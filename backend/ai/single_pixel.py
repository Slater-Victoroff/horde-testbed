import os
import re
import json
import imageio.v3 as iio
from datetime import datetime

import OpenEXR
import Imath
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset

from make_shader import decoder_to_glsl, compare_decoder_and_shader, save_weights_to_exr, save_latent_to_exr

INPUT_IMAGE_CHANNELS = 4  # RGBA
LATENT_IMAGE_CHANNELS = 4 # RGBA
POS_CHANNELS = 6
CONTROL_CHANNELS = 2


def _kernel_expand(raw_pos, height, width, kernel_size=3):
    """
    Expand the raw positions to include neighboring pixels based on the kernel size.
    Handles edges by padding with a default value.
    :param raw_pos: Tensor of shape [H*W, 2] containing x and y coordinates.
    :param height: Height of the image.
    :param width: Width of the image.
    :param kernel_size: Size of the kernel for expansion.
    :param padding_value: Value to use for padding at the edges.
    :return: Tensor of shape [H*W, kernel_size^2, 2] containing expanded positions.
    """
    H, W = height, width
    x_coords = raw_pos[:, 0]
    y_coords = raw_pos[:, 1]

    # Create a grid of offsets based on the kernel size
    offsets = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=raw_pos.device)
    x_offsets, y_offsets = torch.meshgrid(offsets, offsets, indexing="ij")

    # Expand the coordinates
    expanded_x = torch.clamp(x_coords.unsqueeze(-1) + x_offsets.flatten(), 0, width - 1)
    expanded_y = torch.clamp(y_coords.unsqueeze(-1) + y_offsets.flatten(), 0, height - 1)

    return torch.stack([expanded_x, expanded_y], dim=-1).view(-1, kernel_size ** 2, 2)


def _compute_positional_encodings(raw_pos, height, width, num_encodings=POS_CHANNELS, exponential=False):
    """
    Generate positional encodings for the given raw positions.
    :param raw_pos: Tensor of shape [..., 2] containing x and y coordinates in the last axis.
    :param height: Height of the image.
    :param width: Width of the image.
    :param num_encodings: Number of positional encoding channels to generate.
    :return: Tensor of shape [..., num_encodings] containing positional encodings.
    """
    x_coords = torch.clamp(raw_pos[..., 0:1], 0, width - 1) / width  # Clamp and normalize x-coordinates
    y_coords = torch.clamp(raw_pos[..., 1:2], 0, height - 1) / height  # Clamp and normalize y-coordinates

    encodings = [x_coords, y_coords]
    
    # Generate additional sinusoidal encodings
    for i in range(1, (num_encodings - 2) // 2 + 1):
        if exponential:
            encodings.extend([
                torch.sin(2 * torch.pi * (2 ** i) * x_coords),
                torch.sin(2 * torch.pi * (2 ** i) * y_coords)
            ])
        else:
            encodings.extend([
                torch.sin(2 * torch.pi * i * x_coords),
                torch.sin(2 * torch.pi * i * y_coords)
            ])

    # If num_encodings is odd, add one more cosine term
    if len(encodings) < num_encodings:
        for i in range(1, (num_encodings - len(encodings)) // 2 + 1):
            if exponential:
                encodings.extend([
                    torch.cos(2 * torch.pi * (2 ** i) * x_coords),
                    torch.cos(2 * torch.pi * (2 ** i) * y_coords)
                ])
            else:
                encodings.extend([
                    torch.cos(2 * torch.pi * i * x_coords),
                    torch.cos(2 * torch.pi * i * y_coords)
                ])

    positional_encodings = torch.cat(encodings[:num_encodings], dim=-1)
    return positional_encodings


class SineLayer(nn.Module):
    """Siren activation function."""
    def __init__(self, in_features, out_features, is_first=False, omega=10.0):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.linear.in_features) / self.omega,
                    np.sqrt(6 / self.linear.in_features) / self.omega
                )

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class Tanh01(nn.Module):
    def forward(self, x):
        return 0.5 * (torch.tanh(x) + 1.0)


class VFXSpiralNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_hidden_dim = 64
        self.prefilm_dims = 16
        self.input_dim = LATENT_IMAGE_CHANNELS + POS_CHANNELS

        self.control_embed = nn.Sequential(
            nn.Linear(CONTROL_CHANNELS, self.prefilm_dims),
            nn.ReLU(),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(6, self.prefilm_dims),  # time is a scalar (normalized frame index)
            nn.ReLU(),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(12, self.prefilm_dims),  # Don't use positional encodings in Film layer
            nn.ReLU(),
        )

        self.film = nn.Linear(3 * self.prefilm_dims, self.first_hidden_dim * 2)
        
        self.layers = nn.ModuleList([
            nn.Linear(self.input_dim, self.first_hidden_dim),
            nn.GELU(),
            nn.Linear(self.first_hidden_dim, self.first_hidden_dim),
            nn.GELU(),
            nn.Linear(self.first_hidden_dim, INPUT_IMAGE_CHANNELS),
            nn.Sigmoid()
        ])
    
    def forward(self, latent, raw_pos, control, return_hidden_layer=None):
        H, W, C = latent.shape
        latent_flat = latent.view(-1, C)  # Flatten latent to [H*W, C]
        linear_indices = raw_pos[:, 1] * W + raw_pos[:, 0]
        indexed_latent = latent_flat[linear_indices]
        pos_enc = _compute_positional_encodings(raw_pos, H, W)
        main_input = torch.concat([indexed_latent, pos_enc], dim=1)  # [B, LATENT_IMAGE_CHANNELS + POS_CHANNELS]


        control_feat = self.control_embed(control)
        x = pos_enc[:, 0:1]  # normalized x in [0, 1]
        y = pos_enc[:, 1:2]  # normalized y in [0, 1]

        # Scale to [0, 2π]
        x = x * 2 * torch.pi
        y = y * 2 * torch.pi

        # Spiral-style position embedding
        spiral_pos = torch.cat([
            torch.sin(x), torch.cos(x),
            torch.sin(2 * x), torch.cos(2 * x),
            torch.sin(3 * x), torch.cos(3 * x),
            torch.sin(y), torch.cos(y),
            torch.sin(2 * y), torch.cos(2 * y),
            torch.sin(3 * y), torch.cos(3 * y)
        ], dim=-1)  # [B, 12]

        pos_feat = self.pos_embed(spiral_pos)
        t = control[:, 0:1] * 2 * torch.pi  # map to [0, 2π]

        # Spiral embedding: sin/cos for base cycle, sin/cos of harmonic for texture
        spiral_time = torch.cat([
            torch.sin(t), torch.cos(t),
            torch.sin(2 * t), torch.cos(2 * t),
            torch.sin(3 * t), torch.cos(3 * t)
        ], dim=-1)  # [B, 6]

        time_feat = self.time_embed(spiral_time)

        film_input = torch.cat([control_feat, pos_feat, time_feat], dim=-1)  # [B, 3 * pref_dim]

        outputs = main_input
        for i, layer in enumerate(self.layers):
            if i == 1:  # First layer, apply film
                gamma, beta = self.film(film_input).chunk(2, dim=-1)
                outputs = layer((gamma * outputs) + beta)
            else:    
                outputs = layer(outputs)
            if return_hidden_layer is not None and i == return_hidden_layer:
                return outputs
        return outputs


class VFXNetContextDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_hidden_dim = 64
        self.prefilm_dims = 16
        self.input_dim = LATENT_IMAGE_CHANNELS + POS_CHANNELS

        self.control_embed = nn.Sequential(
            nn.Linear(CONTROL_CHANNELS, self.prefilm_dims),
            nn.ReLU(),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.prefilm_dims),  # time is a scalar (normalized frame index)
            nn.ReLU(),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(2, self.prefilm_dims),  # Don't use positional encodings in Film layer
            nn.ReLU(),
        )

        self.film = nn.Linear(3 * self.prefilm_dims, self.first_hidden_dim * 2)
        
        self.layers = nn.ModuleList([
            nn.Linear(self.input_dim, self.first_hidden_dim),
            nn.GELU(),
            nn.Linear(self.first_hidden_dim, 2 * self.first_hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.first_hidden_dim, self.first_hidden_dim),
            nn.GELU(),
            nn.Linear(self.first_hidden_dim, INPUT_IMAGE_CHANNELS),
            nn.Sigmoid()
        ])
    
    def forward(self, latent, raw_pos, control, return_hidden_layer=None):
        H, W, C = latent.shape
        latent_flat = latent.view(-1, C)  # Flatten latent to [H*W, C]
        linear_indices = raw_pos[:, 1] * W + raw_pos[:, 0]
        indexed_latent = latent_flat[linear_indices]
        pos_enc = _compute_positional_encodings(raw_pos, H, W)
        x = torch.concat([indexed_latent, pos_enc], dim=1)  # [B, LATENT_IMAGE_CHANNELS + POS_CHANNELS]


        control_feat = self.control_embed(control)
        pos_feat = self.pos_embed(pos_enc[:, 0:2])  # First two channels are x and y
        time_feat = self.time_embed(control[:, 0:1])  # Assuming time is the first control channel
        film_input = torch.cat([control_feat, pos_feat, time_feat], dim=-1)  # [B, 3 * pref_dim]

        outputs = x
        for i, layer in enumerate(self.layers):
            if i == 1:  # First layer, apply film
                gamma, beta = self.film(film_input).chunk(2, dim=-1)
                outputs = layer((gamma * outputs) + beta)
            else:    
                outputs = layer(outputs)
            if return_hidden_layer is not None and i == return_hidden_layer:
                return outputs
        return outputs


class VFXNetContextSirenDecoder(VFXNetContextDecoder):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            SineLayer(self.input_dim, self.first_hidden_dim, is_first=True),
            SineLayer(self.first_hidden_dim, 2 * self.first_hidden_dim),
            SineLayer(2 * self.first_hidden_dim, self.first_hidden_dim),
            nn.Linear(self.first_hidden_dim, INPUT_IMAGE_CHANNELS),
            Tanh01()
        ])

class VFXNetPixelDecoder(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.film = nn.Linear(CONTROL_CHANNELS, 64 * 2)
        self.layers = nn.ModuleList([
            nn.Linear(LATENT_IMAGE_CHANNELS + POS_CHANNELS + CONTROL_CHANNELS, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, INPUT_IMAGE_CHANNELS),
            nn.Sigmoid()
        ])

    def forward(self, latent, raw_pos, control, return_hidden_layer=None):
        # Use raw_pos to index into the latent
        H, W, C = latent.shape
        assert (H, W) == (self.height, self.width)
        latent_flat = latent.view(-1, C)  # Flatten latent to [H*W, C]
        linear_indices = raw_pos[:, 1] * self.width + raw_pos[:, 0]  # Compute linear indices
        indexed_latent = latent_flat[linear_indices]  # Gather latent values based on raw_pos
        pos_enc = _compute_positional_encodings(raw_pos, self.height, self.width)
        x = torch.cat([indexed_latent, pos_enc, control], dim=1)  # [B, LATENT_IMAGE_CHANNELS + POS_CHANNELS + CONTROL_CHANNELS]

        outputs = x
        for i, layer in enumerate(self.layers):
            if i == 1:  # First layer, apply film
                gamma, beta = self.film(control).chunk(2, dim=-1)
                outputs = layer((gamma * outputs) + beta)
            else:    
                outputs = layer(outputs)
            if return_hidden_layer is not None and i == return_hidden_layer:
                return outputs
        return outputs


class VFXNetSirenDecoder(VFXNetPixelDecoder):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.layers = nn.ModuleList([
            SineLayer(LATENT_IMAGE_CHANNELS + POS_CHANNELS + CONTROL_CHANNELS, 64, is_first=True),
            SineLayer(64, 128),
            SineLayer(128, 64),
            nn.Linear(64, INPUT_IMAGE_CHANNELS),
            Tanh01()
        ])


class VFXNetPatchDecoder(nn.Module):
    def __init__(self, height, width, kernel_size=3):
        super().__init__()
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.input_channels = LATENT_IMAGE_CHANNELS * (kernel_size ** 2) + POS_CHANNELS * (kernel_size ** 2) + CONTROL_CHANNELS * (kernel_size ** 2)

        self.layers = nn.ModuleList([
            nn.Linear(self.input_channels, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, INPUT_IMAGE_CHANNELS),
            nn.Sigmoid()
        ])
    
    def _forward(self, x, return_hidden_layer=None):
        output = x
        for i, layer in enumerate(self.layers):
            output = layer(output)
            if return_hidden_layer is not None and i == return_hidden_layer:
                return output
        return output
    
    def forward(self, latent, raw_pos, control):
        """
        latent:   [B, H, W, latent_dim]
        pos_enc:  [B, H, W, pos_dim]
        control:  [B, control_dim]
        """
        expanded_pos = _kernel_expand(raw_pos, self.height, self.width, kernel_size=self.kernel_size)
        H, W, C = latent.shape
        assert (H, W) == (self.height, self.width)

        # Flatten the latent image for easier indexing
        latent_flat = latent.view(-1, latent.shape[-1])  # [H*W, C]
        linear_indices = expanded_pos[..., 1] * self.width + expanded_pos[..., 0]  # [B, 9]
        latent_values = latent_flat[linear_indices]  # [B, 9, C]
        pos_enc = _compute_positional_encodings(expanded_pos, self.height, self.width)

        control_expanded = control.unsqueeze(1).expand(-1, self.kernel_size ** 2, -1)

        patches = torch.cat([latent_values, pos_enc, control_expanded], dim=-1)
        unrolled = patches.view(patches.shape[0], -1)  # Unroll to [B, kernel**2 * Something]
        return self._forward(unrolled)


class VFXNet(nn.Module):
    def __init__(self, height, width, device='cuda', experiment_name=None):
        super().__init__()
        self.experiment_name = experiment_name
        self.height = height
        self.width = width
        _x_coords = torch.arange(width).repeat(height, 1).view(-1, 1)
        _y_coords = torch.arange(height).repeat(width, 1).t().contiguous().view(-1, 1)
        self.raw_pos = torch.cat([_x_coords, _y_coords], dim=1)  # [H*W, 2]
        self.pos_enc = _compute_positional_encodings(self.raw_pos, height, width)

        self.shared_latent = nn.Parameter(torch.randn(height, width, LATENT_IMAGE_CHANNELS))
        self.decoder = VFXSpiralNetDecoder()
        self.raw_pos = self.raw_pos.to(device)
        self.pos_enc = self.pos_enc.to(device)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def forward(self, raw_pos, control):
        return self.decoder(self.shared_latent, raw_pos, control)
    
    def full_image(self, control):
        # Expand control to match the first dimension of self.raw_pos
        expanded_control = control.unsqueeze(1).expand(-1, self.raw_pos.size(0), -1).reshape(-1, control.size(-1))
        response = self.decoder(self.shared_latent, self.raw_pos, expanded_control)
        shaped_image = response.view(self.height, self.width, INPUT_IMAGE_CHANNELS)
        return shaped_image

    def save_as_glsl(self, directory, source_images, test=True):
        #TODO: Make work.
        directory = str(directory)
        if not directory.endswith('/'):
            directory += '/'

        # Get grayscale image
        grayscale = self.get_grayscale(source_images)

        # Save grayscale as EXR
        grayscale_path = f"{directory}grayscale.exr"
        save_latent_to_exr(grayscale, grayscale_path)
        print(f"Grayscale saved to {grayscale_path}")
        
        # Save weights as EXR
        weights_path = f"{directory}weights.exr"
        layer_offsets = save_weights_to_exr(self.grayscale_to_rgb, weights_path)
        print(f"Weights saved to {weights_path}")
        offsets_path = f"{directory}offsets.json"
        with open(offsets_path, "w") as f:
            json.dump({"layer_offsets": layer_offsets}, f)
        print(f"Offsets saved to {offsets_path}")

        # Generate GLSL code
        glsl_code = decoder_to_glsl(self.grayscale_to_rgb, weights_path, debug_mode=test)

        # Save GLSL code as .frag file
        glsl_path = f"{directory}shader.frag"
        with open(glsl_path, "w") as f:
            f.write(glsl_code)
        print(f"GLSL shader saved to {glsl_path}")

        # Test the shader and decoder if required
        if test:
            compare_decoder_and_shader(self.grayscale_to_rgb, glsl_path, grayscale_path, weights_path, offsets_path)


def train_vfx_model(image_dir, criterion, device='cuda', epochs=1000, batch_size=8192, save_every=5, experiment_name=None):
    # Load images and create tensors
    image_tensor, raw_pos, control_tensor, shape = load_images(image_dir, device)
    print("image_tensor shape:", image_tensor.shape)
    print("shape:", shape)
    print("control_tensor shape:", control_tensor.shape)
    print("control_tensor start:", control_tensor[0:5])
    print("control_tensor end:", control_tensor[-5:])

    # Create a dataset and dataloader
    dataset = TensorDataset(image_tensor, raw_pos, control_tensor)

    model = VFXNet(shape[0], shape[1]).to(device)
    model.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0.0
        batch_num = 0
        # Training loop
        for image, raw_pos, control in train_dataloader:
            batch_num += 1
            if batch_num % 100 == 0:
                print(f"Batch {batch_num}/{len(train_dataloader)}")
            reconstructed_image = model(raw_pos, control)

            loss = criterion(reconstructed_image, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.6f}")
        if (epoch + 1) % save_every == 0:
            model.eval()  # Set model to evaluation mode
            save_images(model, control_tensor, epoch)
            print(f"Saved grayscale and reconstructed RGB images for epoch {epoch + 1}.")


def load_exr_image(image_path):
    # Open the EXR file
    rgba = None
    with OpenEXR.File(str(image_path)) as exr_file:
        header = exr_file.header()
        # Get the image dimensions
        header = exr_file.header()
        min_val, max_val = header['dataWindow']
        width = max_val[0] - min_val[0] + 1
        height = max_val[1] - min_val[1] + 1
        if "RGB" in exr_file.channels():
            rgb = exr_file.channels()["RGB"].pixels
            alpha = np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.float16)
            rgba = np.concatenate([rgb, alpha], axis=2)
        elif "RGBA" in exr_file.channels():
            rgba = exr_file.channels()["RGBA"].pixels
        else:
            raise ValueError("EXR file does not contain RGB or RGBA channels.")
    return rgba


def load_images(image_dir, device):
    image_tensors = []
    pos_tensors = []
    control_tensors = []
    shape = None

    for image_path in sorted(image_dir.glob("**/*.exr")):  # Assuming EXR images
        # Load the image and convert to tensor
        image = load_exr_image(image_path)
        if shape is None:
            shape = image.shape
        elif shape != image.shape:
            raise ValueError(f"Image shape mismatch: {shape} != {image.shape}")
        image_tensor = torch.tensor(image, dtype=torch.float32, device=device).view(-1, INPUT_IMAGE_CHANNELS)  # Flatten to [H*W, 4]
        image_tensors.append(image_tensor)
        # Extract control values from the filename using regex
        matches = re.findall(r'\d+', image_path.stem)
        control_values = [int(match) for match in matches]

        # Pad with zeros if less than CONTROL_CHANNELS
        control_tensor = torch.tensor(control_values + [0] * (CONTROL_CHANNELS - len(control_values)), device=device)
        control_tensors.append(control_tensor.repeat(image_tensor.size(0), 1))  # [H*W, CONTROL_CHANNELS]
        # Generate positional embeddings
        H, W, _ = image.shape
        x_coords = torch.arange(W, device=device).repeat(H, 1).view(-1, 1)  # [H*W, 1]
        y_coords = torch.arange(H, device=device).repeat(W, 1).t().contiguous().view(-1, 1)  # [H*W, 1]
        pos_tensor = torch.cat([x_coords, y_coords], dim=1)  # [H*W, 2]
        pos_tensors.append(pos_tensor)

    # Combine all tensors
    image_tensor = torch.cat(image_tensors, dim=0)  # [H*W*images, INPUT_IMAGE_CHANNELS]
    raw_pos = torch.cat(pos_tensors, dim=0)  # [H*W*images, 2]
    control = torch.cat(control_tensors, dim=0)  # [H*W*images, CONTROL_CHANNELS]

    # Identify the time axis
    time_axis = None
    for i in range(control.size(1)):
        unique_values = torch.unique(control[:, i])
        print(unique_values[1:] - unique_values[:-1])
        if len(unique_values) > 1 and torch.all(unique_values[1:] - unique_values[:-1] == 1):  # Check for sequential increments
            time_axis = i
            break

    if time_axis is not None and time_axis != 0:
        # Swap the time axis to the first position
        control = torch.cat([control[:, time_axis:time_axis + 1], control[:, :time_axis], control[:, time_axis + 1:]], dim=1)
    elif time_axis is None:
        raise ValueError("Unable to identify a time axis in the control tensor.")

    # Normalize each control channel to the range [0, 1]
    control_min, _ = control.min(dim=0, keepdim=True)
    control_max, _ = control.max(dim=0, keepdim=True)
    control = (control - control_min) / (control_max - control_min + 1e-8)  # Add epsilon to avoid division by zero

    print("Normalized control tensor:", control[0:5])
    image_tensor = image_tensor.to(device)
    control = control.to(device)
    raw_pos = raw_pos.to(device)
    return image_tensor, raw_pos, control, shape


def save_images(model, control_tensor, epoch, n=5):
    # Create directory structure
    base_dir = "debug_outputs"
    epoch_dir = os.path.join(base_dir, model.experiment_name, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    with torch.no_grad():
        # Randomly select n unique control vectors
        indices = torch.randperm(control_tensor.size(0))[:n]
        selected_controls = control_tensor[indices]

        # Log details about the selected control vectors
        print(f"Selected control vectors for epoch {epoch}:")
        for idx, control in enumerate(selected_controls):
            print(f"Control vector {idx + 1}: {control.cpu().numpy()}")

        # Generate reconstructed RGB images for each selected control vector
        for i, control in enumerate(selected_controls):
            reconstructed_rgb = model.full_image(control.unsqueeze(0))  # Pass control vector to model
            rgb_image = reconstructed_rgb.squeeze(0).permute(2, 0, 1)  # [C, H, W]

            # Save the reconstructed RGB image with control vector details in the filename
            control_details = "_".join(f"{val:.2f}" for val in control.cpu().numpy())
            rgb_path = os.path.join(epoch_dir, f"reconstructed_rgb_{control_details}.png")
            save_image(rgb_image, rgb_path)

        # Save the shared latent image as an EXR file
        latent_path = os.path.join(epoch_dir, "shared_latent.exr")
        save_latent_to_exr(model.shared_latent, latent_path)
        print(f"Shared latent image saved to {latent_path}")

        # Save the model/weights
        model_path = os.path.join(epoch_dir, "model_weights.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")

        # Generate and save a GIF from evenly sampled control vectors
        gif_path = os.path.join(epoch_dir, "control_animation.gif")
        generate_control_animation(model, control_tensor, gif_path)
        print(f"Control animation GIF saved to {gif_path}")


def generate_control_animation(model, control_tensor, gif_path, num_frames=50):
    """
    Generate a GIF by evenly sampling control vectors and rendering frames.
    :param model: The model to generate frames.
    :param control_tensor: The tensor of control vectors.
    :param gif_path: Path to save the generated GIF.
    :param num_frames: Number of frames in the GIF.
    """
    # Evenly sample control vectors
    indices = torch.linspace(0, control_tensor.size(0) - 1, steps=num_frames).long()
    sampled_controls = control_tensor[indices]

    frames = []
    with torch.no_grad():
        for control in sampled_controls:
            reconstructed_rgb = model.full_image(control.unsqueeze(0))  # Pass control vector to model
            rgb_image = reconstructed_rgb.squeeze(0).cpu().numpy()  # [H, W, C]
            frame = (rgb_image * 255).astype(np.uint8)  # Convert to uint8 for GIF
            frames.append(frame)

    iio.imwrite(gif_path, frames)
