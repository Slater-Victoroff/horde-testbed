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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset

from make_shader import decoder_to_glsl, compare_decoder_and_shader, save_weights_to_exr, save_latent_to_exr

INPUT_IMAGE_CHANNELS = 4  # RGBA
LATENT_IMAGE_CHANNELS = 4 # RGBA
POS_CHANNELS = 6
CONTROL_CHANNELS = 2


class VFXNetDecoder(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
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

    def forward(self, latent, pos_enc, control, return_hidden_layer=None):
        # Expects properly formatted positional encodings. They should already be on the right device
        x = torch.cat([latent, pos_enc, control], dim=1)  # [B, LATENT_IMAGE_CHANNELS + POS_CHANNELS + CONTROL_CHANNELS]

        outputs = x
        for i, layer in enumerate(self.layers):
            outputs = layer(outputs)
            if return_hidden_layer is not None and i == return_hidden_layer:
                return outputs
        return outputs


class VFXNet(nn.Module):
    def __init__(self, height, width, device='cuda', experiment_name=None):
        super().__init__()
        self.experiment_name = experiment_name
        self.height = height
        self.width = width
        _x_coords = torch.arange(width).repeat(height, 1).view(-1, 1)
        _y_coords = torch.arange(height).repeat(width, 1).t().contiguous().view(-1, 1)
        self.raw_pos = torch.cat([_x_coords, _y_coords], dim=1)  # [H*W, 2]
        self.pos_enc = self._compute_positional_encodings(self.raw_pos, height, width)

        self.shared_latent = nn.Parameter(torch.randn(height, width, LATENT_IMAGE_CHANNELS))  # Shared latent image
        self.decoder = VFXNetDecoder(height, width)
        self.raw_pos = self.raw_pos.to(device)  # Move raw positions to the correct device
        self.pos_enc = self.pos_enc.to(device)  # Move positional encodings to the correct device
        self._initialize_weights() # Apply Xavier initialization to all layers

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compute_positional_encodings(self, raw_pos, height, width, num_encodings=POS_CHANNELS):
        """
        Generate positional encodings for the given raw positions.
        :param raw_pos: Tensor of shape [H*W, 2] containing x and y coordinates.
        :param height: Height of the image.
        :param width: Width of the image.
        :param num_encodings: Number of positional encoding channels to generate.
        :return: Tensor of shape [H, W, num_encodings] containing positional encodings.
        """
        x_coords = raw_pos[:, 0:1] / width  # Normalize x-coordinates
        y_coords = raw_pos[:, 1:2] / height  # Normalize y-coordinates

        encodings = [x_coords, y_coords]

        # Generate additional sinusoidal encodings
        for i in range(1, (num_encodings - 2) // 2 + 1):
            x_sin_coords = torch.sin(2 * torch.pi * i * x_coords)
            y_sin_coords = torch.sin(2 * torch.pi * i * y_coords)
            # x_sin_coords = torch.sin(2 * torch.pi * (2 ** i) * x_coords)
            # y_sin_coords = torch.sin(2 * torch.pi * (2 ** i) * y_coords)
            encodings.extend([x_sin_coords, y_sin_coords])

        # If num_encodings is odd, add one more cosine term
        if len(encodings) < num_encodings:
            for i in range(1, (num_encodings - len(encodings)) // 2 + 1):
                x_cos_coords = torch.cos(2 * torch.pi * i * x_coords)
                y_cos_coords = torch.cos(2 * torch.pi * i * y_coords)
                encodings.extend([x_cos_coords, y_cos_coords])

        # Concatenate encodings and reshape to [H, W, num_encodings]
        positional_encodings = torch.cat(encodings[:num_encodings], dim=1)
        return positional_encodings.view(height, width, num_encodings)

    def forward(self, raw_pos, control):
        # Convert raw_pos (Nx2) into indices for shared_latent (HxWx4)
        H, W, _ = self.shared_latent.shape
        x_indices = raw_pos[:, 0].long().clamp(0, W - 1)  # Ensure indices are within bounds
        y_indices = raw_pos[:, 1].long().clamp(0, H - 1)  # Ensure indices are within bounds
        latent_section = self.shared_latent[y_indices, x_indices]  # Nx4

        pos_subset = self.pos_enc[y_indices, x_indices]
        return self.decoder(latent_section, pos_subset, control)
    
    def full_image(self, control, device='cuda'):
        full_control = control.repeat(self.raw_pos.shape[0], 1).to(device)  # Repeat control for all pixels
        print("full_control shape:", full_control.shape)
        flattened_latent = self.shared_latent.view(-1, LATENT_IMAGE_CHANNELS)  # Flatten shared_latent
        print("flattened_latent shape:", flattened_latent.shape)
        flattened_latent = flattened_latent.to(device)  # Move to the correct device

        all_response = self.decoder(flattened_latent, self.pos_enc.view(-1, POS_CHANNELS), full_control)
        print("all_response shape:", all_response.shape)
        shaped_image = all_response.view(self.height, self.width, INPUT_IMAGE_CHANNELS)
        print("shaped_image shape:", shaped_image.shape)
        return shaped_image

    def save_as_glsl(self, directory, source_images, test=True):
        # Ensure the directory is a string and ends with a slash
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
            print(f"Reconstructed RGB shape: {reconstructed_rgb.shape}")
            rgb_image = reconstructed_rgb.squeeze(0).cpu().numpy()  # [H, W, C]
            print(f"RGB image shape: {rgb_image.shape}")
            frame = (rgb_image * 255).astype(np.uint8)  # Convert to uint8 for GIF
            print(f"Frame shape: {frame.size}")
            frames.append(frame)

    iio.imwrite(gif_path, frames)
