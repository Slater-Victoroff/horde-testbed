import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.utils.data import DataLoader, TensorDataset

from decoders import VFXSpiralNetDecoder, VFXNetPixelDecoder
from losses import DCTLoss
from utils import compute_positional_encodings
from make_shader import decoder_to_glsl, compare_decoder_and_shader, save_weights_to_exr
from image_utils import load_images, save_images

INPUT_IMAGE_CHANNELS = 4  # RGBA
LATENT_IMAGE_CHANNELS = 4 # RGBA
POS_CHANNELS = 6
CONTROL_CHANNELS = 2


class VFXNet(nn.Module):
    def __init__(self, height, width, device='cuda', experiment_name=None):
        super().__init__()
        self.experiment_name = experiment_name
        self.height = height
        self.width = width
        _x_coords = torch.arange(width).repeat(height, 1).view(-1, 1)
        _y_coords = torch.arange(height).repeat(width, 1).t().contiguous().view(-1, 1)
        self.raw_pos = torch.cat([_x_coords, _y_coords], dim=1)  # [H*W, 2]
        self.pos_enc = compute_positional_encodings(self.raw_pos, height, width, POS_CHANNELS)

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
        print(f"Expanded control shape: {expanded_control.shape}")
        response = self.decoder(self.shared_latent, self.raw_pos, expanded_control)
        shaped_image = response.view(self.height, self.width, INPUT_IMAGE_CHANNELS)
        return shaped_image


def train_vfx_model(image_dir, device='cuda', epochs=1000, batch_size=8192, save_every=5, perceptual_epoch=5, experiment_name=None):
    # Load images and create tensors
    image_tensor, raw_pos, control_tensor, shape = load_images(image_dir, device)
    # print some stats to make sure these are correct
    print(f"image_tensor shape: {image_tensor.shape}")
    print(f"raw_pos shape: {raw_pos.shape}")
    print(f"control_tensor shape: {control_tensor.shape}")
    print(f"image_tensor min: {image_tensor.min()}, max: {image_tensor.max()}")
    print(f"image_tensor mean: {image_tensor.mean()}, std: {image_tensor.std()}")
    print(f"image_tensor dtype: {image_tensor.dtype}")
    print(f"raw_pos min: {raw_pos.min()}, max: {raw_pos.max()}")
    print(f"control_tensor min: {control_tensor.min()}, max: {control_tensor.max()}")
    pixel_loss = nn.MSELoss().to(device)
    dct_loss = DCTLoss().to(device)

    print("image_tensor shape:", image_tensor.shape)
    print("raw_pos shape:", raw_pos.shape)
    print("control_tensor shape:", control_tensor.shape)

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

            loss = pixel_loss(reconstructed_image, image)

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
