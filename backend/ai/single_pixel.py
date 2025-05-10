import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader, TensorDataset
from piq import ssim

from decoders import VFXSpiralNetDecoder, VFXNetPixelDecoder
from losses import DCTLoss
from make_shader import decoder_to_glsl, compare_decoder_and_shader, save_weights_to_exr
from image_utils import load_images, save_images

INPUT_IMAGE_CHANNELS = 4  # RGBA
LATENT_IMAGE_CHANNELS = 4 # RGBA
POS_CHANNELS = 8
CONTROL_CHANNELS = 2


class VFXNet(nn.Module):
    def __init__(self, height, width, decoder_config=None, device='cuda', experiment_name=None):
        super().__init__()
        self.experiment_name = experiment_name
        self.height = height
        self.width = width
        _x_coords = torch.arange(width).repeat(height, 1).view(-1, 1)
        _y_coords = torch.arange(height).repeat(width, 1).t().contiguous().view(-1, 1)
        self.raw_pos = torch.cat([_x_coords, _y_coords], dim=1)

        self.shared_latent = nn.Parameter(torch.randn(height, width, LATENT_IMAGE_CHANNELS))
        self.decoder = VFXSpiralNetDecoder(**decoder_config or {})
        self.raw_pos = self.raw_pos.to(device)
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


def train_vfx_model(image_dir, device='cuda', epochs=100, batch_size=8192, experiment_name=None, decoder_config=None):
    # Load images and create tensors
    image_tensor, raw_pos, control_tensor, shape = load_images(image_dir, device)
    mse_loss = nn.MSELoss().to(device)
    dct_loss = DCTLoss().to(device)

    # Create a dataset and dataloader
    dataset = TensorDataset(image_tensor, raw_pos, control_tensor)

    model = VFXNet(shape[0], shape[1], decoder_config=decoder_config).to(device)
    model.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

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

            loss = (mse_loss(reconstructed_image, image) + dct_loss(reconstructed_image, image)) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.6f}")
        if epoch in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 99]:
            model.eval()  # Set model to evaluation mode
            reconstructed_batch, sampled_controls = save_images(model, control_tensor, epoch)
            print(f"Saved grayscale and reconstructed RGB images for epoch {epoch + 1}.")
            H, W = model.height, model.width
            pixels_per_frame = H * W
            T = control_tensor.shape[0] // pixels_per_frame
            ssim_scores = []
            for recon, control in zip(reconstructed_batch, sampled_controls):
                t = control[0].item()
                frame_idx = min(int(t * (T - 1)), T - 1)

                # Slice out the corresponding ground truth frame
                start = frame_idx * pixels_per_frame
                end = start + pixels_per_frame
                gt_frame = image_tensor[start:end].view(H, W, -1)  # [H, W, C]

                # Compute SSIM
                recon_img = recon.permute(2, 0, 1).unsqueeze(0)
                gt_img = gt_frame.permute(2, 0, 1).unsqueeze(0)
                score = ssim(recon_img, gt_img, data_range=1.0).item()
                ssim_scores.append(score)
            avg_ssim = sum(ssim_scores) / len(ssim_scores)
            print(f"Average SSIM for epoch {epoch + 1}: {avg_ssim:.4f}")
            metrics_path = f"final_data/{model.experiment_name}/epoch_{epoch}/summary.json"
            with open(metrics_path, "w") as f:
                json.dump({
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "avg_ssim": avg_ssim,
                    "ssim_per_frame": ssim_scores,
                    "decoder_config": decoder_config
                }, f, indent=2)
            print(f"Saved epoch summary to {metrics_path}")
