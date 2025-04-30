import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from full_image import RGBImageModel, train_image_model
from single_pixel import train_vfx_model
import argparse

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_exponent=2.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')  # Compute element-wise MSE
        self.weight_exponent = weight_exponent

    def forward(self, reconstructed, target):
        # Compute the mean RGB value for the target
        mean_rgb = target.mean(dim=0, keepdim=True)

        # Compute weights based on distance from the mean
        weights = torch.norm(target - mean_rgb, dim=1, keepdim=True)  # L2 norm (distance from mean)

        # Normalize weights to [0, 1]
        weights = (weights / weights.max()) ** self.weight_exponent

        # Compute weighted MSE loss
        loss = self.mse(reconstructed, target)
        weighted_loss = (weights * loss).mean()  # Apply weights and average
        return weighted_loss


class GammaCorrectedMSELoss(nn.Module):
    def __init__(self, gamma=2.2):
        super().__init__()
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(self, reconstructed, target):
        # Apply gamma correction
        reconstructed_gamma = torch.pow(reconstructed, self.gamma)
        target_gamma = torch.pow(target, self.gamma)
        return self.mse(reconstructed_gamma, target_gamma)


class FrequencyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, reconstructed, target):
        # Compute the Fourier transform of the reconstructed and target images
        reconstructed_fft = torch.fft.fft2(reconstructed, norm="ortho")
        target_fft = torch.fft.fft2(target, norm="ortho")

        # Compute the magnitude of the frequency components
        reconstructed_mag = torch.abs(reconstructed_fft)
        target_mag = torch.abs(target_fft)

        # Compute the loss as the mean squared error of the magnitudes
        freq_loss = torch.mean((reconstructed_mag - target_mag) ** 2)
        return self.weight * freq_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    STATIC_DIR = Path("/app/static")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # criterion = GammaCorrectedMSELoss().to(device)
    criterion = FrequencyLoss().to(device)
    # criterion = nn.MSELoss().to(device)
    model = train_vfx_model(STATIC_DIR / "VFX/hollow-flame/", criterion, device=device, experiment_name="freq-loss")
    
    # Save model state
    model_path = results_dir / f"vfx_model_combined{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(model.state_dict(), model_path)

    # model.save_as_glsl(results_dir, test_images)
    
    print(f"Model saved to {model_path}")
    print(f"GLSL shader saved to {results_dir}")

if __name__ == "__main__":
    main()
