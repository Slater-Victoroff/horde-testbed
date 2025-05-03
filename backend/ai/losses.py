import torch
import torch.nn as nn
from torchvision import models


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


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def _normalize(self, x):
        x = x[:, :3, :, :]  # Assume input is RGBA and clip to RGB
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, pred, target):
        pred = self._normalize(pred)
        target = self._normalize(target)
        return nn.functional.mse_loss(self.vgg(pred), self.vgg(target))
