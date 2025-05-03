import numpy as np

import torch
import torch.nn as nn


def kernel_expand(raw_pos, height, width, kernel_size=3):
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


def compute_positional_encodings(raw_pos, height, width, num_encodings, exponential=False):
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
