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

def compute_helix_encoding(x, num_harmonics):
    out = []
    for i in range(1, num_harmonics + 1):
        out += [torch.sin(i * x), torch.cos(i * x)]
    return torch.cat(out, dim=-1)


def compute_spiral_encoding(x, num_harmonics):
    out = []
    for i in range(1, num_harmonics + 1):
        out += [torch.sin(i * x) / i, torch.cos(i * x) / i]
    return torch.cat(out, dim=-1)


def compute_sinusoidal_encoding(coords, num_harmonics):
    encodings = []
    for i in range(1, num_harmonics + 1):
        encodings += [torch.sin(2 * torch.pi * i * coords), torch.cos(2 * torch.pi * i * coords)]
    return torch.cat(encodings, dim=-1)


def compute_targeted_encodings(x, target_dim, scheme="spiral", norm_2pi=True, include_norm=False, include_raw=False):
    _, N = x.shape
    encodings = []

    if include_raw:
        encodings.append(x)

    if norm_2pi:
        x = x * 2 * torch.pi
        if include_norm:
            encodings.append(x)

    if scheme in ["helix", "spiral", "sinusoidal"]:
        num_harmonics = ((target_dim - (N if include_raw else 0)) // 2) + 1
        encoding_fn = {
            "helix": compute_helix_encoding,
            "spiral": compute_spiral_encoding,
            "sinusoidal": compute_sinusoidal_encoding,
        }[scheme]
        encodings.append(encoding_fn(x, num_harmonics=num_harmonics))
    elif scheme is None:
        encodings.append(torch.zeros(x.shape[0], target_dim, device=x.device))
    else:
        raise ValueError(f"Unknown encoding scheme: {scheme}")

    return torch.cat(encodings, dim=-1)[:, :target_dim]


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
