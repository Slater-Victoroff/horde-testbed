import math
import numpy as np

import torch
import torch.nn as nn

from functools import partial

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


def compute_spiral_encoding(x, target_dim, freqs=None, amps=None):
    out = []
    if freqs is None:
        num_harmonics = (target_dim // 2) + 1
        freqs = torch.arange(1, num_harmonics + 1, device=x.device).repeat_interleave(2)
    if amps is None:
        amps = torch.ones_like(freqs)
    for i in range(freqs.shape[0]):
        if i % 2 == 1:
            out += torch.sin(freqs[i] * x * 2 * np.pi) * amps[i]
        elif i % 2 == 0:
            out += torch.cos(freqs[i] * x * 2 * np.pi) * amps[i]
    return torch.cat(out, dim=-1)


def compute_sinusoidal_encoding(coords, target_dim, freqs=None):
    cycle_length = 2
    N = int(coords.shape[1])

    num_harmonics = int(math.ceil(target_dim / (cycle_length * N)))
    if freqs is None:
        freqs = torch.arange(1, num_harmonics + 1, device=coords.device).repeat_interleave(N).repeat(cycle_length)
        freqs = freqs[:target_dim]  # Ensure we only take as many frequencies as needed

    reshaped_coords = coords.repeat([1, math.ceil(target_dim / N)])
    updated_coords = freqs.unsqueeze(0) * reshaped_coords

    sin_chunk, cos_chunk = updated_coords.split(num_harmonics * N, dim=1)

    encodings = [
        torch.sin(sin_chunk * 2 * np.pi),
        torch.cos(cos_chunk * 2 * np.pi)
    ]

    encoded_tensor = torch.cat(encodings, dim=-1)
    return torch.cat(encodings, dim=-1)


def compute_mathy_encoding(coords, target_dim, freqs=None):
    cycle_length = 8
    N = int(coords.shape[1])

    num_harmonics = int(math.ceil(target_dim / (cycle_length * N)))
    if freqs is None:
        freqs = torch.arange(1, num_harmonics + 1, device=coords.device).repeat_interleave(N).repeat(cycle_length)
        freqs = freqs[:target_dim]  # Ensure we only take as many frequencies as needed

    reshaped_coords = coords.repeat([1, math.ceil(target_dim / N)])
    updated_coords = torch.clamp(freqs.unsqueeze(0) * reshaped_coords, min=1e-5)  # epsilon to help with stability at 0

    sin_chunk, cos_chunk, exp_chunk, log1p_chunk, sqrt_chunk, inv_chunk, rsqrt_chunk, sq_chunk = updated_coords.split(num_harmonics * N, dim=1)

    encodings = torch.cat([
        torch.sin(sin_chunk * 2 * np.pi),
        torch.cos(cos_chunk * 2 * np.pi),
        torch.exp(exp_chunk),
        torch.log1p(log1p_chunk),
        torch.sqrt(sqrt_chunk),
        1 / (inv_chunk),
        torch.rsqrt(rsqrt_chunk),
        sq_chunk * sq_chunk,
    ], dim=1)

    return encodings


def compute_less_mathy_encoding(coords, target_dim, freqs=None):
    cycle_length = 4
    N = int(coords.shape[1])

    num_harmonics = int(math.ceil(target_dim / (cycle_length * N)))
    if freqs is None:
        freqs = torch.arange(1, num_harmonics + 1, device=coords.device).repeat_interleave(N).repeat(cycle_length)
        freqs = freqs[:target_dim]  # Ensure we only take as many frequencies as needed

    reshaped_coords = coords.repeat([1, math.ceil(target_dim / N)])
    updated_coords = torch.clamp(freqs.unsqueeze(0) * reshaped_coords, min=1e-5)  # epsilon to help with stability at 0

    sin_chunk, cos_chunk, exp_chunk, log1p_chunk = updated_coords.split(num_harmonics * N, dim=1)

    encodings = torch.cat([
        torch.sin(sin_chunk * 2 * np.pi),
        torch.cos(cos_chunk * 2 * np.pi),
        torch.exp(exp_chunk),
        torch.log1p(log1p_chunk),
    ], dim=1)

    return encodings


def compute_sinexp_encoding(coords, target_dim, freqs=None):
    cycle_length = 2
    N = int(coords.shape[1])

    num_harmonics = int(math.ceil(target_dim / (cycle_length * N)))
    if freqs is None:
        freqs = torch.arange(1, num_harmonics + 1, device=coords.device).repeat_interleave(N).repeat(cycle_length)
        freqs = freqs[:target_dim]  # Ensure we only take as many frequencies as needed

    reshaped_coords = coords.repeat([1, math.ceil(target_dim / N)])
    updated_coords = torch.clamp(freqs.unsqueeze(0) * reshaped_coords, min=1e-5)  # epsilon to help with stability at 0

    sin_chunk, exp_chunk = updated_coords.split(num_harmonics * N, dim=1)

    encodings = torch.cat([
        torch.sin(sin_chunk * 2 * np.pi),
        torch.exp(exp_chunk),
    ], dim=1)

    return encodings


def compute_coslog_encoding(coords, target_dim, freqs=None):
    cycle_length = 2
    N = int(coords.shape[1])

    num_harmonics = int(math.ceil(target_dim / (cycle_length * N)))
    if freqs is None:
        freqs = torch.arange(1, num_harmonics + 1, device=coords.device).repeat_interleave(N).repeat(cycle_length)
        freqs = freqs[:target_dim]  # Ensure we only take as many frequencies as needed

    reshaped_coords = coords.repeat([1, math.ceil(target_dim / N)])
    updated_coords = torch.clamp(freqs.unsqueeze(0) * reshaped_coords, min=1e-5)  # epsilon to help with stability at 0

    cos_chunk, log1p_chunk = updated_coords.split(num_harmonics * N, dim=1)

    encodings = torch.cat([
        torch.cos(cos_chunk * 2 * np.pi),
        torch.log1p(log1p_chunk),
    ], dim=1)

    return encodings


def compute_analytic_encoding(coords, target_dim, freqs=None, encoding_cycle=["sin", "cos", "exp", "log1p"]):
    cycle_length = len(encoding_cycle)
    N = int(coords.shape[1])

    num_harmonics = int(math.ceil(target_dim / (cycle_length * N)))
    if freqs is None:
        freqs = torch.arange(1, num_harmonics + 1, device=coords.device).repeat_interleave(N).repeat(cycle_length)
        freqs = freqs[:target_dim]  # Ensure we only take as many frequencies as needed

    reshaped_coords = coords.repeat([1, math.ceil(target_dim / N)])
    updated_coords = freqs.unsqueeze(0) * reshaped_coords

    encodings = []
    chunk_size = num_harmonics * N
    chunks = updated_coords.split(chunk_size, dim=1)

    for i, func in enumerate(encoding_cycle):
        chunk = chunks[i]
        if func == "sin":
            encodings.append(torch.sin(chunk * 2 * np.pi))
        elif func == "cos":
            encodings.append(torch.cos(chunk * 2 * np.pi))
        elif func == "exp":
            encodings.append(torch.exp(chunk))
        elif func == "log1p":
            chunk = torch.nn.functional.softplus(chunk)  # Ensure positivity for stability
            encodings.append(torch.log1p(chunk))
        else:
            raise ValueError(f"Unknown encoding function: {func}")
    return torch.cat(encodings, dim=1)

def compute_helmholtz_encoding(coords, target_dim, wavevectors):
    transform = (coords * 2 * np.pi) @ wavevectors.T
    return torch.sin(transform)

def compute_full_helmholtz_encoding(coords, target_dim, wavevectors):
    transform = (coords * 2 * np.pi) @ wavevectors.T
    sin_term = torch.sin(transform).unsqueeze(-1)
    cos_term = torch.cos(transform)

    k_norm = torch.linalg.norm(wavevectors, dim=1, keepdim=True).clamp(min=1e-5)
    k_rot  = torch.stack([-wavevectors[:, 1], wavevectors[:, 0]], dim=1) / k_norm
    v_k    = (cos_term[:, :, None] * wavevectors) / k_norm
    v_krot = cos_term[:, :, None] * k_rot
    return torch.cat([sin_term, v_k, v_krot], dim=-1)

def compute_linear_encoding(x, target_dim):
    return x.repeat(1, target_dim // x.shape[-1])[:, :target_dim]


def compute_polynomial_encoding(x, max_degree):
    out = []
    for i in range(1, max_degree + 1):
        out.append(x ** i)
    return torch.cat(out, dim=-1)


def compute_gaussian_encoding(x, target_dim, std=10.0, seed=42):
    generator = torch.Generator(device=x.device).manual_seed(seed)
    B = torch.randn(x.shape[1], target_dim // 2, generator=generator, device=x.device) * std
    x_proj = 2 * torch.pi * x @ B  # [B, F]
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def compute_targeted_encodings(x, target_dim, scheme="spiral", include_raw=False, freqs=None, seed=42, encoding_cycle=None):
    _, N = x.shape
    encodings = []

    if include_raw:
        if scheme == "full_helmholtz":
            target = N * 2 + 1
            encodings.append(x.repeat(target))
        encodings.append(x)
        target_dim -= N
    target_dim = int(target_dim)

    if target_dim > 0:
        if scheme in ["spiral", "sinusoidal", "mathy", "less_mathy", "sinexp", "coslog", "analytic", "helmholtz", "full_helmholtz"]:
            encoding_fn = {
                "spiral": compute_spiral_encoding,
                "sinusoidal": compute_sinusoidal_encoding,
                "mathy": compute_mathy_encoding,
                "less_mathy": compute_less_mathy_encoding,
                "sinexp": compute_sinexp_encoding,
                "coslog": compute_coslog_encoding,
                "analytic": partial(compute_analytic_encoding, encoding_cycle=encoding_cycle),
                "helmholtz": compute_helmholtz_encoding,
                "full_helmholtz": compute_full_helmholtz_encoding,
            }[scheme]
            encodings.append(encoding_fn(x, target_dim, freqs))
        elif scheme == "gaussian":
            encodings.append(compute_gaussian_encoding(x, target_dim, seed=seed))
        elif scheme == "linear":
            encodings.append(compute_linear_encoding(x, target_dim))
        elif scheme == "polynomial":
            deg = target_dim // x.shape[-1]
            encodings.append(compute_polynomial_encoding(x, deg))
        elif scheme is None:
            encodings.append(torch.zeros(x.shape[0], target_dim, device=x.device))
        else:
            raise ValueError(f"Unknown encoding scheme: {scheme}")
    return torch.cat(encodings, dim=-1)


class SineLayer(nn.Module):
    """Siren activation function."""
    def __init__(self, in_features, out_features, is_first=False, omega=30.0):
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
