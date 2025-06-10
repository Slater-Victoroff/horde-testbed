import os
import math
import imageio
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from image_utils import load_images
from encoding_utils import compute_targeted_encodings
from losses import DCTLoss
from piq import SSIMLoss

from soap import SOAP
from losses import SpectralLoss, calculate_local_loss, pos_epoch

def rotate_phase(sinφ, cosφ, ω, dt):
    """
    Exact phase rotation:
        [sinφ, cosφ] · R(ω·dt)
    Args
    ----
    sinφ, cosφ : (B,k)   current phase components
    ω          : (B,k)   frequency (≥0, already softplused)
    dt         : scalar or (B,1) or (B,k)
    """
    sin_dt, cos_dt = torch.sin(ω * dt), torch.cos(ω * dt)
    sin_next = sinφ * cos_dt + cosφ * sin_dt
    cos_next = cosφ * cos_dt - sinφ * sin_dt
    return sin_next, cos_next


class SafeBoundedOutput(nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.register_buffer('min_val', torch.tensor(min_val))
        self.register_buffer('max_val', torch.tensor(max_val))

    def forward(self, x):
        # Smooth nonlinearity that preserves gradient structure
        x = torch.tanh(x)
        # Scale + shift to [min_val, max_val]
        x = 0.5 * (x + 1.0)  # to [0,1]
        return torch.clamp(x, self.min_val, self.max_val)


class ModulatedIncrementer(nn.Module):
    def __init__(self, gauge_dim, hidden_dim, cond_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gauge_dim = gauge_dim
        self.cond_dim = cond_dim

        self.fc1 = nn.Linear(gauge_dim + cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, gauge_dim)


class DrillNet(nn.Module):
    def __init__(self, conserved_cycles=3, nc_terms=3, hidden_dim=64, input_channels=4):
        super().__init__()
        self.k = conserved_cycles
        latent_channels = conserved_cycles * 3 + nc_terms
        self.all_inputs = input_channels + 6  # 4 for pos, 2 for time
        self.encoder = nn.Sequential(
            nn.Linear(self.all_inputs, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_channels),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_channels),
            SafeBoundedOutput(min_val=0.0, max_val=1.0)
        )
    
    def decode(self, packed_latent, pos, time):
        return self.decoder(packed_latent)

    def _split_latent(self, z):
        idx   = 0
        ω_raw = z[:, idx:idx+self.k];   idx += self.k
        sinφ  = z[:, idx:idx+self.k];   idx += self.k
        cosφ  = z[:, idx:idx+self.k];   idx += self.k
        noncon = z[:, idx:]

        ω = F.softplus(ω_raw)      # keep ≥0
        norm = torch.rsqrt(sinφ**2 + cosφ**2 + 1e-6)
        sinφ, cosφ = sinφ * norm, cosφ * norm
        return ω, sinφ, cosφ, noncon

    def _pack_latent(self, ω, sinφ, cosφ, noncon):
        return torch.cat([ω, sinφ, cosφ, noncon], dim=1)

    def increment_latent_time(self, latent, dt, pos, time):
        """
        z  : (B, latent_dim)   current latent
        dt : float or tensor   time step
        returns new latent (B, latent_dim)
        """
        ω, sinφ, cosφ, noncon = self._split_latent(latent)
        dt = torch.as_tensor(dt, device=latent.device).view(-1, 1)  # broadcast
        sinφ_next, cosφ_next = rotate_phase(sinφ, cosφ, ω, dt)
        noncon_next = (noncon + dt) % 1.0
        return self._pack_latent(ω, sinφ_next, cosφ_next, noncon_next)

    def forward(self, color, pos, time, baked_latent=None):
        """
        Color, pos, time: all assumed to be 0-1 normalized.
        """
        torus_pos = compute_targeted_encodings(pos, 4, scheme="sinusoidal", norm_2pi=True)
        torus_time = compute_targeted_encodings(time, 2, scheme="sinusoidal", norm_2pi=True)

        x_in = torch.cat([color, torus_pos, torus_time], dim=-1)

        latent = self.encoder(x_in) if baked_latent is None else baked_latent
        return self.decoder(latent), latent


class AφIncrementer(nn.Module):
    def __init__(self, gauge_dim, hidden_dim, cond_dim):
        super().__init__()
        self.gauge_dim = gauge_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        # Explicit trunk: gauge state → hidden → gauge increment
        self.trunk_fc1 = nn.Linear(gauge_dim, hidden_dim)
        self.trunk_fc2 = nn.Linear(hidden_dim, gauge_dim)

        # Explicit FiLM conditioning: cond → scale + shift
        self.film = nn.Linear(cond_dim, hidden_dim * 2)

        self.activation = nn.GELU()
        self.output_activation = nn.Tanh()

    def forward(self, gauge, cond):
        """
        gauge: [B, gauge_dim] (your latent gauge state)
        cond:  [B, cond_dim] (dx, dy, dt, x, y, t conditions)
        """

        # Explicit trunk forward pass
        hidden = self.activation(self.trunk_fc1(gauge))  # [B, hidden_dim]

        # Explicit FiLM modulation
        scale_shift = self.film(cond)  # [B, 2 * hidden_dim]
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)  # Each [B, hidden_dim]

        # Apply FiLM explicitly
        # hidden = hidden * (1 + torch.tanh(scale)) + torch.tanh(shift)
        hidden = hidden * scale + shift

        # Final trunk projection explicitly
        increment = self.output_activation(self.trunk_fc2(hidden))  # [B, gauge_dim]

        return increment


class DrillNet2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.latent_dim = kwargs.get('latent_dim', 12)
        self._num_dims = 3  # x, y, t
        self.k = kwargs.get('num_harmonics', 3)
        self.g = kwargs.get('gauge_dim', 2 * self._num_dims)  # A φ

        self.ω_tensor = nn.ParameterDict({
            axis: nn.Parameter(torch.tensor([2**i for i in range(self.k)], dtype=torch.float32))
            for axis in ['x', 'y', 't']
        })

        self.hidden_dim = kwargs.get('hidden_dim', 64)
        self.input_channels = kwargs.get('input_channels', 4)
        self.decoder_config = kwargs

        self.fiber_encoder = nn.Sequential(
            nn.Linear(self.input_channels + self._num_dims, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.k * self.g),
            nn.Tanh(),
        )

        # This is our gluing function across the Aφ fiber bundles
        # self.Aφ_incrementers = nn.ModuleList([
        #     AφIncrementer(gauge_dim=self.g, hidden_dim=self.hidden_dim, cond_dim=2 * self._num_dims)
        #     for _ in range(self._num_dims)
        # ])
        self.Aφ_incrementers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.g + 2 * self._num_dims, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.g),
                nn.Tanh(),
            )
            for _ in range(self._num_dims)
        ])

        self.image_decoder = nn.Sequential(
            # Each harmonic ultimately encodes A * sinφ, A * cosφ, then plus raw toroidal coords
            nn.Linear(self.k * (2 * self._num_dims) + self._num_dims, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.input_channels),
            SafeBoundedOutput(min_val=0.0, max_val=1.0)
        )

    def encode_raw(self, color, pos, time):
        x_in = torch.cat([color, pos, time], dim=-1)
        return self.encoder(x_in)

    def transport_bundles(self, bundles, dx, dy, dt, x, y, t):
        """
        Transport (advance) the latent bundles using the dx, dy, dt offsets.
        Performs structured cyclic φ updates and learned amplitude increments.
        """

        dx = dx.unsqueeze(-1) if dx.dim() == 1 else dx
        dy = dy.unsqueeze(-1) if dy.dim() == 1 else dy
        dt = dt.unsqueeze(-1) if dt.dim() == 1 else dt

        delta_pos = [dx, dy, dt]
        axes = ['x', 'y', 't']
        dim_bundles = torch.chunk(bundles, chunks=self._num_dims, dim=-1)

        new_bundles = []
        for i, (bundle, incrementer) in enumerate(zip(dim_bundles, self.Aφ_incrementers)):
            d_axis = delta_pos[i]
            d_others = torch.cat([delta_pos[j] for j in range(3) if j != i], dim=-1)
            norm_other = torch.norm(d_others, dim=-1, keepdim=True)

            # Learned residual updates (ΔA, Δφ_residual)
            # x_in = torch.cat([dx, dy, dt, x, y, t], dim=-1)
            # delta = incrementer(bundle, x_in)  # shape (B, g)
            x_in = torch.cat([bundle, dx, dy, dt, x, y, t], dim=-1)
            delta = incrementer(x_in)  # shape (B, g)

            updated_bundle = bundle.clone()

            for h in range(self.k):
                idx = h * 2  # (A, φ) per harmonic
                A, φ = bundle[:, idx:idx+1], bundle[:, idx+1:idx+2]
                dA, dφ_residual = delta[:, idx:idx+1], delta[:, idx+1:idx+2]
                dφ_residual = dφ_residual * math.pi
                ω_axis = self.ω_tensor[axes[i]][h].view(1, 1)

                # structured φ update along axis + residual elsewhere
                φ_raw = φ + ω_axis * d_axis + dφ_residual * norm_other
                φ_new = (φ_raw + math.pi) % (2*math.pi) - math.pi
                A_new = torch.tanh(A + dA)

                updated_bundle[:, idx] = A_new.squeeze(-1)
                updated_bundle[:, idx+1] = φ_new.squeeze(-1)

            new_bundles.append(updated_bundle)

        return torch.cat(new_bundles, dim=-1)  # (B, k*g)

    def increment_latent_time(self, latent_fiber, dt, pos, time):
        """
        Advance the bundles in time using the dt offset.
        """
        B = latent_fiber.shape[0]
        device = latent_fiber.device

        dt_tensor = torch.full((B, 1), dt, device=device)
        dx = torch.zeros((B, 1), device=device)
        dy = torch.zeros((B, 1), device=device)

        x, y, t = pos[:, 0:1], pos[:, 1:2], time

        incremented_bundle = self.transport_bundles(latent_fiber, dx, dy, dt_tensor, x, y, t)
        return incremented_bundle

    def decode(self, latent_fiber, pos, time):
        """
        Decode the bundles into the final image.
        """
        # Unpack the bundles into their respective components
        Aφ_bundles = torch.chunk(latent_fiber, chunks=self._num_dims, dim=-1)
        expanded_bundle = []

        for bundle in Aφ_bundles:  # x, y, t
            for h in range(self.k):
                idx = h * 2
                A, φ = bundle[:, idx:idx+1], bundle[:, idx+1:idx+2]
                sinφ = torch.sin(φ)
                cosφ = torch.cos(φ)
                expanded_bundle.append(A * sinφ)
                expanded_bundle.append(A * cosφ)
        expanded_bundle = torch.cat(expanded_bundle, dim=-1)
        # Concatenate the expanded bundle with the position and time
        x_in = torch.cat([expanded_bundle, pos, time], dim=-1)
        reconstructed_image = self.image_decoder(x_in)
        return reconstructed_image

    def forward(self, color, pos, time):
        x_in = torch.cat([color, pos, time], dim=-1)
        raw_fiber = self.fiber_encoder(x_in)
        B = raw_fiber.shape[0]
        # reshape to [B, dims=3, harmonics=k, 2 channels=(A,φ)]
        bundles = raw_fiber.view(B, self._num_dims, self.k, 2)
        A   = bundles[..., 0]                    # amplitudes
        φ_raw = bundles[..., 1]                    # phase
        φ = φ_raw * math.pi
        φ = (φ + math.pi) % (2*math.pi) - math.pi
        # re-pack into your “latent_fiber”
        latent_fiber = torch.stack([A, φ], dim=-1)   # (B,3,k,2)
        latent_fiber = latent_fiber.view(B, -1)      # (B, k*g)
        reconstructed_image = self.decode(latent_fiber, pos, time)
        return reconstructed_image, latent_fiber


def get_total_grad_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def create_frame_reference(image_tensor, norm_pos, time, unique_time):
    return [
        torch.utils.data.TensorDataset(
            image_tensor[time.squeeze() == t],
            norm_pos[time.squeeze() == t],
            time[time.squeeze() == t]
        )
        for t in unique_time
    ]


def calculate_step(unique_time):
    dt = torch.mean(torch.diff(unique_time, dim=0))
    if not torch.allclose(torch.diff(unique_time, dim=0), dt.expand_as(torch.diff(unique_time, dim=0)), atol=1e-6):
        raise ValueError("Time steps are not evenly spaced.")
    return dt


def pixel_epoch(model, dataloader, optimizer, mse_loss, dct_loss, dt):
    model.train()
    epoch_loss = 0.0
    for batch_num, (image, image_next, pos, time) in enumerate(dataloader, start=1):
        if batch_num % 100 == 0:
            print(f"Batch {batch_num}/{len(dataloader)}")
        
        loss = calculate_local_loss(model, dct_loss, mse_loss, image, image_next, pos, time, dt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch Loss: {epoch_loss:.6f}")
    return epoch_loss


def time_epoch(model, frame_reference, shape, dt, epoch, experiment_name, ssim_loss, spectral_loss, update=True, optimizer=None):
    if not update:
        model.eval()
        with torch.no_grad():
            evaluate_model(model, frame_reference, shape, dt, epoch, experiment_name, ssim_loss=ssim_loss, spectral_loss=spectral_loss, write_gif=True)
    else:
        model.train()
        return evaluate_model(model, frame_reference, shape, dt, epoch, experiment_name, ssim_loss, spectral_loss=spectral_loss, write_gif=False, update=True, optimizer=optimizer)


def evaluate_model(model, frame_reference, shape, dt, epoch, experiment_name, ssim_loss, spectral_loss, write_gif=False, update=False, optimizer=None):
    n_test_frames = 10
    total_frames = len(frame_reference)
    if not update:
        frame_indices = torch.linspace(0, total_frames - 1, steps=n_test_frames).long()
    else:
        frame_indices = torch.randint(0, total_frames, (n_test_frames,))

    total_loss = 0.0
    losses = []
    for idx in frame_indices:
        frame_data = frame_reference[idx]
        test_image, test_pos, test_time = frame_data[:]
        reconstructed, latent = model(test_image, test_pos, test_time)

        ssim_values, frames= simulation_test(model, frame_reference, shape, dt, idx, latent, reconstructed, test_image, ssim_loss, spectral_loss, update, optimizer)
        if write_gif:
            write_metrics_and_gif(ssim_values, frames, f"gauge_dir/{experiment_name}", epoch, idx)
        added_loss = sum(ssim_values) / len(ssim_values)
        total_loss += added_loss
        losses.append(added_loss)
    print(f"Epoch {epoch} Loss: {total_loss:.6f}")
    return total_loss


def simulation_test(model, frame_reference, shape, dt, idx, latent, reconstructed, test_image, ssim_loss, spectral_loss, update=False, optimizer=None):
    H, W, C = shape
    flat_img = test_image.view(-1, *shape).permute(0,3,1,2).reshape(-1, C)
    if update and optimizer is not None:
        optimizer.zero_grad()

    ssim_values = []
    frames = [reconstructed.view(-1, *shape)]

    # Reshape and calculate initial metrics
    test_image_reshaped, reconstructed_reshaped = reshape_images(test_image, reconstructed, shape)
    ssim_reconstructed, spectral_reconstructed = calculate_losses(
        reconstructed_reshaped, test_image_reshaped, ssim_loss, spectral_loss
    )
    ssim_values.append(ssim_reconstructed.item())
    if update and optimizer is not None:
        recon_loss = 0.5 * (ssim_reconstructed + spectral_reconstructed)
        optimizer.zero_grad()
        recon_loss.backward(retain_graph=True)
        optimizer.step()

    # Generate advanced frames
    advanced_ssim_values, advanced_frames = generate_frames(
        model, frame_reference, shape, dt, idx, latent, ssim_loss, spectral_loss, update, optimizer, forward=True
    )
    ssim_values.extend(advanced_ssim_values)
    frames.extend(advanced_frames)

    # Generate rewound frames
    rewound_ssim_values, rewound_frames = generate_frames(
        model, frame_reference, shape, dt, idx, latent, ssim_loss, spectral_loss, update, optimizer, forward=False
    )
    ssim_values = rewound_ssim_values + ssim_values
    frames = rewound_frames + frames

    return ssim_values, frames


def reshape_images(test_image, reconstructed, shape):
    test_image_reshaped = test_image.view(-1, *shape).permute(0, 3, 1, 2)
    reconstructed_reshaped = reconstructed.view(-1, *shape).permute(0, 3, 1, 2)
    return test_image_reshaped, reconstructed_reshaped


def calculate_losses(predicted, target, ssim_loss, spectral_loss):
    ssim_value = ssim_loss(predicted, target)
    spectral_value = spectral_loss(predicted, target)
    return ssim_value, spectral_value


def spiral_epoch(model, frame_reference, epoch, dx, dy, dt, mse_loss, dct_loss, optimizer, n_steps, max_step_size, test_pixels=2500):
    """
    Perform spiral learning by selecting random pixels that are at least n_steps in from the edges.
    """
    device = next(model.parameters()).device

    _, norm_pos_sample, _ = frame_reference[0][:]

    # Generate random samples within the valid range
    x_indices = torch.empty(test_pixels, device=device).uniform_(0, 1)
    y_indices = torch.empty(test_pixels, device=device).uniform_(0, 1)
    t_indices = torch.empty(test_pixels, device=device).uniform_(0, 1)

    # Precompute frame times as a tensor
    frame_times = torch.tensor([frame[:][2][0].item() for frame in frame_reference], device=device)

    # Find the closest frame for each t in valid_t_indices
    time_differences = torch.abs(frame_times.unsqueeze(0) - t_indices.unsqueeze(1))  # Shape: (len(valid_t_indices), len(frame_times))
    closest_frame_indices = torch.argmin(time_differences, dim=1)  # Shape: (len(valid_t_indices),)

    # Gather the corresponding frame data
    frame_data = torch.stack([frame_reference[idx][:][0] for idx in closest_frame_indices])  # Shape: (len(valid_t_indices), ...)

    # Compute distances between norm_pos_sample and (x, y) pairs
    positions = torch.stack([x_indices, y_indices], dim=1)  # Shape: (len(valid_x_indices), 2)
    distances = torch.cdist(norm_pos_sample, positions)  # Shape: (len(norm_pos_sample), len(valid_x_indices))
    closest_pixel_indices = torch.argmin(distances, dim=0)  # Shape: (len(valid_x_indices),)

    # Gather closest pixels
    closest_pixels = norm_pos_sample[closest_pixel_indices]  # Shape: (len(valid_x_indices), 2)

    # Combine colors, positions, and times
    colors = frame_data[torch.arange(len(closest_pixel_indices)), closest_pixel_indices]  # Shape: (len(valid_x_indices), ...)
    times = t_indices.unsqueeze(1)  # Shape: (len(valid_t_indices), 1)
    base_pixels = torch.cat([colors, closest_pixels, times], dim=1)  # Shape: (len(valid_x_indices), ...)

    gauge = model.fiber_encoder(base_pixels)
    x, y, t = base_pixels[:, 4:5], base_pixels[:, 5:6], base_pixels[:, 6:7]

    B = base_pixels.shape[0]
    device = base_pixels.device
    dt_tensor = torch.full((B, 1), dt, device=device)
    dx_tensor = torch.full((B, 1), dx, device=device)
    dy_tensor = torch.full((B, 1), dy, device=device)

    total_loss = 0.0
    max_loss = 0.0
    for _ in range(n_steps):
        signs = torch.randint(0, 2, (B,3), device=device, dtype=torch.float32)*2 - 1  
        sx, sy, st = signs[:,0:1], signs[:,1:2], signs[:,2:3]
        multipliers = torch.randint(0, max_step_size + 1, (B, 3), device=device)

        signed_dx = dx_tensor * multipliers[:, 0:1] * sx
        signed_dy = dy_tensor * multipliers[:, 1:2] * sy
        signed_dt = dt_tensor * multipliers[:, 2:3] * st

        updated_gauge = model.transport_bundles(gauge, signed_dx, signed_dy, signed_dt, x, y, t)
        updated_pixels = increment_base_space(frame_reference, signed_dx, signed_dy, signed_dt, base_pixels)
        updated_pos = updated_pixels[:, 4:6]
        updated_time = updated_pixels[:, 6:7]

        recon = model.decode(updated_gauge, updated_pos, updated_time)
        ref_colors = updated_pixels[:, :4]
        loss = mse_loss(recon, ref_colors)
        if loss > max_loss:
            max_loss = loss
        total_loss += loss

        gauge = updated_gauge
        base_pixels = updated_pixels
        x, y, t = updated_pos[:, 0:1], updated_pos[:, 1:2], updated_time
    total_loss = total_loss / n_steps
    optimizer.zero_grad()
    total_loss.backward()
    print(f"Grad Norm: {get_total_grad_norm(model):.6f}")
    optimizer.step()
    return total_loss.item(), max_loss.item()

def increment_base_space(frame_reference, dx, dy, dt, base_pixels):
    """
    Increment the base space using the dx, dy, dt offsets and update colors from the frame reference.
    """
    _, norm_pos_sample, _ = frame_reference[0][:]
    B = base_pixels.shape[0]
    device = base_pixels.device

    incremented_base = []
    x, y, t = base_pixels[:, 4:5], base_pixels[:, 5:6], base_pixels[:, 6:7]
    new_x = torch.clamp(x + dx, 0, 1)
    new_y = torch.clamp(y + dy, 0, 1)
    new_t = torch.clamp(t + dt, 0, 1)
    times = torch.tensor([frame[:][2][0].item() for frame in frame_reference], device=new_t.device)
    closest_indices = torch.argmin(torch.abs(new_t - times.unsqueeze(0)), dim=1, keepdim=True)  # Resulting shape (B, 1)

    # Vectorized computation for closest pixels
    new_positions = torch.cat([new_x, new_y], dim=-1)  # Shape: (B, 2)
    distances = torch.cdist(new_positions, norm_pos_sample)
    closest_pixel_indices = torch.argmin(distances, dim=1)  # Shape: (B,)
    positions = norm_pos_sample[closest_pixel_indices]  # Shape: (B, 2)

    # Iterate through time indices and positions to get the right pixel
    for i in range(B):
        time_idx = closest_indices[i].item()
        pos_idx = closest_pixel_indices[i].item()

        # Get the corresponding frame and pixel color
        frame_data = frame_reference[time_idx]
        color = frame_data[:][0][pos_idx]  # Extract the color for the closest pixel

        # Combine color, position, and time into the incremented base
        incremented_base.append(torch.cat([color, positions[i], new_t[i]]))

    incremented_base = torch.stack(incremented_base)
    return incremented_base

def generate_frames(model, frame_reference, shape, dt, idx, latent, ssim_loss, spectral_loss, update, optimizer, forward=True):
    N = 10
    frames = []
    ssim_values = []
    latent_copy = latent.detach().clone()
    H, W, _ = shape
    device = latent.device
    pos, time = initialize_position_and_time(H, W, device, idx, len(frame_reference))
    total_loss = 0.0

    for i in range(N):
        time = time + dt if forward else time - dt
        latent_copy = model.increment_latent_time(latent_copy, dt if forward else -dt, pos, time)
        predicted_image = model.decode(latent_copy, pos, time).view(-1, *shape)

        predicted_reshaped = predicted_image.permute(0, 3, 1, 2)
        frame_data = get_frame_data(frame_reference, idx, i, forward)
        test_image = frame_data[:][0].view(-1, *shape).permute(0, 3, 1, 2)
        ssim_value, spectral_value = calculate_losses(predicted_reshaped, test_image, ssim_loss, spectral_loss)
        if update and optimizer is not None:
            total_loss += 0.5 * (ssim_value + spectral_value)

        if forward:
            frames.append(predicted_image)
            ssim_values.append(ssim_value.item())
        else:
            frames.insert(0, predicted_image)
            ssim_values.insert(0, ssim_value.item())

    if update and optimizer is not None:
        total_loss = total_loss / N
        optimizer.zero_grad()
        total_loss.backward()
        print(f"Grad Norm: {get_total_grad_norm(model):.6f}")
        optimizer.step()
    return ssim_values, frames


def initialize_position_and_time(H, W, device, idx, total_frames):
    x_lin = torch.linspace(0, 1, W, device=device)
    y_lin = torch.linspace(0, 1, H, device=device)
    x, y = torch.meshgrid(x_lin, y_lin, indexing='xy')
    pos = torch.stack([x.flatten(), y.flatten()], dim=-1)
    t_init = idx / total_frames
    time = torch.full((H * W, 1), t_init, device=device)
    return pos, time


def get_frame_data(frame_reference, idx, i, forward):
    if forward:
        return frame_reference[min(idx + i + 1, len(frame_reference) - 1)]
    else:
        return frame_reference[max(idx - i - 1, 0)]


def finalize_loss_and_step(model, total_loss, num_frames, optimizer):
    total_loss = total_loss / num_frames
    total_loss.backward()
    # log_layer_gradients(model)
    print(f"Grad Norm: {get_total_grad_norm(model):.6f}")
    optimizer.step()


def log_layer_gradients(model):
    """
    After loss.backward(), call this to print per-parameter grad norms.
    """
    print("#### Layer gradient norms ####")
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"{name:50s} | no grad")
        else:
            gn = param.grad.data.norm(2).item()
            print(f"{name:50s} | {gn:8.4e}")
    print("#" * 60)


def write_metrics_and_gif(metrics, frames, output_dir, epoch, idx):
    # Ensure the base output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a subdirectory for the specific epoch
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Write metrics to a file in the epoch directory
    metrics_path = os.path.join(epoch_dir, f"metrics_{idx}.txt")
    with open(metrics_path, "w") as f:
        for metric in metrics:
            f.write(f"{metric}\n")

    # Write frames as a GIF in the epoch directory
    squeezed_frames = [frame.squeeze(0)[..., :3] for frame in frames]  # Remove the prepended dimension and keep only the first three channels
    gif_path = os.path.join(epoch_dir, f"debug_{idx}.gif")
    imageio.mimsave(
        gif_path,
        [(frame.detach().cpu().numpy() * 255).astype(np.uint8) for frame in squeezed_frames],
        fps=10
    )


def train_drill_model(image_dir, device='cuda', epochs=10000, batch_size=8192, experiment_name=None, decoder_config=None):
    image_tensor, image_tensor_next, norm_pos, control_tensor, shape = load_images(image_dir, device, norm=True, add_next=True)
    time = control_tensor[:, 0:1]

    unique_x = torch.unique(norm_pos[:, 0])
    dx = calculate_step(unique_x)
    
    unique_y = torch.unique(norm_pos[:, 1])
    dy = calculate_step(unique_y)
    
    unique_time = torch.unique(time)
    dt = calculate_step(unique_time)

    frame_reference = create_frame_reference(image_tensor, norm_pos, time, unique_time)
    print(f"image_tensor shape: {image_tensor.shape}")

    
    dataset = torch.utils.data.TensorDataset(image_tensor, image_tensor_next, norm_pos, time)
    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mse_loss = nn.MSELoss().to(device)
    dct_loss = DCTLoss().to(device)
    ssim_loss = SSIMLoss().to(device)
    spectral_loss = SpectralLoss().to(device)
    model = DrillNet2(**decoder_config).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = SOAP(model.parameters(), lr=5e-5, betas=(0.9, 0.95))
    optimizer = SOAP(model.parameters())
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    # log_epochs = list(range(0, 11, 1)) + list(range(10, 51, 5)) + list(range(50, 101, 10)) + list(range(100, 501, 50)) + list(range(500, 1001, 100)) + list(range(1000, 2501, 250))
    if epochs > 1000:
        log_epochs = list(range(0, epochs, 100))
    curriculum = {
        "pixel_loss": 0,
        "time_loss": 100,
        "pos_loss": 200,
    }

    n_steps = 1
    step_size = 1
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}, N_steps: {n_steps}, Step_size: {step_size}")
        # time_loss = time_epoch(model, frame_reference, shape, dt, epoch, experiment_name, ssim_loss, spectral_loss, update=True, optimizer=optimizer)
        total_loss, max_loss = spiral_epoch(model, frame_reference, epoch, dx, dy, dt, mse_loss, dct_loss, optimizer, n_steps = n_steps, max_step_size=step_size)
        print(f"Total Loss: {total_loss:.6f}, Max Loss: {max_loss:.6f}")
        if max_loss < 0.001:  #arbitrary threshold
            n_steps += 1
            # if n_steps < step_size:
            #     n_steps += 1
            #     print(f"Incrementing n_steps to {n_steps}")
            # else:
            #     step_size += 1
            #     print(f"Incrementing step_size to {step_size}")

        # scheduler.step(frame_loss)
        if epoch in log_epochs:
            time_epoch(model, frame_reference, shape, dt, epoch, experiment_name, ssim_loss, spectral_loss, update=False)
