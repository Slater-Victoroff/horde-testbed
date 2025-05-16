import os
import imageio
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from image_utils import load_images
from encoding_utils import compute_targeted_encodings
from losses import DCTLoss
from piq import ssim


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
            nn.GELU(),
        )

        # self.Aφ_decoder = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.k * self.g),
        # )

        # This is our gluing function across the Aφ fiber bundles
        self.Aφ_incrementers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.g + 2 * self._num_dims, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.g)
            )
            for _ in range(self._num_dims)
        ])

        # # Create one ψ-decoder per Aωφ field (9 in total: 3 for x, y, t * 3)
        # self.ψ_decoders = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(9, 3),  # Each ψ-decoder processes 9 inputs to produce 3 outputs
        #     )
        #     for _ in range(self.k)
        # ])

        self.image_decoder = nn.Sequential(
            # Each harmonic ultimately encodes A * sinφ, A * cosφ, then plus raw toroidal coords
            nn.Linear(self.k * (2 * self._num_dims) + self._num_dims, self.hidden_dim),
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
            x_in = torch.cat([bundle, dx, dy, dt, x, y, t], dim=-1)
            delta = incrementer(x_in)  # shape (B, g)

            updated_bundle = bundle.clone()

            for h in range(self.k):
                idx = h * 2  # (A, φ) per harmonic
                A, φ = bundle[:, idx:idx+1], bundle[:, idx+1:idx+2]
                dA, dφ_residual = delta[:, idx:idx+1], delta[:, idx+1:idx+2]

                ω_axis = self.ω_tensor[axes[i]][h].view(1, 1)

                # structured φ update along axis + residual elsewhere
                φ_new = φ + ω_axis * d_axis + dφ_residual * norm_other
                A_new = A + dA

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
        latent_fiber = self.fiber_encoder(x_in)
        reconstructed_image = self.decode(latent_fiber, pos, time)
        return reconstructed_image, latent_fiber


def calculate_losses(model, dct_loss, mse_loss, image, image_next, pos, time, dt):
    """
    Calculate the reconstruction and prediction losses for the model.
    """
    reconstructed_image, latent = model(image, pos, time)
    reconstructed_next, latent_next = model(image_next, pos, time + dt)

    reconstruction_error = (dct_loss(reconstructed_image, image) + mse_loss(reconstructed_image, image)) / 2.0
    reconstructed_next_error = (dct_loss(reconstructed_next, image_next) + mse_loss(reconstructed_next, image_next)) / 2.0
    full_reconstruction_error = (reconstruction_error + reconstructed_next_error) / 2.0

    predicted_latent = model.increment_latent_time(latent, dt, pos, time)
    predicted_image = model.decode(predicted_latent, pos, time + dt)
    prediction_loss = (dct_loss(predicted_image, image_next) + mse_loss(predicted_image, image_next)) / 2.0

    rewound_latent = model.increment_latent_time(latent_next, -dt, pos, time)
    rewound_image = model.decode(rewound_latent, pos, time - dt)
    rewound_loss = (dct_loss(rewound_image, image) + mse_loss(rewound_image, image)) / 2.0

    full_prediction_error = (prediction_loss + rewound_loss) / 2.0

    loss = (full_reconstruction_error + 2 * full_prediction_error) / 3.0  # weighted toward prediction
    return loss

def create_frame_reference(image_tensor, norm_pos, time, unique_time):
    return [
        torch.utils.data.TensorDataset(
            image_tensor[time.squeeze() == t],
            norm_pos[time.squeeze() == t],
            time[time.squeeze() == t]
        )
        for t in unique_time
    ]


def calculate_time_step(unique_time):
    dt = torch.mean(torch.diff(unique_time, dim=0))
    if not torch.allclose(torch.diff(unique_time, dim=0), dt.expand_as(torch.diff(unique_time, dim=0)), atol=1e-6):
        raise ValueError("Time steps are not evenly spaced.")
    return dt


def train_epoch(model, dataloader, optimizer, mse_loss, dct_loss, dt):
    model.train()
    epoch_loss = 0.0
    for batch_num, (image, image_next, pos, time) in enumerate(dataloader, start=1):
        if batch_num % 100 == 0:
            print(f"Batch {batch_num}/{len(dataloader)}")
        loss = calculate_losses(model, dct_loss, mse_loss, image, image_next, pos, time, dt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch Loss: {epoch_loss:.6f}")


def log_epoch(model, frame_reference, shape, dt, epoch, experiment_name):
    model.eval()
    with torch.no_grad():
        evaluate_model(model, frame_reference, shape, dt, epoch, experiment_name)


def evaluate_model(model, frame_reference, shape, dt, epoch, experiment_name):
    n_test_frames = 5
    total_frames = len(frame_reference)
    frame_indices = torch.linspace(0, total_frames - 1, steps=n_test_frames).long()

    for idx in frame_indices:
        frame_data = frame_reference[idx]
        test_image, test_pos, test_time = frame_data[:]
        reconstructed, latent = model(test_image, test_pos, test_time)

        ssim_values, frames = simulation_test(model, frame_reference, shape, dt, idx, latent, reconstructed, test_image)
        write_metrics_and_gif(ssim_values, frames, f"gauge_dir/{experiment_name}", epoch, idx)


def simulation_test(model, frame_reference, shape, dt, idx, latent, reconstructed, test_image):
    ssim_values = []
    test_image_reshaped = test_image.view(-1, *shape).permute(0, 3, 1, 2)
    reconstructed = reconstructed.view(-1, *shape)
    frames = [reconstructed]
    reconstructed_reshaped = reconstructed.permute(0, 3, 1, 2)
    ssim_reconstructed = ssim(reconstructed_reshaped, test_image_reshaped)
    ssim_values.append(ssim_reconstructed.item())

    N = 10
    advanced_latent = latent.detach().clone()
    rewound_latent = latent.detach().clone()

    H, W, _ = shape
    device = test_image.device
    x_lin = torch.linspace(0, 1, W, device=device)
    y_lin = torch.linspace(0, 1, H, device=device)
    x, y = torch.meshgrid(x_lin, y_lin, indexing='xy')
    pos = torch.stack([x.flatten(), y.flatten()], dim=-1)

    # Set initial time from frame index
    total_frames = len(frame_reference)
    t_init = idx / total_frames
    time = torch.full((H * W, 1), t_init, device=device)

    for i in range(N):
        time = time + dt
        advanced_latent = model.increment_latent_time(advanced_latent, dt, pos, time)
        predicted_image = model.decode(advanced_latent, pos, time).view(-1, *shape)
        frames.append(predicted_image)

        predicted_image_reshaped = predicted_image.permute(0, 3, 1, 2)
        next_frame_data = frame_reference[min(idx + i + 1, len(frame_reference) - 1)]
        next_test_image = next_frame_data[:][0].view(-1, *shape).permute(0, 3, 1, 2)
        ssim_predicted = ssim(predicted_image_reshaped, next_test_image)
        ssim_values.append(ssim_predicted.item())

    time = torch.full((H * W, 1), t_init, device=device)
    for i in range(N):
        time = time - dt
        rewound_latent = model.increment_latent_time(rewound_latent, -dt, pos, time)
        rewound_image = model.decode(rewound_latent, pos, time).view(-1, *shape)
        frames.insert(0, rewound_image)

        rewound_image_reshaped = rewound_image.permute(0, 3, 1, 2)
        prev_frame_data = frame_reference[max(idx - i - 1, 0)]
        prev_test_image = prev_frame_data[:][0].view(-1, *shape).permute(0, 3, 1, 2)
        ssim_rewound = ssim(rewound_image_reshaped, prev_test_image)
        ssim_values.insert(0, ssim_rewound.item())
    return ssim_values, frames

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
    squeezed_frames = [frame.squeeze(0) for frame in frames]  # Remove the prepended dimension
    gif_path = os.path.join(epoch_dir, f"debug_{idx}.gif")
    imageio.mimsave(
        gif_path,
        [(frame.cpu().numpy() * 255).astype(np.uint8) for frame in squeezed_frames],
        fps=10
    )


def train_drill_model(image_dir, device='cuda', epochs=100, batch_size=8192, experiment_name=None, decoder_config=None):
    image_tensor, image_tensor_next, norm_pos, control_tensor, shape = load_images(image_dir, device, norm=True, add_next=True)
    time = control_tensor[:, 0:1]
    unique_time = torch.unique(time)
    frame_reference = create_frame_reference(image_tensor, norm_pos, time, unique_time)

    dt = calculate_time_step(unique_time)
    dataset = torch.utils.data.TensorDataset(image_tensor, image_tensor_next, norm_pos, time)
    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mse_loss = nn.MSELoss().to(device)
    dct_loss = DCTLoss().to(device)
    model = DrillNet2(**decoder_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    log_epochs = list(range(0, 11, 1)) + list(range(10, 51, 5)) + list(range(50, 101, 10)) + list(range(100, 501, 50)) + list(range(500, 1001, 100))
        
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_epoch(model, training_dataloader, optimizer, mse_loss, dct_loss, dt)
        if epoch in log_epochs:
            log_epoch(model, frame_reference, shape, dt, epoch, experiment_name)
