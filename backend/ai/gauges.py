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
            nn.Sigmoid(),  # TODO: replace. This breaks divergence free, but I still need 0-1 range
        )
    
    def decode(self, packed_latent, pos, time):
        ω, sinφ, cosφ, noncon = self._split_latent(packed_latent)
        
        # Concatenate the unpacked components with the position and time
        x_in = torch.cat([ω, sinφ, cosφ, noncon, pos, time], dim=-1)
        
        # Pass through the decoder
        return self.decoder(x_in)

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

    def advance_latent(self, z, dt):
        """
        z  : (B, latent_dim)   current latent
        dt : float or tensor   time step
        returns new latent (B, latent_dim)
        """
        ω, sinφ, cosφ, noncon = self._split_latent(z)
        dt = torch.as_tensor(dt, device=z.device).view(-1, 1)  # broadcast
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


def calculate_losses(model, dct_loss, mse_loss, image, image_next, pos, time, dt):
    """
    Calculate the reconstruction and prediction losses for the model.
    """
    reconstructed_image, latent = model(image, pos, time)
    reconstructed_next, latent_next = model(image_next, pos, time + dt)

    reconstruction_error = (dct_loss(reconstructed_image, image) + mse_loss(reconstructed_image, image)) / 2.0
    reconstructed_next_error = (dct_loss(reconstructed_next, image_next) + mse_loss(reconstructed_next, image_next)) / 2.0
    full_reconstruction_error = (reconstruction_error + reconstructed_next_error) / 2.0

    predicted_latent = model.advance_latent(latent, dt)
    predicted_image = model.decoder(predicted_latent)
    prediction_loss = (dct_loss(predicted_image, image_next) + mse_loss(predicted_image, image_next)) / 2.0

    rewound_latent = model.advance_latent(latent_next, -dt)
    rewound_image = model.decoder(rewound_latent)
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
    for i in range(N):
        advanced_latent = model.advance_latent(advanced_latent, dt)
        predicted_image = model.decoder(advanced_latent).view(-1, *shape)
        frames.append(predicted_image)

        predicted_image_reshaped = predicted_image.permute(0, 3, 1, 2)
        next_frame_data = frame_reference[min(idx + i + 1, len(frame_reference) - 1)]
        next_test_image = next_frame_data[:][0].view(-1, *shape).permute(0, 3, 1, 2)
        ssim_predicted = ssim(predicted_image_reshaped, next_test_image)
        ssim_values.append(ssim_predicted.item())

    rewound_latent = latent.detach().clone()
    for i in range(N):
        rewound_latent = model.advance_latent(rewound_latent, -dt)
        rewound_image = model.decoder(rewound_latent).view(-1, *shape)
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
    print(f"Frame sizes after squeeze: {[frame.shape for frame in squeezed_frames]}")
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
    model = DrillNet(**decoder_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    log_epochs = list(range(0, 11, 1)) + list(range(10, 51, 5)) + list(range(50, 101, 10)) + list(range(100, 501, 50)) + list(range(500, 1001, 100))
        
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_epoch(model, training_dataloader, optimizer, mse_loss, dct_loss, dt)
        if epoch in log_epochs:
            log_epoch(model, frame_reference, shape, dt, epoch, experiment_name)
