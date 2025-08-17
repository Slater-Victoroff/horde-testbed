import os
import time
import json
from datetime import datetime

import torch
import torch.profiler as profiler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, TensorDataset
from piq import ssim, SSIMLoss, psnr

from decoders import VFXSpiralNetDecoder, VFXNetPixelDecoder
from losses import DCTLoss, GradientLoss
from make_shader import decoder_to_glsl, compare_decoder_and_shader, save_weights_to_exr
from image_utils import load_images, save_images
from soap import SOAP


LATENT_IMAGE_CHANNELS = 4 # RGBA


class VFXNet(nn.Module):
    def __init__(self, height, width, decoder_config=None, device='cuda', experiment_name=None, freeze_decoder=False):
        super().__init__()
        self.device = device
        self.experiment_name = experiment_name
        self.height = height
        self.width = width

        self.decoder = VFXSpiralNetDecoder(**decoder_config or {})
        if self.decoder.latent_dim > 0:
            self.latent = nn.Embedding(self.height * self.width, LATENT_IMAGE_CHANNELS)
        self.output_channels = self.decoder.output_channels
        self.freeze_decoder = freeze_decoder
        if self.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def forward(self, raw_pos, control):
        if self.decoder.latent_dim > 0:
            return self.decoder(raw_pos, control[:, 0:1], self.latent)
        else:
            return self.decoder(raw_pos, control[:, 0:1])

    def run_patch(self, pos_patch, time_patch):
        flat_raw_pos = pos_patch.view(-1, pos_patch.shape[-1])
        flat_control = time_patch.view(-1, time_patch.shape[-1])

        if self.decoder.latent_dim > 0:
            raise NotImplementedError("Latent images not supported in run_patch")
        else:
            return self.decoder(flat_raw_pos, flat_control).view(*pos_patch.shape[:-1], self.output_channels)

    def full_image(self, time, H=512, W=512):
        x_coords = torch.linspace(0, 1, W, device=self.device)
        y_coords = torch.linspace(0, 1, H, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        raw_pos = torch.stack([grid_x, grid_y], dim=-1)
        expanded_time = time.repeat(H, W, 1)

        if self.decoder.latent_dim > 0:
            print("Again, the world has ended.")
            latent = self.latents(control[:, 1].long())
            latent = latent.view(-1, H, W, LATENT_IMAGE_CHANNELS)
            response = self.decoder(raw_pos, expanded_time, latent)
        else:
            flat_pos = raw_pos.view(-1, 2)
            flat_time = expanded_time.view(-1, 1)
            response = self.decoder(flat_pos, flat_time)
        shaped_image = response.view(H, W, self.output_channels)
        return shaped_image


def compute_ssim_scores(
    reconstructed_batch,
    sampled_controls,
    image_tensor,
    shape,
    control_tensor,
    update=False,
    ssim_loss_fn=None,
    optimizer=None
):
    H, W = shape
    pixels_per_frame = H * W
    T = control_tensor.shape[0] // pixels_per_frame
    ssim_scores = []

    if not update:
        torch.set_grad_enabled(False)  # Ensure gradients off for pure evaluation
    else:
        assert ssim_loss_fn is not None, "Must provide SSIM loss function when update=True"
        torch.set_grad_enabled(True)

    for recon, control in zip(reconstructed_batch, sampled_controls):
        t = control[0].item()
        frame_idx = min(int(t * (T - 1)), T - 1)

        start = frame_idx * pixels_per_frame
        end = start + pixels_per_frame
        gt_frame = image_tensor[start:end].view(H, W, -1)

        recon_img = recon.permute(2, 0, 1).unsqueeze(0).detach()
        gt_img = gt_frame.permute(2, 0, 1).unsqueeze(0).detach()

        if update:
            recon_img.requires_grad_(True)
            gt_img.requires_grad_(True)
            loss = ssim_loss_fn(recon_img, gt_img)
            print(f"SSIM Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ssim_scores.append(loss.item())
        else:
            score = 1 - ssim_loss_fn(recon_img, gt_img).item()
            ssim_scores.append(score)

    torch.set_grad_enabled(True)  # reset default
    return ssim_scores


def compute_gradient_scores(
    reconstructed_batch,
    sampled_controls,
    image_tensor,
    shape,
    control_tensor,
    update=False,
    grad_loss_fn=None,
    optimizer=None
):
    """
    Same API as compute_ssim_scores but runs your gradient loss.
    Returns a list of per-frame grad losses; if update=True, does one
    backward()/step() on the mean grad loss.
    """
    H, W = shape
    pixels = H * W
    T = control_tensor.shape[0] // pixels

    if update:
        assert grad_loss_fn is not None, "Need grad_loss_fn when update=True"
        assert optimizer     is not None, "Need optimizer when update=True"
        torch.set_grad_enabled(True)
    else:
        torch.set_grad_enabled(False)

    total_grad = 0.0
    grad_scores = []

    for recon, control in zip(reconstructed_batch, sampled_controls):
        # pick matching GT frame
        t = control[0].item()
        idx = min(int(t * (T - 1)), T - 1)
        start = idx * pixels
        gt = image_tensor[start:start+pixels].view(H, W, -1)

        # NCHW tensors
        p = recon.permute(2, 0, 1).unsqueeze(0)
        g = gt.permute(2, 0, 1).unsqueeze(0).detach()

        loss_g = grad_loss_fn(p, g)
        grad_scores.append(loss_g.item())
        if update:
            total_grad += loss_g

    if update:
        mean_grad = total_grad / len(reconstructed_batch)
        optimizer.zero_grad()
        mean_grad.backward()
        optimizer.step()

    torch.set_grad_enabled(True)
    return grad_scores


def subsample_random_pixels(num_samples, image_tensor, raw_pos, control_tensor):
    total_pixels = image_tensor.shape[0]
    indices = torch.randperm(total_pixels)[:num_samples]
    return (
        image_tensor[indices],
        raw_pos[indices],
        control_tensor[indices],
    )


class PatchSampler(torch.utils.data.IterableDataset):
    def __init__(self, image_tensor, tile_size=32, batch_size=8, dtype=torch.float32, device='cuda'):
        self.device = device
        self.tile_size = tile_size
        self.margin_size = self.tile_size // 2
        self.image_tensor = image_tensor.permute(0, 3, 1, 2)  # Change to NCHW format
        self.T, self.C, self.H, self.W = self.image_tensor.shape

        dx = self.margin_size / self.W
        dy = self.margin_size / self.H
        self.kernel_x = torch.linspace(-dx, dx, tile_size, dtype=image_tensor.dtype, device=device)
        self.kernel_y = torch.linspace(-dy, dy, tile_size, dtype=image_tensor.dtype, device=device)

        self.patch_kernel = torch.stack(torch.meshgrid(self.kernel_y, self.kernel_x, indexing='ij'), dim=-1)  # (tile_size, tile_size, 2)
        self.batch_size = batch_size
        self.dtype = dtype

    def __iter__(self):
        while True:
            times = torch.randint(0, self.T, (self.batch_size,), device=self.image_tensor.device)
            image_batch = self.image_tensor[times].to(device=self.device)  # Move to GPU
            patch_centers = torch.rand((self.batch_size, 2), dtype=self.image_tensor.dtype, device=self.device) * 2 - 1
            patch_grid = self.patch_kernel.unsqueeze(0) + patch_centers[:, None, None, :]  # [B, tile, tile, 2]

            patch_batch = F.grid_sample(image_batch, patch_grid).requires_grad_(True)  # [B, C, tile, tile]
            patch_grid = (patch_grid + 1) / 2

            times_patch = times.float().view(-1, 1, 1, 1).expand(-1, self.tile_size, self.tile_size, 1)
            yield (
                patch_batch.to(dtype=self.dtype, device=self.device),
                patch_grid.to(dtype=self.dtype, device=self.device),
                (times_patch / self.T).to(dtype=self.dtype, device=self.device)
            )


def train_vfx_model(image_dir, device='cuda', epochs=1000, batch_size=8192, experiment_name=None, decoder_config=None, max_image_size=512):
    """
    Train VFX model on images from the given directory.
    
    Args:
        image_dir: Path to directory containing training images
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of training epochs
        batch_size: Batch size for training
        experiment_name: Name for this experiment (for logging)
        decoder_config: Configuration for the decoder network
        max_image_size: Maximum height/width for images. Images larger than this will be
                       downsampled while preserving aspect ratio. Set to None to use original size.
    """
    image_tensor = load_images(image_dir, max_size=max_image_size)
    shape = image_tensor.shape[1:]
    print(f"Training on images with shape: {shape}")

    mse_loss = nn.MSELoss().to(device)
    dct_loss = DCTLoss().to(device)
    gradient_loss = GradientLoss().to(device)
    ssim_loss = SSIMLoss(data_range=1.0).to(device)

    model = VFXNet(shape[0], shape[1], decoder_config=decoder_config).to(device)
    model.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer = SOAP(
        model.parameters(),
        weight_decay=0,
    )

    base_path = f"anim_tests/{model.experiment_name}"
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(f'runs/{model.experiment_name}')

    # Log model architecture and config
    writer.add_text('Config/Decoder', json.dumps(decoder_config or {}, indent=2), 0)
    writer.add_text('Config/Training', f'Epochs: {epochs}, Batch Size: {batch_size}', 0)

    dataset = PatchSampler(image_tensor, batch_size=64)
    dataloader = DataLoader(dataset, batch_size=None)
    log_epochs = list(range(0, 11, 1)) + list(range(10, 51, 5)) + list(range(50, 101, 10)) + list(range(100, 501, 50)) + list(range(500, 1001, 100))

    batches_per_epoch = 1000
    total_pixel_loss = 0.0
    total_patch_loss = 0.0
    global_step = 0
    for epoch in range(epochs):
        model.train()
        batch_num = 0
        total_pixel_loss = 0.0
        total_mse_loss = 0.0
        total_dct_loss = 0.0
        total_l1_loss = 0.0
        total_ssim_loss = 0.0
        total_psnr = 0.0
        for image_patches, pos_patches, time_patches in dataloader:
            # Permute reconstructed_image from [B, H, W, C] to [B, C, H, W] and select first 3 channels
            reconstructed_image = model.run_patch(pos_patches, time_patches)[..., :3]
            reconstructed_image_patches = reconstructed_image.permute(0, 3, 1, 2)
            patch_loss = ssim_loss(reconstructed_image_patches, image_patches)

            if batch_num % 100 == 0:
                print(f"Batch {batch_num+1}/{batches_per_epoch}")

            # Calculate individual loss components
            mse_component = mse_loss(reconstructed_image_patches, image_patches)
            dct_component = dct_loss(reconstructed_image_patches, image_patches)
            l1_component = F.l1_loss(reconstructed_image_patches, image_patches)

            pixel_loss = (
                0.1 * mse_component +
                0.15 * dct_component +
                0.75 * l1_component
            )

            # Track individual losses
            total_mse_loss += mse_component.item()
            total_dct_loss += dct_component.item()
            total_l1_loss += l1_component.item()
            total_ssim_loss += patch_loss.item()
            
            # Calculate PSNR
            psnr_value = psnr(reconstructed_image_patches, image_patches, data_range=1.0)
            total_psnr += psnr_value.item()

            total_loss = pixel_loss + patch_loss
            optimizer.zero_grad()
            total_loss.backward()

            # Log gradient norm before step
            grad_norm = get_total_grad_norm(model)

            optimizer.step()
            total_pixel_loss += pixel_loss.item()
            total_patch_loss += patch_loss.item()

            # Log batch metrics to TensorBoard
            if batch_num % 50 == 0:
                writer.add_scalar('Loss/Batch/Total', total_loss.item(), global_step)
                writer.add_scalar('Loss/Batch/Pixel', pixel_loss.item(), global_step)
                writer.add_scalar('Loss/Batch/SSIM', patch_loss.item(), global_step)
                writer.add_scalar('Loss/Batch/MSE', mse_component.item(), global_step)
                writer.add_scalar('Loss/Batch/DCT', dct_component.item(), global_step)
                writer.add_scalar('Loss/Batch/L1', l1_component.item(), global_step)
                writer.add_scalar('Metrics/Batch/PSNR', psnr_value.item(), global_step)
                writer.add_scalar('Training/GradientNorm', grad_norm, global_step)
                writer.add_scalar('Training/LearningRate', optimizer.param_groups[0]['lr'], global_step)

            global_step += 1
            batch_num += 1
            if batch_num >= batches_per_epoch:
                break
        total_pixel_loss /= batches_per_epoch
        total_patch_loss /= batches_per_epoch
        total_mse_loss /= batches_per_epoch
        total_dct_loss /= batches_per_epoch
        total_l1_loss /= batches_per_epoch
        total_ssim_loss /= batches_per_epoch
        avg_psnr = total_psnr / batches_per_epoch

        # Log epoch metrics
        writer.add_scalar('Loss/Epoch/Total', total_pixel_loss + total_patch_loss, epoch)
        writer.add_scalar('Loss/Epoch/Pixel', total_pixel_loss, epoch)
        writer.add_scalar('Loss/Epoch/SSIM', total_ssim_loss, epoch)
        writer.add_scalar('Loss/Epoch/MSE', total_mse_loss, epoch)
        writer.add_scalar('Loss/Epoch/DCT', total_dct_loss, epoch)
        writer.add_scalar('Loss/Epoch/L1', total_l1_loss, epoch)
        writer.add_scalar('Metrics/Epoch/PSNR', avg_psnr, epoch)

        if epoch in log_epochs:
            print(f"Epoch {epoch+1} - Pixel loss: {total_pixel_loss:.4f}, Patch loss: {total_patch_loss:.4f}, PSNR: {avg_psnr:.2f} dB")
            epoch_dir = f"{base_path}/epoch_{epoch}"
            os.makedirs(epoch_dir, exist_ok=True)
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                save_images(model, H=512, W=1024, n_images=5, gif_frames=250, base_dir=epoch_dir)

                # Generate and log image comparisons to TensorBoard
                # Sample a few time points for visualization
                sample_times = torch.linspace(0, 1, 4, device=device)
                psnr_values = []
                ssim_values = []
                
                # Use the actual image dimensions (already scaled if needed during loading)
                H_img, W_img = shape[0], shape[1]
                
                for i, t in enumerate(sample_times):
                    # Generate image at the training resolution
                    generated = model.full_image(t.unsqueeze(0), H=H_img, W=W_img)

                    # Get corresponding ground truth frame
                    frame_idx = min(int(t.item() * (image_tensor.shape[0] - 1)), image_tensor.shape[0] - 1)
                    gt_frame = image_tensor[frame_idx][:, :, :3]  # Full frame, just select RGB channels

                    # Convert to NCHW format for metrics calculation
                    gen_img = generated[:, :, :3].permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
                    gt_img = gt_frame.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
                    
                    # Calculate metrics for this frame
                    frame_psnr = psnr(gen_img, gt_img, data_range=1.0).item()
                    frame_ssim = ssim(gen_img, gt_img, data_range=1.0).item()
                    psnr_values.append(frame_psnr)
                    ssim_values.append(frame_ssim)
                    
                    # Log individual frame metrics
                    writer.add_scalar(f'Metrics/Frame/PSNR_t{i}', frame_psnr, epoch)
                    writer.add_scalar(f'Metrics/Frame/SSIM_t{i}', frame_ssim, epoch)

                    # Create side-by-side comparison (images are already at appropriate size)
                    comparison = torch.cat([gt_img, gen_img], dim=3)  # Concatenate along width
                    writer.add_image(f'Images/Comparison_t{i}', comparison[0], epoch)

                    # Also log individual images
                    writer.add_image(f'Images/Original_t{i}', gt_img[0], epoch)
                    writer.add_image(f'Images/Generated_t{i}', gen_img[0], epoch)
                
                # Log average metrics across all sampled frames
                avg_frame_psnr = sum(psnr_values) / len(psnr_values)
                avg_frame_ssim = sum(ssim_values) / len(ssim_values)
                writer.add_scalar('Metrics/Evaluation/AvgPSNR', avg_frame_psnr, epoch)
                writer.add_scalar('Metrics/Evaluation/AvgSSIM', avg_frame_ssim, epoch)
                print(f"  Evaluation - Avg PSNR: {avg_frame_psnr:.2f} dB, Avg SSIM: {avg_frame_ssim:.4f}")

    # Close TensorBoard writer
    writer.close()


def get_total_grad_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm
