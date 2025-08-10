import os
import time
import json
from datetime import datetime
import time

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
from image_utils import load_images, save_images, generate_comparison_gif
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
            out = self.decoder(raw_pos, control[:, 0:1])
            return out
    
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


def corresponding_original_frames(sampled_controls, image_tensor, shape, control_tensor):
    """
    Extract original frames that correspond to the sampled control values.
    
    Args:
        sampled_controls: List of control tensors with time values
        image_tensor: Full tensor of all image data
        shape: Tuple of (H, W)
        control_tensor: Full control tensor
    
    Returns:
        List of original frames matching the sampled controls
    """
    H, W = shape
    pixels_per_frame = H * W
    T = control_tensor.shape[0] // pixels_per_frame
    original_frames = []
    
    for control in sampled_controls:
        t = control[0].item()
        frame_idx = min(int(t * (T - 1)), T - 1)
        start = frame_idx * pixels_per_frame
        end = start + pixels_per_frame
        gt_frame = image_tensor[start:end].view(H, W, -1)
        original_frames.append(gt_frame)
    
    return original_frames


def compute_psnr_scores(reconstructed_frames, original_frames):
    """
    Compute PSNR scores between reconstructed frames and original frames.
    
    Args:
        reconstructed_frames: List of reconstructed frames (H, W, C) tensors
        original_frames: List of original frames (H, W, C) tensors
    
    Returns:
        List of PSNR values in dB
    """
    psnr_scores = []
    
    with torch.no_grad():
        for recon, orig in zip(reconstructed_frames, original_frames):
            # Convert to NCHW format for piq.psnr
            recon_img = recon.permute(2, 0, 1).unsqueeze(0)
            orig_img = orig.permute(2, 0, 1).unsqueeze(0)
            
            # Compute PSNR
            psnr_value = psnr(recon_img, orig_img, data_range=1.0)
            psnr_scores.append(psnr_value.item())
    
    return psnr_scores


def log_psnr_info(psnr_scores, sampled_controls, epoch):
    """
    Log PSNR statistics including min, max, and average values with corresponding frame info.
    
    Args:
        psnr_scores: List of PSNR values
        sampled_controls: List of control tensors corresponding to each PSNR score
        epoch: Current epoch number
        
    Returns:
        avg_psnr: Average PSNR value
    """
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    min_psnr = min(psnr_scores)
    max_psnr = max(psnr_scores)
    min_idx = psnr_scores.index(min_psnr)
    max_idx = psnr_scores.index(max_psnr)
    
    # Get corresponding control values for min/max frames
    min_control = sampled_controls[min_idx]
    max_control = sampled_controls[max_idx]
    min_t = min_control[0].item()
    max_t = max_control[0].item()
    
    print(f"Average PSNR for epoch {epoch}: {avg_psnr:.2f} dB")
    print(f"PSNR stats: min={min_psnr:.2f}dB (t={min_t:.3f}, frame_idx={min_idx}), max={max_psnr:.2f}dB (t={max_t:.3f}, frame_idx={max_idx})")
    
    return avg_psnr


def compute_ssim_scores(
    reconstructed_batch,
    original_frames,
    update=False,
    ssim_loss_fn=None,
    optimizer=None
):
    ssim_scores = []

    if not update:
        torch.set_grad_enabled(False)  # Ensure gradients off for pure evaluation
    else:
        assert ssim_loss_fn is not None, "Must provide SSIM loss function when update=True"
        torch.set_grad_enabled(True)

    for recon, orig in zip(reconstructed_batch, original_frames):
        recon_img = recon.permute(2, 0, 1).unsqueeze(0).detach()
        gt_img = orig.permute(2, 0, 1).unsqueeze(0).detach()

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
    original_frames,
    update=False,
    grad_loss_fn=None,
    optimizer=None
):
    """
    Same API as compute_ssim_scores but runs your gradient loss.
    Returns a list of per-frame grad losses; if update=True, does one
    backward()/step() on the mean grad loss.
    """

    if update:
        assert grad_loss_fn is not None, "Need grad_loss_fn when update=True"
        assert optimizer     is not None, "Need optimizer when update=True"
        torch.set_grad_enabled(True)
    else:
        torch.set_grad_enabled(False)

    total_grad = 0.0
    grad_scores = []

    for recon, gt in zip(reconstructed_batch, original_frames):

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


def train_vfx_model(image_dir, device='cuda', epochs=1000, batch_size=8192, experiment_name=None, decoder_config=None):
    image_tensor = load_images(image_dir)
    shape = image_tensor.shape[1:]
    mse_loss = nn.MSELoss().to(device)
    dct_loss = DCTLoss().to(device)
    gradient_loss = GradientLoss().to(device)
    ssim_loss = SSIMLoss(data_range=1.0).to(device)
    
    # Initialize TensorBoard writer
    experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'runs/{experiment_name}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    global_step = 0

    model = VFXNet(shape[0], shape[1], decoder_config=decoder_config).to(device)
    model.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    optimizer = SOAP(
        model.parameters(),
        weight_decay=0,
    )
    
    # Log model architecture and hyperparameters
    writer.add_text('Model/Architecture', str(model), 0)
    writer.add_text('Model/Config', json.dumps(decoder_config or {}, indent=2), 0)
    writer.add_scalar('Hyperparameters/batch_size', batch_size, 0)
    writer.add_scalar('Hyperparameters/epochs', epochs, 0)
    print(f"Train_dataloader_stats: {len(train_dataloader)} batches, {len(train_dataloader.dataset)} samples")

    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    base_path = f"anim_tests/{model.experiment_name}"
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    dataset = PatchSampler(image_tensor, batch_size=64)
    dataloader = DataLoader(dataset, batch_size=None)
    log_epochs = list(range(0, 11, 1)) + list(range(10, 51, 5)) + list(range(50, 101, 10)) + list(range(100, 501, 50)) + list(range(500, 1001, 100))

    batches_per_epoch = 1000
    total_pixel_loss = 0.0
    total_patch_loss = 0.0
    for epoch in range(epochs):
        model.train()
        batch_num = 0
        total_pixel_loss = 0.0
        epoch_losses = []
        epoch_start_time = time.time()
        for image_patches, pos_patches, time_patches in dataloader:
            batch_num += 1
            global_step += 1
            
            if batch_num % 100 == 0:
                print(f"Batch {batch_num}/{len(dataloader)}")
                examples_per_sec = batch_num * batch_size / (time.time() - epoch_start_time)
                print(f"Examples per second: {examples_per_sec:.2f}")
                writer.add_scalar('Training/examples_per_second', examples_per_sec, global_step)
            # Permute reconstructed_image from [B, H, W, C] to [B, C, H, W] and select first 3 channels
            reconstructed_image = model.run_patch(pos_patches, time_patches)[..., :3]
            reconstructed_image_patches = reconstructed_image.permute(0, 3, 1, 2)
            patch_loss = ssim_loss(reconstructed_image_patches, image_patches)

            if batch_num % 100 == 0:
                print(f"Batch {batch_num+1}/{batches_per_epoch}")
                
            pixel_loss = (
                0.1 * mse_loss(reconstructed_image_patches, image_patches) +
                0.15 * dct_loss(reconstructed_image_patches, image_patches) +
                0.75 * F.l1_loss(reconstructed_image_patches, image_patches)
            )
            
            # Log losses to TensorBoard
            writer.add_scalar('Loss/total', pixel_loss.item(), global_step)
            writer.add_scalar('Loss/mse', mse_loss_val.item(), global_step)
            writer.add_scalar('Loss/dct', dct_loss_val.item(), global_step)
            writer.add_scalar('Loss/l1', l1_loss_val.item(), global_step)
            
            epoch_losses.append(pixel_loss.item())

            total_loss = pixel_loss + patch_loss
            optimizer.zero_grad()
            pixel_loss.backward()
            
            # Log gradient norms
            if batch_num % 100 == 0:
                grad_norm = get_total_grad_norm(model)
                writer.add_scalar('Training/gradient_norm', grad_norm, global_step)
            
            optimizer.step()
            
            # Log learning rate
            writer.add_scalar('Training/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        
        # Log epoch metrics
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        epoch_duration = time.time() - epoch_start_time
        writer.add_scalar('Loss/epoch_average', avg_epoch_loss, epoch)
        writer.add_scalar('Training/epoch_duration', epoch_duration, epoch)
        
        reconstructed, sampled = save_images(model, control_tensor, write_files=False)
        if epoch > 100:
            H, W = model.height, model.width
            original_frames = corresponding_original_frames(sampled, image_tensor, (H, W), control_tensor)
            ssim_scores = compute_ssim_scores(
                reconstructed, original_frames,
                update=True, ssim_loss_fn=ssim_loss, optimizer=optimizer
            )
            # Log SSIM scores from the update
            if ssim_scores:
                avg_ssim_loss = sum(ssim_scores) / len(ssim_scores)
                writer.add_scalar('Loss/ssim', avg_ssim_loss, epoch)
            batch_num += 1
            if batch_num >= batches_per_epoch:
                break

        if epoch in log_epochs:
            epoch_dir = f"{base_path}/epoch_{epoch}"
            os.makedirs(epoch_dir, exist_ok=True)
            model.eval()
      
            print(f"Epoch {epoch+1} - Pixel loss: {total_pixel_loss:.4f}, Patch loss: {total_patch_loss:.4f}")
            reconstructed_batch, sampled_controls = save_images(model, control_tensor, base_dir=epoch_dir)
            
            # Get original frames once
            H, W = shape[0], shape[1]
            original_frames_tensors = corresponding_original_frames(sampled_controls, image_tensor, (H, W), control_tensor)
            
            # Compute SSIM scores
            ssim_scores = compute_ssim_scores(reconstructed_batch, original_frames_tensors, ssim_loss_fn=ssim_loss, update=False)
            avg_ssim = sum(ssim_scores) / len(ssim_scores)
            
            # Compute PSNR scores
            psnr_scores = compute_psnr_scores(reconstructed_batch, original_frames_tensors)
            avg_psnr = log_psnr_info(psnr_scores, sampled_controls, epoch)
            
            # Log quality metrics to TensorBoard
            writer.add_scalar('Metrics/PSNR', avg_psnr, epoch)
            writer.add_scalar('Metrics/SSIM', avg_ssim, epoch)
            
            # Log per-frame metrics as histograms
            writer.add_histogram('Metrics/PSNR_distribution', torch.tensor(psnr_scores), epoch)
            writer.add_histogram('Metrics/SSIM_distribution', torch.tensor(ssim_scores), epoch)
            
            # Log sample images to TensorBoard
            if len(reconstructed_batch) > 0 and len(original_frames_tensors) > 0:
                # Take first 4 frames for visualization
                num_samples = min(4, len(reconstructed_batch))
                for i in range(num_samples):
                    # Convert from HWC to CHW for TensorBoard
                    recon_img = reconstructed_batch[i].permute(2, 0, 1)
                    orig_img = original_frames_tensors[i].permute(2, 0, 1)
                    
                    writer.add_image(f'Reconstructed/frame_{i}', recon_img, epoch)
                    writer.add_image(f'Original/frame_{i}', orig_img, epoch)
                    
                    # Add difference image
                    diff_img = torch.abs(recon_img - orig_img)
                    writer.add_image(f'Difference/frame_{i}', diff_img, epoch)

            # Generate comparison GIF
            original_frames = [frame.cpu().numpy() for frame in original_frames_tensors]
            control_frames = [frame.squeeze(0).cpu().numpy() for frame in reconstructed_batch]
            
            # Generate the comparison WebP
            comparison_path = os.path.join(epoch_dir, "comparison.webp")
            generate_comparison_gif(original_frames, control_frames, comparison_path)

            print(f"Average SSIM for epoch {epoch}: {avg_ssim:.4f}")
            metrics_path = f"png_tests/{model.experiment_name}/epoch_{epoch}/summary.json"
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump({
                    "epoch": epoch,
                    # "loss": pixel_loss.item(),
                    "avg_ssim": avg_ssim,
                    "avg_psnr": avg_psnr,
                    "ssim_per_frame": ssim_scores,
                    "psnr_per_frame": psnr_scores,
                    "decoder_config": decoder_config
                }, f, indent=2)
                
            with torch.no_grad():
                save_images(model, H=512, W=1024, n_images=5, gif_frames=250, base_dir=epoch_dir)
    
    # Close TensorBoard writer
    writer.close()
    print(f"\nTraining complete! View results with: tensorboard --logdir=runs/{experiment_name}")
    return model



def get_total_grad_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

