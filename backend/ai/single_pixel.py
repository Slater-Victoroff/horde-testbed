import os
import time
import json
from datetime import datetime

import torch
import torch.profiler as profiler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader, TensorDataset
from piq import ssim, SSIMLoss, psnr
from torch.utils.tensorboard import SummaryWriter

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

        return self.decoder(flat_raw_pos, flat_control).view(*pos_patch.shape[:-1], self.output_channels)

    def full_image(self, time, H=512, W=512):
        x_coords = torch.linspace(0, 1, W, device=self.device)
        y_coords = torch.linspace(0, 1, H, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        raw_pos = torch.stack([grid_x, grid_y], dim=-1)
        expanded_time = time.repeat(H, W, 1)

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
            patch_grid = ((patch_grid + 1) / 2).clamp(0.0, 1.0)

            times_patch = times.float().view(-1, 1, 1, 1).expand(-1, self.tile_size, self.tile_size, 1)
            yield (
                patch_batch.to(dtype=self.dtype, device=self.device),
                patch_grid.to(dtype=self.dtype, device=self.device),
                (times_patch / self.T).to(dtype=self.dtype, device=self.device)
            )


def log_vector(writer, name: str, vec: torch.Tensor, step: int, bins: int = 64):
    v = vec.detach().float().view(-1)
    writer.add_histogram(f"{name}/hist", v, step, bins=bins)


def _first_linear(module: nn.Module):
    if module is None:
        return None
    if isinstance(module, nn.Sequential):
        for m in module:
            if isinstance(m, nn.Linear):
                return m
    if isinstance(module, nn.Linear):
        return module
    # common patterns: custom SineLayer may wrap a linear as .linear or .fc
    for attr in ("linear", "fc", "proj"):
        if hasattr(module, attr) and isinstance(getattr(module, attr), nn.Linear):
            return getattr(module, attr)
    return None

def _log_linear_stats(writer: SummaryWriter, name: str, linear: nn.Linear, step: int):
    W = linear.weight.detach().cpu().float()   # [out, in]
    col_norm = torch.linalg.vector_norm(W, dim=0)  # [in]

    for i, val in enumerate(col_norm):
        writer.add_scalar(f"{name}/col_norm/index_{i}", val.item(), step)

def log_enc_debug(writer: SummaryWriter, model, epoch: int):
    # 1) Learned frequency distributions
    decoder = model.decoder
    if decoder.learned_encodings:
        log_vector(writer, "pos_freqs", decoder.pos_freqs, epoch)
        log_vector(writer, "time_freqs", decoder.time_freqs, epoch)

    # 2) First linear(s) behind time/pos embeddings (handles SineLayer or Sequential)
    if decoder.time_embed:
        lin = _first_linear(decoder.time_embed)
        _log_linear_stats(writer, "time_embed", lin, epoch)

    if decoder.pos_embed:
        lin = _first_linear(decoder.pos_embed)
        _log_linear_stats(writer, "pos_embed", lin, epoch)

def train_vfx_model(image_dir, device='cuda', epochs=1000, batch_size=8192, experiment_name=None, decoder_config=None):
    image_tensor = load_images(image_dir)
    T, H, W, C = image_tensor.shape

    mse_loss = nn.MSELoss().to(device)
    dct_loss = DCTLoss().to(device)
    gradient_loss = GradientLoss().to(device)
    ssim_loss = SSIMLoss(data_range=1.0).to(device)

    model = VFXNet(H, W, decoder_config=decoder_config).to(device)
    model.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer = SOAP(
        model.parameters(),
        weight_decay=0,
    )

    base_path = f"anim_tests/{model.experiment_name}"
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    dataset = PatchSampler(image_tensor, batch_size=64)
    dataloader = DataLoader(dataset, batch_size=None)
    log_epochs = list(range(0, 11, 1)) + list(range(10, 51, 5)) + list(range(50, 101, 10)) + list(range(100, 501, 50)) + list(range(500, 1001, 100))

    writer = SummaryWriter(f'runs/{model.experiment_name}')
    writer.add_text('Config/Decoder', json.dumps(decoder_config or {}, indent=2), 0)
    writer.add_graph(
        model,
        (
            torch.zeros((7, 2), device=device),
            torch.zeros((7, 1), device=device),
        )
    )
    batches_per_epoch = 1000
    total_pixel_loss = 0.0
    total_patch_loss = 0.0
    global_step = 0
    for epoch in range(epochs):
        model.train()
        batch_num = 0
        total_pixel_loss = 0.0
        for image_patches, pos_patches, time_patches in dataloader:
            # Permute reconstructed_image from [B, H, W, C] to [B, C, H, W] and select first 3 channels
            reconstructed_image = model.run_patch(pos_patches, time_patches)[..., :3]
            reconstructed_image_patches = reconstructed_image.permute(0, 3, 1, 2)
            patch_loss = ssim_loss(reconstructed_image_patches, image_patches)

            if batch_num % 100 == 0:
                print(f"Batch {batch_num+1}/{batches_per_epoch}")
            
            mse_weight, mse_component = 0.1, mse_loss(reconstructed_image_patches, image_patches)
            dct_weight, dct_component = 0.15, dct_loss(reconstructed_image_patches, image_patches)
            l1_weight, l1_component = 0.75, F.l1_loss(reconstructed_image_patches, image_patches)

            pixel_loss = (
                mse_weight * mse_component +
                dct_weight * dct_component +
                l1_weight * l1_component
            )

            total_loss = pixel_loss + patch_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/MSE', mse_component.item(), global_step)
            writer.add_scalar('Loss/DCT', dct_component.item(), global_step)
            writer.add_scalar('Loss/L1', l1_component.item(), global_step)
            writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            writer.add_scalar('Loss/Pixel', pixel_loss.item(), global_step)
            writer.add_scalar('Loss/Patch', patch_loss.item(), global_step)
            total_pixel_loss += pixel_loss.item()
            total_patch_loss += patch_loss.item()
            batch_num += 1
            if batch_num >= batches_per_epoch:
                break
        total_pixel_loss /= batches_per_epoch
        total_patch_loss /= batches_per_epoch

        if epoch in log_epochs:
            print(f"Epoch {epoch+1} - Pixel loss: {total_pixel_loss:.4f}, Patch loss: {total_patch_loss:.4f}")
            epoch_dir = f"{base_path}/epoch_{epoch}"
            os.makedirs(epoch_dir, exist_ok=True)
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                debug_frames = 5
                test_frames = 10
                save_images(model, H=H, W=W, n_images=debug_frames, gif_frames=T, base_dir=epoch_dir)
                
                test_controls = torch.randint(0, T, (test_frames,), device=device)
                base_batch = image_tensor[test_controls]
                pred_batch = [model.full_image(test_control / T, H, W) for test_control in test_controls]
                pred_batch = torch.stack(pred_batch).to(dtype=torch.float32)
                print(f"Base batch shape: {base_batch.shape}, Pred batch shape: {pred_batch.shape}")

                base_batch_nchw = base_batch.permute(0, 3, 1, 2)
                pred_batch_nchw = pred_batch.permute(0, 3, 1, 2)
                comparison = torch.cat([base_batch_nchw, pred_batch_nchw], dim=3)  # Concatenate along width

                ssim_result = ssim(base_batch_nchw, pred_batch_nchw)
                psnr_result = psnr(base_batch_nchw, pred_batch_nchw)

                for i, t in enumerate(test_controls):
                    writer.add_image(f'Images/Comparison_t{t}', comparison[i], epoch)

                print(f"SSIM result: {ssim_result.item():.4f}, PSNR result: {psnr_result.item():.4f}")
                log_enc_debug(writer, model, epoch)
        global_step += 1


def get_total_grad_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

