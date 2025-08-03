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
from piq import ssim, SSIMLoss

from decoders import VFXSpiralNetDecoder, VFXNetPixelDecoder
from losses import DCTLoss, GradientLoss
from make_shader import decoder_to_glsl, compare_decoder_and_shader, save_weights_to_exr
from image_utils import load_images, save_images
from soap import SOAP


LATENT_IMAGE_CHANNELS = 4 # RGBA


class VFXNet(nn.Module):
    def __init__(self, height, width, decoder_config=None, device='cuda', experiment_name=None, freeze_decoder=False):
        super().__init__()
        self.experiment_name = experiment_name
        self.height = height
        self.width = width
        print(f"VFXNet: height={height}, width={width}")
        _x_coords = torch.arange(width).repeat(height, 1).view(-1, 1) / width
        _y_coords = torch.arange(height).repeat(width, 1).t().contiguous().view(-1, 1) / height
        self.raw_pos = torch.cat([_x_coords, _y_coords], dim=1)

        self.decoder = VFXSpiralNetDecoder(**decoder_config or {})
        if self.decoder.latent_dim > 0:
            self.latent = nn.Embedding(self.height * self.width, LATENT_IMAGE_CHANNELS)
        self.output_channels = self.decoder.output_channels
        self.freeze_decoder = freeze_decoder
        if self.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
        self.raw_pos = self.raw_pos.to(device)
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

    def full_image(self, control, H=None, W=None):
        # Expand control to match the first dimension of self.raw_pos
        if H is None or W is None:
            print("WARNING: H and W are None, using model's height and width.")
            H, W = self.height, self.width
            raw_pos = self.raw_pos
        else:
            # Generate normalized raw_pos coordinates for H x W grid
            x_coords = torch.arange(W, device=control.device).repeat(H, 1).view(-1, 1) / (W - 1)
            y_coords = torch.arange(H, device=control.device).repeat(W, 1).t().contiguous().view(-1, 1) / (H - 1)
            raw_pos = torch.cat([x_coords, y_coords], dim=1)
        t = control[:, 0:1]
        expanded_time = t.unsqueeze(1).expand(-1, raw_pos.size(0), -1).reshape(-1, t.size(-1))
        if self.decoder.latent_dim > 0:
            print("Again, the world has ended.")
            latent = self.latents(control[:, 1].long())
            latent = latent.view(-1, H, W, LATENT_IMAGE_CHANNELS)
            response = self.decoder(raw_pos, expanded_time, latent)
        else:
            response = self.decoder(raw_pos, expanded_time)
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
    def __init__(self, image_tensor, tile_size=32):
        self.image_tensor = image_tensor
        self.tile_size = tile_size
        self.margin_size = self.tile_size // 2
        self.T, self.H, self.W, self.C = self.image_tensor.shape
        xs = torch.arange(self.W) / (self.W - 1)
        ys = torch.arange(self.H) / (self.H - 1)
        self.ts = torch.arange(self.T) / (self.T - 1)
        self.raw_pos = torch.stack(torch.meshgrid(ys, xs, indexing='ij'), dim=-1)

    def __iter__(self):
        t = torch.randint(0, self.T, (1,)).item()
        x = torch.randint(self.margin_size, self.W - self.margin_size, (1,)).item()
        y = torch.randint(self.margin_size, self.H - self.margin_size, (1,)).item()
        image_patch = self.image_tensor[t, (y - self.margin_size):(y + self.margin_size), (x - self.margin_size):(x + self.margin_size), :]
        raw_pos_patch = self.raw_pos[(y - self.margin_size):(y + self.margin_size), (x - self.margin_size):(x + self.margin_size), :]
        t_patch = self.ts[t].repeat(self.tile_size, self.tile_size, 1)
        yield (image_patch, raw_pos_patch, t_patch)


def train_vfx_model(image_dir, device='cuda', epochs=1000, batch_size=8192, experiment_name=None, decoder_config=None):
    image_tensor = load_images(image_dir, device=device)
    shape = image_tensor.shape[1:]
    dataset = PatchSampler(image_tensor)

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

    log_epochs = list(range(0, 11, 1)) + list(range(10, 51, 5)) + list(range(50, 101, 10)) + list(range(100, 501, 50)) + list(range(500, 1001, 100))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
    )

    total_pixel_loss = 0.0
    for epoch in range(epochs):
        model.train()

        print(f"Epoch {epoch+1}/{epochs} - Pixel loss: {total_pixel_loss:.4f}")
        total_pixel_loss = 0.0
        for batch_num, (image, raw_pos_batch, control_batch) in enumerate(dataloader):
            print(f"Batch {batch_num+1} - Image shape: {image.shape}, Raw pos shape: {raw_pos_batch.shape}, Control shape: {control_batch.shape}")
            reconstructed_image = model(raw_pos_batch, control_batch)
            if batch_num % 100 == 0:
                print(f"Batch {batch_num+1}/{num_batches}")
            pixel_loss = (
                0.1 * mse_loss(reconstructed_image, image) +
                0.15 * dct_loss(reconstructed_image, image) +
                0.75 * F.l1_loss(reconstructed_image, image)
            )
            optimizer.zero_grad()
            pixel_loss.backward()
            optimizer.step()
            total_pixel_loss += pixel_loss.item()
        
        if epoch in log_epochs:
            epoch_dir = f"{base_path}/epoch_{epoch}"
            os.makedirs(epoch_dir, exist_ok=True)
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                reconstructed_batch, sampled_controls = save_images(model, control_tensor, H=512, W=1024, base_dir=epoch_dir)

                print(f"batch device: {reconstructed_batch.device}, control_tensor device: {control_tensor.device}")

                ssim_scores = compute_ssim_scores(reconstructed_batch, sampled_controls, image_tensor, (model.height, model.width), control_tensor, ssim_loss_fn=ssim_loss, update=False)
                avg_ssim = sum(ssim_scores) / len(ssim_scores)

                # print(f"Average SSIM for epoch {epoch}: {avg_ssim:.4f}")
                metrics_path = f"png_tests/{model.experiment_name}/epoch_{epoch}/summary.json"
                os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
                with open(metrics_path, "w") as f:
                    json.dump({
                        "epoch": epoch,
                        # "loss": pixel_loss.item(),
                        "avg_ssim": avg_ssim,
                        "ssim_per_frame": ssim_scores,
                        "decoder_config": decoder_config
                    }, f, indent=2)


def get_total_grad_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

