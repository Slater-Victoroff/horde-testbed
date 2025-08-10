import os
import json
from datetime import datetime
import time

import torch
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
            out = self.decoder(raw_pos, control[:, 0:1])
            return out
    
    def full_image(self, control):
        # Expand control to match the first dimension of self.raw_pos
        t = control[:, 0:1]
        expanded_time = t.unsqueeze(1).expand(-1, self.raw_pos.size(0), -1).reshape(-1, t.size(-1))
        if self.decoder.latent_dim > 0:
            print("Again, the world has ended.")
            latent = self.latents(control[:, 1].long())
            latent = latent.view(-1, self.height, self.width, LATENT_IMAGE_CHANNELS)
            response = self.decoder(self.raw_pos, expanded_time, latent)
        else:
            response = self.decoder(self.raw_pos, expanded_time)
        shaped_image = response.view(self.height, self.width, self.output_channels)
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


def train_vfx_model(image_dir, device='cuda', epochs=1000, batch_size=8196, experiment_name=None, decoder_config=None):
    # Load images and create tensors
    image_tensor, raw_pos, control_tensor, shape = load_images(image_dir, device)
    print(f"image_tensor shape: {image_tensor.shape}")
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

    # Create a dataset and dataloader
    dataset = TensorDataset(image_tensor, raw_pos, control_tensor)

    num_sequences = int(control_tensor[:, 1].max().item()) + 1
    print(f"num_sequences: {num_sequences}")
    model = VFXNet(shape[0], shape[1], decoder_config=decoder_config).to(device)
    model.experiment_name = experiment_name
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = SOAP(
        model.parameters(),
        weight_decay=0,
    )
    
    # Log model architecture and hyperparameters
    writer.add_text('Model/Architecture', str(model), 0)
    writer.add_text('Model/Config', json.dumps(decoder_config or {}, indent=2), 0)
    writer.add_scalar('Hyperparameters/batch_size', batch_size, 0)
    writer.add_scalar('Hyperparameters/epochs', epochs, 0)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Train_dataloader_stats: {len(train_dataloader)} batches, {len(train_dataloader.dataset)} samples")

    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    base_path = f"png_tests/{model.experiment_name}"
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    log_epochs = list(range(0, 11, 1)) + list(range(10, 51, 5)) + list(range(50, 101, 10)) + list(range(100, 501, 50)) + list(range(500, 1001, 100))

    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}")
        model.train()
        epoch_losses = []
        batch_num = 0
        epoch_start_time = time.time()
        for image, raw_pos, control in train_dataloader:
            batch_num += 1
            global_step += 1
            
            if batch_num % 100 == 0:
                print(f"Batch {batch_num}/{len(train_dataloader)}")
                examples_per_sec = batch_num * batch_size / (time.time() - epoch_start_time)
                print(f"Examples per second: {examples_per_sec:.2f}")
                writer.add_scalar('Training/examples_per_second', examples_per_sec, global_step)
            
            # Move batch data to device
            image = image.to(device)
            raw_pos = raw_pos.to(device) 
            control = control.to(device)
            
            reconstructed_image = model.forward(raw_pos, control)

            mse_weight = 0.1
            dct_weight = 0.15
            l1_weight = 0.75

            # Calculate individual losses
            mse_loss_val = mse_loss(reconstructed_image, image)
            dct_loss_val = dct_loss(reconstructed_image, image)
            l1_loss_val = F.l1_loss(reconstructed_image, image)
            
            pixel_loss = (
                mse_weight * mse_loss_val +
                dct_weight * dct_loss_val +
                l1_weight * l1_loss_val
            )
            
            # Log losses to TensorBoard
            writer.add_scalar('Loss/total', pixel_loss.item(), global_step)
            writer.add_scalar('Loss/mse', mse_loss_val.item(), global_step)
            writer.add_scalar('Loss/dct', dct_loss_val.item(), global_step)
            writer.add_scalar('Loss/l1', l1_loss_val.item(), global_step)
            
            epoch_losses.append(pixel_loss.item())

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

        # H, W = model.height, model.width
        # original_frames = corresponding_original_frames(sampled, image_tensor, (H, W), control_tensor)
        # compute_gradient_scores(
        #     reconstructed, original_frames,
        #     update=True,
        #     grad_loss_fn=gradient_loss,  # your Sobel‐based function
        #     optimizer=optimizer
        # )
        # scheduler.step()

        # print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.6f}")
        # Log specific epochs for detailed analysis
        if epoch in log_epochs:
            epoch_dir = f"{base_path}/epoch_{epoch}"
            os.makedirs(epoch_dir, exist_ok=True)
            model.eval()  # Set model to evaluation mode
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

