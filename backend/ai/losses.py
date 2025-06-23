import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torchvision import models
import torch_dct as dct


class WeightedMSELoss(nn.Module):
    def __init__(self, weight_exponent=2.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')  # Compute element-wise MSE
        self.weight_exponent = weight_exponent

    def forward(self, reconstructed, target):
        # Compute the mean RGB value for the target
        mean_rgb = target.mean(dim=0, keepdim=True)

        # Compute weights based on distance from the mean
        weights = torch.norm(target - mean_rgb, dim=1, keepdim=True)  # L2 norm (distance from mean)

        # Normalize weights to [0, 1]
        weights = (weights / weights.max()) ** self.weight_exponent

        # Compute weighted MSE loss
        loss = self.mse(reconstructed, target)
        weighted_loss = (weights * loss).mean()  # Apply weights and average
        return weighted_loss


class GradientLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=device
                            ).view(1,1,3,3)
        self.sobel_y = self.sobel_x.transpose(2,3)

    def forward(self, pred, target):
        # using Rec.709: Y = 0.2126 R + 0.7152 G + 0.0722 B
        coeffs = torch.tensor([0.2126,0.7152,0.0722],device=pred.device).view(1,3,1,1)
        p_y = (pred * coeffs).sum(dim=1, keepdim=True)
        t_y = (target * coeffs).sum(dim=1, keepdim=True)

        gx_p = F.conv2d(p_y, self.sobel_x, padding=1)
        gy_p = F.conv2d(p_y, self.sobel_y, padding=1)
        gx_t = F.conv2d(t_y, self.sobel_x, padding=1)
        gy_t = F.conv2d(t_y, self.sobel_y, padding=1)

        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


def gradient_loss(pred, target):
    # convert to grayscale luminance
    # using Rec.709: Y = 0.2126 R + 0.7152 G + 0.0722 B
    coeffs = torch.tensor([0.2126,0.7152,0.0722],device=pred.device).view(1,3,1,1)
    p_y = (pred * coeffs).sum(dim=1, keepdim=True)
    t_y = (target * coeffs).sum(dim=1, keepdim=True)

    gx_p = F.conv2d(p_y, sobel_x, padding=1)
    gy_p = F.conv2d(p_y, sobel_y, padding=1)
    gx_t = F.conv2d(t_y, sobel_x, padding=1)
    gy_t = F.conv2d(t_y, sobel_y, padding=1)

    return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
class GammaCorrectedMSELoss(nn.Module):
    def __init__(self, gamma=2.2):
        super().__init__()
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(self, reconstructed, target):
        # Apply gamma correction
        reconstructed_gamma = torch.pow(reconstructed, self.gamma)
        target_gamma = torch.pow(target, self.gamma)
        return self.mse(reconstructed_gamma, target_gamma)


class FrequencyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, reconstructed, target):
        # Compute the Fourier transform of the reconstructed and target images
        reconstructed_fft = torch.fft.fft2(reconstructed, norm="ortho")
        target_fft = torch.fft.fft2(target, norm="ortho")

        # Compute the magnitude of the frequency components
        reconstructed_mag = torch.abs(reconstructed_fft)
        target_mag = torch.abs(target_fft)

        # Compute the loss as the mean squared error of the magnitudes
        freq_loss = torch.mean((reconstructed_mag - target_mag) ** 2)
        return self.weight * freq_loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def _normalize(self, x):
        x = x[:, :3, :, :]  # Assume input is RGBA and clip to RGB
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, pred, target):
        pred = self._normalize(pred)
        target = self._normalize(target)
        return nn.functional.mse_loss(self.vgg(pred), self.vgg(target))


class DCTLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # pred/target: [B, C] with C=4 for RGBA
        dct_pred = dct.dct(pred, norm='ortho')
        dct_target = dct.dct(target, norm='ortho')
        loss = F.mse_loss(dct_pred, dct_target, reduction=self.reduction)
        return loss


class SpectralLoss(nn.Module):
    def __init__(self, freq_exponent=2.0):
        """
        Initializes spectral loss explicitly.

        Args:
            freq_exponent (float): 
                Controls explicit emphasis on high frequencies.
                Higher = more aggressive penalty explicitly on blur.
        """
        super().__init__()
        self.freq_exponent = freq_exponent

    def forward(self, predicted, target):
        # Compute FFT clearly and explicitly:
        pred_fft = torch.fft.fft2(predicted, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')

        # Shift explicitly to put high frequencies at edges clearly:
        pred_fft_shift = torch.fft.fftshift(pred_fft)
        target_fft_shift = torch.fft.fftshift(target_fft)

        # Frequency difference explicitly:
        freq_diff = pred_fft_shift - target_fft_shift

        # Explicitly build radial frequency weighting clearly:
        _, _, H, W = freq_diff.shape
        y = torch.linspace(-1, 1, H, device=predicted.device).view(-1, 1).repeat(1, W)
        x = torch.linspace(-1, 1, W, device=predicted.device).repeat(H, 1)
        radius = torch.sqrt(x**2 + y**2)
        radius = radius / radius.max()

        # Clearly explicit frequency weighting (high-radius emphasis explicitly):
        freq_weight = radius ** self.freq_exponent

        # Match explicit dimensions clearly:
        freq_weight = freq_weight.unsqueeze(0).unsqueeze(0)

        # Weighted frequency loss explicitly computed clearly:
        weighted_freq_diff = freq_weight * freq_diff.abs()

        # Explicit scalar spectral loss clearly computed:
        loss = weighted_freq_diff.mean()

        return loss


def calculate_local_loss(model, dct_loss, mse_loss, image, image_next, pos, time, dt):
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


def calculate_global_loss(model, color, pos, time, num_samples=2048, time_weight=10.0, pos_sigma=0.1, time_sigma=0.25):
    """
    Calculate the global loss for the model.
    """
    B = color.shape[0]
    device = color.device

    latent_fiber = model.fiber_encoder(torch.cat([color, pos, time], dim=-1))

    idx1 = torch.randint(0, B, (num_samples,), device=device)
    idx2 = torch.randint(0, B, (num_samples,), device=device)

    
    latent1, pos1, time1, color1 = latent_fiber[idx1], pos[idx1], time[idx1], color[idx1]
    latent2, pos2, time2, color2 = latent_fiber[idx2], pos[idx2], time[idx2], color[idx2]

    # Compute coordinate/time differences
    dpos_forward = pos2 - pos1
    dtime_forward = time2 - time1

    dpos_backward = pos1 - pos2
    dtime_backward = time1 - time2

    spatial_dist_sq = torch.sum(dpos_forward**2, dim=-1)
    temporal_dist_sq = (dtime_forward.squeeze(-1))**2

    weight_forward = torch.exp(-spatial_dist_sq / (2 * pos_sigma**2)
                               - temporal_dist_sq / (2 * time_sigma**2))

    weight_backward = weight_forward  # symmetric

    # Normalize weights
    weight_forward_norm = weight_forward / (weight_forward.max().detach() + 1e-6)
    weight_backward_norm = weight_forward_norm  # symmetric

    # Forward: 1 → 2
    transported_latent_forward = model.transport_bundles(
        latent1,
        dx=dpos_forward[:, 0:1],
        dy=dpos_forward[:, 1:2],
        dt=dtime_forward,
        x=pos1[:, 0:1],
        y=pos1[:, 1:2],
        t=time1
    )
    pred_color_forward = model.decode(transported_latent_forward, pos2, time2)

    loss_forward = F.mse_loss(pred_color_forward, color2, reduction='none').mean(dim=-1)
    loss_forward_weighted = (loss_forward * weight_forward_norm).mean()

    # Backward: 2 → 1
    transported_latent_backward = model.transport_bundles(
        latent2,
        dx=dpos_backward[:, 0:1],
        dy=dpos_backward[:, 1:2],
        dt=dtime_backward,
        x=pos2[:, 0:1],
        y=pos2[:, 1:2],
        t=time2
    )
    pred_color_backward = model.decode(transported_latent_backward, pos1, time1)

    loss_backward = F.mse_loss(pred_color_backward, color1, reduction='none').mean(dim=-1)
    loss_backward_weighted = (loss_backward * weight_backward_norm).mean()

    # Total bidirectional loss
    total_loss = (loss_forward_weighted + loss_backward_weighted) / 2.0

    return total_loss


def spatial_transport_loss(
    model,
    latent,         # (B, L) flattened latent for current frame
    flat_img,       # (B, C) flattened RGB(A) pixels of current frame
    H, W,           # height, width of frame
    sigma=0.025,      # Gaussian σ in 0/1 norm
    K=512,          # how many pixel‐pairs to sample
    device=None
):
    """
    Samples K random “centers” i, then for each draws a neighbor j
    via a rounded Gaussian offset N(0, sigma^2). Computes
    MSE between model’s transport+decode(i→j) vs flat_img[j].
    """
    if device is None:
        device = latent.device

    B, L = latent.shape
    # 1) pick K random centers
    idx_i = torch.randint(0, B, (K,), device=device)

    # 2) their 2D coords
    rows = (idx_i // W).float()   # float for gaussian
    cols = (idx_i %  W).float()

    # 3) gaussian offsets
    dr = torch.randn(K, device=device) * sigma
    dc = torch.randn(K, device=device) * sigma

    # 4) neighbor coords, clamped to image bounds
    rows2 = torch.clamp((rows + dr).round().long(), 0, H-1)
    cols2 = torch.clamp((cols + dc).round().long(), 0, W-1)
    idx_j = rows2 * W + cols2         # flat index of neighbor

    # 5) gather latents & colors
    latent_i = latent[idx_i]          # (K, L)
    color_j  = flat_img[idx_j]        # (K, C)

    # 6) build normalized pos tensors for model transport/decode
    x_i = cols  .unsqueeze(-1) / (W-1)
    y_i = rows  .unsqueeze(-1) / (H-1)
    x_j = cols2.unsqueeze(-1) / (W-1)
    y_j = rows2.unsqueeze(-1) / (H-1)
    dt = torch.zeros_like(x_i)        # no time shift

    # 7) transport latent from i→j
    latent_j = model.transport_bundles(
        latent_i,
        dx = (x_j - x_i),
        dy = (y_j - y_i),
        dt = dt,
        x  = x_i, y = y_i,
        t  = torch.zeros_like(x_i)      # zero base time
    )

    # 8) decode and compute pixel MSE
    pred_j = model.decode(latent_j, torch.cat([x_j, y_j], dim=-1), torch.zeros_like(x_i))
    return F.mse_loss(pred_j, color_j)


def pos_epoch(model,
              frame_reference,
              shape,       # (H, W, C)
              dx, dy, dt,  # spacings in x,y and time (we only use dx,dy here)
              epoch,
              experiment_name,
              mse_loss,
              dct_loss,
              optimizer):
    """
    One epoch of purely spatial training:
      - Enforce that  ψ(x+dx,y) ≈ transport_bundles(ψ(x,y), dx,0,0) decoded at (x+dx,y)
      - And      ψ(x,y+dy) ≈ transport_bundles(ψ(x,y), 0,dy,0) decoded at (x,y+dy)
    """
    model.train()
    H, W, C = shape
    total_loss = 0.0

    for frame_data in frame_reference:
        # unpack one time‐slice
        img, pos, time = frame_data[:]    # each is (H*W, ...)
        # encode to latent
        _, latent = model(img, pos, time) # (H*W, D)
        D = latent.shape[-1]

        # reshape onto grid
        latent_grid = latent.view(H, W, D)
        img_grid    = img.view(H, W, C)
        pos_grid    = pos.view(H, W, 2)
        time_grid   = time.view(H, W, 1)
        device      = latent.device

        # 1) shift‐right consistency
        # ───────────────────────
        src_latent = latent_grid[:, :-1].reshape(-1, D)   # drop last column
        src_pos    = pos_grid   [:, :-1].reshape(-1, 2)
        src_time   = time_grid  [:, :-1].reshape(-1, 1)

        # transport dx to the right
        dx_t = torch.full((src_latent.size(0),1), dx, device=device)
        dy_t = torch.zeros_like(dx_t)
        dt_t = torch.zeros_like(dx_t)

        transported = model.transport_bundles(
            src_latent, dx_t, dy_t, dt_t,
            src_pos[:,0:1], src_pos[:,1:2], src_time
        )  # (N_right, D)

        # decode at the _destination_ coords
        dest_pos  = src_pos.clone()
        dest_pos[:,0] += dx
        dest_time = src_time  # unchanged
        pred_right = model.decode(transported, dest_pos, dest_time)  # (N_right, C)

        # ground‐truth right neighbor colors
        gt_right = img_grid[:,1:].reshape(-1, C)

        loss_right = mse_loss(pred_right, gt_right)

        # 2) shift‐down consistency
        # ───────────────────────
        src_latent = latent_grid[:-1, :].reshape(-1, D)   # drop last row
        src_pos    = pos_grid   [:-1, :].reshape(-1, 2)
        src_time   = time_grid  [:-1, :].reshape(-1, 1)

        dx_t = torch.zeros_like(src_pos[:,0:1])
        dy_t = torch.full_like(dx_t, dy)
        dt_t = torch.zeros_like(dx_t)

        transported = model.transport_bundles(
            src_latent, dx_t, dy_t, dt_t,
            src_pos[:,0:1], src_pos[:,1:2], src_time
        )
        dest_pos   = src_pos.clone()
        dest_pos[:,1] += dy
        dest_time  = src_time
        pred_down  = model.decode(transported, dest_pos, dest_time)

        gt_down = img_grid[1:,:].reshape(-1, C)
        loss_down = mse_loss(pred_down, gt_down)

        # average spatial loss for this frame
        loss = 0.5*(loss_right + loss_down)

        # step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(frame_reference)
    print(f"[pos_epoch] epoch {epoch:3d} spatial loss: {avg_loss:.6e}")
    return avg_loss
