import re
import os

import torch
import numpy as np
import OpenEXR
import imageio.v3 as iio
from PIL import Image
from torchvision.utils import save_image
from torchvision.datasets import MovingMNIST

from make_shader import save_latent_to_exr

def remove_background(frame, tolerance=0.02):
    h, w, c = frame.shape
    # Sample outer 5% margin
    edge = 0.05
    margin_px = int(min(h, w) * edge)

    top    = frame[:margin_px, :, :]
    bottom = frame[-margin_px:, :, :]
    left   = frame[:, :margin_px, :]
    right  = frame[:, -margin_px:, :]
    border_pixels = np.concatenate([
        top.reshape(-1, c),
        bottom.reshape(-1, c),
        left.reshape(-1, c),
        right.reshape(-1, c)
    ], axis=0)
    # Estimate background color (median is more robust than mean)
    bg_color = np.median(border_pixels.reshape(-1, c), axis=0)

    # Compute mask: distance from background color
    dist = np.linalg.norm(frame[..., :3] - bg_color[:3], axis=-1)  # Only consider RGB channels for distance
    mask = dist > tolerance  # foreground = True, background = False

    # Apply alpha mask
    alpha = mask.astype(np.float32)[..., np.newaxis]
    if c == 4:
        frame[..., 3:] = alpha  # Replace existing alpha channel
    else:
        frame = np.concatenate([frame, alpha], axis=-1)  # Add alpha channel if not present

    frame[..., :3] *= alpha  # Zero out background in RGB channels
    return frame

def load_exr_image(image_path):
    # Open the EXR file
    rgba = None
    with OpenEXR.File(str(image_path)) as exr_file:
        if "RGB" in exr_file.channels():
            rgb = exr_file.channels()["RGB"].pixels
            alpha = np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.float16)
            rgba = np.concatenate([rgb, alpha], axis=2)
        elif "RGBA" in exr_file.channels():
            rgba = exr_file.channels()["RGBA"].pixels
        else:
            raise ValueError("EXR file does not contain RGB or RGBA channels.")
    rgba = np.clip(rgba, 0, None)
    rgba = rgba / (1 + rgba)
    return rgba


def load_images(image_dir, device, input_image_channels=4, control_channels=2, norm=True, add_next=False):
    if "moving_mnist" in image_dir.name:
        input_image_channels = 1
    # Control tensor assumed to be [time (0/1 normalized), latent(int)]
    image_tensors, pos_tensors, control_tensors = [], [], []
    if add_next:
        image_tensors_next = []
    shape = None

    def _process_frame(frame, frame_idx=None, sequence_idx=0, prior_frame=None):
        nonlocal shape
        # Ensure correct channels
        h, w, c = frame.shape
        if c < input_image_channels:
            pad = np.ones((h, w, input_image_channels - c), dtype=frame.dtype)
            frame = np.concatenate([frame, pad], axis=-1)
        if shape is None:
            shape = frame.shape
        elif shape != frame.shape:
            raise ValueError(f"Frame shape mismatch: {shape} vs {frame.shape}")
        # flatten image
        img = torch.tensor(frame, dtype=torch.float32, device=device).view(-1, input_image_channels)
        if add_next and prior_frame is not None:
            image_tensors_next.append(img)
            prior_frame = torch.tensor(prior_frame, dtype=torch.float32, device=device).view(-1, input_image_channels)
            image_tensors.append(prior_frame)
        else:
            image_tensors.append(img)

        # control vector
        ctrl = torch.tensor([frame_idx] + [sequence_idx]*(control_channels-1), device=device)
        control_tensors.append(ctrl.repeat(img.size(0),1))
        # positional coords
        if norm:
            xs = torch.arange(w, device=device).repeat(h,1).view(-1,1) / w
            ys = torch.arange(h, device=device).repeat(w,1).t().reshape(-1,1) / h
        else:
            xs = torch.arange(w, device=device).repeat(h,1).view(-1,1)
            ys = torch.arange(h, device=device).repeat(w,1).t().reshape(-1,1)
        pos = torch.cat([xs, ys], dim=1)
        pos_tensors.append(pos)
        return img

    if "moving_mnist" in image_dir.name:
        dataset = MovingMNIST(root="ref_data", split=None, download=True)
        for sequence_idx, video in enumerate(dataset[:5]):
            # video: [T, 1, H, W], convert to numpy for consistency
            video = video.squeeze(1).numpy()  # [T, H, W]
            video = video / 255.0  # normalize

            if add_next:
                prior_frame = None
            for frame_idx, frame in enumerate(video):
                frame = np.expand_dims(frame, axis=-1)
                if add_next:
                    if prior_frame is not None:
                        _process_frame(frame, frame_idx, sequence_idx, prior_frame)
                    prior_frame = frame
                else:
                    _process_frame(frame, frame_idx, sequence_idx)
    else:
        # Check for GIF files in the directory
        gif_files = list(image_dir.glob("**/*.gif"))
        if gif_files:
            if len(gif_files) > 1:
                raise ValueError("Multiple GIF files found. Please provide only one GIF file.")
            gif_path = gif_files[0]
            gif_frames = iio.imread(gif_path, plugin="pillow")  # Load all frames from the GIF
            normalized_frames = [gif_frame / 255.0 for gif_frame in gif_frames]
            for frame_idx, frame in enumerate(normalized_frames):
                _process_frame(frame, frame_idx)
        else:
            # Process EXR files
            if add_next:
                prior_frame = None
            for image_path in sorted(image_dir.glob("**/*.exr")):  # Assuming EXR images
                image = load_exr_image(image_path)
                # Extract control values from the filename using regex
                matches = re.findall(r'\d+', image_path.stem)
                time = int(matches[1])
                if add_next:
                    if prior_frame is not None:
                        _process_frame(image, time, int(matches[0]), prior_frame)
                    prior_frame = image


    # Combine all tensors
    image_tensor = torch.cat(image_tensors, dim=0)  # [H*W*images, INPUT_IMAGE_CHANNELS]
    image_tensor = image_tensor / image_tensor.max()  # Normalize to [0, 1]
    if add_next:
        image_tensor_next = torch.cat(image_tensors_next, dim=0)
        image_tensor_next = image_tensor_next / image_tensor_next.max()
    raw_pos = torch.cat(pos_tensors, dim=0)  # [H*W*images, 2]
    control = torch.cat(control_tensors, dim=0)  # [H*W*images, CONTROL_CHANNELS]

    # control[:, 0] is the time axis, control[:, 1] is the sequence ID
    seq_ids   = control[:,1].long()            # keep these intact
    time_vals = control[:,0:1]                 # just the time axis

    # normalize *only* the continuous dims:
    if norm:
        t_min, _ = time_vals.min(0, keepdim=True)
        t_max, _ = time_vals.max(0, keepdim=True)
        t_norm   = (time_vals - t_min) / (t_max - t_min + 1e-8)

        # rebuild:
        control = torch.cat([t_norm, seq_ids.unsqueeze(1)], dim=1)

    image_tensor = image_tensor.to(device)
    if add_next:
        image_tensor_next = image_tensor_next.to(device)
    control = control.to(device)
    raw_pos = raw_pos.to(device)
    if add_next:
        return image_tensor, image_tensor_next, raw_pos, control, shape
    else:
        return image_tensor, raw_pos, control, shape


def save_images(model, control_tensor, epoch, n=5):
    # Create directory structure
    base_dir = "final_data"
    epoch_dir = os.path.join(base_dir, model.experiment_name, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    with torch.no_grad():
        # Randomly select n unique control vectors
        indices = torch.randperm(control_tensor.size(0))[:n]
        selected_controls = control_tensor[indices]

        # Log details about the selected control vectors
        print(f"Selected control vectors for epoch {epoch}:")
        for idx, control in enumerate(selected_controls):
            print(f"Control vector {idx + 1}: {control.cpu().numpy()}")

        # Generate reconstructed RGB images for each selected control vector
        for i, control in enumerate(selected_controls):
            reconstructed_rgb = model.full_image(control.unsqueeze(0))  # Pass control vector to model
            print(f"Reconstructed RGB shape: {reconstructed_rgb.shape}")
            rgb_image = reconstructed_rgb.squeeze(0).permute(2, 0, 1)  # [C, H, W]

            # Save the reconstructed RGB image with control vector details in the filename
            control_details = "_".join(f"{val:.2f}" for val in control.cpu().numpy())
            rgb_path = os.path.join(epoch_dir, f"reconstructed_rgb_{control_details}.png")
            save_image(rgb_image, rgb_path)

        # Save the shared latent image as an EXR file
        latent_path = os.path.join(epoch_dir, "test_latent.exr")
        sample_latent = model.latents(torch.tensor([0], device="cuda"))  # [1, H*W*C]
        latent_grid = sample_latent.view(model.height, model.width, 4)
        save_latent_to_exr(latent_grid, latent_path)
        print(f"Shared latent image saved to {latent_path}")

        # Save the model/weights
        model_path = os.path.join(epoch_dir, "model_weights.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")

        # Generate and save a GIF from evenly sampled control vectors
        gif_path = os.path.join(epoch_dir, "control_animation.gif")
        print(f"Control tensor shape: {control_tensor.shape}")
        reconstructed_batch, sampled_controls = generate_control_animation(model, control_tensor, gif_path)
        print(f"Control animation GIF saved to {gif_path}")
    return reconstructed_batch, sampled_controls


def generate_control_animation(model, control_tensor, gif_path, num_frames=20, control_axis=None, fixed_values=None):
    """
    Generate a GIF by sampling control vectors along specified axes and rendering frames.
    :param model: The model to generate frames.
    :param control_tensor: The tensor of control vectors.
    :param gif_path: Path to save the generated GIF.
    :param num_frames: Number of frames in the GIF.
    :param control_axis: The axis to vary for animation (default: time axis, 0).
    :param fixed_values: A dictionary specifying fixed values for other control axes (e.g., {1: 0.5}).
    """
    print(f"Generating control animation with {num_frames} frames.")
    print(f"Control tensor shape: {control_tensor.shape}")

    if control_axis is None:
        control_axis = 0  # Default to time axis

    # Create a base control vector with fixed values
    base_control = torch.zeros(control_tensor.size(1), device=control_tensor.device)
    if fixed_values:
        for axis, value in fixed_values.items():
            base_control[axis] = value

    # Generate sampled control vectors along the specified axis
    sampled_controls = []
    for i in range(num_frames):
        control = base_control.clone()
        control[control_axis] = i / (num_frames - 1)  # Linearly interpolate from 0 to 1
        sampled_controls.append(control)
    sampled_controls = torch.stack(sampled_controls)
    print(f"Sampled control tensor shape: {sampled_controls.shape}")
    # Generate frames
    frames = []
    outputs = []
    with torch.no_grad():
        for control in sampled_controls:
            reconstructed = model.full_image(control.unsqueeze(0))  # Pass control vector to model
            outputs.append(reconstructed)
            output_image = reconstructed.squeeze(0).cpu().numpy()  # [H, W, C]

            # Gamma correction
            output_image = np.clip(output_image, 0, 1)
            # output_image = output_image ** (1 / 2.2)
            
            # Check if the output has 4 channels and remove the fourth channel if present
            if output_image.shape[-1] == 1:
                frame = (output_image[..., 0] * 255).astype(np.uint8)
                pil_mode = "L"
            elif output_image.shape[-1] in [3, 4]:
                frame = (output_image[..., :3] * 255).astype(np.uint8)
                pil_mode = "RGB"
            frames.append(Image.fromarray(frame, mode=pil_mode))

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0
    )
    print(f"Control animation GIF saved to {gif_path}")
    return torch.stack(outputs), sampled_controls

def generate_control_grid_animation(model, control_tensor, gif_path, num_frames=50, axis_1=0, axis_2=1, axis_2_values=None):
    """
    Generate a grid of GIFs by varying two control axes.
    :param model: The model to generate frames.
    :param control_tensor: The tensor of control vectors.
    :param gif_path: Path to save the generated GIF.
    :param num_frames: Number of frames in the GIF.
    :param axis_1: The primary axis to vary for animation (default: time axis, 0).
    :param axis_2: The secondary axis to vary (default: 1).
    :param axis_2_values: Specific values to sample for the secondary axis (default: [0, 0.25, 0.5, 0.75]).
    """
    print(f"Generating control grid animation with {num_frames} frames per axis.")
    print(f"Control tensor shape: {control_tensor.shape}")

    if axis_2_values is None:
        axis_2_values = [0, 0.25, 0.5, 0.75]

    # Generate GIFs for each value of the secondary axis
    for value in axis_2_values:
        gif_path_with_value = gif_path.replace(".gif", f"_axis2_{value:.2f}.gif")
        fixed_values = {axis_2: value}
        generate_control_animation(
            model, control_tensor, gif_path_with_value, num_frames=num_frames, control_axis=axis_1, fixed_values=fixed_values
        )
        print(f"Generated GIF for axis_2 value {value:.2f} at {gif_path_with_value}")
