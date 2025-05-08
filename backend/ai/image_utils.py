import re
import os

import torch
import numpy as np
import OpenEXR
import imageio.v3 as iio
from torchvision.utils import save_image

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


def load_images(image_dir, device, input_image_channels=4, control_channels=2):
    image_tensors = []
    pos_tensors = []
    control_tensors = []
    shape = None

    def _process_frame(frame, frame_idx=None):
        nonlocal shape
        # frame = remove_background(frame)
        if frame.shape[-1] < input_image_channels:
            # Pad with 1s to match the required number of channels
            padding = np.ones((*frame.shape[:-1], input_image_channels - frame.shape[-1]), dtype=frame.dtype)
            frame = np.concatenate([frame, padding], axis=-1)
        if shape is None:
            shape = frame.shape
        elif shape != frame.shape:
            raise ValueError(f"Frame shape mismatch: {shape} != {frame.shape}")
        image_tensor = torch.tensor(frame, dtype=torch.float32, device=device).view(-1, input_image_channels)  # Flatten to [H*W, 4]
        image_tensors.append(image_tensor)

        # Generate control tensor
        control_values = [frame_idx] if frame_idx is not None else []
        control_tensor = torch.tensor(control_values + [0] * (control_channels - len(control_values)), device=device)
        control_tensors.append(control_tensor.repeat(image_tensor.size(0), 1))  # [H*W, CONTROL_CHANNELS]
        # Generate positional embeddings
        H, W, _ = frame.shape
        x_coords = torch.arange(W, device=device).repeat(H, 1).view(-1, 1)  # [H*W, 1]
        y_coords = torch.arange(H, device=device).repeat(W, 1).t().contiguous().view(-1, 1)  # [H*W, 1]
        pos_tensor = torch.cat([x_coords, y_coords], dim=1)  # [H*W, 2]
        pos_tensors.append(pos_tensor)

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
        for image_path in sorted(image_dir.glob("**/*.exr")):  # Assuming EXR images
            image = load_exr_image(image_path)
            # Extract control values from the filename using regex
            matches = re.findall(r'\d+', image_path.stem)
            control_values = [int(match) for match in matches]
            _process_frame(image, control_values[1] if control_values else None)


    # Combine all tensors
    image_tensor = torch.cat(image_tensors, dim=0)  # [H*W*images, INPUT_IMAGE_CHANNELS]
    image_tensor = image_tensor / image_tensor.max()  # Normalize to [0, 1]
    raw_pos = torch.cat(pos_tensors, dim=0)  # [H*W*images, 2]
    control = torch.cat(control_tensors, dim=0)  # [H*W*images, CONTROL_CHANNELS]
    print("image_tensor shape:", image_tensor.shape)
    print("raw_pos shape:", raw_pos.shape)
    print("control tensor shape:", control.shape)
    print("torch unique control tensor:", torch.unique(control[:, 1], dim=0))
    # Identify the time axis
    time_axis = None
    h, w = shape[:2]  # Height and width of the image
    step = h * w  # Step size to sample every frame
    print(f"Step size for sampling: {step}")

    for i in range(control.size(1)):
        sampled_values = control[::step, i]  # Sample every h*w values
        unique_values = torch.unique(sampled_values)
        if len(unique_values) > 1 and torch.all(unique_values[1:] - unique_values[:-1] > 0):  # Check for increasing values
            time_axis = i
            break

    if time_axis is not None and time_axis != 0:
        # Swap the time axis to the first position
        control = torch.cat([control[:, time_axis:time_axis + 1], control[:, :time_axis], control[:, time_axis + 1:]], dim=1)
    elif time_axis is None:
        raise ValueError("Unable to identify a time axis in the control tensor.")

    # Normalize each control channel to the range [0, 1]
    control_min, _ = control.min(dim=0, keepdim=True)
    control_max, _ = control.max(dim=0, keepdim=True)
    control = (control - control_min) / (control_max - control_min + 1e-8)  # Add epsilon to avoid division by zero

    image_tensor = image_tensor.to(device)
    control = control.to(device)
    raw_pos = raw_pos.to(device)
    return image_tensor, raw_pos, control, shape

def save_images(model, control_tensor, epoch, n=5):
    # Create directory structure
    base_dir = "debug_outputs"
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
        latent_path = os.path.join(epoch_dir, "shared_latent.exr")
        save_latent_to_exr(model.shared_latent, latent_path)
        print(f"Shared latent image saved to {latent_path}")

        # Save the model/weights
        model_path = os.path.join(epoch_dir, "model_weights.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")

        # Generate and save a GIF from evenly sampled control vectors
        gif_path = os.path.join(epoch_dir, "control_animation.gif")
        print(f"Control tensor shape: {control_tensor.shape}")
        generate_control_animation(model, control_tensor, gif_path)
        print(f"Control animation GIF saved to {gif_path}")


def generate_control_animation(model, control_tensor, gif_path, num_frames=50, control_axis=None, fixed_values=None):
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
    with torch.no_grad():
        for control in sampled_controls:
            reconstructed_rgb = model.full_image(control.unsqueeze(0))  # Pass control vector to model
            rgb_image = reconstructed_rgb.squeeze(0).cpu().numpy()  # [H, W, C]
            rgb_image = np.clip(rgb_image, 0, 1)
            rgb_image = rgb_image ** (1 / 2.2)
            
            # Check if the output has 4 channels and remove the fourth channel if present
            if rgb_image.shape[-1] == 4:
                rgb_image = rgb_image[..., :3]
            
            frame = (rgb_image * 255).astype(np.uint8)  # Convert to uint8 for GIF
            frames.append(frame)

    iio.imwrite(gif_path, frames)
    print(f"Control animation GIF saved to {gif_path}")

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
