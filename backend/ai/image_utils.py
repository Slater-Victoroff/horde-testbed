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
        gif_files = [gif for gif in image_dir.glob("**/*.gif") if gif.name != "debug.gif"]
        png_files = [png for png in sorted(image_dir.glob("**/*.png")) if png.name != "debug.png"]
        exr_files = [exr for exr in sorted(image_dir.glob("**/*.exr")) if exr.name != "debug.exr"]
        if gif_files:
            if len(gif_files) > 1:
                raise ValueError("Multiple GIF files found (excluding debug.gif). Please provide only one GIF file.")
            gif_path = gif_files[0]
            gif_frames = iio.imread(gif_path, plugin="pillow")  # Load all frames from the GIF
            # Check overall size of the GIF frames
            num_bytes = gif_frames.nbytes
            if num_bytes > 4 * 1024 * 1024 * 1024:  # 4 GB limit
                raise ValueError(f"GIF file is too large to load into GPU: {num_bytes} bytes")
            else:
                with torch.no_grad():
                    gif_frames = torch.tensor(gif_frames, dtype=torch.float32, device=device)
                    normalized_frames = gif_frames / 255.0


        elif png_files:
            if add_next:
                prior_frame = None
            for image_path in png_files:
                image = Image.open(image_path).convert("RGBA")
                image = np.array(image) / 255.0
                matches = re.findall(r'\d+', image_path.stem)
                time = int(matches[0])
                if add_next:
                    if prior_frame is not None:
                        _process_frame(image, time, int(matches[1]), prior_frame)
                    prior_frame = image
                else:
                    _process_frame(image, time)

        elif exr_files:
            # Process EXR files
            if add_next:
                prior_frame = None
            for image_path in exr_files:  # Assuming EXR images
                image = load_exr_image(image_path)
                # Extract control values from the filename using regex
                matches = re.findall(r'\d+', image_path.stem)
                time = int(matches[1])
                if add_next:
                    if prior_frame is not None:
                        _process_frame(image, time, int(matches[0]), prior_frame)
                    prior_frame = image
                else:
                    _process_frame(image, time)

    return normalized_frames

def save_images(model, H=256, W=256, n_images=5, gif_frames=20, base_dir=None, write_files=True):
    with torch.no_grad():
        times = torch.rand(n_images, device=model.device)

        # Generate reconstructed RGB images for each selected control vector
        for i, time in enumerate(times):
            reconstructed_rgb = model.full_image(time, H=H, W=W)  # Pass control vector to model
            rgb_image = reconstructed_rgb.squeeze(0).permute(2, 0, 1)  # [C, H, W]

            if write_files:
                rgb_path = os.path.join(base_dir, f"reconstructed_rgb_{time}.png")
                save_image(rgb_image, rgb_path)

        gif_path = None
        if write_files:
            # save_latent_to_exr(latent_grid, latent_path)
            model_path = os.path.join(base_dir, "model_weights.pth")
            torch.save(model.state_dict(), model_path)
            gif_path = os.path.join(base_dir, "control_animation.gif")

        generate_control_animation(model, H=H, W=W, num_frames=gif_frames, gif_path=gif_path, write_files=write_files)


def generate_control_animation(model, H, W, num_frames=20, gif_path=None, write_files=True):
    time_control = torch.linspace(0, 1, num_frames, device=model.device).unsqueeze(1)  # [num_frames, 1]

    # Generate frames
    frames = []
    outputs = []
    with torch.no_grad():
        for time in time_control:
            reconstructed = model.full_image(time, H=H, W=W)  # Pass control vector to model
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

    if write_files:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=40,
            loop=0
        )

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
