import re
import os

import torch
import numpy as np
import OpenEXR
import imageio.v3 as iio
from PIL import Image
from torchvision.utils import save_image
from torchvision.datasets import MovingMNIST
from skimage.color import yuv2rgb

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


def load_images(image_dir, input_image_channels=4, control_channels=2, norm=True, add_next=False):
    gif_files = [gif for gif in image_dir.glob("**/*.gif") if gif.name != "debug.gif"]
    png_files = [png for png in image_dir.glob("**/*.png") if png.name != "debug.png"]

    if gif_files:
        if len(gif_files) > 1:
            raise ValueError("Multiple GIF files found (excluding debug.gif). Please provide only one GIF file.")
        gif_path = gif_files[0]
        frames = iio.imread(gif_path, plugin="pillow")  # Load all frames from the GIF
        # Check overall size of the GIF frames

    elif png_files:
        # sort files by numeric portion of the filename
        png_files.sort(key=lambda x: int(re.search(r'\d+', x.name).group()))
        png_frames = []
        for png_file in png_files:
            png_frame = iio.imread(png_file, plugin="pillow")
            png_frames.append(png_frame)
        frames = np.array(png_frames, dtype=np.float16)
    num_bytes = frames.nbytes
    if num_bytes > 2 * 1024 * 1024 * 1024:
        device = "cpu"
        print(f"PNG files are too large to load into GPU: {num_bytes} bytes, loading on CPU instead.")
    else:
        device = "cuda"
    with torch.no_grad():
        frames = torch.tensor(frames, dtype=torch.float16, device=device)
        normalized_frames = frames / 255.0

    return normalized_frames

def save_images(model, H=256, W=256, n_images=5, gif_frames=20, base_dir=None, write_files=True):
    print(f"Saving images at {H} x {W} resolution")
    reconstructured_images = []
    with torch.no_grad():
        times = torch.rand(n_images, device=model.device)

        # Generate reconstructed RGB images for each selected control vector
        for i, time in enumerate(times):
            reconstructed_rgb = model.full_image(time, H=H, W=W)  # Pass control vector to model
            rgb_image = reconstructed_rgb.squeeze(0).permute(2, 0, 1)  # [C, H, W]
            reconstructured_images.append(rgb_image)

            if write_files:
                rgb_path = os.path.join(base_dir, f"reconstructed_rgb_{time}.png")
                save_image(rgb_image, rgb_path)

        gif_path = None
        if write_files:
            # save_latent_to_exr(latent_grid, latent_path)
            model_path = os.path.join(base_dir, "model_weights.pth")
            torch.save(model.state_dict(), model_path)
            gif_path = os.path.join(base_dir, "control_animation.webp")

        generate_control_animation(model, H=H, W=W, num_frames=gif_frames, gif_path=gif_path, write_files=write_files)
    return torch.stack(reconstructured_images), times

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

            # Preserve all channels including alpha
            if output_image.shape[-1] == 1:
                frame = (output_image[..., 0] * 255).astype(np.uint8)
                pil_mode = "L"
            elif output_image.shape[-1] == 4:
                frame = (output_image * 255).astype(np.uint8)  # Keep all 4 channels
                pil_mode = "RGBA"
            elif output_image.shape[-1] == 3:
                frame = (output_image * 255).astype(np.uint8)
                pil_mode = "RGB"
            frames.append(Image.fromarray(frame, mode=pil_mode))

    if write_files:
        # Change extension to .webp
        webp_path = gif_path.replace('.gif', '.webp')

        # Save as animated WebP with fast encoding
        frames[0].save(
            webp_path,
            save_all=True,
            append_images=frames[1:],
            duration=40,
            loop=0,
            lossless=True,  # Still lossless to avoid artifacts
            method=0        # Fastest encoding method
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
        gif_path_with_value = gif_path.replace(".webp", f"_axis2_{value:.2f}.webp").replace(".gif", f"_axis2_{value:.2f}.webp")
        fixed_values = {axis_2: value}
        generate_control_animation(
            model, control_tensor, gif_path_with_value, num_frames=num_frames, control_axis=axis_1, fixed_values=fixed_values
        )
        print(f"Generated GIF for axis_2 value {value:.2f} at {gif_path_with_value}")


def generate_comparison_gif(original_frames, control_frames, gif_path, duration=40):
    """
    Generate a 4-quadrant WebP animation showing:
    - Top left: Original animation
    - Top right: Control animation (reconstruction)
    - Bottom left: Diff between original and control
    - Bottom right: Renormalized diff (emphasizing areas with largest difference)

    :param original_frames: List or tensor of original animation frames
    :param control_frames: List or tensor of control/reconstructed animation frames
    :param gif_path: Path to save the generated animation (will be saved as WebP)
    :param duration: Duration between frames in milliseconds
    """
    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(original_frames):
        original_frames = [frame.cpu().numpy() for frame in original_frames]
    if torch.is_tensor(control_frames):
        control_frames = [frame.cpu().numpy() for frame in control_frames]

    if len(original_frames) != len(control_frames):
        raise ValueError(f"Frame count mismatch: {len(original_frames)} original vs {len(control_frames)} control")

    composite_frames = []

    for orig_frame, ctrl_frame in zip(original_frames, control_frames):
        # Ensure frames are in [0, 1] range
        orig_frame = np.clip(orig_frame, 0, 1)
        ctrl_frame = np.clip(ctrl_frame, 0, 1)

        # Calculate diff
        diff = (np.abs(orig_frame - ctrl_frame) + 1) / 2
        diff = np.clip(diff, 0, 1)
        # Add one and divide by two to get a range of [0, 1]
        directional_diff = np.clip((ctrl_frame - orig_frame + 1) / 2, 0, 1)

        # Get dimensions
        h, w = orig_frame.shape[:2]

        # Create 2x2 grid
        top_row = np.hstack([orig_frame, ctrl_frame])
        bottom_row = np.hstack([diff, directional_diff])
        composite = np.vstack([top_row, bottom_row])

        # Convert to uint8
        composite_uint8 = (composite * 255).astype(np.uint8)

        # Determine PIL mode based on channels - preserve alpha if present
        if len(composite_uint8.shape) == 2 or composite_uint8.shape[2] == 1:
            if len(composite_uint8.shape) == 3:
                composite_uint8 = composite_uint8[..., 0]
            pil_mode = "L"
        elif composite_uint8.shape[2] == 4:
            pil_mode = "RGBA"  # Preserve alpha channel
        else:
            composite_uint8 = composite_uint8[..., :3]
            pil_mode = "RGB"

        composite_frames.append(Image.fromarray(composite_uint8, mode=pil_mode))

    # Save as WebP animation
    if composite_frames:
        # Change extension to .webp
        webp_path = gif_path.replace('.gif', '.webp')

        # Save as animated WebP with fast encoding
        composite_frames[0].save(
            webp_path,
            save_all=True,
            append_images=composite_frames[1:],
            duration=duration,
            loop=0,
            lossless=True,  # Still lossless to avoid artifacts
            method=0        # Fastest encoding method
        )
        print(f"Saved comparison WebP to {webp_path}")
