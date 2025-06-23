import os
from PIL import Image
import imageio
import argparse
import numpy as np


def recenter_frame(frame: Image.Image, method='center') -> Image.Image:
    """
    Re-center the non-transparent pixels of `frame` in a new same-sized RGBA canvas.
    method:
      - 'center'     → content center → cell center
      - 'bottom'     → content bottom → cell bottom (good for ground-anchored VFX)
    """
    frame = frame.convert('RGBA')
    alpha = frame.split()[-1]
    bbox = alpha.getbbox()  # (left, upper, right, lower) of non-transparent pixels
    if not bbox:
        return frame  # entirely empty cell

    left, upper, right, lower = bbox
    bw, bh = right - left, lower - upper
    fw, fh = frame.size

    if method == 'center':
        x_off = (fw - bw)//2 - left
        y_off = (fh - bh)//2 - upper
    elif method == 'bottom':
        x_off = (fw - bw)//2 - left
        y_off = fh - lower
    else:
        raise ValueError(f"Unknown recenter method: {method}")

    canvas = Image.new('RGBA', (fw, fh), (0,0,0,0))
    canvas.paste(frame, (x_off, y_off))
    return canvas


def fix_frame_alignment(frames, cell_w, cell_h, pad=2):
    """
    Crop every frame to the *same* union bbox of all non-transparent pixels,
    then bottom-anchor each into a new (cell_w x cell_h) canvas.

    Args:
      frames   – list of PIL.Image RGBA frames
      cell_w   – original frame width
      cell_h   – original frame height
      pad      – extra pixels padding around the union bbox
    Returns:
      new_frames – list of PIL.Image, all size (cell_w, cell_h)
    """
    # 1) Gather all non-empty bboxes
    bboxes = []
    for f in frames:
        alpha = f.split()[-1]
        # optional threshold to ignore tiny specks:
        # alpha = alpha.point(lambda v: 255 if v > 16 else 0)
        bb = alpha.getbbox()
        if bb:
            bboxes.append(bb)

    if not bboxes:
        return frames  # nothing to do

    # 2) Compute union bbox
    min_l = min(bb[0] for bb in bboxes) - pad
    min_u = min(bb[1] for bb in bboxes) - pad
    max_r = max(bb[2] for bb in bboxes) + pad
    max_b = max(bb[3] for bb in bboxes) + pad

    # clamp to cell
    min_l = max(0, min_l)
    min_u = max(0, min_u)
    max_r = min(cell_w,  max_r)
    max_b = min(cell_h, max_b)

    fixed_w = max_r - min_l
    fixed_h = max_b - min_u

    # 3) Re-crop & paste into new canvases
    new_frames = []
    for f in frames:
        cropped = f.crop((min_l, min_u, max_r, max_b))
        canvas = Image.new("RGBA", (cell_w, cell_h), (0,0,0,0))
        # bottom-anchor
        x = (cell_w - fixed_w)//2
        y = cell_h - fixed_h
        canvas.paste(cropped, (x, y))
        new_frames.append(canvas)

    return new_frames


def compute_pivots(frames, method='centroid'):
    """
    Returns a list of (cx,cy) pivots for each frame.
    - 'centroid' finds the center of mass of the alpha mask.
    - 'bottom_center' finds the midpoint of the bottom edge of the bbox.
    """
    pivots = []
    for f in frames:
        alpha = np.array(f.getchannel('A'))
        mask = alpha > 16    # ignore tiny specks
        if not mask.any():
            pivots.append((f.width/2, f.height/2))
            continue

        ys, xs = np.nonzero(mask)
        if method == 'centroid':
            cx, cy = xs.mean(), ys.mean()
        else:  # bottom_center
            # bbox bottom:
            min_x, max_x = xs.min(), xs.max()
            bottom_y = ys.max()
            cx, cy = (min_x + max_x)/2, bottom_y
        pivots.append((cx, cy))
    return pivots


def smooth_pivots(pivots, radius=2):
    """
    Simple moving‐average over a window of 2*radius+1 frames.
    """
    N = len(pivots)
    smoothed = []
    for i in range(N):
        lo = max(0, i-radius)
        hi = min(N, i+radius+1)
        window = pivots[lo:hi]
        xs, ys = zip(*window)
        smoothed.append((sum(xs)/len(xs), sum(ys)/len(ys)))
    return smoothed


def apply_smoothed_alignment(frames, smoothed_pivots, cell_w, cell_h, target_pivot):
    """
    Paste each frame onto a new (cell_w,cell_h) canvas so that
    its smoothed pivot lands at target_pivot (x0,y0).
    """
    aligned = []
    for f, (cx, cy) in zip(frames, smoothed_pivots):
        dx = target_pivot[0] - cx
        dy = target_pivot[1] - cy
        canvas = Image.new('RGBA', (cell_w, cell_h), (0,0,0,0))
        # round offsets
        canvas.paste(f, (int(dx), int(dy)), f)
        aligned.append(canvas)
    return aligned


def slice_sprite_sheet(image_path, output_folder, rows=None, cols=None, frame_width=None, frame_height=None):
    """
    Slice a flipsheet (sprite sheet) into individual frame images.
    You can specify either:
      - rows and cols (number of rows/columns in the sheet), or
      - frame_width and frame_height (dimensions of each frame).
    """
    img = Image.open(image_path)
    sheet_w, sheet_h = img.size

    # Determine layout
    if rows and cols:
        fw = sheet_w // cols
        fh = sheet_h // rows
    elif frame_width and frame_height:
        fw = frame_width
        fh = frame_height
        cols = sheet_w // fw
        rows = sheet_h // fh
    else:
        raise ValueError("Specify either rows & cols or frame_width & frame_height")

    os.makedirs(output_folder, exist_ok=True)
    frames = []
    count = 0

    for r in range(rows):
        for c in range(cols):
            left = c * fw
            upper = r * fh
            right = left + fw
            lower = upper + fh
            frame = img.crop((left, upper, right, lower))
            frame = recenter_frame(frame, method=args.recenter or 'center')
            frame_path = os.path.join(output_folder, f"frame_{count:03d}.png")
            frame.save(frame_path)
            frames.append(frame)
            count += 1

    frames = fix_frame_alignment(frames, fw, fh, pad=2)
    raw_pivots = compute_pivots(frames, method='bottom_center')
    smoothed   = smooth_pivots(raw_pivots, radius=2)

    target_pivot = (fw/2, fh - 10)

    # 4) re-anchor
    frames = apply_smoothed_alignment(frames, smoothed, fw, fh, target_pivot)
    return frames

def create_gif(frames, gif_path, duration=0.1):
    """Combine extracted frames into a looping GIF."""
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration * 1000),
        loop=0,
        disposal=2,
        transparency=0
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice a flipsheet and optionally generate a GIF for debugging.")
    parser.add_argument("image_path", help="Path to the flipsheet PNG")
    parser.add_argument("output_folder", help="Directory to save individual frames/GIF")
    parser.add_argument("--rows", type=int, help="Number of rows in the sheet")
    parser.add_argument("--cols", type=int, help="Number of columns in the sheet")
    parser.add_argument("--frame-width", type=int, help="Width of each frame (px)")
    parser.add_argument("--frame-height", type=int, help="Height of each frame (px)")
    parser.add_argument("--gif", action="store_true", help="Also create a GIF from the frames")
    parser.add_argument("--duration", type=float, default=0.1, help="Frame duration for GIF (seconds)")
    parser.add_argument(
        '--recenter',
        choices=['center','bottom'],
        help="re-align each sprite's content bbox to the cell (center or bottom)"
    )
    args = parser.parse_args()

    # Slice into frames
    frames = slice_sprite_sheet(
        args.image_path,
        args.output_folder,
        rows=args.rows,
        cols=args.cols,
        frame_width=args.frame_width,
        frame_height=args.frame_height
    )
    print(f"Extracted {len(frames)} frames into '{args.output_folder}'")

    # Optionally build a GIF
    if args.gif:
        gif_path = os.path.join(args.output_folder, "debug.gif")
        create_gif(frames, gif_path, duration=args.duration)
        print(f"Saved debug GIF to '{gif_path}'")
