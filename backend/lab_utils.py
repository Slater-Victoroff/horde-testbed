import torch

def normalize_lab(lab_tensor):
    l, a, b = lab_tensor.unbind(-1)
    l = l / 100.0  # Normalize L to [0, 1]
    a = (a + 128) / 255.0  # Normalize a to [0, 1]
    b = (b + 128) / 255.0  # Normalize b to [0, 1]
    return torch.stack([l, a, b], dim=-1)


def denormalize_lab(lab_tensor):
    l, a, b = lab_tensor.unbind(-1)
    l = l * 100.0  # Denormalize L to [0, 100]
    a = a * 255.0 - 128  # Denormalize a to [-128, 127]
    b = b * 255.0 - 128  # Denormalize b to [-128, 127]
    return torch.stack([l, a, b], dim=-1)


def rgb_to_lab(rgb_tensor):
    
    # Convert RGB to LAB using PyTorch operations
    if rgb_tensor.max() > 1.0:  # Check if values are in [0, 255]
        print("Max value greater than 1.0, normalizing to [0, 1]")
        print("max:", rgb_tensor.max())
        rgb_tensor = rgb_tensor / 255.0  # Normalize to [0, 1]
    rgb_tensor = rgb_tensor.clamp(0, 1)  # Ensure values are in [0, 1]
    r, g, b = rgb_tensor.unbind(-1)

    # Convert to XYZ color space
    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b

    # Normalize for D65 illuminant
    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883

    # Apply the f(t) function
    epsilon = 6 / 29
    f = lambda t: torch.where(t > epsilon**3, t**(1/3), (t / (3 * epsilon**2)) + (4 / 29))

    fx = f(x)
    fy = f(y)
    fz = f(z)

    # Convert to LAB
    l = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    lab_tensor = torch.stack([l, a, b], dim=-1)
    return lab_tensor


def lab_to_rgb(lab_tensor):
    # Convert LAB to RGB using PyTorch operations
    l, a, b = lab_tensor.unbind(-1)

    # Convert to XYZ color space
    fy = (l + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    epsilon = 6 / 29
    f_inv = lambda t: torch.where(t > epsilon, t**3, 3 * epsilon**2 * (t - 4 / 29))

    x = f_inv(fx) * 0.95047
    y = f_inv(fy) * 1.00000
    z = f_inv(fz) * 1.08883

    # Convert to RGB color space
    r = 3.240479 * x - 1.537150 * y - 0.498535 * z
    g = -0.969256 * x + 1.875992 * y + 0.041556 * z
    b = 0.055648 * x - 0.204043 * y + 1.057311 * z

    rgb_tensor = torch.stack([r, g, b], dim=-1).clamp(0, 1)
    return rgb_tensor
