import os
import json

import torch
import numpy as np

from single_pixel import VFXNet
from encoding_utils import compute_targeted_encodings

RESULTS_DIR = "results"
DEBUG_DIR = "debug_outputs"

# --- Helpers --------------------------------------------------------------

def resolve_experiment_subfolder(experiment_name: str, epoch: int = None) -> str:
    base = os.path.join(DEBUG_DIR, experiment_name)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Experiment folder not found: {base}")
    if epoch is not None:
        return os.path.join(base, f"epoch_{epoch}")
    # pick latest
    epochs = [d for d in os.listdir(base) if d.startswith("epoch_")]
    if not epochs:
        raise FileNotFoundError(f"No epoch_ folders in: {base}")
    latest = max(int(d.split("_")[1]) for d in epochs)
    return os.path.join(base, f"epoch_{latest}")


def convert_to_webgpu(experiment_name: str, epoch: int = None) -> None:
    """
    Export .pth weights to flat .bin + manifest JSON for WebGPU.
    """
    sub = resolve_experiment_subfolder(experiment_name, epoch)
    src = os.path.join(sub, "model_weights.pth")
    dst_bin = os.path.join(RESULTS_DIR, "model_weights.bin")
    dst_manifest = os.path.join(RESULTS_DIR, "model_manifest.json")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    state = torch.load(src, map_location='cpu')

    manifest = {"layers": [], "dtype": "float32", "endianness": "little"}
    offset = 0

    with open(dst_bin, "wb") as f:
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Unexpected type for {name}: {type(tensor)}")
            arr = tensor.cpu().numpy().astype(np.float32)
            f.write(arr.tobytes())
            manifest["layers"].append({
                "name": name,
                "shape": list(arr.shape),
                "offset": offset,
                "size": arr.size
            })
            offset += arr.nbytes

    with open(dst_manifest, "w") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"Exported weights → {dst_bin}")
    print(f"Exported manifest → {dst_manifest}")


def load_vfxnet(height: int, width: int, ckpt_path: str, device: str = "cpu") -> VFXNet:
    """
    Instantiate VFXNet, load checkpoint, return eval model.
    """
    model = VFXNet(height, width, device=device).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def debug_pixel(
    model: VFXNet,
    x: int,
    y: int,
    t: float,
    layer_idx: int = None
) -> torch.Tensor:
    """
    Return activation at pixel (x,y) and time t.
    layer_idx=None for final RGBA; else hidden-layer index.
    """
    # flat index for raw_pos
    idx = y * model.width + x
    raw_pos = model.raw_pos[idx:idx+1]
    control = torch.tensor([[t, 0.0]], device=raw_pos.device)
    with torch.no_grad():
        out = model.decoder(
            model.shared_latent,
            raw_pos,
            control,
            return_hidden_layer=layer_idx
        )
    return out.squeeze(0).cpu()


def save_debug(x: int, y: int, t: float, vec: torch.Tensor, name: str) -> None:
    """
    Dump a JSON of the vector under RESULTS_DIR.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}_{x}x{y}_t{t:.2f}.json")
    data = {"x": x, "y": y, "t": t, "vec": vec.tolist()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved debug -> {path}")


def load_flat_weights(bin_path: str, manifest_path: str):
    with open(manifest_path) as f:
        manifest = json.load(f)["layers"]
    flat = torch.from_numpy(np.fromfile(bin_path, dtype=np.float32))
    return manifest, flat


def get_weight(manifest, flat, name: str, i: int, j: int = None) -> float:
    m = next(m for m in manifest if m["name"] == name)
    off = m["offset"] // 4
    if j is not None:
        stride = m["shape"][1]
        idx = off + i*stride + j
    else:
        idx = off + i
    return float(flat[idx])

def inspect_film(
        model: VFXNet,
        x: int, y: int, t: float,
        *,
        pos_channels=8,
        spiral_pos_channels=16,
        spiral_time_channels=8,
        device="cpu"
):
    """Return the tensors that matter for the FiLM modulation path."""
    
    H, W = model.height, model.width   # 240, 128 in your case
    
    # ----- latent & “main” input (what you already have) -----------------
    control  = torch.tensor([[t]], device=device)
    u, v     = x / W, y / H

    # ----- build FiLM input exactly like the network does ---------------
    # 1. spiral-position   (16)
    spiral_pos = compute_targeted_encodings(
        torch.tensor([[u, v]], device=device, dtype=torch.float32),
        spiral_pos_channels,
        scheme="spiral", norm_2pi=True, include_norm=True
    )
    
    # 2. spiral-time       (8)
    spiral_time = compute_targeted_encodings(
        control[:, :1],
        spiral_time_channels,
        scheme="spiral", include_raw=True, norm_2pi=True, include_norm=True
    )
    
    # 3. pass through the *learned* embed MLPs inside the decoder
    with torch.no_grad():
        pos_feat  = model.decoder.pos_embed( spiral_pos )
        time_feat = model.decoder.time_embed( spiral_time )
        film_in   = torch.cat([pos_feat, time_feat], dim=-1)      # (32)
        film_out  = model.decoder.film( film_in )                 # (128)
        gamma, beta = film_out.chunk(2, dim=-1)                   # each (64)
    
    return {
        "pos_feat" : pos_feat.squeeze(0).cpu().numpy(),  # 32
        "time_feat": time_feat.squeeze(0).cpu().numpy(), # 32
        "film_in"  : film_in.squeeze(0).cpu().numpy(),   # 32
        "gamma"    : gamma.squeeze(0).cpu().numpy(),     # 64
        "beta"     : beta.squeeze(0).cpu().numpy(),      # 64
    }

# --- Main ----------------------------------------------------------------
if __name__ == "__main__":
    # 1) Export weights + manifest
    # convert_to_webgpu("spiral-spiral-baseline", 34)

    # 2) Inspect a few weights
    manifest, flat = load_flat_weights(
        os.path.join(RESULTS_DIR, "model_weights.bin"),
        os.path.join(RESULTS_DIR, "model_manifest.json")
    )
    # print("Py decoder.layers.0.weight[0,0] =", get_weight(manifest, flat, "decoder.layers.0.weight", 0, 0))

    # 3) Pixel debug
    H, W = 240, 128
    ckpt = os.path.join(
        resolve_experiment_subfolder("spiral-spiral-baseline", 34),
        "model_weights.pth"
    )
    model = load_vfxnet(H, W, ckpt, device="cpu")
    u, v = 0.44, 0.63
    x = int((u * W))               # texCoord.x in WGSL
    y = int((v * H))               # texCoord.y in WGSL
    t = 0.69

    payload = inspect_film(model, x=x, y=y, t=t)
    print("pos_feat[0..7] :", payload["pos_feat"][:8])
    print("time_feat[0..7]:", payload["time_feat"][:8])
    print("gamma    [0..7] :", payload["gamma"])
    print("beta     [0..7] :", payload["beta"])

    trunk = debug_pixel(model, x, y, t, layer_idx=0)
    gamma_tensor = torch.tensor(payload["gamma"], device=trunk.device)
    beta_tensor = torch.tensor(payload["beta"], device=trunk.device)
    preGelu = gamma_tensor * trunk + beta_tensor
    compute_test = debug_pixel(model, x, y, t)
    print("trunk   =", trunk.numpy())
    print("preGelu  :", preGelu.numpy())
    print("compute_test:", compute_test.numpy())

    # x, y, t = W//2, H//2, 0.25
    # trunk = debug_pixel(model, x, y, t, layer_idx=1)
    # final = debug_pixel(model, x, y, t)
    # print("trunk[:8]   =", trunk[:8].numpy())
    # print("final (RGBA)=", final[:4].numpy())
    # save_debug(x, y, t, trunk, "trunk")
    # save_debug(x, y, t, final, "final")
