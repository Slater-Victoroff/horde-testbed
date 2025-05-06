import os
import json
import torch
import numpy as np
import OpenEXR
import Imath


def resolve_experiment_subfolder(experiment_name, epoch=None):
    experiment_path = os.path.join("debug_outputs", experiment_name)
    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f"Experiment folder not found: {experiment_path}")

    if epoch is not None:
        return os.path.join(experiment_path, f"epoch_{epoch}")
    
    epoch_folders = [f for f in os.listdir(experiment_path) if f.startswith("epoch_")]
    if not epoch_folders:
        raise FileNotFoundError(f"No epoch folders found in: {experiment_path}")
    
    latest_epoch = max(int(f.split('_')[1]) for f in epoch_folders)
    return os.path.join(experiment_path, f"epoch_{latest_epoch}")


def convert_to_webgpu(experiment_name, epoch=None):
    subfolder = resolve_experiment_subfolder(experiment_name, epoch)
    input_path = os.path.join(subfolder, "model_weights.pth")
    weights_output = os.path.join("results", "model_weights.bin")
    manifest_output = os.path.join("results", "model_manifest.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model = torch.load(input_path, map_location='cpu')
    os.makedirs("results", exist_ok=True)

    manifest = {
        "layers": [],
        "dtype": "float32",
        "endianness": "little"
    }

    with open(weights_output, "wb") as f:
        offset = 0
        for name, tensor in model.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Unsupported parameter type: {type(tensor)} for {name}")
            array = tensor.numpy().astype(np.float32)
            f.write(array.tobytes())
            manifest["layers"].append({
                "name": name,
                "shape": list(array.shape),
                "offset": offset,
                "size": array.size
            })
            offset += array.nbytes

    with open(manifest_output, "w") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"Wrote weights to {weights_output}")
    print(f"Wrote manifest to {manifest_output}")


if __name__ == "__main__":
    experiment = "combined-loss"
    convert_to_webgpu(experiment)
