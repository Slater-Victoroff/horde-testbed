import json
from datetime import datetime
from pathlib import Path

import torch

from single_pixel import train_vfx_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    STATIC_DIR = Path("/app/static")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    model = train_vfx_model(STATIC_DIR / "VFX/clouds/cloud01", device=device, experiment_name="no-film-cloud")
    
    # Save model state
    model_path = results_dir / f"vfx_model_combined{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(model.state_dict(), model_path)

    # model.save_as_glsl(results_dir, test_images)
    
    print(f"Model saved to {model_path}")
    print(f"GLSL shader saved to {results_dir}")

if __name__ == "__main__":
    main()
