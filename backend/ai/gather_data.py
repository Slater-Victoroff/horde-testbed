import json
import pandas as pd
from pathlib import Path

def collect_results(root="final_data"):
    rows = []

    for exp_dir in Path(root).iterdir():
        if not exp_dir.is_dir():
            continue

        for epoch_dir in exp_dir.glob("epoch_*"):
            metrics_file = epoch_dir / "summary.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                row = {
                    "experiment": exp_dir.name,
                    "epoch": data.get("epoch"),
                    "loss": data.get("loss"),
                    "ssim": data.get("avg_ssim"),
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(by=["experiment", "epoch"])

if __name__ == "__main__":
    df = collect_results()
    print(df.head())
    df.to_csv("results/all_results.csv", index=False)
