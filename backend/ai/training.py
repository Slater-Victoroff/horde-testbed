import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Process
from collections import deque

import torch

from single_pixel import train_vfx_model
from gauges import train_drill_model
from experiments import EXPERIMENTS


def run_single_job(job, static_dir):
    try:
        name = job["name"]
        dataset_path = static_dir / job["dataset"]
        model_type = job.get("model_type", "vfx")
        if model_type == "vfx":
            train_job = train_vfx_model
        elif model_type == "drill":
            train_job = train_drill_model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        decoder_config = job["config"]
        print(f"Starting VFX Model: {name}")
        train_job(
            image_dir=dataset_path,
            device=torch.device("cuda"),
            experiment_name=name,
            decoder_config=decoder_config,
        )
        print(f"Finished: {name}")
        return "done"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"error: {e}")
            print(f"OOM on {name}")
            sys.exit(42)
        raise


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    STATIC_DIR = Path("/app/static")

    queue = deque()
    for exp in EXPERIMENTS:
        queue.append(exp)
    running = []
    mem_full = False

    def launch_next():
        nonlocal mem_full
        if not queue or mem_full:
            return
        job = queue.popleft()
        p = Process(target=run_single_job, args=(job, STATIC_DIR))
        try:
            p.start()
            running.append((p, job))
            time.sleep(5)
            launch_next()
        except RuntimeError as e:
            raise

    launch_next()

    # Monitor loop
    while queue or any(p.is_alive() for p, _ in running):
        still_running = []
        for p, job in running:
            if p.is_alive():
                still_running.append((p, job))
            else:
                exitcode = p.exitcode
                if exitcode == 42:
                    print(f"Job {job['name']} OOM, requeuing.")
                    mem_full = True
                    queue.appendleft(job)
                elif exitcode != 0:
                    print(f"Job {job['name']} exited with code {exitcode}, requeuing.")
                    queue.append(job)  # non-oom failure = back of the queue
                else:
                    print(f"Job {job['name']} completed successfully.")
                    mem_full = False
                    launch_next()
        running[:] = still_running
        time.sleep(5)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
