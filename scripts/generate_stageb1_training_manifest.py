"""
Pre-generates an immutable training manifest for INTERVAL-ROBUST STAGE B.1.
Guarantees that ALL matched arms (Uniform and Randomized, across rank 4 and rank 8)
receive the exact same sequence of:
- step index
- scan name and video index
- sample/view seed
- sampled K trajectory

Saved to outputs/stageb1_training_manifest.csv.
"""

import os
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from dvlt.data.datasets.parser.dataverse import DataverseTrainDataset


def generate_training_manifest(data_root="datasets/test/dtu", output_dir="outputs", total_steps=200, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    manifest_csv = os.path.join(output_dir, "stageb1_training_manifest.csv")

    train_scan_names = ["scan10", "scan11", "scan12", "scan15", "scan23", "scan29", "scan33", "scan48", "scan110"]
    cfg = OmegaConf.create({"target": "dtu.DTU", "params": {"root_path": data_root}})
    ds_train = DataverseTrainDataset(dataverse_cfg=cfg)
    ds_train.set_image_params(504, 14)

    train_video_indices = []
    video_idx_to_name = {}
    for idx in range(ds_train.ds.num_videos()):
        sname = ds_train.ds._get_scan_name(idx)
        if sname in train_scan_names:
            train_video_indices.append(idx)
            video_idx_to_name[idx] = sname

    assert len(train_video_indices) == len(train_scan_names), f"Found {len(train_video_indices)} scans, expected 9."

    data_rng = np.random.RandomState(seed)

    # Beta sampler parameters matching DVLT defaults: num_steps=16, min_steps=8, beta_a=2, beta_b=2
    num_steps = 16
    min_steps = 8
    beta_a = 2
    beta_b = 2

    records = []
    for step in range(total_steps):
        vidx = train_video_indices[data_rng.randint(0, len(train_video_indices))]
        sname = video_idx_to_name[vidx]
        sample_seed = seed + step

        # Sample K deterministically for this step using a dedicated generator
        k_gen = torch.Generator().manual_seed(2026 + step)
        e = torch.empty(beta_a + beta_b).exponential_(generator=k_gen)
        ga = e[:beta_a].sum()
        gb = e[beta_a:].sum()
        m = (ga / (ga + gb)).item()
        span = num_steps - min_steps
        k_sampled = min_steps + min(int(round(m * span)), span)

        records.append({
            "step": step,
            "scan_name": sname,
            "video_idx": vidx,
            "sample_seed": sample_seed,
            "sampled_k": k_sampled,
        })

    df = pd.DataFrame(records)
    df.to_csv(manifest_csv, index=False)
    print(f"Generated immutable training manifest with {len(df)} steps at: {manifest_csv}")
    print("Sample head:")
    print(df.head(10))


if __name__ == "__main__":
    generate_training_manifest()
