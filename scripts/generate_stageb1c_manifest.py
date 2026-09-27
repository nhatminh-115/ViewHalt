"""
STAGE B.1C: GENERATE OFFICIAL-K TRAINING MANIFEST
Preserves exactly:
- step
- scan_name
- video_idx
- sample_seed
from outputs/stageb1_training_manifest.csv.

Replaces ONLY sampled_k using the exact DVLT runtime implementation:
    generator.manual_seed(42 + step)
    K = inner._sample_K(generator)
with confirmed parameters: min_steps=8, num_steps=16, beta_a=2, beta_b=1.

Saves to: outputs/stageb1c_official_k_manifest.csv
"""

import os
import pandas as pd
import torch
from dvlt.model.dvlt.model import DVLT

def generate_stageb1c_manifest():
    old_manifest_path = "outputs/stageb1_training_manifest.csv"
    assert os.path.exists(old_manifest_path), f"Missing {old_manifest_path}"
    df_old = pd.read_csv(old_manifest_path)

    # Initialize DVLT to use its exact native _sample_K
    model = DVLT(img_size=504, depth_head_type="conv")
    inner = model.model

    print("=" * 80)
    print("STAGE B.1C: OFFICIAL-K MANIFEST GENERATION")
    print("=" * 80)
    print("Runtime confirmed parameters:")
    print(f"  min_steps: {inner.min_steps}")
    print(f"  num_steps: {inner.num_steps}")
    print(f"  beta_a: {inner.k_sampler_beta_a}")
    print(f"  beta_b: {inner.k_sampler_beta_b}")

    df_new = df_old.copy()
    new_k_list = []

    generator = torch.Generator()
    for step in range(len(df_new)):
        generator.manual_seed(42 + step)
        k_val = inner._sample_K(generator)
        new_k_list.append(k_val)

    df_new["sampled_k"] = new_k_list

    out_manifest = "outputs/stageb1c_official_k_manifest.csv"
    df_new.to_csv(out_manifest, index=False)
    print(f"\nSaved corrected manifest: {out_manifest}")

    old_mean = df_old["sampled_k"].mean()
    new_mean = df_new["sampled_k"].mean()
    print(f"\nOld manifest mean K (Beta 2,2): {old_mean:.4f}")
    print(f"Corrected manifest mean K (Beta 2,1): {new_mean:.4f}")

    print("\nCorrected K histogram (distribution across 200 steps):")
    counts = df_new["sampled_k"].value_counts().sort_index()
    for k, count in counts.items():
        bar = "#" * int(count)
        print(f"  K={k:2d}: {count:3d} ({count / len(df_new) * 100:5.1f}%) {bar}")

    print("\nHead of corrected manifest (first 10 steps):")
    print(df_new.head(10))

if __name__ == "__main__":
    generate_stageb1c_manifest()
