"""
Generates publication-quality visualizations for INTERVAL-ROBUST STAGE B.1 REPAIR:
1. Treatment Effect vs Training Step (Rank 4 and Rank 8)
2. Quality vs Training Step (Uniform vs Rand vs Pretrained K=14 Hurdle)
3. Robustness Curves at Selected Checkpoints (e.g., Step 50, Step 100, Step 200)
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


def plot_stageb1_repair(output_dir="outputs", artifact_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)

    matched_csv = os.path.join(output_dir, "stageb1_checkpoint_matched.csv")
    nonunif_csv = os.path.join(output_dir, "stageb1_nonuniform_by_checkpoint.csv")
    robust_csv = os.path.join(output_dir, "stageb1_robustness_by_checkpoint.csv")

    assert os.path.exists(matched_csv), f"Missing {matched_csv}"
    assert os.path.exists(nonunif_csv), f"Missing {nonunif_csv}"
    assert os.path.exists(robust_csv), f"Missing {robust_csv}"

    df_matched = pd.read_csv(matched_csv)
    df_nonunif = pd.read_csv(nonunif_csv)
    df_robust = pd.read_csv(robust_csv)

    pre_k14_hurdle = float(df_matched["pre_k14_hurdle"].iloc[0]) * 1000

    # ----------------------------------------------------
    # Plot 1: Treatment Effect vs Training Step
    # ----------------------------------------------------
    # Treatment gain = Uniform Control - Rand FT (Positive = Rand is better)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    ranks = [4, 8]
    colors = {4: "#e63946", 8: "#2a9d8f"}

    for r in ranks:
        sub = df_matched[df_matched["rank"] == r].sort_values("step")
        steps = sub["step"].values
        gain_k12 = sub["k12_treatment_gain"].values * 1000
        gain_nonunif = sub["nonunif_treatment_gain"].values * 1000

        ax1.plot(steps, gain_k12, "o-", color=colors[r], linewidth=2.0, markersize=7, label=f"Rank {r} (LoRA)")
        ax2.plot(steps, gain_nonunif, "s-", color=colors[r], linewidth=2.0, markersize=7, label=f"Rank {r} (LoRA)")

    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel(r"Treatment Gain on Uniform K=12 $(\times 10^{-3})$" + "\n" + r"[$\mathrm{Uniform} - \mathrm{Rand}$, Positive = Rand Better]")
    ax1.set_title("Treatment Effect on Uniform K=12 vs. Step", pad=10, fontweight="bold")
    ax1.set_xticks([0, 25, 50, 100, 200])
    ax1.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel(r"Treatment Gain on Non-Uniform K=12 Mean $(\times 10^{-3})$" + "\n" + r"[$\mathrm{Uniform} - \mathrm{Rand}$, Positive = Rand Better]")
    ax2.set_title("Treatment Effect on Non-Uniform Schedules vs. Step", pad=10, fontweight="bold")
    ax2.set_xticks([0, 25, 50, 100, 200])
    ax2.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plot1_path = os.path.join(output_dir, "stageb1_treatment_effect_vs_step.png")
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")

    # ----------------------------------------------------
    # Plot 2: Quality vs Training Step (Rank 4 & Rank 8)
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    for ax, r, title in [(ax1, 4, "Stage B1 (LoRA Rank 4)"), (ax2, 8, "Stage B2 (LoRA Rank 8)")]:
        sub = df_matched[df_matched["rank"] == r].sort_values("step")
        steps = sub["step"].values

        ax.plot(steps, sub["uniform_control_k12"] * 1000, "o--", color="#457b9d", linewidth=1.8, markersize=6, label="Uniform FT Control (K=12)")
        ax.plot(steps, sub["mild_rand_k12"] * 1000, "*-", color="#e63946", linewidth=2.0, markersize=8, label="Mild Rand FT (K=12)")
        ax.plot(steps, sub["uniform_control_nonunif_mean"] * 1000, "d--", color="#2a9d8f", linewidth=1.8, markersize=6, label="Uniform FT (Non-Unif Mean)")
        ax.plot(steps, sub["mild_rand_nonunif_mean"] * 1000, "P-", color="#f4a261", linewidth=2.0, markersize=8, label="Mild Rand FT (Non-Unif Mean)")

        ax.axhline(pre_k14_hurdle, color="#6c757d", linestyle=":", linewidth=1.5, label=f"Pretrained Uniform K=14 ({pre_k14_hurdle:.3f})")

        ax.set_xlabel("Training Step")
        ax.set_ylabel(r"Depth AbsRel Error $(\times 10^{-3})$")
        ax.set_title(title, pad=10, fontweight="bold")
        ax.set_xticks([0, 25, 50, 100, 200])
        ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", loc="upper left", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plot2_path = os.path.join(output_dir, "stageb1_quality_vs_step.png")
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

    # ----------------------------------------------------
    # Plot 3: Robustness Curves at Key Checkpoints (Step 50 vs Step 100 vs Step 200)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    plot_steps = [50, 100, 200]

    for ax, s in zip(axes, plot_steps):
        # Pretrained curve
        pre_sub = df_robust[(df_robust["rank"] == 4) & (df_robust["step"] == 0)].sort_values("target_cv")
        ax.plot(pre_sub["target_cv"], pre_sub["pretrained_absrel"] * 1000, "k-", linewidth=2.0, label="Pretrained Baseline")

        # Rank 4
        sub_r4 = df_robust[(df_robust["rank"] == 4) & (df_robust["step"] == s)].sort_values("target_cv")
        ax.plot(sub_r4["target_cv"], sub_r4["uniform_control_absrel"] * 1000, "s--", color="#457b9d", linewidth=1.8, label="R4 Uniform Control")
        ax.plot(sub_r4["target_cv"], sub_r4["mild_rand_absrel"] * 1000, "*-", color="#e63946", linewidth=2.0, label="R4 Mild Rand FT")

        # Rank 8
        sub_r8 = df_robust[(df_robust["rank"] == 8) & (df_robust["step"] == s)].sort_values("target_cv")
        ax.plot(sub_r8["target_cv"], sub_r8["uniform_control_absrel"] * 1000, "d--", color="#2a9d8f", linewidth=1.8, label="R8 Uniform Control")
        ax.plot(sub_r8["target_cv"], sub_r8["mild_rand_absrel"] * 1000, "P-", color="#f4a261", linewidth=2.0, label="R8 Mild Rand FT")

        ax.axhline(pre_k14_hurdle, color="#6c757d", linestyle=":", linewidth=1.2, label=f"Pretrained K=14 ({pre_k14_hurdle:.3f})")

        ax.set_xlabel(r"Interval Perturbation $CV(\Delta t)$")
        ax.set_ylabel(r"Depth AbsRel Error $(\times 10^{-3})$")
        ax.set_title(f"Checkpoint Step {s}", pad=10, fontweight="bold")
        ax.set_xticks([0.0, 0.2, 0.4, 0.6])
        if s == 50:
            ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=8.5, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)

    plot3_path = os.path.join(output_dir, "stageb1_robustness_at_checkpoints.png")
    plt.tight_layout()
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path}")

    # Copy to artifacts directory
    if artifact_dir:
        import shutil
        for p in [plot1_path, plot2_path, plot3_path]:
            shutil.copy(p, artifact_dir)
        print(f"Copied all plots to {artifact_dir}")


if __name__ == "__main__":
    plot_stageb1_repair(artifact_dir="C:/Users/Minh/.gemini/antigravity-ide/brain/b788712f-4175-4ebd-a0ba-b4c45e1d8582")
