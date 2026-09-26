"""
Generates publication-quality visualizations for INTERVAL-ROBUST STAGE B:
1. Capacity vs Treatment Gain (Stage A vs Stage B1 vs Stage B2)
2. Robustness Curves (CV vs AbsRel / degradation)
3. Quality vs K (Uniform K in {10, 12, 14, 16} with Pretrained K=14 hurdle)
4. Hidden-State Drift by Iteration ({4, 8, 12, 14, 16} under K=16)
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

def plot_stageb(output_dir="outputs", artifact_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)

    matched_csv = os.path.join(output_dir, "stageb_matched_results.csv")
    nonunif_csv = os.path.join(output_dir, "stageb_nonuniform_results.csv")
    robust_csv = os.path.join(output_dir, "stageb_robustness_curve.csv")
    hidden_csv = os.path.join(output_dir, "stageb_hidden_drift.csv")
    audit_json = os.path.join(output_dir, "stageb_capacity_audit.json")

    df_matched = pd.read_csv(matched_csv)
    df_nonunif = pd.read_csv(nonunif_csv)
    df_robust = pd.read_csv(robust_csv)
    df_hidden = pd.read_csv(hidden_csv)
    with open(audit_json, "r") as f:
        capacity_audit = json.load(f)

    # ----------------------------------------------------
    # Plot 1: Robustness Curves (CV vs AbsRel)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    models = df_robust["model"].unique()
    palette = {
        "Pretrained": ("#2b2d42", "o-", 2.0, "Pretrained Baseline"),
        "B1_Uniform_FT": ("#457b9d", "s--", 1.8, "B1 (r=4) Uniform FT Control"),
        "B1_Mild_Rand_FT": ("#e63946", "*-", 2.2, "B1 (r=4) Mild Rand FT (sigma=0.2)"),
        "B2_Uniform_FT": ("#2a9d8f", "d--", 1.8, "B2 (r=8) Uniform FT Control"),
        "B2_Mild_Rand_FT": ("#f4a261", "P-", 2.2, "B2 (r=8) Mild Rand FT (sigma=0.2)"),
    }

    for m in models:
        sub = df_robust[df_robust["model"] == m].sort_values("target_cv")
        color, fmt, lw, label = palette.get(m, ("gray", "o-", 1.5, m))
        ax.plot(sub["target_cv"], sub["abs_rel"] * 1000, fmt, color=color, linewidth=lw, markersize=7, label=label)

    pre_k14_row = df_matched[(df_matched["model"] == "Pretrained") & (df_matched["K"] == 14)]
    if not pre_k14_row.empty:
        pre_k14_absrel = pre_k14_row.iloc[0]["abs_rel"] * 1000
        ax.axhline(pre_k14_absrel, color="#6c757d", linestyle=":", linewidth=1.5, label=f"Pretrained Uniform K=14 ({pre_k14_absrel:.3f})")

    ax.set_xlabel(r"Interval Schedule Perturbation $CV(\Delta t)$")
    ax.set_ylabel(r"Depth AbsRel Error $(\times 10^{-3})$")
    ax.set_title("Stage B Robustness Curves: Recurrent LoRA Under Non-Uniform Intervals", pad=12, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plot1_path = os.path.join(output_dir, "stageb_robustness_curves.png")
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")

    # ----------------------------------------------------
    # Plot 2: Quality vs K (Uniform K in {10, 12, 14, 16})
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    for m in models:
        sub = df_matched[df_matched["model"] == m].sort_values("K")
        color, fmt, lw, label = palette.get(m, ("gray", "o-", 1.5, m))
        ax.plot(sub["K"], sub["abs_rel"] * 1000, fmt, color=color, linewidth=lw, markersize=7, label=label)

    if not pre_k14_row.empty:
        ax.axhline(pre_k14_absrel, color="#6c757d", linestyle=":", linewidth=1.5, label=f"K=14 Compute Hurdle ({pre_k14_absrel:.3f})")

    ax.set_xlabel("Number of Recurrent Iterations (K)")
    ax.set_ylabel(r"Depth AbsRel Error $(\times 10^{-3})$")
    ax.set_title("Stage B Quality vs. Compute (Uniform Recurrent Schedules)", pad=12, fontweight="bold")
    ax.set_xticks([10, 12, 14, 16])
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plot2_path = os.path.join(output_dir, "stageb_quality_vs_k.png")
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

    # ----------------------------------------------------
    # Plot 3: Capacity Ladder vs Treatment Gain
    # ----------------------------------------------------
    # Calculate treatment gain: (Uniform FT - Rand FT) across Stage A, Stage B1, Stage B2
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)

    # Stage A reference from Stage A evaluation
    # Stage A: 316,032 params. K=12 non-uniform avg gain ~= -0.000007 (or negligible)
    stages = ["Stage A\n(Gate Only)", "Stage B1\n(LoRA r=4)", "Stage B2\n(LoRA r=8)"]
    params = [
        capacity_audit["interval_depth_scaling_alone"]["trainable_params"],
        capacity_audit["stage_b1_rank4"]["total_trainable_params"],
        capacity_audit["stage_b2_rank8"]["total_trainable_params"]
    ]

    # Compute mean treatment gain on K=12 across non-uniform schedules
    # Treatment gain = AbsRel(Uniform_FT) - AbsRel(Mild_Rand_FT) (positive means Rand is better)
    nonunif_sub = df_nonunif[df_nonunif["schedule_key"] != "uniform_K12"]
    
    # B1 gain
    b1_u = nonunif_sub[nonunif_sub["model"] == "B1_Uniform_FT"]["abs_rel"].mean()
    b1_r = nonunif_sub[nonunif_sub["model"] == "B1_Mild_Rand_FT"]["abs_rel"].mean()
    b1_gain = (b1_u - b1_r) * 1000

    # B2 gain
    b2_u = nonunif_sub[nonunif_sub["model"] == "B2_Uniform_FT"]["abs_rel"].mean()
    b2_r = nonunif_sub[nonunif_sub["model"] == "B2_Mild_Rand_FT"]["abs_rel"].mean()
    b2_gain = (b2_u - b2_r) * 1000

    # Stage A gain (from Stage A report: Uniform FT avg 0.005822 vs Mild 0.005829 -> gain = -0.007e-3)
    stage_a_gain = (0.005822 - 0.005829) * 1000

    gains = [stage_a_gain, b1_gain, b2_gain]
    colors = ["#2b2d42" if g >= 0 else "#d90429" for g in gains]

    bars = ax.bar(stages, gains, color=["#457b9d", "#e63946", "#2a9d8f"], width=0.45, edgecolor="black", alpha=0.85)
    ax.axhline(0.0, color="black", linestyle="-", linewidth=1.0)

    for bar, g, p in zip(bars, gains, params):
        y_pos = bar.get_height() + (0.005 if g >= 0 else -0.015)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{g:+.3f}e-3\n({p:,} params)",
                ha="center", va="bottom" if g >= 0 else "top", fontsize=9.5, fontweight="bold")

    ax.set_ylabel(r"Treatment Gain on Non-Uniform Schedules $(\times 10^{-3})$" + "\n" + r"[$\Delta \mathrm{AbsRel} = \mathrm{Uniform} - \mathrm{Rand}$, Positive = Rand Better]")
    ax.set_title("Capacity vs. Treatment Gain: Impact of Recurrent Adaptation", pad=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plot3_path = os.path.join(output_dir, "stageb_capacity_vs_gain.png")
    plt.tight_layout()
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path}")

    # ----------------------------------------------------
    # Plot 4: Hidden-State Drift by Recurrent Iteration
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)

    hidden_models = df_hidden["model"].unique()
    for m in hidden_models:
        sub = df_hidden[df_hidden["model"] == m].sort_values("iteration")
        color, fmt, lw, label = palette.get(m, ("gray", "o-", 1.8, m))
        ax.plot(sub["iteration"], sub["relative_hidden_drift"] * 100, fmt, color=color, linewidth=lw, markersize=8, label=label)

    ax.set_xlabel("Recurrent Iteration Step")
    ax.set_ylabel("Relative Latent Drift from Pretrained Baseline (%)")
    ax.set_title("Recurrent Dynamics Drift: Hidden State Deviation Under Uniform K=16", pad=12, fontweight="bold")
    ax.set_xticks([4, 8, 12, 14, 16])
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plot4_path = os.path.join(output_dir, "stageb_hidden_state_drift.png")
    plt.tight_layout()
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved: {plot4_path}")

    # Copy to artifacts directory if provided
    if artifact_dir:
        import shutil
        for p in [plot1_path, plot2_path, plot3_path, plot4_path]:
            shutil.copy(p, artifact_dir)
        print(f"Copied all plots to {artifact_dir}")

if __name__ == "__main__":
    plot_stageb(artifact_dir="C:/Users/Minh/.gemini/antigravity-ide/brain/b788712f-4175-4ebd-a0ba-b4c45e1d8582")
