"""
INTERVAL-ROBUST V0.1: Visualizations for LR Stability and Matched Treatment Comparison.
Generates:
1. outputs/interval_robust_v01_lr_stability.png
2. outputs/interval_robust_v01_matched_comparison.png
"""

import os
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
    "figure.titlesize": 15,
})


def plot_lr_stability(calib_csv="outputs/interval_robust_v01_lr_calibration.csv", out_png="outputs/interval_robust_v01_lr_stability.png"):
    df = pd.read_csv(calib_csv)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    lrs = sorted(df["lr"].unique())
    colors = {1e-6: "#2ca02c", 3e-6: "#1f77b4", 1e-5: "#ff7f0e", 3e-5: "#d62728"}
    markers = {1e-6: "o", 3e-6: "s", 1e-5: "^", 3e-5: "D"}

    # 1. Delta AbsRel (%) vs Steps (K=14)
    ax1 = axes[0, 0]
    for lr in lrs:
        sub = df[df["lr"] == lr].sort_values("step")
        ax1.plot(sub["step"], sub["delta_k14_pct"], marker=markers[lr], color=colors[lr], label=f"LR = {lr:.1e}", linewidth=2, markersize=6)
    ax1.axhline(0, color="black", linestyle="--", alpha=0.6, label="Pretrained Baseline")
    ax1.axhline(5, color="red", linestyle=":", linewidth=1.5, label="5% Calibration Bound")
    ax1.axhline(-5, color="red", linestyle=":", linewidth=1.5)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Delta K=14 AbsRel (%) vs Pretrained")
    ax1.set_title("A. Calibration Stability across Learning Rates (K=14)", fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.set_ylim(-2, 30)

    # 2. Delta AbsRel (%) vs Steps (K=12)
    ax2 = axes[0, 1]
    for lr in lrs:
        sub = df[df["lr"] == lr].sort_values("step")
        ax2.plot(sub["step"], sub["delta_k12_pct"], marker=markers[lr], color=colors[lr], label=f"LR = {lr:.1e}", linewidth=2, markersize=6)
    ax2.axhline(0, color="black", linestyle="--", alpha=0.6, label="Pretrained Baseline")
    ax2.axhline(5, color="red", linestyle=":", linewidth=1.5, label="5% Calibration Bound")
    ax2.axhline(-5, color="red", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Delta K=12 AbsRel (%) vs Pretrained")
    ax2.set_title("B. Calibration Stability across Learning Rates (K=12)", fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.set_ylim(-2, 30)

    # 3. Parameter Drift & Gate Drift Dynamics (Log Scale)
    ax3 = axes[1, 0]
    for lr in lrs:
        sub = df[df["lr"] == lr].sort_values("step")
        ax3.plot(sub["step"], sub["param_drift"], marker=markers[lr], color=colors[lr], label=f"Param Drift (LR={lr:.1e})", linewidth=1.8, markersize=5)
        ax3.plot(sub["step"], sub["gate_drift"], marker=markers[lr], color=colors[lr], linestyle="--", alpha=0.7, label=f"Gate Drift (LR={lr:.1e})", linewidth=1.5)
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Frobenius Drift Distance (log scale)")
    ax3.set_yscale("log")
    ax3.set_title("C. Trainable Parameter & Gate Output Drift Dynamics", fontweight="bold")
    ax3.legend(loc="lower right", ncol=2, fontsize=8)

    # 4. Training Loss & Camera Pose Robustness
    ax4 = axes[1, 1]
    for lr in lrs:
        sub = df[df["lr"] == lr].sort_values("step")
        # filter step > 0 for loss
        sub_loss = sub[sub["step"] > 0]
        ax4.plot(sub_loss["step"], sub_loss["train_loss"], marker=markers[lr], color=colors[lr], label=f"Loss (LR={lr:.1e})", linewidth=1.8)
    ax4.set_xlabel("Training Step")
    ax4.set_ylabel("Unweighted Depth L1 Loss")
    ax4.set_title("D. Optimization Loss Dynamics (Scene-Normalized)", fontweight="bold")
    ax4.legend(loc="upper right")

    plt.suptitle("INTERVAL-ROBUST V0.1: Protocol Repair & Learning Rate Calibration Sweep", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


def plot_matched_comparison(matched_csv="outputs/interval_robust_v01_matched_comparison.csv", out_png="outputs/interval_robust_v01_matched_comparison.png"):
    if not os.path.exists(matched_csv):
        print(f"CSV not found: {matched_csv}, skipping matched plot.")
        return

    df = pd.read_csv(matched_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pretrained reference values
    pre_k12 = 0.005801
    pre_k14 = 0.005713

    # Add Pretrained row to df for visualization
    names = ["Pretrained DVLT"] + list(df["name"])
    k12_vals = [pre_k12] + list(df["k12_absrel"])
    k14_vals = [pre_k14] + list(df["k14_absrel"])
    rot_vals = [0.0] + list(df["k14_rot_deg"])

    x = np.arange(len(names))
    width = 0.35

    # Panel A: AbsRel Depth Error
    ax1 = axes[0]
    rects1 = ax1.bar(x - width/2, [v * 1000 for v in k12_vals], width, label="K=12 AbsRel (x10^-3)", color="#4575b4")
    rects2 = ax1.bar(x + width/2, [v * 1000 for v in k14_vals], width, label="K=14 AbsRel (x10^-3)", color="#74add1")
    ax1.axhline(pre_k14 * 1000, color="crimson", linestyle="--", linewidth=1.5, label="Pretrained K=14 Hurdle")
    ax1.set_ylabel("AbsRel Error (x 10^-3)")
    ax1.set_title("A. Absolute Depth Error Comparison", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace(" (", "\n(") for n in names], rotation=15, ha="right")
    ax1.legend(loc="upper right")
    ax1.set_ylim(5.0, 6.5)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Panel B: Delta vs Pretrained K=14 (%)
    ax2 = axes[1]
    delta_pct = [(v - pre_k14) / pre_k14 * 100 for v in k14_vals]
    bar_colors = ["#2ca02c" if d <= 0 else ("#1f77b4" if d <= 5.0 else "#d62728") for d in delta_pct]
    rects3 = ax2.bar(x, delta_pct, width=0.5, color=bar_colors)
    ax2.axhline(0, color="black", linestyle="-", linewidth=1)
    ax2.axhline(5, color="red", linestyle=":", linewidth=1.5, label="5% Calibration Bound")
    ax2.set_ylabel("Delta AbsRel vs Pretrained K=14 (%)")
    ax2.set_title("B. Relative Degradation vs Pretrained K=14 (%)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace(" (", "\n(") for n in names], rotation=15, ha="right")
    ax2.legend(loc="upper right")

    for rect in rects3:
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        ax2.annotate(f"{h:+.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3 if h >= 0 else -10), textcoords="offset points", ha="center", va=va, fontsize=9, fontweight="bold")

    plt.suptitle("INTERVAL-ROBUST V0.1: Matched Treatment Comparison (LR=1e-06, 200 Steps)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    plot_lr_stability()
    plot_matched_comparison()
