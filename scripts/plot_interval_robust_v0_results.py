"""Generates publication-quality figures for DVLT Interval-Robust Training V0:
1. outputs/interval_robust_v0_robustness_curve.png: Schedule Irregularity CV(Delta_t) vs Reconstruction Error
2. outputs/interval_robust_v0_quality_latency.png: Reconstruction Quality vs Real GPU Recurrent Latency
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_robustness_curves(robustness_csv="outputs/interval_robust_v0_robustness_curve.csv", output_dir="outputs"):
    if not os.path.exists(robustness_csv):
        print(f"File not found: {robustness_csv}")
        return

    df = pd.read_csv(robustness_csv)

    # Style setup
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    model_styles = {
        "control1_pretrained": {"label": "Control 1: Pretrained DVLT", "color": "#1f77b4", "marker": "o", "ls": "--", "lw": 2.0},
        "control2_uniform_ft": {"label": "Control 2: Uniform FT (Delta_t = const)", "color": "#2ca02c", "marker": "s", "ls": "-", "lw": 2.0},
        "treatment1_mild_ft":  {"label": "Treatment 1: Mild Rand FT (sigma=0.2)", "color": "#ff7f0e", "marker": "^", "ls": "-", "lw": 2.2},
        "treatment2_mod_ft":   {"label": "Treatment 2: Moderate Rand FT (sigma=0.5)", "color": "#d62728", "marker": "D", "ls": "-", "lw": 2.2},
    }

    # Left Panel: AbsRel vs CV(Delta t)
    ax0 = axes[0]
    for model_key, style in model_styles.items():
        sub = df[df["model_key"] == model_key].sort_values("actual_cv")
        if len(sub) == 0:
            continue
        ax0.plot(sub["actual_cv"], sub["abs_rel"] * 100.0, label=style["label"],
                 color=style["color"], marker=style["marker"], linestyle=style["ls"],
                 linewidth=style["lw"], markersize=7)

    ax0.set_title("Schedule Irregularity vs AbsRel Error (K=12)", fontsize=13, fontweight="bold", pad=10)
    ax0.set_xlabel(r"Interval Irregularity $\mathrm{CV}(\Delta t) = \sigma_{\Delta t} / \mu_{\Delta t}$", fontsize=11)
    ax0.set_ylabel("AbsRel Error (%) [Lower is Better]", fontsize=11)
    ax0.grid(True, linestyle="--", alpha=0.6)
    ax0.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)

    # Right Panel: RMSE vs CV(Delta t)
    ax1 = axes[1]
    for model_key, style in model_styles.items():
        sub = df[df["model_key"] == model_key].sort_values("actual_cv")
        if len(sub) == 0:
            continue
        ax1.plot(sub["actual_cv"], sub["rmse"], label=style["label"],
                 color=style["color"], marker=style["marker"], linestyle=style["ls"],
                 linewidth=style["lw"], markersize=7)

    ax1.set_title("Schedule Irregularity vs RMSE (K=12)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel(r"Interval Irregularity $\mathrm{CV}(\Delta t) = \sigma_{\Delta t} / \mu_{\Delta t}$", fontsize=11)
    ax1.set_ylabel("RMSE [Lower is Better]", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "interval_robust_v0_robustness_curve.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved robustness curve plot: {out_path}")


def plot_quality_latency_pareto(summary_csv="outputs/interval_robust_v0_summary.csv", output_dir="outputs"):
    if not os.path.exists(summary_csv):
        print(f"File not found: {summary_csv}")
        return

    df = pd.read_csv(summary_csv)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    # Find hurdle value: Pretrained Uniform K=14
    hurdle_sub = df[(df["model_key"] == "control1_pretrained") & (df["sched_key"] == "uniform_K14")]
    if len(hurdle_sub) > 0:
        hurdle_absrel = hurdle_sub["abs_rel"].iloc[0] * 100.0
        hurdle_lat = hurdle_sub["recurrent_latency_ms"].iloc[0]
        ax.axhline(hurdle_absrel, color="#9467bd", linestyle=":", linewidth=1.8,
                   label=f"Hurdle: Pretrained Uniform K=14 ({hurdle_absrel:.3f}%)")

    # Plot baseline pretrained curve
    pre_u = df[(df["model_key"] == "control1_pretrained") & (df["family"] == "baseline_uniform")].sort_values("K")
    if len(pre_u) > 0:
        ax.plot(pre_u["recurrent_latency_ms"], pre_u["abs_rel"] * 100.0,
                color="#1f77b4", linestyle="--", marker="o", linewidth=2.0, markersize=8,
                label="Control 1: Pretrained Uniform Baseline")
        for _, r in pre_u.iterrows():
            ax.annotate(f"Pretr K={int(r['K'])}",
                        (r["recurrent_latency_ms"], r["abs_rel"] * 100.0),
                        textcoords="offset points", xytext=(8, -4), fontsize=8.5, color="#1f77b4")

    # Models & markers
    model_meta = {
        "control2_uniform_ft": {"name": "Control 2 (Uniform FT)", "color": "#2ca02c", "marker": "s"},
        "treatment1_mild_ft":  {"name": "Treatment 1 (Mild Rand FT)", "color": "#ff7f0e", "marker": "^"},
        "treatment2_mod_ft":   {"name": "Treatment 2 (Moderate Rand FT)", "color": "#d62728", "marker": "D"},
    }

    for model_key, meta in model_meta.items():
        sub_m = df[df["model_key"] == model_key]
        
        # Plot K=12 uniform
        u12 = sub_m[sub_m["sched_key"] == "uniform_K12"]
        if len(u12) > 0:
            ax.scatter(u12["recurrent_latency_ms"], u12["abs_rel"] * 100.0,
                       color=meta["color"], marker=meta["marker"], s=100, edgecolor="black", linewidth=1.2,
                       label=f"{meta['name']} (K=12 Uniform)")
            ax.annotate(f"{meta['name'].split()[0]} K=12 Unif",
                        (u12["recurrent_latency_ms"].iloc[0], u12["abs_rel"].iloc[0] * 100.0),
                        textcoords="offset points", xytext=(8, 4), fontsize=8.5, color=meta["color"], fontweight="bold")

        # Plot candidate non-uniform points for K=12
        non_u12 = sub_m[(sub_m["K"] == 12) & (sub_m["family"] != "baseline_uniform")]
        if len(non_u12) > 0:
            ax.scatter(non_u12["recurrent_latency_ms"], non_u12["abs_rel"] * 100.0,
                       color=meta["color"], marker=meta["marker"], s=45, alpha=0.6,
                       edgecolor=meta["color"])

    ax.set_title("DVLT Interval-Robust Training V0: Quality vs Recurrent Latency", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Real GPU Recurrent Latency (ms) [RTX 5070 Laptop, Lower is Better]", fontsize=11)
    ax.set_ylabel("AbsRel Reconstruction Error (%) [Lower is Better]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "interval_robust_v0_quality_latency.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved quality-latency plot: {out_path}")


def main():
    os.makedirs("outputs", exist_ok=True)
    plot_robustness_curves()
    plot_quality_latency_pareto()


if __name__ == "__main__":
    main()
