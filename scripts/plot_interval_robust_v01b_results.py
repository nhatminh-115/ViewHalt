"""
INTERVAL-ROBUST V0.1B: Plotting script for Robustness Curve and Non-Uniform Schedule Evaluation.
Generates:
- outputs/interval_robust_v01b_robustness.png
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


def plot_stagea_results(
    robust_csv="outputs/interval_robust_v01b_robustness.csv",
    nonuniform_csv="outputs/interval_robust_v01b_nonuniform.csv",
    out_png="outputs/interval_robust_v01b_robustness.png"
):
    if not os.path.exists(robust_csv) or not os.path.exists(nonuniform_csv):
        print(f"CSVs missing ({robust_csv} or {nonuniform_csv}), skipping plot.")
        return

    df_rob = pd.read_csv(robust_csv)
    df_non = pd.read_csv(nonuniform_csv)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Color palette
    colors = {
        "Pretrained": "#333333",
        "Uniform_FT": "#1f77b4",
        "Mild_Rand_FT": "#2ca02c",
        "Mod_Rand_FT": "#d62728",
    }
    labels = {
        "Pretrained": "Pretrained DVLT",
        "Uniform_FT": "Control 2: Uniform FT",
        "Mild_Rand_FT": "Treatment 1: Mild Rand FT (sigma=0.2)",
        "Mod_Rand_FT": "Treatment 2: Mod Rand FT (sigma=0.5)",
    }
    markers = {
        "Pretrained": "o",
        "Uniform_FT": "s",
        "Mild_Rand_FT": "^",
        "Mod_Rand_FT": "D",
    }

    # Panel A: Robustness Curve (CV vs AbsRel)
    ax1 = axes[0]
    for model_name in ["Pretrained", "Uniform_FT", "Mild_Rand_FT", "Mod_Rand_FT"]:
        sub = df_rob[df_rob["model"] == model_name].sort_values("target_cv")
        ax1.plot(
            sub["actual_cv"],
            sub["abs_rel"] * 1000.0,
            marker=markers[model_name],
            color=colors[model_name],
            label=labels[model_name],
            linewidth=2.2,
            markersize=7,
        )

    pre_k14_hurdle = 0.005713 * 1000.0
    ax1.axhline(pre_k14_hurdle, color="crimson", linestyle="--", alpha=0.7, label="Pretrained K=14 Hurdle (5.71)")
    ax1.set_xlabel("Interval Irregularity CV(Delta_t)")
    ax1.set_ylabel("Depth AbsRel Error (x 10^-3)")
    ax1.set_title("A. Robustness Curve: AbsRel Error vs Interval Irregularity", fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.set_ylim(5.0, 7.5)

    # Panel B: Non-Uniform Schedule Evaluation (Delta vs Uniform FT Control %)
    ax2 = axes[1]
    # Filter out Uniform K12 to focus on non-uniform schedules
    sub_non = df_non[df_non["schedule_key"] != "uniform_K12"]
    sched_names = sub_non["schedule_name"].unique()
    x = np.arange(len(sched_names))
    width = 0.25

    t1_deltas = []
    t2_deltas = []
    pre_deltas = []

    for s_name in sched_names:
        row_t1 = sub_non[(sub_non["schedule_name"] == s_name) & (sub_non["model"] == "Mild_Rand_FT")].iloc[0]
        row_t2 = sub_non[(sub_non["schedule_name"] == s_name) & (sub_non["model"] == "Mod_Rand_FT")].iloc[0]
        row_pre = sub_non[(sub_non["schedule_name"] == s_name) & (sub_non["model"] == "Pretrained")].iloc[0]
        t1_deltas.append(row_t1["delta_vs_ctrl_absrel_pct"])
        t2_deltas.append(row_t2["delta_vs_ctrl_absrel_pct"])
        pre_deltas.append(row_pre["delta_vs_ctrl_absrel_pct"])

    rects1 = ax2.bar(x - width, pre_deltas, width, label="Pretrained", color=colors["Pretrained"], alpha=0.7)
    rects2 = ax2.bar(x, t1_deltas, width, label="Treatment 1 (Mild)", color=colors["Mild_Rand_FT"])
    rects3 = ax2.bar(x + width, t2_deltas, width, label="Treatment 2 (Moderate)", color=colors["Mod_Rand_FT"])

    ax2.axhline(0, color="black", linestyle="-", linewidth=1.2, label="Matched Uniform FT Control Baseline (0%)")
    ax2.set_ylabel("Delta AbsRel vs Matched Uniform FT (%) [Lower is Better]")
    ax2.set_title("B. Non-Uniform Schedules: Treatment Advantage over Uniform FT", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace(" (", "\n(") for n in sched_names], rotation=20, ha="right")
    ax2.legend(loc="upper left")

    plt.suptitle("INTERVAL-ROBUST V0.1B: Stage-A Non-Uniform & Robustness Evaluation", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    plot_stagea_results()
