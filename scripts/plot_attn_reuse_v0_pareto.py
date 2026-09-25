"""Plot Attention Reuse V0 Pareto Frontier and Diagnostics."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pareto_and_diagnostics(summary_csv="outputs/attn_reuse_v0_summary.csv", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(summary_csv)

    fixed_14 = df[df["sched_key"] == "fixed_14"].iloc[0]
    fixed_16 = df[df["sched_key"] == "fixed_16"].iloc[0]

    # Create 2-panel figure: Left = Latency, Right = FLOPs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), dpi=150)

    # -------------------------------------------------------------
    # Panel 1: Quality vs Real GPU Recurrent Latency
    # -------------------------------------------------------------
    df_fixed = df[df["sched_type"] == "fixed"].sort_values("latency_ms")
    ax1.plot(
        df_fixed["latency_ms"],
        df_fixed["abs_rel"],
        marker="o",
        markersize=9,
        color="#111111",
        linewidth=2.5,
        linestyle="-",
        label="Fixed Stopping Frontier (i=13..16)",
        zorder=4,
    )
    for _, r in df_fixed.iterrows():
        ax1.annotate(
            r["sched_name"],
            (r["latency_ms"], r["abs_rel"]),
            textcoords="offset points",
            xytext=(8, -4),
            fontweight="bold",
            fontsize=9,
            color="#111111",
        )

    # Variant A
    df_varA = df[df["sched_type"] == "variantA"]
    ax1.scatter(
        df_varA["latency_ms"],
        df_varA["abs_rel"],
        marker="s",
        s=85,
        color="#d62728",
        label="Variant A: Global Skip",
        alpha=0.85,
        zorder=5,
    )
    for _, r in df_varA.iterrows():
        name_short = r["sched_name"].replace("VarA: ", "")
        ax1.annotate(
            name_short,
            (r["latency_ms"], r["abs_rel"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            color="#d62728",
        )

    # Variant B
    df_varB = df[df["sched_type"] == "variantB"]
    ax1.scatter(
        df_varB["latency_ms"],
        df_varB["abs_rel"],
        marker="^",
        s=75,
        color="#1f77b4",
        label="Variant B: Residual Reuse (alpha in {0.5, 0.75, 1.0})",
        alpha=0.85,
        zorder=5,
    )

    # Variant C
    df_varC = df[df["sched_type"] == "variantC"]
    ax1.scatter(
        df_varC["latency_ms"],
        df_varC["abs_rel"],
        marker="D",
        s=70,
        color="#2ca02c",
        label="Variant C: Frozen Global K/V",
        alpha=0.85,
        zorder=5,
    )

    # Reference lines
    ax1.axhline(
        fixed_14["abs_rel"],
        color="#666666",
        linestyle="--",
        linewidth=1.5,
        label=f"Fixed i=14 AbsRel Bar ({fixed_14['abs_rel']:.6f})",
    )
    target_lat_bar = fixed_14["latency_ms"] * 0.95
    ax1.axvline(
        target_lat_bar,
        color="#e377c2",
        linestyle="--",
        linewidth=1.5,
        label=f"Target 5% Speedup Bar ({target_lat_bar:.1f} ms)",
    )
    ax1.axvline(
        fixed_14["latency_ms"],
        color="#8c564b",
        linestyle=":",
        linewidth=1.2,
        label=f"Fixed i=14 Latency ({fixed_14['latency_ms']:.1f} ms)",
    )

    ax1.set_title("Reconstruction Quality vs Real GPU Latency", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Recurrent Latency (ms, RTX 5070 Laptop, bf16)", fontsize=11)
    ax1.set_ylabel("Mean Reconstruction AbsRel (Lower is Better)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 2: Quality vs Theoretical Recurrent FLOPs
    # -------------------------------------------------------------
    df_fixed_f = df[df["sched_type"] == "fixed"].sort_values("recurrent_flops_gflops")
    ax2.plot(
        df_fixed_f["recurrent_flops_gflops"] / 1000.0,
        df_fixed_f["abs_rel"],
        marker="o",
        markersize=9,
        color="#111111",
        linewidth=2.5,
        linestyle="-",
        label="Fixed Stopping Frontier (i=13..16)",
        zorder=4,
    )
    for _, r in df_fixed_f.iterrows():
        ax2.annotate(
            r["sched_name"],
            (r["recurrent_flops_gflops"] / 1000.0, r["abs_rel"]),
            textcoords="offset points",
            xytext=(8, -4),
            fontweight="bold",
            fontsize=9,
            color="#111111",
        )

    ax2.scatter(
        df_varA["recurrent_flops_gflops"] / 1000.0,
        df_varA["abs_rel"],
        marker="s",
        s=85,
        color="#d62728",
        label="Variant A: Global Skip",
        alpha=0.85,
        zorder=5,
    )

    ax2.scatter(
        df_varB["recurrent_flops_gflops"] / 1000.0,
        df_varB["abs_rel"],
        marker="^",
        s=75,
        color="#1f77b4",
        label="Variant B: Residual Reuse",
        alpha=0.85,
        zorder=5,
    )

    ax2.scatter(
        df_varC["recurrent_flops_gflops"] / 1000.0,
        df_varC["abs_rel"],
        marker="D",
        s=70,
        color="#2ca02c",
        label="Variant C: Frozen Global K/V",
        alpha=0.85,
        zorder=5,
    )

    ax2.axhline(
        fixed_14["abs_rel"],
        color="#666666",
        linestyle="--",
        linewidth=1.5,
        label=f"Fixed i=14 AbsRel Bar ({fixed_14['abs_rel']:.6f})",
    )
    target_flops_bar = (fixed_14["recurrent_flops_gflops"] * 0.95) / 1000.0
    ax2.axvline(
        target_flops_bar,
        color="#e377c2",
        linestyle="--",
        linewidth=1.5,
        label=f"Target 5% FLOP Reduction ({target_flops_bar:.2f} TFLOPs)",
    )
    ax2.axvline(
        fixed_14["recurrent_flops_gflops"] / 1000.0,
        color="#8c564b",
        linestyle=":",
        linewidth=1.2,
        label=f"Fixed i=14 FLOPs ({fixed_14['recurrent_flops_gflops']/1000.0:.2f} TFLOPs)",
    )

    ax2.set_title("Reconstruction Quality vs Recurrent Compute (TFLOPs)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Recurrent Compute (TFLOPs)", fontsize=11)
    ax2.set_ylabel("Mean Reconstruction AbsRel (Lower is Better)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    pareto_png = os.path.join(output_dir, "attn_reuse_v0_pareto_frontier.png")
    plt.tight_layout()
    plt.savefig(pareto_png)
    plt.close()
    print(f"Saved Pareto frontier plot: {pareto_png}")


if __name__ == "__main__":
    plot_pareto_and_diagnostics()
