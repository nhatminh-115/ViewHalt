"""Generate visualizations for DVLT Non-Uniform Schedule V0 Kill-Test.

Produces:
1. outputs/nonuniform_v0_time_grid_vs_quality.png (Time partitions and schedule trajectories)
2. outputs/nonuniform_v0_pareto_frontier.png (Quality vs Real Latency & FLOPs)
3. outputs/nonuniform_v0_interval_importance.png (Interval diagnostics vs continuous time t)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_nonuniform_results(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, "nonuniform_v0_summary.csv")
    diag_csv = os.path.join(output_dir, "nonuniform_v0_interval_diagnostics.csv")
    sched_json = os.path.join(output_dir, "nonuniform_v0_schedule_definitions.json")

    df = pd.read_csv(summary_csv)
    with open(sched_json, "r", encoding="utf-8") as f:
        schedules = json.load(f)

    u14 = df[df["sched_key"] == "uniform_K14"].iloc[0]
    u16 = df[df["sched_key"] == "uniform_K16"].iloc[0]

    # =========================================================================
    # Figure 1: Time Grid Shapes and Trajectories (Family comparison for K=12)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)

    # 1.1 Time grid curves for K=12
    ax = axes[0]
    k_norm = np.linspace(0, 1, 100)
    ax.plot(k_norm, k_norm, "k--", lw=2, label="Uniform (Linear)")

    # Power
    ax.plot(k_norm, k_norm**0.5, color="#1f77b4", label="Power γ=0.5 (Late-Dense)")
    ax.plot(k_norm, k_norm**2.0, color="#aec7e8", label="Power γ=2.0 (Early-Dense)")
    # Cosine
    ax.plot(k_norm, 1.0 - np.cos(k_norm * np.pi / 2), color="#2ca02c", label="Cosine Late-Dense")
    ax.plot(k_norm, np.sin(k_norm * np.pi / 2), color="#98df8a", label="Cosine Early-Dense")
    ax.plot(k_norm, 0.5 * (1.0 - np.cos(k_norm * np.pi)), color="#d62728", label="Cosine Endpoints-Dense")

    ax.set_title("Time Partition Curves t_k vs Step Ratio", fontsize=12, fontweight="bold")
    ax.set_xlabel("Normalized Step Index k / (K - 1)")
    ax.set_ylabel("Continuous Time t ∈ [0, 1]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=8.5)

    # 1.2 AbsRel vs Gamma for Power Schedules (K=8, 10, 12)
    ax = axes[1]
    for K, col, marker in [(8, "#7f7f7f", "o"), (10, "#ff7f0e", "s"), (12, "#1f77b4", "^")]:
        df_p = df[(df["family"] == "power") & (df["K"] == K)].sort_values("gamma", inplace=False) if "gamma" in df.columns else None
        if df_p is not None and len(df_p) > 0:
            gammas = [schedules[k]["gamma"] for k in df_p["sched_key"]]
            ax.plot(gammas, df_p["abs_rel"], marker=marker, color=col, lw=2, label=f"Power K={K}")
            # Add uniform baseline horizontal dashed line
            u_val = df[df["sched_key"] == f"uniform_K{K}"]["abs_rel"].values[0]
            ax.axhline(u_val, color=col, linestyle=":", alpha=0.7)

    ax.axhline(u14["abs_rel"], color="#d62728", linestyle="--", lw=1.5, label=f"Uniform K=14 Hurdle ({u14['abs_rel']:.6f})")
    ax.set_title("Quality vs Power Exponent γ", fontsize=12, fontweight="bold")
    ax.set_xlabel("Power Exponent γ (<1: Late-Dense, >1: Early-Dense)")
    ax.set_ylabel("Reconstruction AbsRel")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5)

    # 1.3 Best Schedule per Family vs Uniform Baselines
    ax = axes[2]
    families = ["baseline", "power", "cosine", "piecewise", "random"]
    fam_names = ["Uniform Baseline", "Power", "Cosine", "Piecewise", "Random"]
    colors = ["#111111", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    for K, offset, marker in [(8, -0.2, "o"), (10, 0.0, "s"), (12, 0.2, "^")]:
        vals = []
        for fam in families:
            sub = df[(df["family"] == fam) & (df["K"] == K)]
            vals.append(sub["abs_rel"].min() if len(sub) > 0 else np.nan)
        ax.bar(np.arange(len(families)) + offset, vals, width=0.18, label=f"Best K={K}", alpha=0.85)

    ax.axhline(u14["abs_rel"], color="#d62728", linestyle="--", lw=1.5, label=f"Uniform K=14 ({u14['abs_rel']:.6f})")
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(fam_names, rotation=20, ha="right", fontsize=9)
    ax.set_title("Best AbsRel per Family by Step Count K", fontsize=12, fontweight="bold")
    ax.set_ylabel("Reconstruction AbsRel (Lower is Better)")
    ax.set_ylim(0.005, 0.0075)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "nonuniform_v0_time_grid_vs_quality.png")
    plt.savefig(plot1_path)
    plt.close()
    print(f"Saved: {plot1_path}")

    # =========================================================================
    # Figure 2: Quality-vs-Latency & FLOPs Pareto Frontier
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), dpi=150)

    # Panel 1: Quality vs Real GPU Recurrent Latency
    df_u = df[df["family"] == "baseline"].sort_values("latency_ms")
    ax1.plot(
        df_u["latency_ms"],
        df_u["abs_rel"],
        marker="o",
        markersize=9,
        color="#111111",
        lw=2.5,
        linestyle="-",
        label="Uniform Baseline Frontier (K=8..16)",
        zorder=4,
    )
    for _, r in df_u.iterrows():
        ax1.annotate(
            f"K={int(r['K'])}",
            (r["latency_ms"], r["abs_rel"]),
            textcoords="offset points",
            xytext=(7, -4),
            fontweight="bold",
            fontsize=9,
        )

    # Scatter points by family
    fam_configs = [
        ("power", "Power Schedules", "#1f77b4", "s", 65),
        ("cosine", "Cosine Schedules", "#2ca02c", "D", 65),
        ("piecewise", "Piecewise Schedules", "#d62728", "^", 70),
        ("random", "Random Monotonic", "#9467bd", "v", 60),
    ]

    for fam_key, fam_label, color, marker, size in fam_configs:
        sub = df[df["family"] == fam_key]
        ax1.scatter(
            sub["latency_ms"],
            sub["abs_rel"],
            marker=marker,
            s=size,
            color=color,
            label=fam_label,
            alpha=0.75,
            zorder=3,
        )

    # Hurdle line: Uniform K=14
    ax1.axhline(
        u14["abs_rel"],
        color="#d62728",
        linestyle="--",
        lw=1.8,
        label=f"Uniform K=14 Quality Hurdle ({u14['abs_rel']:.6f})",
    )
    ax1.axvline(
        u14["latency_ms"],
        color="#7f7f7f",
        linestyle=":",
        lw=1.5,
        label=f"Uniform K=14 Latency ({u14['latency_ms']:.1f} ms)",
    )

    # GO Zone highlight (Latency < Uniform K=14 and AbsRel <= Uniform K=14)
    x_min, x_max = ax1.get_xlim()
    y_min, y_max = ax1.get_ylim()
    ax1.axvspan(
        x_min,
        u14["latency_ms"] * 0.95,
        ymin=0,
        ymax=(u14["abs_rel"] - y_min) / max(y_max - y_min, 1e-6),
        color="#2ca02c",
        alpha=0.08,
        label="GO Zone (K<=12, AbsRel <= K=14, Speedup >= 5%)",
    )

    ax1.set_title("Reconstruction Quality vs Real GPU Recurrent Latency", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Recurrent Latency (ms, RTX 5070 Laptop, bf16)", fontsize=11)
    ax1.set_ylabel("Mean Reconstruction AbsRel (Lower is Better)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # Panel 2: Quality vs Recurrent Compute (TFLOPs)
    ax2.plot(
        df_u["recurrent_flops_gflops"] / 1000.0,
        df_u["abs_rel"],
        marker="o",
        markersize=9,
        color="#111111",
        lw=2.5,
        linestyle="-",
        label="Uniform Baseline Frontier",
        zorder=4,
    )
    for _, r in df_u.iterrows():
        ax2.annotate(
            f"K={int(r['K'])}",
            (r["recurrent_flops_gflops"] / 1000.0, r["abs_rel"]),
            textcoords="offset points",
            xytext=(7, -4),
            fontweight="bold",
            fontsize=9,
        )

    for fam_key, fam_label, color, marker, size in fam_configs:
        sub = df[df["family"] == fam_key]
        ax2.scatter(
            sub["recurrent_flops_gflops"] / 1000.0,
            sub["abs_rel"],
            marker=marker,
            s=size,
            color=color,
            label=fam_label,
            alpha=0.75,
            zorder=3,
        )

    ax2.axhline(
        u14["abs_rel"],
        color="#d62728",
        linestyle="--",
        lw=1.8,
        label=f"Uniform K=14 Quality Hurdle ({u14['abs_rel']:.6f})",
    )
    ax2.axvline(
        u14["recurrent_flops_gflops"] / 1000.0,
        color="#7f7f7f",
        linestyle=":",
        lw=1.5,
        label=f"Uniform K=14 FLOPs ({u14['recurrent_flops_gflops']/1000.0:.2f} TF)",
    )

    ax2.set_title("Reconstruction Quality vs Recurrent Compute (TFLOPs)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Recurrent Compute (TFLOPs)", fontsize=11)
    ax2.set_ylabel("Mean Reconstruction AbsRel (Lower is Better)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "nonuniform_v0_pareto_frontier.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"Saved: {plot2_path}")

    # =========================================================================
    # Figure 3: Interval Importance Diagnostics vs Continuous Time t
    # =========================================================================
    if os.path.exists(diag_csv):
        df_d = pd.read_csv(diag_csv)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)

        # 3.1 Frame Residual vs t_mid
        ax = axes[0]
        ax.scatter(df_d["t_mid"], df_d["frame_residual_norm"], alpha=0.25, color="#1f77b4", s=15)
        # Binned average
        df_d["t_bin"] = pd.cut(df_d["t_mid"], bins=np.linspace(0, 1, 11))
        bin_mean = df_d.groupby("t_bin", observed=False)["frame_residual_norm"].mean()
        bin_mids = [b.mid for b in bin_mean.index]
        ax.plot(bin_mids, bin_mean.values, color="#111111", lw=2.5, marker="o", label="Binned Mean")
        ax.set_title("Frame-Attention Residual vs Continuous Time", fontsize=12, fontweight="bold")
        ax.set_xlabel("Continuous Time Interval Midpoint t_mid")
        ax.set_ylabel("Frame Residual Norm ||x_frame - x_k||")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

        # 3.2 Global Residual vs t_mid
        ax = axes[1]
        ax.scatter(df_d["t_mid"], df_d["global_residual_norm"], alpha=0.25, color="#2ca02c", s=15)
        bin_mean_g = df_d.groupby("t_bin", observed=False)["global_residual_norm"].mean()
        ax.plot(bin_mids, bin_mean_g.values, color="#111111", lw=2.5, marker="o", label="Binned Mean")
        ax.set_title("Global Cross-View Residual vs Continuous Time", fontsize=12, fontweight="bold")
        ax.set_xlabel("Continuous Time Interval Midpoint t_mid")
        ax.set_ylabel("Global Residual Norm ||x_global - x_frame||")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

        # 3.3 Hidden State Change vs t_mid
        ax = axes[2]
        ax.scatter(df_d["t_mid"], df_d["hidden_change_norm"], alpha=0.25, color="#d62728", s=15)
        bin_mean_h = df_d.groupby("t_bin", observed=False)["hidden_change_norm"].mean()
        ax.plot(bin_mids, bin_mean_h.values, color="#111111", lw=2.5, marker="o", label="Binned Mean")
        ax.set_title("Total Hidden-State Update vs Continuous Time", fontsize=12, fontweight="bold")
        ax.set_xlabel("Continuous Time Interval Midpoint t_mid")
        ax.set_ylabel("Hidden Update Norm ||x_{k+1} - x_k||")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

        plt.tight_layout()
        plot3_path = os.path.join(output_dir, "nonuniform_v0_interval_importance.png")
        plt.savefig(plot3_path)
        plt.close()
        print(f"Saved: {plot3_path}")


if __name__ == "__main__":
    plot_nonuniform_results()
