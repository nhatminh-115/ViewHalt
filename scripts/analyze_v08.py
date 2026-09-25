"""ViewHalt V0.8: Analysis, Matched Latency Pareto Frontier & Statistical Validation.

Processes:
1. outputs/v08_regimes_comparison.csv
2. outputs/v08_timing_benchmark_raw.csv & summary.json
3. outputs/v07_analysis_summary.json (to preserve V0.7 theoretical ceiling)

Generates:
- outputs/v08_static_vs_overwrite_reconstruction.png
- outputs/v08_active_view_degradation_analysis.png
- outputs/v08_real_compute_latency_pareto_frontier.png
- outputs/v08_flop_vs_latency_scaling.png
- outputs/v08_analysis_summary.json
"""

import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_v08(output_dir="outputs"):
    regimes_csv = os.path.join(output_dir, "v08_regimes_comparison.csv")
    timing_csv = os.path.join(output_dir, "v08_timing_benchmark_raw.csv")
    timing_summary_json = os.path.join(output_dir, "v08_timing_benchmark_summary.json")

    assert os.path.exists(regimes_csv), f"Missing {regimes_csv}"
    assert os.path.exists(timing_csv), f"Missing {timing_csv}"

    df_reg = pd.read_csv(regimes_csv)
    df_time = pd.read_csv(timing_csv)
    with open(timing_summary_json, "r") as f:
        timing_summary = json.load(f)

    # -------------------------------------------------------------------------
    # 1. Verification of Numerical Match (Regime C vs Regime D)
    # -------------------------------------------------------------------------
    max_cd_diff = float(df_reg["diff_c_vs_d_max_depth"].max())
    print(f"Max absolute depth difference between Regime C and D: {max_cd_diff:.6e}")

    # -------------------------------------------------------------------------
    # 2. Quality Comparison across Regimes (A, B, C/D)
    # -------------------------------------------------------------------------
    # 5% tolerance subset
    df_5pct = df_reg[df_reg["tolerance"] == "5pct"]
    df_2pct = df_reg[df_reg["tolerance"] == "2pct"]

    metrics = ["abs_rel", "rmse", "rot_err", "trans_err"]
    summary_quality = {}

    for tol_name, df_sub in [("5pct", df_5pct), ("2pct", df_2pct)]:
        summary_quality[tol_name] = {
            "fixed16": {
                "abs_rel": float(df_sub["fixed16_abs_rel"].mean()),
                "rmse": float(df_sub["fixed16_rmse"].mean()),
                "rot_err_deg": float(df_sub["fixed16_rot_err"].mean()),
                "trans_err_deg": float(df_sub["fixed16_trans_err"].mean()),
            },
            "fixed15": {
                "abs_rel": float(df_sub["fixed15_abs_rel"].mean()),
            },
            "fixed14": {
                "abs_rel": float(df_sub["fixed14_abs_rel"].mean()),
            },
            "regime_b_overwrite": {
                "abs_rel": float(df_sub["regime_b_abs_rel"].mean()),
                "rmse": float(df_sub["regime_b_rmse"].mean()),
                "rot_err_deg": float(df_sub["regime_b_rot_err"].mean()),
                "trans_err_deg": float(df_sub["regime_b_trans_err"].mean()),
                "active_view_deg_pct": float(df_sub[df_sub["is_active_at_end"] == 1]["regime_b_active_deg_pct"].mean()),
            },
            "regime_c_true_static": {
                "abs_rel": float(df_sub["regime_c_abs_rel"].mean()),
                "rmse": float(df_sub["regime_c_rmse"].mean()),
                "rot_err_deg": float(df_sub["regime_c_rot_err"].mean()),
                "trans_err_deg": float(df_sub["regime_c_trans_err"].mean()),
                "active_view_deg_pct": float(df_sub[df_sub["is_active_at_end"] == 1]["regime_c_active_deg_pct"].mean()),
            },
            "regime_d_sparse": {
                "abs_rel": float(df_sub["regime_d_abs_rel"].mean()),
                "rmse": float(df_sub["regime_d_rmse"].mean()),
                "rot_err_deg": float(df_sub["regime_d_rot_err"].mean()),
                "trans_err_deg": float(df_sub["regime_d_trans_err"].mean()),
            },
            "diff_b_vs_c": {
                "abs_rel_diff": float((df_sub["regime_c_abs_rel"] - df_sub["regime_b_abs_rel"]).mean()),
                "rmse_diff": float((df_sub["regime_c_rmse"] - df_sub["regime_b_rmse"]).mean()),
            }
        }

    # -------------------------------------------------------------------------
    # 3. Real Latency Headroom Analysis
    # -------------------------------------------------------------------------
    # Compare Sparse_Oracle_5pct vs Fixed_i15 (Matched Quality)
    k16_lat = timing_summary["Fixed_K16"]["median_latency_ms"]
    i15_lat = timing_summary["Fixed_i15"]["median_latency_ms"]
    i14_lat = timing_summary["Fixed_i14"]["median_latency_ms"]
    sparse5_lat = timing_summary["Sparse_Oracle_5pct"]["median_latency_ms"]
    sparse2_lat = timing_summary["Sparse_Oracle_2pct"]["median_latency_ms"]

    latency_saving_vs_k16_5pct = timing_summary["Sparse_Oracle_5pct"]["mean_latency_savings_pct"]
    latency_saving_vs_k16_i15 = timing_summary["Fixed_i15"]["mean_latency_savings_pct"]
    real_latency_headroom_5pct = latency_saving_vs_k16_5pct - latency_saving_vs_k16_i15

    # -------------------------------------------------------------------------
    # 4. Bootstrap Real Latency Headroom (10,000 resamples over benchmarked sequences)
    # -------------------------------------------------------------------------
    np.random.seed(42)
    bench_seqs = sorted(list(set(zip(df_time["scene"], df_time["subset"]))))
    n_seqs = len(bench_seqs)

    boot_headrooms = []
    for _ in range(10000):
        sample_seqs = [bench_seqs[idx] for idx in np.random.choice(n_seqs, size=n_seqs, replace=True)]
        # For each sample, compute mean latency savings for sparse5 and i15
        s5_savings = []
        i15_savings = []
        for sc, sub in sample_seqs:
            row_k16 = df_time[(df_time["scene"] == sc) & (df_time["subset"] == sub) & (df_time["policy"] == "Fixed_K16")]
            row_s5 = df_time[(df_time["scene"] == sc) & (df_time["subset"] == sub) & (df_time["policy"] == "Sparse_Oracle_5pct")]
            row_i15 = df_time[(df_time["scene"] == sc) & (df_time["subset"] == sub) & (df_time["policy"] == "Fixed_i15")]
            if len(row_k16) > 0 and len(row_s5) > 0 and len(row_i15) > 0:
                t16 = row_k16["latency_median_ms"].values[0]
                ts5 = row_s5["latency_median_ms"].values[0]
                ti15 = row_i15["latency_median_ms"].values[0]
                s5_savings.append((t16 - ts5) / t16 * 100.0)
                i15_savings.append((t16 - ti15) / t16 * 100.0)
        boot_headrooms.append(np.mean(s5_savings) - np.mean(i15_savings))

    boot_headrooms = np.array(boot_headrooms)
    boot_mean = float(np.mean(boot_headrooms))
    ci_lower = float(np.percentile(boot_headrooms, 2.5))
    ci_upper = float(np.percentile(boot_headrooms, 97.5))
    pct_fail = float((boot_headrooms <= 0.0).mean() * 100.0)

    summary_headroom = {
        "real_latency_headroom_5pct_discrete": round(real_latency_headroom_5pct, 2),
        "bootstrap_mean_headroom_pct": round(boot_mean, 2),
        "bootstrap_95ci_lower_pct": round(ci_lower, 2),
        "bootstrap_95ci_upper_pct": round(ci_upper, 2),
        "bootstrap_failure_rate_pct": round(pct_fail, 2),
    }

    # -------------------------------------------------------------------------
    # 5. Plotting Publication Figures
    # -------------------------------------------------------------------------
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "figure.autolayout": True})

    # FIGURE 1: Static Freeze (Regime C/D) vs Overwrite Freeze (Regime B)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # 5% tol AbsRel scatter
    ax = axes[0]
    ax.scatter(df_5pct["regime_b_abs_rel"], df_5pct["regime_c_abs_rel"], alpha=0.7, color="#2563EB", edgecolors="none", s=40, label="Views (Tol 5%)")
    lims = [min(df_5pct["regime_b_abs_rel"].min(), df_5pct["regime_c_abs_rel"].min()) * 0.95,
            max(df_5pct["regime_b_abs_rel"].max(), df_5pct["regime_c_abs_rel"].max()) * 1.05]
    ax.plot(lims, lims, "r--", alpha=0.8, label="Identity line (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Regime B (Post-Block Overwrite) AbsRel", fontweight="bold")
    ax.set_ylabel("Regime C (True Static Reference) AbsRel", fontweight="bold")
    ax.set_title("Reconstruction Quality: True Static vs Overwrite", fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Active view delta distribution
    ax = axes[1]
    active_b = df_5pct[df_5pct["is_active_at_end"] == 1]["regime_b_active_deg_pct"]
    active_c = df_5pct[df_5pct["is_active_at_end"] == 1]["regime_c_active_deg_pct"]
    data_to_plot = [active_b, active_c]
    bp = ax.boxplot(data_to_plot, labels=["Regime B\n(Overwrite)", "Regime C\n(True Static)"], patch_artist=True)
    colors = ["#94A3B8", "#10B981"]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax.axhline(0, color="red", linestyle="--", alpha=0.8)
    ax.set_ylabel("Active-View Error Delta vs K=16 (%)", fontweight="bold")
    ax.set_title("Active-View Degradation on Non-Halted Views", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig1_path = os.path.join(output_dir, "v08_static_vs_overwrite_reconstruction.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # FIGURE 2: Active-View Degradation Analysis
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.hist(active_c, bins=15, color="#10B981", alpha=0.7, edgecolor="black", label=f"True Static (Mean: {active_c.mean():+.3f}%)")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Degradation Baseline")
    ax.axvline(active_c.mean(), color="#047857", linestyle="-", linewidth=2, label=f"Mean Delta ({active_c.mean():+.3f}%)")
    ax.set_xlabel("Relative Error Change on Active Views (%)", fontweight="bold")
    ax.set_ylabel("Number of Active Views", fontweight="bold")
    ax.set_title("Distribution of Active-View Degradation under True Static Freezing", fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig2_path = os.path.join(output_dir, "v08_active_view_degradation_analysis.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # FIGURE 3: Real Wall-Clock Latency vs AbsRel Pareto Frontier
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    # Fixed policies
    fixed_names = ["Fixed_i8", "Fixed_i12", "Fixed_i14", "Fixed_i15", "Fixed_K16"]
    fixed_steps = [8, 12, 14, 15, 16]
    fixed_lats = [timing_summary[name]["median_latency_ms"] for name in fixed_names]
    fixed_errs = [
        float(df_reg["fixed8_abs_rel"].mean()),
        float(df_reg["fixed12_abs_rel"].mean()),
        float(df_reg["fixed14_abs_rel"].mean()),
        float(df_reg["fixed15_abs_rel"].mean()),
        float(df_reg["fixed16_abs_rel"].mean()),
    ]

    ax.plot(fixed_lats, fixed_errs, "o-", color="#64748B", linewidth=2.5, markersize=8, label="Fixed Stopping Frontier (i=8..16)")
    for s, l, e in zip(fixed_steps, fixed_lats, fixed_errs):
        ax.annotate(f"i={s}", (l, e), textcoords="offset points", xytext=(-15, 10), fontweight="bold", color="#334155")

    # Sparse Oracles
    s5_lat = timing_summary["Sparse_Oracle_5pct"]["median_latency_ms"]
    s5_err = summary_quality["5pct"]["regime_d_sparse"]["abs_rel"]
    ax.scatter([s5_lat], [s5_err], color="#EF4444", s=140, zorder=5, marker="*", label=f"Sparse Oracle Tol 5% ({s5_lat:.1f}ms, AbsRel={s5_err:.6f})")

    s2_lat = timing_summary["Sparse_Oracle_2pct"]["median_latency_ms"]
    s2_err = summary_quality["2pct"]["regime_d_sparse"]["abs_rel"]
    ax.scatter([s2_lat], [s2_err], color="#F59E0B", s=120, zorder=5, marker="D", label=f"Sparse Oracle Tol 2% ({s2_lat:.1f}ms, AbsRel={s2_err:.6f})")

    # Reference Static (unoptimized)
    ref_lat = timing_summary["Ref_Static_5pct"]["median_latency_ms"]
    ax.scatter([ref_lat], [s5_err], color="#8B5CF6", s=100, zorder=4, marker="x", label=f"Ref Static Tol 5% ({ref_lat:.1f}ms, No Q/KV sparsity)")

    # Matched comparison line at s5_err
    ax.axhline(s5_err, color="#EF4444", linestyle=":", alpha=0.5)
    # Matched saving annotation
    i15_l = timing_summary["Fixed_i15"]["median_latency_ms"]
    ax.annotate(
        f"Real Latency Headroom vs Fixed i=15: {i15_l - s5_lat:+.1f} ms\n({real_latency_headroom_5pct:+.1f}% compute advantage)",
        xy=((s5_lat + i15_l)/2, s5_err),
        xytext=(0, -35),
        textcoords="offset points",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF2F2", edgecolor="#EF4444", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1.5),
        fontweight="bold"
    )

    ax.set_xlabel("GPU Recurrent Block Latency per Sequence (ms)", fontweight="bold")
    ax.set_ylabel("Mean AbsRel Depth Error", fontweight="bold")
    ax.set_title("Real Compute Latency vs Quality: Matched Pareto Frontier", fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig3_path = os.path.join(output_dir, "v08_real_compute_latency_pareto_frontier.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig3_path}")

    # FIGURE 4: FLOP vs Latency Scaling
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    pol_labels = ["Fixed i=15", "Fixed i=14", "Fixed i=12", "Sparse Tol 5%", "Sparse Tol 2%"]
    pol_keys = ["Fixed_i15", "Fixed_i14", "Fixed_i12", "Sparse_Oracle_5pct", "Sparse_Oracle_2pct"]
    flop_sav = [timing_summary[k]["mean_flop_savings_pct"] for k in pol_keys]
    lat_sav = [timing_summary[k]["mean_latency_savings_pct"] for k in pol_keys]

    x = np.arange(len(pol_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, flop_sav, width, label="Analytical FLOP Saving (%)", color="#3B82F6", alpha=0.85)
    rects2 = ax.bar(x + width/2, lat_sav, width, label="Measured GPU Latency Saving (%)", color="#10B981", alpha=0.85)

    ax.set_ylabel("Savings vs Fixed K=16 (%)", fontweight="bold")
    ax.set_title("Theoretical FLOP Savings vs Measured GPU Latency Savings", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pol_labels, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    fig4_path = os.path.join(output_dir, "v08_flop_vs_latency_scaling.png")
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig4_path}")

    # Compile final comprehensive summary
    final_summary = {
        "metadata": {
            "version": "V0.8",
            "num_view_instances": len(df_reg) // 2,
            "max_c_vs_d_diff": max_cd_diff,
        },
        "quality_summary": summary_quality,
        "timing_summary": timing_summary,
        "headroom_summary": summary_headroom,
    }

    final_json_path = os.path.join(output_dir, "v08_analysis_summary.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)
    print(f"Saved final analysis summary: {final_json_path}")
    return final_summary


if __name__ == "__main__":
    analyze_v08()
