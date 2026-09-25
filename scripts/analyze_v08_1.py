"""ViewHalt V0.8.1: Correct Matched-Quality Real-Latency & Scene-Cluster Bootstrap Analysis.

Features:
1. Reconciled FLOP calculation with exact token count P = 977 (378x504 resolution).
2. Dynamic selection of matched-quality fixed policy (Fixed AbsRel <= Sparse Oracle AbsRel).
3. Strict scene-cluster bootstrap (10,000 resamples over 14 physical DTU scenes),
   keeping subset_middle and subset_uniform together.
4. In every bootstrap resample:
   - recompute Sparse Oracle mean quality;
   - recompute fixed-policy quality frontier;
   - dynamically choose fastest matching fixed policy;
   - compute real latency headroom against that matched policy.
5. Strict separation of:
   - FLOP saving vs K=16
   - Latency saving vs K=16
   - Matched-quality latency advantage vs best fixed policy
6. Publication plots:
   - outputs/v08_1_matched_quality_pareto_frontier.png
   - outputs/v08_1_scene_cluster_bootstrap_distribution.png
   - outputs/v08_1_efficiency_quantities_comparison.png
"""

import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_analysis_v08_1(output_dir="outputs"):
    v07_csv = os.path.join(output_dir, "v07_raw_results.csv")
    v08_csv = os.path.join(output_dir, "v08_regimes_comparison.csv")
    timing_csv = os.path.join(output_dir, "v08_1_timing_benchmark_raw.csv")
    timing_summary_json = os.path.join(output_dir, "v08_1_timing_benchmark_summary.json")

    assert os.path.exists(v07_csv), f"Missing {v07_csv}"
    assert os.path.exists(v08_csv), f"Missing {v08_csv}"
    assert os.path.exists(timing_csv), f"Missing {timing_csv}"

    df_v07 = pd.read_csv(v07_csv)
    df_v08 = pd.read_csv(v08_csv)
    df_timing = pd.read_csv(timing_csv)
    with open(timing_summary_json, "r") as f:
        timing_summary = json.load(f)

    # -------------------------------------------------------------------------
    # 1. Build Merged View-Level Quality DataFrame
    # -------------------------------------------------------------------------
    # From v08, get Sparse Oracle (Regime D) for 5% and 2%
    df_v08_5 = df_v08[df_v08["tolerance"] == "5pct"].copy()
    df_v08_2 = df_v08[df_v08["tolerance"] == "2pct"].copy()

    # From v07, get fixed step results for each step i in {12, 13, 14, 15, 16}
    fixed_dfs = {}
    for step in [12, 13, 14, 15, 16]:
        fixed_dfs[step] = df_v07[df_v07["step_i"] == step].set_index(["scene", "subset", "view_idx"])["abs_rel"]

    view_records = []
    for _, row in df_v08_5.iterrows():
        sc = row["scene"]
        sub = row["subset"]
        v = row["view_idx"]
        key = (sc, sub, v)

        row_2pct = df_v08_2[(df_v08_2["scene"] == sc) & (df_v08_2["subset"] == sub) & (df_v08_2["view_idx"] == v)].iloc[0]

        rec = {
            "scene": sc,
            "subset": sub,
            "view_idx": v,
            "frame_id": row["frame_id"],
            "oracle_k_star_5pct": row["k_star"],
            "oracle_k_star_2pct": row_2pct["k_star"],
            "sparse_oracle_5pct_abs_rel": row["regime_d_abs_rel"],
            "sparse_oracle_2pct_abs_rel": row_2pct["regime_d_abs_rel"],
            "fixed_16_abs_rel": fixed_dfs[16].loc[key],
            "fixed_15_abs_rel": fixed_dfs[15].loc[key],
            "fixed_14_abs_rel": fixed_dfs[14].loc[key],
            "fixed_13_abs_rel": fixed_dfs[13].loc[key],
            "fixed_12_abs_rel": fixed_dfs[12].loc[key],
        }
        view_records.append(rec)
    df_views = pd.DataFrame(view_records)

    # -------------------------------------------------------------------------
    # 2. Point-Estimate Quality & Dynamic Matched-Policy Selection
    # -------------------------------------------------------------------------
    mean_q_oracle5 = float(df_views["sparse_oracle_5pct_abs_rel"].mean())
    mean_q_oracle2 = float(df_views["sparse_oracle_2pct_abs_rel"].mean())

    fixed_steps = [12, 13, 14, 15, 16]
    mean_q_fixed = {s: float(df_views[f"fixed_{s}_abs_rel"].mean()) for s in fixed_steps}

    print("\n" + "=" * 80)
    print("V0.8.1 POINT ESTIMATE: Quality Frontier across 168 View Instances")
    print(f"Sparse Oracle Tol 5% Mean AbsRel : {mean_q_oracle5:.6f}")
    print(f"Sparse Oracle Tol 2% Mean AbsRel : {mean_q_oracle2:.6f}")
    for s in fixed_steps:
        print(f"Fixed i={s:<2} Mean AbsRel             : {mean_q_fixed[s]:.6f}")

    # Dynamically select fastest fixed policy with AbsRel <= Oracle AbsRel
    def select_matched_policy(q_oracle, q_fixed_dict):
        for s in [12, 13, 14, 15, 16]:
            if q_fixed_dict[s] <= q_oracle:
                return s
        return 16

    point_matched_policy_5pct = select_matched_policy(mean_q_oracle5, mean_q_fixed)
    point_matched_policy_2pct = select_matched_policy(mean_q_oracle2, mean_q_fixed)

    print(f"\nPoint-Estimate Matched Policy for Tol 5%: Fixed i={point_matched_policy_5pct} (AbsRel {mean_q_fixed[point_matched_policy_5pct]:.6f} <= {mean_q_oracle5:.6f})")
    print(f"Point-Estimate Matched Policy for Tol 2%: Fixed i={point_matched_policy_2pct} (AbsRel {mean_q_fixed[point_matched_policy_2pct]:.6f} <= {mean_q_oracle2:.6f})")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 3. Point-Estimate Latency & Three Separated Efficiency Quantities
    # -------------------------------------------------------------------------
    # Sequence-level latency lookup: (scene, subset, policy) -> latency_median_ms
    seq_latencies = {}
    for _, r in df_timing.iterrows():
        seq_latencies[(r["scene"], r["subset"], r["policy"])] = r["latency_median_ms"]

    mean_lat_k16 = float(timing_summary["Fixed_K16"]["mean_latency_ms"])
    med_lat_k16 = float(timing_summary["Fixed_K16"]["median_latency_ms"])

    mean_lat_s5 = float(timing_summary["Sparse_Oracle_5pct"]["mean_latency_ms"])
    med_lat_s5 = float(timing_summary["Sparse_Oracle_5pct"]["median_latency_ms"])

    mean_lat_s2 = float(timing_summary["Sparse_Oracle_2pct"]["mean_latency_ms"])
    med_lat_s2 = float(timing_summary["Sparse_Oracle_2pct"]["median_latency_ms"])

    matched_name_5pct = f"Fixed_i{point_matched_policy_5pct}" if point_matched_policy_5pct < 16 else "Fixed_K16"
    mean_lat_matched5 = float(timing_summary[matched_name_5pct]["mean_latency_ms"])
    med_lat_matched5 = float(timing_summary[matched_name_5pct]["median_latency_ms"])

    # Quantity 1: FLOP saving vs K=16
    flop_sav_5pct = float(timing_summary["Sparse_Oracle_5pct"]["mean_flop_savings_pct"])
    flop_sav_2pct = float(timing_summary["Sparse_Oracle_2pct"]["mean_flop_savings_pct"])

    # Quantity 2: Latency saving vs K=16
    lat_sav_vs_k16_5pct = float(timing_summary["Sparse_Oracle_5pct"]["mean_latency_savings_vs_k16_pct"])
    lat_sav_vs_k16_2pct = float(timing_summary["Sparse_Oracle_2pct"]["mean_latency_savings_vs_k16_pct"])

    # Quantity 3: Matched-quality real latency advantage vs fastest fixed policy
    # Difference in latency
    diff_ms_5pct = mean_lat_matched5 - mean_lat_s5
    headroom_k16_norm_5pct = (diff_ms_5pct / mean_lat_k16) * 100.0
    headroom_matched_norm_5pct = (diff_ms_5pct / mean_lat_matched5) * 100.0
    speedup_matched_5pct = mean_lat_matched5 / mean_lat_s5

    print("\n" + "=" * 80)
    print("V0.8.1 POINT ESTIMATE: Three Separated Efficiency Quantities (Tol 5%)")
    print(f"1. FLOP saving vs K=16                              : {flop_sav_5pct:.2f}%")
    print(f"2. Latency saving vs K=16                           : {lat_sav_vs_k16_5pct:.2f}% (Speedup: {mean_lat_k16/mean_lat_s5:.3f}x)")
    print(f"3. Matched-Quality Latency Advantage vs Fixed i={point_matched_policy_5pct} :")
    print(f"   - Absolute Latency Delta                         : {diff_ms_5pct:+.2f} ms ({mean_lat_matched5:.1f}ms vs {mean_lat_s5:.1f}ms)")
    print(f"   - Headroom (normalized to K=16)                  : {headroom_k16_norm_5pct:+.2f}%")
    print(f"   - Relative Advantage (normalized to Fixed i={point_matched_policy_5pct}) : {headroom_matched_norm_5pct:+.2f}%")
    print(f"   - Real Speedup vs Fixed i={point_matched_policy_5pct}                     : {speedup_matched_5pct:.3f}x")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 4. Strict 10,000 Scene-Cluster Bootstrap
    # -------------------------------------------------------------------------
    # Cluster unit: Physical DTU Scene (14 scenes)
    # Both subset_middle and subset_uniform are sampled together
    unique_scenes = sorted(list(df_views["scene"].unique()))
    n_scenes = len(unique_scenes)
    assert n_scenes == 14, f"Expected 14 physical scenes, got {n_scenes}"

    # Pre-index data by scene to make 10,000 resamples blisteringly fast
    scene_views = {sc: df_views[df_views["scene"] == sc] for sc in unique_scenes}
    scene_timings = {sc: df_timing[df_timing["scene"] == sc] for sc in unique_scenes}

    np.random.seed(42)
    n_boot = 10000

    boot_headrooms_k16_norm = []
    boot_headrooms_matched_norm = []
    boot_speedups = []
    boot_matched_policies = []
    boot_lat_sav_vs_k16 = []

    for _ in range(n_boot):
        # Sample 14 scenes with replacement
        sample_scs = np.random.choice(unique_scenes, size=n_scenes, replace=True)

        # Aggregate quality for this sample
        sample_views = pd.concat([scene_views[sc] for sc in sample_scs], ignore_index=True)
        q_s5 = sample_views["sparse_oracle_5pct_abs_rel"].mean()
        q_fix = {s: sample_views[f"fixed_{s}_abs_rel"].mean() for s in fixed_steps}

        # Dynamically determine matched policy
        m_step = select_matched_policy(q_s5, q_fix)
        boot_matched_policies.append(m_step)
        m_name = f"Fixed_i{m_step}" if m_step < 16 else "Fixed_K16"

        # Aggregate latency for this sample across both subsets
        sample_timing = pd.concat([scene_timings[sc] for sc in sample_scs], ignore_index=True)
        lat_k16 = sample_timing[sample_timing["policy"] == "Fixed_K16"]["latency_median_ms"].mean()
        lat_s5 = sample_timing[sample_timing["policy"] == "Sparse_Oracle_5pct"]["latency_median_ms"].mean()
        lat_matched = sample_timing[sample_timing["policy"] == m_name]["latency_median_ms"].mean()

        headroom_k16 = ((lat_matched - lat_s5) / lat_k16) * 100.0
        headroom_m = ((lat_matched - lat_s5) / lat_matched) * 100.0
        speedup = lat_matched / lat_s5
        saving_k16 = ((lat_k16 - lat_s5) / lat_k16) * 100.0

        boot_headrooms_k16_norm.append(headroom_k16)
        boot_headrooms_matched_norm.append(headroom_m)
        boot_speedups.append(speedup)
        boot_lat_sav_vs_k16.append(saving_k16)

    boot_headrooms_k16_norm = np.array(boot_headrooms_k16_norm)
    boot_headrooms_matched_norm = np.array(boot_headrooms_matched_norm)
    boot_speedups = np.array(boot_speedups)
    boot_matched_policies = np.array(boot_matched_policies)

    # Policy distribution in bootstrap
    policy_counts = pd.Series(boot_matched_policies).value_counts(normalize=True) * 100.0
    pol_dist_str = ", ".join([f"Fixed i={k}: {v:.1f}%" for k, v in sorted(policy_counts.items())])

    boot_mean_headroom = float(np.mean(boot_headrooms_k16_norm))
    ci_lower = float(np.percentile(boot_headrooms_k16_norm, 2.5))
    ci_upper = float(np.percentile(boot_headrooms_k16_norm, 97.5))
    fail_rate = float((boot_headrooms_k16_norm <= 0.0).mean() * 100.0)

    boot_mean_speedup = float(np.mean(boot_speedups))
    ci_speedup_lower = float(np.percentile(boot_speedups, 2.5))
    ci_speedup_upper = float(np.percentile(boot_speedups, 97.5))

    print("\n" + "=" * 80)
    print("V0.8.1 STRICT SCENE-CLUSTER BOOTSTRAP (10,000 RESAMPLES OVER 14 PHYSICAL SCENES)")
    print(f"Dynamically Selected Matched Policy Distribution: {pol_dist_str}")
    print(f"Matched-Quality Latency Headroom Mean           : {boot_mean_headroom:+.2f}%")
    print(f"Matched-Quality Latency Headroom 95% CI         : [{ci_lower:+.2f}%, {ci_upper:+.2f}%]")
    print(f"Fraction of Bootstrap Samples with Headroom <= 0 : {fail_rate:.2f}% (Failure rate)")
    print(f"Speedup vs Matched Policy Mean                  : {boot_mean_speedup:.3f}x [95% CI: {ci_speedup_lower:.3f}x, {ci_speedup_upper:.3f}x]")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 5. Publication Plots
    # -------------------------------------------------------------------------
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "figure.autolayout": True})

    # FIGURE 1: Dynamic Matched-Quality Latency Pareto Frontier
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    
    fixed_plot_steps = [12, 13, 14, 15, 16]
    plot_lats = [timing_summary[f"Fixed_i{s}" if s < 16 else "Fixed_K16"]["mean_latency_ms"] for s in fixed_plot_steps]
    plot_errs = [mean_q_fixed[s] for s in fixed_plot_steps]

    ax.plot(plot_lats, plot_errs, "o-", color="#475569", linewidth=2.5, markersize=8, label="Fixed Stopping Frontier (i=12..16)")
    for s, l, e in zip(fixed_plot_steps, plot_lats, plot_errs):
        ax.annotate(f"Fixed i={s}", (l, e), textcoords="offset points", xytext=(-15, 12), fontweight="bold", color="#1E293B")

    # Sparse Oracle 5%
    ax.scatter([mean_lat_s5], [mean_q_oracle5], color="#DC2626", s=160, zorder=5, marker="*", label=f"Sparse Oracle Tol 5% ({mean_lat_s5:.1f}ms, AbsRel={mean_q_oracle5:.6f})")
    # Sparse Oracle 2%
    ax.scatter([mean_lat_s2], [mean_q_oracle2], color="#D97706", s=120, zorder=5, marker="D", label=f"Sparse Oracle Tol 2% ({mean_lat_s2:.1f}ms, AbsRel={mean_q_oracle2:.6f})")

    # Highlight point-estimate matched policy: Fixed i=14
    ax.axhline(mean_q_oracle5, color="#DC2626", linestyle=":", alpha=0.6, label="Sparse Oracle 5% Quality Level")
    matched_l = timing_summary[matched_name_5pct]["mean_latency_ms"]
    ax.scatter([matched_l], [mean_q_fixed[point_matched_policy_5pct]], facecolors="none", edgecolors="#2563EB", s=250, linewidth=2.5, zorder=6, label=f"Matched Policy: Fixed i={point_matched_policy_5pct}")

    ax.annotate(
        f"Dynamic Matched Baseline: Fixed i={point_matched_policy_5pct}\n"
        f"Real Latency Advantage: {matched_l - mean_lat_s5:+.1f} ms ({headroom_k16_norm_5pct:+.1f}% compute advantage)\n"
        f"Real Speedup: {speedup_matched_5pct:.3f}x over Fixed i={point_matched_policy_5pct}",
        xy=((mean_lat_s5 + matched_l) / 2, mean_q_oracle5),
        xytext=(0, -45),
        textcoords="offset points",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF2F2", edgecolor="#DC2626", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
        fontweight="bold"
    )

    ax.set_xlabel("Measured GPU Recurrent Block Latency (ms) [Full 14 Scans, 28 Seqs]", fontweight="bold")
    ax.set_ylabel("Mean AbsRel Depth Error", fontweight="bold")
    ax.set_title("V0.8.1: Real Latency vs Quality — Dynamic Matched Frontier", fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig1_path = os.path.join(output_dir, "v08_1_matched_quality_pareto_frontier.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # FIGURE 2: 10,000 Scene-Cluster Bootstrap Distribution
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax.hist(boot_headrooms_k16_norm, bins=40, color="#3B82F6", alpha=0.75, edgecolor="black", label=f"10,000 Scene-Cluster Resamples\nMean: {boot_mean_headroom:+.2f}%\n95% CI: [{ci_lower:+.2f}%, {ci_upper:+.2f}%]")
    ax.axvline(0, color="#DC2626", linestyle="--", linewidth=2, label=f"Breakeven (Failure rate: {fail_rate:.2f}%)")
    ax.axvline(ci_lower, color="#1E293B", linestyle=":", linewidth=1.5, label=f"95% CI Lower Bound: {ci_lower:+.2f}%")
    ax.axvline(boot_mean_headroom, color="#1D4ED8", linestyle="-", linewidth=2.5, label=f"Bootstrap Mean: {boot_mean_headroom:+.2f}%")

    ax.set_xlabel("Matched-Quality Real Latency Headroom (% vs K=16)", fontweight="bold")
    ax.set_ylabel("Number of Bootstrap Resamples", fontweight="bold")
    ax.set_title("V0.8.1: Scene-Cluster Bootstrap of Matched Real Latency Headroom", fontweight="bold")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig2_path = os.path.join(output_dir, "v08_1_scene_cluster_bootstrap_distribution.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # FIGURE 3: Separation of the Three Efficiency Quantities
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    quantities = [
        "FLOP Saving\nvs K=16",
        "Measured Latency\nSaving vs K=16",
        f"Matched-Quality Real Latency\nAdvantage vs Fixed i={point_matched_policy_5pct}"
    ]
    values = [flop_sav_5pct, lat_sav_vs_k16_5pct, headroom_k16_norm_5pct]
    bar_colors = ["#60A5FA", "#34D399", "#F87171"]

    bars = ax.bar(quantities, values, color=bar_colors, edgecolor="black", alpha=0.9, width=0.55)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Efficiency Advantage (%)", fontweight="bold")
    ax.set_title("V0.8.1: Strict Separation of Three Efficiency Quantities (Tol 5%)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax.annotate(f"{val:+.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold", fontsize=11)

    fig3_path = os.path.join(output_dir, "v08_1_efficiency_quantities_comparison.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig3_path}")

    # -------------------------------------------------------------------------
    # 6. Save JSON Summary
    # -------------------------------------------------------------------------
    final_summary_v08_1 = {
        "metadata": {
            "version": "V0.8.1",
            "num_physical_scenes": n_scenes,
            "num_sequences": len(df_timing) // len(timing_summary),
            "num_view_instances": len(df_views),
            "runtime_token_count_P": 977,
            "runtime_resolution": [378, 504],
        },
        "point_estimates": {
            "sparse_oracle_5pct": {
                "mean_abs_rel": round(mean_q_oracle5, 6),
                "mean_latency_ms": round(mean_lat_s5, 2),
                "median_latency_ms": round(med_lat_s5, 2),
                "analytical_flops_gflops": round(float(timing_summary["Sparse_Oracle_5pct"]["mean_recurrent_flops_gflops"]), 2),
            },
            "sparse_oracle_2pct": {
                "mean_abs_rel": round(mean_q_oracle2, 6),
                "mean_latency_ms": round(mean_lat_s2, 2),
                "median_latency_ms": round(med_lat_s2, 2),
                "analytical_flops_gflops": round(float(timing_summary["Sparse_Oracle_2pct"]["mean_recurrent_flops_gflops"]), 2),
            },
            "fixed_frontier": {
                s: {
                    "mean_abs_rel": round(mean_q_fixed[s], 6),
                    "mean_latency_ms": round(float(timing_summary[f"Fixed_i{s}" if s < 16 else "Fixed_K16"]["mean_latency_ms"]), 2),
                    "median_latency_ms": round(float(timing_summary[f"Fixed_i{s}" if s < 16 else "Fixed_K16"]["median_latency_ms"]), 2),
                    "analytical_flops_gflops": round(float(timing_summary[f"Fixed_i{s}" if s < 16 else "Fixed_K16"]["mean_recurrent_flops_gflops"]), 2),
                } for s in fixed_steps
            },
            "dynamic_matched_selection_5pct": {
                "matched_policy": f"Fixed_i{point_matched_policy_5pct}",
                "matched_step": point_matched_policy_5pct,
                "matched_abs_rel": round(mean_q_fixed[point_matched_policy_5pct], 6),
                "matched_mean_latency_ms": round(mean_lat_matched5, 2),
            },
            "separated_efficiency_quantities_5pct": {
                "quantity_1_flop_saving_vs_k16_pct": round(flop_sav_5pct, 2),
                "quantity_2_latency_saving_vs_k16_pct": round(lat_sav_vs_k16_5pct, 2),
                "quantity_3_matched_quality_latency_advantage_vs_matched_pct": round(headroom_k16_norm_5pct, 2),
                "quantity_3_relative_advantage_vs_matched_pct": round(headroom_matched_norm_5pct, 2),
                "quantity_3_speedup_vs_matched": round(speedup_matched_5pct, 3),
                "quantity_3_delta_latency_ms": round(diff_ms_5pct, 2),
            }
        },
        "scene_cluster_bootstrap": {
            "num_resamples": n_boot,
            "cluster_level": "physical_scene (14 scenes)",
            "matched_policy_distribution": {int(k): round(float(v), 2) for k, v in policy_counts.items()},
            "matched_latency_headroom_mean_pct": round(boot_mean_headroom, 2),
            "matched_latency_headroom_95ci_lower_pct": round(ci_lower, 2),
            "matched_latency_headroom_95ci_upper_pct": round(ci_upper, 2),
            "bootstrap_failure_rate_pct": round(fail_rate, 2),
            "speedup_vs_matched_mean": round(boot_mean_speedup, 3),
            "speedup_vs_matched_95ci_lower": round(ci_speedup_lower, 3),
            "speedup_vs_matched_95ci_upper": round(ci_speedup_upper, 3),
        }
    }

    out_json = os.path.join(output_dir, "v08_1_analysis_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_summary_v08_1, f, indent=2)
    print(f"Saved: {out_json}")
    return final_summary_v08_1


if __name__ == "__main__":
    run_analysis_v08_1()
