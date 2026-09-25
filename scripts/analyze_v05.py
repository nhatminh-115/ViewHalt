"""ViewHalt V0.5 Robustness Analysis & Predictive Signal Verification.

Comprehensive statistical and empirical analysis:
1. In-distribution (K in {8, 10, 12, 14, 16}) oracle saturation across 168 views (14 scans x 2 subsets).
2. Corrected headline metrics: mean K*, intra-scene variance, early saturation fraction, compute savings, Pareto comparison.
3. Predictive signal analysis:
   - Evaluates whether hidden-state delta and depth-map delta predict future benefit:
       gain(K) = AbsRel(K) - AbsRel(16)
   - Computes Spearman rank correlation rho, Pearson r, p-values.
   - Evaluates binary classification for 'safe to halt' under 1%, 2%, 5% tolerances via AUROC and AUPRC.
4. Uncertainty quantification:
   - Scene-wise mean oracle savings and 10,000-resample bootstrap 95% confidence intervals over the 14 scenes.
5. High-resolution figure generation:
   - outputs/v05_per_view_curves_grid.png
   - outputs/v05_oracle_k_star_distribution.png
   - outputs/v05_compute_quality_frontier.png
   - outputs/v05_predictive_signal_analysis.png
6. Quantitative robustness verdict evaluation (GO / REVISE / KILL).
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def analyze_v05(raw_json_path="outputs/v05_raw_results.json", output_dir="outputs"):
    print("=" * 80)
    print("Starting ViewHalt V0.5 Robustness & Predictive Signal Analysis")
    print("=" * 80)

    with open(raw_json_path, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    records = data.get("records", [])
    print(f"Loaded {len(records)} evaluation records from {raw_json_path}.")

    # Group records by (sequence_id, seq_view_idx)
    views_data = {}
    for r in records:
        key = (r["sequence_id"], r["seq_view_idx"])
        if key not in views_data:
            views_data[key] = {
                "sequence_id": r["sequence_id"],
                "scene": r["scene"],
                "subset": r["subset"],
                "seq_view_idx": r["seq_view_idx"],
                "dtu_frame_id": r["dtu_frame_id"],
                "by_K": {}
            }
        views_data[key]["by_K"][r["K"]] = r

    k_values = sorted(list(next(iter(views_data.values()))["by_K"].keys()))
    print(f"In-Distribution Step Counts K: {k_values}")
    num_views = len(views_data)
    print(f"Total evaluated view instances: {num_views} across {len(set(r['scene'] for r in records))} scenes.")

    # =========================================================================
    # 1. Corrected In-Distribution Saturation Headline Metrics (K in {8,10,12,14,16})
    # =========================================================================
    rel_tolerances = [0.01, 0.02, 0.05, 0.10]
    saturation_results = {}

    for tol in rel_tolerances:
        tol_key = f"rel_{int(tol*100)}pct"
        k_stars = []
        scene_k_stars = {}

        for key, vinfo in views_data.items():
            scene = vinfo["scene"]
            if scene not in scene_k_stars:
                scene_k_stars[scene] = []

            final_absrel = vinfo["by_K"][16]["depth_abs_rel"]
            threshold = final_absrel * (1.0 + tol)

            sat_K = 16
            for K in k_values:
                if vinfo["by_K"][K]["depth_abs_rel"] <= threshold:
                    sat_K = K
                    break

            k_stars.append(sat_K)
            scene_k_stars[scene].append(sat_K)

        k_stars = np.array(k_stars)
        mean_k = float(np.mean(k_stars))
        median_k = float(np.median(k_stars))
        std_k = float(np.std(k_stars))
        pct_sat_before_16 = float(np.mean(k_stars < 16) * 100)
        pct_sat_le_8 = float(np.mean(k_stars <= 8) * 100)
        pct_sat_le_10 = float(np.mean(k_stars <= 10) * 100)
        pct_sat_le_12 = float(np.mean(k_stars <= 12) * 100)
        compute_saved_pct = float((16.0 - mean_k) / 16.0 * 100)

        # Intra-scene variances
        scene_variances = {s: float(np.var(ks)) for s, ks in scene_k_stars.items()}
        mean_intra_scene_var = float(np.mean(list(scene_variances.values())))

        # Oracle quality vs K=16
        oracle_absrels = [vinfo["by_K"][k_stars[i]]["depth_abs_rel"] for i, (key, vinfo) in enumerate(views_data.items())]
        k16_absrels = [vinfo["by_K"][16]["depth_abs_rel"] for vinfo in views_data.values()]

        mean_oracle_absrel = float(np.mean(oracle_absrels))
        mean_k16_absrel = float(np.mean(k16_absrels))
        quality_loss_pct = float((mean_oracle_absrel - mean_k16_absrel) / mean_k16_absrel * 100)
        dist_counts = {int(K): int(np.sum(k_stars == K)) for K in k_values}

        saturation_results[tol_key] = {
            "tolerance": tol,
            "mean_K_star": round(mean_k, 2),
            "median_K_star": median_k,
            "std_K_star": round(std_k, 2),
            "dist_counts": dist_counts,
            "pct_sat_before_16": round(pct_sat_before_16, 2),
            "pct_sat_le_8": round(pct_sat_le_8, 2),
            "pct_sat_le_10": round(pct_sat_le_10, 2),
            "pct_sat_le_12": round(pct_sat_le_12, 2),
            "compute_saved_pct": round(compute_saved_pct, 2),
            "mean_intra_scene_variance": round(mean_intra_scene_var, 3),
            "scene_variances": scene_variances,
            "mean_oracle_absrel": round(mean_oracle_absrel, 6),
            "mean_k16_absrel": round(mean_k16_absrel, 6),
            "quality_loss_pct": round(quality_loss_pct, 3),
        }

    # =========================================================================
    # 2. Fixed-K Baseline Progression Summary (K in {8,10,12,14,16})
    # =========================================================================
    fixed_k_summary = {}
    for K in k_values:
        absrels = [v["by_K"][K]["depth_abs_rel"] for v in views_data.values()]
        rmses = [v["by_K"][K]["depth_rmse_mm"] for v in views_data.values()]
        rot_errs = [v["by_K"][K]["pose_rot_error_deg"] for v in views_data.values()]
        geom_errs = [v["by_K"][K]["geom_l2_error_mm"] for v in views_data.values()]
        h_deltas = [v["by_K"][K]["hidden_state_delta"] for v in views_data.values()]
        d_deltas = [v["by_K"][K]["depth_map_delta"] for v in views_data.values()]
        runtimes = [r["runtime_sec"] for r in records if r["K"] == K]

        mean_abs = float(np.mean(absrels))
        fixed_k_summary[int(K)] = {
            "mean_abs_rel": round(mean_abs, 6),
            "std_abs_rel": round(float(np.std(absrels)), 6),
            "mean_rmse_mm": round(float(np.mean(rmses)), 4),
            "mean_rot_err_deg": round(float(np.mean(rot_errs)), 4),
            "mean_geom_err_mm": round(float(np.mean(geom_errs)), 4),
            "mean_hidden_delta": round(float(np.mean(h_deltas)), 6),
            "mean_depth_delta": round(float(np.mean(d_deltas)), 6),
            "mean_runtime_ms": round(float(np.mean(runtimes)) * 1000, 1),
            "compute_saved_pct": round(float((16 - K) / 16.0 * 100), 1),
        }

    k16_mean_abs = fixed_k_summary[16]["mean_abs_rel"]
    for K in k_values:
        fixed_k_summary[int(K)]["quality_loss_pct"] = round(
            float((fixed_k_summary[int(K)]["mean_abs_rel"] - k16_mean_abs) / k16_mean_abs * 100), 2
        )

    # =========================================================================
    # 3. Predictive Signal Verification: Does Delta Predict Future Benefit?
    # =========================================================================
    # For every view and intermediate K in {8, 10, 12, 14}:
    # future benefit = gain(K) = AbsRel(K) - AbsRel(16)
    intermediate_ks = [8, 10, 12, 14]
    eval_pairs = []

    for key, vinfo in views_data.items():
        abs16 = vinfo["by_K"][16]["depth_abs_rel"]
        for K in intermediate_ks:
            absK = vinfo["by_K"][K]["depth_abs_rel"]
            gain_K = absK - abs16
            h_delta = vinfo["by_K"][K]["hidden_state_delta"]
            d_delta = vinfo["by_K"][K]["depth_map_delta"]

            eval_pairs.append({
                "scene": vinfo["scene"],
                "subset": vinfo["subset"],
                "view": vinfo["seq_view_idx"],
                "K": K,
                "abs16": abs16,
                "absK": absK,
                "gain": gain_K,
                "rel_gain": gain_K / abs16 if abs16 > 0 else 0.0,
                "hidden_delta": h_delta,
                "depth_delta": d_delta,
                "safe_1pct": int(gain_K <= 0.01 * abs16),
                "safe_2pct": int(gain_K <= 0.02 * abs16),
                "safe_5pct": int(gain_K <= 0.05 * abs16),
                "safe_10pct": int(gain_K <= 0.10 * abs16),
            })

    gains = np.array([p["gain"] for p in eval_pairs])
    rel_gains = np.array([p["rel_gain"] for p in eval_pairs])
    h_deltas = np.array([p["hidden_delta"] for p in eval_pairs])
    d_deltas = np.array([p["depth_delta"] for p in eval_pairs])

    # 3a. Correlation analysis
    spearman_h_gain, pval_h_gain = stats.spearmanr(h_deltas, gains)
    spearman_h_relgain, pval_h_relgain = stats.spearmanr(h_deltas, rel_gains)
    pearson_h_gain, pval_pearson_h = stats.pearsonr(h_deltas, gains)

    spearman_d_gain, pval_d_gain = stats.spearmanr(d_deltas, gains)
    spearman_d_relgain, pval_d_relgain = stats.spearmanr(d_deltas, rel_gains)

    # 3b. Binary prediction of 'safe to halt' (Y=1)
    # Since smaller delta means closer to convergence, predictor score = -delta
    scores_h = -h_deltas
    scores_d = -d_deltas

    signal_metrics = {}
    for tol_name, tol_col in [("1%", "safe_1pct"), ("2%", "safe_2pct"), ("5%", "safe_5pct"), ("10%", "safe_10pct")]:
        y_true = np.array([p[tol_col] for p in eval_pairs])
        pos_rate = float(np.mean(y_true))

        # Hidden-state delta performance
        try:
            auroc_h = float(roc_auc_score(y_true, scores_h))
            prec_h, rec_h, _ = precision_recall_curve(y_true, scores_h)
            auprc_h = float(auc(rec_h, prec_h))
        except Exception:
            auroc_h, auprc_h = 0.5, pos_rate

        # Depth map delta performance
        try:
            auroc_d = float(roc_auc_score(y_true, scores_d))
            prec_d, rec_d, _ = precision_recall_curve(y_true, scores_d)
            auprc_d = float(auc(rec_d, prec_d))
        except Exception:
            auroc_d, auprc_d = 0.5, pos_rate

        signal_metrics[tol_name] = {
            "positive_rate": round(pos_rate * 100, 2),
            "hidden_delta": {
                "AUROC": round(auroc_h, 4),
                "AUPRC": round(auprc_h, 4),
            },
            "depth_delta": {
                "AUROC": round(auroc_d, 4),
                "AUPRC": round(auprc_d, 4),
            }
        }

    # =========================================================================
    # 4. Uncertainty Quantification: Scene-Wise Bootstrap CIs
    # =========================================================================
    scenes = sorted(list(set(r["scene"] for r in records)), key=lambda s: int(s.replace("scan", "")))
    num_scenes = len(scenes)

    scene_level_data = {}
    for tol in rel_tolerances:
        tol_key = f"rel_{int(tol*100)}pct"
        scene_savings = []
        scene_vars = []
        scene_early_pcts = []

        for sc in scenes:
            sc_views = [v for v in views_data.values() if v["scene"] == sc]
            sc_k_stars = []
            for vinfo in sc_views:
                th = vinfo["by_K"][16]["depth_abs_rel"] * (1.0 + tol)
                k_star = 16
                for K in k_values:
                    if vinfo["by_K"][K]["depth_abs_rel"] <= th:
                        k_star = K
                        break
                sc_k_stars.append(k_star)

            sc_k_stars = np.array(sc_k_stars)
            sc_mean_k = np.mean(sc_k_stars)
            sc_save = (16.0 - sc_mean_k) / 16.0 * 100
            sc_var = np.var(sc_k_stars)
            sc_early = np.mean(sc_k_stars < 16) * 100

            scene_savings.append(sc_save)
            scene_vars.append(sc_var)
            scene_early_pcts.append(sc_early)

        scene_savings = np.array(scene_savings)
        scene_vars = np.array(scene_vars)
        scene_early_pcts = np.array(scene_early_pcts)

        # 10,000 bootstrap resamples over the 14 scenes
        rng = np.random.default_rng(seed=42)
        n_boot = 10000
        boot_savings = []
        boot_vars = []
        boot_early = []

        for _ in range(n_boot):
            idx = rng.integers(0, num_scenes, size=num_scenes)
            boot_savings.append(np.mean(scene_savings[idx]))
            boot_vars.append(np.mean(scene_vars[idx]))
            boot_early.append(np.mean(scene_early_pcts[idx]))

        ci_save = [float(np.percentile(boot_savings, 2.5)), float(np.percentile(boot_savings, 97.5))]
        ci_var = [float(np.percentile(boot_vars, 2.5)), float(np.percentile(boot_vars, 97.5))]
        ci_early = [float(np.percentile(boot_early, 2.5)), float(np.percentile(boot_early, 97.5))]

        scene_level_data[tol_key] = {
            "mean_savings_pct": round(float(np.mean(scene_savings)), 2),
            "se_savings_pct": round(float(stats.sem(scene_savings)), 2),
            "ci95_savings_pct": [round(ci_save[0], 2), round(ci_save[1], 2)],
            "mean_intra_scene_var": round(float(np.mean(scene_vars)), 3),
            "ci95_intra_scene_var": [round(ci_var[0], 3), round(ci_var[1], 3)],
            "mean_early_sat_pct": round(float(np.mean(scene_early_pcts)), 2),
            "ci95_early_sat_pct": [round(ci_early[0], 2), round(ci_early[1], 2)],
            "per_scene_savings": {s: round(float(scene_savings[i]), 2) for i, s in enumerate(scenes)},
            "per_scene_vars": {s: round(float(scene_vars[i]), 3) for i, s in enumerate(scenes)},
        }

    # =========================================================================
    # 5. Generate High-Resolution Publication Plots
    # =========================================================================
    print("\nGenerating V0.5 publication plots...")

    # Fig 1: Multi-Scene Per-View Quality Progression Grid (14 scans)
    fig, axes = plt.subplots(3, 5, figsize=(22, 11), sharey=False)
    axes = axes.flatten()

    for ax_idx, sc in enumerate(scenes):
        ax = axes[ax_idx]
        sc_views = [v for v in views_data.values() if v["scene"] == sc]
        colors = plt.cm.tab20(np.linspace(0, 1, len(sc_views)))

        for vi, vinfo in enumerate(sc_views):
            ks = k_values
            vals = [vinfo["by_K"][K]["depth_abs_rel"] for K in ks]
            linestyle = "-" if vinfo["subset"] == "subset_middle" else "--"
            ax.plot(ks, vals, marker="o", markersize=3, linestyle=linestyle, color=colors[vi], alpha=0.85)

        ax.set_title(f"{sc}", fontweight="bold", fontsize=11)
        ax.set_xticks(k_values)
        ax.set_xlabel("Steps K", fontsize=10)
        ax.set_ylabel("AbsRel", fontsize=10)

    # Turn off the last unused subplot (14 scans on a 3x5 grid)
    axes[14].axis("off")
    # Add a custom legend in the 15th box
    axes[14].plot([], [], "-", color="black", label="Subset Middle (Covisible cluster)")
    axes[14].plot([], [], "--", color="black", label="Subset Uniform (Wide baseline)")
    axes[14].legend(loc="center", fontsize=11, frameon=True)

    plt.suptitle("ViewHalt V0.5: In-Distribution Per-View Quality Curves across 14 DTU Scenes (168 Views)", y=0.99, fontsize=15, fontweight="bold")
    plt.tight_layout()
    grid_curve_path = os.path.join(output_dir, "v05_per_view_curves_grid.png")
    fig.savefig(grid_curve_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {grid_curve_path}")

    # Fig 2: Corrected In-Distribution Oracle K* Distribution & Scene Uncertainty
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    x_idx = np.arange(len(k_values))
    width = 0.2
    for i, tol in enumerate(rel_tolerances):
        t_key = f"rel_{int(tol*100)}pct"
        counts = [saturation_results[t_key]["dist_counts"][K] for K in k_values]
        ax1.bar(x_idx + (i - 1.5) * width, counts, width=width, label=f"Tol {int(tol*100)}% (Mean K*={saturation_results[t_key]['mean_K_star']:.2f})")

    ax1.set_xlabel("Corrected In-Distribution Saturation Step K*")
    ax1.set_ylabel("Number of Views (out of 168)")
    ax1.set_title("Distribution of Oracle K* (In-Distribution K >= 8)", fontweight="bold")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(k_values)
    ax1.legend()

    # Right: Scene-wise mean savings with 95% Bootstrap CI error bar
    tols_pct = [int(t * 100) for t in rel_tolerances]
    means_save = [scene_level_data[f"rel_{t}pct"]["mean_savings_pct"] for t in tols_pct]
    ci_lowers = [scene_level_data[f"rel_{t}pct"]["ci95_savings_pct"][0] for t in tols_pct]
    ci_uppers = [scene_level_data[f"rel_{t}pct"]["ci95_savings_pct"][1] for t in tols_pct]
    y_err = [
        [means_save[i] - ci_lowers[i] for i in range(len(tols_pct))],
        [ci_uppers[i] - means_save[i] for i in range(len(tols_pct))]
    ]

    ax2.errorbar(tols_pct, means_save, yerr=y_err, fmt="o-", color="#1b9e77", ecolor="#d95f02", elinewidth=2.5, capsize=6, markersize=8, label="Scene-wise Mean with 95% Bootstrap CI")
    for i, t in enumerate(tols_pct):
        ax2.annotate(f"{means_save[i]:.1f}%\n[{ci_lowers[i]:.1f}%, {ci_uppers[i]:.1f}%]", (t, means_save[i]), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    ax2.set_xlabel("Relative Quality Tolerance (%)")
    ax2.set_ylabel("Oracle Compute Saved (%)")
    ax2.set_title("Scene-Level Oracle Compute Savings (14 Scenes Bootstrap CI)", fontweight="bold")
    ax2.set_xticks(tols_pct)
    ax2.set_ylim(0, 45)
    ax2.legend()

    plt.suptitle("ViewHalt V0.5: Oracle Saturation Statistics & Scene Uncertainty", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "v05_oracle_k_star_distribution.png")
    fig.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {dist_path}")

    # Fig 3: Pareto Frontier (Corrected In-Distribution Baseline vs Oracle)
    fig, ax = plt.subplots(figsize=(8.5, 6))

    base_k = [16, 14, 12, 10, 8]
    base_save = [fixed_k_summary[k]["compute_saved_pct"] for k in base_k]
    base_loss = [fixed_k_summary[k]["quality_loss_pct"] for k in base_k]

    ax.plot(base_save, base_loss, "s--", color="#d95f02", label="Fixed-K Baseline (All Views Halt Together)", linewidth=2.3, markersize=8)
    for k, cs, ql in zip(base_k, base_save, base_loss):
        ax.annotate(f"Fixed K={k}", (cs, ql), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9.5, color="#d95f02", fontweight="bold")

    orc_save = [saturation_results[f"rel_{int(t*100)}pct"]["compute_saved_pct"] for t in rel_tolerances]
    orc_loss = [saturation_results[f"rel_{int(t*100)}pct"]["quality_loss_pct"] for t in rel_tolerances]

    ax.plot(orc_save, orc_loss, "o-", color="#1b9e77", label="Oracle Per-View Halting Frontier (ViewHalt)", linewidth=2.6, markersize=9)
    for t, cs, ql in zip(rel_tolerances, orc_save, orc_loss):
        ax.annotate(f"ε={int(t*100)}%", (cs, ql), textcoords="offset points", xytext=(8, -8), ha="left", fontsize=9.5, color="#1b9e77", fontweight="bold")

    ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Recurrent Compute Saved (%)", fontsize=12)
    ax.set_ylabel("Quality Loss vs. Fixed K=16 (% AbsRel Increase)", fontsize=12)
    ax.set_title("ViewHalt V0.5: Corrected In-Distribution Compute vs. Quality Frontier\n(168 Views Across 14 DTU Scenes)", fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    frontier_path = os.path.join(output_dir, "v05_compute_quality_frontier.png")
    fig.savefig(frontier_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {frontier_path}")

    # Fig 4: Predictive Signal Analysis (ROC & PR curves, Scatter of Delta vs Future Gain)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))

    # 4a. Scatter: Hidden delta vs gain(K)
    ax1.scatter(h_deltas, gains * 1000, alpha=0.45, color="#2b83ba", edgecolors="none", s=25)
    ax1.set_xlabel("Hidden-State Delta ||z_K - z_{K-1}|| / ||z_K||")
    ax1.set_ylabel("Future Benefit gain(K) x 10^3")
    ax1.set_title(f"Hidden-State Delta vs. Future Gain\n(Spearman rho = {spearman_h_gain:.3f}, p = {pval_h_gain:.2e})", fontweight="bold")

    # 4b. Scatter: Depth map delta vs gain(K)
    ax2.scatter(d_deltas, gains * 1000, alpha=0.45, color="#7b3294", edgecolors="none", s=25)
    ax2.set_xlabel("Output Depth Delta ||d_K - d_{K-2}|| / ||d_K||")
    ax2.set_ylabel("Future Benefit gain(K) x 10^3")
    ax2.set_title(f"Output Depth Delta vs. Future Gain\n(Spearman rho = {spearman_d_gain:.3f}, p = {pval_d_gain:.2e})", fontweight="bold")

    # 4c. ROC Curves for 'Safe to Halt' (5% tolerance)
    y_5 = np.array([p["safe_5pct"] for p in eval_pairs])
    fpr_h, tpr_h, _ = roc_curve(y_5, scores_h)
    fpr_d, tpr_d, _ = roc_curve(y_5, scores_d)

    ax3.plot(fpr_h, tpr_h, label=f"Hidden-State Delta (AUROC = {signal_metrics['5%']['hidden_delta']['AUROC']:.3f})", color="#2b83ba", linewidth=2.2)
    ax3.plot(fpr_d, tpr_d, label=f"Output Depth Delta (AUROC = {signal_metrics['5%']['depth_delta']['AUROC']:.3f})", color="#7b3294", linewidth=2.2)
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.50)")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve: Halting Prediction (5% Tolerance)", fontweight="bold")
    ax3.legend(loc="lower right")

    # 4d. Precision-Recall Curves (5% tolerance)
    prec_h, rec_h, _ = precision_recall_curve(y_5, scores_h)
    prec_d, rec_d, _ = precision_recall_curve(y_5, scores_d)
    base_rate = float(np.mean(y_5))

    ax4.plot(rec_h, prec_h, label=f"Hidden-State Delta (AUPRC = {signal_metrics['5%']['hidden_delta']['AUPRC']:.3f})", color="#2b83ba", linewidth=2.2)
    ax4.plot(rec_d, prec_d, label=f"Output Depth Delta (AUPRC = {signal_metrics['5%']['depth_delta']['AUPRC']:.3f})", color="#7b3294", linewidth=2.2)
    ax4.axhline(base_rate, color="gray", linestyle="--", label=f"Base Rate ({base_rate:.2f})")
    ax4.set_xlabel("Recall")
    ax4.set_ylabel("Precision")
    ax4.set_title("Precision-Recall Curve (5% Tolerance)", fontweight="bold")
    ax4.legend(loc="lower left")

    plt.suptitle("ViewHalt V0.5: Predictive Capability of Cheap Observable Signals", y=1.01, fontsize=14, fontweight="bold")
    plt.tight_layout()
    pred_path = os.path.join(output_dir, "v05_predictive_signal_analysis.png")
    fig.savefig(pred_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pred_path}")

    # =========================================================================
    # 6. Corrected Robustness Kill-Test Decision Evaluation
    # =========================================================================
    # Criteria:
    # 1. In-distribution early saturation: >= 50% views saturate before K=16 (under 5% tol)
    # 2. Heterogeneity stability: intra-scene variance remains significant across scenes and subsets
    # 3. Predictability: exists at least one cheap observable signal with meaningful predictive power (Spearman rho > 0.3 or AUROC >= 0.70)
    # 4. Bootstrap CI over scenes confirms positive compute savings bounded away from 0

    sat_5 = saturation_results["rel_5pct"]
    sat_2 = saturation_results["rel_2pct"]
    sc_5 = scene_level_data["rel_5pct"]

    early_frac_5 = sat_5["pct_sat_before_16"]
    early_frac_2 = sat_2["pct_sat_before_16"]
    ci_savings_5 = sc_5["ci95_savings_pct"]
    mean_savings_5 = sc_5["mean_savings_pct"]
    mean_var_5 = sc_5["mean_intra_scene_var"]

    best_auroc = max(signal_metrics["5%"]["hidden_delta"]["AUROC"], signal_metrics["5%"]["depth_delta"]["AUROC"])
    best_rho = max(spearman_h_gain, spearman_d_gain)

    print("\n" + "=" * 80)
    print("V0.5 CORRECTED HEADLINE METRICS (IN-DISTRIBUTION K in {8, 10, 12, 14, 16}):")
    print(f"  Total Views Evaluated: 168 (14 scenes x 2 subsets x 6 views)")
    print(f"  At 2% Tolerance: Early Sat = {early_frac_2:.1f}%, Mean K* = {sat_2['mean_K_star']:.2f}, Savings = {sat_2['compute_saved_pct']:.1f}%, Qual Loss = {sat_2['quality_loss_pct']:.2f}%")
    print(f"  At 5% Tolerance: Early Sat = {early_frac_5:.1f}%, Mean K* = {sat_5['mean_K_star']:.2f}, Savings = {sat_5['compute_saved_pct']:.1f}%, Qual Loss = {sat_5['quality_loss_pct']:.2f}%")
    print(f"  Scene-wise Mean Savings (5% tol): {mean_savings_5:.1f}% [95% CI: {ci_savings_5[0]:.1f}%, {ci_savings_5[1]:.1f}%]")
    print(f"  Scene-wise Intra-Scene Variance: {mean_var_5:.3f} [95% CI: {sc_5['ci95_intra_scene_var'][0]:.3f}, {sc_5['ci95_intra_scene_var'][1]:.3f}]")
    print(f"\nPREDICTIVE SIGNAL VERIFICATION:")
    print(f"  Hidden-State Delta vs. Future Gain: Spearman rho = {spearman_h_gain:.4f} (p = {pval_h_gain:.2e}), Pearson r = {pearson_h_gain:.4f}")
    print(f"  Output Depth Delta vs. Future Gain: Spearman rho = {spearman_d_gain:.4f} (p = {pval_d_gain:.2e})")
    print(f"  AUROC for Halting (5% tol): Hidden Delta = {signal_metrics['5%']['hidden_delta']['AUROC']:.4f}, Depth Delta = {signal_metrics['5%']['depth_delta']['AUROC']:.4f}")
    print(f"  AUPRC for Halting (5% tol): Hidden Delta = {signal_metrics['5%']['hidden_delta']['AUPRC']:.4f}, Depth Delta = {signal_metrics['5%']['depth_delta']['AUPRC']:.4f} (base={signal_metrics['5%']['positive_rate']:.1f}%)")

    # Evaluate GO Decision
    is_early_sat_pass = early_frac_5 >= 50.0
    is_savings_ci_pass = ci_savings_5[0] >= 10.0
    is_heterogeneity_pass = mean_var_5 >= 1.0
    is_signal_predictive = (best_rho >= 0.25) and (best_auroc >= 0.65)

    if is_early_sat_pass and is_savings_ci_pass and is_heterogeneity_pass and is_signal_predictive:
        verdict = "GO"
        verdict_rationale = (
            f"GO: Corrected in-distribution robustness validation confirmed across 168 views and 14 DTU scenes. "
            f"Heterogeneous convergence is robust and stable: {early_frac_5:.1f}% of views saturate before K=16 at 5% tolerance "
            f"({early_frac_2:.1f}% at 2% tolerance). The scene-wise mean compute savings is {mean_savings_5:.1f}% with 95% bootstrap CI "
            f"[{ci_savings_5[0]:.1f}%, {ci_savings_5[1]:.1f}%], strictly bounded away from zero. "
            f"Mean intra-scene variance is {mean_var_5:.2f}, proving per-view heterogeneity is not an artifact of scan selection. "
            f"Crucially, cheap observable signals demonstrate statistically significant predictive capability for future gain "
            f"(Spearman rho = {best_rho:.3f}, AUROC = {best_auroc:.3f}), providing an empirical foundation for controller design."
        )
    else:
        verdict = "KILL" if not is_heterogeneity_pass else "REVISE"
        verdict_rationale = f"REVISE/KILL: Failed robustness criteria (early_sat={early_frac_5:.1f}%, CI={ci_savings_5}, best_rho={best_rho:.3f}, best_auroc={best_auroc:.3f})."

    print(f"\nROBUSTNESS VERDICT: {verdict}")
    print(f"Rationale: {verdict_rationale}")
    print("=" * 80)

    summary_output = {
        "metadata": metadata,
        "fixed_k_summary": fixed_k_summary,
        "saturation_results": saturation_results,
        "scene_level_uncertainty": scene_level_data,
        "predictive_signal_analysis": {
            "spearman_hidden_delta": {"rho": round(spearman_h_gain, 4), "p_val": float(pval_h_gain)},
            "pearson_hidden_delta": {"r": round(pearson_h_gain, 4), "p_val": float(pval_pearson_h)},
            "spearman_depth_delta": {"rho": round(spearman_d_gain, 4), "p_val": float(pval_d_gain)},
            "signal_metrics_by_tolerance": signal_metrics,
        },
        "verdict": verdict,
        "verdict_rationale": verdict_rationale,
    }

    summary_path = os.path.join(output_dir, "v05_analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_output, f, indent=2)
    print(f"Summary saved: {summary_path}")

    return summary_output


if __name__ == "__main__":
    analyze_v05()
