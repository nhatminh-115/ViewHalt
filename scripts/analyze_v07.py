"""ViewHalt V0.7: Dense Common-Trajectory Pareto Resolution & Economic Headroom Analysis.

Core Analyses:
1. Complete fixed-stop frontier for every dense integer i ∈ {8..16}.
2. Recomputed dense oracle saturation K* across tolerances {1%, 2%, 5%, 10%}.
3. Matched-quality economic comparison:
   headroom = oracle_savings - best_fixed_savings_at_matched_quality
4. Non-parametric bootstrap of matched-quality headroom over 14 physical scenes (10,000 resamples).
5. Corrected out-of-sample predictive evaluation: Leave-One-Scene-Out (LOSO) cross-validation
   comparing Iteration-only vs Hidden-delta-only vs Iteration + Hidden-delta.
6. Row-aligned within-iteration z-score analysis via groupby("step_i").transform().
7. Clear scientific verdict: GO vs. KILL based on economic headroom.
8. Generation of 5 high-resolution publication figures.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, log_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut


def analyze_v07(
    raw_json_path="outputs/v07_raw_results.json",
    output_dir="outputs",
):
    print("=" * 80)
    print("Starting ViewHalt V0.7 Dense Pareto Resolution & Headroom Analysis")
    print("=" * 80)

    if not os.path.exists(raw_json_path):
        raise FileNotFoundError(f"Missing {raw_json_path}. Run run_v07_dense_common_trajectory.py first.")

    with open(raw_json_path, "r") as f:
        data = json.load(f)

    meta = data["metadata"]
    traj_records = data["trajectory_records"]
    frz_records = data["freeze_oracle_records"]

    df_traj = pd.DataFrame(traj_records)
    df_frz = pd.DataFrame(frz_records)

    eval_steps = meta["eval_steps"]  # [8, 9, 10, 11, 12, 13, 14, 15, 16]
    scans = meta["scans"]
    tolerances = meta["tolerances"]  # [0.01, 0.02, 0.05, 0.10]
    num_view_instances = meta["num_view_instances"]
    num_unique_physical_frames = meta["num_unique_physical_frames"]

    print(f"Loaded {len(df_traj)} trajectory records and {len(df_frz)} freeze-oracle records.")
    print(f"Total View Instances: {num_view_instances} (14 scans x 2 subsets x 6 views)")
    print(f"Total Unique Physical Frames: {num_unique_physical_frames}")

    # =========================================================================
    # 1. DENSE FIXED-STOP FRONTIER FOR EVERY i ∈ {8..16}
    # =========================================================================
    fixed_k_summary = {}
    k16_mean_absrel = float(df_traj[df_traj["step_i"] == 16]["abs_rel"].mean())

    fixed_savings_list = []
    fixed_absrel_list = []

    for step in eval_steps:
        sub = df_traj[df_traj["step_i"] == step]
        step_saved_pct = ((16.0 - step) / 16.0) * 100.0
        mean_absrel = float(sub["abs_rel"].mean())
        qual_loss_pct = ((mean_absrel - k16_mean_absrel) / k16_mean_absrel) * 100.0

        fixed_savings_list.append(step_saved_pct)
        fixed_absrel_list.append(mean_absrel)

        fixed_k_summary[str(step)] = {
            "step": step,
            "step_saved_pct": round(step_saved_pct, 2),
            "mean_abs_rel": round(mean_absrel, 6),
            "std_abs_rel": round(float(sub["abs_rel"].std()), 6),
            "median_abs_rel": round(float(sub["abs_rel"].median()), 6),
            "mean_rmse_mm": round(float(sub["rmse"].mean()), 4),
            "mean_mae_mm": round(float(sub["mae"].mean()), 4),
            "mean_delta1": round(float(sub["delta1"].mean()), 4),
            "mean_rot_err_deg": round(float(sub["rot_err_deg"].mean()), 4),
            "mean_trans_err_deg": round(float(sub["trans_err_deg"].mean()), 4),
            "mean_geom_err_mm": round(float(sub["geom_l2"].mean()), 4),
            "quality_loss_pct": round(qual_loss_pct, 2),
        }

    # =========================================================================
    # 2. RECOMPUTE DENSE ORACLE SATURATION OVER {8..16}
    # =========================================================================
    df_step16 = df_traj[df_traj["step_i"] == 16].copy()
    tol_names = {0.01: "1pct", 0.02: "2pct", 0.05: "5pct", 0.10: "10pct"}
    dense_oracle_summary = {}

    for tol in tolerances:
        tol_key = f"rel_{tol_names[tol]}"
        col_kstar = f"oracle_k_star_{tol_names[tol]}"
        k_stars = df_step16[col_kstar].values

        mean_k = float(np.mean(k_stars))
        median_k = float(np.median(k_stars))
        std_k = float(np.std(k_stars))

        dist_counts = {str(s): int(np.sum(k_stars == s)) for s in eval_steps}
        pct_sat_before_16 = float(np.mean(k_stars < 16) * 100.0)

        step_saved_pct = float(((16.0 - mean_k) / 16.0) * 100.0)

        scene_variances = {}
        for sc in scans:
            sc_k = df_step16[df_step16["scene"] == sc][col_kstar].values
            scene_variances[sc] = float(np.var(sc_k, ddof=1)) if len(sc_k) > 1 else 0.0
        mean_intra_var = float(np.mean(list(scene_variances.values())))

        # Decode oracle AbsRel
        oracle_absrels = []
        for _, row in df_step16.iterrows():
            sc = row["scene"]
            sub = row["subset"]
            v = row["view_idx"]
            k_star_v = row[col_kstar]
            match_row = df_traj[
                (df_traj["scene"] == sc) &
                (df_traj["subset"] == sub) &
                (df_traj["view_idx"] == v) &
                (df_traj["step_i"] == k_star_v)
            ]
            oracle_absrels.append(match_row["abs_rel"].values[0])

        mean_oracle_absrel = float(np.mean(oracle_absrels))
        qual_loss_pct = float(((mean_oracle_absrel - k16_mean_absrel) / k16_mean_absrel) * 100.0)

        dense_oracle_summary[tol_key] = {
            "tolerance": tol,
            "mean_K_star": round(mean_k, 2),
            "median_K_star": round(median_k, 2),
            "std_K_star": round(std_k, 2),
            "dist_counts": dist_counts,
            "pct_sat_before_16": round(pct_sat_before_16, 2),
            "step_equivalent_savings_pct": round(step_saved_pct, 2),
            "mean_intra_scene_variance": round(mean_intra_var, 3),
            "scene_variances": scene_variances,
            "mean_decode_oracle_absrel": round(mean_oracle_absrel, 6),
            "mean_k16_absrel": round(k16_mean_absrel, 6),
            "quality_loss_pct": round(qual_loss_pct, 2),
        }

    # =========================================================================
    # 3. DENSE SIMULATION FREEZE ORACLE SUMMARY
    # =========================================================================
    dense_freeze_summary = {}

    for tol_name, tol_val in [("5pct", 0.05), ("2pct", 0.02)]:
        sub_frz = df_frz[df_frz["tolerance"] == tol_name].copy()

        mean_frz_absrel = float(sub_frz["freeze_abs_rel"].mean())
        mean_dec_absrel = float(sub_frz["decode_oracle_abs_rel"].mean())
        mean_k16_absrel = float(sub_frz["fixed16_abs_rel"].mean())

        k_stars = sub_frz["oracle_k_star"].values
        mean_k_star = float(np.mean(k_stars))
        step_saved_pct = float(((16.0 - mean_k_star) / 16.0) * 100.0)

        active_later_mask = sub_frz["is_active_later"].values
        sub_active = sub_frz[active_later_mask]
        deltas_active = sub_active["delta_freeze_vs_decode"].values
        mean_delta_active = float(np.mean(deltas_active)) if len(deltas_active) > 0 else 0.0
        mean_delta_active_vs_k16 = float(np.mean((sub_active["freeze_abs_rel"] - sub_active["fixed16_abs_rel"]).values)) if len(sub_active) > 0 else 0.0

        dense_freeze_summary[tol_name] = {
            "tolerance_val": tol_val,
            "mean_k_star": round(mean_k_star, 2),
            "step_equivalent_savings_pct": round(step_saved_pct, 2),
            "mean_freeze_oracle_absrel": round(mean_frz_absrel, 6),
            "mean_decode_oracle_absrel": round(mean_dec_absrel, 6),
            "mean_fixed_k16_absrel": round(mean_k16_absrel, 6),
            "quality_loss_freeze_vs_k16_pct": round(((mean_frz_absrel - mean_k16_absrel) / mean_k16_absrel) * 100.0, 2),
            "mean_delta_active_views": round(mean_delta_active, 6),
            "mean_delta_active_views_vs_k16": round(mean_delta_active_vs_k16, 6),
        }

    # =========================================================================
    # 4. MATCHED-QUALITY ECONOMIC COMPARISON (THE CORE ECONOMIC QUANTITY)
    # =========================================================================
    # Helper to find best fixed savings at matched quality
    # Fixed policies are sorted by step: i=16 (sav=0%), i=15 (sav=6.25%), ..., i=8 (sav=50%)
    fixed_steps_arr = np.array(eval_steps)  # [8..16]
    fixed_savings_arr = np.array([fixed_k_summary[str(s)]["step_saved_pct"] for s in eval_steps])  # [50%, ..., 0%]
    fixed_absrel_arr = np.array([fixed_k_summary[str(s)]["mean_abs_rel"] for s in eval_steps])    # [0.0185, ..., 0.0089]

    # Create interpolation function: given error, what is fixed saving?
    # Monotonic part: sort by AbsRel increasing (from i=16 to i=8)
    sort_idx = np.argsort(fixed_absrel_arr)
    err_sorted = fixed_absrel_arr[sort_idx]
    sav_sorted = fixed_savings_arr[sort_idx]

    # Interpolator
    f_interp_saving = interpolate.interp1d(err_sorted, sav_sorted, kind="linear", fill_value="extrapolate")

    def get_matched_quality_savings(target_absrel):
        # Discrete: best fixed policy whose mean AbsRel <= target_absrel
        eligible_steps = [s for s in eval_steps if fixed_k_summary[str(s)]["mean_abs_rel"] <= target_absrel + 1e-9]
        if eligible_steps:
            best_discrete_step = min(eligible_steps)  # minimum step = maximum savings
            best_discrete_savings = fixed_k_summary[str(best_discrete_step)]["step_saved_pct"]
        else:
            best_discrete_step = 16
            best_discrete_savings = 0.0

        # Continuous / interpolated
        if target_absrel <= err_sorted[0]:
            interp_saving = 0.0
        elif target_absrel >= err_sorted[-1]:
            interp_saving = float(sav_sorted[-1])
        else:
            interp_saving = float(f_interp_saving(target_absrel))

        return best_discrete_step, best_discrete_savings, interp_saving

    matched_headroom_analysis = {}

    # Evaluate for Freeze Oracle (5% and 2%) and Decode Oracle (all tolerances)
    eval_oracles = [
        ("Freeze_Oracle_5pct", dense_freeze_summary["5pct"]["mean_freeze_oracle_absrel"], dense_freeze_summary["5pct"]["step_equivalent_savings_pct"]),
        ("Freeze_Oracle_2pct", dense_freeze_summary["2pct"]["mean_freeze_oracle_absrel"], dense_freeze_summary["2pct"]["step_equivalent_savings_pct"]),
        ("Decode_Oracle_5pct", dense_oracle_summary["rel_5pct"]["mean_decode_oracle_absrel"], dense_oracle_summary["rel_5pct"]["step_equivalent_savings_pct"]),
        ("Decode_Oracle_2pct", dense_oracle_summary["rel_2pct"]["mean_decode_oracle_absrel"], dense_oracle_summary["rel_2pct"]["step_equivalent_savings_pct"]),
        ("Decode_Oracle_1pct", dense_oracle_summary["rel_1pct"]["mean_decode_oracle_absrel"], dense_oracle_summary["rel_1pct"]["step_equivalent_savings_pct"]),
        ("Decode_Oracle_10pct", dense_oracle_summary["rel_10pct"]["mean_decode_oracle_absrel"], dense_oracle_summary["rel_10pct"]["step_equivalent_savings_pct"]),
    ]

    for name, err_val, sav_val in eval_oracles:
        best_step, best_disc_sav, interp_sav = get_matched_quality_savings(err_val)
        disc_headroom = sav_val - best_disc_sav
        interp_headroom = sav_val - interp_sav

        matched_headroom_analysis[name] = {
            "oracle_absrel": round(err_val, 6),
            "oracle_step_savings_pct": round(sav_val, 2),
            "matched_fixed_step_discrete": best_step,
            "matched_fixed_savings_discrete_pct": round(best_disc_sav, 2),
            "adaptive_headroom_discrete_pct": round(disc_headroom, 2),
            "matched_fixed_savings_interpolated_pct": round(interp_sav, 2),
            "adaptive_headroom_interpolated_pct": round(interp_headroom, 2),
        }

    # =========================================================================
    # 5. BOOTSTRAP UNCERTAINTY OVER 14 PHYSICAL SCENES (10,000 RESAMPLES)
    # =========================================================================
    np.random.seed(42)
    n_bootstrap = 10000
    n_scenes = len(scans)

    # For each scene, precompute:
    # 1. scene-mean AbsRel for each fixed step i ∈ {8..16}
    # 2. scene-mean AbsRel and savings for Freeze Oracle 5%
    # 3. scene-mean AbsRel and savings for Freeze Oracle 2%
    scene_fixed_absrel = {sc: {} for sc in scans}
    for sc in scans:
        df_sc = df_traj[df_traj["scene"] == sc]
        for s in eval_steps:
            scene_fixed_absrel[sc][s] = float(df_sc[df_sc["step_i"] == s]["abs_rel"].mean())

    scene_frz_5pct = {}
    scene_frz_2pct = {}
    for sc in scans:
        df_sc_frz5 = df_frz[(df_frz["scene"] == sc) & (df_frz["tolerance"] == "5pct")]
        df_sc_frz2 = df_frz[(df_frz["scene"] == sc) & (df_frz["tolerance"] == "2pct")]

        sav5 = float(((16.0 - df_sc_frz5["oracle_k_star"].mean()) / 16.0) * 100.0)
        err5 = float(df_sc_frz5["freeze_abs_rel"].mean())
        scene_frz_5pct[sc] = (sav5, err5)

        sav2 = float(((16.0 - df_sc_frz2["oracle_k_star"].mean()) / 16.0) * 100.0)
        err2 = float(df_sc_frz2["freeze_abs_rel"].mean())
        scene_frz_2pct[sc] = (sav2, err2)

    boot_headroom_discrete_5pct = []
    boot_headroom_interp_5pct = []
    boot_headroom_discrete_2pct = []
    boot_headroom_interp_2pct = []

    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n_scenes, size=n_scenes, replace=True)
        resampled_scans = [scans[j] for j in boot_idx]

        # Resampled fixed curve
        boot_fixed_err = [np.mean([scene_fixed_absrel[sc][s] for sc in resampled_scans]) for s in eval_steps]

        # 5% Freeze Oracle
        boot_frz_sav_5 = np.mean([scene_frz_5pct[sc][0] for sc in resampled_scans])
        boot_frz_err_5 = np.mean([scene_frz_5pct[sc][1] for sc in resampled_scans])

        # Best fixed step whose error <= boot_frz_err_5
        elig_5 = [s for s, err in zip(eval_steps, boot_fixed_err) if err <= boot_frz_err_5 + 1e-9]
        best_s_5 = min(elig_5) if elig_5 else 16
        best_sav_5 = ((16.0 - best_s_5) / 16.0) * 100.0
        boot_headroom_discrete_5pct.append(boot_frz_sav_5 - best_sav_5)

        # Interpolated
        s_idx5 = np.argsort(boot_fixed_err)
        b_err_sort = np.array(boot_fixed_err)[s_idx5]
        b_sav_sort = fixed_savings_arr[s_idx5]
        if boot_frz_err_5 <= b_err_sort[0]:
            b_interp_sav5 = 0.0
        elif boot_frz_err_5 >= b_err_sort[-1]:
            b_interp_sav5 = float(b_sav_sort[-1])
        else:
            b_interp_sav5 = float(interpolate.interp1d(b_err_sort, b_sav_sort)(boot_frz_err_5))
        boot_headroom_interp_5pct.append(boot_frz_sav_5 - b_interp_sav5)

        # 2% Freeze Oracle
        boot_frz_sav_2 = np.mean([scene_frz_2pct[sc][0] for sc in resampled_scans])
        boot_frz_err_2 = np.mean([scene_frz_2pct[sc][1] for sc in resampled_scans])

        elig_2 = [s for s, err in zip(eval_steps, boot_fixed_err) if err <= boot_frz_err_2 + 1e-9]
        best_s_2 = min(elig_2) if elig_2 else 16
        best_sav_2 = ((16.0 - best_s_2) / 16.0) * 100.0
        boot_headroom_discrete_2pct.append(boot_frz_sav_2 - best_sav_2)

        if boot_frz_err_2 <= b_err_sort[0]:
            b_interp_sav2 = 0.0
        elif boot_frz_err_2 >= b_err_sort[-1]:
            b_interp_sav2 = float(b_sav_sort[-1])
        else:
            b_interp_sav2 = float(interpolate.interp1d(b_err_sort, b_sav_sort)(boot_frz_err_2))
        boot_headroom_interp_2pct.append(boot_frz_sav_2 - b_interp_sav2)

    ci95_headroom_disc_5 = [round(float(np.percentile(boot_headroom_discrete_5pct, 2.5)), 2), round(float(np.percentile(boot_headroom_discrete_5pct, 97.5)), 2)]
    ci95_headroom_interp_5 = [round(float(np.percentile(boot_headroom_interp_5pct, 2.5)), 2), round(float(np.percentile(boot_headroom_interp_5pct, 97.5)), 2)]

    ci95_headroom_disc_2 = [round(float(np.percentile(boot_headroom_discrete_2pct, 2.5)), 2), round(float(np.percentile(boot_headroom_discrete_2pct, 97.5)), 2)]
    ci95_headroom_interp_2 = [round(float(np.percentile(boot_headroom_interp_2pct, 2.5)), 2), round(float(np.percentile(boot_headroom_interp_2pct, 97.5)), 2)]

    bootstrap_results = {
        "Freeze_Oracle_5pct": {
            "mean_discrete_headroom_pct": round(float(np.mean(boot_headroom_discrete_5pct)), 2),
            "ci95_discrete_headroom_pct": ci95_headroom_disc_5,
            "mean_interpolated_headroom_pct": round(float(np.mean(boot_headroom_interp_5pct)), 2),
            "ci95_interpolated_headroom_pct": ci95_headroom_interp_5,
            "pct_bootstrap_samples_headroom_le_zero": round(float(np.mean(np.array(boot_headroom_discrete_5pct) <= 0.0) * 100.0), 2),
        },
        "Freeze_Oracle_2pct": {
            "mean_discrete_headroom_pct": round(float(np.mean(boot_headroom_discrete_2pct)), 2),
            "ci95_discrete_headroom_pct": ci95_headroom_disc_2,
            "mean_interpolated_headroom_pct": round(float(np.mean(boot_headroom_interp_2pct)), 2),
            "ci95_interpolated_headroom_pct": ci95_headroom_interp_2,
            "pct_bootstrap_samples_headroom_le_zero": round(float(np.mean(np.array(boot_headroom_discrete_2pct) <= 0.0) * 100.0), 2),
        }
    }

    # =========================================================================
    # 6. CORRECTED PREDICTIVE-SIGNAL EVALUATION (LEAVE-ONE-SCENE-OUT CV)
    # =========================================================================
    # Align within-iteration z-score properly using groupby transform
    df_eval = df_traj[df_traj["step_i"].isin([8, 9, 10, 11, 12, 13, 14, 15])].copy()
    df_eval["hidden_delta_zscore"] = df_eval.groupby("step_i")["hidden_delta_1step"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    # Leave-One-Scene-Out (LOSO) Cross-Validation
    logo = LeaveOneGroupOut()
    groups = df_eval["scene"].values

    loso_results = {}

    for tol in [0.01, 0.02, 0.05]:
        tol_str = f"{int(tol*100)}%"
        col_safe = f"safe_to_halt_{tol_names[tol]}"
        y = df_eval[col_safe].values

        X_iter = df_eval[["step_i"]].values
        X_delta = df_eval[["hidden_delta_1step"]].values
        X_both = df_eval[["step_i", "hidden_delta_1step"]].values

        oof_prob_iter = np.zeros(len(df_eval))
        oof_prob_delta = np.zeros(len(df_eval))
        oof_prob_both = np.zeros(len(df_eval))

        for train_idx, test_idx in logo.split(X_both, y, groups=groups):
            # Model 1: Iteration only
            clf1 = LogisticRegression(C=1e5, solver="lbfgs").fit(X_iter[train_idx], y[train_idx])
            oof_prob_iter[test_idx] = clf1.predict_proba(X_iter[test_idx])[:, 1]

            # Model 2: Hidden delta only
            clf2 = LogisticRegression(C=1e5, solver="lbfgs").fit(X_delta[train_idx], y[train_idx])
            oof_prob_delta[test_idx] = clf2.predict_proba(X_delta[test_idx])[:, 1]

            # Model 3: Both (Iteration + Hidden Delta)
            clf3 = LogisticRegression(C=1e5, solver="lbfgs").fit(X_both[train_idx], y[train_idx])
            oof_prob_both[test_idx] = clf3.predict_proba(X_both[test_idx])[:, 1]

        # Out-of-Sample AUROC and AUPRC
        auroc_iter = float(roc_auc_score(y, oof_prob_iter))
        auroc_delta = float(roc_auc_score(y, oof_prob_delta))
        auroc_both = float(roc_auc_score(y, oof_prob_both))

        prec_i, rec_i, _ = precision_recall_curve(y, oof_prob_iter)
        prec_d, rec_d, _ = precision_recall_curve(y, oof_prob_delta)
        prec_b, rec_b, _ = precision_recall_curve(y, oof_prob_both)

        auprc_iter = float(auc(rec_i, prec_i))
        auprc_delta = float(auc(rec_d, prec_d))
        auprc_both = float(auc(rec_b, prec_b))

        delta_auroc_oos = auroc_both - auroc_iter
        delta_auprc_oos = auprc_both - auprc_iter

        loso_results[tol_str] = {
            "positive_rate_pct": round(float(np.mean(y) * 100.0), 2),
            "model1_iteration_only": {
                "OOS_AUROC": round(auroc_iter, 4),
                "OOS_AUPRC": round(auprc_iter, 4),
            },
            "model2_hidden_delta_only": {
                "OOS_AUROC": round(auroc_delta, 4),
                "OOS_AUPRC": round(auprc_delta, 4),
            },
            "model3_iteration_plus_hidden": {
                "OOS_AUROC": round(auroc_both, 4),
                "OOS_AUPRC": round(auprc_both, 4),
            },
            "incremental_oos_improvement": {
                "delta_AUROC_OOS": round(delta_auroc_oos, 4),
                "delta_AUPRC_OOS": round(delta_auprc_oos, 4),
                "hidden_delta_adds_generalizable_value": bool(delta_auroc_oos > 0.02 and delta_auprc_oos > 0.0),
            }
        }

    # =========================================================================
    # 7. SCIENTIFIC VERDICT: GO VS. KILL (PRIMARY CRITERION: ECONOMIC HEADROOM)
    # =========================================================================
    # In V0.7:
    # "GO if adaptive freeze oracle has a meaningful, statistically robust matched-quality compute advantage over the best uniform stopping policy."
    # "KILL if the matched-quality advantage is negligible after dense-step evaluation."
    headroom_5_disc = matched_headroom_analysis["Freeze_Oracle_5pct"]["adaptive_headroom_discrete_pct"]
    headroom_5_interp = matched_headroom_analysis["Freeze_Oracle_5pct"]["adaptive_headroom_interpolated_pct"]
    ci_lower_5 = bootstrap_results["Freeze_Oracle_5pct"]["ci95_interpolated_headroom_pct"][0]

    # Meaningful headroom requires at least 4% matched compute savings with 95% CI bounded safely away from zero
    is_headroom_meaningful = bool(headroom_5_interp >= 4.0 and ci_lower_5 > 1.0)

    if is_headroom_meaningful:
        verdict = "GO"
        verdict_rationale = (
            f"GO: Adaptive freeze oracle achieves a meaningful, statistically robust matched-quality compute advantage "
            f"over uniform fixed stopping. Headroom = {headroom_5_interp:.2f}% [95% CI: {ci_lower_5:.2f}%, {bootstrap_results['Freeze_Oracle_5pct']['ci95_interpolated_headroom_pct'][1]:.2f}%]."
        )
    else:
        verdict = "KILL"
        verdict_rationale = (
            f"KILL: Matched-quality economic headroom is negligible after dense-step evaluation. "
            f"At 5% tolerance, the Freeze-Oracle achieves AbsRel = {dense_freeze_summary['5pct']['mean_freeze_oracle_absrel']:.6f} with {dense_freeze_summary['5pct']['step_equivalent_savings_pct']:.2f}% savings. "
            f"However, uniform stopping at i=15 already achieves AbsRel = {fixed_k_summary['15']['mean_abs_rel']:.6f} with {fixed_k_summary['15']['step_saved_pct']:.2f}% savings, and "
            f"interpolated matched-quality fixed savings is {matched_headroom_analysis['Freeze_Oracle_5pct']['matched_fixed_savings_interpolated_pct']:.2f}%. "
            f"The net adaptive headroom is only {headroom_5_interp:.2f}% [95% bootstrap CI: {ci_lower_5:.2f}%, {bootstrap_results['Freeze_Oracle_5pct']['ci95_interpolated_headroom_pct'][1]:.2f}%]. "
            f"Such a marginal compute advantage cannot justify the runtime controller latency, dynamic attention masking overhead, and KV-cache complexity of per-view halting. "
            f"Uniform step tuning (e.g. fixed i=14 or i=15) captures virtually the entire Pareto benefit with zero architectural modification."
        )

    print("\n" + "=" * 80)
    print(f"V0.7 SCIENTIFIC VERDICT: {verdict}")
    print(f"Rationale: {verdict_rationale}")
    print("=" * 80)

    # =========================================================================
    # 8. GENERATE PUBLICATION FIGURES
    # =========================================================================
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig_dir = output_dir

    # Figure 1: Matched-Quality Pareto Frontier
    fig, ax = plt.subplots(figsize=(11, 7))
    # Dense Fixed-K curve
    ax.plot(fixed_savings_arr, fixed_absrel_arr, marker="o", color="#1f77b4", linewidth=2.5, markersize=7, label="Dense Uniform Stopping (Fixed i ∈ {8..16})")
    for s, sav, err in zip(eval_steps, fixed_savings_arr, fixed_absrel_arr):
        offset = (-12, 10) if s % 2 == 0 else (8, -12)
        ax.annotate(f"i={s}", (sav, err), textcoords="offset points", xytext=offset, fontsize=10, fontweight="bold", color="#1f77b4")

    # Oracle Points
    frz_sav_5 = dense_freeze_summary["5pct"]["step_equivalent_savings_pct"]
    frz_err_5 = dense_freeze_summary["5pct"]["mean_freeze_oracle_absrel"]
    ax.scatter([frz_sav_5], [frz_err_5], marker="*", s=260, color="#d62728", zorder=6, label="Freeze-Oracle (τ=5%)")
    ax.annotate(f"Freeze τ=5% ({frz_sav_5:.1f}%)", (frz_sav_5, frz_err_5), textcoords="offset points", xytext=(-35, 12), fontweight="bold", color="#d62728")

    # Matched-quality horizontal projection line
    matched_fixed_sav_5 = matched_headroom_analysis["Freeze_Oracle_5pct"]["matched_fixed_savings_interpolated_pct"]
    ax.hlines(y=frz_err_5, xmin=matched_fixed_sav_5, xmax=frz_sav_5, color="#d62728", linestyle=":", linewidth=2.5,
              label=f"Matched-Quality Headroom = {headroom_5_interp:.2f}%")

    # Fixed i=15 and i=14 highlight
    ax.scatter([fixed_k_summary["15"]["step_saved_pct"]], [fixed_k_summary["15"]["mean_abs_rel"]], s=120, facecolors='none', edgecolors='#2ca02c', linewidths=2.5, label="Fixed i=15 (Matched Benchmark)")

    ax.set_xlabel("Recurrent Step-Equivalent Compute Savings (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Reconstruction Error (AbsRel, Lower is Better)", fontsize=12, fontweight="bold")
    ax.set_title("ViewHalt V0.7: Dense Common-Trajectory Matched-Quality Pareto Resolution", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()
    p1 = os.path.join(fig_dir, "v07_matched_quality_pareto_frontier.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # Figure 2: Bootstrap Distribution of Matched-Quality Headroom
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(boot_headroom_interp_5pct, bins=35, color="#1f77b4", edgecolor="black", alpha=0.75, density=True)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=2.5, label="Zero Headroom (Breakeven)")
    ax.axvline(float(np.mean(boot_headroom_interp_5pct)), color="black", linestyle="-", linewidth=2.0, label=f"Mean Headroom = {np.mean(boot_headroom_interp_5pct):.2f}%")
    ax.axvline(ci95_headroom_interp_5[0], color="green", linestyle=":", linewidth=2.0, label=f"95% CI Lower Bound = {ci95_headroom_interp_5[0]:.2f}%")
    ax.axvline(ci95_headroom_interp_5[1], color="green", linestyle=":", linewidth=2.0, label=f"95% CI Upper Bound = {ci95_headroom_interp_5[1]:.2f}%")
    ax.set_xlabel("Interpolated Matched-Quality Adaptive Headroom (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
    ax.set_title("ViewHalt V0.7: Scene-Bootstrap Distribution of Matched-Quality Headroom (10,000 Resamples)", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(fig_dir, "v07_bootstrap_headroom_distribution.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    # Figure 3: Dense K* Saturation Distribution
    fig, ax = plt.subplots(figsize=(11, 6))
    x_indices = np.arange(len(eval_steps))
    width = 0.2
    palette = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    for idx, tol in enumerate(tolerances):
        tol_key = f"rel_{tol_names[tol]}"
        counts = [dense_oracle_summary[tol_key]["dist_counts"][str(s)] for s in eval_steps]
        ax.bar(x_indices + idx * width, counts, width=width, label=f"τ = {int(tol*100)}% (Mean K*={dense_oracle_summary[tol_key]['mean_K_star']})", color=palette[idx], alpha=0.85)
    ax.set_xticks(x_indices + 1.5 * width)
    ax.set_xticklabels([f"i={s}" for s in eval_steps], fontsize=11)
    ax.set_xlabel("Earliest Dense Saturation Step (K*)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of View Instances (N=168)", fontsize=12, fontweight="bold")
    ax.set_title("ViewHalt V0.7: Dense Common-Trajectory Oracle Saturation Distribution", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    p3 = os.path.join(fig_dir, "v07_dense_k_star_distribution.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")

    # Figure 4: Leave-One-Scene-Out (LOSO) ROC and PR Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    y_5 = df_eval["safe_to_halt_5pct"].values

    # Get out-of-sample predictions for 5% tolerance
    # Re-run quick loop to get exact arrays for plotting
    oof_iter = np.zeros(len(df_eval))
    oof_delta = np.zeros(len(df_eval))
    oof_both = np.zeros(len(df_eval))
    for train_idx, test_idx in logo.split(df_eval[["step_i", "hidden_delta_1step"]].values, y_5, groups=groups):
        clf1 = LogisticRegression(C=1e5).fit(df_eval[["step_i"]].values[train_idx], y_5[train_idx])
        oof_iter[test_idx] = clf1.predict_proba(df_eval[["step_i"]].values[test_idx])[:, 1]
        clf2 = LogisticRegression(C=1e5).fit(df_eval[["hidden_delta_1step"]].values[train_idx], y_5[train_idx])
        oof_delta[test_idx] = clf2.predict_proba(df_eval[["hidden_delta_1step"]].values[test_idx])[:, 1]
        clf3 = LogisticRegression(C=1e5).fit(df_eval[["step_i", "hidden_delta_1step"]].values[train_idx], y_5[train_idx])
        oof_both[test_idx] = clf3.predict_proba(df_eval[["step_i", "hidden_delta_1step"]].values[test_idx])[:, 1]

    # ROC
    fpr_i, tpr_i, _ = roc_curve(y_5, oof_iter)
    fpr_d, tpr_d, _ = roc_curve(y_5, oof_delta)
    fpr_b, tpr_b, _ = roc_curve(y_5, oof_both)
    ax1.plot(fpr_i, tpr_i, label=f"Model 1: Iteration Only (OOS AUROC={loso_results['5%']['model1_iteration_only']['OOS_AUROC']})", color="#1f77b4", linewidth=2.2)
    ax1.plot(fpr_d, tpr_d, label=f"Model 2: Hidden Delta Only (OOS AUROC={loso_results['5%']['model2_hidden_delta_only']['OOS_AUROC']})", color="#ff7f0e", linestyle="--", linewidth=2.0)
    ax1.plot(fpr_b, tpr_b, label=f"Model 3: Iteration + Delta (OOS AUROC={loso_results['5%']['model3_iteration_plus_hidden']['OOS_AUROC']})", color="#2ca02c", linewidth=2.2)
    ax1.plot([0, 1], [0, 1], color="gray", linestyle=":")
    ax1.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax1.set_title("Out-of-Scene Generalization: ROC Curves (LOSO CV, τ=5%)", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, fontsize=9.5)

    # PR
    prec_i, rec_i, _ = precision_recall_curve(y_5, oof_iter)
    prec_d, rec_d, _ = precision_recall_curve(y_5, oof_delta)
    prec_b, rec_b, _ = precision_recall_curve(y_5, oof_both)
    ax2.plot(rec_i, prec_i, label=f"Model 1: Iteration Only (OOS AUPRC={loso_results['5%']['model1_iteration_only']['OOS_AUPRC']})", color="#1f77b4", linewidth=2.2)
    ax2.plot(rec_d, prec_d, label=f"Model 2: Hidden Delta Only (OOS AUPRC={loso_results['5%']['model2_hidden_delta_only']['OOS_AUPRC']})", color="#ff7f0e", linestyle="--", linewidth=2.0)
    ax2.plot(rec_b, prec_b, label=f"Model 3: Iteration + Delta (OOS AUPRC={loso_results['5%']['model3_iteration_plus_hidden']['OOS_AUPRC']})", color="#2ca02c", linewidth=2.2)
    ax2.axhline(float(np.mean(y_5)), color="gray", linestyle=":", label=f"Base Rate ({float(np.mean(y_5)*100):.1f}%)")
    ax2.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax2.set_title("Out-of-Scene Generalization: Precision-Recall (LOSO CV, τ=5%)", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, fontsize=9.5)
    plt.tight_layout()
    p4 = os.path.join(fig_dir, "v07_out_of_sample_predictive_evaluation.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved: {p4}")

    # Summary JSON export
    summary_data = {
        "metadata": meta,
        "dense_fixed_stopping_frontier": fixed_k_summary,
        "dense_oracle_saturation_summary": dense_oracle_summary,
        "dense_freeze_oracle_summary": dense_freeze_summary,
        "matched_quality_headroom_analysis": matched_headroom_analysis,
        "scene_bootstrap_headroom_10k": bootstrap_results,
        "leave_one_scene_out_predictive_evaluation": loso_results,
        "verdict": verdict,
        "verdict_rationale": verdict_rationale,
    }

    summary_path = os.path.join(output_dir, "v07_analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved Analysis Summary: {summary_path}")

    return summary_data


if __name__ == "__main__":
    analyze_v07()
