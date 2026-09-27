"""DVLT Iteration-Role V0.1 — Recovery-Horizon Control Diagnostic.

Investigates whether the Frame-vs-Global role shift observed in V0 is due to:
A. genuine iteration-specific functional specialization (immediate intrinsic causal effect), or
B. a recovery-horizon confound (equal intrinsic importance, but early iterations have more downstream steps to recover).

Protocol:
- Untouched pretrained NVIDIA DVLT (bf16).
- 5 DTU validation scans (scan1, scan4, scan9, scan24, scan62) x 2 subsets (middle, uniform) = 10 sequences / 60 views.
- Common Uniform K=16 trajectory: ts = torch.linspace(0, 1, 16).
- For each intervention (Skip Frame at step i, Skip Global at step i):
  Evaluate after recovery horizons R in {0, 1, 2, 4} where i + R <= 16.
  Compare against the untouched baseline at step i + R:
  delta_absrel(i, R) = AbsRel_ablated(i+R) - AbsRel_baseline(i+R).

Strict hardware rule: torch.cuda.set_per_process_memory_fraction(0.85).
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dvlt.common.constants import DataField, PredictionField
from dvlt.common.rotation import so3_relative_angle
from dvlt.data.collate import default_collate_fn
from dvlt.data.datasets.multi_source import MultiSourceDataset
from dvlt.data.datasets.parser.dataverse import DataverseEvalDataset
from dvlt.metric.depth import apply_alignment
from dvlt.model.dvlt.model import DVLT, _slice_expand_flatten


def compute_metrics_for_predictions(preds, batch, device):
    """Compute depth AbsRel, RMSE and relative camera pose errors."""
    gt_depths = batch[DataField.DEPTHS][0]
    gt_masks = batch[DataField.POINT_MASKS][0]
    gt_c2w = batch[DataField.EXTRINSICS_C2W][0]
    pred_depths = preds[PredictionField.DEPTHS][0]
    pred_c2w_raw = preds[PredictionField.CAMERAS][0].camera_to_worlds
    S = pred_depths.shape[0]

    if pred_c2w_raw.shape[-2] == 3:
        homo = torch.tensor([0, 0, 0, 1], device=device, dtype=pred_c2w_raw.dtype).expand(S, 1, 4)
        pred_c2w = torch.cat([pred_c2w_raw, homo], dim=-2)
    else:
        pred_c2w = pred_c2w_raw

    if gt_c2w.shape[-2] == 3:
        homo_gt = torch.tensor([0, 0, 0, 1], device=device, dtype=gt_c2w.dtype).expand(S, 1, 4)
        gt_c2w = torch.cat([gt_c2w, homo_gt], dim=-2)

    rel_gt = torch.linalg.inv(gt_c2w[:1]).expand(S, -1, -1) @ gt_c2w
    rel_pred = torch.linalg.inv(pred_c2w[:1]).expand(S, -1, -1) @ pred_c2w

    absrel_list, rmse_list, rot_list, trans_list = [], [], [], []
    for v in range(S):
        mask_v = gt_masks[v] & (gt_depths[v] > 0)
        pred_d_aligned = apply_alignment(pred_depths[v], gt_depths[v], mask_v, align="median")

        if mask_v.sum() > 0:
            d_pred_val = pred_d_aligned[mask_v]
            d_gt_val = gt_depths[v][mask_v]
            abs_rel = (torch.abs(d_pred_val - d_gt_val) / d_gt_val).mean().item()
            rmse = torch.sqrt(torch.mean((d_pred_val - d_gt_val) ** 2)).item()
        else:
            abs_rel, rmse = 0.0, 0.0

        r_pred_v = rel_pred[v, :3, :3].unsqueeze(0)
        r_gt_v = rel_gt[v, :3, :3].unsqueeze(0)
        rot_err = so3_relative_angle(r_pred_v, r_gt_v).item() * (180.0 / np.pi)

        t_pred_v = rel_pred[v, :3, 3]
        t_gt_v = rel_gt[v, :3, 3]
        t_norm_pred = torch.norm(t_pred_v).clamp(min=1e-6)
        t_norm_gt = torch.norm(t_gt_v).clamp(min=1e-6)
        cos_sim = (torch.dot(t_pred_v, t_gt_v) / (t_norm_pred * t_norm_gt)).clamp(-1.0, 1.0)
        trans_err = torch.acos(cos_sim).item() * (180.0 / np.pi)

        absrel_list.append(abs_rel)
        rmse_list.append(rmse)
        rot_list.append(rot_err)
        trans_list.append(trans_err)

    return {
        "abs_rel": float(np.mean(absrel_list)),
        "rmse": float(np.mean(rmse_list)),
        "rot_err_deg": float(np.mean(rot_list)),
        "trans_err_deg": float(np.mean(trans_list)),
    }


def run_iteration_role_v01_recovery(
    data_root="datasets/test/dtu",
    output_dir="outputs",
    max_seqs=None,
):
    os.makedirs(output_dir, exist_ok=True)
    t_start_total = time.time()

    # Hardware safety cap
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    print("=" * 80)
    print("DVLT ITERATION-ROLE V0.1 — RECOVERY-HORIZON CONTROL")
    print(f"Device: {device} | Precision: bf16")
    print("CUDA memory fraction limit: 0.85 (Safety Cap)")
    print("=" * 80)

    val_scan_names = ["scan1", "scan4", "scan9", "scan24", "scan62"]
    subsets_def = [
        {"name": "subset_middle", "ranking": "middle_first", "sampling": "first"},
        {"name": "subset_uniform", "ranking": "index", "sampling": "uniform"},
    ]

    print("\n[1/4] Loading validation sequences (10 sequences, 60 views)...")
    cfg = OmegaConf.create({"target": "dtu.DTU", "params": {"root_path": data_root}})
    test_config = {
        "normalize_scene": False,
        "load_data_fields": ["images", "extrinsics_c2w", "intrinsics", "depths", "world_points", "point_masks"],
    }

    val_batches = {}
    for scan_name in val_scan_names:
        for sub in subsets_def:
            seq_id = f"{scan_name}_{sub['name']}"
            ds_val = DataverseEvalDataset(
                dataverse_cfg=cfg,
                view_ranking=sub["ranking"],
                view_sampling=sub["sampling"],
                max_frames=6,
            )
            ds_val.set_image_params(504, 14)
            video_idx = None
            for idx in range(ds_val.ds.num_videos()):
                if ds_val.ds._get_scan_name(idx) == scan_name:
                    video_idx = idx
                    break
            assert video_idx is not None, f"Scan {scan_name} not found!"

            ms_val = MultiSourceDataset({"dtu": ds_val}, training=False, **test_config)
            sample = ms_val[video_idx]
            batch = default_collate_fn([sample])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            val_batches[seq_id] = {
                "scan_name": scan_name,
                "subset_name": sub["name"],
                "batch": batch,
            }

    if max_seqs is not None and max_seqs < len(val_batches):
        val_batches = dict(list(val_batches.items())[:max_seqs])
        print(f"Limiting to first {max_seqs} sequences for smoke test.")
    print(f"Loaded {len(val_batches)} sequences successfully.")

    print("\n[2/4] Loading untouched pretrained NVIDIA DVLT...")
    base_model = DVLT(img_size=504, depth_head_type="conv")
    base_model.load_pretrained("nvidia/dvlt", strict=True)
    inner = base_model.model
    inner.eval()
    inner.to(device)
    recurrent_block = inner.recurrent_blocks[0]

    K = 16
    ts = torch.linspace(0.0, 1.0, K).tolist()
    horizons = [0, 1, 2, 4]
    print(f"Recurrent schedule: Uniform K={K}, Horizons evaluated: {horizons}")

    recovery_records = []
    seq_idx = 0
    total_seqs = len(val_batches)

    for seq_id, seq_info in val_batches.items():
        seq_idx += 1
        scan_name = seq_info["scan_name"]
        subset_name = seq_info["subset_name"]
        batch = seq_info["batch"]

        print(f"\n--- [{seq_idx}/{total_seqs}] Processing: {seq_id} ---")
        t0 = time.time()

        with torch.inference_mode(), accelerator.autocast():
            images = batch[DataField.IMAGES]
            B, S, _, H, W = images.shape

            # Initial embedding
            z_0 = inner._encode_images(images)
            reg_tok = inner.register_token.expand(B, S, -1, -1).reshape(B * S, inner.num_register_tokens, -1)
            cam_tok = _slice_expand_flatten(inner.camera_token, B, S)
            x_init = torch.cat([cam_tok, reg_tok, z_0], dim=1)
            rope_pos = inner._get_rope_positions(B * S, H, W, images.device)

            P = x_init.shape[1]
            C = x_init.shape[2]

            # ---------------------------------------------------------
            # 1. Untouched Baseline Trajectory: Cache h_i for i=0..16
            # ---------------------------------------------------------
            h_baseline = {0: x_init.clone()}
            x_curr = x_init.clone()
            for i in range(1, K + 1):
                t_now = ts[i - 1]
                t_next = ts[i] if i < K else 1.0
                t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)
                k_frame = t_pair.expand(B * S, -1)
                k_batch = t_pair.expand(B, -1)

                h_af = recurrent_block.frame_attn(x_curr, k_frame, pos=rope_pos.reshape(B * S, P, 2))
                x_g_in = h_af.reshape(B, S * P, C)
                x_g_out = recurrent_block.global_attn(x_g_in, k_batch, pos=None)
                x_curr = x_g_out.reshape(B * S, P, C)
                h_baseline[i] = x_curr.clone()

            # Decode baseline at every iteration 0..16
            baseline_metrics = {}
            for i in range(0, K + 1):
                raw_preds = inner._decode(h_baseline[i], H, W, B, S, rope_pos)
                preds = base_model._postprocess_predictions(batch, raw_preds)
                baseline_metrics[i] = compute_metrics_for_predictions(preds, batch, device)

            print(f"  Baseline AbsRel: i=1: {baseline_metrics[1]['abs_rel']:.6f} | i=8: {baseline_metrics[8]['abs_rel']:.6f} | i=16: {baseline_metrics[16]['abs_rel']:.6f}")

            # ---------------------------------------------------------
            # 2. Interventions with Controlled Recovery Horizons R={0,1,2,4}
            # ---------------------------------------------------------
            for i in range(1, K + 1):
                t_now = ts[i - 1]
                t_next = ts[i] if i < K else 1.0
                t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)
                k_frame = t_pair.expand(B * S, -1)
                k_batch = t_pair.expand(B, -1)

                # =====================================================
                # A. Skip Frame Attention at Iteration i
                # =====================================================
                # Start from untouched baseline state h_{i-1}
                x_f_in = h_baseline[i - 1].clone()
                # Skip frame: x_frame = x_f_in
                x_f_g_in = x_f_in.reshape(B, S * P, C)
                x_f_g_out = recurrent_block.global_attn(x_f_g_in, k_batch, pos=None)
                x_state = x_f_g_out.reshape(B * S, P, C)

                # Horizon R=0 (immediate effect at step i)
                raw_p = inner._decode(x_state, H, W, B, S, rope_pos)
                preds = base_model._postprocess_predictions(batch, raw_p)
                m_abl_r0 = compute_metrics_for_predictions(preds, batch, device)
                base_r0 = baseline_metrics[i]

                delta_absrel_r0 = m_abl_r0["abs_rel"] - base_r0["abs_rel"]
                delta_rmse_r0 = m_abl_r0["rmse"] - base_r0["rmse"]
                delta_rot_r0 = m_abl_r0["rot_err_deg"] - base_r0["rot_err_deg"]
                delta_trans_r0 = m_abl_r0["trans_err_deg"] - base_r0["trans_err_deg"]

                recovery_records.append({
                    "seq_id": seq_id,
                    "scan_name": scan_name,
                    "subset_name": subset_name,
                    "iteration": i,
                    "intervention": "skip_frame",
                    "recovery_horizon_R": 0,
                    "eval_iteration": i,
                    "ablated_absrel": m_abl_r0["abs_rel"],
                    "baseline_absrel": base_r0["abs_rel"],
                    "delta_absrel": delta_absrel_r0,
                    "delta_rmse": delta_rmse_r0,
                    "delta_rot": delta_rot_r0,
                    "delta_trans": delta_trans_r0,
                    "immediate_effect_E0": delta_absrel_r0,
                    "recovery_fraction": 0.0,
                })

                # Now step downstream forward through recovery horizons R=1, 2, 3, 4
                curr_step_idx = i
                for r_step in range(1, 5):
                    target_step = i + r_step
                    if target_step > K:
                        break

                    tj_now = ts[target_step - 1]
                    tj_next = ts[target_step] if target_step < K else 1.0
                    x_state = inner._interval_step(x_state, tj_now, tj_next, rope_pos, B, S)

                    if r_step in [1, 2, 4]:
                        raw_p = inner._decode(x_state, H, W, B, S, rope_pos)
                        preds = base_model._postprocess_predictions(batch, raw_p)
                        m_abl_r = compute_metrics_for_predictions(preds, batch, device)
                        base_r = baseline_metrics[target_step]

                        delta_absrel_r = m_abl_r["abs_rel"] - base_r["abs_rel"]
                        delta_rmse_r = m_abl_r["rmse"] - base_r["rmse"]
                        delta_rot_r = m_abl_r["rot_err_deg"] - base_r["rot_err_deg"]
                        delta_trans_r = m_abl_r["trans_err_deg"] - base_r["trans_err_deg"]

                        # Recovery fraction (only if |E0| > 1e-5 to avoid division by near-zero)
                        if abs(delta_absrel_r0) >= 1e-5:
                            rec_frac = 1.0 - (abs(delta_absrel_r) / abs(delta_absrel_r0))
                        else:
                            rec_frac = np.nan

                        recovery_records.append({
                            "seq_id": seq_id,
                            "scan_name": scan_name,
                            "subset_name": subset_name,
                            "iteration": i,
                            "intervention": "skip_frame",
                            "recovery_horizon_R": r_step,
                            "eval_iteration": target_step,
                            "ablated_absrel": m_abl_r["abs_rel"],
                            "baseline_absrel": base_r["abs_rel"],
                            "delta_absrel": delta_absrel_r,
                            "delta_rmse": delta_rmse_r,
                            "delta_rot": delta_rot_r,
                            "delta_trans": delta_trans_r,
                            "immediate_effect_E0": delta_absrel_r0,
                            "recovery_fraction": rec_frac,
                        })

                # =====================================================
                # B. Skip Global Attention at Iteration i
                # =====================================================
                # Start from untouched baseline state h_{i-1}
                x_g_in = h_baseline[i - 1].clone()
                # Run frame attention:
                x_g_frame = recurrent_block.frame_attn(x_g_in, k_frame, pos=rope_pos.reshape(B * S, P, 2))
                # Skip global attention: x_state = x_g_frame
                x_state = x_g_frame

                # Horizon R=0 (immediate effect at step i)
                raw_p = inner._decode(x_state, H, W, B, S, rope_pos)
                preds = base_model._postprocess_predictions(batch, raw_p)
                m_abl_r0 = compute_metrics_for_predictions(preds, batch, device)
                base_r0 = baseline_metrics[i]

                delta_absrel_r0 = m_abl_r0["abs_rel"] - base_r0["abs_rel"]
                delta_rmse_r0 = m_abl_r0["rmse"] - base_r0["rmse"]
                delta_rot_r0 = m_abl_r0["rot_err_deg"] - base_r0["rot_err_deg"]
                delta_trans_r0 = m_abl_r0["trans_err_deg"] - base_r0["trans_err_deg"]

                recovery_records.append({
                    "seq_id": seq_id,
                    "scan_name": scan_name,
                    "subset_name": subset_name,
                    "iteration": i,
                    "intervention": "skip_global",
                    "recovery_horizon_R": 0,
                    "eval_iteration": i,
                    "ablated_absrel": m_abl_r0["abs_rel"],
                    "baseline_absrel": base_r0["abs_rel"],
                    "delta_absrel": delta_absrel_r0,
                    "delta_rmse": delta_rmse_r0,
                    "delta_rot": delta_rot_r0,
                    "delta_trans": delta_trans_r0,
                    "immediate_effect_E0": delta_absrel_r0,
                    "recovery_fraction": 0.0,
                })

                # Now step downstream forward through recovery horizons R=1, 2, 3, 4
                for r_step in range(1, 5):
                    target_step = i + r_step
                    if target_step > K:
                        break

                    tj_now = ts[target_step - 1]
                    tj_next = ts[target_step] if target_step < K else 1.0
                    x_state = inner._interval_step(x_state, tj_now, tj_next, rope_pos, B, S)

                    if r_step in [1, 2, 4]:
                        raw_p = inner._decode(x_state, H, W, B, S, rope_pos)
                        preds = base_model._postprocess_predictions(batch, raw_p)
                        m_abl_r = compute_metrics_for_predictions(preds, batch, device)
                        base_r = baseline_metrics[target_step]

                        delta_absrel_r = m_abl_r["abs_rel"] - base_r["abs_rel"]
                        delta_rmse_r = m_abl_r["rmse"] - base_r["rmse"]
                        delta_rot_r = m_abl_r["rot_err_deg"] - base_r["rot_err_deg"]
                        delta_trans_r = m_abl_r["trans_err_deg"] - base_r["trans_err_deg"]

                        # Recovery fraction
                        if abs(delta_absrel_r0) >= 1e-5:
                            rec_frac = 1.0 - (abs(delta_absrel_r) / abs(delta_absrel_r0))
                        else:
                            rec_frac = np.nan

                        recovery_records.append({
                            "seq_id": seq_id,
                            "scan_name": scan_name,
                            "subset_name": subset_name,
                            "iteration": i,
                            "intervention": "skip_global",
                            "recovery_horizon_R": r_step,
                            "eval_iteration": target_step,
                            "ablated_absrel": m_abl_r["abs_rel"],
                            "baseline_absrel": base_r["abs_rel"],
                            "delta_absrel": delta_absrel_r,
                            "delta_rmse": delta_rmse_r,
                            "delta_rot": delta_rot_r,
                            "delta_trans": delta_trans_r,
                            "immediate_effect_E0": delta_absrel_r0,
                            "recovery_fraction": rec_frac,
                        })

        seq_elapsed = time.time() - t0
        print(f"  Seq {seq_id} done in {seq_elapsed:.2f}s | Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**2):.1f} MB")

    total_elapsed = time.time() - t_start_total
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print("\n" + "=" * 80)
    print("V0.1 RECOVERY AUDIT COMPLETED!")
    print(f"Total sequences evaluated: {total_seqs}")
    print(f"Total elapsed runtime: {total_elapsed:.2f}s ({total_elapsed / 60.0:.2f} min)")
    print(f"Peak VRAM: {peak_vram_mb:.1f} MB")
    print("=" * 80)

    # Convert to DataFrame
    df_rec = pd.DataFrame(recovery_records)
    rec_csv_path = os.path.join(output_dir, "iteration_role_v01_recovery.csv")
    df_rec.to_csv(rec_csv_path, index=False)
    print(f"Saved: {rec_csv_path}")

    # =========================================================
    # Aggregations & Scene Summaries
    # =========================================================
    print("\nGenerating Scene Summaries...")
    scene_summary_records = []
    scenes = sorted(df_rec["scan_name"].unique())

    # Build per-scene summary across horizons and interventions
    for sc in scenes:
        df_sc = df_rec[df_rec["scan_name"] == sc]
        for interv in ["skip_frame", "skip_global"]:
            for r in [0, 1, 2, 4]:
                sub = df_sc[(df_sc["intervention"] == interv) & (df_sc["recovery_horizon_R"] == r)]
                # Split by iteration range: early (1-5), middle (6-11), late (12-16)
                for phase_name, (it_min, it_max) in [("Early (1-5)", (1, 5)), ("Middle (6-11)", (6, 11)), ("Late (12-16)", (12, 16))]:
                    sub_p = sub[(sub["iteration"] >= it_min) & (sub["iteration"] <= it_max)]
                    if len(sub_p) > 0:
                        scene_summary_records.append({
                            "scan_name": sc,
                            "intervention": interv,
                            "recovery_horizon_R": r,
                            "phase": phase_name,
                            "num_measurements": len(sub_p),
                            "delta_absrel_mean": sub_p["delta_absrel"].mean(),
                            "delta_absrel_std": sub_p["delta_absrel"].std(),
                            "delta_rmse_mean": sub_p["delta_rmse"].mean(),
                            "recovery_fraction_mean": sub_p["recovery_fraction"].dropna().mean(),
                        })

    # Overall pooled summary
    for interv in ["skip_frame", "skip_global"]:
        for r in [0, 1, 2, 4]:
            sub = df_rec[(df_rec["intervention"] == interv) & (df_rec["recovery_horizon_R"] == r)]
            for phase_name, (it_min, it_max) in [("Early (1-5)", (1, 5)), ("Middle (6-11)", (6, 11)), ("Late (12-16)", (12, 16))]:
                sub_p = sub[(sub["iteration"] >= it_min) & (sub["iteration"] <= it_max)]
                if len(sub_p) > 0:
                    scene_summary_records.append({
                        "scan_name": "OVERALL_ALL_SCENES",
                        "intervention": interv,
                        "recovery_horizon_R": r,
                        "phase": phase_name,
                        "num_measurements": len(sub_p),
                        "delta_absrel_mean": sub_p["delta_absrel"].mean(),
                        "delta_absrel_std": sub_p["delta_absrel"].std(),
                        "delta_rmse_mean": sub_p["delta_rmse"].mean(),
                        "recovery_fraction_mean": sub_p["recovery_fraction"].dropna().mean(),
                    })

    df_scene_summary = pd.DataFrame(scene_summary_records)
    scene_summary_csv = os.path.join(output_dir, "iteration_role_v01_scene_summary.csv")
    df_scene_summary.to_csv(scene_summary_csv, index=False)
    print(f"Saved: {scene_summary_csv}")

    # =========================================================
    # PLOTTING
    # =========================================================
    print("\nGenerating plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Immediate Effect (R=0)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    df_r0_f = df_rec[(df_rec["intervention"] == "skip_frame") & (df_rec["recovery_horizon_R"] == 0)].groupby("iteration")["delta_absrel"].agg(["mean", "std"]).reset_index()
    df_r0_g = df_rec[(df_rec["intervention"] == "skip_global") & (df_rec["recovery_horizon_R"] == 0)].groupby("iteration")["delta_absrel"].agg(["mean", "std"]).reset_index()

    ax.plot(df_r0_f["iteration"], df_r0_f["mean"], marker="o", color="#2563eb", linewidth=2.5, label="Skip Frame: Immediate Effect ($E_0$, $R=0$)")
    ax.fill_between(df_r0_f["iteration"], df_r0_f["mean"] - df_r0_f["std"], df_r0_f["mean"] + df_r0_f["std"], color="#2563eb", alpha=0.15)

    ax.plot(df_r0_g["iteration"], df_r0_g["mean"], marker="s", color="#dc2626", linewidth=2.5, label="Skip Global: Immediate Effect ($E_0$, $R=0$)")
    ax.fill_between(df_r0_g["iteration"], df_r0_g["mean"] - df_r0_g["std"], df_r0_g["mean"] + df_r0_g["std"], color="#dc2626", alpha=0.15)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Iteration $i$", fontsize=12, fontweight="bold")
    ax.set_ylabel("Immediate Metric Degradation $\\Delta \\text{AbsRel}(i, 0)$", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 17))
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.axvspan(0.5, 5.5, color="#e0f2fe", alpha=0.3, label="Early Phase (1-5)")
    ax.axvspan(5.5, 11.5, color="#fef3c7", alpha=0.3, label="Middle Phase (6-11)")
    ax.axvspan(11.5, 16.5, color="#f1f5f9", alpha=0.3, label="Late Phase (12-16)")

    plt.title("Immediate Intrinsic Effect ($R=0$): Frame vs Global Attention", fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "iteration_role_v01_immediate_effect.png")
    plt.savefig(plot1_path, dpi=200)
    plt.close()

    # Plot 2: Fixed Horizon R=2 (i <= 14)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    df_r2_f = df_rec[(df_rec["intervention"] == "skip_frame") & (df_rec["recovery_horizon_R"] == 2)].groupby("iteration")["delta_absrel"].agg(["mean", "std"]).reset_index()
    df_r2_g = df_rec[(df_rec["intervention"] == "skip_global") & (df_rec["recovery_horizon_R"] == 2)].groupby("iteration")["delta_absrel"].agg(["mean", "std"]).reset_index()

    ax.plot(df_r2_f["iteration"], df_r2_f["mean"], marker="o", color="#2563eb", linewidth=2.5, label="Skip Frame ($R=2$ steps)")
    ax.fill_between(df_r2_f["iteration"], df_r2_f["mean"] - df_r2_f["std"], df_r2_f["mean"] + df_r2_f["std"], color="#2563eb", alpha=0.15)

    ax.plot(df_r2_g["iteration"], df_r2_g["mean"], marker="s", color="#dc2626", linewidth=2.5, label="Skip Global ($R=2$ steps)")
    ax.fill_between(df_r2_g["iteration"], df_r2_g["mean"] - df_r2_g["std"], df_r2_g["mean"] + df_r2_g["std"], color="#dc2626", alpha=0.15)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Iteration i (where i <= 14)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Degradation after 2 Recovery Steps $\\Delta \\text{AbsRel}(i, 2)$", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 15))
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.axvspan(0.5, 5.5, color="#e0f2fe", alpha=0.3, label="Early Phase (1-5)")
    ax.axvspan(5.5, 11.5, color="#fef3c7", alpha=0.3, label="Middle Phase (6-11)")
    ax.axvspan(11.5, 14.5, color="#f1f5f9", alpha=0.3, label="Late Phase (12-14)")

    plt.title("Controlled Horizon Comparison at Fixed $R=2$ Recovery Steps", fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "iteration_role_v01_fixed_R2.png")
    plt.savefig(plot2_path, dpi=200)
    plt.close()

    # Plot 3: Fixed Horizon R=4 (i <= 12)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    df_r4_f = df_rec[(df_rec["intervention"] == "skip_frame") & (df_rec["recovery_horizon_R"] == 4)].groupby("iteration")["delta_absrel"].agg(["mean", "std"]).reset_index()
    df_r4_g = df_rec[(df_rec["intervention"] == "skip_global") & (df_rec["recovery_horizon_R"] == 4)].groupby("iteration")["delta_absrel"].agg(["mean", "std"]).reset_index()

    ax.plot(df_r4_f["iteration"], df_r4_f["mean"], marker="o", color="#2563eb", linewidth=2.5, label="Skip Frame ($R=4$ steps)")
    ax.fill_between(df_r4_f["iteration"], df_r4_f["mean"] - df_r4_f["std"], df_r4_f["mean"] + df_r4_f["std"], color="#2563eb", alpha=0.15)

    ax.plot(df_r4_g["iteration"], df_r4_g["mean"], marker="s", color="#dc2626", linewidth=2.5, label="Skip Global ($R=4$ steps)")
    ax.fill_between(df_r4_g["iteration"], df_r4_g["mean"] - df_r4_g["std"], df_r4_g["mean"] + df_r4_g["std"], color="#dc2626", alpha=0.15)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Iteration i (where i <= 12)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Degradation after 4 Recovery Steps $\\Delta \\text{AbsRel}(i, 4)$", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 13))
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.axvspan(0.5, 5.5, color="#e0f2fe", alpha=0.3, label="Early Phase (1-5)")
    ax.axvspan(5.5, 11.5, color="#fef3c7", alpha=0.3, label="Middle Phase (6-11)")
    ax.axvspan(11.5, 12.5, color="#f1f5f9", alpha=0.3, label="Late Phase (12)")

    plt.title("Controlled Horizon Comparison at Fixed $R=4$ Recovery Steps", fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "iteration_role_v01_fixed_R4.png")
    plt.savefig(plot3_path, dpi=200)
    plt.close()

    # Plot 4: Recovery Curves across R={0, 1, 2, 4} for selected representative iterations
    fig, (ax_rf, ax_rg) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    rep_iters = [2, 6, 10, 12]
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]

    for it, c in zip(rep_iters, colors):
        sub_f = df_rec[(df_rec["intervention"] == "skip_frame") & (df_rec["iteration"] == it)].groupby("recovery_horizon_R")["delta_absrel"].mean().reset_index()
        sub_g = df_rec[(df_rec["intervention"] == "skip_global") & (df_rec["iteration"] == it)].groupby("recovery_horizon_R")["delta_absrel"].mean().reset_index()

        ax_rf.plot(sub_f["recovery_horizon_R"], sub_f["delta_absrel"], marker="o", linewidth=2.2, color=c, label=f"Iteration $i={it}$")
        ax_rg.plot(sub_g["recovery_horizon_R"], sub_g["delta_absrel"], marker="s", linewidth=2.2, color=c, label=f"Iteration $i={it}$")

    ax_rf.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_rg.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax_rf.set_title("Skip Frame: Degradation vs Recovery Horizon $R$", fontsize=12, fontweight="bold")
    ax_rf.set_xlabel("Recovery Horizon $R$ (steps)", fontsize=11, fontweight="bold")
    ax_rf.set_ylabel("$\\Delta \\text{AbsRel}(i, R)$", fontsize=11, fontweight="bold")
    ax_rf.set_xticks([0, 1, 2, 4])
    ax_rf.legend(frameon=True)
    ax_rf.grid(True, linestyle="--", alpha=0.5)

    ax_rg.set_title("Skip Global: Degradation vs Recovery Horizon $R$", fontsize=12, fontweight="bold")
    ax_rg.set_xlabel("Recovery Horizon $R$ (steps)", fontsize=11, fontweight="bold")
    ax_rg.set_xticks([0, 1, 2, 4])
    ax_rg.legend(frameon=True)
    ax_rg.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Downstream Recovery Trajectories Across Horizons $R \\in \\{0, 1, 2, 4\\}$", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot4_path = os.path.join(output_dir, "iteration_role_v01_recovery_curves.png")
    plt.savefig(plot4_path, dpi=200)
    plt.close()

    print(f"Generated plots:")
    print(f"  - {plot1_path}")
    print(f"  - {plot2_path}")
    print(f"  - {plot3_path}")
    print(f"  - {plot4_path}")

    return {
        "total_elapsed_sec": total_elapsed,
        "peak_vram_mb": peak_vram_mb,
        "total_seqs": total_seqs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVLT Iteration-Role V0.1 Recovery-Horizon Control")
    parser.add_argument("--data_root", type=str, default="datasets/test/dtu")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_seqs", type=int, default=None, help="Limit number of sequences for smoke test")
    args = parser.parse_args()

    run_iteration_role_v01_recovery(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_seqs=args.max_seqs,
    )
