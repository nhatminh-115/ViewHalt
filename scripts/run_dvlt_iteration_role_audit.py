"""DVLT Iteration-Role Audit V0 — Cheap Mechanistic Diagnostic.

Investigates what the shared DVLT recurrent block is doing at early, middle, and late iterations:
1. Dense trajectory audit: frame, global, and total update magnitudes, cosine similarities.
2. Decode-at-every-iteration curve: AbsRel, RMSE, camera rotation/translation error and marginal gains.
3. Causal single-iteration ablations: Skip Frame at step i vs. Skip Global at step i.
4. Descriptive phase analysis (Early 1-5, Middle 6-11, Late 12-16) and physical scene consistency.

No training, no adapters, strictly diagnostic.
"""

import os
import sys
import time
import math
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
        "per_view_absrel": absrel_list,
        "per_view_rmse": rmse_list,
        "per_view_rot": rot_list,
        "per_view_trans": trans_list,
    }


def run_iteration_role_audit(
    data_root="datasets/test/dtu",
    output_dir="outputs",
    max_seqs=None,
):
    os.makedirs(output_dir, exist_ok=True)
    t_start_total = time.time()

    # Hardware constraint
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    print("=" * 80)
    print("DVLT ITERATION-ROLE AUDIT V0 — CHEAP MECHANISTIC DIAGNOSTIC")
    print(f"Device: {device} | Precision: bf16")
    print(f"CUDA memory fraction limit: 0.85")
    print("=" * 80)

    val_scan_names = ["scan1", "scan4", "scan9", "scan24", "scan62"]
    subsets_def = [
        {"name": "subset_middle", "ranking": "middle_first", "sampling": "first"},
        {"name": "subset_uniform", "ranking": "index", "sampling": "uniform"},
    ]

    print("\n[1/5] Loading validation dataset (10 sequences, 60 views)...")
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
    print(f"Successfully loaded {len(val_batches)} sequences.")

    print("\n[2/5] Loading untouched pretrained NVIDIA DVLT...")
    base_model = DVLT(img_size=504, depth_head_type="conv")
    base_model.load_pretrained("nvidia/dvlt", strict=True)
    inner = base_model.model
    inner.eval()
    inner.to(device)
    recurrent_block = inner.recurrent_blocks[0]

    K = 16
    ts = torch.linspace(0.0, 1.0, K).tolist()
    print(f"Recurrent schedule: Uniform K={K}, ts: {ts}")

    # Data structures to collect results
    dense_curve_records = []     # decoded metrics at step 0..16
    update_norms_records = []    # frame/global/total update norms & cosine similarities
    causal_ablation_records = [] # frame skip and global skip causal degradation

    seq_idx = 0
    total_seqs = len(val_batches)

    # First test 1 sequence to estimate timing
    t_seq_start = time.time()

    for seq_id, seq_info in val_batches.items():
        seq_idx += 1
        scan_name = seq_info["scan_name"]
        subset_name = seq_info["subset_name"]
        batch = seq_info["batch"]

        print(f"\n--- [{seq_idx}/{total_seqs}] Processing sequence: {seq_id} ---")
        t0 = time.time()

        with torch.no_grad(), accelerator.autocast():
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

            # Store hidden states: h[0] = x_init, h[1..16]
            h_states = {0: x_init.clone()}
            h_before_frames = {}
            h_after_frames = {}
            h_after_globals = {}

            delta_frames = {}
            delta_globals = {}
            delta_steps = {}

            # Execute dense K=16 forward pass
            x_curr = x_init.clone()
            for i in range(1, K + 1):
                t_now = ts[i - 1]
                t_next = ts[i] if i < K else 1.0
                t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)
                k_frame = t_pair.expand(B * S, -1)
                k_batch = t_pair.expand(B, -1)

                h_bf = x_curr.clone()
                h_before_frames[i] = h_bf

                # Frame attention
                h_af = recurrent_block.frame_attn(h_bf, k_frame, pos=rope_pos.reshape(B * S, P, 2))
                h_after_frames[i] = h_af

                # Global attention
                x_g_in = h_af.reshape(B, S * P, C)
                x_g_out = recurrent_block.global_attn(x_g_in, k_batch, pos=None)
                h_ag = x_g_out.reshape(B * S, P, C)
                h_after_globals[i] = h_ag

                x_curr = h_ag
                h_states[i] = x_curr.clone()

                d_frame = h_af - h_bf
                d_global = h_ag - h_af
                d_step = h_ag - h_bf

                delta_frames[i] = d_frame
                d_global = h_ag - h_af
                d_step = h_ag - h_bf

                delta_frames[i] = d_frame
                delta_globals[i] = d_global
                delta_steps[i] = d_step

            # =========================================================
            # SECTION 2: Dense Trajectory Update Norms & Cosine Similarities
            # =========================================================
            h_terminal = h_states[K]

            for i in range(1, K + 1):
                h_bf = h_before_frames[i]
                h_af = h_after_frames[i]
                h_ag = h_after_globals[i]
                h_i = h_states[i]

                d_frame = delta_frames[i]
                d_global = delta_globals[i]
                d_step = delta_steps[i]

                # Update magnitudes
                mag_frame = (torch.norm(d_frame.float()) / torch.norm(h_bf.float()).clamp(min=1e-6)).item()
                mag_global = (torch.norm(d_global.float()) / torch.norm(h_af.float()).clamp(min=1e-6)).item()
                mag_step = (torch.norm(d_step.float()) / torch.norm(h_bf.float()).clamp(min=1e-6)).item()

                # Cosine similarity h_i vs h_16
                cos_h_terminal_flat = F.cosine_similarity(h_i.flatten().float(), h_terminal.flatten().float(), dim=0).item()
                cos_h_terminal_tok = F.cosine_similarity(h_i.float(), h_terminal.float(), dim=-1).mean().item()

                # Consecutive directional alignment
                if i < K:
                    d_frame_next = delta_frames[i + 1]
                    d_global_next = delta_globals[i + 1]
                    cos_frame_consec = F.cosine_similarity(d_frame.flatten().float(), d_frame_next.flatten().float(), dim=0).item()
                    cos_global_consec = F.cosine_similarity(d_global.flatten().float(), d_global_next.flatten().float(), dim=0).item()
                else:
                    cos_frame_consec = np.nan
                    cos_global_consec = np.nan

                update_norms_records.append({
                    "seq_id": seq_id,
                    "scan_name": scan_name,
                    "subset_name": subset_name,
                    "iteration": i,
                    "mag_frame": mag_frame,
                    "mag_global": mag_global,
                    "mag_step": mag_step,
                    "cos_to_terminal_flat": cos_h_terminal_flat,
                    "cos_to_terminal_token_mean": cos_h_terminal_tok,
                    "cos_frame_consecutive": cos_frame_consec,
                    "cos_global_consecutive": cos_global_consec,
                })

            # =========================================================
            # SECTION 3: Decode at Every Iteration (0..16)
            # =========================================================
            step_metrics = {}
            for i in range(0, K + 1):
                feat = h_states[i]
                raw_preds = inner._decode(feat, H, W, B, S, rope_pos)
                preds = base_model._postprocess_predictions(batch, raw_preds)
                m = compute_metrics_for_predictions(preds, batch, device)
                step_metrics[i] = m

            for i in range(0, K + 1):
                m = step_metrics[i]
                if i > 0:
                    marginal_absrel_gain = step_metrics[i - 1]["abs_rel"] - m["abs_rel"]
                else:
                    marginal_absrel_gain = 0.0

                dense_curve_records.append({
                    "seq_id": seq_id,
                    "scan_name": scan_name,
                    "subset_name": subset_name,
                    "iteration": i,
                    "abs_rel": m["abs_rel"],
                    "rmse": m["rmse"],
                    "rot_err_deg": m["rot_err_deg"],
                    "trans_err_deg": m["trans_err_deg"],
                    "marginal_absrel_gain": marginal_absrel_gain,
                })

            baseline_k16 = step_metrics[K]
            print(f"  K=16 Untouched Baseline: AbsRel={baseline_k16['abs_rel']:.6f} | RMSE={baseline_k16['rmse']:.6f} | Rot={baseline_k16['rot_err_deg']:.3f} deg")

            # =========================================================
            # SECTION 4 & 5: Causal Single-Iteration Ablations (1..16)
            # =========================================================
            # We perform full 16-step sweep for completeness
            for i in range(1, K + 1):
                # -----------------------------------------------------
                # A. Skip Frame at iteration i
                # -----------------------------------------------------
                # Start from untouched h_{i-1}
                x_abl = h_states[i - 1].clone()

                # Step i with frame attention skipped: x_frame = x_abl
                t_now = ts[i - 1]
                t_next = ts[i] if i < K else 1.0
                t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)
                k_batch = t_pair.expand(B, -1)

                # Skip frame: x_frame = x_abl
                x_g_in = x_abl.reshape(B, S * P, C)
                x_g_out = recurrent_block.global_attn(x_g_in, k_batch, pos=None)
                x_abl = x_g_out.reshape(B * S, P, C)

                # Continue normal remaining steps i+1..16
                for j in range(i + 1, K + 1):
                    tj_now = ts[j - 1]
                    tj_next = ts[j] if j < K else 1.0
                    x_abl = inner._interval_step(x_abl, tj_now, tj_next, rope_pos, B, S)

                raw_preds = inner._decode(x_abl, H, W, B, S, rope_pos)
                preds = base_model._postprocess_predictions(batch, raw_preds)
                m_frame_skip = compute_metrics_for_predictions(preds, batch, device)

                delta_absrel_f = m_frame_skip["abs_rel"] - baseline_k16["abs_rel"]
                delta_rmse_f = m_frame_skip["rmse"] - baseline_k16["rmse"]
                delta_rot_f = m_frame_skip["rot_err_deg"] - baseline_k16["rot_err_deg"]
                delta_trans_f = m_frame_skip["trans_err_deg"] - baseline_k16["trans_err_deg"]

                causal_ablation_records.append({
                    "seq_id": seq_id,
                    "scan_name": scan_name,
                    "subset_name": subset_name,
                    "iteration": i,
                    "ablation_type": "skip_frame",
                    "final_absrel": m_frame_skip["abs_rel"],
                    "final_rmse": m_frame_skip["rmse"],
                    "final_rot_err_deg": m_frame_skip["rot_err_deg"],
                    "final_trans_err_deg": m_frame_skip["trans_err_deg"],
                    "delta_absrel": delta_absrel_f,
                    "delta_rmse": delta_rmse_f,
                    "delta_rot": delta_rot_f,
                    "delta_trans": delta_trans_f,
                })

                # -----------------------------------------------------
                # B. Skip Global at iteration i
                # -----------------------------------------------------
                x_abl = h_states[i - 1].clone()

                # Step i with global attention skipped: x_ag = x_frame
                k_frame = t_pair.expand(B * S, -1)
                x_frame = recurrent_block.frame_attn(x_abl, k_frame, pos=rope_pos.reshape(B * S, P, 2))
                x_abl = x_frame  # skip global attention

                # Continue normal remaining steps i+1..16
                for j in range(i + 1, K + 1):
                    tj_now = ts[j - 1]
                    tj_next = ts[j] if j < K else 1.0
                    x_abl = inner._interval_step(x_abl, tj_now, tj_next, rope_pos, B, S)

                raw_preds = inner._decode(x_abl, H, W, B, S, rope_pos)
                preds = base_model._postprocess_predictions(batch, raw_preds)
                m_global_skip = compute_metrics_for_predictions(preds, batch, device)

                delta_absrel_g = m_global_skip["abs_rel"] - baseline_k16["abs_rel"]
                delta_rmse_g = m_global_skip["rmse"] - baseline_k16["rmse"]
                delta_rot_g = m_global_skip["rot_err_deg"] - baseline_k16["rot_err_deg"]
                delta_trans_g = m_global_skip["trans_err_deg"] - baseline_k16["trans_err_deg"]

                causal_ablation_records.append({
                    "seq_id": seq_id,
                    "scan_name": scan_name,
                    "subset_name": subset_name,
                    "iteration": i,
                    "ablation_type": "skip_global",
                    "final_absrel": m_global_skip["abs_rel"],
                    "final_rmse": m_global_skip["rmse"],
                    "final_rot_err_deg": m_global_skip["rot_err_deg"],
                    "final_trans_err_deg": m_global_skip["trans_err_deg"],
                    "delta_absrel": delta_absrel_g,
                    "delta_rmse": delta_rmse_g,
                    "delta_rot": delta_rot_g,
                    "delta_trans": delta_trans_g,
                })

        seq_elapsed = time.time() - t0
        print(f"  Seq {seq_id} completed in {seq_elapsed:.2f}s | Peak VRAM: {torch.cuda.max_memory_allocated() / (1024 ** 2):.1f} MB")

    total_elapsed = time.time() - t_start_total
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print("\n" + "=" * 80)
    print(f"AUDIT EXECUTION COMPLETE!")
    print(f"Total sequences evaluated: {total_seqs}")
    print(f"Total elapsed time: {total_elapsed:.2f}s ({total_elapsed / 60.0:.2f} min)")
    print(f"Peak VRAM: {peak_vram_mb:.1f} MB")
    print("=" * 80)

    # Convert to DataFrames
    df_dense = pd.DataFrame(dense_curve_records)
    df_norms = pd.DataFrame(update_norms_records)
    df_causal = pd.DataFrame(causal_ablation_records)

    # Save raw CSVs
    dense_csv_path = os.path.join(output_dir, "iteration_role_dense_curve.csv")
    norms_csv_path = os.path.join(output_dir, "iteration_role_update_norms.csv")
    causal_csv_path = os.path.join(output_dir, "iteration_role_causal_ablation.csv")

    df_dense.to_csv(dense_csv_path, index=False)
    df_norms.to_csv(norms_csv_path, index=False)
    df_causal.to_csv(causal_csv_path, index=False)

    print(f"\nSaved:")
    print(f"  - {dense_csv_path}")
    print(f"  - {norms_csv_path}")
    print(f"  - {causal_csv_path}")

    # =========================================================
    # Aggregation & Scene Summary (SECTION 7 & 8)
    # =========================================================
    print("\nComputing Phase Analysis and Scene Summaries...")

    # Define phases: Early (1-5), Middle (6-11), Late (12-16)
    def assign_phase(it):
        if 1 <= it <= 5:
            return "Early (1-5)"
        elif 6 <= it <= 11:
            return "Middle (6-11)"
        elif 12 <= it <= 16:
            return "Late (12-16)"
        return "Initial (0)"

    df_dense["phase"] = df_dense["iteration"].apply(assign_phase)
    df_norms["phase"] = df_norms["iteration"].apply(assign_phase)
    df_causal["phase"] = df_causal["iteration"].apply(assign_phase)

    # Scene summary table
    scene_summary_records = []
    scenes = sorted(df_causal["scan_name"].unique())

    for sc in scenes:
        df_c_sc = df_causal[df_causal["scan_name"] == sc]
        df_n_sc = df_norms[df_norms["scan_name"] == sc]
        df_d_sc = df_dense[df_dense["scan_name"] == sc]

        for phase in ["Early (1-5)", "Middle (6-11)", "Late (12-16)"]:
            c_f = df_c_sc[(df_c_sc["phase"] == phase) & (df_c_sc["ablation_type"] == "skip_frame")]["delta_absrel"]
            c_g = df_c_sc[(df_c_sc["phase"] == phase) & (df_c_sc["ablation_type"] == "skip_global")]["delta_absrel"]
            n_f = df_n_sc[df_n_sc["phase"] == phase]["mag_frame"]
            n_g = df_n_sc[df_n_sc["phase"] == phase]["mag_global"]
            n_tot = df_n_sc[df_n_sc["phase"] == phase]["mag_step"]
            g_m = df_d_sc[df_d_sc["phase"] == phase]["marginal_absrel_gain"]

            scene_summary_records.append({
                "scan_name": sc,
                "phase": phase,
                "frame_update_mag_mean": n_f.mean(),
                "frame_update_mag_std": n_f.std(),
                "global_update_mag_mean": n_g.mean(),
                "global_update_mag_std": n_g.std(),
                "total_update_mag_mean": n_tot.mean(),
                "marginal_gain_absrel_mean": g_m.mean(),
                "marginal_gain_absrel_sum": g_m.sum(),
                "frame_skip_delta_absrel_mean": c_f.mean(),
                "frame_skip_delta_absrel_std": c_f.std(),
                "global_skip_delta_absrel_mean": c_g.mean(),
                "global_skip_delta_absrel_std": c_g.std(),
            })

    # Add overall pooled summary across scenes
    for phase in ["Early (1-5)", "Middle (6-11)", "Late (12-16)"]:
        c_f = df_causal[(df_causal["phase"] == phase) & (df_causal["ablation_type"] == "skip_frame")]["delta_absrel"]
        c_g = df_causal[(df_causal["phase"] == phase) & (df_causal["ablation_type"] == "skip_global")]["delta_absrel"]
        n_f = df_norms[df_norms["phase"] == phase]["mag_frame"]
        n_g = df_norms[df_norms["phase"] == phase]["mag_global"]
        n_tot = df_norms[df_norms["phase"] == phase]["mag_step"]
        g_m = df_dense[df_dense["phase"] == phase]["marginal_absrel_gain"]

        scene_summary_records.append({
            "scan_name": "OVERALL_ALL_SCENES",
            "phase": phase,
            "frame_update_mag_mean": n_f.mean(),
            "frame_update_mag_std": n_f.std(),
            "global_update_mag_mean": n_g.mean(),
            "global_update_mag_std": n_g.std(),
            "total_update_mag_mean": n_tot.mean(),
            "marginal_gain_absrel_mean": g_m.mean(),
            "marginal_gain_absrel_sum": g_m.sum(),
            "frame_skip_delta_absrel_mean": c_f.mean(),
            "frame_skip_delta_absrel_std": c_f.std(),
            "global_skip_delta_absrel_mean": c_g.mean(),
            "global_skip_delta_absrel_std": c_g.std(),
        })

    df_scene_summary = pd.DataFrame(scene_summary_records)
    scene_summary_csv = os.path.join(output_dir, "iteration_role_scene_summary.csv")
    df_scene_summary.to_csv(scene_summary_csv, index=False)
    print(f"  - {scene_summary_csv}")

    # =========================================================
    # PLOTTING
    # =========================================================
    print("\nGenerating figures...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Quality vs Iteration Curve
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    mean_dense = df_dense.groupby("iteration").mean(numeric_only=True).reset_index()

    color_absrel = "#1f77b4"
    color_gain = "#2ca02c"
    ax1.set_xlabel("Recurrent Iteration $i$ ($K=16$)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Depth AbsRel (Mean across 10 sequences)", color=color_absrel, fontsize=12, fontweight="bold")
    ax1.plot(mean_dense["iteration"], mean_dense["abs_rel"], marker="o", color=color_absrel, linewidth=2.5, label="Depth AbsRel")
    ax1.tick_params(axis="y", labelcolor=color_absrel)
    ax1.set_xticks(range(0, 17))
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color_gain = "#d62728"
    ax2.set_ylabel("Marginal AbsRel Gain $\\Delta = \\text{AbsRel}_{i-1} - \\text{AbsRel}_i$", color=color_gain, fontsize=12, fontweight="bold")
    # Plot bar for marginal gain starting from i=1
    gain_data = mean_dense[mean_dense["iteration"] >= 1]
    ax2.bar(gain_data["iteration"], gain_data["marginal_absrel_gain"], alpha=0.35, color=color_gain, width=0.5, label="Marginal Gain")
    ax2.tick_params(axis="y", labelcolor=color_gain)

    # Shaded phases
    ax1.axvspan(0.5, 5.5, color="#e0f2fe", alpha=0.4, label="Early (1-5)")
    ax1.axvspan(5.5, 11.5, color="#fef3c7", alpha=0.4, label="Middle (6-11)")
    ax1.axvspan(11.5, 16.5, color="#f1f5f9", alpha=0.4, label="Late (12-16)")

    plt.title("DVLT Decode-at-Every-Iteration Curve & Marginal AbsRel Gain", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "iteration_role_quality_vs_iteration.png")
    plt.savefig(plot1_path, dpi=200)
    plt.close()

    # 2. Update Magnitudes
    fig, ax = plt.subplots(figsize=(9, 5.5))
    mean_norms = df_norms.groupby("iteration").mean(numeric_only=True).reset_index()

    ax.plot(mean_norms["iteration"], mean_norms["mag_frame"], marker="s", color="#3b82f6", linewidth=2.2, label="Frame Update Magnitude (||Δ_frame|| / ||h_before||)")
    ax.plot(mean_norms["iteration"], mean_norms["mag_global"], marker="^", color="#ef4444", linewidth=2.2, label="Global Update Magnitude (||Δ_global|| / ||h_frame||)")
    ax.plot(mean_norms["iteration"], mean_norms["mag_step"], marker="o", color="#10b981", linewidth=2.2, linestyle="--", label="Total Step Update (||Δ_step|| / ||h_before||)")

    ax.set_xlabel("Recurrent Iteration $i$", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative Update Magnitude", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 17))
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.axvspan(0.5, 5.5, color="#e0f2fe", alpha=0.3)
    ax.axvspan(5.5, 11.5, color="#fef3c7", alpha=0.3)
    ax.axvspan(11.5, 16.5, color="#f1f5f9", alpha=0.3)

    plt.title("Relative Hidden-State Update Magnitudes (Frame vs Global)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "iteration_role_update_magnitudes.png")
    plt.savefig(plot2_path, dpi=200)
    plt.close()

    # 3. Frame vs Global Causal Importance
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    df_f = df_causal[df_causal["ablation_type"] == "skip_frame"].groupby("iteration")["delta_absrel"].agg(["mean", "std"]).reset_index()
    df_g = df_causal[df_causal["ablation_type"] == "skip_global"].groupby("iteration")["delta_absrel"].agg(["mean", "std"]).reset_index()

    ax.plot(df_f["iteration"], df_f["mean"], marker="o", color="#2563eb", linewidth=2.5, label="Skip Frame Attention (at step $i$)")
    ax.fill_between(df_f["iteration"], df_f["mean"] - df_f["std"], df_f["mean"] + df_f["std"], color="#2563eb", alpha=0.15)

    ax.plot(df_g["iteration"], df_g["mean"], marker="s", color="#dc2626", linewidth=2.5, label="Skip Global Attention (at step $i$)")
    ax.fill_between(df_g["iteration"], df_g["mean"] - df_g["std"], df_g["mean"] + df_g["std"], color="#dc2626", alpha=0.15)

    ax.set_xlabel("Ablated Iteration $i$", fontsize=12, fontweight="bold")
    ax.set_ylabel("Final Metric Degradation $\\Delta \\text{AbsRel}$ (vs untouched K=16)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 17))
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.axvspan(0.5, 5.5, color="#e0f2fe", alpha=0.3, label="Early Phase (1-5)")
    ax.axvspan(5.5, 11.5, color="#fef3c7", alpha=0.3, label="Middle Phase (6-11)")
    ax.axvspan(11.5, 16.5, color="#f1f5f9", alpha=0.3, label="Late Phase (12-16)")

    plt.title("Single-Iteration Causal Ablations: Frame vs Global Attention Importance", fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "iteration_role_frame_vs_global_importance.png")
    plt.savefig(plot3_path, dpi=200)
    plt.close()

    # 4. Per-Scene Heatmap
    fig, (ax_h_f, ax_h_g) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    pivot_f = df_causal[df_causal["ablation_type"] == "skip_frame"].groupby(["scan_name", "iteration"])["delta_absrel"].mean().unstack()
    pivot_g = df_causal[df_causal["ablation_type"] == "skip_global"].groupby(["scan_name", "iteration"])["delta_absrel"].mean().unstack()

    im_f = ax_h_f.imshow(pivot_f.values * 1000.0, cmap="YlOrRd", aspect="auto")
    ax_h_f.set_title("Skip Frame: $\\Delta \\text{AbsRel} \\times 10^3$", fontsize=12, fontweight="bold")
    ax_h_f.set_xticks(range(16))
    ax_h_f.set_xticklabels(range(1, 17))
    ax_h_f.set_yticks(range(len(pivot_f.index)))
    ax_h_f.set_yticklabels(pivot_f.index, fontweight="bold")
    ax_h_f.set_xlabel("Iteration $i$", fontsize=11, fontweight="bold")
    fig.colorbar(im_f, ax=ax_h_f, orientation="horizontal", pad=0.15, label="Δ AbsRel × 10³")

    im_g = ax_h_g.imshow(pivot_g.values * 1000.0, cmap="YlOrRd", aspect="auto")
    ax_h_g.set_title("Skip Global: $\\Delta \\text{AbsRel} \\times 10^3$", fontsize=12, fontweight="bold")
    ax_h_g.set_xticks(range(16))
    ax_h_g.set_xticklabels(range(1, 17))
    ax_h_g.set_xlabel("Iteration $i$", fontsize=11, fontweight="bold")
    fig.colorbar(im_g, ax=ax_h_g, orientation="horizontal", pad=0.15, label="Δ AbsRel × 10³")

    plt.suptitle("Per-Scene Causal Importance Heatmap Across Iterations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot4_path = os.path.join(output_dir, "iteration_role_scene_heatmap.png")
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
        "df_dense": df_dense,
        "df_norms": df_norms,
        "df_causal": df_causal,
        "df_scene_summary": df_scene_summary,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DVLT Iteration-Role Audit V0")
    parser.add_argument("--data_root", type=str, default="datasets/test/dtu")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_seqs", type=int, default=None, help="Limit number of sequences for quick smoke test")
    args = parser.parse_args()

    run_iteration_role_audit(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_seqs=args.max_seqs,
    )
