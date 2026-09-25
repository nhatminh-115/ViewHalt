"""ViewHalt V0.6: Common-Trajectory Halting Validation & Simulation-Only Freeze Oracle.

Methodological Goal:
Resolve the linspace grid solver discrepancy by extracting all intermediate candidate
halt points from a single common K=16 recurrent solver trajectory.
Test whether:
1. Heterogeneous convergence remains strong along the common trajectory.
2. Hidden-state delta predicts future gain independently at each iteration (unconfounded by K).
3. Simulation-only freeze oracle preserves reconstruction quality under real cross-view attention coupling.

Protocol:
- Single K=16 recurrent forward pass per sequence (torch.linspace(0, 1, 16)).
- Cache hidden states h_i after every recurrent step i in {1..16}.
- Decode intermediate states for i in {8, 10, 12, 14, 16} using DVLT decoder heads.
- Compute ground truth metrics against DTU metric depth and camera poses.
- Identify oracle K* for each view under relative tolerances (1%, 2%, 5%, 10%).
- Execute simulation-only freeze oracle (freezing early views post-block while active views continue).
- Compare Fixed K=16, Uniform early stopping, Decode-Oracle, and Freeze-Oracle execution.

Saves:
- outputs/v06_raw_results.json
- outputs/v06_raw_results.csv
"""

import csv
import json
import os
import time
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from dvlt.common.constants import DataField, PredictionField
from dvlt.common.rotation import so3_relative_angle
from dvlt.data.collate import default_collate_fn
from dvlt.data.datasets.multi_source import MultiSourceDataset
from dvlt.data.datasets.parser.dataverse import DataverseEvalDataset
from dvlt.metric.depth import apply_alignment
from dvlt.model.dvlt.model import DVLT, _slice_expand_flatten


def run_v06_common_trajectory(
    data_root="datasets/test/dtu",
    output_dir="outputs",
):
    os.makedirs(output_dir, exist_ok=True)

    scans = [
        "scan1", "scan4", "scan9", "scan10", "scan11", "scan12",
        "scan15", "scan23", "scan24", "scan29", "scan33", "scan48",
        "scan62", "scan110"
    ]
    eval_steps = [8, 10, 12, 14, 16]
    tolerances = [0.01, 0.02, 0.05, 0.10]

    print("=" * 80)
    print("ViewHalt V0.6: Common-Trajectory Halting Validation & Freeze Oracle")
    print(f"Scans ({len(scans)}): {scans}")
    print(f"Common Trajectory Total Steps: K=16")
    print(f"Intermediate Decoded Checkpoints: {eval_steps}")
    print("=" * 80)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    print("Loading pretrained DVLT model (nvidia/dvlt)...")
    model = DVLT(img_size=504, depth_head_type="conv")
    model.load_pretrained("nvidia/dvlt", strict=True)
    model.setup_test(accelerator)

    test_config = {
        "normalize_scene": False,
        "load_data_fields": [
            "images",
            "extrinsics_c2w",
            "intrinsics",
            "depths",
            "world_points",
            "point_masks",
        ],
    }

    subsets_def = [
        {"name": "subset_middle", "ranking": "middle_first", "sampling": "first"},
        {"name": "subset_uniform", "ranking": "index", "sampling": "uniform"},
    ]

    metadata = {
        "version": "V0.6",
        "scans": scans,
        "common_trajectory_k": 16,
        "eval_steps": eval_steps,
        "tolerances": tolerances,
        "subsets": [s["name"] for s in subsets_def],
        "num_views_per_sequence": 6,
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        "precision": "bf16",
        "resolution": [378, 504],
    }

    raw_trajectory_records = []
    freeze_oracle_records = []
    summary_by_sequence = []

    total_seqs = len(scans) * len(subsets_def)
    seq_counter = 0

    # Track unique physical camera frames
    all_physical_frames = set()

    for scan_name in scans:
        for sub in subsets_def:
            seq_counter += 1
            seq_id = f"{scan_name}_{sub['name']}"
            print(f"\n[{seq_counter}/{total_seqs}] Processing: {seq_id} (ranking={sub['ranking']}, sampling={sub['sampling']})")

            cfg = OmegaConf.create({
                "target": "dtu.DTU",
                "params": {"root_path": data_root}
            })
            ds = DataverseEvalDataset(
                dataverse_cfg=cfg,
                view_ranking=sub["ranking"],
                view_sampling=sub["sampling"],
                max_frames=6,
            )
            ds.set_image_params(504, 14)

            video_idx = None
            for idx in range(ds.ds.num_videos()):
                if ds.ds._get_scan_name(idx) == scan_name:
                    video_idx = idx
                    break
            assert video_idx is not None, f"Scan {scan_name} not found in DTU dataset."

            ms_ds = MultiSourceDataset({"dtu": ds}, training=False, **test_config)
            sample = ms_ds[video_idx]
            batch = default_collate_fn([sample])

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            images = batch[DataField.IMAGES]
            B, S, _, H, W = images.shape
            inner = model.model
            dtu_frame_ids = batch[DataField.IDS][0].cpu().tolist()

            for f_id in dtu_frame_ids:
                all_physical_frames.add((scan_name, int(f_id)))

            gt_depths = batch[DataField.DEPTHS][0]
            gt_masks = batch[DataField.POINT_MASKS][0]
            gt_c2w = batch[DataField.EXTRINSICS_C2W][0]
            gt_world_pts = batch[DataField.WORLD_POINTS][0]

            with torch.no_grad(), accelerator.autocast():
                # -------------------------------------------------------------
                # 1. Single K=16 Common Trajectory Forward Pass
                # -------------------------------------------------------------
                z_0 = inner._encode_images(images)
                register_token = inner.register_token.expand(B, S, -1, -1).reshape(B * S, inner.num_register_tokens, -1)
                camera_token = _slice_expand_flatten(inner.camera_token, B, S)
                x_init = torch.cat([camera_token, register_token, z_0], dim=1)
                rope_pos = inner._get_rope_positions(B * S, H, W, images.device)

                K = 16
                ts = torch.linspace(0.0, 1.0, K).tolist()
                cached_states = {}
                x = x_init.clone()

                t0 = time.time()
                for i in range(K):
                    t_now = ts[i]
                    t_next = ts[i + 1] if i + 1 < K else 1.0
                    x = inner._interval_step(x, t_now, t_next, rope_pos, B, S)
                    step_idx = i + 1
                    cached_states[step_idx] = x.clone()
                common_traj_time_ms = (time.time() - t0) * 1000.0

                # -------------------------------------------------------------
                # 2. Compute Hidden-State Deltas (1-step and 2-step)
                # -------------------------------------------------------------
                # For each view v and step i:
                # delta_h_1step: ||h_i - h_{i-1}|| / ||h_i||
                # delta_h_2step: ||h_i - h_{i-2}|| / ||h_i||
                hidden_deltas_1step = {}
                hidden_deltas_2step = {}

                for i in range(1, K + 1):
                    h_curr = cached_states[i]  # shape: (S, P, D)
                    # 1-step delta
                    if i > 1:
                        h_prev1 = cached_states[i - 1]
                        diff1 = (h_curr - h_prev1).norm(dim=-1).mean(dim=-1)
                        denom = h_curr.norm(dim=-1).mean(dim=-1).clamp(min=1e-6)
                        hidden_deltas_1step[i] = (diff1 / denom).cpu().tolist()
                    else:
                        hidden_deltas_1step[i] = [0.0] * S

                    # 2-step delta
                    if i > 2:
                        h_prev2 = cached_states[i - 2]
                        diff2 = (h_curr - h_prev2).norm(dim=-1).mean(dim=-1)
                        denom = h_curr.norm(dim=-1).mean(dim=-1).clamp(min=1e-6)
                        hidden_deltas_2step[i] = (diff2 / denom).cpu().tolist()
                    else:
                        hidden_deltas_2step[i] = [0.0] * S

                # -------------------------------------------------------------
                # 3. Decode Intermediate States & Evaluate Against DTU GT
                # -------------------------------------------------------------
                decoded_metrics_by_step = {}
                decoded_depths_by_step = {}

                for step in eval_steps:
                    feat = cached_states[step]
                    raw_preds = inner._decode(feat, H, W, B, S, rope_pos)
                    preds = model._postprocess_predictions(batch, raw_preds)
                    pred_depths = preds[PredictionField.DEPTHS][0]
                    pred_world_pts = preds[PredictionField.WORLD_POINTS][0]
                    decoded_depths_by_step[step] = pred_depths.clone()
                    pred_c2w_raw = preds[PredictionField.CAMERAS][0].camera_to_worlds
                    if pred_c2w_raw.shape[-2] == 3:
                        homo = torch.tensor([0, 0, 0, 1], device=device, dtype=pred_c2w_raw.dtype).expand(S, 1, 4)
                        pred_c2w = torch.cat([pred_c2w_raw, homo], dim=-2)
                    else:
                        pred_c2w = pred_c2w_raw

                    if gt_c2w.shape[-2] == 3:
                        homo_gt = torch.tensor([0, 0, 0, 1], device=device, dtype=gt_c2w.dtype).expand(S, 1, 4)
                        gt_c2w_4x4 = torch.cat([gt_c2w, homo_gt], dim=-2)
                    else:
                        gt_c2w_4x4 = gt_c2w

                    inv_pred_0 = torch.linalg.inv(pred_c2w[0])
                    inv_gt_0 = torch.linalg.inv(gt_c2w_4x4[0])
                    rel_pred = torch.bmm(inv_pred_0.unsqueeze(0).expand(S, 4, 4), pred_c2w)
                    rel_gt = torch.bmm(inv_gt_0.unsqueeze(0).expand(S, 4, 4), gt_c2w_4x4)

                    step_view_metrics = []
                    for v in range(S):
                        mask_v = gt_masks[v] & (gt_depths[v] > 0)
                        pred_d_aligned = apply_alignment(pred_depths[v], gt_depths[v], mask_v, align="median")

                        if mask_v.sum() > 0:
                            d_pred_val = pred_d_aligned[mask_v]
                            d_gt_val = gt_depths[v][mask_v]
                            abs_rel = (torch.abs(d_pred_val - d_gt_val) / d_gt_val).mean().item()
                            rmse = torch.sqrt(torch.mean((d_pred_val - d_gt_val) ** 2)).item()
                            mae = torch.mean(torch.abs(d_pred_val - d_gt_val)).item()
                            ratio = torch.maximum(d_pred_val / d_gt_val, d_gt_val / d_pred_val)
                            delta1 = (ratio < 1.25).float().mean().item()

                            pt_pred_val = pred_world_pts[v][mask_v]
                            pt_gt_val = gt_world_pts[v][mask_v]
                            geom_l2 = torch.norm(pt_pred_val - pt_gt_val, dim=-1).mean().item()
                        else:
                            abs_rel, rmse, mae, delta1, geom_l2 = 0.0, 0.0, 0.0, 1.0, 0.0

                        r_pred_v = rel_pred[v, :3, :3].unsqueeze(0)
                        r_gt_v = rel_gt[v, :3, :3].unsqueeze(0)
                        rot_err_deg = so3_relative_angle(r_pred_v, r_gt_v).item()

                        t_pred_v = rel_pred[v, :3, 3]
                        t_gt_v = rel_gt[v, :3, 3]
                        norm_p = t_pred_v.norm().clamp(min=1e-6)
                        norm_g = t_gt_v.norm().clamp(min=1e-6)
                        cos_sim = (torch.dot(t_pred_v, t_gt_v) / (norm_p * norm_g)).clamp(-1.0, 1.0)
                        trans_err_deg = torch.rad2deg(torch.acos(cos_sim)).item() if v > 0 else 0.0

                        step_view_metrics.append({
                            "abs_rel": abs_rel,
                            "rmse": rmse,
                            "mae": mae,
                            "delta1": delta1,
                            "geom_l2": geom_l2,
                            "rot_err_deg": rot_err_deg,
                            "trans_err_deg": trans_err_deg,
                        })

                    decoded_metrics_by_step[step] = step_view_metrics

                # -------------------------------------------------------------
                # 4. Determine Common-Trajectory Oracle K* for Each View
                # -------------------------------------------------------------
                # Terminal step is 16
                k16_metrics = decoded_metrics_by_step[16]
                oracle_k_by_tol = {}
                for tol in tolerances:
                    k_stars = []
                    for v in range(S):
                        target_absrel = (1.0 + tol) * k16_metrics[v]["abs_rel"]
                        k_star = 16
                        for s in eval_steps:
                            if decoded_metrics_by_step[s][v]["abs_rel"] <= target_absrel:
                                k_star = s
                                break
                        k_stars.append(k_star)
                    oracle_k_by_tol[tol] = k_stars

                # -------------------------------------------------------------
                # 5. Record Common-Trajectory View Records
                # -------------------------------------------------------------
                for step in eval_steps:
                    for v in range(S):
                        abs_rel_v = decoded_metrics_by_step[step][v]["abs_rel"]
                        abs_rel_16_v = k16_metrics[v]["abs_rel"]
                        future_gain = abs_rel_v - abs_rel_16_v

                        # Depth delta vs previous evaluated step
                        prev_step_idx = eval_steps.index(step) - 1
                        if prev_step_idx >= 0:
                            prev_s = eval_steps[prev_step_idx]
                            d_diff = torch.abs(decoded_depths_by_step[step][v] - decoded_depths_by_step[prev_s][v]).mean()
                            d_denom = decoded_depths_by_step[step][v].abs().mean().clamp(min=1e-6)
                            depth_delta = (d_diff / d_denom).item()
                        else:
                            depth_delta = 0.0

                        record = {
                            "scene": scan_name,
                            "subset": sub["name"],
                            "view_idx": v,
                            "frame_id": int(dtu_frame_ids[v]),
                            "step_i": step,
                            "abs_rel": abs_rel_v,
                            "rmse": decoded_metrics_by_step[step][v]["rmse"],
                            "mae": decoded_metrics_by_step[step][v]["mae"],
                            "delta1": decoded_metrics_by_step[step][v]["delta1"],
                            "geom_l2": decoded_metrics_by_step[step][v]["geom_l2"],
                            "rot_err_deg": decoded_metrics_by_step[step][v]["rot_err_deg"],
                            "trans_err_deg": decoded_metrics_by_step[step][v]["trans_err_deg"],
                            "hidden_delta_1step": hidden_deltas_1step[step][v],
                            "hidden_delta_2step": hidden_deltas_2step[step][v],
                            "depth_delta": depth_delta,
                            "abs_rel_16": abs_rel_16_v,
                            "future_gain": future_gain,
                            "oracle_k_star_1pct": oracle_k_by_tol[0.01][v],
                            "oracle_k_star_2pct": oracle_k_by_tol[0.02][v],
                            "oracle_k_star_5pct": oracle_k_by_tol[0.05][v],
                            "oracle_k_star_10pct": oracle_k_by_tol[0.10][v],
                            "safe_to_halt_1pct": int(future_gain <= 0.01 * abs_rel_16_v),
                            "safe_to_halt_2pct": int(future_gain <= 0.02 * abs_rel_16_v),
                            "safe_to_halt_5pct": int(future_gain <= 0.05 * abs_rel_16_v),
                            "safe_to_halt_10pct": int(future_gain <= 0.10 * abs_rel_16_v),
                        }
                        raw_trajectory_records.append(record)

                # -------------------------------------------------------------
                # 6. Critical Executable-Oracle Test (Simulation Freeze Oracle)
                # -------------------------------------------------------------
                # We test freeze execution for both 5% tolerance (primary) and 2% tolerance
                for tol_name, tol_val in [("5pct", 0.05), ("2pct", 0.02)]:
                    target_k_stars = oracle_k_by_tol[tol_val]

                    x_freeze = x_init.clone()
                    frozen_states = {}

                    t_frz_0 = time.time()
                    for i in range(K):
                        t_now = ts[i]
                        t_next = ts[i + 1] if i + 1 < K else 1.0
                        step_idx = i + 1

                        # Execute normal interval step through block
                        x_freeze = inner._interval_step(x_freeze, t_now, t_next, rope_pos, B, S)

                        # Overwrite post-block representation of frozen views
                        for v in range(S):
                            if step_idx == target_k_stars[v]:
                                frozen_states[v] = x_freeze[v].clone()
                            elif step_idx > target_k_stars[v]:
                                x_freeze[v] = frozen_states[v].clone()

                    frz_time_ms = (time.time() - t_frz_0) * 1000.0

                    # Decode final freeze-oracle representation
                    raw_frz_preds = inner._decode(x_freeze, H, W, B, S, rope_pos)
                    frz_preds = model._postprocess_predictions(batch, raw_frz_preds)
                    frz_depths = frz_preds[PredictionField.DEPTHS][0]
                    frz_world_pts = frz_preds[PredictionField.WORLD_POINTS][0]
                    frz_c2w_raw = frz_preds[PredictionField.CAMERAS][0].camera_to_worlds
                    if frz_c2w_raw.shape[-2] == 3:
                        homo_frz = torch.tensor([0, 0, 0, 1], device=device, dtype=frz_c2w_raw.dtype).expand(S, 1, 4)
                        frz_c2w = torch.cat([frz_c2w_raw, homo_frz], dim=-2)
                    else:
                        frz_c2w = frz_c2w_raw

                    inv_frz_0 = torch.linalg.inv(frz_c2w[0])
                    rel_frz = torch.bmm(inv_frz_0.unsqueeze(0).expand(S, 4, 4), frz_c2w)

                    for v in range(S):
                        mask_v = gt_masks[v] & (gt_depths[v] > 0)
                        pred_d_aligned = apply_alignment(frz_depths[v], gt_depths[v], mask_v, align="median")

                        if mask_v.sum() > 0:
                            d_pred_val = pred_d_aligned[mask_v]
                            d_gt_val = gt_depths[v][mask_v]
                            frz_abs_rel = (torch.abs(d_pred_val - d_gt_val) / d_gt_val).mean().item()
                            frz_rmse = torch.sqrt(torch.mean((d_pred_val - d_gt_val) ** 2)).item()
                            frz_mae = torch.mean(torch.abs(d_pred_val - d_gt_val)).item()
                            frz_delta1 = (torch.maximum(d_pred_val / d_gt_val, d_gt_val / d_pred_val) < 1.25).float().mean().item()
                            frz_geom = torch.norm(frz_world_pts[v][mask_v] - gt_world_pts[v][mask_v], dim=-1).mean().item()
                        else:
                            frz_abs_rel, frz_rmse, frz_mae, frz_delta1, frz_geom = 0.0, 0.0, 0.0, 1.0, 0.0

                        r_frz_v = rel_frz[v, :3, :3].unsqueeze(0)
                        frz_rot = so3_relative_angle(r_frz_v, rel_gt[v, :3, :3].unsqueeze(0)).item()

                        decode_oracle_absrel = decoded_metrics_by_step[target_k_stars[v]][v]["abs_rel"]
                        fixed16_absrel = k16_metrics[v]["abs_rel"]

                        # Check if view was active during the freeze of other views
                        # A view is an "active view under early frozen neighbors" if K*_v > min(K*_all)
                        is_active_later = bool(target_k_stars[v] > min(target_k_stars))
                        was_early_frozen = bool(target_k_stars[v] < 16)

                        freeze_oracle_records.append({
                            "scene": scan_name,
                            "subset": sub["name"],
                            "tolerance": tol_name,
                            "view_idx": v,
                            "frame_id": int(dtu_frame_ids[v]),
                            "oracle_k_star": target_k_stars[v],
                            "was_early_frozen": was_early_frozen,
                            "is_active_later": is_active_later,
                            "freeze_abs_rel": frz_abs_rel,
                            "decode_oracle_abs_rel": decode_oracle_absrel,
                            "fixed16_abs_rel": fixed16_absrel,
                            "delta_freeze_vs_decode": frz_abs_rel - decode_oracle_absrel,
                            "delta_freeze_vs_k16": frz_abs_rel - fixed16_absrel,
                            "freeze_rmse": frz_rmse,
                            "freeze_mae": frz_mae,
                            "freeze_delta1": frz_delta1,
                            "freeze_geom_l2": frz_geom,
                            "freeze_rot_err_deg": frz_rot,
                        })

                # Log progression for sequence
                mean_absrel_steps = [np.mean([decoded_metrics_by_step[s][v]["abs_rel"] for v in range(S)]) for s in eval_steps]
                print(f"  Common Trajectory AbsRel: {dict(zip(eval_steps, [round(m, 5) for m in mean_absrel_steps]))}")
                print(f"  Oracle K* (5% tol): {oracle_k_by_tol[0.05]} | (2% tol): {oracle_k_by_tol[0.02]}")

    metadata["num_view_instances"] = len(raw_trajectory_records) // len(eval_steps)
    metadata["num_unique_physical_frames"] = len(all_physical_frames)

    print("\n" + "=" * 80)
    print("V0.6 Data Generation Complete.")
    print(f"Total Trajectory Records (Intermediate Steps): {len(raw_trajectory_records)}")
    print(f"Total Freeze-Oracle Records: {len(freeze_oracle_records)}")
    print(f"Total View Instances: {metadata['num_view_instances']}")
    print(f"Total Unique Physical Frames: {metadata['num_unique_physical_frames']}")
    print("=" * 80)

    # Save to JSON
    json_path = os.path.join(output_dir, "v06_raw_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "metadata": metadata,
            "trajectory_records": raw_trajectory_records,
            "freeze_oracle_records": freeze_oracle_records,
        }, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Save Trajectory CSV
    csv_path = os.path.join(output_dir, "v06_raw_results.csv")
    if raw_trajectory_records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=raw_trajectory_records[0].keys())
            writer.writeheader()
            writer.writerows(raw_trajectory_records)
        print(f"Saved CSV: {csv_path}")

    # Save Freeze-Oracle CSV
    frz_csv_path = os.path.join(output_dir, "v06_freeze_oracle_results.csv")
    if freeze_oracle_records:
        with open(frz_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=freeze_oracle_records[0].keys())
            writer.writeheader()
            writer.writerows(freeze_oracle_records)
        print(f"Saved Freeze Oracle CSV: {frz_csv_path}")

    return json_path


if __name__ == "__main__":
    run_v06_common_trajectory()
