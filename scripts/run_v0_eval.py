"""Run V0 fixed-K evaluation on DTU dataset for ViewHalt.

Evaluates frozen pretrained DVLT across recurrent step counts:
  K ∈ {4, 6, 8, 10, 12, 14, 16}
on identical multi-view scenes from the DTU benchmark (scan1, scan10, scan24, scan33, scan110).

Saves per-scene and per-view metrics:
- Depth AbsRel, RMSE, MAE, Delta1 (aligned to real DTU GT depth on foreground masks)
- Camera pose rotation and translation errors (against real DTU GT camera poses)
- 3D Geometry point error (mean Euclidean distance to GT world points)
- Inference wall-clock time and peak VRAM
- Per-view hidden-state convergence signals (layer hook norms)
Outputs machine-readable raw results in JSON and CSV formats.
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
from dvlt.model.dvlt.model import DVLT


def run_v0_eval(
    data_root="datasets/test/dtu",
    scans=None,
    num_views_per_scene=6,
    k_values=None,
    output_dir="outputs",
):
    if scans is None:
        scans = ["scan1", "scan10", "scan24", "scan33", "scan110"]
    if k_values is None:
        k_values = [4, 6, 8, 10, 12, 14, 16]

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("ViewHalt V0 Multi-K Oracle Evaluation")
    print(f"Scans: {scans}")
    print(f"K values: {k_values}")
    print(f"Views per scene: {num_views_per_scene}")
    print("=" * 70)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    print("Loading DVLT model from nvidia/dvlt...")
    model = DVLT(img_size=504, depth_head_type="conv")
    model.load_pretrained("nvidia/dvlt", strict=True)
    model.setup_test(accelerator)

    # Setup hidden-state delta tracking via hook
    step_hidden_states = []

    def hook_fn(module, input_tensor, output_tensor):
        # output_tensor is [B*S, N_tokens, C]
        step_hidden_states.append(output_tensor.detach())

    hook = model.model.recurrent_blocks[0].register_forward_hook(hook_fn)

    # Prepare datasets for each scan
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

    all_results = []
    metadata = {
        "upstream_commit": "134b21f2af02d98039e79ab5dd48f36dc8123c97",
        "checkpoint": "nvidia/dvlt",
        "checkpoint_hash": "20f919875a00ee3ab2e840f2f31580f4652b7c5e",
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        "precision": "bf16",
        "resolution": [378, 504],
        "k_values": k_values,
        "scans": scans,
        "num_views_per_scene": num_views_per_scene,
    }

    cfg = OmegaConf.create({
        "target": "dtu.DTU",
        "params": {"root_path": data_root}
    })
    ds = DataverseEvalDataset(
        dataverse_cfg=cfg,
        view_ranking="middle_first",
        max_frames=num_views_per_scene,
    )
    ds.set_image_params(504, 14)
    ms_ds = MultiSourceDataset({"dtu": ds}, training=False, **test_config)

    for scan_idx in range(len(ms_ds)):
        sample = ms_ds[scan_idx]
        scan_name = sample[DataField.SEQ_NAME].replace("null_", "")
        print(f"\n[{scan_idx + 1}/{len(ms_ds)}] Evaluating scene: {scan_name}")
        batch = default_collate_fn([sample])
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        gt_depths = batch[DataField.DEPTHS][0]       # (S, H, W)
        gt_masks = batch[DataField.POINT_MASKS][0]   # (S, H, W)
        gt_c2w = batch[DataField.EXTRINSICS_C2W][0]   # (S, 4, 4)
        gt_world_pts = batch[DataField.WORLD_POINTS][0] # (S, H, W, 3)
        dtu_frame_ids = batch[DataField.IDS][0].cpu().tolist() # list of int
        S = gt_depths.shape[0]

        for K in k_values:
            model.inference_steps = K
            model.model.inference_steps = K
            step_hidden_states.clear()

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad(), accelerator.autocast():
                preds = model.predict(batch, accelerator)
            torch.cuda.synchronize()
            elapsed_sec = time.perf_counter() - t0
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

            pred_depths = preds[PredictionField.DEPTHS][0]       # (S, H, W)
            pred_world_pts = preds[PredictionField.WORLD_POINTS][0] # (S, H, W, 3)
            pred_c2w_raw = preds[PredictionField.CAMERAS][0].camera_to_worlds # (S, 3, 4)

            # Homogenize predicted poses to 4x4
            if pred_c2w_raw.shape[-2] == 3:
                homo = torch.tensor([0, 0, 0, 1], device=device, dtype=pred_c2w_raw.dtype).expand(S, 1, 4)
                pred_c2w = torch.cat([pred_c2w_raw, homo], dim=-2)
            else:
                pred_c2w = pred_c2w_raw

            # Compute hidden-state delta for final step K if available
            final_hidden_delta = None
            if len(step_hidden_states) >= 2:
                # delta between step K-1 and step K
                h_prev = step_hidden_states[-2]  # [S, N, C]
                h_curr = step_hidden_states[-1]  # [S, N, C]
                norm_diff = (h_curr - h_prev).norm(dim=-1).mean(dim=-1) # [S]
                norm_curr = h_curr.norm(dim=-1).mean(dim=-1).clamp(min=1e-6) # [S]
                rel_hidden_delta = (norm_diff / norm_curr).cpu().tolist()
            else:
                rel_hidden_delta = [0.0] * S

            # Compute reference-aligned relative poses
            # Rel to view 0: T_rel = inv(T_0) @ T_v
            inv_pred_0 = torch.linalg.inv(pred_c2w[0])
            inv_gt_0 = torch.linalg.inv(gt_c2w[0])
            rel_pred = torch.bmm(inv_pred_0.unsqueeze(0).expand(S, 4, 4), pred_c2w)
            rel_gt = torch.bmm(inv_gt_0.unsqueeze(0).expand(S, 4, 4), gt_c2w)

            # Evaluate each individual view v
            view_abs_rels = []
            for v in range(S):
                mask_v = gt_masks[v] & (gt_depths[v] > 0)
                gt_d_v = gt_depths[v]
                pred_d_v = pred_depths[v]

                # Depth alignment
                pred_d_aligned = apply_alignment(pred_d_v, gt_d_v, mask_v, align="median")

                if mask_v.sum() > 0:
                    d_pred_val = pred_d_aligned[mask_v]
                    d_gt_val = gt_d_v[mask_v]
                    abs_rel = (torch.abs(d_pred_val - d_gt_val) / d_gt_val).mean().item()
                    rmse = torch.sqrt(torch.mean((d_pred_val - d_gt_val) ** 2)).item()
                    mae = torch.mean(torch.abs(d_pred_val - d_gt_val)).item()
                    ratio = torch.maximum(d_pred_val / d_gt_val, d_gt_val / d_pred_val)
                    delta1 = (ratio < 1.25).float().mean().item()

                    # Geometry: 3D point error (in mm)
                    pt_pred_val = pred_world_pts[v][mask_v]
                    pt_gt_val = gt_world_pts[v][mask_v]
                    geom_l2 = torch.norm(pt_pred_val - pt_gt_val, dim=-1).mean().item()
                else:
                    abs_rel, rmse, mae, delta1, geom_l2 = 0.0, 0.0, 0.0, 1.0, 0.0

                # Camera pose errors for view v (relative to view 0)
                r_pred_v = rel_pred[v, :3, :3].unsqueeze(0)
                r_gt_v = rel_gt[v, :3, :3].unsqueeze(0)
                rot_err_deg = so3_relative_angle(r_pred_v, r_gt_v).item()

                t_pred_v = rel_pred[v, :3, 3]
                t_gt_v = rel_gt[v, :3, 3]
                norm_p = t_pred_v.norm().clamp(min=1e-6)
                norm_g = t_gt_v.norm().clamp(min=1e-6)
                cos_sim = torch.dot(t_pred_v, t_gt_v) / (norm_p * norm_g)
                cos_sim = cos_sim.clamp(-1.0, 1.0)
                trans_err_deg = torch.rad2deg(torch.acos(cos_sim)).item() if v > 0 else 0.0

                record = {
                    "scene": scan_name,
                    "seq_view_idx": v,
                    "dtu_frame_id": dtu_frame_ids[v],
                    "K": K,
                    "depth_abs_rel": round(abs_rel, 6),
                    "depth_rmse_mm": round(rmse, 4),
                    "depth_mae_mm": round(mae, 4),
                    "depth_delta1": round(delta1, 6),
                    "pose_rot_error_deg": round(rot_err_deg, 4),
                    "pose_trans_error_deg": round(trans_err_deg, 4),
                    "geom_l2_error_mm": round(geom_l2, 4),
                    "hidden_state_delta": round(rel_hidden_delta[v], 6),
                    "runtime_sec": round(elapsed_sec, 4),
                    "peak_vram_mb": round(peak_vram_mb, 2),
                }
                all_results.append(record)
                view_abs_rels.append(abs_rel)

            mean_absrel = np.mean(view_abs_rels)
            print(
                f"  K={K:2d} | Mean AbsRel: {mean_absrel:.5f} | "
                f"Time: {elapsed_sec*1000:.1f}ms | Peak VRAM: {peak_vram_mb:.1f}MiB"
            )

    hook.remove()

    # Save machine-readable raw outputs
    json_path = os.path.join(output_dir, "v0_raw_results.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "records": all_results}, f, indent=2)

    csv_path = os.path.join(output_dir, "v0_raw_results.csv")
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 70)
    print(f"Multi-K Evaluation Complete! Total records: {len(all_results)}")
    print(f"JSON saved to: {json_path}")
    print(f"CSV saved to:  {csv_path}")
    print("=" * 70)
    return all_results


if __name__ == "__main__":
    run_v0_eval()
