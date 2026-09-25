"""ViewHalt V0.5: Expanded Multi-Scene Robustness Evaluation Script.

Evaluates frozen pretrained DVLT across the in-distribution step counts:
  K ∈ {8, 10, 12, 14, 16}
across 14 DTU scans and multiple view subsets per scene:
  - Subset A: 'middle_first' (6 views clustered around trajectory center)
  - Subset B: 'uniform_stride' (6 views uniformly strided across all 49 views)
Total: 28 multi-view sequences × 6 views = 168 individual view evaluations per K.

Saves:
- Depth AbsRel, RMSE, MAE, Delta1 (aligned to real DTU GT depth on foreground masks)
- Camera pose rotation & translation errors
- 3D point geometry error
- Per-view recurrent hidden-state deltas at each step
- Output depth map delta between steps
Outputs:
- outputs/v05_raw_results.json
- outputs/v05_raw_results.csv
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


def run_v05_eval(
    data_root="datasets/test/dtu",
    output_dir="outputs",
):
    os.makedirs(output_dir, exist_ok=True)

    # 14 distinct scans available
    scans = [
        "scan1", "scan4", "scan9", "scan10", "scan11", "scan12",
        "scan15", "scan23", "scan24", "scan29", "scan33", "scan48",
        "scan62", "scan110"
    ]
    # In-distribution step counts only
    k_values = [8, 10, 12, 14, 16]

    print("=" * 75)
    print("ViewHalt V0.5: Expanded DTU Robustness Evaluation")
    print(f"Scans ({len(scans)}): {scans}")
    print(f"In-Distribution K: {k_values}")
    print("=" * 75)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    print("Loading DVLT model from nvidia/dvlt...")
    model = DVLT(img_size=504, depth_head_type="conv")
    model.load_pretrained("nvidia/dvlt", strict=True)
    model.setup_test(accelerator)

    # Hidden-state delta tracker
    step_hidden_states = []

    def hook_fn(module, input_tensor, output_tensor):
        step_hidden_states.append(output_tensor.detach())

    hook = model.model.recurrent_blocks[0].register_forward_hook(hook_fn)

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

    # Define subsets:
    # 1. middle_first (6 frames)
    # 2. uniform (6 frames sampled across the 49 views: e.g. [0, 9, 19, 29, 39, 48])
    subsets_def = [
        {"name": "subset_middle", "ranking": "middle_first", "sampling": "first"},
        {"name": "subset_uniform", "ranking": "index", "sampling": "uniform"},
    ]

    all_results = []
    metadata = {
        "version": "V0.5",
        "scans": scans,
        "k_values": k_values,
        "subsets": [s["name"] for s in subsets_def],
        "num_views_per_sequence": 6,
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        "precision": "bf16",
        "resolution": [378, 504],
    }

    total_seqs = len(scans) * len(subsets_def)
    seq_counter = 0

    for scan_name in scans:
        for sub in subsets_def:
            seq_counter += 1
            seq_id = f"{scan_name}_{sub['name']}"
            print(f"\n[{seq_counter}/{total_seqs}] Evaluating: {seq_id} (ranking={sub['ranking']}, sampling={sub['sampling']})")

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

            # Find video index for scan_name
            video_idx = None
            for idx in range(ds.ds.num_videos()):
                if ds.ds._get_scan_name(idx) == scan_name:
                    video_idx = idx
                    break
            assert video_idx is not None, f"Could not find video index for {scan_name}"

            ms_ds = MultiSourceDataset({"dtu": ds}, training=False, **test_config)
            sample = ms_ds[video_idx]
            batch = default_collate_fn([sample])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            gt_depths = batch[DataField.DEPTHS][0]
            gt_masks = batch[DataField.POINT_MASKS][0]
            gt_c2w = batch[DataField.EXTRINSICS_C2W][0]
            gt_world_pts = batch[DataField.WORLD_POINTS][0]
            dtu_frame_ids = batch[DataField.IDS][0].cpu().tolist()
            S = gt_depths.shape[0]

            prev_pred_depths = None

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

                pred_depths = preds[PredictionField.DEPTHS][0]
                pred_world_pts = preds[PredictionField.WORLD_POINTS][0]
                pred_c2w_raw = preds[PredictionField.CAMERAS][0].camera_to_worlds

                if pred_c2w_raw.shape[-2] == 3:
                    homo = torch.tensor([0, 0, 0, 1], device=device, dtype=pred_c2w_raw.dtype).expand(S, 1, 4)
                    pred_c2w = torch.cat([pred_c2w_raw, homo], dim=-2)
                else:
                    pred_c2w = pred_c2w_raw

                # Per-view hidden-state delta at step K
                if len(step_hidden_states) >= 2:
                    h_prev = step_hidden_states[-2]
                    h_curr = step_hidden_states[-1]
                    norm_diff = (h_curr - h_prev).norm(dim=-1).mean(dim=-1)
                    norm_curr = h_curr.norm(dim=-1).mean(dim=-1).clamp(min=1e-6)
                    rel_hidden_delta = (norm_diff / norm_curr).cpu().tolist()
                else:
                    rel_hidden_delta = [0.0] * S

                # Output depth map relative delta between current K and previous K
                if prev_pred_depths is not None:
                    d_diff = torch.abs(pred_depths - prev_pred_depths).mean(dim=[-2, -1])
                    d_denom = pred_depths.abs().mean(dim=[-2, -1]).clamp(min=1e-6)
                    depth_delta = (d_diff / d_denom).cpu().tolist()
                else:
                    depth_delta = [0.0] * S
                prev_pred_depths = pred_depths.clone()

                # Align relative poses
                inv_pred_0 = torch.linalg.inv(pred_c2w[0])
                inv_gt_0 = torch.linalg.inv(gt_c2w[0])
                rel_pred = torch.bmm(inv_pred_0.unsqueeze(0).expand(S, 4, 4), pred_c2w)
                rel_gt = torch.bmm(inv_gt_0.unsqueeze(0).expand(S, 4, 4), gt_c2w)

                absrel_list = []
                for v in range(S):
                    mask_v = gt_masks[v] & (gt_depths[v] > 0)
                    gt_d_v = gt_depths[v]
                    pred_d_v = pred_depths[v]

                    pred_d_aligned = apply_alignment(pred_d_v, gt_d_v, mask_v, align="median")

                    if mask_v.sum() > 0:
                        d_pred_val = pred_d_aligned[mask_v]
                        d_gt_val = gt_d_v[mask_v]
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
                    cos_sim = torch.dot(t_pred_v, t_gt_v) / (norm_p * norm_g)
                    cos_sim = cos_sim.clamp(-1.0, 1.0)
                    trans_err_deg = torch.rad2deg(torch.acos(cos_sim)).item() if v > 0 else 0.0

                    record = {
                        "sequence_id": seq_id,
                        "scene": scan_name,
                        "subset": sub["name"],
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
                        "depth_map_delta": round(depth_delta[v], 6),
                        "runtime_sec": round(elapsed_sec, 4),
                        "peak_vram_mb": round(peak_vram_mb, 2),
                    }
                    all_results.append(record)
                    absrel_list.append(abs_rel)

                print(f"    K={K:2d} | Mean AbsRel: {np.mean(absrel_list):.5f} | Time: {elapsed_sec*1000:.0f}ms")

    hook.remove()

    json_path = os.path.join(output_dir, "v05_raw_results.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "records": all_results}, f, indent=2)

    csv_path = os.path.join(output_dir, "v05_raw_results.csv")
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 75)
    print(f"V0.5 Evaluation Complete! Total records: {len(all_results)}")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print("=" * 75)
    return all_results


if __name__ == "__main__":
    run_v05_eval()
