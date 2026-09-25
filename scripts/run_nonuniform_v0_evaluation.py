"""DVLT Non-Uniform Schedule V0: Dense Continuous-Time Partition Kill-Test.

Evaluates whether changing the continuous-time partition (t_0, ..., t_{K-1})
in DVLT's interval conditioning allows a smaller K (e.g. K=10 or K=12)
to match or beat the reconstruction quality of Uniform K=14.

Families tested for K in {8, 10, 12}:
- Family A: Power schedules: t_k = (k / (K - 1))^gamma, gamma in {0.5, 0.75, 1.25, 1.5, 2.0}
- Family B: Cosine schedules (late-dense, early-dense, endpoints-dense, middle-dense)
- Family C: Piecewise schedules (early-dense, late-dense, middle-dense, endpoints-dense)
- Family D: Random monotonic partitions (deterministic seeds)
- Baselines: Uniform K in {8, 10, 12, 13, 14, 16}

Metrics reported:
- AbsRel, RMSE, camera rotation error (deg), camera translation error (deg)
- Recurrent FLOPs (GFLOPs)
- Real GPU recurrent latency (torch.cuda.Event, 3 warmups, 10 reps)
- Peak VRAM (MB)
- Per-interval diagnostics (frame residual, global residual, hidden state change)
"""

import csv
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf

from dvlt.common.constants import DataField, PredictionField
from dvlt.data.collate import default_collate_fn
from dvlt.data.datasets.multi_source import MultiSourceDataset
from dvlt.data.datasets.parser.dataverse import DataverseEvalDataset
from dvlt.metric.depth import apply_alignment
from dvlt.metric.pose import so3_relative_angle
from dvlt.model.dvlt.model import DVLT, _slice_expand_flatten

# Per-step compute constants (B=1, S=6, P=977, C=768, N=S*P=5862)
FLOPS_FRAME_STEP = 100.576315  # GFLOPs
FLOPS_GLOBAL_FULL = 188.558673  # GFLOPs
FLOPS_PER_STEP = FLOPS_FRAME_STEP + FLOPS_GLOBAL_FULL  # 289.134988 GFLOPs


def generate_all_schedules():
    """Defines all 54 schedules across baselines and candidate families."""
    schedules = {}

    # 1. Baselines: Uniform schedules
    for K in [8, 10, 12, 13, 14, 16]:
        ts = np.linspace(0.0, 1.0, K).tolist()
        schedules[f"uniform_K{K}"] = {
            "sched_key": f"uniform_K{K}",
            "name": f"Uniform K={K}",
            "family": "baseline",
            "K": K,
            "ts": ts,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

    # Candidate families for K in {8, 10, 12}
    for K in [8, 10, 12]:
        k_grid = np.arange(K) / (K - 1)

        # Family A: Power schedules
        for gamma in [0.5, 0.75, 1.25, 1.5, 2.0]:
            ts = (k_grid ** gamma).tolist()
            ts[0] = 0.0
            ts[-1] = 1.0
            schedules[f"power_K{K}_g{gamma}"] = {
                "sched_key": f"power_K{K}_g{gamma}",
                "name": f"Power K={K} (gamma={gamma})",
                "family": "power",
                "K": K,
                "gamma": gamma,
                "ts": ts,
                "recurrent_flops": K * FLOPS_PER_STEP,
            }

        # Family B: Cosine schedules
        ts_late = (1.0 - np.cos(k_grid * (np.pi / 2.0))).tolist()
        ts_late[0], ts_late[-1] = 0.0, 1.0
        schedules[f"cosine_late_K{K}"] = {
            "sched_key": f"cosine_late_K{K}",
            "name": f"Cosine Late-Dense K={K}",
            "family": "cosine",
            "K": K,
            "ts": ts_late,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

        ts_early = np.sin(k_grid * (np.pi / 2.0)).tolist()
        ts_early[0], ts_early[-1] = 0.0, 1.0
        schedules[f"cosine_early_K{K}"] = {
            "sched_key": f"cosine_early_K{K}",
            "name": f"Cosine Early-Dense K={K}",
            "family": "cosine",
            "K": K,
            "ts": ts_early,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

        ts_endpoints = (0.5 * (1.0 - np.cos(k_grid * np.pi))).tolist()
        ts_endpoints[0], ts_endpoints[-1] = 0.0, 1.0
        schedules[f"cosine_endpoints_K{K}"] = {
            "sched_key": f"cosine_endpoints_K{K}",
            "name": f"Cosine Endpoints-Dense K={K}",
            "family": "cosine",
            "K": K,
            "ts": ts_endpoints,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

        ts_mid = (k_grid + 0.15 * np.sin(2 * np.pi * k_grid)).tolist()
        ts_mid[0], ts_mid[-1] = 0.0, 1.0
        schedules[f"cosine_middle_K{K}"] = {
            "sched_key": f"cosine_middle_K{K}",
            "name": f"Cosine Middle-Dense K={K}",
            "family": "cosine",
            "K": K,
            "ts": ts_mid,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

        # Family C: Piecewise schedules
        n_early = int(round(K * 0.6))
        n_late = K - n_early
        ts_p_early = np.concatenate([np.linspace(0.0, 0.3, n_early, endpoint=False), np.linspace(0.3, 1.0, n_late)]).tolist()
        ts_p_early[0], ts_p_early[-1] = 0.0, 1.0
        schedules[f"piecewise_early_K{K}"] = {
            "sched_key": f"piecewise_early_K{K}",
            "name": f"Piecewise Early-Dense K={K}",
            "family": "piecewise",
            "K": K,
            "ts": ts_p_early,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

        n_early2 = int(round(K * 0.4))
        n_late2 = K - n_early2
        ts_p_late = np.concatenate([np.linspace(0.0, 0.7, n_early2, endpoint=False), np.linspace(0.7, 1.0, n_late2)]).tolist()
        ts_p_late[0], ts_p_late[-1] = 0.0, 1.0
        schedules[f"piecewise_late_K{K}"] = {
            "sched_key": f"piecewise_late_K{K}",
            "name": f"Piecewise Late-Dense K={K}",
            "family": "piecewise",
            "K": K,
            "ts": ts_p_late,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

        n1 = int(round(K * 0.25))
        n2 = int(round(K * 0.5))
        n3 = K - n1 - n2
        ts_p_mid = np.concatenate([np.linspace(0.0, 0.25, n1, endpoint=False), np.linspace(0.25, 0.75, n2, endpoint=False), np.linspace(0.75, 1.0, n3)]).tolist()
        ts_p_mid[0], ts_p_mid[-1] = 0.0, 1.0
        schedules[f"piecewise_middle_K{K}"] = {
            "sched_key": f"piecewise_middle_K{K}",
            "name": f"Piecewise Middle-Dense K={K}",
            "family": "piecewise",
            "K": K,
            "ts": ts_p_mid,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

        n_end1 = int(round(K * 0.35))
        n_mid2 = int(round(K * 0.3))
        n_end2 = K - n_end1 - n_mid2
        ts_p_end = np.concatenate([np.linspace(0.0, 0.25, n_end1, endpoint=False), np.linspace(0.25, 0.75, n_mid2, endpoint=False), np.linspace(0.75, 1.0, n_end2)]).tolist()
        ts_p_end[0], ts_p_end[-1] = 0.0, 1.0
        schedules[f"piecewise_endpoints_K{K}"] = {
            "sched_key": f"piecewise_endpoints_K{K}",
            "name": f"Piecewise Endpoints-Dense K={K}",
            "family": "piecewise",
            "K": K,
            "ts": ts_p_end,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

        # Family D: Random monotonic partitions
        for seed in [42, 123, 999]:
            rng = np.random.default_rng(seed)
            weights = rng.exponential(scale=1.0, size=K - 1)
            ts_rand = [0.0] + np.cumsum(weights / weights.sum()).tolist()
            ts_rand[-1] = 1.0
            schedules[f"random_K{K}_s{seed}"] = {
                "sched_key": f"random_K{K}_s{seed}",
                "name": f"Random Monotonic K={K} (seed={seed})",
                "family": "random",
                "K": K,
                "seed": seed,
                "ts": ts_rand,
                "recurrent_flops": K * FLOPS_PER_STEP,
            }

    return schedules


def compute_metrics_for_predictions(preds, batch, device):
    """Compute depth and camera pose evaluation metrics."""
    gt_depths = batch[DataField.DEPTHS][0]
    gt_masks = batch[DataField.POINT_MASKS][0]
    gt_c2w = batch[DataField.EXTRINSICS_C2W][0]
    gt_world_pts = batch[DataField.WORLD_POINTS][0]

    pred_depths = preds[PredictionField.DEPTHS][0]
    pred_world_pts = preds[PredictionField.WORLD_POINTS][0]
    pred_c2w_raw = preds[PredictionField.CAMERAS][0].camera_to_worlds
    S = pred_depths.shape[0]

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

    metrics_list = []
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
        rot_err = so3_relative_angle(r_pred_v, r_gt_v).item() * (180.0 / np.pi)

        t_pred_v = rel_pred[v, :3, 3]
        t_gt_v = rel_gt[v, :3, 3]
        t_norm_pred = torch.norm(t_pred_v).clamp(min=1e-6)
        t_norm_gt = torch.norm(t_gt_v).clamp(min=1e-6)
        cos_sim = (torch.dot(t_pred_v, t_gt_v) / (t_norm_pred * t_norm_gt)).clamp(-1.0, 1.0)
        trans_err = torch.acos(cos_sim).item() * (180.0 / np.pi)

        metrics_list.append({
            "abs_rel": abs_rel,
            "rmse": rmse,
            "mae": mae,
            "delta1": delta1,
            "geom_l2": geom_l2,
            "rot_err_deg": rot_err,
            "trans_err_deg": trans_err,
        })
    return metrics_list


def run_schedule_forward(inner, x_init, rope_pos, B, S, ts, collect_diagnostics=False):
    """Executes the dense recurrent refinement loop for an arbitrary time schedule ts."""
    device = x_init.device
    block = inner.recurrent_blocks[0]
    P = x_init.shape[1]
    dim = x_init.shape[2]

    x_curr = x_init.clone()
    K = len(ts)
    step_diags = []

    for i in range(K):
        t_now = ts[i]
        t_next = ts[i + 1] if i + 1 < K else 1.0
        t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)

        # 1. Frame attention
        x_frame = block.frame_attn(x_curr, t_pair.expand(B * S, -1), pos=rope_pos.reshape(B * S, P, 2))

        # 2. Global attention
        x_global = block.global_attn(x_frame.reshape(B, S * P, dim), t_pair.expand(B, -1), pos=None).reshape(B * S, P, dim)

        if collect_diagnostics:
            with torch.no_grad():
                frame_res = (x_frame - x_curr).norm(dim=-1).mean().item()
                global_res = (x_global - x_frame).norm(dim=-1).mean().item()
                hidden_chg = (x_global - x_curr).norm(dim=-1).mean().item()
                step_diags.append({
                    "step_idx": i,
                    "t_now": float(t_now),
                    "t_next": float(t_next),
                    "t_mid": float(0.5 * (t_now + t_next)),
                    "delta_t": float(t_next - t_now),
                    "frame_residual_norm": frame_res,
                    "global_residual_norm": global_res,
                    "hidden_change_norm": hidden_chg,
                })

        x_curr = x_global

    if collect_diagnostics:
        return x_curr, step_diags
    return x_curr


def run_nonuniform_evaluation(data_root="datasets/test/dtu", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    scans = ["scan1", "scan4", "scan9", "scan24", "scan62"]
    subsets_def = [
        {"name": "subset_middle", "ranking": "middle_first", "sampling": "first"},
        {"name": "subset_uniform", "ranking": "index", "sampling": "uniform"},
    ]

    print("=" * 95)
    print("DVLT Non-Uniform Schedule V0: Dense Continuous-Time Partition Kill-Test")
    print(f"Scans (5): {scans}")
    print(f"Subsets: {[s['name'] for s in subsets_def]} (10 sequences, 60 views)")
    print("=" * 95)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    print("Loading pretrained DVLT model...")
    model = DVLT(img_size=504, depth_head_type="conv")
    model.load_pretrained("nvidia/dvlt", strict=True)
    model.setup_test(accelerator)
    inner = model.model
    inner.eval()

    test_config = {
        "normalize_scene": False,
        "load_data_fields": ["images", "extrinsics_c2w", "intrinsics", "depths", "world_points", "point_masks"],
    }

    schedules = generate_all_schedules()
    print(f"Total schedules configured: {len(schedules)}")

    # Save schedule definitions
    sched_def_path = os.path.join(output_dir, "nonuniform_v0_schedule_definitions.json")
    with open(sched_def_path, "w", encoding="utf-8") as f:
        json.dump(schedules, f, indent=2)
    print(f"Saved schedule definitions: {sched_def_path}")

    # Phase 1: Quality evaluation across 10 sequences
    print("\nPhase 1: Evaluating Reconstruction Quality Across All Sequences...")
    raw_csv = os.path.join(output_dir, "nonuniform_v0_sequence_metrics.csv")
    quality_records = []
    interval_diags = []
    bench_sample = None

    for scan_name in scans:
        for sub in subsets_def:
            seq_id = f"{scan_name}_{sub['name']}"
            print(f"Processing sequence: {seq_id}")

            cfg = OmegaConf.create({
                "target": "dtu.DTU",
                "params": {"root_path": data_root}
            })
            ds = DataverseEvalDataset(dataverse_cfg=cfg, view_ranking=sub["ranking"], view_sampling=sub["sampling"], max_frames=6)
            ds.set_image_params(504, 14)
            video_idx = None
            for idx in range(ds.ds.num_videos()):
                if ds.ds._get_scan_name(idx) == scan_name:
                    video_idx = idx
                    break
            assert video_idx is not None

            ms_ds = MultiSourceDataset({"dtu": ds}, training=False, **test_config)
            sample = ms_ds[video_idx]
            batch = default_collate_fn([sample])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            images = batch[DataField.IMAGES]
            B, S, _, H, W = images.shape

            with torch.no_grad(), accelerator.autocast():
                z_0 = inner._encode_images(images)
                reg_tok = inner.register_token.expand(B, S, -1, -1).reshape(B * S, inner.num_register_tokens, -1)
                cam_tok = _slice_expand_flatten(inner.camera_token, B, S)
                x_init = torch.cat([cam_tok, reg_tok, z_0], dim=1)
                rope_pos = inner._get_rope_positions(B * S, H, W, images.device)

                if bench_sample is None:
                    bench_sample = {
                        "seq_id": seq_id,
                        "x_init": x_init.detach().clone(),
                        "rope_pos": rope_pos.detach().clone(),
                        "B": B,
                        "S": S,
                    }

                for sched_key, sched in schedules.items():
                    ts = sched["ts"]
                    # Collect diagnostics on uniform baselines and representative power schedules
                    need_diags = (sched["family"] == "baseline") or (sched_key.startswith("power") and sched.get("gamma") in [0.5, 2.0])
                    
                    if need_diags:
                        x_out, diags = run_schedule_forward(inner, x_init, rope_pos, B, S, ts, collect_diagnostics=True)
                        for d in diags:
                            d["seq_id"] = seq_id
                            d["sched_key"] = sched_key
                            d["K"] = sched["K"]
                            interval_diags.append(d)
                    else:
                        x_out = run_schedule_forward(inner, x_init, rope_pos, B, S, ts, collect_diagnostics=False)

                    preds = model._postprocess_predictions(batch, inner._decode(x_out, H, W, B, S, rope_pos))
                    view_metrics = compute_metrics_for_predictions(preds, batch, device)

                    m_absrel = np.mean([vm["abs_rel"] for vm in view_metrics])
                    m_rmse = np.mean([vm["rmse"] for vm in view_metrics])
                    m_rot = np.mean([vm["rot_err_deg"] for vm in view_metrics])
                    m_trans = np.mean([vm["trans_err_deg"] for vm in view_metrics])

                    quality_records.append({
                        "seq_id": seq_id,
                        "scan": scan_name,
                        "subset": sub["name"],
                        "sched_key": sched_key,
                        "sched_name": sched["name"],
                        "family": sched["family"],
                        "K": sched["K"],
                        "abs_rel": float(m_absrel),
                        "rmse": float(m_rmse),
                        "rot_err_deg": float(m_rot),
                        "trans_err_deg": float(m_trans),
                        "recurrent_flops_gflops": sched["recurrent_flops"],
                    })

    # Save sequence metrics CSV
    pd.DataFrame(quality_records).to_csv(raw_csv, index=False)
    print(f"Saved sequence quality metrics: {raw_csv}")

    # Save interval diagnostics
    diag_csv = os.path.join(output_dir, "nonuniform_v0_interval_diagnostics.csv")
    pd.DataFrame(interval_diags).to_csv(diag_csv, index=False)
    diag_json = os.path.join(output_dir, "nonuniform_v0_interval_diagnostics.json")
    with open(diag_json, "w", encoding="utf-8") as f:
        json.dump(interval_diags, f, indent=2)
    print(f"Saved interval diagnostics: {diag_csv}")

    # Phase 2: Benchmarking Real GPU Recurrent Latency and Peak VRAM
    print("\nPhase 2: Benchmarking Real GPU Recurrent Latency (torch.cuda.Event, 3 warmups, 10 reps)...")
    timing_results = {}

    assert bench_sample is not None
    x_b = bench_sample["x_init"]
    rope_b = bench_sample["rope_pos"]
    B_b = bench_sample["B"]
    S_b = bench_sample["S"]

    for sched_key, sched in schedules.items():
        ts = sched["ts"]

        # Warmup
        for _ in range(3):
            with torch.no_grad(), accelerator.autocast():
                _ = run_schedule_forward(inner, x_b, rope_b, B_b, S_b, ts, collect_diagnostics=False)
        torch.cuda.synchronize()

        # Timed repetitions
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]

        torch.cuda.reset_peak_memory_stats()
        for rep in range(10):
            start_events[rep].record()
            with torch.no_grad(), accelerator.autocast():
                _ = run_schedule_forward(inner, x_b, rope_b, B_b, S_b, ts, collect_diagnostics=False)
            end_events[rep].record()
        torch.cuda.synchronize()

        latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        mean_lat = float(np.mean(latencies))
        std_lat = float(np.std(latencies))

        timing_results[sched_key] = {
            "mean_latency_ms": mean_lat,
            "std_latency_ms": std_lat,
            "peak_vram_mb": peak_vram_mb,
        }
        print(f"  {sched['name']:<34} | {mean_lat:6.2f} +/- {std_lat:4.2f} ms | VRAM: {peak_vram_mb:6.1f} MB")

    # Phase 3: Aggregated Summary Table
    df_q = pd.DataFrame(quality_records)
    summary_list = []

    uniform_14_absrel = df_q[df_q["sched_key"] == "uniform_K14"]["abs_rel"].mean()
    uniform_14_lat = timing_results["uniform_K14"]["mean_latency_ms"]
    uniform_16_absrel = df_q[df_q["sched_key"] == "uniform_K16"]["abs_rel"].mean()
    uniform_16_lat = timing_results["uniform_K16"]["mean_latency_ms"]

    for sched_key, sched in schedules.items():
        sub_df = df_q[df_q["sched_key"] == sched_key]
        m_absrel = float(sub_df["abs_rel"].mean())
        m_rmse = float(sub_df["rmse"].mean())
        m_rot = float(sub_df["rot_err_deg"].mean())
        m_trans = float(sub_df["trans_err_deg"].mean())

        t_res = timing_results[sched_key]
        lat = t_res["mean_latency_ms"]
        flops = sched["recurrent_flops"]

        # Comparison vs Uniform K=14 (The Primary Hurdle)
        lat_gain_vs_u14 = (uniform_14_lat - lat) / uniform_14_lat * 100.0
        absrel_delta_vs_u14 = (m_absrel - uniform_14_absrel) / uniform_14_absrel * 100.0

        # Comparison vs same-K uniform baseline
        same_k_uniform_key = f"uniform_K{sched['K']}"
        same_k_absrel = float(df_q[df_q["sched_key"] == same_k_uniform_key]["abs_rel"].mean())
        absrel_gain_vs_same_k = (same_k_absrel - m_absrel) / same_k_absrel * 100.0  # positive = better than same-K

        matches_or_beats_u14 = bool(m_absrel <= uniform_14_absrel)
        is_le_12_steps = bool(sched["K"] <= 12)
        improves_over_same_k = bool(m_absrel < same_k_absrel)
        is_go = bool(matches_or_beats_u14 and is_le_12_steps and (lat_gain_vs_u14 > 5.0))

        summary_list.append({
            "sched_key": sched_key,
            "sched_name": sched["name"],
            "family": sched["family"],
            "K": sched["K"],
            "abs_rel": m_absrel,
            "rmse": m_rmse,
            "rot_err_deg": m_rot,
            "trans_err_deg": m_trans,
            "recurrent_flops_gflops": flops,
            "latency_ms": lat,
            "latency_std_ms": t_res["std_latency_ms"],
            "peak_vram_mb": t_res["peak_vram_mb"],
            "lat_gain_vs_uniform14_pct": lat_gain_vs_u14,
            "absrel_delta_vs_uniform14_pct": absrel_delta_vs_u14,
            "absrel_gain_vs_same_k_pct": absrel_gain_vs_same_k,
            "matches_uniform14_quality": matches_or_beats_u14,
            "improves_over_same_k": improves_over_same_k,
            "verdict_go_candidate": is_go,
        })

    summary_df = pd.DataFrame(summary_list)
    summary_csv = os.path.join(output_dir, "nonuniform_v0_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved aggregated summary table: {summary_csv}")

    summary_json = os.path.join(output_dir, "nonuniform_v0_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, indent=2)
    print(f"Saved aggregated summary JSON: {summary_json}")

    print("\n" + "=" * 125)
    print(f"{'Schedule Name':<35} | {'K':<3} | {'AbsRel':<9} | {'RMSE':<8} | {'FLOPs (GF)':<10} | {'Latency (ms)':<13} | {'dLat vs K=14':<12} | {'Matches K=14'}")
    print("-" * 125)
    for _, r in summary_df.iterrows():
        match_str = "YES" if r["matches_uniform14_quality"] else "NO"
        print(f"{r['sched_name']:<35} | {r['K']:<3} | {r['abs_rel']:<9.6f} | {r['rmse']:<8.4f} | {r['recurrent_flops_gflops']:<10.1f} | {r['latency_ms']:6.2f} +/- {r['latency_std_ms']:4.2f} | {r['lat_gain_vs_uniform14_pct']:+6.2f}%      | {match_str}")
    print("=" * 125)

    return summary_df


if __name__ == "__main__":
    run_nonuniform_evaluation()
