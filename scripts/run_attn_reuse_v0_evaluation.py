"""DVLT Attention-Reuse V0: Cross-Iteration Global Attention Reuse Kill-Test.

Evaluates:
- Baselines: Fixed K=16, Fixed i=15, Fixed i=14, Fixed i=13
- Variant A: Global Skip (skip_16, skip_15, skip_15_16, skip_14_16, skip_even_after_10, skip_13_15)
- Variant B: Previous Global Residual Reuse (alpha in {0.5, 0.75, 1.0})
- Variant C: Frozen Global K/V (reuse cached K/V from refresh step)

Metrics reported:
- AbsRel, RMSE, camera rotation error (deg), camera translation error (deg)
- Recurrent FLOPs (theoretical GFLOPs)
- Real GPU recurrent latency (torch.cuda.Event, 3 warmups, 10 repetitions)
- Peak VRAM (MB)
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


# Flop constants for DVLT (B=1, S=6, P=977 tokens/view, C=768, N=S*P=5862)
# Frame attention: 24*S*P*C^2 + 4*S*P^2*C = 100.576 GFLOPs
# Global attention (Full): 24*N*C^2 + 4*N^2*C = 188.559 GFLOPs
# Global attention (Frozen K/V): Q proj only saves 4*N*C^2 = 13.830 GFLOPs -> 174.729 GFLOPs
# Global attention (Skipped): 0 GFLOPs
FLOPS_FRAME_STEP = 100.576315  # GFLOPs
FLOPS_GLOBAL_FULL = 188.558673  # GFLOPs
FLOPS_GLOBAL_FROZEN_KV = 174.728950  # GFLOPs


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


def run_recurrent_schedule(inner, x_init, rope_pos, B, S, ts, sched):
    """Executes the recurrent refinement trajectory under a given reuse schedule."""
    device = x_init.device
    block = inner.recurrent_blocks[0]
    g_block = block.global_attn
    attn_mod = g_block.attn
    dim = x_init.shape[2]
    P = x_init.shape[1]

    x_curr = x_init.clone()
    total_steps = sched["total_steps"]
    stype = sched["type"]
    skip_global = sched.get("skip_global", set())
    alpha = sched.get("alpha", 1.0)

    r_last = None
    cached_k = None
    cached_v = None

    for i in range(total_steps):
        step_i = i + 1
        t_now = ts[i]
        t_next = ts[i + 1] if i + 1 < len(ts) else 1.0
        t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)

        # 1. Frame attention (runs every iteration across all views)
        x_frame = block.frame_attn(x_curr, t_pair.expand(B * S, -1), pos=rope_pos.reshape(B * S, P, 2))
        x_global_in = x_frame.reshape(B, S * P, dim)

        # 2. Global attention handling
        if stype == "fixed":
            x_global_out = g_block(x_global_in, t_pair.expand(B, -1), pos=None)
            x_curr = x_global_out.reshape(B * S, P, dim)

        elif stype == "variantA":  # Global Skip
            if step_i in skip_global:
                x_global_out = x_global_in  # skip global attention entirely
            else:
                x_global_out = g_block(x_global_in, t_pair.expand(B, -1), pos=None)
            x_curr = x_global_out.reshape(B * S, P, dim)

        elif stype == "variantB":  # Previous Global Residual Reuse
            if step_i in skip_global:
                assert r_last is not None, f"No previous residual cached for step {step_i}"
                x_global_out = x_global_in + alpha * r_last
            else:
                x_global_out = g_block(x_global_in, t_pair.expand(B, -1), pos=None)
                r_last = x_global_out - x_global_in
            x_curr = x_global_out.reshape(B * S, P, dim)

        elif stype == "variantC":  # Frozen Global K/V
            normed = g_block.norm1(x_global_in)
            s = g_block.depth_scale(t_pair.expand(B, -1)).unsqueeze(1)
            s_attn, s_mlp, s_out = s.chunk(3, dim=-1)

            if step_i in skip_global:
                assert cached_k is not None, f"No cached K/V for step {step_i}"
                # Sliced Q projection only
                w_q = attn_mod.qkv.weight[:dim]
                b_q = attn_mod.qkv.bias[:dim] if attn_mod.qkv.bias is not None else None
                q = F.linear(normed, w_q, b_q).reshape(B, S * P, attn_mod.num_heads, attn_mod.head_dim).transpose(1, 2)
                q = attn_mod.q_norm(q)
                # SDPA with cached K/V
                attn_out = F.scaled_dot_product_attention(q, cached_k, cached_v)
            else:
                # Refresh step: compute full QKV and cache K, V
                qkv = attn_mod.qkv(normed).reshape(B, S * P, 3, attn_mod.num_heads, attn_mod.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q = attn_mod.q_norm(q)
                k = attn_mod.k_norm(k)
                cached_k = k
                cached_v = v
                attn_out = F.scaled_dot_product_attention(q, k, v)

            attn_out = attn_out.transpose(1, 2).reshape(B, S * P, dim)
            attn_out = attn_mod.proj(attn_out)
            branch = g_block.ls1(attn_out)
            if s_attn is not None:
                branch = s_attn * branch
            x_mid = x_global_in + branch
            branch2 = g_block.ls2(g_block.mlp(g_block.norm2(x_mid)))
            if s_mlp is not None:
                branch2 = s_mlp * branch2
            x_global_out = s_out * (x_mid + branch2)
            x_curr = x_global_out.reshape(B * S, P, dim)
        else:
            raise ValueError(f"Unknown schedule type: {stype}")

    return x_curr


def calculate_schedule_flops(sched):
    total_steps = sched["total_steps"]
    stype = sched["type"]
    skip_global = sched.get("skip_global", set())

    total_flops = 0.0
    for i in range(total_steps):
        step_i = i + 1
        total_flops += FLOPS_FRAME_STEP
        if stype == "fixed":
            total_flops += FLOPS_GLOBAL_FULL
        elif stype in ["variantA", "variantB"]:
            if step_i not in skip_global:
                total_flops += FLOPS_GLOBAL_FULL
        elif stype == "variantC":
            if step_i in skip_global:
                total_flops += FLOPS_GLOBAL_FROZEN_KV
            else:
                total_flops += FLOPS_GLOBAL_FULL
    return total_flops


def get_all_schedules():
    schedules = {}

    # 1. Baselines (Common K=16 linspace trajectory prefixes)
    schedules["fixed_16"] = {"name": "Fixed i=16", "type": "fixed", "total_steps": 16, "skip_global": set()}
    schedules["fixed_15"] = {"name": "Fixed i=15", "type": "fixed", "total_steps": 15, "skip_global": set()}
    schedules["fixed_14"] = {"name": "Fixed i=14", "type": "fixed", "total_steps": 14, "skip_global": set()}
    schedules["fixed_13"] = {"name": "Fixed i=13", "type": "fixed", "total_steps": 13, "skip_global": set()}

    # 2. Variant A: Global Skip
    schedules["varA_skip_16"] = {"name": "VarA: Skip 16", "type": "variantA", "total_steps": 16, "skip_global": {16}}
    schedules["varA_skip_15"] = {"name": "VarA: Skip 15", "type": "variantA", "total_steps": 16, "skip_global": {15}}
    schedules["varA_skip_15_16"] = {"name": "VarA: Skip {15,16}", "type": "variantA", "total_steps": 16, "skip_global": {15, 16}}
    schedules["varA_skip_14_16"] = {"name": "VarA: Skip {14,16}", "type": "variantA", "total_steps": 16, "skip_global": {14, 16}}
    schedules["varA_skip_even_after_10"] = {"name": "VarA: Skip {12,14,16}", "type": "variantA", "total_steps": 16, "skip_global": {12, 14, 16}}
    schedules["varA_skip_13_15"] = {"name": "VarA: Skip {13,15}", "type": "variantA", "total_steps": 16, "skip_global": {13, 15}}

    # 3. Variant B: Previous Global Residual Reuse
    b_skips = {
        "reuse_16": {16},
        "reuse_15": {15},
        "reuse_15_16": {15, 16},
        "reuse_14_16": {14, 16},
        "reuse_even_after_10": {12, 14, 16},
    }
    for b_key, b_set in b_skips.items():
        for alpha in [0.5, 0.75, 1.0]:
            sched_key = f"varB_{b_key}_a{alpha}"
            schedules[sched_key] = {
                "name": f"VarB: {b_key} (alpha={alpha})",
                "type": "variantB",
                "total_steps": 16,
                "skip_global": b_set,
                "alpha": alpha,
            }

    # 4. Variant C: Frozen Global K/V
    schedules["varC_frozen_kv_16"] = {"name": "VarC: Freeze K/V 16", "type": "variantC", "total_steps": 16, "skip_global": {16}}
    schedules["varC_frozen_kv_15"] = {"name": "VarC: Freeze K/V 15", "type": "variantC", "total_steps": 16, "skip_global": {15}}
    schedules["varC_frozen_kv_15_16"] = {"name": "VarC: Freeze K/V {15,16}", "type": "variantC", "total_steps": 16, "skip_global": {15, 16}}
    schedules["varC_frozen_kv_14_16"] = {"name": "VarC: Freeze K/V {14,16}", "type": "variantC", "total_steps": 16, "skip_global": {14, 16}}
    schedules["varC_frozen_kv_even_after_10"] = {"name": "VarC: Freeze K/V {12,14,16}", "type": "variantC", "total_steps": 16, "skip_global": {12, 14, 16}}

    return schedules


def run_full_evaluation(data_root="datasets/test/dtu", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    scans = ["scan1", "scan4", "scan9", "scan24", "scan62"]
    subsets_def = [
        {"name": "subset_middle", "ranking": "middle_first", "sampling": "first"},
        {"name": "subset_uniform", "ranking": "index", "sampling": "uniform"},
    ]

    print("=" * 90)
    print("DVLT Attention-Reuse V0: Comprehensive Feasibility & Kill-Test Evaluation")
    print(f"Scans (5): {scans}")
    print(f"Subsets: {[s['name'] for s in subsets_def]} (10 sequences, 60 views)")
    print("=" * 90)

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

    schedules = get_all_schedules()
    print(f"Total schedules configured: {len(schedules)}")
    for k, s in schedules.items():
        s["recurrent_flops"] = calculate_schedule_flops(s)

    # 1. Quality evaluation across all 10 sequences
    print("\nPhase 1: Evaluating Quality Across All Sequences...")
    raw_csv = os.path.join(output_dir, "attn_reuse_v0_sequence_metrics.csv")
    quality_records = []
    bench_sample = None
    ts = torch.linspace(0.0, 1.0, 16).tolist()

    if os.path.exists(raw_csv) and len(pd.read_csv(raw_csv)) == len(schedules) * 10:
        print(f"Loading existing quality metrics ({len(pd.read_csv(raw_csv))} records) from {raw_csv}...")
        quality_records = pd.read_csv(raw_csv).to_dict(orient="records")

        # Capture one bench sample for timing phase
        cfg0 = OmegaConf.create({"target": "dtu.DTU", "params": {"root_path": data_root}})
        ds0 = DataverseEvalDataset(dataverse_cfg=cfg0, view_ranking="middle_first", view_sampling="first", max_frames=6)
        ds0.set_image_params(504, 14)
        ms_ds0 = MultiSourceDataset({"dtu": ds0}, training=False, **test_config)
        sample0 = ms_ds0[0]
        batch0 = default_collate_fn([sample0])
        for k, v in batch0.items():
            if isinstance(v, torch.Tensor):
                batch0[k] = v.to(device)
        images0 = batch0[DataField.IMAGES]
        B0, S0, _, H0, W0 = images0.shape
        with torch.no_grad(), accelerator.autocast():
            z_0 = inner._encode_images(images0)
            reg_tok = inner.register_token.expand(B0, S0, -1, -1).reshape(B0 * S0, inner.num_register_tokens, -1)
            cam_tok = _slice_expand_flatten(inner.camera_token, B0, S0)
            x_init0 = torch.cat([cam_tok, reg_tok, z_0], dim=1)
            rope_pos0 = inner._get_rope_positions(B0 * S0, H0, W0, images0.device)
            bench_sample = {
                "seq_id": "bench_scan1",
                "x_init": x_init0.detach().clone(),
                "rope_pos": rope_pos0.detach().clone(),
                "B": B0,
                "S": S0,
            }
    else:
        for scan_name in scans:
            for sub in subsets_def:
                seq_id = f"{scan_name}_{sub['name']}"
                print(f"Processing sequence for quality: {seq_id}")

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
                    register_token = inner.register_token.expand(B, S, -1, -1).reshape(B * S, inner.num_register_tokens, -1)
                    camera_token = _slice_expand_flatten(inner.camera_token, B, S)
                    x_init = torch.cat([camera_token, register_token, z_0], dim=1)
                    rope_pos = inner._get_rope_positions(B * S, H, W, images.device)

                    # Store first batch for timing phase
                    if bench_sample is None:
                        bench_sample = {
                            "seq_id": seq_id,
                            "x_init": x_init.detach().clone(),
                            "rope_pos": rope_pos.detach().clone(),
                            "B": B,
                            "S": S,
                        }

                    for sched_key, sched in schedules.items():
                        x_out = run_recurrent_schedule(inner, x_init, rope_pos, B, S, ts, sched)
                        preds = model._postprocess_predictions(batch, inner._decode(x_out, H, W, B, S, rope_pos))
                        view_metrics = compute_metrics_for_predictions(preds, batch, device)

                        # Compute sequence-mean metrics
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
                            "sched_type": sched["type"],
                            "total_steps": sched["total_steps"],
                            "abs_rel": m_absrel,
                            "rmse": m_rmse,
                            "rot_err_deg": m_rot,
                            "trans_err_deg": m_trans,
                            "recurrent_flops_gflops": sched["recurrent_flops"],
                        })

    # Save per-sequence quality metrics
    raw_csv = os.path.join(output_dir, "attn_reuse_v0_sequence_metrics.csv")
    pd.DataFrame(quality_records).to_csv(raw_csv, index=False)
    print(f"Saved sequence quality metrics: {raw_csv}")

    # 2. Benchmarking Real GPU Recurrent Latency and Peak VRAM
    print("\nPhase 2: Benchmarking Real GPU Recurrent Latency (torch.cuda.Event, 3 warmups, 10 reps)...")
    timing_results = {}

    assert bench_sample is not None, "bench_sample not captured"
    x_b = bench_sample["x_init"]
    rope_b = bench_sample["rope_pos"]
    B_b = bench_sample["B"]
    S_b = bench_sample["S"]

    for sched_key, sched in schedules.items():
        # Warmup
        for _ in range(3):
            with torch.no_grad(), accelerator.autocast():
                _ = run_recurrent_schedule(inner, x_b, rope_b, B_b, S_b, ts, sched)
        torch.cuda.synchronize()

        # Timed repetitions
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]

        torch.cuda.reset_peak_memory_stats()
        for rep in range(10):
            start_events[rep].record()
            with torch.no_grad(), accelerator.autocast():
                _ = run_recurrent_schedule(inner, x_b, rope_b, B_b, S_b, ts, sched)
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
        print(f"  {sched['name']:<30} | {mean_lat:6.2f} +/- {std_lat:4.2f} ms | VRAM: {peak_vram_mb:6.1f} MB")

    # 3. Aggregate Summary Table
    df_q = pd.DataFrame(quality_records)
    summary_list = []

    fixed_16_absrel = df_q[df_q["sched_key"] == "fixed_16"]["abs_rel"].mean()
    fixed_14_absrel = df_q[df_q["sched_key"] == "fixed_14"]["abs_rel"].mean()
    fixed_14_lat = timing_results["fixed_14"]["mean_latency_ms"]
    fixed_16_lat = timing_results["fixed_16"]["mean_latency_ms"]

    for sched_key, sched in schedules.items():
        sub_df = df_q[df_q["sched_key"] == sched_key]
        m_absrel = sub_df["abs_rel"].mean()
        m_rmse = sub_df["rmse"].mean()
        m_rot = sub_df["rot_err_deg"].mean()
        m_trans = sub_df["trans_err_deg"].mean()

        t_res = timing_results[sched_key]
        lat = t_res["mean_latency_ms"]
        flops = sched["recurrent_flops"]

        lat_gain_vs_14 = (fixed_14_lat - lat) / fixed_14_lat * 100.0
        lat_gain_vs_16 = (fixed_16_lat - lat) / fixed_16_lat * 100.0
        quality_delta_vs_14 = (m_absrel - fixed_14_absrel) / fixed_14_absrel * 100.0
        quality_delta_vs_16 = (m_absrel - fixed_16_absrel) / fixed_16_absrel * 100.0

        matches_or_beats_14 = (m_absrel <= fixed_14_absrel)
        meets_5pct_latency_vs_14 = (lat_gain_vs_14 >= 5.0)
        is_go = matches_or_beats_14 and meets_5pct_latency_vs_14

        summary_list.append({
            "sched_key": sched_key,
            "sched_name": sched["name"],
            "sched_type": sched["type"],
            "total_steps": sched["total_steps"],
            "abs_rel": m_absrel,
            "rmse": m_rmse,
            "rot_err_deg": m_rot,
            "trans_err_deg": m_trans,
            "recurrent_flops_gflops": flops,
            "latency_ms": lat,
            "latency_std_ms": t_res["std_latency_ms"],
            "peak_vram_mb": t_res["peak_vram_mb"],
            "lat_gain_vs_fixed14_pct": lat_gain_vs_14,
            "lat_gain_vs_fixed16_pct": lat_gain_vs_16,
            "absrel_delta_vs_fixed14_pct": quality_delta_vs_14,
            "absrel_delta_vs_fixed16_pct": quality_delta_vs_16,
            "matches_fixed14_quality": matches_or_beats_14,
            "meets_5pct_speedup_vs_fixed14": meets_5pct_latency_vs_14,
            "verdict_go_candidate": is_go,
        })

    summary_df = pd.DataFrame(summary_list)
    summary_csv = os.path.join(output_dir, "attn_reuse_v0_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved aggregated summary table: {summary_csv}")

    summary_json = os.path.join(output_dir, "attn_reuse_v0_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, indent=2)
    print(f"Saved aggregated summary JSON: {summary_json}")

    print("\n" + "=" * 115)
    print(f"{'Schedule Name':<32} | {'AbsRel':<9} | {'RMSE':<8} | {'Rot (deg)':<9} | {'FLOPs (GF)':<10} | {'Latency (ms)':<13} | {'dLat vs i=14':<12} | {'Matches i=14'}")
    print("-" * 115)
    for _, r in summary_df.iterrows():
        match_str = "YES" if r["matches_fixed14_quality"] else "NO"
        print(f"{r['sched_name']:<32} | {r['abs_rel']:<9.6f} | {r['rmse']:<8.4f} | {r['rot_err_deg']:<9.4f} | {r['recurrent_flops_gflops']:<10.1f} | {r['latency_ms']:6.2f} +/- {r['latency_std_ms']:4.2f} | {r['lat_gain_vs_fixed14_pct']:+6.2f}%      | {match_str}")
    print("=" * 115)

    return summary_df


if __name__ == "__main__":
    run_full_evaluation()
