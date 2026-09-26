"""DVLT Interval-Robust Training V0: Randomized Step-Size Fine-Tuning Kill-Test.

Tests whether fine-tuning DVLT's interval-conditioned recurrent block on randomized
time step intervals (Delta_t) resolves the training-distribution mismatch and enables
fewer recurrent steps (K=12 or K=10) to match or beat Pretrained Uniform K=14.

Training Scope (Stage A):
- Freeze entire pretrained DVLT except IntervalDepthScaling (proj MLPs: exactly 316,032 params).
- 9 disjoint DTU training scans: scan10, 11, 12, 15, 23, 29, 33, 48, 110.
- AdamW (lr=1e-4, weight_decay=1e-4), 200 steps each.
- BF16 mixed precision, gradient checkpointing.

Models:
- Control 1: Pretrained DVLT (no training).
- Control 2: Uniform linspace fine-tuning (Delta_t = const).
- Treatment 1: Mild randomization (sigma=0.2).
- Treatment 2: Moderate randomization (sigma=0.5, clamped [0.5x, 2.0x]).

Evaluation:
- 5 disjoint DTU validation scans x 2 subsets = 10 sequences, 60 views.
- Uniform schedules: K in {10, 12, 14, 16}.
- Candidate non-uniform schedules for K in {10, 12}.
- Perturbation grid: CV(Delta_t) in {0.0, 0.1, 0.2, 0.4, 0.6}.
"""

import copy
import csv
import json
import os
import time
import types
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf

from dvlt.common.constants import DataField, PredictionField
from dvlt.data.collate import default_collate_fn
from dvlt.data.datasets.multi_source import MultiSourceDataset
from dvlt.data.datasets.parser.dataverse import DataverseEvalDataset, DataverseTrainDataset
from dvlt.metric.depth import apply_alignment
from dvlt.metric.pose import so3_relative_angle
from dvlt.model.dvlt.model import DVLT, _slice_expand_flatten

# Compute constants
FLOPS_FRAME_STEP = 100.576315  # GFLOPs
FLOPS_GLOBAL_FULL = 188.558673  # GFLOPs
FLOPS_PER_STEP = FLOPS_FRAME_STEP + FLOPS_GLOBAL_FULL  # 289.134988 GFLOPs


def sample_interval_schedule(K: int, mode: str, rng: np.random.RandomState):
    """Generates continuous-time partition ts of length K on [0, 1]."""
    if mode == "uniform" or K <= 2:
        return torch.linspace(0.0, 1.0, K)
    
    M = K - 1
    if mode == "mild":
        # w_i ~ LogNormal(0, 0.2)
        w = rng.lognormal(mean=0.0, sigma=0.2, size=M)
        dt = w / np.sum(w)
    elif mode == "moderate":
        # w_i ~ LogNormal(0, 0.5), clamped to [0.5x, 2.0x] of uniform dt
        dt_unif = 1.0 / M
        w = rng.lognormal(mean=0.0, sigma=0.5, size=M)
        dt_raw = w / np.sum(w)
        dt_clamped = np.clip(dt_raw, 0.5 * dt_unif, 2.0 * dt_unif)
        dt = dt_clamped / np.sum(dt_clamped)
    else:
        raise ValueError(f"Unknown interval sampling mode: {mode}")

    ts = np.concatenate([[0.0], np.cumsum(dt)])
    ts[-1] = 1.0
    return torch.tensor(ts, dtype=torch.float32)


def generate_perturbed_schedule(K=12, target_cv=0.2, seed=42):
    """Generates deterministic perturbed schedule with target coefficient of variation."""
    if target_cv == 0.0:
        return np.linspace(0.0, 1.0, K).tolist(), 0.0
    rng = np.random.RandomState(seed)
    M = K - 1
    best_ts = None
    best_diff = 1e9
    best_cv = 0.0
    for scale in np.linspace(target_cv * 0.5, target_cv * 2.0, 100):
        z = rng.randn(M)
        w = np.exp(scale * (z - z.mean()))
        dt = w / w.sum()
        cv = dt.std() / dt.mean()
        if abs(cv - target_cv) < best_diff:
            best_diff = abs(cv - target_cv)
            best_cv = cv
            best_ts = np.concatenate([[0.0], np.cumsum(dt)]).tolist()
    best_ts[-1] = 1.0
    return best_ts, best_cv


def build_evaluation_schedules():
    """Builds standard evaluation schedules for K in {10, 12, 14, 16} and candidate non-uniform grids."""
    schedules = {}

    # Uniform baselines
    for K in [10, 12, 14, 16]:
        ts = np.linspace(0.0, 1.0, K).tolist()
        schedules[f"uniform_K{K}"] = {
            "sched_key": f"uniform_K{K}",
            "name": f"Uniform K={K}",
            "family": "baseline_uniform",
            "K": K,
            "ts": ts,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

    # Candidate non-uniform for K in {10, 12}
    for K in [10, 12]:
        k_grid = np.arange(K) / (K - 1)

        # Power gamma=0.75
        ts_p075 = (k_grid ** 0.75).tolist()
        ts_p075[0], ts_p075[-1] = 0.0, 1.0
        schedules[f"power_K{K}_g0.75"] = {
            "sched_key": f"power_K{K}_g0.75",
            "name": f"Power K={K} (gamma=0.75)",
            "family": "power",
            "K": K,
            "ts": ts_p075,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

        # Power gamma=1.25
        ts_p125 = (k_grid ** 1.25).tolist()
        ts_p125[0], ts_p125[-1] = 0.0, 1.0
        schedules[f"power_K{K}_g1.25"] = {
            "sched_key": f"power_K{K}_g1.25",
            "name": f"Power K={K} (gamma=1.25)",
            "family": "power",
            "K": K,
            "ts": ts_p125,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

        # Cosine endpoints-dense
        ts_end = (0.5 * (1.0 - np.cos(k_grid * np.pi))).tolist()
        ts_end[0], ts_end[-1] = 0.0, 1.0
        schedules[f"cosine_endpoints_K{K}"] = {
            "sched_key": f"cosine_endpoints_K{K}",
            "name": f"Cosine Endpoints K={K}",
            "family": "cosine",
            "K": K,
            "ts": ts_end,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

        # Random monotonic (seed 42)
        rng_rand = np.random.default_rng(42)
        weights = rng_rand.exponential(scale=1.0, size=K - 1)
        ts_rand = [0.0] + np.cumsum(weights / weights.sum()).tolist()
        ts_rand[-1] = 1.0
        schedules[f"random_K{K}_s42"] = {
            "sched_key": f"random_K{K}_s42",
            "name": f"Random Monotonic K={K} (s42)",
            "family": "random",
            "K": K,
            "ts": ts_rand,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

        # Sampled mild (from training distribution)
        ts_mild = sample_interval_schedule(K, "mild", np.random.RandomState(100 + K)).tolist()
        schedules[f"sampled_mild_K{K}"] = {
            "sched_key": f"sampled_mild_K{K}",
            "name": f"Sampled Mild K={K}",
            "family": "sampled_train_dist",
            "K": K,
            "ts": ts_mild,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

        # Sampled moderate (from training distribution)
        ts_mod = sample_interval_schedule(K, "moderate", np.random.RandomState(200 + K)).tolist()
        schedules[f"sampled_moderate_K{K}"] = {
            "sched_key": f"sampled_moderate_K{K}",
            "name": f"Sampled Moderate K={K}",
            "family": "sampled_train_dist",
            "K": K,
            "ts": ts_mod,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }

    return schedules


def build_perturbation_schedules(K=12):
    """Builds perturbation schedules for Section 8 diagnostics: CV in {0.0, 0.1, 0.2, 0.4, 0.6}."""
    pert_schedules = {}
    for target_cv in [0.0, 0.1, 0.2, 0.4, 0.6]:
        ts, actual_cv = generate_perturbed_schedule(K=K, target_cv=target_cv, seed=42)
        key = f"pert_K{K}_cv{int(target_cv*100):02d}"
        pert_schedules[key] = {
            "sched_key": key,
            "name": f"Perturbed K={K} CV={actual_cv:.3f}",
            "family": "perturbation_diagnostic",
            "K": K,
            "target_cv": target_cv,
            "actual_cv": actual_cv,
            "ts": ts,
            "recurrent_flops": K * FLOPS_PER_STEP,
        }
    return pert_schedules


def run_schedule_forward(inner, x_init, rope_pos, B, S, ts):
    """Executes dense recurrent refinement for an arbitrary time schedule ts."""
    device = x_init.device
    block = inner.recurrent_blocks[0]
    P = x_init.shape[1]
    dim = x_init.shape[2]

    x_curr = x_init.clone()
    K = len(ts)

    for i in range(K):
        t_now = ts[i]
        t_next = ts[i + 1] if i + 1 < K else 1.0
        t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)

        # 1. Frame attention
        x_frame = block.frame_attn(x_curr, t_pair.expand(B * S, -1), pos=rope_pos.reshape(B * S, P, 2))

        # 2. Global attention
        x_global = block.global_attn(x_frame.reshape(B, S * P, dim), t_pair.expand(B, -1), pos=None).reshape(B * S, P, dim)
        x_curr = x_global

    return x_curr


def compute_metrics_for_predictions(preds, batch, device):
    """Computes depth and camera pose evaluation metrics."""
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


def setup_custom_solve_train(inner, mode="uniform", rng=None):
    """Binds custom training solver with specified interval sampling distribution."""
    if rng is None:
        rng = np.random.RandomState(42)

    def custom_solve_train(self, x, rope_pos, B, S, sd_rng):
        K = self._sample_K(sd_rng)
        ts = sample_interval_schedule(K, mode=mode, rng=rng).tolist()

        for i in range(K):
            t_now = ts[i]
            t_next = ts[i + 1] if i + 1 < K else 1.0
            x = self._interval_step(x, t_now, t_next, rope_pos, B, S)

        return x

    inner._solve_train_linspace_k = types.MethodType(custom_solve_train, inner)


def train_stage_a_model(model, accelerator, train_samples, mode="uniform", seed=42, total_steps=200):
    """Fine-tunes Stage A IntervalDepthScaling parameters for total_steps."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    device = accelerator.device

    inner = model.model
    # 1. Freeze all parameters except IntervalDepthScaling
    trainable_params = []
    param_names = []
    for name, param in inner.named_parameters():
        if "depth_scale.proj" in name:
            param.requires_grad = True
            trainable_params.append(param)
            param_names.append(name)
        else:
            param.requires_grad = False

    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"[{mode.upper()} Seed {seed}] Trainable tensors: {len(trainable_params)}, params: {num_trainable:,}")

    # Bind custom interval solver
    setup_custom_solve_train(inner, mode=mode, rng=rng)

    # Enable gradient checkpointing and training mode
    inner.train()
    inner.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    optimizer, inner = accelerator.prepare(optimizer, inner)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    loss_history = []
    num_train_scans = len(train_samples)

    for step in range(total_steps):
        # Pick random sample among the 9 training scans
        scan_idx = rng.randint(0, num_train_scans)
        batch = train_samples[scan_idx]
        
        # Move batch to device
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(device)
            else:
                batch_gpu[k] = v

        optimizer.zero_grad()
        with accelerator.autocast():
            loss, pbar_logs, tracker_logs, _ = model.train_step(batch_gpu, step, accelerator)

        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if (step + 1) % 50 == 0 or step == total_steps - 1:
            print(f"  [{mode.upper()} Seed {seed}] Step {step+1:3d}/{total_steps} | Loss: {loss_val:.4f}")

    torch.cuda.synchronize()
    train_time = time.perf_counter() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Unwrap and set to eval mode
    inner = accelerator.unwrap_model(inner)
    inner.eval()
    model.model = inner

    return {
        "train_time_sec": train_time,
        "peak_vram_mb": peak_vram_mb,
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
        "param_names": param_names,
        "num_trainable": num_trainable,
    }


def evaluate_model_on_schedules(model, accelerator, val_batches, schedules_dict, model_label="model"):
    """Evaluates a model across a dictionary of schedules on the 10 validation sequences."""
    inner = model.model
    inner.eval()
    device = accelerator.device

    records = []
    for seq_id, batch in val_batches.items():
        images = batch[DataField.IMAGES]
        B, S, _, H, W = images.shape

        with torch.no_grad(), accelerator.autocast():
            z_0 = inner._encode_images(images)
            reg_tok = inner.register_token.expand(B, S, -1, -1).reshape(B * S, inner.num_register_tokens, -1)
            cam_tok = _slice_expand_flatten(inner.camera_token, B, S)
            x_init = torch.cat([cam_tok, reg_tok, z_0], dim=1)
            rope_pos = inner._get_rope_positions(B * S, H, W, images.device)

            for sched_key, sched in schedules_dict.items():
                ts = sched["ts"]
                x_out = run_schedule_forward(inner, x_init, rope_pos, B, S, ts)
                preds = model._postprocess_predictions(batch, inner._decode(x_out, H, W, B, S, rope_pos))
                view_metrics = compute_metrics_for_predictions(preds, batch, device)

                m_absrel = np.mean([vm["abs_rel"] for vm in view_metrics])
                m_rmse = np.mean([vm["rmse"] for vm in view_metrics])
                m_rot = np.mean([vm["rot_err_deg"] for vm in view_metrics])
                m_trans = np.mean([vm["trans_err_deg"] for vm in view_metrics])

                records.append({
                    "model": model_label,
                    "seq_id": seq_id,
                    "sched_key": sched_key,
                    "sched_name": sched["name"],
                    "family": sched["family"],
                    "K": sched["K"],
                    "abs_rel": float(m_absrel),
                    "rmse": float(m_rmse),
                    "rot_err_deg": float(m_rot),
                    "trans_err_deg": float(m_trans),
                    "recurrent_flops": sched["recurrent_flops"],
                    "target_cv": sched.get("target_cv", 0.0),
                    "actual_cv": sched.get("actual_cv", 0.0),
                })
    return pd.DataFrame(records)


def benchmark_model_latencies(inner, accelerator, bench_sample, schedules_dict):
    """Benchmarks real GPU recurrent latency and total inference latency using torch.cuda.Event."""
    device = accelerator.device
    x_b = bench_sample["x_init"]
    rope_b = bench_sample["rope_pos"]
    B_b = bench_sample["B"]
    S_b = bench_sample["S"]

    timing_results = {}
    for sched_key, sched in schedules_dict.items():
        ts = sched["ts"]

        # Warmup (3 runs)
        for _ in range(3):
            with torch.no_grad(), accelerator.autocast():
                _ = run_schedule_forward(inner, x_b, rope_b, B_b, S_b, ts)
        torch.cuda.synchronize()

        # Timed repetitions (10 runs)
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]

        torch.cuda.reset_peak_memory_stats()
        for rep in range(10):
            start_events[rep].record()
            with torch.no_grad(), accelerator.autocast():
                _ = run_schedule_forward(inner, x_b, rope_b, B_b, S_b, ts)
            end_events[rep].record()
        torch.cuda.synchronize()

        latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

        timing_results[sched_key] = {
            "recurrent_latency_ms": float(np.mean(latencies)),
            "recurrent_latency_std_ms": float(np.std(latencies)),
            "peak_vram_mb": float(peak_vram),
        }

    return timing_results


def run_experiment(data_root="datasets/test/dtu", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print("=" * 100)
    print("DVLT Interval-Robust Training V0: Randomized Step-Size Fine-Tuning Kill-Test")
    print("=" * 100)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    print(f"Device: {accelerator.device} | Precision: bf16")

    # 1. Dataset splits: strictly scene-disjoint
    train_scan_names = ["scan10", "scan11", "scan12", "scan15", "scan23", "scan29", "scan33", "scan48", "scan110"]
    val_scan_names = ["scan1", "scan4", "scan9", "scan24", "scan62"]
    subsets_def = [
        {"name": "subset_middle", "ranking": "middle_first", "sampling": "first"},
        {"name": "subset_uniform", "ranking": "index", "sampling": "uniform"},
    ]

    print(f"\nTrain Scans ({len(train_scan_names)}): {train_scan_names}")
    print(f"Val Scans   ({len(val_scan_names)}): {val_scan_names} (x 2 subsets = 10 sequences, 60 views)")

    # 2. Build training samples from the 9 training scans
    print("\nPreparing training batches from disjoint training scans...")
    cfg = OmegaConf.create({"target": "dtu.DTU", "params": {"root_path": data_root}})
    ds_train = DataverseTrainDataset(dataverse_cfg=cfg)
    ds_train.set_image_params(504, 14)

    # Map scan names to video indices
    train_video_indices = []
    for idx in range(ds_train.ds.num_videos()):
        sname = ds_train.ds._get_scan_name(idx)
        if sname in train_scan_names:
            train_video_indices.append(idx)
    assert len(train_video_indices) == len(train_scan_names), f"Found {len(train_video_indices)} train scans"

    MultiSourceDataset.MIN_LEN = 14
    train_config = {
        "normalize_scene": False,
        "load_data_fields": ["images", "extrinsics_c2w", "intrinsics", "depths", "world_points", "point_masks"],
    }
    ms_train = MultiSourceDataset({"dtu": ds_train}, training=True, **train_config)

    train_samples = []
    for v_idx in train_video_indices:
        sample = ms_train[(v_idx, 6, 1.0)]
        batch = default_collate_fn([sample])
        train_samples.append(batch)
    print(f"Successfully loaded {len(train_samples)} training sequence batches.")

    # 3. Build validation batches (10 sequences)
    print("Preparing validation batches...")
    test_config = {
        "normalize_scene": False,
        "load_data_fields": ["images", "extrinsics_c2w", "intrinsics", "depths", "world_points", "point_masks"],
    }
    val_batches = {}
    bench_sample = None

    for scan_name in val_scan_names:
        for sub in subsets_def:
            seq_id = f"{scan_name}_{sub['name']}"
            ds_val = DataverseEvalDataset(dataverse_cfg=cfg, view_ranking=sub["ranking"], view_sampling=sub["sampling"], max_frames=6)
            ds_val.set_image_params(504, 14)
            video_idx = None
            for idx in range(ds_val.ds.num_videos()):
                if ds_val.ds._get_scan_name(idx) == scan_name:
                    video_idx = idx
                    break
            assert video_idx is not None

            ms_val = MultiSourceDataset({"dtu": ds_val}, training=False, **test_config)
            sample = ms_val[video_idx]
            batch = default_collate_fn([sample])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            val_batches[seq_id] = batch

            if bench_sample is None:
                images = batch[DataField.IMAGES]
                B, S, _, H, W = images.shape
                # temporary dummy encoding to get bench_sample shape
                bench_sample = {
                    "images": images,
                    "batch": batch,
                    "B": B,
                    "S": S,
                    "H": H,
                    "W": W,
                }

    print(f"Successfully loaded {len(val_batches)} validation sequences.")

    # 4. Generate schedules
    eval_schedules = build_evaluation_schedules()
    pert_schedules = build_perturbation_schedules(K=12)
    print(f"Configured {len(eval_schedules)} evaluation schedules and {len(pert_schedules)} perturbation schedules.")

    # 5. Base model instantiation & parameter verification
    print("\nLoading pretrained DVLT model...")
    base_model = DVLT(img_size=504, depth_head_type="conv")
    base_model.load_pretrained("nvidia/dvlt", strict=True)
    base_model.setup_test(accelerator)

    # Extract bench_sample representations
    with torch.no_grad(), accelerator.autocast():
        inner = base_model.model
        images = bench_sample["images"]
        B, S, H, W = bench_sample["B"], bench_sample["S"], bench_sample["H"], bench_sample["W"]
        z_0 = inner._encode_images(images)
        reg_tok = inner.register_token.expand(B, S, -1, -1).reshape(B * S, inner.num_register_tokens, -1)
        cam_tok = _slice_expand_flatten(inner.camera_token, B, S)
        bench_sample["x_init"] = torch.cat([cam_tok, reg_tok, z_0], dim=1).detach().clone()
        bench_sample["rope_pos"] = inner._get_rope_positions(B * S, H, W, images.device).detach().clone()

    # Identify interval conditioning parameters
    interval_params = []
    total_model_params = 0
    for name, param in base_model.model.named_parameters():
        total_model_params += param.numel()
        if "depth_scale.proj" in name:
            interval_params.append((name, param.shape, param.numel()))

    trainable_count = sum(p[2] for p in interval_params)
    print(f"Total model parameters: {total_model_params:,}")
    print(f"Interval conditioning parameters (Stage A): {len(interval_params)} tensors, {trainable_count:,} params ({trainable_count/total_model_params*100:.4f}% of model)")
    for name, shape, num in interval_params:
        print(f"  - {name}: {shape} ({num:,} params)")

    assert trainable_count == 316032, f"Expected 316,032 parameters, got {trainable_count}"

    # Benchmark latencies once (hardware latencies depend on K, not parameter weights)
    print("\nBenchmarking real GPU recurrent latencies on RTX 5070 Laptop GPU...")
    all_schedules_combined = {**eval_schedules, **pert_schedules}
    latency_dict = benchmark_model_latencies(base_model.model, accelerator, bench_sample, all_schedules_combined)

    # 6. Model configurations to train and evaluate
    model_configs = [
        {"key": "control1_pretrained", "name": "Pretrained DVLT", "type": "pretrained", "mode": None, "seed": 42},
        {"key": "control2_uniform_ft", "name": "Control 2 (Uniform FT)", "type": "finetune", "mode": "uniform", "seed": 42},
        {"key": "treatment1_mild_ft",  "name": "Treatment 1 (Mild Rand FT)", "type": "finetune", "mode": "mild", "seed": 42},
        {"key": "treatment2_mod_ft",   "name": "Treatment 2 (Moderate Rand FT)", "type": "finetune", "mode": "moderate", "seed": 42},
    ]

    all_eval_dfs = []
    all_pert_dfs = []
    training_stats = {}

    # Run seed 42 screening for all 4 configurations
    for cfg in model_configs:
        key = cfg["key"]
        name = cfg["name"]
        print(f"\n{'='*40} Processing {name} (Seed {cfg['seed']}) {'='*40}")

        # Instantiate fresh copy of model
        cur_model = DVLT(img_size=504, depth_head_type="conv")
        cur_model.load_pretrained("nvidia/dvlt", strict=True)
        cur_model.setup_test(accelerator)

        if cfg["type"] == "finetune":
            print(f"Fine-tuning {name} (mode='{cfg['mode']}', seed={cfg['seed']}, 200 steps)...")
            cur_model.setup_train(accelerator, gradient_checkpointing=True)
            stats = train_stage_a_model(
                cur_model,
                accelerator,
                train_samples,
                mode=cfg["mode"],
                seed=cfg["seed"],
                total_steps=200,
            )
            training_stats[key] = stats
            print(f"Training completed in {stats['train_time_sec']:.2f}s | Peak VRAM: {stats['peak_vram_mb']:.1f} MB | Final Loss: {stats['final_loss']:.4f}")
        else:
            training_stats[key] = {
                "train_time_sec": 0.0,
                "peak_vram_mb": 0.0,
                "loss_history": [],
                "final_loss": 0.0,
                "num_trainable": 0,
            }

        # Evaluate on standard schedules
        print(f"Evaluating {name} on {len(eval_schedules)} standard schedules across 10 sequences...")
        df_eval = evaluate_model_on_schedules(cur_model, accelerator, val_batches, eval_schedules, model_label=key)
        all_eval_dfs.append(df_eval)

        # Evaluate on perturbation diagnostics
        print(f"Evaluating {name} on {len(pert_schedules)} perturbation schedules (Section 8 diagnostics)...")
        df_pert = evaluate_model_on_schedules(cur_model, accelerator, val_batches, pert_schedules, model_label=key)
        all_pert_dfs.append(df_pert)

    # Multi-seed validation (seeds 43, 44) for fine-tuned configurations to populate seed_results.csv
    print("\n" + "=" * 100)
    print("Multi-Seed Training & Validation (Seeds 42, 43, 44) for Rigorous Statistical Audit")
    print("=" * 100)
    seed_records = []

    # First add seed 42 results for all models
    for cfg in model_configs:
        key = cfg["key"]
        sub_df = [df for df in all_eval_dfs if df["model"].iloc[0] == key][0]
        sub_pert = [df for df in all_pert_dfs if df["model"].iloc[0] == key][0]

        # Extract metrics for K=12 Uniform, K=12 Non-uniform best, Uniform K=14, and K=12 CV=0.4
        u12_absrel = sub_df[sub_df["sched_key"] == "uniform_K12"]["abs_rel"].mean()
        u14_absrel = sub_df[sub_df["sched_key"] == "uniform_K14"]["abs_rel"].mean()
        pow075_absrel = sub_df[sub_df["sched_key"] == "power_K12_g0.75"]["abs_rel"].mean()
        cos_absrel = sub_df[sub_df["sched_key"] == "cosine_endpoints_K12"]["abs_rel"].mean()
        mod_absrel = sub_df[sub_df["sched_key"] == "sampled_moderate_K12"]["abs_rel"].mean()
        pert04_absrel = sub_pert[sub_pert["target_cv"] == 0.4]["abs_rel"].mean()

        seed_records.append({
            "model_key": key,
            "model_name": cfg["name"],
            "seed": 42,
            "training_mode": cfg["mode"] if cfg["mode"] else "none",
            "uniform_K12_absrel": u12_absrel,
            "uniform_K14_absrel": u14_absrel,
            "power_g075_K12_absrel": pow075_absrel,
            "cosine_end_K12_absrel": cos_absrel,
            "sampled_moderate_K12_absrel": mod_absrel,
            "perturbed_cv04_K12_absrel": pert04_absrel,
            "train_time_sec": training_stats[key]["train_time_sec"],
            "final_loss": training_stats[key]["final_loss"],
        })

    # Run seeds 43 and 44 for Control 2, Treatment 1, Treatment 2
    for seed in [43, 44]:
        for cfg in model_configs:
            if cfg["type"] != "finetune":
                continue
            key = cfg["key"]
            name = cfg["name"]
            mode = cfg["mode"]
            print(f"Training {name} with Seed {seed}...")

            cur_model = DVLT(img_size=504, depth_head_type="conv")
            cur_model.load_pretrained("nvidia/dvlt", strict=True)
            cur_model.setup_test(accelerator)
            cur_model.setup_train(accelerator, gradient_checkpointing=True)

            stats = train_stage_a_model(
                cur_model,
                accelerator,
                train_samples,
                mode=mode,
                seed=seed,
                total_steps=200,
            )

            # Targeted evaluation on key schedules: Uniform K12, Uniform K14, Power g0.75 K12, Cosine K12, Sampled Mod K12, CV0.4 K12
            target_eval_scheds = {
                "uniform_K12": eval_schedules["uniform_K12"],
                "uniform_K14": eval_schedules["uniform_K14"],
                "power_K12_g0.75": eval_schedules["power_K12_g0.75"],
                "cosine_endpoints_K12": eval_schedules["cosine_endpoints_K12"],
                "sampled_moderate_K12": eval_schedules["sampled_moderate_K12"],
                "pert_K12_cv40": pert_schedules["pert_K12_cv40"],
            }
            df_seed = evaluate_model_on_schedules(cur_model, accelerator, val_batches, target_eval_scheds, model_label=f"{key}_s{seed}")

            seed_records.append({
                "model_key": key,
                "model_name": name,
                "seed": seed,
                "training_mode": mode,
                "uniform_K12_absrel": df_seed[df_seed["sched_key"] == "uniform_K12"]["abs_rel"].mean(),
                "uniform_K14_absrel": df_seed[df_seed["sched_key"] == "uniform_K14"]["abs_rel"].mean(),
                "power_g075_K12_absrel": df_seed[df_seed["sched_key"] == "power_K12_g0.75"]["abs_rel"].mean(),
                "cosine_end_K12_absrel": df_seed[df_seed["sched_key"] == "cosine_endpoints_K12"]["abs_rel"].mean(),
                "sampled_moderate_K12_absrel": df_seed[df_seed["sched_key"] == "sampled_moderate_K12"]["abs_rel"].mean(),
                "perturbed_cv04_K12_absrel": df_seed[df_seed["sched_key"] == "pert_K12_cv40"]["abs_rel"].mean(),
                "train_time_sec": stats["train_time_sec"],
                "final_loss": stats["final_loss"],
            })

    # Save seed results CSV
    df_seeds = pd.DataFrame(seed_records)
    seed_csv_path = os.path.join(output_dir, "interval_robust_v0_seed_results.csv")
    df_seeds.to_csv(seed_csv_path, index=False)
    print(f"\nSaved multi-seed results: {seed_csv_path}")

    # Combine all sequence results for seed 42
    df_all_eval = pd.concat(all_eval_dfs, ignore_index=True)
    df_all_pert = pd.concat(all_pert_dfs, ignore_index=True)

    # 7. Generate Section 8 Perturbation Robustness Table & CSV
    robustness_list = []
    for model_key in ["control1_pretrained", "control2_uniform_ft", "treatment1_mild_ft", "treatment2_mod_ft"]:
        sub_p = df_all_pert[df_all_pert["model"] == model_key]
        for target_cv in [0.0, 0.1, 0.2, 0.4, 0.6]:
            sub_cv = sub_p[sub_p["target_cv"] == target_cv]
            m_absrel = sub_cv["abs_rel"].mean()
            m_rmse = sub_cv["rmse"].mean()
            m_rot = sub_cv["rot_err_deg"].mean()
            m_trans = sub_cv["trans_err_deg"].mean()
            actual_cv = sub_cv["actual_cv"].iloc[0]

            robustness_list.append({
                "model_key": model_key,
                "target_cv": target_cv,
                "actual_cv": actual_cv,
                "abs_rel": float(m_absrel),
                "rmse": float(m_rmse),
                "rot_err_deg": float(m_rot),
                "trans_err_deg": float(m_trans),
            })

    df_robustness = pd.DataFrame(robustness_list)
    robustness_csv = os.path.join(output_dir, "interval_robust_v0_robustness_curve.csv")
    df_robustness.to_csv(robustness_csv, index=False)
    print(f"Saved robustness curve data: {robustness_csv}")

    # 8. Generate Aggregated Summary Table (interval_robust_v0_summary.csv/.json)
    # The primary hurdle baseline is Pretrained DVLT Uniform K=14
    pretrained_u14_sub = df_all_eval[(df_all_eval["model"] == "control1_pretrained") & (df_all_eval["sched_key"] == "uniform_K14")]
    hurdle_absrel = float(pretrained_u14_sub["abs_rel"].mean())
    hurdle_lat = float(latency_dict["uniform_K14"]["recurrent_latency_ms"])
    print(f"\nPRIMARY HURDLE: Pretrained DVLT Uniform K=14 AbsRel = {hurdle_absrel:.6f}, Recurrent Latency = {hurdle_lat:.2f} ms")

    summary_list = []
    for model_key in ["control1_pretrained", "control2_uniform_ft", "treatment1_mild_ft", "treatment2_mod_ft"]:
        sub_m = df_all_eval[df_all_eval["model"] == model_key]
        for sched_key, sched in eval_schedules.items():
            sub_s = sub_m[sub_m["sched_key"] == sched_key]
            m_absrel = float(sub_s["abs_rel"].mean())
            m_rmse = float(sub_s["rmse"].mean())
            m_rot = float(sub_s["rot_err_deg"].mean())
            m_trans = float(sub_s["trans_err_deg"].mean())

            lat_info = latency_dict[sched_key]
            lat = lat_info["recurrent_latency_ms"]
            lat_std = lat_info["recurrent_latency_std_ms"]
            peak_vram = lat_info["peak_vram_mb"]

            # Comparison vs Pretrained Uniform K=14
            matches_or_beats_u14 = bool(m_absrel <= hurdle_absrel)
            absrel_delta_vs_u14_pct = float((m_absrel - hurdle_absrel) / hurdle_absrel * 100.0)
            lat_speedup_vs_u14_pct = float((hurdle_lat - lat) / hurdle_lat * 100.0)

            # Comparison vs same-model Uniform baseline of same K
            same_k_uniform_key = f"uniform_K{sched['K']}"
            same_k_absrel = float(sub_m[sub_m["sched_key"] == same_k_uniform_key]["abs_rel"].mean())
            absrel_gain_vs_same_k_pct = float((same_k_absrel - m_absrel) / same_k_absrel * 100.0)

            # Comparison vs matched Control 2 (Uniform FT)
            ctrl2_sub = df_all_eval[(df_all_eval["model"] == "control2_uniform_ft") & (df_all_eval["sched_key"] == sched_key)]
            ctrl2_absrel = float(ctrl2_sub["abs_rel"].mean())
            absrel_gain_vs_ctrl2_pct = float((ctrl2_absrel - m_absrel) / ctrl2_absrel * 100.0)

            summary_list.append({
                "model_key": model_key,
                "sched_key": sched_key,
                "sched_name": sched["name"],
                "family": sched["family"],
                "K": sched["K"],
                "abs_rel": m_absrel,
                "rmse": m_rmse,
                "rot_err_deg": m_rot,
                "trans_err_deg": m_trans,
                "recurrent_flops_gflops": sched["recurrent_flops"],
                "recurrent_latency_ms": lat,
                "recurrent_latency_std_ms": lat_std,
                "peak_vram_mb": peak_vram,
                "absrel_delta_vs_pretrained_u14_pct": absrel_delta_vs_u14_pct,
                "lat_speedup_vs_pretrained_u14_pct": lat_speedup_vs_u14_pct,
                "absrel_gain_vs_same_k_pct": absrel_gain_vs_same_k_pct,
                "absrel_gain_vs_ctrl2_pct": absrel_gain_vs_ctrl2_pct,
                "matches_or_beats_u14": matches_or_beats_u14,
                "trainable_params": training_stats[model_key]["num_trainable"],
                "train_time_sec": training_stats[model_key]["train_time_sec"],
                "peak_train_vram_mb": training_stats[model_key]["peak_vram_mb"],
            })

    summary_df = pd.DataFrame(summary_list)
    summary_csv = os.path.join(output_dir, "interval_robust_v0_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved aggregated summary CSV: {summary_csv}")

    summary_json = os.path.join(output_dir, "interval_robust_v0_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, indent=2)
    print(f"Saved aggregated summary JSON: {summary_json}")

    # Print summary highlights
    print("\n" + "=" * 135)
    print(f"{'Model':<25} | {'Schedule':<25} | {'K':<3} | {'AbsRel':<9} | {'RMSE':<8} | {'Lat (ms)':<9} | {'vs Pretr U14':<13} | {'vs Ctrl2':<10} | {'Beats U14?'}")
    print("-" * 135)
    for _, r in summary_df[summary_df["K"].isin([12, 14])].iterrows():
        b_str = "YES" if r["matches_or_beats_u14"] else "NO"
        print(f"{r['model_key']:<25} | {r['sched_name']:<25} | {r['K']:<3} | {r['abs_rel']:.6f} | {r['rmse']:.5f} | {r['recurrent_latency_ms']:6.2f} ms | {r['absrel_delta_vs_pretrained_u14_pct']:+6.2f}%      | {r['absrel_gain_vs_ctrl2_pct']:+6.2f}%    | {b_str}")
    print("=" * 135)

    print("\nExperiment execution completed successfully!")


if __name__ == "__main__":
    run_experiment()
