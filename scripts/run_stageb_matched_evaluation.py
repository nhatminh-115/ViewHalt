"""
INTERVAL-ROBUST STAGE B: RECURRENT-DYNAMICS ADAPTATION WITH SHARED LoRA
Evaluates matched pairs for Stage B1 (rank 4) and Stage B2 (rank 8):
- Arm A: Uniform FT Control
- Arm B: Mild Randomized FT (sigma=0.2)
Benchmarked against Pretrained DVLT baseline across:
- Uniform schedules: K in {10, 12, 14, 16}
- Non-uniform schedules: Power (0.75, 1.25), Cosine endpoints, Best Random (s=999), Sampled Mild, Sampled Moderate
- Robustness curve: CV in {0.0, 0.1, 0.2, 0.4, 0.6}
- Mechanistic diagnostics: Hidden-state drift at iterations {4, 8, 12, 14, 16} under K=16, gate drift, and per-module LoRA norms.

Outputs:
- outputs/stageb_matched_results.csv
- outputs/stageb_nonuniform_results.csv
- outputs/stageb_robustness_curve.csv
- outputs/stageb_hidden_drift.csv
"""

import copy
import gc
import json
import os
import sys
import time
import types
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Hardware cap: max 85% VRAM
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85, device=0)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

from dvlt.common.constants import DataField, PredictionField
from dvlt.data.collate import default_collate_fn
from dvlt.data.datasets.multi_source import MultiSourceDataset
from dvlt.data.datasets.parser.dataverse import DataverseEvalDataset, DataverseTrainDataset
from dvlt.metric.depth import apply_alignment
from dvlt.metric.pose import so3_relative_angle
from dvlt.model.dvlt.model import DVLT, _slice_expand_flatten

CANONICAL_INTERVAL_GRID = torch.tensor([
    [0.00, 0.10], [0.10, 0.20], [0.20, 0.30], [0.30, 0.40], [0.40, 0.50],
    [0.50, 0.60], [0.60, 0.70], [0.70, 0.80], [0.80, 0.90], [0.90, 1.00],
    [0.00, 0.05], [0.05, 0.20], [0.20, 0.50], [0.50, 0.70], [0.70, 1.00]
], dtype=torch.float32)


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 4, alpha: float = 4.0):
        super().__init__()
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scale = alpha / rank

        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(self.in_features, rank, dtype=original_linear.weight.dtype, device=original_linear.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features, dtype=original_linear.weight.dtype, device=original_linear.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = self.original_linear(x)
        lora = self.scale * (x @ self.lora_A.to(x.dtype)) @ self.lora_B.to(x.dtype)
        return orig + lora


def inject_lora(inner, rank=4, alpha=None):
    """Wraps attention projections with shared LoRA."""
    if alpha is None:
        alpha = float(rank)
    block = inner.recurrent_blocks[0]
    targets = [
        (block.frame_attn.attn, "qkv", "frame_qkv"),
        (block.frame_attn.attn, "proj", "frame_proj"),
        (block.global_attn.attn, "qkv", "global_qkv"),
        (block.global_attn.attn, "proj", "global_proj"),
    ]
    lora_dict = {}
    for parent, attr, key in targets:
        orig_layer = getattr(parent, attr)
        lora_layer = LoRALinear(orig_layer, rank=rank, alpha=alpha)
        setattr(parent, attr, lora_layer)
        lora_dict[key] = lora_layer
    return lora_dict


def sample_interval_schedule(K=12, mode="uniform", rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    if mode == "uniform":
        return torch.linspace(0.0, 1.0, K)
    M = K - 1
    if mode == "mild":
        w = rng.lognormal(mean=0.0, sigma=0.2, size=M)
        dt = w / np.sum(w)
    elif mode == "moderate":
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
    schedules = {}

    # 1. Uniform family
    for K in [10, 12, 14, 16]:
        schedules[f"uniform_K{K}"] = {
            "key": f"uniform_K{K}",
            "name": f"Uniform K={K}",
            "type": "uniform",
            "K": K,
            "ts": np.linspace(0.0, 1.0, K).tolist(),
        }

    # 2. Non-uniform family on K=12
    K = 12
    k_grid = np.arange(K) / (K - 1)

    ts_p075 = (k_grid ** 0.75).tolist()
    ts_p075[0], ts_p075[-1] = 0.0, 1.0
    schedules["power_g075"] = {
        "key": "power_g075", "name": "Power gamma=0.75", "type": "nonuniform", "K": K, "ts": ts_p075
    }

    ts_p125 = (k_grid ** 1.25).tolist()
    ts_p125[0], ts_p125[-1] = 0.0, 1.0
    schedules["power_g125"] = {
        "key": "power_g125", "name": "Power gamma=1.25", "type": "nonuniform", "K": K, "ts": ts_p125
    }

    ts_cos = (0.5 * (1.0 - np.cos(k_grid * np.pi))).tolist()
    ts_cos[0], ts_cos[-1] = 0.0, 1.0
    schedules["cosine_endpoints"] = {
        "key": "cosine_endpoints", "name": "Cosine endpoints-dense", "type": "nonuniform", "K": K, "ts": ts_cos
    }

    rng_best = np.random.default_rng(999)
    weights = rng_best.exponential(scale=1.0, size=K - 1)
    ts_best_rand = [0.0] + np.cumsum(weights / weights.sum()).tolist()
    ts_best_rand[-1] = 1.0
    schedules["best_random_v0"] = {
        "key": "best_random_v0", "name": "Best Random (V0 seed=999)", "type": "nonuniform", "K": K, "ts": ts_best_rand
    }

    ts_mild = sample_interval_schedule(K=12, mode="mild", rng=np.random.RandomState(42)).tolist()
    schedules["sampled_mild"] = {
        "key": "sampled_mild", "name": "Sampled Mild (sigma=0.2)", "type": "nonuniform", "K": K, "ts": ts_mild
    }

    ts_mod = sample_interval_schedule(K=12, mode="moderate", rng=np.random.RandomState(42)).tolist()
    schedules["sampled_moderate"] = {
        "key": "sampled_moderate", "name": "Sampled Moderate (sigma=0.5)", "type": "nonuniform", "K": K, "ts": ts_mod
    }

    # 3. Robustness curve CV in {0.0, 0.1, 0.2, 0.4, 0.6}
    for target_cv in [0.0, 0.1, 0.2, 0.4, 0.6]:
        ts_cv, actual_cv = generate_perturbed_schedule(K=K, target_cv=target_cv, seed=42)
        schedules[f"robust_cv_{target_cv:.1f}"] = {
            "key": f"robust_cv_{target_cv:.1f}",
            "name": f"Robust CV={actual_cv:.3f}",
            "type": "robustness",
            "K": K,
            "target_cv": target_cv,
            "actual_cv": actual_cv,
            "ts": ts_cv,
        }

    return schedules


def setup_custom_solve_train(inner, mode="uniform", rng=None):
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


def run_schedule_forward_with_intermediates(inner, x_init, rope_pos, B, S, ts, capture_iters=None):
    device = x_init.device
    block = inner.recurrent_blocks[0]
    P = x_init.shape[1]
    dim = x_init.shape[2]

    x_curr = x_init.clone()
    K = len(ts)
    intermediates = {}

    for i in range(K):
        t_now = ts[i]
        t_next = ts[i + 1] if i + 1 < K else 1.0
        t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)

        x_frame = block.frame_attn(x_curr, t_pair.expand(B * S, -1), pos=rope_pos.reshape(B * S, P, 2))
        x_global = block.global_attn(x_frame.reshape(B, S * P, dim), t_pair.expand(B, -1), pos=None).reshape(B * S, P, dim)
        x_curr = x_global

        iter_idx = i + 1
        if capture_iters is not None and iter_idx in capture_iters:
            intermediates[iter_idx] = x_curr.detach().cpu()

    return x_curr, intermediates


def compute_metrics_for_predictions(preds, batch, device):
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

    metrics_list = []
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

        metrics_list.append({
            "abs_rel": abs_rel, "rmse": rmse, "rot_err_deg": rot_err, "trans_err_deg": trans_err
        })
    return metrics_list


def evaluate_model_full(cur_model, accelerator, val_batches, schedules, capture_k16_hidden=False):
    inner = cur_model.model
    inner.eval()
    device = accelerator.device

    schedule_results = {sk: [] for sk in schedules}
    k16_intermediates_accum = {it: [] for it in [4, 8, 12, 14, 16]} if capture_k16_hidden else None

    u16_ts = schedules["uniform_K16"]["ts"]

    with torch.no_grad(), accelerator.autocast():
        for seq_id, batch in val_batches.items():
            images = batch[DataField.IMAGES]
            B, S, _, H, W = images.shape

            z_0 = inner._encode_images(images)
            reg_tok = inner.register_token.expand(B, S, -1, -1).reshape(B * S, inner.num_register_tokens, -1)
            cam_tok = _slice_expand_flatten(inner.camera_token, B, S)
            x_init = torch.cat([cam_tok, reg_tok, z_0], dim=1)
            rope_pos = inner._get_rope_positions(B * S, H, W, images.device)

            for sk, s_info in schedules.items():
                ts = s_info["ts"]
                x_final, _ = run_schedule_forward_with_intermediates(inner, x_init, rope_pos, B, S, ts)
                preds = cur_model._postprocess_predictions(batch, inner._decode(x_final, H, W, B, S, rope_pos))
                m_list = compute_metrics_for_predictions(preds, batch, device)
                schedule_results[sk].append({
                    "abs_rel": np.mean([v["abs_rel"] for v in m_list]),
                    "rmse": np.mean([v["rmse"] for v in m_list]),
                    "rot_err_deg": np.mean([v["rot_err_deg"] for v in m_list]),
                    "trans_err_deg": np.mean([v["trans_err_deg"] for v in m_list]),
                })

            if capture_k16_hidden:
                _, inter_dict = run_schedule_forward_with_intermediates(inner, x_init, rope_pos, B, S, u16_ts, capture_iters=[4, 8, 12, 14, 16])
                for it_k, tensor_val in inter_dict.items():
                    k16_intermediates_accum[it_k].append(tensor_val)

    summary = {}
    for sk, res in schedule_results.items():
        summary[sk] = {
            "abs_rel": float(np.mean([v["abs_rel"] for v in res])),
            "rmse": float(np.mean([v["rmse"] for v in res])),
            "rot_err_deg": float(np.mean([v["rot_err_deg"] for v in res])),
            "trans_err_deg": float(np.mean([v["trans_err_deg"] for v in res])),
        }

    return summary, k16_intermediates_accum


def train_lora_arm(accelerator, ms_train, train_video_indices, mode="uniform", rank=4, lora_lr=3e-6, gate_lr=1e-6, steps=200):
    device = accelerator.device
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.RandomState(42)

    cur_model = DVLT(img_size=504, depth_head_type="conv")
    cur_model.load_pretrained("nvidia/dvlt", strict=True)
    cur_model.setup_test(accelerator)

    inner = cur_model.model
    lora_dict = inject_lora(inner, rank=rank, alpha=float(rank))
    setup_custom_solve_train(inner, mode=mode, rng=rng)
    cur_model.setup_train(accelerator, gradient_checkpointing=True)

    gate_params = []
    lora_params = []
    for n, p in inner.named_parameters():
        if "depth_scale.proj" in n:
            p.requires_grad = True
            gate_params.append(p)
        elif "lora_" in n:
            p.requires_grad = True
            lora_params.append(p)
        else:
            p.requires_grad = False

    optimizer = torch.optim.AdamW([
        {"params": gate_params, "lr": gate_lr, "weight_decay": 1e-4},
        {"params": lora_params, "lr": lora_lr, "weight_decay": 1e-4},
    ])
    optimizer, inner = accelerator.prepare(optimizer, inner)
    cur_model.model = inner

    print(f"Training arm [mode={mode}, rank={rank}] for {steps} steps (gate_lr={gate_lr:.1e}, lora_lr={lora_lr:.1e})...")
    start_t = time.time()
    for step in range(steps):
        scan_vidx = train_video_indices[rng.randint(0, len(train_video_indices))]
        sample = ms_train[(scan_vidx, 6, 1.0, 42 + step)]
        batch = default_collate_fn([sample])
        batch_gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        optimizer.zero_grad()
        with accelerator.autocast():
            loss, _, _, _ = cur_model.train_step(batch_gpu, step, accelerator)
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(gate_params + lora_params, max_norm=1.0)
        optimizer.step()

        if (step + 1) % 50 == 0 or step == 0:
            print(f"  Step {step+1:3d}/{steps} | Loss: {loss.item():.4f}")

    train_time_s = time.time() - start_t
    inner_eval = accelerator.unwrap_model(inner)
    inner_eval.eval()
    cur_model.model = inner_eval
    return cur_model, lora_dict, train_time_s


def compute_gate_drift(inner, test_grid_device, baseline_gate_out):
    block = inner.recurrent_blocks[0]
    with torch.no_grad():
        s_frame = block.frame_attn.depth_scale(test_grid_device).flatten()
        s_global = block.global_attn.depth_scale(test_grid_device).flatten()
    gate_out = torch.cat([s_frame, s_global])
    return float(torch.norm(gate_out - baseline_gate_out).item())


def compute_module_lora_norms(lora_dict):
    norms = {}
    for key, l_module in lora_dict.items():
        W_delta = l_module.scale * (l_module.lora_A @ l_module.lora_B)
        norms[key] = float(torch.norm(W_delta).item())
    return norms


def run_stageb_full_protocol(data_root="datasets/test/dtu", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print("=" * 100)
    print("INTERVAL-ROBUST STAGE B: RECURRENT-DYNAMICS ADAPTATION WITH SHARED LoRA")
    print("=" * 100)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    print(f"Device: {device} | Precision: bf16")

    train_scan_names = ["scan10", "scan11", "scan12", "scan15", "scan23", "scan29", "scan33", "scan48", "scan110"]
    val_scan_names = ["scan1", "scan4", "scan9", "scan24", "scan62"]
    subsets_def = [
        {"name": "subset_middle", "ranking": "middle_first", "sampling": "first"},
        {"name": "subset_uniform", "ranking": "index", "sampling": "uniform"},
    ]

    print("\nPreparing validation batches (10 sequences)...")
    cfg = OmegaConf.create({"target": "dtu.DTU", "params": {"root_path": data_root}})
    test_config = {
        "normalize_scene": False,
        "load_data_fields": ["images", "extrinsics_c2w", "intrinsics", "depths", "world_points", "point_masks"],
    }
    val_batches = {}
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
    print(f"Loaded {len(val_batches)} validation sequences successfully.")

    print("\nPreparing dynamic training dataset with normalize_scene=True...")
    ds_train = DataverseTrainDataset(dataverse_cfg=cfg)
    ds_train.set_image_params(504, 14)

    train_video_indices = []
    for idx in range(ds_train.ds.num_videos()):
        sname = ds_train.ds._get_scan_name(idx)
        if sname in train_scan_names:
            train_video_indices.append(idx)
    assert len(train_video_indices) == len(train_scan_names)

    MultiSourceDataset.MIN_LEN = 14
    train_config = {
        "normalize_scene": True,
        "load_data_fields": ["images", "extrinsics_c2w", "intrinsics", "depths", "world_points", "point_masks"],
    }
    ms_train = MultiSourceDataset({"dtu": ds_train}, training=True, **train_config)

    schedules = build_evaluation_schedules()
    print(f"Built {len(schedules)} evaluation schedules across Uniform, Non-uniform, and Robustness sets.")

    # 1. Evaluate Pretrained Baseline
    print("\n--- Evaluating Pretrained Baseline ---")
    model_pre = DVLT(img_size=504, depth_head_type="conv")
    model_pre.load_pretrained("nvidia/dvlt", strict=True)
    model_pre.setup_test(accelerator)

    test_grid_device = CANONICAL_INTERVAL_GRID.to(device)
    block_pre = model_pre.model.recurrent_blocks[0]
    with torch.no_grad():
        sf = block_pre.frame_attn.depth_scale(test_grid_device).flatten()
        sg = block_pre.global_attn.depth_scale(test_grid_device).flatten()
        baseline_gate_out = torch.cat([sf, sg])

    res_pre, hidden_pre = evaluate_model_full(model_pre, accelerator, val_batches, schedules, capture_k16_hidden=True)
    del model_pre
    torch.cuda.empty_cache()
    gc.collect()

    # LoRA LR selected from calibration
    lora_lr_selected = 3e-6
    gate_lr_selected = 1e-6
    steps_budget = 200

    arms_def = [
        {"stage": "B1", "rank": 4, "mode": "uniform", "name": "B1_Uniform_FT", "desc": "B1 rank=4 Uniform FT Control"},
        {"stage": "B1", "rank": 4, "mode": "mild", "name": "B1_Mild_Rand_FT", "desc": "B1 rank=4 Mild Rand FT (sigma=0.2)"},
        {"stage": "B2", "rank": 8, "mode": "uniform", "name": "B2_Uniform_FT", "desc": "B2 rank=8 Uniform FT Control"},
        {"stage": "B2", "rank": 8, "mode": "mild", "name": "B2_Mild_Rand_FT", "desc": "B2 rank=8 Mild Rand FT (sigma=0.2)"},
    ]

    all_models_eval = {"Pretrained": res_pre}
    hidden_states_all = {"Pretrained": hidden_pre}
    diagnostics_records = []

    for arm in arms_def:
        arm_name = arm["name"]
        print(f"\n--- Running Arm: {arm['desc']} ---")
        cur_model, lora_dict, train_t = train_lora_arm(
            accelerator, ms_train, train_video_indices,
            mode=arm["mode"], rank=arm["rank"],
            lora_lr=lora_lr_selected, gate_lr=gate_lr_selected, steps=steps_budget
        )

        # Diagnostics
        gate_drift = compute_gate_drift(cur_model.model, test_grid_device, baseline_gate_out)
        lora_norms = compute_module_lora_norms(lora_dict)

        diagnostics_records.append({
            "arm": arm_name,
            "stage": arm["stage"],
            "rank": arm["rank"],
            "mode": arm["mode"],
            "train_time_s": train_t,
            "gate_drift": gate_drift,
            "frame_qkv_norm": lora_norms["frame_qkv"],
            "frame_proj_norm": lora_norms["frame_proj"],
            "global_qkv_norm": lora_norms["global_qkv"],
            "global_proj_norm": lora_norms["global_proj"],
            "total_lora_norm": float(np.sqrt(sum(v**2 for v in lora_norms.values()))),
        })

        eval_res, hidden_arm = evaluate_model_full(cur_model, accelerator, val_batches, schedules, capture_k16_hidden=True)
        all_models_eval[arm_name] = eval_res
        hidden_states_all[arm_name] = hidden_arm

        del cur_model, lora_dict
        torch.cuda.empty_cache()
        gc.collect()

    # Output 1: Uniform Matched Results (K in 10, 12, 14, 16)
    uniform_keys = ["uniform_K10", "uniform_K12", "uniform_K14", "uniform_K16"]
    matched_records = []
    for arm_name in ["Pretrained"] + [a["name"] for a in arms_def]:
        for uk in uniform_keys:
            m = all_models_eval[arm_name][uk]
            matched_records.append({
                "model": arm_name,
                "schedule_key": uk,
                "K": schedules[uk]["K"],
                "abs_rel": m["abs_rel"],
                "rmse": m["rmse"],
                "rot_err_deg": m["rot_err_deg"],
                "trans_err_deg": m["trans_err_deg"],
                "delta_vs_pre_pct": ((m["abs_rel"] - all_models_eval["Pretrained"][uk]["abs_rel"]) / all_models_eval["Pretrained"][uk]["abs_rel"]) * 100.0,
                "beats_pre_k14": bool(m["abs_rel"] <= all_models_eval["Pretrained"]["uniform_K14"]["abs_rel"]),
            })
    df_matched = pd.DataFrame(matched_records)
    out_matched_csv = os.path.join(output_dir, "stageb_matched_results.csv")
    df_matched.to_csv(out_matched_csv, index=False)
    print(f"\nSaved matched uniform results: {out_matched_csv}")

    # Output 2: Non-uniform results (K=12 schedules)
    nonuniform_keys = ["uniform_K12", "power_g075", "power_g125", "cosine_endpoints", "best_random_v0", "sampled_mild", "sampled_moderate"]
    nonuniform_records = []
    for arm_name in ["Pretrained"] + [a["name"] for a in arms_def]:
        for nk in nonuniform_keys:
            m = all_models_eval[arm_name][nk]
            nonuniform_records.append({
                "model": arm_name,
                "schedule_key": nk,
                "schedule_name": schedules[nk]["name"],
                "abs_rel": m["abs_rel"],
                "rmse": m["rmse"],
                "rot_err_deg": m["rot_err_deg"],
                "trans_err_deg": m["trans_err_deg"],
                "delta_vs_pre_pct": ((m["abs_rel"] - all_models_eval["Pretrained"][nk]["abs_rel"]) / all_models_eval["Pretrained"][nk]["abs_rel"]) * 100.0,
            })
    df_nonunif = pd.DataFrame(nonuniform_records)
    out_nonunif_csv = os.path.join(output_dir, "stageb_nonuniform_results.csv")
    df_nonunif.to_csv(out_nonunif_csv, index=False)
    print(f"Saved non-uniform results: {out_nonunif_csv}")

    # Output 3: Robustness Curve
    robust_keys = [f"robust_cv_{cv:.1f}" for cv in [0.0, 0.1, 0.2, 0.4, 0.6]]
    robust_records = []
    for arm_name in ["Pretrained"] + [a["name"] for a in arms_def]:
        for rk in robust_keys:
            m = all_models_eval[arm_name][rk]
            target_cv = schedules[rk]["target_cv"]
            actual_cv = schedules[rk]["actual_cv"]
            robust_records.append({
                "model": arm_name,
                "target_cv": target_cv,
                "actual_cv": actual_cv,
                "abs_rel": m["abs_rel"],
                "rmse": m["rmse"],
                "rot_err_deg": m["rot_err_deg"],
                "trans_err_deg": m["trans_err_deg"],
                "delta_vs_pre_pct": ((m["abs_rel"] - all_models_eval["Pretrained"][rk]["abs_rel"]) / all_models_eval["Pretrained"][rk]["abs_rel"]) * 100.0,
            })
    df_robust = pd.DataFrame(robust_records)
    out_robust_csv = os.path.join(output_dir, "stageb_robustness_curve.csv")
    df_robust.to_csv(out_robust_csv, index=False)
    print(f"Saved robustness curve results: {out_robust_csv}")

    # Output 4: Hidden State Drift at iterations 4, 8, 12, 14, 16
    hidden_records = []
    iters_track = [4, 8, 12, 14, 16]
    for arm_name in [a["name"] for a in arms_def]:
        for it_k in iters_track:
            # compute mean Frobenius distance across all 10 sequences
            diffs = []
            for s_idx in range(len(val_batches)):
                h_pre = hidden_states_all["Pretrained"][it_k][s_idx]
                h_arm = hidden_states_all[arm_name][it_k][s_idx]
                frob_diff = torch.norm(h_arm - h_pre).item() / torch.norm(h_pre).item()
                diffs.append(frob_diff)
            hidden_records.append({
                "model": arm_name,
                "iteration": it_k,
                "relative_hidden_drift": float(np.mean(diffs)),
            })
    df_hidden = pd.DataFrame(hidden_records)
    out_hidden_csv = os.path.join(output_dir, "stageb_hidden_drift.csv")
    df_hidden.to_csv(out_hidden_csv, index=False)
    print(f"Saved hidden drift results: {out_hidden_csv}")

    # Diagnostics summary
    df_diag = pd.DataFrame(diagnostics_records)
    out_diag_csv = os.path.join(output_dir, "stageb_diagnostics.csv")
    df_diag.to_csv(out_diag_csv, index=False)
    print(f"Saved diagnostics results: {out_diag_csv}")

    print("\nStage B full matched protocol completed successfully!")


if __name__ == "__main__":
    run_stageb_full_protocol()
