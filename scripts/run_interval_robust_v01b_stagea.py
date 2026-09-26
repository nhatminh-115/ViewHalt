"""
INTERVAL-ROBUST V0.1B — COMPLETE THE STAGE-A DECISION
Evaluates:
1. Non-Uniform Schedule Evaluation across 7 candidate schedules on K=12.
2. Robustness Curve across CV(Delta_t) in {0.0, 0.1, 0.2, 0.4, 0.6}.
Models:
- Pretrained DVLT (Untouched)
- Control 2: Uniform FT (LR=1e-6, 200 steps, repaired protocol)
- Treatment 1: Mild Rand FT (sigma=0.2, LR=1e-6, 200 steps)
- Treatment 2: Moderate Rand FT (sigma=0.5, LR=1e-6, 200 steps)

Outputs:
- outputs/interval_robust_v01b_nonuniform.csv
- outputs/interval_robust_v01b_robustness.csv
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
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf

# Force line-buffering on stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Hard memory fraction cap requested by user: max 0.85
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


def sample_interval_schedule(K=12, mode="uniform", rng=None):
    """Draws continuous-time schedule ts in [0, 1] for K steps."""
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
    """Builds the required non-uniform and perturbation schedules on K=12."""
    K = 12
    k_grid = np.arange(K) / (K - 1)

    schedules = {}

    # 1. Non-Uniform Set
    # - Uniform K=12
    schedules["uniform_K12"] = {
        "key": "uniform_K12",
        "name": "Uniform K=12",
        "category": "nonuniform",
        "ts": np.linspace(0.0, 1.0, K).tolist(),
    }
    # - Power gamma=0.75
    ts_p075 = (k_grid ** 0.75).tolist()
    ts_p075[0], ts_p075[-1] = 0.0, 1.0
    schedules["power_g075"] = {
        "key": "power_g075",
        "name": "Power gamma=0.75",
        "category": "nonuniform",
        "ts": ts_p075,
    }
    # - Power gamma=1.25
    ts_p125 = (k_grid ** 1.25).tolist()
    ts_p125[0], ts_p125[-1] = 0.0, 1.0
    schedules["power_g125"] = {
        "key": "power_g125",
        "name": "Power gamma=1.25",
        "category": "nonuniform",
        "ts": ts_p125,
    }
    # - Cosine endpoints-dense
    ts_cos = (0.5 * (1.0 - np.cos(k_grid * np.pi))).tolist()
    ts_cos[0], ts_cos[-1] = 0.0, 1.0
    schedules["cosine_endpoints"] = {
        "key": "cosine_endpoints",
        "name": "Cosine endpoints-dense",
        "category": "nonuniform",
        "ts": ts_cos,
    }
    # - Best random schedule from NONUNIFORM_V0 (seed=999)
    rng_best = np.random.default_rng(999)
    weights = rng_best.exponential(scale=1.0, size=K - 1)
    ts_best_rand = [0.0] + np.cumsum(weights / weights.sum()).tolist()
    ts_best_rand[-1] = 1.0
    schedules["best_random_v0"] = {
        "key": "best_random_v0",
        "name": "Best Random (V0 seed=999)",
        "category": "nonuniform",
        "ts": ts_best_rand,
    }
    # - Sampled Mild schedule
    ts_mild = sample_interval_schedule(K=12, mode="mild", rng=np.random.RandomState(42)).tolist()
    schedules["sampled_mild"] = {
        "key": "sampled_mild",
        "name": "Sampled Mild (sigma=0.2)",
        "category": "nonuniform",
        "ts": ts_mild,
    }
    # - Sampled Moderate schedule
    ts_mod = sample_interval_schedule(K=12, mode="moderate", rng=np.random.RandomState(42)).tolist()
    schedules["sampled_moderate"] = {
        "key": "sampled_moderate",
        "name": "Sampled Moderate (sigma=0.5)",
        "category": "nonuniform",
        "ts": ts_mod,
    }

    # 2. Robustness Curve Set: CV(Delta_t) in {0.0, 0.1, 0.2, 0.4, 0.6}
    for target_cv in [0.0, 0.1, 0.2, 0.4, 0.6]:
        ts_cv, actual_cv = generate_perturbed_schedule(K=K, target_cv=target_cv, seed=42)
        schedules[f"robust_cv_{target_cv:.1f}"] = {
            "key": f"robust_cv_{target_cv:.1f}",
            "name": f"Robust CV={actual_cv:.3f}",
            "category": "robustness",
            "target_cv": target_cv,
            "actual_cv": actual_cv,
            "ts": ts_cv,
        }

    return schedules


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

        x_frame = block.frame_attn(x_curr, t_pair.expand(B * S, -1), pos=rope_pos.reshape(B * S, P, 2))
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
            "abs_rel": abs_rel,
            "rmse": rmse,
            "rot_err_deg": rot_err,
            "trans_err_deg": trans_err,
        })
    return metrics_list


def evaluate_model_on_schedules(cur_model, accelerator, val_batches, schedules):
    """Evaluates a model across all given schedules over the 10 validation sequences."""
    inner = cur_model.model
    inner.eval()
    device = accelerator.device

    schedule_results = {s_key: [] for s_key in schedules}

    with torch.no_grad(), accelerator.autocast():
        for seq_id, batch in val_batches.items():
            images = batch[DataField.IMAGES]
            B, S, _, H, W = images.shape

            z_0 = inner._encode_images(images)
            reg_tok = inner.register_token.expand(B, S, -1, -1).reshape(B * S, inner.num_register_tokens, -1)
            cam_tok = _slice_expand_flatten(inner.camera_token, B, S)
            x_init = torch.cat([cam_tok, reg_tok, z_0], dim=1)
            rope_pos = inner._get_rope_positions(B * S, H, W, images.device)

            for s_key, s_info in schedules.items():
                ts = s_info["ts"]
                x_final = run_schedule_forward(inner, x_init, rope_pos, B, S, ts)
                preds = cur_model._postprocess_predictions(batch, inner._decode(x_final, H, W, B, S, rope_pos))
                m_list = compute_metrics_for_predictions(preds, batch, device)
                schedule_results[s_key].append({
                    "abs_rel": np.mean([v["abs_rel"] for v in m_list]),
                    "rmse": np.mean([v["rmse"] for v in m_list]),
                    "rot_err_deg": np.mean([v["rot_err_deg"] for v in m_list]),
                    "trans_err_deg": np.mean([v["trans_err_deg"] for v in m_list]),
                })

    summary = {}
    for s_key, res in schedule_results.items():
        summary[s_key] = {
            "abs_rel": float(np.mean([v["abs_rel"] for v in res])),
            "rmse": float(np.mean([v["rmse"] for v in res])),
            "rot_err_deg": float(np.mean([v["rot_err_deg"] for v in res])),
            "trans_err_deg": float(np.mean([v["trans_err_deg"] for v in res])),
        }
    return summary


def train_arm(accelerator, ms_train, train_video_indices, mode="uniform", steps=200, lr=1e-6):
    """Trains a single arm under the repaired stable protocol."""
    device = accelerator.device
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.RandomState(42)

    cur_model = DVLT(img_size=504, depth_head_type="conv")
    cur_model.load_pretrained("nvidia/dvlt", strict=True)
    cur_model.setup_test(accelerator)

    inner = cur_model.model
    trainable_params = []
    for n, p in inner.named_parameters():
        if "depth_scale.proj" in n:
            p.requires_grad = True
            trainable_params.append(p)
        else:
            p.requires_grad = False

    setup_custom_solve_train(inner, mode=mode, rng=rng)
    cur_model.setup_train(accelerator, gradient_checkpointing=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    optimizer, inner = accelerator.prepare(optimizer, inner)
    cur_model.model = inner

    print(f"Training arm [{mode}] for {steps} steps with LR={lr:.1e}...")
    for step in range(steps):
        scan_vidx = train_video_indices[rng.randint(0, len(train_video_indices))]
        sample = ms_train[(scan_vidx, 6, 1.0, 42 + step)]
        batch = default_collate_fn([sample])
        batch_gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        optimizer.zero_grad()
        with accelerator.autocast():
            loss, _, _, _ = cur_model.train_step(batch_gpu, step, accelerator)
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        if (step + 1) % 50 == 0 or step == 0:
            print(f"  Step {step+1:3d}/{steps} | Loss: {loss.item():.4f}")

    inner_eval = accelerator.unwrap_model(inner)
    inner_eval.eval()
    cur_model.model = inner_eval
    return cur_model


def main(data_root="datasets/test/dtu", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print("=" * 100)
    print("INTERVAL-ROBUST V0.1B: COMPLETE THE STAGE-A DECISION")
    print("=" * 100)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    print(f"Device: {device} | Precision: bf16")

    # Dataset splits
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

    # Dynamic Training Dataset Setup (WITH normalize_scene=True)
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
        "normalize_scene": True,  # CRITICAL: normalized scene for official DVLT loss scaling
        "load_data_fields": ["images", "extrinsics_c2w", "intrinsics", "depths", "world_points", "point_masks"],
    }
    ms_train = MultiSourceDataset({"dtu": ds_train}, training=True, **train_config)

    all_schedules = build_evaluation_schedules()
    print(f"\nEvaluated Schedules: {len(all_schedules)} schedules defined.")

    # 1. Evaluate Pretrained Baseline
    print("\n--- Evaluating Pretrained Baseline ---")
    model_pre = DVLT(img_size=504, depth_head_type="conv")
    model_pre.load_pretrained("nvidia/dvlt", strict=True)
    model_pre.setup_test(accelerator)
    res_pre = evaluate_model_on_schedules(model_pre, accelerator, val_batches, all_schedules)
    del model_pre
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Train and Evaluate Control 2 (Uniform FT)
    print("\n--- Training & Evaluating Control 2 (Uniform FT) ---")
    model_ctrl = train_arm(accelerator, ms_train, train_video_indices, mode="uniform", steps=200, lr=1e-6)
    res_ctrl = evaluate_model_on_schedules(model_ctrl, accelerator, val_batches, all_schedules)
    del model_ctrl
    torch.cuda.empty_cache()
    gc.collect()

    # 3. Train and Evaluate Treatment 1 (Mild Rand FT)
    print("\n--- Training & Evaluating Treatment 1 (Mild Rand FT, sigma=0.2) ---")
    model_mild = train_arm(accelerator, ms_train, train_video_indices, mode="mild", steps=200, lr=1e-6)
    res_mild = evaluate_model_on_schedules(model_mild, accelerator, val_batches, all_schedules)
    del model_mild
    torch.cuda.empty_cache()
    gc.collect()

    # 4. Train and Evaluate Treatment 2 (Moderate Rand FT)
    print("\n--- Training & Evaluating Treatment 2 (Moderate Rand FT, sigma=0.5) ---")
    model_mod = train_arm(accelerator, ms_train, train_video_indices, mode="moderate", steps=200, lr=1e-6)
    res_mod = evaluate_model_on_schedules(model_mod, accelerator, val_batches, all_schedules)
    del model_mod
    torch.cuda.empty_cache()
    gc.collect()

    # Consolidate results
    models_dict = {
        "Pretrained": res_pre,
        "Uniform_FT": res_ctrl,
        "Mild_Rand_FT": res_mild,
        "Mod_Rand_FT": res_mod,
    }

    # Output 1: Non-uniform schedules CSV
    nonuniform_keys = [
        "uniform_K12", "power_g075", "power_g125", "cosine_endpoints",
        "best_random_v0", "sampled_mild", "sampled_moderate"
    ]
    records_nonuniform = []
    for sk in nonuniform_keys:
        s_name = all_schedules[sk]["name"]
        for m_name, res in models_dict.items():
            metrics = res[sk]
            records_nonuniform.append({
                "model": m_name,
                "schedule_key": sk,
                "schedule_name": s_name,
                "abs_rel": metrics["abs_rel"],
                "rmse": metrics["rmse"],
                "rot_err_deg": metrics["rot_err_deg"],
                "trans_err_deg": metrics["trans_err_deg"],
                "delta_vs_ctrl_absrel_pct": ((metrics["abs_rel"] - models_dict["Uniform_FT"][sk]["abs_rel"]) / models_dict["Uniform_FT"][sk]["abs_rel"]) * 100.0,
                "delta_vs_pre_absrel_pct": ((metrics["abs_rel"] - models_dict["Pretrained"][sk]["abs_rel"]) / models_dict["Pretrained"][sk]["abs_rel"]) * 100.0,
            })
    df_nonuniform = pd.DataFrame(records_nonuniform)
    out_nonuniform_csv = os.path.join(output_dir, "interval_robust_v01b_nonuniform.csv")
    df_nonuniform.to_csv(out_nonuniform_csv, index=False)
    print(f"\nSaved non-uniform schedule results: {out_nonuniform_csv}")

    # Output 2: Robustness curve CSV
    cv_list = [0.0, 0.1, 0.2, 0.4, 0.6]
    records_robust = []
    for cv in cv_list:
        sk = f"robust_cv_{cv:.1f}"
        actual_cv = all_schedules[sk]["actual_cv"]
        for m_name, res in models_dict.items():
            metrics = res[sk]
            records_robust.append({
                "model": m_name,
                "target_cv": cv,
                "actual_cv": actual_cv,
                "abs_rel": metrics["abs_rel"],
                "rmse": metrics["rmse"],
                "rot_err_deg": metrics["rot_err_deg"],
                "trans_err_deg": metrics["trans_err_deg"],
                "delta_vs_ctrl_absrel_pct": ((metrics["abs_rel"] - models_dict["Uniform_FT"][sk]["abs_rel"]) / models_dict["Uniform_FT"][sk]["abs_rel"]) * 100.0,
                "delta_vs_pre_absrel_pct": ((metrics["abs_rel"] - models_dict["Pretrained"][sk]["abs_rel"]) / models_dict["Pretrained"][sk]["abs_rel"]) * 100.0,
            })
    df_robust = pd.DataFrame(records_robust)
    out_robust_csv = os.path.join(output_dir, "interval_robust_v01b_robustness.csv")
    df_robust.to_csv(out_robust_csv, index=False)
    print(f"Saved robustness curve results: {out_robust_csv}")

    print("\nV0.1B evaluation completed successfully!")


if __name__ == "__main__":
    main()
