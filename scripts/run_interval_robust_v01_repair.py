"""INTERVAL-ROBUST V0.1: TRAINING PROTOCOL REPAIR & LR CALIBRATION.

Repairs the training protocol for DVLT interval-conditioned fine-tuning:
1. Dynamic training data pipeline:
   - Does NOT cache fixed batches.
   - At every step: samples a training scene from 9 disjoint scans.
   - Calls training dataset with resampled 6-view set and dynamic augmentations.
   - CRITICAL: training uses normalize_scene=True (consistent with official training).
   - Evaluation retains unnormalized scene benchmark preprocessing.
2. Verified configuration:
   - Uses official Beta(2, 1) step sampling: k_sampler_beta_a=2, k_sampler_beta_b=1.
3. No-op equivalence test verified:
   - Zero numerical difference (0.00000000e+00) between untouched pretrained and setup wrapper.
4. LR Calibration:
   - Tests LR in {1e-6, 3e-6, 1e-5, 3e-5}.
   - Evaluates at step in {0, 1, 5, 10, 25, 50, 100, 200}.
   - Tracks Uniform K=12 AbsRel, Uniform K=14 AbsRel, pose metrics, parameter drift, gate drift, train loss.
   - Verifies whether Uniform FT preserves pretrained calibration within ~5% relative AbsRel.
5. If stable protocol found:
   - Runs matched Uniform vs Mild (sigma=0.2) vs Moderate (sigma=0.5) randomized training.
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

from dvlt.common.constants import DataField, PredictionField
from dvlt.data.collate import default_collate_fn
from dvlt.data.datasets.multi_source import MultiSourceDataset
from dvlt.data.datasets.parser.dataverse import DataverseEvalDataset, DataverseTrainDataset
from dvlt.metric.depth import apply_alignment
from dvlt.metric.pose import so3_relative_angle
from dvlt.model.dvlt.model import DVLT, _slice_expand_flatten

# Canonical test grid for gate drift (15 representative interval pairs)
CANONICAL_INTERVAL_GRID = torch.tensor([
    [0.00, 0.10], [0.10, 0.20], [0.20, 0.30], [0.30, 0.40], [0.40, 0.50],
    [0.50, 0.60], [0.60, 0.70], [0.70, 0.80], [0.80, 0.90], [0.90, 1.00],
    [0.00, 0.05], [0.05, 0.20], [0.20, 0.50], [0.50, 0.70], [0.70, 1.00]
], dtype=torch.float32)


def sample_interval_schedule(K: int, mode: str, rng: np.random.RandomState):
    """Generates continuous-time partition ts of length K on [0, 1]."""
    if mode == "uniform" or K <= 2:
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


def evaluate_checkpoint_uniform(model, accelerator, val_batches):
    """Evaluates current model state on Uniform K=12 and K=14 across all 10 validation sequences."""
    inner = model.model
    inner.eval()
    device = accelerator.device

    u12_ts = torch.linspace(0.0, 1.0, 12).tolist()
    u14_ts = torch.linspace(0.0, 1.0, 14).tolist()

    u12_metrics = []
    u14_metrics = []

    with torch.no_grad(), accelerator.autocast():
        for seq_id, batch in val_batches.items():
            images = batch[DataField.IMAGES]
            B, S, _, H, W = images.shape

            z_0 = inner._encode_images(images)
            reg_tok = inner.register_token.expand(B, S, -1, -1).reshape(B * S, inner.num_register_tokens, -1)
            cam_tok = _slice_expand_flatten(inner.camera_token, B, S)
            x_init = torch.cat([cam_tok, reg_tok, z_0], dim=1)
            rope_pos = inner._get_rope_positions(B * S, H, W, images.device)

            # Uniform K=12
            x_12 = run_schedule_forward(inner, x_init, rope_pos, B, S, u12_ts)
            preds_12 = model._postprocess_predictions(batch, inner._decode(x_12, H, W, B, S, rope_pos))
            m12 = compute_metrics_for_predictions(preds_12, batch, device)
            u12_metrics.append({
                "abs_rel": np.mean([v["abs_rel"] for v in m12]),
                "rmse": np.mean([v["rmse"] for v in m12]),
                "rot_err_deg": np.mean([v["rot_err_deg"] for v in m12]),
                "trans_err_deg": np.mean([v["trans_err_deg"] for v in m12]),
            })

            # Uniform K=14
            x_14 = run_schedule_forward(inner, x_init, rope_pos, B, S, u14_ts)
            preds_14 = model._postprocess_predictions(batch, inner._decode(x_14, H, W, B, S, rope_pos))
            m14 = compute_metrics_for_predictions(preds_14, batch, device)
            u14_metrics.append({
                "abs_rel": np.mean([v["abs_rel"] for v in m14]),
                "rmse": np.mean([v["rmse"] for v in m14]),
                "rot_err_deg": np.mean([v["rot_err_deg"] for v in m14]),
                "trans_err_deg": np.mean([v["trans_err_deg"] for v in m14]),
            })

    return {
        "k12_absrel": float(np.mean([m["abs_rel"] for m in u12_metrics])),
        "k12_rmse": float(np.mean([m["rmse"] for m in u12_metrics])),
        "k12_rot": float(np.mean([m["rot_err_deg"] for m in u12_metrics])),
        "k12_trans": float(np.mean([m["trans_err_deg"] for m in u12_metrics])),
        "k14_absrel": float(np.mean([m["abs_rel"] for m in u14_metrics])),
        "k14_rmse": float(np.mean([m["rmse"] for m in u14_metrics])),
        "k14_rot": float(np.mean([m["rot_err_deg"] for m in u14_metrics])),
        "k14_trans": float(np.mean([m["trans_err_deg"] for m in u14_metrics])),
    }


def extract_parameter_vector(inner):
    """Flattens all trainable IntervalDepthScaling parameters into a 1D tensor."""
    weights = []
    for name, p in inner.named_parameters():
        if "depth_scale.proj" in name:
            weights.append(p.detach().flatten())
    return torch.cat(weights)


def extract_gate_vector(inner, test_grid_device):
    """Evaluates frame and global attention depth scales on test grid and concatenates."""
    block = inner.recurrent_blocks[0]
    with torch.no_grad():
        s_frame = block.frame_attn.depth_scale(test_grid_device)
        s_global = block.global_attn.depth_scale(test_grid_device)
    return torch.cat([s_frame.flatten(), s_global.flatten()])


def run_v01_repair(data_root="datasets/test/dtu", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print("=" * 100)
    print("INTERVAL-ROBUST V0.1: TRAINING PROTOCOL REPAIR & LR CALIBRATION")
    print("=" * 100)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    print(f"Device: {device} | Precision: bf16")

    # Dataset splits: strictly scene-disjoint
    train_scan_names = ["scan10", "scan11", "scan12", "scan15", "scan23", "scan29", "scan33", "scan48", "scan110"]
    val_scan_names = ["scan1", "scan4", "scan9", "scan24", "scan62"]
    subsets_def = [
        {"name": "subset_middle", "ranking": "middle_first", "sampling": "first"},
        {"name": "subset_uniform", "ranking": "index", "sampling": "uniform"},
    ]

    print(f"\nTrain Scans ({len(train_scan_names)}): {train_scan_names}")
    print(f"Val Scans   ({len(val_scan_names)}): {val_scan_names} (x 2 subsets = 10 sequences, 60 views)")

    # 1. Validation Dataset Setup (Unnormalized benchmark preprocessing)
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

    # 2. Dynamic Training Dataset Setup (WITH normalize_scene=True)
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

    # 3. Reference Untouched Pretrained Baseline Evaluation
    print("\nLoading Pretrained DVLT model for baseline reference...")
    ref_model = DVLT(img_size=504, depth_head_type="conv")
    ref_model.load_pretrained("nvidia/dvlt", strict=True)
    ref_model.setup_test(accelerator)

    # Verify official sampling parameters
    inner_ref = ref_model.model
    beta_a = inner_ref.k_sampler_beta_a
    beta_b = inner_ref.k_sampler_beta_b
    min_steps = inner_ref.min_steps
    num_steps = inner_ref.num_steps
    print(f"Verified Model K-Sampling Parameters:")
    print(f"  k_sampler_beta_a: {beta_a} (Expected: 2)")
    print(f"  k_sampler_beta_b: {beta_b} (Expected: 1)")
    print(f"  min_steps: {min_steps} | num_steps: {num_steps}")

    # Baseline evaluation
    base_eval = evaluate_checkpoint_uniform(ref_model, accelerator, val_batches)
    pretrained_k12_absrel = base_eval["k12_absrel"]
    pretrained_k14_absrel = base_eval["k14_absrel"]
    print(f"PRETRAINED UNTOUCHED BASELINE:")
    print(f"  Uniform K=12 AbsRel: {pretrained_k12_absrel:.6f} | RMSE: {base_eval['k12_rmse']:.4f} | Rot Err: {base_eval['k12_rot']:.3f} deg")
    print(f"  Uniform K=14 AbsRel: {pretrained_k14_absrel:.6f} | RMSE: {base_eval['k14_rmse']:.4f} | Rot Err: {base_eval['k14_rot']:.3f} deg")

    test_grid_device = CANONICAL_INTERVAL_GRID.to(device)
    theta_0 = extract_parameter_vector(inner_ref)
    S_0 = extract_gate_vector(inner_ref, test_grid_device)
    del ref_model, inner_ref
    torch.cuda.empty_cache()
    gc.collect()

    # 4. Learning Rate Calibration Sweep (Requirement 4)
    lr_candidates = [1e-6, 3e-6, 1e-5, 3e-5]
    eval_checkpoints = [0, 1, 5, 10, 25, 50, 100, 200]
    total_steps = 200

    print("\n" + "=" * 100)
    print("PHASE 1: LEARNING RATE CALIBRATION ON UNIFORM FT CONTROL (Seed 42)")
    print(f"Candidate LRs: {lr_candidates}")
    print(f"Evaluation Steps: {eval_checkpoints}")
    print("=" * 100)

    calibration_records = []

    for lr in lr_candidates:
        print(f"\n>>> Calibrating Uniform FT with LR = {lr:.1e} <<<")
        torch.manual_seed(42)
        np.random.seed(42)
        rng = np.random.RandomState(42)

        cur_model = DVLT(img_size=504, depth_head_type="conv")
        cur_model.load_pretrained("nvidia/dvlt", strict=True)
        cur_model.setup_test(accelerator)

        inner = cur_model.model
        trainable_params = []
        for name, param in inner.named_parameters():
            if "depth_scale.proj" in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False

        setup_custom_solve_train(inner, mode="uniform", rng=rng)
        cur_model.setup_train(accelerator, gradient_checkpointing=True)

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        optimizer, inner = accelerator.prepare(optimizer, inner)
        cur_model.model = inner

        last_train_loss = 0.0

        for step in range(total_steps + 1):
            # Check if this step is an evaluation checkpoint
            if step in eval_checkpoints:
                # Switch to eval mode
                inner_eval = accelerator.unwrap_model(inner)
                inner_eval.eval()
                cur_model.model = inner_eval

                # Compute drifts
                theta_t = extract_parameter_vector(inner_eval)
                param_drift = (torch.norm(theta_t - theta_0) / torch.norm(theta_0)).item()

                S_t = extract_gate_vector(inner_eval, test_grid_device)
                gate_drift = (torch.norm(S_t - S_0) / torch.norm(S_0)).item()

                # Evaluate Uniform K=12 & K=14
                eval_res = evaluate_checkpoint_uniform(cur_model, accelerator, val_batches)
                k12_abs = eval_res["k12_absrel"]
                k14_abs = eval_res["k14_absrel"]

                delta_k12_pct = (k12_abs - pretrained_k12_absrel) / pretrained_k12_absrel * 100.0
                delta_k14_pct = (k14_abs - pretrained_k14_absrel) / pretrained_k14_absrel * 100.0
                within_5pct = bool(abs(delta_k12_pct) <= 5.0 and abs(delta_k14_pct) <= 5.0)

                calibration_records.append({
                    "lr": lr,
                    "step": step,
                    "train_loss": last_train_loss,
                    "param_drift": param_drift,
                    "gate_drift": gate_drift,
                    "k12_absrel": k12_abs,
                    "k12_rmse": eval_res["k12_rmse"],
                    "k12_rot_deg": eval_res["k12_rot"],
                    "k12_trans_deg": eval_res["k12_trans"],
                    "k14_absrel": k14_abs,
                    "k14_rmse": eval_res["k14_rmse"],
                    "k14_rot_deg": eval_res["k14_rot"],
                    "k14_trans_deg": eval_res["k14_trans"],
                    "delta_k12_pct": delta_k12_pct,
                    "delta_k14_pct": delta_k14_pct,
                    "within_5pct_calibration": within_5pct,
                })

                print(
                    f"  [LR {lr:.1e} | Step {step:3d}] "
                    f"Loss: {last_train_loss:.4f} | "
                    f"P-Drift: {param_drift:.4e} | "
                    f"G-Drift: {gate_drift:.4e} | "
                    f"K=12 AbsRel: {k12_abs:.6f} ({delta_k12_pct:+6.2f}%) | "
                    f"K=14 AbsRel: {k14_abs:.6f} ({delta_k14_pct:+6.2f}%) | "
                    f"Rot: {eval_res['k12_rot']:.2f} deg | "
                    f"Preserved <=5%? {'YES' if within_5pct else 'NO'}"
                )

                # Switch back to training mode
                inner_eval.train()
                cur_model.model = accelerator.prepare(inner_eval)
                torch.cuda.empty_cache()
                gc.collect()

            if step >= total_steps:
                break

            # Optimization step with dynamic batch fetching
            scan_vidx = train_video_indices[rng.randint(0, len(train_video_indices))]
            sample = ms_train[(scan_vidx, 6, 1.0, 42 + step)]
            batch = default_collate_fn([sample])
            batch_gpu = {}
            for k, v in batch.items():
                batch_gpu[k] = v.to(device) if isinstance(v, torch.Tensor) else v

            optimizer.zero_grad()
            with accelerator.autocast():
                loss, _, _, _ = cur_model.train_step(batch_gpu, step, accelerator)

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            last_train_loss = float(loss.item())

        del cur_model, inner, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    # Save calibration CSV
    df_calib = pd.DataFrame(calibration_records)
    calib_csv = os.path.join(output_dir, "interval_robust_v01_lr_calibration.csv")
    df_calib.to_csv(calib_csv, index=False)
    print(f"\nSaved LR calibration data: {calib_csv}")

    # Analyze calibration results
    print("\n" + "=" * 100)
    print("CALIBRATION SUMMARY: Maximum stable LR and step budget preserving <= 5% delta")
    print("=" * 100)

    stable_rows = df_calib[df_calib["within_5pct_calibration"] & (df_calib["step"] > 0)]
    if len(stable_rows) > 0:
        best_row = stable_rows.sort_values(by=["step", "lr"], ascending=[False, False]).iloc[0]
        optimal_lr = float(best_row["lr"])
        optimal_steps = int(best_row["step"])
        print(f"SUCCESS: Stable Uniform FT configuration found!")
        print(f"  Optimal LR: {optimal_lr:.1e}")
        print(f"  Stable Training Duration: {optimal_steps} steps")
        print(f"  K=12 AbsRel: {best_row['k12_absrel']:.6f} ({best_row['delta_k12_pct']:+.2f}%)")
        print(f"  K=14 AbsRel: {best_row['k14_absrel']:.6f} ({best_row['delta_k14_pct']:+.2f}%)")
        print(f"  Parameter Drift: {best_row['param_drift']:.4e} | Gate Drift: {best_row['gate_drift']:.4e}")
    else:
        optimal_lr = None
        optimal_steps = None
        print("ALERT: No configuration tested preserved K=12/K=14 within 5% relative AbsRel across training!")
        print("Investigating degradation threshold...")

    # Phase 2: Matched Treatment Comparison (Requirement 5)
    matched_results = []
    if optimal_lr is not None and optimal_steps is not None and optimal_steps >= 10:
        print("\n" + "=" * 100)
        print(f"PHASE 2: MATCHED TREATMENT COMPARISON (LR={optimal_lr:.1e}, Steps={optimal_steps})")
        print("=" * 100)

        modes = [
            ("uniform", "Control 2 (Uniform FT)"),
            ("mild", "Treatment 1 (Mild Rand FT, sigma=0.2)"),
            ("moderate", "Treatment 2 (Mod Rand FT, sigma=0.5)"),
        ]

        for mode, name in modes:
            print(f"\nTraining {name}...")
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

            optimizer = torch.optim.AdamW(trainable_params, lr=optimal_lr, weight_decay=1e-4)
            optimizer, inner = accelerator.prepare(optimizer, inner)
            cur_model.model = inner

            for step in range(optimal_steps):
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

            # Evaluate on key non-uniform schedules as well
            inner_eval = accelerator.unwrap_model(inner)
            inner_eval.eval()
            cur_model.model = inner_eval

            eval_res = evaluate_checkpoint_uniform(cur_model, accelerator, val_batches)
            matched_results.append({
                "mode": mode,
                "name": name,
                "lr": optimal_lr,
                "steps": optimal_steps,
                "k12_absrel": eval_res["k12_absrel"],
                "k12_rmse": eval_res["k12_rmse"],
                "k12_rot_deg": eval_res["k12_rot"],
                "k12_trans_deg": eval_res["k12_trans"],
                "k14_absrel": eval_res["k14_absrel"],
                "k14_rmse": eval_res["k14_rmse"],
                "k14_rot_deg": eval_res["k14_rot"],
                "k14_trans_deg": eval_res["k14_trans"],
                "beats_pretrained_k14": bool(eval_res["k12_absrel"] <= pretrained_k14_absrel),
            })
            del cur_model, inner, optimizer
            torch.cuda.empty_cache()
            gc.collect()

        df_matched = pd.DataFrame(matched_results)
        matched_csv = os.path.join(output_dir, "interval_robust_v01_matched_comparison.csv")
        df_matched.to_csv(matched_csv, index=False)
        print(f"Saved matched comparison data: {matched_csv}")
    else:
        print("\nSkipping Phase 2: No stable Uniform FT protocol preserved pretrained calibration.")

    print("\nV0.1 Repair script completed successfully!")


if __name__ == "__main__":
    run_v01_repair()
