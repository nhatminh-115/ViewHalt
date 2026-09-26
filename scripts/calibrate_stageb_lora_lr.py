"""
STAGE B1: LEARNING RATE CALIBRATION FOR SHARED RECURRENT ATTENTION LoRA (Rank 4)
Screens lora_lr in {3e-6, 1e-5, 3e-5, 1e-4} with gate_lr fixed at 1e-6 on Uniform FT control.
Evaluates at step in {0, 10, 25, 50, 100, 200}.
Saves results to outputs/stageb_lr_calibration.csv.
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


def inject_lora(inner, rank=4, alpha=4.0):
    """Wraps the 4 attention projections in recurrent block 0 with shared LoRA."""
    block = inner.recurrent_blocks[0]
    targets = [
        (block.frame_attn.attn, "qkv"),
        (block.frame_attn.attn, "proj"),
        (block.global_attn.attn, "qkv"),
        (block.global_attn.attn, "proj"),
    ]
    lora_modules = []
    for parent, attr in targets:
        orig_layer = getattr(parent, attr)
        lora_layer = LoRALinear(orig_layer, rank=rank, alpha=alpha)
        setattr(parent, attr, lora_layer)
        lora_modules.append(lora_layer)
    return lora_modules


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


def compute_lora_norm(lora_layers):
    """Computes Frobenius norm of LoRA updates: scale * ||A @ B||."""
    total_norm_sq = 0.0
    for l in lora_layers:
        W_delta = l.scale * (l.lora_A @ l.lora_B)
        total_norm_sq += (W_delta ** 2).sum().item()
    return float(np.sqrt(total_norm_sq))


def calibrate_lora_lr(data_root="datasets/test/dtu", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print("STAGE B1: LEARNING RATE CALIBRATION FOR SHARED RECURRENT ATTENTION LoRA")
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
        "normalize_scene": True,
        "load_data_fields": ["images", "extrinsics_c2w", "intrinsics", "depths", "world_points", "point_masks"],
    }
    ms_train = MultiSourceDataset({"dtu": ds_train}, training=True, **train_config)

    # Pretrained baseline evaluation
    print("\nLoading Pretrained DVLT model for baseline reference...")
    ref_model = DVLT(img_size=504, depth_head_type="conv")
    ref_model.load_pretrained("nvidia/dvlt", strict=True)
    ref_model.setup_test(accelerator)
    base_eval = evaluate_checkpoint_uniform(ref_model, accelerator, val_batches)
    pretrained_k12_absrel = base_eval["k12_absrel"]
    pretrained_k14_absrel = base_eval["k14_absrel"]
    print(f"PRETRAINED UNTOUCHED BASELINE:")
    print(f"  Uniform K=12 AbsRel: {pretrained_k12_absrel:.6f} | RMSE: {base_eval['k12_rmse']:.4f} | Rot: {base_eval['k12_rot']:.3f} deg")
    print(f"  Uniform K=14 AbsRel: {pretrained_k14_absrel:.6f} | RMSE: {base_eval['k14_rmse']:.4f} | Rot: {base_eval['k14_rot']:.3f} deg")
    del ref_model
    torch.cuda.empty_cache()
    gc.collect()

    # Calibration parameters
    gate_lr = 1e-6
    lora_lrs = [3e-6, 1e-5, 3e-5, 1e-4]
    eval_checkpoints = [0, 10, 25, 50, 100, 200]
    total_steps = 200

    calibration_records = []

    for lora_lr in lora_lrs:
        print(f"\n{'='*80}\n>>> Calibrating LoRA LR = {lora_lr:.1e} (Gate LR = {gate_lr:.1e}) <<<\n{'='*80}")
        torch.manual_seed(42)
        np.random.seed(42)
        rng = np.random.RandomState(42)

        cur_model = DVLT(img_size=504, depth_head_type="conv")
        cur_model.load_pretrained("nvidia/dvlt", strict=True)
        cur_model.setup_test(accelerator)

        inner = cur_model.model
        lora_layers = inject_lora(inner, rank=4, alpha=4.0)
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

        last_train_loss = 0.0

        for step in range(total_steps + 1):
            if step in eval_checkpoints:
                inner_eval = accelerator.unwrap_model(inner)
                inner_eval.eval()
                cur_model.model = inner_eval

                eval_res = evaluate_checkpoint_uniform(cur_model, accelerator, val_batches)
                k12_abs = eval_res["k12_absrel"]
                k14_abs = eval_res["k14_absrel"]
                delta_k12_pct = (k12_abs - pretrained_k12_absrel) / pretrained_k12_absrel * 100.0
                delta_k14_pct = (k14_abs - pretrained_k14_absrel) / pretrained_k14_absrel * 100.0
                within_5pct = (abs(delta_k12_pct) <= 5.0) and (abs(delta_k14_pct) <= 5.0)

                lora_update_norm = compute_lora_norm(lora_layers)

                calibration_records.append({
                    "gate_lr": gate_lr,
                    "lora_lr": lora_lr,
                    "step": step,
                    "train_loss": last_train_loss,
                    "lora_update_norm": lora_update_norm,
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
                    f"  [LoRA LR {lora_lr:.1e} | Step {step:3d}] "
                    f"Loss: {last_train_loss:.4f} | "
                    f"LoRA Norm: {lora_update_norm:.4e} | "
                    f"K=12: {k12_abs:.6f} ({delta_k12_pct:+6.2f}%) | "
                    f"K=14: {k14_abs:.6f} ({delta_k14_pct:+6.2f}%) | "
                    f"Rot: {eval_res['k12_rot']:.2f} deg | "
                    f"<=5%? {'YES' if within_5pct else 'NO'}"
                )

                inner_eval.train()
                cur_model.model = accelerator.prepare(inner_eval)
                torch.cuda.empty_cache()
                gc.collect()

            if step >= total_steps:
                break

            scan_vidx = train_video_indices[rng.randint(0, len(train_video_indices))]
            sample = ms_train[(scan_vidx, 6, 1.0, 42 + step)]
            batch = default_collate_fn([sample])
            batch_gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            optimizer.zero_grad()
            with accelerator.autocast():
                loss, _, _, _ = cur_model.train_step(batch_gpu, step, accelerator)
            accelerator.backward(loss)
            all_trainable = gate_params + lora_params
            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            optimizer.step()
            last_train_loss = float(loss.item())

        del cur_model, inner, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    df_calib = pd.DataFrame(calibration_records)
    calib_csv = os.path.join(output_dir, "stageb_lr_calibration.csv")
    df_calib.to_csv(calib_csv, index=False)
    print(f"\nSaved calibration results: {calib_csv}")

    # Summary
    print("\n" + "=" * 100)
    print("STAGE B1 CALIBRATION SUMMARY:")
    print("=" * 100)
    stable_rows = df_calib[df_calib["within_5pct_calibration"] & (df_calib["step"] == 200)]
    if len(stable_rows) > 0:
        best_row = stable_rows.sort_values(by="lora_lr", ascending=False).iloc[0]
        opt_lora_lr = float(best_row["lora_lr"])
        print(f"SUCCESS: Highest stable LoRA LR at step 200 is {opt_lora_lr:.1e}!")
        print(f"  K=12 AbsRel: {best_row['k12_absrel']:.6f} ({best_row['delta_k12_pct']:+.2f}%)")
        print(f"  K=14 AbsRel: {best_row['k14_absrel']:.6f} ({best_row['delta_k14_pct']:+.2f}%)")
        print(f"  LoRA Update Norm: {best_row['lora_update_norm']:.4e}")
    else:
        print("ALERT: No screened LoRA LR preserved calibration at step 200! Selecting best sub-200 step configuration...")


if __name__ == "__main__":
    calibrate_lora_lr()
