"""
INDEPENDENT RANK-8 LoRA LEARNING RATE CALIBRATION (STAGE B.1)
Screens lora_lr in {5e-7, 1e-6, 3e-6, 1e-5} with gate_lr fixed at 1e-6
on the Uniform FT control across checkpoints {0, 25, 50, 100, 200}.
Uses the pre-generated immutable training manifest: outputs/stageb1_training_manifest.csv.
Saves results to: outputs/stageb1_rank8_lr_calibration.csv.
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
from accelerate import Accelerator
from omegaconf import OmegaConf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Hardware cap: max 85% VRAM (<= 6.9 GB)
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
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 8.0):
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


def inject_lora(inner, rank=8, alpha=None):
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


def setup_custom_solve_train(inner, manifest_df):
    def custom_solve_train(self, x, rope_pos, B, S, sd_rng, step=0):
        # Read pre-sampled K from manifest
        K = int(manifest_df.loc[step, "sampled_k"])
        ts = torch.linspace(0.0, 1.0, K).tolist()
        for i in range(K):
            t_now = ts[i]
            t_next = ts[i + 1] if i + 1 < K else 1.0
            x = self._interval_step(x, t_now, t_next, rope_pos, B, S)
        return x

    inner._solve_train_linspace_k = types.MethodType(
        lambda self, x, rope_pos, B, S, sd_rng: custom_solve_train(self, x, rope_pos, B, S, sd_rng, step=self._current_step),
        inner
    )


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
            "abs_rel": abs_rel,
            "rmse": rmse,
            "rot_err_deg": rot_err,
            "trans_err_deg": trans_err,
        })
    return metrics_list


def evaluate_checkpoint_uniform(model, accelerator, val_batches):
    inner = model.model
    inner.eval()
    device = accelerator.device

    results = {"k12_absrel": [], "k12_rmse": [], "k12_rot": [], "k12_trans": [],
               "k14_absrel": [], "k14_rmse": [], "k14_rot": [], "k14_trans": []}

    ts_k12 = torch.linspace(0.0, 1.0, 12).tolist()
    ts_k14 = torch.linspace(0.0, 1.0, 14).tolist()

    with torch.no_grad(), accelerator.autocast():
        for seq_id, batch in val_batches.items():
            images = batch[DataField.IMAGES]
            B, S, _, H, W = images.shape

            z_0 = inner._encode_images(images)
            reg_tok = inner.register_token.expand(B, S, -1, -1).reshape(B * S, inner.num_register_tokens, -1)
            cam_tok = _slice_expand_flatten(inner.camera_token, B, S)
            x_init = torch.cat([cam_tok, reg_tok, z_0], dim=1)
            rope_pos = inner._get_rope_positions(B * S, H, W, images.device)

            for ts, tag in [(ts_k12, "k12"), (ts_k14, "k14")]:
                x_curr = x_init.clone()
                K = len(ts)
                for i in range(K):
                    t_now = ts[i]
                    t_next = ts[i + 1] if i + 1 < K else 1.0
                    t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)
                    block = inner.recurrent_blocks[0]
                    x_frame = block.frame_attn(x_curr, t_pair.expand(B * S, -1), pos=rope_pos.reshape(B * S, x_curr.shape[1], 2))
                    x_global = block.global_attn(x_frame.reshape(B, S * x_curr.shape[1], -1), t_pair.expand(B, -1), pos=None).reshape(B * S, x_curr.shape[1], -1)
                    x_curr = x_global

                preds = model._postprocess_predictions(batch, inner._decode(x_curr, H, W, B, S, rope_pos))
                m_list = compute_metrics_for_predictions(preds, batch, device)

                results[f"{tag}_absrel"].append(np.mean([v["abs_rel"] for v in m_list]))
                results[f"{tag}_rmse"].append(np.mean([v["rmse"] for v in m_list]))
                results[f"{tag}_rot"].append(np.mean([v["rot_err_deg"] for v in m_list]))
                results[f"{tag}_trans"].append(np.mean([v["trans_err_deg"] for v in m_list]))

    return {k: float(np.mean(v)) for k, v in results.items()}


def calibrate_rank8_lora_lr(data_root="datasets/test/dtu", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    manifest_csv = os.path.join(output_dir, "stageb1_training_manifest.csv")
    assert os.path.exists(manifest_csv), f"Manifest {manifest_csv} must exist!"
    manifest_df = pd.read_csv(manifest_csv)

    print("=" * 100)
    print("STAGE B.1: INDEPENDENT RANK-8 LoRA LEARNING RATE CALIBRATION")
    print("=" * 100)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    print(f"Device: {device} | Precision: bf16")

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

    MultiSourceDataset.MIN_LEN = 14
    train_config = {
        "normalize_scene": True,
        "load_data_fields": ["images", "extrinsics_c2w", "intrinsics", "depths", "world_points", "point_masks"],
    }
    ms_train = MultiSourceDataset({"dtu": ds_train}, training=True, **train_config)

    # Pretrained baseline evaluation
    print("\nLoading Pretrained DVLT model for baseline reference...")
    base_model = DVLT(img_size=504, depth_head_type="conv")
    base_model.load_pretrained("nvidia/dvlt", strict=True)
    base_model.setup_test(accelerator)

    eval_baseline = evaluate_checkpoint_uniform(base_model, accelerator, val_batches)
    baseline_k12_abs = eval_baseline["k12_absrel"]
    baseline_k14_abs = eval_baseline["k14_absrel"]
    print("PRETRAINED UNTOUCHED BASELINE:")
    print(f"  Uniform K=12 AbsRel: {baseline_k12_abs:.6f} | RMSE: {eval_baseline['k12_rmse']:.4f} | Rot: {eval_baseline['k12_rot']:.3f} deg")
    print(f"  Uniform K=14 AbsRel: {baseline_k14_abs:.6f} | RMSE: {eval_baseline['k14_rmse']:.4f} | Rot: {eval_baseline['k14_rot']:.3f} deg")

    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    # Screen lora_lr in {5e-7, 1e-6, 3e-6, 1e-5}
    candidates_lora_lr = [5e-7, 1e-06, 3e-06, 1e-05]
    checkpoints = [0, 25, 50, 100, 200]
    total_steps = 200
    gate_lr = 1.0e-06

    calibration_records = []

    for lora_lr in candidates_lora_lr:
        print("\n" + "=" * 80)
        print(f">>> Calibrating Rank-8 LoRA LR = {lora_lr:.1e} (Gate LR = {gate_lr:.1e}) <<<")
        print("=" * 80)

        torch.manual_seed(42)
        np.random.seed(42)

        cur_model = DVLT(img_size=504, depth_head_type="conv")
        cur_model.load_pretrained("nvidia/dvlt", strict=True)
        cur_model.setup_test(accelerator)

        inner = cur_model.model
        lora_dict = inject_lora(inner, rank=8, alpha=8.0)
        setup_custom_solve_train(inner, manifest_df)
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
            inner._current_step = step

            if step in checkpoints:
                inner_eval = accelerator.unwrap_model(inner)
                inner_eval.eval()
                cur_model.model = inner_eval

                eval_res = evaluate_checkpoint_uniform(cur_model, accelerator, val_batches)
                k12_abs = eval_res["k12_absrel"]
                k14_abs = eval_res["k14_absrel"]
                delta_k12_pct = ((k12_abs - baseline_k12_abs) / baseline_k12_abs) * 100.0
                delta_k14_pct = ((k14_abs - baseline_k14_abs) / baseline_k14_abs) * 100.0

                within_5pct = (delta_k12_pct <= 5.0) and (delta_k14_pct <= 5.0) and (eval_res["k12_rot"] < 5.0)

                lora_update_norm = 0.0
                for lm in lora_dict.values():
                    W_delta = lm.scale * (lm.lora_A @ lm.lora_B)
                    lora_update_norm += float(torch.norm(W_delta).item() ** 2)
                lora_update_norm = float(np.sqrt(lora_update_norm))

                calibration_records.append({
                    "rank": 8,
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
                    f"  [Rank-8 LoRA LR {lora_lr:.1e} | Step {step:3d}] "
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

            # Strictly feed from immutable manifest
            row = manifest_df.loc[step]
            scan_vidx = int(row["video_idx"])
            sample_seed = int(row["sample_seed"])

            sample = ms_train[(scan_vidx, 6, 1.0, sample_seed)]
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
    calib_csv = os.path.join(output_dir, "stageb1_rank8_lr_calibration.csv")
    df_calib.to_csv(calib_csv, index=False)
    print(f"\nSaved Rank-8 calibration results: {calib_csv}")

    # Summary
    print("\n" + "=" * 100)
    print("STAGE B.1 RANK-8 CALIBRATION SUMMARY:")
    print("=" * 100)
    for lr in candidates_lora_lr:
        sub = df_calib[df_calib["lora_lr"] == lr]
        step200 = sub[sub["step"] == 200].iloc[0]
        step50 = sub[sub["step"] == 50].iloc[0]
        print(f"LR {lr:.1e}: Step 50 K=12={step50['k12_absrel']:.6f} ({step50['delta_k12_pct']:+.2f}%) | Step 200 K=12={step200['k12_absrel']:.6f} ({step200['delta_k12_pct']:+.2f}%) | <=5%? {step200['within_5pct_calibration']}")


if __name__ == "__main__":
    calibrate_rank8_lora_lr()
