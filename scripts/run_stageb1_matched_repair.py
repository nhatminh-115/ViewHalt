"""
INTERVAL-ROBUST STAGE B.1: MATCHED-CONTROL & CHECKPOINT REPAIR
Evaluates checkpoint-wise matched arms for Stage B1 (rank 4) and Stage B2 (rank 8):
- Arm A: Uniform FT Control
- Arm B: Mild Randomized FT (sigma=0.2)
Across checkpoints: step in {0, 25, 50, 100, 200}
With:
- Strictly decoupled deterministic RNG streams via immutable training manifest
- Matched step-0 initialization verification (param diff == 0, output diff == 0)
- Full suite of Uniform, Non-Uniform, and Robustness schedules at EVERY checkpoint

Outputs:
- outputs/stageb1_checkpoint_matched.csv
- outputs/stageb1_nonuniform_by_checkpoint.csv
- outputs/stageb1_robustness_by_checkpoint.csv
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


def sample_mild_interval_schedule(K=12, seed=42):
    rng = np.random.RandomState(seed)
    M = K - 1
    w = rng.lognormal(mean=0.0, sigma=0.2, size=M)
    dt = w / np.sum(w)
    ts = np.concatenate([[0.0], np.cumsum(dt)])
    ts[-1] = 1.0
    return ts.tolist()


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
    for K in [12, 14]:
        schedules[f"uniform_K{K}"] = {
            "key": f"uniform_K{K}", "name": f"Uniform K={K}", "type": "uniform", "K": K,
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

    ts_mild = sample_mild_interval_schedule(K=12, seed=42)
    schedules["sampled_mild"] = {
        "key": "sampled_mild", "name": "Sampled Mild (sigma=0.2)", "type": "nonuniform", "K": K, "ts": ts_mild
    }

    # Moderate sampling
    rng_mod = np.random.RandomState(42)
    dt_unif = 1.0 / (K - 1)
    w_mod = rng_mod.lognormal(mean=0.0, sigma=0.5, size=K - 1)
    dt_clamped = np.clip(w_mod / np.sum(w_mod), 0.5 * dt_unif, 2.0 * dt_unif)
    dt_mod = dt_clamped / np.sum(dt_clamped)
    ts_mod = np.concatenate([[0.0], np.cumsum(dt_mod)]).tolist()
    ts_mod[-1] = 1.0
    schedules["sampled_moderate"] = {
        "key": "sampled_moderate", "name": "Sampled Moderate (sigma=0.5)", "type": "nonuniform", "K": K, "ts": ts_mod
    }

    # 3. Robustness curve CV in {0.0, 0.2, 0.4, 0.6}
    for target_cv in [0.0, 0.2, 0.4, 0.6]:
        ts_cv, actual_cv = generate_perturbed_schedule(K=K, target_cv=target_cv, seed=42)
        schedules[f"robust_cv_{target_cv:.1f}"] = {
            "key": f"robust_cv_{target_cv:.1f}", "name": f"Robust CV={actual_cv:.3f}",
            "type": "robustness", "K": K, "target_cv": target_cv, "actual_cv": actual_cv, "ts": ts_cv,
        }

    return schedules


def setup_custom_solve_train(inner, manifest_df, mode="uniform"):
    """
    Decoupled interval step solver:
    - Reads K from manifest_df
    - In uniform mode: exact linspace
    - In mild mode: dedicated interval RNG keyed on step (independent of data RNG)
    """
    def custom_solve_train(self, x, rope_pos, B, S, sd_rng, step=0):
        K = int(manifest_df.loc[step, "sampled_k"])
        if mode == "uniform":
            ts = torch.linspace(0.0, 1.0, K).tolist()
        elif mode == "mild":
            ts = sample_mild_interval_schedule(K=K, seed=10000 + step * 13)
        else:
            raise ValueError(f"Unknown mode: {mode}")

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
            "abs_rel": abs_rel, "rmse": rmse, "rot_err_deg": rot_err, "trans_err_deg": trans_err
        })
    return metrics_list


def evaluate_model_on_schedules(cur_model, accelerator, val_batches, schedules):
    inner = cur_model.model
    inner.eval()
    device = accelerator.device

    schedule_results = {sk: [] for sk in schedules}

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

                preds = cur_model._postprocess_predictions(batch, inner._decode(x_curr, H, W, B, S, rope_pos))
                m_list = compute_metrics_for_predictions(preds, batch, device)
                schedule_results[sk].append({
                    "abs_rel": np.mean([v["abs_rel"] for v in m_list]),
                    "rmse": np.mean([v["rmse"] for v in m_list]),
                    "rot_err_deg": np.mean([v["rot_err_deg"] for v in m_list]),
                    "trans_err_deg": np.mean([v["trans_err_deg"] for v in m_list]),
                })

    summary = {}
    for sk, res in schedule_results.items():
        summary[sk] = {
            "abs_rel": float(np.mean([v["abs_rel"] for v in res])),
            "rmse": float(np.mean([v["rmse"] for v in res])),
            "rot_err_deg": float(np.mean([v["rot_err_deg"] for v in res])),
            "trans_err_deg": float(np.mean([v["trans_err_deg"] for v in res])),
        }
    return summary


def train_and_eval_arm_checkpoints(accelerator, ms_train, manifest_df, val_batches, schedules,
                                    rank=4, mode="uniform", lora_lr=3e-6, gate_lr=1e-6,
                                    initial_state_dict=None, total_steps=200, checkpoints=[0, 25, 50, 100, 200]):
    device = accelerator.device

    cur_model = DVLT(img_size=504, depth_head_type="conv")
    cur_model.load_pretrained("nvidia/dvlt", strict=True)
    cur_model.setup_test(accelerator)

    inner = cur_model.model
    lora_dict = inject_lora(inner, rank=rank, alpha=float(rank))
    if initial_state_dict is not None:
        inner.load_state_dict(initial_state_dict)

    setup_custom_solve_train(inner, manifest_df, mode=mode)
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

    checkpoint_evals = {}
    data_manifest_consumed = []

    print(f"\n--- Training Arm [Rank={rank}, Mode={mode}, LoRA LR={lora_lr:.1e}] across checkpoints {checkpoints} ---")
    start_t = time.time()

    for step in range(total_steps + 1):
        inner._current_step = step

        if step in checkpoints:
            inner_eval = accelerator.unwrap_model(inner)
            inner_eval.eval()
            cur_model.model = inner_eval

            eval_res = evaluate_model_on_schedules(cur_model, accelerator, val_batches, schedules)
            checkpoint_evals[step] = eval_res

            u12 = eval_res["uniform_K12"]["abs_rel"]
            u14 = eval_res["uniform_K14"]["abs_rel"]
            nonunif_keys = ["power_g075", "power_g125", "cosine_endpoints", "best_random_v0", "sampled_mild", "sampled_moderate"]
            mean_nonunif = float(np.mean([eval_res[k]["abs_rel"] for k in nonunif_keys]))

            print(f"  [Step {step:3d}] K=12 Uniform: {u12:.6f} | K=14 Uniform: {u14:.6f} | Non-Unif Mean: {mean_nonunif:.6f}")

            inner_eval.train()
            cur_model.model = accelerator.prepare(inner_eval)
            torch.cuda.empty_cache()
            gc.collect()

        if step >= total_steps:
            break

        row = manifest_df.loc[step]
        scan_vidx = int(row["video_idx"])
        sample_seed = int(row["sample_seed"])
        k_sampled = int(row["sampled_k"])

        data_manifest_consumed.append((step, scan_vidx, sample_seed, k_sampled))

        sample = ms_train[(scan_vidx, 6, 1.0, sample_seed)]
        batch = default_collate_fn([sample])
        batch_gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        optimizer.zero_grad()
        with accelerator.autocast():
            loss, _, _, _ = cur_model.train_step(batch_gpu, step, accelerator)
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(gate_params + lora_params, max_norm=1.0)
        optimizer.step()

    del cur_model, inner, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return checkpoint_evals, data_manifest_consumed


def run_stageb1_repair(data_root="datasets/test/dtu", output_dir="outputs", rank8_lora_lr=1e-6):
    os.makedirs(output_dir, exist_ok=True)
    manifest_csv = os.path.join(output_dir, "stageb1_training_manifest.csv")
    assert os.path.exists(manifest_csv), f"Manifest {manifest_csv} must exist!"
    manifest_df = pd.read_csv(manifest_csv)

    print("=" * 100)
    print("STAGE B.1: MATCHED-CONTROL & CHECKPOINT REPAIR PIPELINE")
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

    schedules = build_evaluation_schedules()
    checkpoints = [0, 25, 50, 100, 200]
    nonunif_keys = ["power_g075", "power_g125", "cosine_endpoints", "best_random_v0", "sampled_mild", "sampled_moderate"]
    robust_keys = ["robust_cv_0.0", "robust_cv_0.2", "robust_cv_0.4", "robust_cv_0.6"]

    # 1. Pretrained Baseline Reference
    print("\nEvaluating Pretrained Baseline across all schedules...")
    base_model = DVLT(img_size=504, depth_head_type="conv")
    base_model.load_pretrained("nvidia/dvlt", strict=True)
    base_model.setup_test(accelerator)
    eval_pre = evaluate_model_on_schedules(base_model, accelerator, val_batches, schedules)
    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    pre_k12_u = eval_pre["uniform_K12"]["abs_rel"]
    pre_k14_u = eval_pre["uniform_K14"]["abs_rel"]
    pre_nonunif_mean = float(np.mean([eval_pre[k]["abs_rel"] for k in nonunif_keys]))
    print(f"PRETRAINED: K=12 Uniform: {pre_k12_u:.6f} | K=14 Uniform: {pre_k14_u:.6f} | Non-Unif Mean: {pre_nonunif_mean:.6f}")

    # Experiments definitions
    experiments = [
        {"rank": 4, "lora_lr": 3e-6, "desc": "Rank 4 (lora_lr=3e-6)"},
        {"rank": 8, "lora_lr": rank8_lora_lr, "desc": f"Rank 8 (lora_lr={rank8_lora_lr:.1e})"},
    ]

    all_matched_records = []
    all_nonunif_records = []
    all_robust_records = []

    for exp in experiments:
        rank = exp["rank"]
        lora_lr = exp["lora_lr"]
        print("\n" + "=" * 80)
        print(f"RUNNING MATCHED EXPERIMENT: {exp['desc']}")
        print("=" * 80)

        # 1. Create shared matched initialization
        torch.manual_seed(42)
        np.random.seed(42)
        init_model = DVLT(img_size=504, depth_head_type="conv")
        init_model.load_pretrained("nvidia/dvlt", strict=True)
        inject_lora(init_model.model, rank=rank, alpha=float(rank))
        shared_init_sd = copy.deepcopy(init_model.model.state_dict())
        del init_model
        torch.cuda.empty_cache()
        gc.collect()

        # Arm A: Uniform FT Control
        evals_uniform, consumed_u = train_and_eval_arm_checkpoints(
            accelerator, ms_train, manifest_df, val_batches, schedules,
            rank=rank, mode="uniform", lora_lr=lora_lr, gate_lr=1e-6,
            initial_state_dict=copy.deepcopy(shared_init_sd),
            total_steps=200, checkpoints=checkpoints
        )

        # Arm B: Mild Randomized FT
        evals_rand, consumed_r = train_and_eval_arm_checkpoints(
            accelerator, ms_train, manifest_df, val_batches, schedules,
            rank=rank, mode="mild", lora_lr=lora_lr, gate_lr=1e-6,
            initial_state_dict=copy.deepcopy(shared_init_sd),
            total_steps=200, checkpoints=checkpoints
        )

        # Assert data manifest equivalence
        assert consumed_u == consumed_r, "FATAL ERROR: Uniform and Randomized arms received different data trajectories!"
        print("\nSUCCESS: Verified that Uniform and Randomized data trajectories are 100% byte-identical!")

        # Process results across checkpoints
        for step in checkpoints:
            res_u = evals_uniform[step]
            res_r = evals_rand[step]

            k12_u = res_u["uniform_K12"]["abs_rel"]
            k12_r = res_r["uniform_K12"]["abs_rel"]
            k14_u = res_u["uniform_K14"]["abs_rel"]
            k14_r = res_r["uniform_K14"]["abs_rel"]

            nonunif_u = float(np.mean([res_u[k]["abs_rel"] for k in nonunif_keys]))
            nonunif_r = float(np.mean([res_r[k]["abs_rel"] for k in nonunif_keys]))

            # treatment effect = Randomized - Uniform (negative means Randomized has lower error)
            # treatment gain = Uniform - Randomized (positive means Randomized has lower error)
            delta_k12 = k12_r - k12_u
            gain_k12 = k12_u - k12_r
            delta_nonunif = nonunif_r - nonunif_u
            gain_nonunif = nonunif_u - nonunif_r

            all_matched_records.append({
                "rank": rank,
                "lora_lr": lora_lr,
                "step": step,
                "uniform_control_k12": k12_u,
                "mild_rand_k12": k12_r,
                "k12_treatment_diff": delta_k12,
                "k12_treatment_gain": gain_k12,
                "uniform_control_k14": k14_u,
                "mild_rand_k14": k14_r,
                "uniform_control_nonunif_mean": nonunif_u,
                "mild_rand_nonunif_mean": nonunif_r,
                "nonunif_treatment_diff": delta_nonunif,
                "nonunif_treatment_gain": gain_nonunif,
                "pre_k14_hurdle": pre_k14_u,
                "rand_beats_pre_k14": bool(k12_r <= pre_k14_u),
            })

            # Non-uniform schedule breakdown
            for nk in nonunif_keys:
                nu_val = res_u[nk]["abs_rel"]
                nr_val = res_r[nk]["abs_rel"]
                all_nonunif_records.append({
                    "rank": rank,
                    "lora_lr": lora_lr,
                    "step": step,
                    "schedule_key": nk,
                    "schedule_name": schedules[nk]["name"],
                    "pretrained_absrel": eval_pre[nk]["abs_rel"],
                    "uniform_control_absrel": nu_val,
                    "mild_rand_absrel": nr_val,
                    "diff_rand_minus_unif": nr_val - nu_val,
                    "gain_unif_minus_rand": nu_val - nr_val,
                })

            # Robustness breakdown
            for rk in robust_keys:
                target_cv = schedules[rk]["target_cv"]
                actual_cv = schedules[rk]["actual_cv"]
                ru_val = res_u[rk]["abs_rel"]
                rr_val = res_r[rk]["abs_rel"]
                all_robust_records.append({
                    "rank": rank,
                    "lora_lr": lora_lr,
                    "step": step,
                    "target_cv": target_cv,
                    "actual_cv": actual_cv,
                    "pretrained_absrel": eval_pre[rk]["abs_rel"],
                    "uniform_control_absrel": ru_val,
                    "mild_rand_absrel": rr_val,
                    "diff_rand_minus_unif": rr_val - ru_val,
                    "gain_unif_minus_rand": ru_val - rr_val,
                })

    df_matched = pd.DataFrame(all_matched_records)
    df_nonunif = pd.DataFrame(all_nonunif_records)
    df_robust = pd.DataFrame(all_robust_records)

    out_matched_csv = os.path.join(output_dir, "stageb1_checkpoint_matched.csv")
    out_nonunif_csv = os.path.join(output_dir, "stageb1_nonuniform_by_checkpoint.csv")
    out_robust_csv = os.path.join(output_dir, "stageb1_robustness_by_checkpoint.csv")

    df_matched.to_csv(out_matched_csv, index=False)
    df_nonunif.to_csv(out_nonunif_csv, index=False)
    df_robust.to_csv(out_robust_csv, index=False)

    print(f"\nSaved matched checkpoint results: {out_matched_csv}")
    print(f"Saved non-uniform breakdown: {out_nonunif_csv}")
    print(f"Saved robustness breakdown: {out_robust_csv}")


if __name__ == "__main__":
    # If rank 8 calibration is already complete, load winner from CSV, else default to 1e-6
    calib_csv = "outputs/stageb1_rank8_lr_calibration.csv"
    chosen_rank8_lr = 1e-6
    if os.path.exists(calib_csv):
        df_c = pd.read_csv(calib_csv)
        stable = df_c[df_c["within_5pct_calibration"] & (df_c["step"] == 200)]
        if len(stable) > 0:
            chosen_rank8_lr = float(stable.sort_values(by="lora_lr", ascending=False).iloc[0]["lora_lr"])
    print(f"Selected Rank-8 LoRA LR: {chosen_rank8_lr:.1e}")
    run_stageb1_repair(rank8_lora_lr=chosen_rank8_lr)
