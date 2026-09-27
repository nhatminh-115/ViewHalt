"""
STAGE B.1C: FINAL 50-STEP OFFICIAL-K CONFIRMATION
Focus confirmatory test on Step 50 using official Beta(2,1) K-sampling:
- Rank 4 (lora_lr=3e-6, gate_lr=1e-6): Uniform FT vs Mild Rand FT
- Rank 8 (lora_lr=1e-6, gate_lr=1e-6): Uniform FT vs Mild Rand FT

Evaluates minimal schedule suite at step 50:
- Uniform K=12, K=14
- Non-uniform K=12: Power (0.75, 1.25), Cosine, Best Random (s=999), Sampled Mild, Sampled Moderate
- Robustness: CV=0.0, CV=0.6

Outputs:
- outputs/stageb1c_results.csv
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


def generate_perturbed_schedule(K=12, target_cv=0.6, seed=42):
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

    # 3. Robustness curve CV=0.6 (CV=0.0 is uniform_K12)
    ts_cv06, actual_cv06 = generate_perturbed_schedule(K=K, target_cv=0.6, seed=42)
    schedules["robust_cv_0.6"] = {
        "key": "robust_cv_0.6", "name": f"Robust CV={actual_cv06:.3f}",
        "type": "robustness", "K": K, "target_cv": 0.6, "actual_cv": actual_cv06, "ts": ts_cv06,
    }

    return schedules


def setup_custom_solve_train(inner, manifest_df, mode="uniform"):
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


def train_50_steps_arm(accelerator, ms_train, manifest_df, rank=4, mode="uniform",
                       lora_lr=3e-6, gate_lr=1e-6, initial_state_dict=None):
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

    data_manifest_consumed = []
    print(f"\nTraining Arm [Rank={rank}, Mode={mode}, LoRA LR={lora_lr:.1e}] for 50 steps...")

    for step in range(50):
        inner._current_step = step

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

        if (step + 1) % 25 == 0 or step == 0:
            print(f"  Step {step+1:2d}/50 | Loss: {loss.item():.4f} | K={k_sampled}")

    inner_eval = accelerator.unwrap_model(inner)
    inner_eval.eval()
    cur_model.model = inner_eval

    return cur_model, data_manifest_consumed


def run_stageb1c_confirmation(data_root="datasets/test/dtu", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    manifest_csv = os.path.join(output_dir, "stageb1c_official_k_manifest.csv")
    assert os.path.exists(manifest_csv), f"Manifest {manifest_csv} must exist!"
    manifest_df = pd.read_csv(manifest_csv)

    print("=" * 100)
    print("STAGE B.1C: FINAL 50-STEP OFFICIAL-K CONFIRMATION")
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
    nonunif_keys = ["power_g075", "power_g125", "cosine_endpoints", "best_random_v0", "sampled_mild", "sampled_moderate"]

    # 1. Evaluate Pretrained Baseline
    print("\nEvaluating Pretrained Baseline across minimal confirmation schedules...")
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
    print(f"PRETRAINED: K=12: {pre_k12_u:.6f} | K=14: {pre_k14_u:.6f} | Non-Unif Mean: {pre_nonunif_mean:.6f}")

    experiments = [
        {"rank": 4, "lora_lr": 3e-6, "desc": "Rank 4 (lora_lr=3e-6)"},
        {"rank": 8, "lora_lr": 1e-6, "desc": "Rank 8 (lora_lr=1e-6)"},
    ]

    all_results_records = []

    for exp in experiments:
        rank = exp["rank"]
        lora_lr = exp["lora_lr"]
        print("\n" + "=" * 80)
        print(f"RUNNING 50-STEP CONFIRMATION: {exp['desc']}")
        print("=" * 80)

        # 1. Matched initialization
        torch.manual_seed(42)
        np.random.seed(42)
        init_model = DVLT(img_size=504, depth_head_type="conv")
        init_model.load_pretrained("nvidia/dvlt", strict=True)
        inject_lora(init_model.model, rank=rank, alpha=float(rank))
        shared_init_sd = copy.deepcopy(init_model.model.state_dict())
        del init_model
        torch.cuda.empty_cache()
        gc.collect()

        # Arm A: Uniform FT Control (50 steps)
        model_u, consumed_u = train_50_steps_arm(
            accelerator, ms_train, manifest_df,
            rank=rank, mode="uniform", lora_lr=lora_lr, gate_lr=1e-6,
            initial_state_dict=copy.deepcopy(shared_init_sd)
        )
        eval_u = evaluate_model_on_schedules(model_u, accelerator, val_batches, schedules)
        del model_u
        torch.cuda.empty_cache()
        gc.collect()

        # Arm B: Mild Randomized FT (50 steps)
        model_r, consumed_r = train_50_steps_arm(
            accelerator, ms_train, manifest_df,
            rank=rank, mode="mild", lora_lr=lora_lr, gate_lr=1e-6,
            initial_state_dict=copy.deepcopy(shared_init_sd)
        )
        eval_r = evaluate_model_on_schedules(model_r, accelerator, val_batches, schedules)
        del model_r
        torch.cuda.empty_cache()
        gc.collect()

        # Assert data manifest equivalence
        assert consumed_u == consumed_r, "FATAL ERROR: Manifest mismatch!"
        print(f"ASSERTION PASSED: Rank {rank} Uniform and Randomized data trajectories are 100% byte-identical!")

        # Detailed schedule records
        for sk, s_info in schedules.items():
            u_abs = eval_u[sk]["abs_rel"]
            r_abs = eval_r[sk]["abs_rel"]
            p_abs = eval_pre[sk]["abs_rel"]

            treatment_gain = u_abs - r_abs
            treatment_gain_pct = (treatment_gain / u_abs) * 100.0
            beats_pre_k14 = bool(r_abs <= pre_k14_u)

            all_results_records.append({
                "rank": rank,
                "schedule_key": sk,
                "schedule_name": s_info["name"],
                "schedule_type": s_info["type"],
                "pretrained_absrel": p_abs,
                "uniform_control_absrel": u_abs,
                "mild_rand_absrel": r_abs,
                "treatment_gain_absrel": treatment_gain,
                "treatment_gain_pct": treatment_gain_pct,
                "beats_pre_k14_hurdle": beats_pre_k14,
            })

        # Summary rows for this rank
        u_nonunif_mean = float(np.mean([eval_u[k]["abs_rel"] for k in nonunif_keys]))
        r_nonunif_mean = float(np.mean([eval_r[k]["abs_rel"] for k in nonunif_keys]))
        nonunif_gain = u_nonunif_mean - r_nonunif_mean
        nonunif_gain_pct = (nonunif_gain / u_nonunif_mean) * 100.0

        all_results_records.append({
            "rank": rank,
            "schedule_key": "nonunif_mean",
            "schedule_name": "Non-Uniform K=12 Mean (6 schedules)",
            "schedule_type": "summary",
            "pretrained_absrel": pre_nonunif_mean,
            "uniform_control_absrel": u_nonunif_mean,
            "mild_rand_absrel": r_nonunif_mean,
            "treatment_gain_absrel": nonunif_gain,
            "treatment_gain_pct": nonunif_gain_pct,
            "beats_pre_k14_hurdle": bool(r_nonunif_mean <= pre_k14_u),
        })

    df_results = pd.DataFrame(all_results_records)
    out_csv = os.path.join(output_dir, "stageb1c_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved Stage B.1C results: {out_csv}")

    # Print summary table
    print("\n" + "=" * 100)
    print("STAGE B.1C FINAL CONFIRMATION SUMMARY:")
    print("=" * 100)
    summary_sub = df_results[df_results["schedule_key"].isin(["uniform_K12", "uniform_K14", "nonunif_mean", "robust_cv_0.6"])]
    print(summary_sub[["rank", "schedule_name", "uniform_control_absrel", "mild_rand_absrel", "treatment_gain_absrel", "treatment_gain_pct", "beats_pre_k14_hurdle"]].to_string(index=False))


if __name__ == "__main__":
    run_stageb1c_confirmation()
