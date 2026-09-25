"""ViewHalt V0.8: True Asynchronous Freeze & Multi-Regime Correctness Evaluation.

Compares four regimes on all 14 DTU scans x 2 subsets (168 view instances):
- Regime A: Fixed K=16 (and intermediate Fixed i=15, 14, 12, 8)
- Regime B: V0.7 Post-block overwrite freeze oracle (Tol 2% & 5%)
- Regime C: True Asynchronous static reference freeze oracle (Tol 2% & 5%)
- Regime D: Sparse Active-Q / Frozen-KV execution (Tol 2% & 5%)

Validates:
1. Exact numerical equivalence between Regime C and Regime D.
2. Direct comparison between Regime B (temporary overwrite) and Regime C/D (true static).
3. Active-view degradation when neighboring views truly stop updating.
"""

import csv
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf

from dvlt.common.constants import DataField, PredictionField
from dvlt.common.rotation import so3_relative_angle
from dvlt.data.collate import default_collate_fn
from dvlt.data.datasets.multi_source import MultiSourceDataset
from dvlt.data.datasets.parser.dataverse import DataverseEvalDataset
from dvlt.metric.depth import apply_alignment
from dvlt.model.dvlt.model import DVLT, _slice_expand_flatten


def compute_metrics_for_predictions(preds, batch, device):
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


def run_regime_b_post_block_overwrite(inner, x_init, target_k_stars, rope_pos, B, S, K, ts):
    """Regime B: V0.7 post-block overwrite simulation."""
    x = x_init.clone()
    frozen = {}
    for i in range(K):
        t_now = ts[i]
        t_next = ts[i + 1] if i + 1 < K else 1.0
        step_idx = i + 1
        x = inner._interval_step(x, t_now, t_next, rope_pos, B, S)
        for v in range(S):
            if step_idx == target_k_stars[v]:
                frozen[v] = x[v].clone()
            elif step_idx > target_k_stars[v]:
                x[v] = frozen[v].clone()
    return x


def run_regime_c_true_static_reference(inner, x_init, target_k_stars, rope_pos, B, S, K, ts, device):
    """Regime C: Reference True-Static Asynchronous execution."""
    block = inner.recurrent_blocks[0]
    P = x_init.shape[1]
    dim = x_init.shape[2]
    x = x_init.clone()
    frozen = {}

    for i in range(K):
        t_now = ts[i]
        t_next = ts[i + 1] if i + 1 < K else 1.0
        step_idx = i + 1
        t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)

        # Step A: Frame attention (bypass frame update for frozen views)
        x_frame = block.frame_attn(x, t_pair.expand(B * S, -1), pos=rope_pos.reshape(B * S, P, 2))
        for v in range(S):
            if step_idx > target_k_stars[v]:
                x_frame[v] = frozen[v].clone()

        # Step B: Global attention
        x_global_in = x_frame.reshape(B, S * P, dim)
        x_global_out = block.global_attn(x_global_in, t_pair.expand(B, -1), pos=None).reshape(B * S, P, dim)

        for v in range(S):
            if step_idx == target_k_stars[v]:
                frozen[v] = x_global_out[v].clone()
            elif step_idx > target_k_stars[v]:
                x_global_out[v] = frozen[v].clone()

        x = x_global_out
    return x


def run_regime_d_sparse_active_q_frozen_kv(inner, x_init, target_k_stars, rope_pos, B, S, K, ts, device):
    """Regime D: Sparse Active-Q / Frozen-KV Execution with KV Caching."""
    block = inner.recurrent_blocks[0]
    g_attn_block = block.global_attn
    attn_mod = g_attn_block.attn
    num_heads = attn_mod.num_heads
    head_dim = attn_mod.head_dim
    P = x_init.shape[1]
    dim = x_init.shape[2]

    x = x_init.clone()
    frozen = {}
    frozen_kv = {}

    for i in range(K):
        t_now = ts[i]
        t_next = ts[i + 1] if i + 1 < K else 1.0
        step_idx = i + 1
        t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)

        active_v = [v for v in range(S) if step_idx <= target_k_stars[v]]
        frozen_v = [v for v in range(S) if step_idx > target_k_stars[v]]

        # Step A: Frame attention ONLY for active views
        x_frame = torch.empty_like(x)
        for v in frozen_v:
            x_frame[v] = frozen[v]

        if len(active_v) > 0:
            act_idx = torch.tensor(active_v, device=device)
            x_act = x[act_idx]
            pos_act = rope_pos.reshape(B * S, P, 2)[act_idx]
            t_act = t_pair.expand(len(active_v), -1)
            x_act_frame = block.frame_attn(x_act, t_act, pos=pos_act)
            for idx, v in enumerate(active_v):
                x_frame[v] = x_act_frame[idx]

        if len(active_v) == 0:
            x = x_frame
            continue

        # Step B: Global attention sparse execution
        s = g_attn_block.depth_scale(t_pair.expand(B, -1)).unsqueeze(1)
        s_attn, s_mlp, s_out = s.chunk(3, dim=-1)

        x_act_in = x_frame[active_v]
        norm_act = g_attn_block.norm1(x_act_in)
        qkv_act = attn_mod.qkv(norm_act).reshape(len(active_v), P, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q_act_views, k_act_views, v_act_views = qkv_act.unbind(0)
        q_act_views = attn_mod.q_norm(q_act_views)
        k_act_views = attn_mod.k_norm(k_act_views)

        # Cache K, V for frozen views
        for v in frozen_v:
            if v not in frozen_kv:
                norm_v = g_attn_block.norm1(frozen[v].unsqueeze(0))
                qkv_v = attn_mod.qkv(norm_v).reshape(1, P, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                _, k_v, v_v = qkv_v.unbind(0)
                k_v = attn_mod.k_norm(k_v)
                frozen_kv[v] = (k_v.squeeze(0), v_v.squeeze(0))

        # Full K and V
        k_list, v_list = [], []
        for v in range(S):
            if v in frozen_v:
                k_list.append(frozen_kv[v][0])
                v_list.append(frozen_kv[v][1])
            else:
                act_local_idx = active_v.index(v)
                k_list.append(k_act_views[act_local_idx])
                v_list.append(v_act_views[act_local_idx])

        full_k = torch.cat(k_list, dim=1).unsqueeze(0)
        full_v = torch.cat(v_list, dim=1).unsqueeze(0)
        q_act_cat = torch.cat([q_act_views[i] for i in range(len(active_v))], dim=1).unsqueeze(0)

        # SDPA
        attn_out_act = F.scaled_dot_product_attention(q_act_cat, full_k, full_v, dropout_p=0.0)
        attn_out_act = attn_out_act.transpose(1, 2).reshape(len(active_v), P, dim)
        proj_act = attn_mod.proj(attn_out_act)

        branch1_act = s_attn * g_attn_block.ls1(proj_act)
        x_act_res1 = x_act_in + branch1_act

        norm2_act = g_attn_block.norm2(x_act_res1)
        mlp_act = g_attn_block.mlp(norm2_act)
        branch2_act = s_mlp * g_attn_block.ls2(mlp_act)
        x_act_res2 = x_act_res1 + branch2_act

        if s_out is not None:
            x_act_res2 = s_out * x_act_res2

        x_next = torch.empty_like(x)
        for v in frozen_v:
            x_next[v] = frozen[v]
        for idx, v in enumerate(active_v):
            x_next[v] = x_act_res2[idx]
            if step_idx == target_k_stars[v]:
                frozen[v] = x_act_res2[idx].clone()

        x = x_next
    return x


def run_v08_evaluation(data_root="datasets/test/dtu", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Load V0.7 results to extract exact GT-derived dense K*
    v07_json_path = os.path.join(output_dir, "v07_raw_results.json")
    assert os.path.exists(v07_json_path), f"Cannot find {v07_json_path}. Run V0.7 first."

    print("Loading V0.7 dense oracle K* targets...")
    with open(v07_json_path, "r") as f:
        v07_data = json.load(f)

    # Build oracle lookup dictionary: (scene, subset, view_idx) -> {2pct, 5pct}
    oracle_k_lookup = {}
    for r in v07_data["trajectory_records"]:
        if r["step_i"] == 16:
            key = (r["scene"], r["subset"], r["view_idx"])
            oracle_k_lookup[key] = {
                "2pct": int(r["oracle_k_star_2pct"]),
                "5pct": int(r["oracle_k_star_5pct"]),
                "frame_id": int(r["frame_id"]),
            }

    scans = v07_data["metadata"]["scans"]
    subsets_def = [
        {"name": "subset_middle", "ranking": "middle_first", "sampling": "first"},
        {"name": "subset_uniform", "ranking": "index", "sampling": "uniform"},
    ]

    print("=" * 80)
    print("ViewHalt V0.8: True Asynchronous Freeze & Multi-Regime Evaluation")
    print(f"Scans ({len(scans)}): {scans}")
    print(f"Total sequences: {len(scans) * len(subsets_def)} (168 view instances)")
    print("=" * 80)

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
        "load_data_fields": [
            "images", "extrinsics_c2w", "intrinsics", "depths", "world_points", "point_masks"
        ],
    }

    all_records = []
    max_c_vs_d_diffs = []

    seq_idx = 0
    total_seqs = len(scans) * len(subsets_def)

    for scan_name in scans:
        for sub in subsets_def:
            seq_idx += 1
            seq_id = f"{scan_name}_{sub['name']}"
            print(f"\n[{seq_idx}/{total_seqs}] Processing: {seq_id}")

            cfg = OmegaConf.create({
                "target": "dtu.DTU",
                "params": {"root_path": data_root}
            })
            ds = DataverseEvalDataset(
                dataverse_cfg=cfg,
                view_ranking=sub["ranking"],
                view_sampling=sub["sampling"],
                max_frames=6,
            )
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
            dtu_frame_ids = batch[DataField.IDS][0].cpu().tolist()

            target_k_2pct = [oracle_k_lookup[(scan_name, sub["name"], v)]["2pct"] for v in range(S)]
            target_k_5pct = [oracle_k_lookup[(scan_name, sub["name"], v)]["5pct"] for v in range(S)]

            with torch.no_grad(), accelerator.autocast():
                z_0 = inner._encode_images(images)
                register_token = inner.register_token.expand(B, S, -1, -1).reshape(B * S, inner.num_register_tokens, -1)
                camera_token = _slice_expand_flatten(inner.camera_token, B, S)
                x_init = torch.cat([camera_token, register_token, z_0], dim=1)
                rope_pos = inner._get_rope_positions(B * S, H, W, images.device)

                K = 16
                ts = torch.linspace(0.0, 1.0, K).tolist()

                # -------------------------------------------------------------
                # Regime A: Standard Full K=16 Trajectory
                # Also save intermediate checkpoints for Fixed i=15, 14, 12, 8
                # -------------------------------------------------------------
                x_curr = x_init.clone()
                fixed_checkpoints = {}
                for i in range(K):
                    t_now = ts[i]
                    t_next = ts[i + 1] if i + 1 < K else 1.0
                    x_curr = inner._interval_step(x_curr, t_now, t_next, rope_pos, B, S)
                    step_num = i + 1
                    if step_num in [8, 12, 14, 15, 16]:
                        fixed_checkpoints[step_num] = x_curr.clone()

                # Decode Fixed regimes
                fixed_metrics = {}
                for step_num in [8, 12, 14, 15, 16]:
                    preds_fixed = model._postprocess_predictions(
                        batch, inner._decode(fixed_checkpoints[step_num], H, W, B, S, rope_pos)
                    )
                    fixed_metrics[step_num] = compute_metrics_for_predictions(preds_fixed, batch, device)

                # -------------------------------------------------------------
                # Tolerances: 2% and 5%
                # Compare Regime B (V0.7 overwrite), Regime C (Ref static), Regime D (Sparse)
                # -------------------------------------------------------------
                tol_configs = [
                    ("2pct", 0.02, target_k_2pct),
                    ("5pct", 0.05, target_k_5pct),
                ]

                for tol_str, tol_val, k_stars in tol_configs:
                    # 1. Regime B: Post-block overwrite
                    x_b = run_regime_b_post_block_overwrite(inner, x_init, k_stars, rope_pos, B, S, K, ts)
                    preds_b = model._postprocess_predictions(batch, inner._decode(x_b, H, W, B, S, rope_pos))
                    metrics_b = compute_metrics_for_predictions(preds_b, batch, device)

                    # 2. Regime C: Reference True-Static
                    x_c = run_regime_c_true_static_reference(inner, x_init, k_stars, rope_pos, B, S, K, ts, device)
                    preds_c = model._postprocess_predictions(batch, inner._decode(x_c, H, W, B, S, rope_pos))
                    metrics_c = compute_metrics_for_predictions(preds_c, batch, device)

                    # 3. Regime D: Sparse Active-Q / Frozen-KV
                    x_d = run_regime_d_sparse_active_q_frozen_kv(inner, x_init, k_stars, rope_pos, B, S, K, ts, device)
                    preds_d = model._postprocess_predictions(batch, inner._decode(x_d, H, W, B, S, rope_pos))
                    metrics_d = compute_metrics_for_predictions(preds_d, batch, device)

                    # Numerical equivalence verification between C and D
                    diff_cd_x = (x_c - x_d).abs().max().item()
                    diff_cd_depth = (preds_c[PredictionField.DEPTHS][0] - preds_d[PredictionField.DEPTHS][0]).abs().max().item()
                    max_c_vs_d_diffs.append(max(diff_cd_x, diff_cd_depth))

                    # Difference between B and C
                    diff_bc_x = (x_b - x_c).abs().max().item()
                    diff_bc_depth = (preds_b[PredictionField.DEPTHS][0] - preds_c[PredictionField.DEPTHS][0]).abs().max().item()

                    for v in range(S):
                        is_active_at_end = (k_stars[v] == 16)
                        m_fixed16 = fixed_metrics[16][v]
                        m_b = metrics_b[v]
                        m_c = metrics_c[v]
                        m_d = metrics_d[v]

                        # Active-view degradation (for views active at end vs Fixed 16)
                        active_deg_b = ((m_b["abs_rel"] - m_fixed16["abs_rel"]) / m_fixed16["abs_rel"]) * 100.0
                        active_deg_c = ((m_c["abs_rel"] - m_fixed16["abs_rel"]) / m_fixed16["abs_rel"]) * 100.0

                        rec = {
                            "scene": scan_name,
                            "subset": sub["name"],
                            "view_idx": v,
                            "frame_id": int(dtu_frame_ids[v]),
                            "tolerance": tol_str,
                            "k_star": k_stars[v],
                            "is_active_at_end": int(is_active_at_end),
                            # Fixed K=16 baseline
                            "fixed16_abs_rel": m_fixed16["abs_rel"],
                            "fixed16_rmse": m_fixed16["rmse"],
                            "fixed16_rot_err": m_fixed16["rot_err_deg"],
                            "fixed16_trans_err": m_fixed16["trans_err_deg"],
                            # Regime B (V0.7 overwrite)
                            "regime_b_abs_rel": m_b["abs_rel"],
                            "regime_b_rmse": m_b["rmse"],
                            "regime_b_rot_err": m_b["rot_err_deg"],
                            "regime_b_trans_err": m_b["trans_err_deg"],
                            "regime_b_active_deg_pct": active_deg_b,
                            # Regime C (True Static Reference)
                            "regime_c_abs_rel": m_c["abs_rel"],
                            "regime_c_rmse": m_c["rmse"],
                            "regime_c_rot_err": m_c["rot_err_deg"],
                            "regime_c_trans_err": m_c["trans_err_deg"],
                            "regime_c_active_deg_pct": active_deg_c,
                            # Regime D (Sparse Active-Q / Frozen-KV)
                            "regime_d_abs_rel": m_d["abs_rel"],
                            "regime_d_rmse": m_d["rmse"],
                            "regime_d_rot_err": m_d["rot_err_deg"],
                            "regime_d_trans_err": m_d["trans_err_deg"],
                            # Differences
                            "diff_b_vs_c_abs_rel": m_c["abs_rel"] - m_b["abs_rel"],
                            "diff_c_vs_d_max_depth": diff_cd_depth,
                            # Fixed 15, 14 baselines for matched comparison
                            "fixed15_abs_rel": fixed_metrics[15][v]["abs_rel"],
                            "fixed14_abs_rel": fixed_metrics[14][v]["abs_rel"],
                            "fixed12_abs_rel": fixed_metrics[12][v]["abs_rel"],
                            "fixed8_abs_rel": fixed_metrics[8][v]["abs_rel"],
                        }
                        all_records.append(rec)

    print("\n" + "=" * 80)
    print("VERIFICATION OF REGIME C vs REGIME D NUMERICAL EQUIVALENCE:")
    print(f"Max C vs D absolute difference across ALL sequences: {max(max_c_vs_d_diffs):.6e}")
    if max(max_c_vs_d_diffs) < 1e-4:
        print("PASS: Regime D (Sparse Active-Q / Frozen-KV) matches Regime C (True Static) to bf16 precision!")
    else:
        print("WARNING: Difference exceeds expected numerical precision.")
    print("=" * 80)

    # Save CSV and JSON
    csv_path = os.path.join(output_dir, "v08_regimes_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    print(f"Saved: {csv_path}")

    json_path = os.path.join(output_dir, "v08_regimes_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "version": "V0.8",
                "scans": scans,
                "subsets": [s["name"] for s in subsets_def],
                "num_records": len(all_records),
                "max_c_vs_d_diff": max(max_c_vs_d_diffs),
            },
            "records": all_records,
        }, f, indent=2)
    print(f"Saved: {json_path}")
    return all_records


if __name__ == "__main__":
    run_v08_evaluation()
