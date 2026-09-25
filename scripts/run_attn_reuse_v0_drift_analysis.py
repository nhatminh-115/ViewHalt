"""DVLT Attention-Reuse V0: Trajectory Drift Analysis.

Instruments the K=16 recurrent trajectory on 5 representative DTU scans:
Measures across consecutive iterations (especially steps 8-16):
- hidden-state cosine similarity and L2 drift
- global-attention input drift (x_frame_i vs x_frame_{i-1})
- global-attention output drift (x_global_i vs x_global_{i-1})
- global-attention residual drift (r_i vs r_{i-1} where r = x_global - x_frame)
- global-attention residual magnitude ratio (||r_i|| / ||x_frame_i||)
- global K/V drift and cosine similarity
"""

import csv
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf

from dvlt.common.constants import DataField
from dvlt.data.collate import default_collate_fn
from dvlt.data.datasets.multi_source import MultiSourceDataset
from dvlt.data.datasets.parser.dataverse import DataverseEvalDataset
from dvlt.model.dvlt.model import DVLT, _slice_expand_flatten


def run_drift_analysis(data_root="datasets/test/dtu", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    scans = ["scan1", "scan4", "scan9", "scan24", "scan62"]
    subsets_def = [
        {"name": "subset_middle", "ranking": "middle_first", "sampling": "first"},
        {"name": "subset_uniform", "ranking": "index", "sampling": "uniform"},
    ]

    print("=" * 80)
    print("DVLT Attention-Reuse V0: Recurrent Block Drift Analysis")
    print(f"Scans (5): {scans}")
    print(f"Subsets: {[s['name'] for s in subsets_def]} (10 sequences, 60 views)")
    print("=" * 80)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    print("Loading pretrained DVLT model...")
    model = DVLT(img_size=504, depth_head_type="conv")
    model.load_pretrained("nvidia/dvlt", strict=True)
    model.setup_test(accelerator)
    inner = model.model
    inner.eval()
    block = inner.recurrent_blocks[0]
    g_attn_block = block.global_attn
    attn_mod = g_attn_block.attn

    test_config = {
        "normalize_scene": False,
        "load_data_fields": ["images", "extrinsics_c2w", "intrinsics", "depths", "world_points", "point_masks"],
    }

    all_step_drifts = []

    for scan_name in scans:
        for sub in subsets_def:
            seq_id = f"{scan_name}_{sub['name']}"
            print(f"Processing sequence: {seq_id}")

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
                x_curr = torch.cat([camera_token, register_token, z_0], dim=1)
                rope_pos = inner._get_rope_positions(B * S, H, W, images.device)

                K = 16
                ts = torch.linspace(0.0, 1.0, K).tolist()
                P = x_curr.shape[1]
                dim = x_curr.shape[2]

                # Storage per step i
                history = {}

                for i in range(K):
                    step_i = i + 1
                    t_now = ts[i]
                    t_next = ts[i + 1] if i + 1 < K else 1.0
                    t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)

                    # 1. Frame attention
                    x_frame = block.frame_attn(x_curr, t_pair.expand(B * S, -1), pos=rope_pos.reshape(B * S, P, 2))

                    # 2. Global attention internals
                    x_global_in = x_frame.reshape(B, S * P, dim)

                    # Extract Q, K, V
                    norm1_out = g_attn_block.norm1(x_global_in)
                    qkv = attn_mod.qkv(norm1_out).reshape(B, S * P, 3, attn_mod.num_heads, attn_mod.head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    q = attn_mod.q_norm(q)
                    k = attn_mod.k_norm(k)

                    # Run global attention
                    x_global_out = g_attn_block(x_global_in, t_pair.expand(B, -1), pos=None)
                    r_i = x_global_out - x_global_in

                    # Next input
                    x_next = x_global_out.reshape(B * S, P, dim)

                    history[step_i] = {
                        "h": x_next.clone(),
                        "x_frame": x_global_in.clone(),
                        "x_global_out": x_global_out.clone(),
                        "r": r_i.clone(),
                        "k": k.clone(),
                        "v": v.clone(),
                    }
                    x_curr = x_next

                # Compute consecutive drifts
                for step_i in range(2, K + 1):
                    prev = history[step_i - 1]
                    curr = history[step_i]

                    # 1. Hidden state drift
                    h_p = prev["h"].reshape(-1, dim)
                    h_c = curr["h"].reshape(-1, dim)
                    cos_h = F.cosine_similarity(h_p, h_c, dim=-1).mean().item()
                    diff_h = (h_c - h_p).norm(dim=-1).mean().item()
                    norm_h = h_c.norm(dim=-1).mean().item()
                    rel_drift_h = diff_h / max(norm_h, 1e-6)

                    # 2. Global input drift (x_frame)
                    gf_p = prev["x_frame"].reshape(-1, dim)
                    gf_c = curr["x_frame"].reshape(-1, dim)
                    cos_gf = F.cosine_similarity(gf_p, gf_c, dim=-1).mean().item()
                    diff_gf = (gf_c - gf_p).norm(dim=-1).mean().item()
                    norm_gf = gf_c.norm(dim=-1).mean().item()
                    rel_drift_gf = diff_gf / max(norm_gf, 1e-6)

                    # 3. Global output drift
                    go_p = prev["x_global_out"].reshape(-1, dim)
                    go_c = curr["x_global_out"].reshape(-1, dim)
                    cos_go = F.cosine_similarity(go_p, go_c, dim=-1).mean().item()
                    diff_go = (go_c - go_p).norm(dim=-1).mean().item()
                    norm_go = go_c.norm(dim=-1).mean().item()
                    rel_drift_go = diff_go / max(norm_go, 1e-6)

                    # 4. Global residual drift & magnitude
                    r_p = prev["r"].reshape(-1, dim)
                    r_c = curr["r"].reshape(-1, dim)
                    cos_r = F.cosine_similarity(r_p, r_c, dim=-1).mean().item()
                    diff_r = (r_c - r_p).norm(dim=-1).mean().item()
                    norm_r = r_c.norm(dim=-1).mean().item()
                    rel_drift_r = diff_r / max(norm_r, 1e-6)
                    ratio_r_to_frame = norm_r / max(norm_gf, 1e-6)

                    # 5. K and V drift
                    k_p = prev["k"].reshape(-1, attn_mod.head_dim)
                    k_c = curr["k"].reshape(-1, attn_mod.head_dim)
                    cos_k = F.cosine_similarity(k_p, k_c, dim=-1).mean().item()
                    diff_k = (k_c - k_p).norm(dim=-1).mean().item()
                    norm_k = k_c.norm(dim=-1).mean().item()
                    rel_drift_k = diff_k / max(norm_k, 1e-6)

                    v_p = prev["v"].reshape(-1, attn_mod.head_dim)
                    v_c = curr["v"].reshape(-1, attn_mod.head_dim)
                    cos_v = F.cosine_similarity(v_p, v_c, dim=-1).mean().item()
                    diff_v = (v_c - v_p).norm(dim=-1).mean().item()
                    norm_v = v_c.norm(dim=-1).mean().item()
                    rel_drift_v = diff_v / max(norm_v, 1e-6)

                    rec = {
                        "scene": scan_name,
                        "subset": sub["name"],
                        "step_i": step_i,
                        "cos_h": cos_h,
                        "rel_drift_h": rel_drift_h,
                        "cos_global_in": cos_gf,
                        "rel_drift_global_in": rel_drift_gf,
                        "cos_global_out": cos_go,
                        "rel_drift_global_out": rel_drift_go,
                        "cos_residual": cos_r,
                        "rel_drift_residual": rel_drift_r,
                        "residual_ratio": ratio_r_to_frame,
                        "cos_k": cos_k,
                        "rel_drift_k": rel_drift_k,
                        "cos_v": cos_v,
                        "rel_drift_v": rel_drift_v,
                    }
                    all_step_drifts.append(rec)

    # Save CSV
    csv_path = os.path.join(output_dir, "attn_reuse_v0_drift_raw.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_step_drifts[0].keys()))
        writer.writeheader()
        writer.writerows(all_step_drifts)
    print(f"\nSaved raw drift metrics: {csv_path}")

    # Summary table by step i
    df = pd.DataFrame(all_step_drifts)
    summary_by_step = df.groupby("step_i").mean(numeric_only=True).reset_index()

    print("\n" + "=" * 95)
    print(f"{'Step i':<8} | {'Cos(h)':<8} | {'Drift(h)':<9} | {'Cos(G_in)':<9} | {'Cos(G_out)':<10} | {'Cos(r)':<8} | {'||r||/||x||':<11} | {'Cos(K)':<8} | {'Cos(V)':<8}")
    print("-" * 95)
    for _, r in summary_by_step.iterrows():
        s = int(r["step_i"])
        print(f"{s:<8} | {r['cos_h']:<8.4f} | {r['rel_drift_h']:<9.4f} | {r['cos_global_in']:<9.4f} | {r['cos_global_out']:<10.4f} | {r['cos_residual']:<8.4f} | {r['residual_ratio']:<11.4f} | {r['cos_k']:<8.4f} | {r['cos_v']:<8.4f}")
    print("=" * 95)

    summary_json_path = os.path.join(output_dir, "attn_reuse_v0_drift_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_by_step.to_dict(orient="records"), f, indent=2)
    print(f"Saved drift summary: {summary_json_path}")
    return summary_by_step


if __name__ == "__main__":
    run_drift_analysis()
