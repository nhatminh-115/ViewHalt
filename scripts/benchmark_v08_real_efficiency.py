"""ViewHalt V0.8: Real Compute Efficiency & Wall-Clock Latency Benchmark.

Measures:
1. Recurrent-block analytical FLOPs across full sequences for each policy.
2. GPU wall-clock latency (median, IQR, std) using torch.cuda.Event.
3. Peak VRAM allocation for each regime.
4. Controller-free oracle execution overhead.
5. Real speedup and latency savings versus:
   - Fixed K=16
   - Fixed i=15 (matched quality for Tol 5%)
   - Fixed i=14
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

from dvlt.common.constants import DataField
from dvlt.data.collate import default_collate_fn
from dvlt.data.datasets.multi_source import MultiSourceDataset
from dvlt.data.datasets.parser.dataverse import DataverseEvalDataset
from dvlt.model.dvlt.model import DVLT, _slice_expand_flatten


def compute_analytical_recurrent_flops(S, P=1301, C=768, num_heads=12):
    """Computes analytical FLOPs for 1 step of LoopedAABlock with S active views."""
    # Frame block (per view):
    # - norm1: 2 P C
    # - qkv: 6 P C^2
    # - q/k norm: 2 P C
    # - rope: 2 P C
    # - attn: 2 P^2 C
    # - proj: 2 P C^2
    # - ls1 + res: 2 P C
    # - norm2: 2 P C
    # - mlp: 16 P C^2
    # - ls2 + res: 2 P C
    flops_frame_per_view = (
        2 * P * C + 6 * P * (C**2) + 2 * P * C + 2 * P * C
        + 2 * (P**2) * C + 2 * P * (C**2) + 2 * P * C
        + 2 * P * C + 16 * P * (C**2) + 2 * P * C
    )
    return flops_frame_per_view


def compute_sequence_analytical_flops(k_stars, total_K=16, S=6, P=1301, C=768):
    """Computes total recurrent FLOPs for a sequence under sparse execution."""
    flops_frame_per_view = (
        2 * P * C + 6 * P * (C**2) + 2 * P * C + 2 * P * C
        + 2 * (P**2) * C + 2 * P * (C**2) + 2 * P * C
        + 2 * P * C + 16 * P * (C**2) + 2 * P * C
    )
    
    total_flops = 0.0
    for step_i in range(1, total_K + 1):
        # Active views at step_i
        n_act = sum(1 for k in k_stars if step_i <= k)
        n_frz = S - n_act

        # 1. Frame block: only active views
        flops_frame = n_act * flops_frame_per_view

        # 2. Global block:
        # norm1: n_act * 2 P C
        # qkv: Q for n_act (2 n_act P C^2), K/V for n_act (4 n_act P C^2). (Cached frozen K/V: 0)
        # global attn: 2 * (n_act * P) * (S * P) * C
        # proj: 2 n_act P C^2
        # norm2 + mlp: n_act * (2 P C + 16 P C^2)
        flops_global = (
            n_act * (2 * P * C)
            + 6 * n_act * P * (C**2)
            + 2 * (n_act * P) * (S * P) * C
            + 2 * n_act * P * (C**2)
            + n_act * (2 * P * C + 16 * P * (C**2))
            + n_act * (2 * P * C) # ls + res
        )
        total_flops += (flops_frame + flops_global)
    return total_flops


def benchmark_v08_efficiency(data_root="datasets/test/dtu", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    v07_json_path = os.path.join(output_dir, "v07_raw_results.json")
    with open(v07_json_path, "r") as f:
        v07_data = json.load(f)

    oracle_k_lookup = {}
    for r in v07_data["trajectory_records"]:
        if r["step_i"] == 16:
            key = (r["scene"], r["subset"], r["view_idx"])
            oracle_k_lookup[key] = {
                "2pct": int(r["oracle_k_star_2pct"]),
                "5pct": int(r["oracle_k_star_5pct"]),
            }

    scans = v07_data["metadata"]["scans"]
    subsets_def = [
        {"name": "subset_middle", "ranking": "middle_first", "sampling": "first"},
        {"name": "subset_uniform", "ranking": "index", "sampling": "uniform"},
    ]

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    print("Loading pretrained DVLT model for latency benchmarking...")
    model = DVLT(img_size=504, depth_head_type="conv")
    model.load_pretrained("nvidia/dvlt", strict=True)
    model.setup_test(accelerator)
    inner = model.model
    inner.eval()
    block = inner.recurrent_blocks[0]
    g_attn_block = block.global_attn
    attn_mod = g_attn_block.attn
    num_heads = attn_mod.num_heads
    head_dim = attn_mod.head_dim

    test_config = {
        "normalize_scene": False,
        "load_data_fields": ["images", "extrinsics_c2w", "intrinsics", "depths", "world_points", "point_masks"],
    }

    # Policies to benchmark
    # 1. Fixed K=16
    # 2. Fixed i=15
    # 3. Fixed i=14
    # 4. Fixed i=12
    # 5. Fixed i=8
    # 6. Sparse Active-Q / Frozen-KV (Tol 5%)
    # 7. Sparse Active-Q / Frozen-KV (Tol 2%)
    # 8. Reference True-Static (Tol 5%)

    warmup_reps = 3
    timed_reps = 10

    # Select representative subset of sequences to benchmark thoroughly across scans
    # We will benchmark 7 scans x 2 subsets = 14 sequences with 10 repetitions each
    bench_scans = scans[::2] # 7 scans
    print(f"Benchmarking on {len(bench_scans)} scans x 2 subsets = {len(bench_scans)*2} sequences")
    print(f"Repetitions per condition: {warmup_reps} warmup + {timed_reps} timed runs")

    all_timing_records = []

    for scan_name in bench_scans:
        for sub in subsets_def:
            seq_id = f"{scan_name}_{sub['name']}"
            print(f"\nBenchmarking sequence: {seq_id}")

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

            k_stars_5pct = [oracle_k_lookup[(scan_name, sub["name"], v)]["5pct"] for v in range(S)]
            k_stars_2pct = [oracle_k_lookup[(scan_name, sub["name"], v)]["2pct"] for v in range(S)]

            with torch.no_grad(), accelerator.autocast():
                z_0 = inner._encode_images(images)
                register_token = inner.register_token.expand(B, S, -1, -1).reshape(B * S, inner.num_register_tokens, -1)
                camera_token = _slice_expand_flatten(inner.camera_token, B, S)
                x_init = torch.cat([camera_token, register_token, z_0], dim=1)
                rope_pos = inner._get_rope_positions(B * S, H, W, images.device)

                K = 16
                ts = torch.linspace(0.0, 1.0, K).tolist()
                P = x_init.shape[1]
                dim = x_init.shape[2]

                # Policy definitions
                policies = [
                    {"name": "Fixed_K16", "type": "fixed", "steps": 16},
                    {"name": "Fixed_i15", "type": "fixed", "steps": 15},
                    {"name": "Fixed_i14", "type": "fixed", "steps": 14},
                    {"name": "Fixed_i12", "type": "fixed", "steps": 12},
                    {"name": "Fixed_i8", "type": "fixed", "steps": 8},
                    {"name": "Sparse_Oracle_5pct", "type": "sparse", "k_stars": k_stars_5pct},
                    {"name": "Sparse_Oracle_2pct", "type": "sparse", "k_stars": k_stars_2pct},
                    {"name": "Ref_Static_5pct", "type": "ref_static", "k_stars": k_stars_5pct},
                ]

                for pol in policies:
                    pol_name = pol["name"]

                    # Compute analytical FLOPs
                    if pol["type"] == "fixed":
                        flops = compute_sequence_analytical_flops([pol["steps"]] * S, total_K=pol["steps"], S=S, P=P, C=dim)
                    else:
                        flops = compute_sequence_analytical_flops(pol["k_stars"], total_K=16, S=S, P=P, C=dim)

                    # Warmup
                    for _ in range(warmup_reps):
                        if pol["type"] == "fixed":
                            x_w = x_init.clone()
                            for i in range(pol["steps"]):
                                t_now = ts[i]
                                t_next = ts[i + 1] if i + 1 < K else 1.0
                                x_w = inner._interval_step(x_w, t_now, t_next, rope_pos, B, S)
                        elif pol["type"] == "sparse":
                            # Run sparse loop
                            x_w = x_init.clone()
                            frozen_w, frozen_kv_w = {}, {}
                            for i in range(K):
                                t_now = ts[i]
                                t_next = ts[i + 1] if i + 1 < K else 1.0
                                step_idx = i + 1
                                t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)
                                active_v = [v for v in range(S) if step_idx <= pol["k_stars"][v]]
                                frozen_v = [v for v in range(S) if step_idx > pol["k_stars"][v]]
                                x_frame_w = torch.empty_like(x_w)
                                for v in frozen_v:
                                    x_frame_w[v] = frozen_w[v]
                                if len(active_v) > 0:
                                    act_idx = torch.tensor(active_v, device=device)
                                    x_act = x_w[act_idx]
                                    pos_act = rope_pos.reshape(B * S, P, 2)[act_idx]
                                    t_act = t_pair.expand(len(active_v), -1)
                                    x_act_frame = block.frame_attn(x_act, t_act, pos=pos_act)
                                    for idx, v in enumerate(active_v):
                                        x_frame_w[v] = x_act_frame[idx]
                                if len(active_v) == 0:
                                    x_w = x_frame_w
                                    continue
                                s = g_attn_block.depth_scale(t_pair.expand(B, -1)).unsqueeze(1)
                                s_attn, s_mlp, s_out = s.chunk(3, dim=-1)
                                x_act_in = x_frame_w[active_v]
                                norm_act = g_attn_block.norm1(x_act_in)
                                qkv_act = attn_mod.qkv(norm_act).reshape(len(active_v), P, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                                q_act_views, k_act_views, v_act_views = qkv_act.unbind(0)
                                q_act_views = attn_mod.q_norm(q_act_views)
                                k_act_views = attn_mod.k_norm(k_act_views)
                                for v in frozen_v:
                                    if v not in frozen_kv_w:
                                        norm_v = g_attn_block.norm1(frozen_w[v].unsqueeze(0))
                                        qkv_v = attn_mod.qkv(norm_v).reshape(1, P, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                                        _, k_v, v_v = qkv_v.unbind(0)
                                        k_v = attn_mod.k_norm(k_v)
                                        frozen_kv_w[v] = (k_v.squeeze(0), v_v.squeeze(0))
                                k_list, v_list = [], []
                                for v in range(S):
                                    if v in frozen_v:
                                        k_list.append(frozen_kv_w[v][0])
                                        v_list.append(frozen_kv_w[v][1])
                                    else:
                                        act_local_idx = active_v.index(v)
                                        k_list.append(k_act_views[act_local_idx])
                                        v_list.append(v_act_views[act_local_idx])
                                full_k = torch.cat(k_list, dim=1).unsqueeze(0)
                                full_v = torch.cat(v_list, dim=1).unsqueeze(0)
                                q_act_cat = torch.cat([q_act_views[i] for i in range(len(active_v))], dim=1).unsqueeze(0)
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
                                x_next = torch.empty_like(x_w)
                                for v in frozen_v:
                                    x_next[v] = frozen_w[v]
                                for idx, v in enumerate(active_v):
                                    x_next[v] = x_act_res2[idx]
                                    if step_idx == pol["k_stars"][v]:
                                        frozen_w[v] = x_act_res2[idx].clone()
                                x_w = x_next
                        elif pol["type"] == "ref_static":
                            x_w = x_init.clone()
                            frozen_w = {}
                            for i in range(K):
                                t_now = ts[i]
                                t_next = ts[i + 1] if i + 1 < K else 1.0
                                step_idx = i + 1
                                t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)
                                x_frame = block.frame_attn(x_w, t_pair.expand(B * S, -1), pos=rope_pos.reshape(B * S, P, 2))
                                for v in range(S):
                                    if step_idx > pol["k_stars"][v]:
                                        x_frame[v] = frozen_w[v].clone()
                                x_global_in = x_frame.reshape(B, S * P, dim)
                                x_global_out = block.global_attn(x_global_in, t_pair.expand(B, -1), pos=None).reshape(B * S, P, dim)
                                for v in range(S):
                                    if step_idx == pol["k_stars"][v]:
                                        frozen_w[v] = x_global_out[v].clone()
                                    elif step_idx > pol["k_stars"][v]:
                                        x_global_out[v] = frozen_w[v].clone()
                                x_w = x_global_out

                    torch.cuda.synchronize()

                    # Timed repetitions
                    recurrent_times = []
                    torch.cuda.reset_peak_memory_stats(device)

                    for rep in range(timed_reps):
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)

                        start_event.record()
                        if pol["type"] == "fixed":
                            x_run = x_init.clone()
                            for i in range(pol["steps"]):
                                t_now = ts[i]
                                t_next = ts[i + 1] if i + 1 < K else 1.0
                                x_run = inner._interval_step(x_run, t_now, t_next, rope_pos, B, S)
                        elif pol["type"] == "sparse":
                            x_run = x_init.clone()
                            frozen_run, frozen_kv_run = {}, {}
                            for i in range(K):
                                t_now = ts[i]
                                t_next = ts[i + 1] if i + 1 < K else 1.0
                                step_idx = i + 1
                                t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)
                                active_v = [v for v in range(S) if step_idx <= pol["k_stars"][v]]
                                frozen_v = [v for v in range(S) if step_idx > pol["k_stars"][v]]
                                x_frame_run = torch.empty_like(x_run)
                                for v in frozen_v:
                                    x_frame_run[v] = frozen_run[v]
                                if len(active_v) > 0:
                                    act_idx = torch.tensor(active_v, device=device)
                                    x_act = x_run[act_idx]
                                    pos_act = rope_pos.reshape(B * S, P, 2)[act_idx]
                                    t_act = t_pair.expand(len(active_v), -1)
                                    x_act_frame = block.frame_attn(x_act, t_act, pos=pos_act)
                                    for idx, v in enumerate(active_v):
                                        x_frame_run[v] = x_act_frame[idx]
                                if len(active_v) == 0:
                                    x_run = x_frame_run
                                    continue
                                s = g_attn_block.depth_scale(t_pair.expand(B, -1)).unsqueeze(1)
                                s_attn, s_mlp, s_out = s.chunk(3, dim=-1)
                                x_act_in = x_frame_run[active_v]
                                norm_act = g_attn_block.norm1(x_act_in)
                                qkv_act = attn_mod.qkv(norm_act).reshape(len(active_v), P, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                                q_act_views, k_act_views, v_act_views = qkv_act.unbind(0)
                                q_act_views = attn_mod.q_norm(q_act_views)
                                k_act_views = attn_mod.k_norm(k_act_views)
                                for v in frozen_v:
                                    if v not in frozen_kv_run:
                                        norm_v = g_attn_block.norm1(frozen_run[v].unsqueeze(0))
                                        qkv_v = attn_mod.qkv(norm_v).reshape(1, P, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                                        _, k_v, v_v = qkv_v.unbind(0)
                                        k_v = attn_mod.k_norm(k_v)
                                        frozen_kv_run[v] = (k_v.squeeze(0), v_v.squeeze(0))
                                k_list, v_list = [], []
                                for v in range(S):
                                    if v in frozen_v:
                                        k_list.append(frozen_kv_run[v][0])
                                        v_list.append(frozen_kv_run[v][1])
                                    else:
                                        act_local_idx = active_v.index(v)
                                        k_list.append(k_act_views[act_local_idx])
                                        v_list.append(v_act_views[act_local_idx])
                                full_k = torch.cat(k_list, dim=1).unsqueeze(0)
                                full_v = torch.cat(v_list, dim=1).unsqueeze(0)
                                q_act_cat = torch.cat([q_act_views[i] for i in range(len(active_v))], dim=1).unsqueeze(0)
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
                                x_next = torch.empty_like(x_run)
                                for v in frozen_v:
                                    x_next[v] = frozen_run[v]
                                for idx, v in enumerate(active_v):
                                    x_next[v] = x_act_res2[idx]
                                    if step_idx == pol["k_stars"][v]:
                                        frozen_run[v] = x_act_res2[idx].clone()
                                x_run = x_next
                        elif pol["type"] == "ref_static":
                            x_run = x_init.clone()
                            frozen_run = {}
                            for i in range(K):
                                t_now = ts[i]
                                t_next = ts[i + 1] if i + 1 < K else 1.0
                                step_idx = i + 1
                                t_pair = torch.tensor([[t_now, t_next]], device=device, dtype=torch.float32)
                                x_frame = block.frame_attn(x_run, t_pair.expand(B * S, -1), pos=rope_pos.reshape(B * S, P, 2))
                                for v in range(S):
                                    if step_idx > pol["k_stars"][v]:
                                        x_frame[v] = frozen_run[v].clone()
                                x_global_in = x_frame.reshape(B, S * P, dim)
                                x_global_out = block.global_attn(x_global_in, t_pair.expand(B, -1), pos=None).reshape(B * S, P, dim)
                                for v in range(S):
                                    if step_idx == pol["k_stars"][v]:
                                        frozen_run[v] = x_global_out[v].clone()
                                    elif step_idx > pol["k_stars"][v]:
                                        x_global_out[v] = frozen_run[v].clone()
                                x_run = x_global_out

                        end_event.record()
                        torch.cuda.synchronize()
                        recurrent_times.append(start_event.elapsed_time(end_event))

                    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

                    recurrent_times = np.array(recurrent_times)
                    med_ms = float(np.median(recurrent_times))
                    iqr_ms = float(np.percentile(recurrent_times, 75) - np.percentile(recurrent_times, 25))
                    mean_ms = float(np.mean(recurrent_times))
                    std_ms = float(np.std(recurrent_times))

                    rec = {
                        "scene": scan_name,
                        "subset": sub["name"],
                        "policy": pol_name,
                        "policy_type": pol["type"],
                        "recurrent_flops_gflops": round(flops / 1e9, 2),
                        "latency_median_ms": round(med_ms, 2),
                        "latency_iqr_ms": round(iqr_ms, 2),
                        "latency_mean_ms": round(mean_ms, 2),
                        "latency_std_ms": round(std_ms, 2),
                        "peak_vram_mb": round(peak_vram_mb, 1),
                    }
                    all_timing_records.append(rec)
                    print(f"  {pol_name:<20}: FLOPs={flops/1e9:6.1f} GF | Latency={med_ms:5.1f} ms (IQR={iqr_ms:4.1f}) | VRAM={peak_vram_mb:5.0f} MB")

    csv_path = os.path.join(output_dir, "v08_timing_benchmark_raw.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_timing_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_timing_records)
    print(f"\nSaved raw timing benchmark: {csv_path}")

    # Compute aggregate summary across all benchmarked sequences
    policies_set = sorted(list(set(r["policy"] for r in all_timing_records)))
    summary = {}
    
    # Baseline K=16 per sequence lookup
    k16_lookup = {(r["scene"], r["subset"]): r["latency_median_ms"] for r in all_timing_records if r["policy"] == "Fixed_K16"}
    k16_flops_lookup = {(r["scene"], r["subset"]): r["recurrent_flops_gflops"] for r in all_timing_records if r["policy"] == "Fixed_K16"}
    i15_lookup = {(r["scene"], r["subset"]): r["latency_median_ms"] for r in all_timing_records if r["policy"] == "Fixed_i15"}
    i14_lookup = {(r["scene"], r["subset"]): r["latency_median_ms"] for r in all_timing_records if r["policy"] == "Fixed_i14"}

    for pol in policies_set:
        pol_recs = [r for r in all_timing_records if r["policy"] == pol]
        latencies = [r["latency_median_ms"] for r in pol_recs]
        flops_list = [r["recurrent_flops_gflops"] for r in pol_recs]
        vram_list = [r["peak_vram_mb"] for r in pol_recs]

        # Latency savings vs K16
        savings_vs_k16 = [
            ((k16_lookup[(r["scene"], r["subset"])] - r["latency_median_ms"]) / k16_lookup[(r["scene"], r["subset"])]) * 100.0
            for r in pol_recs
        ]
        # FLOP savings vs K16
        flop_savings_vs_k16 = [
            ((k16_flops_lookup[(r["scene"], r["subset"])] - r["recurrent_flops_gflops"]) / k16_flops_lookup[(r["scene"], r["subset"])]) * 100.0
            for r in pol_recs
        ]
        # Speedups
        speedup_vs_k16 = [k16_lookup[(r["scene"], r["subset"])] / r["latency_median_ms"] for r in pol_recs]
        speedup_vs_i15 = [i15_lookup[(r["scene"], r["subset"])] / r["latency_median_ms"] for r in pol_recs]
        speedup_vs_i14 = [i14_lookup[(r["scene"], r["subset"])] / r["latency_median_ms"] for r in pol_recs]

        # Matched quality saving (vs i15)
        savings_vs_i15 = [
            ((i15_lookup[(r["scene"], r["subset"])] - r["latency_median_ms"]) / i15_lookup[(r["scene"], r["subset"])]) * 100.0
            for r in pol_recs
        ]

        summary[pol] = {
            "mean_recurrent_flops_gflops": round(float(np.mean(flops_list)), 2),
            "mean_flop_savings_pct": round(float(np.mean(flop_savings_vs_k16)), 2),
            "median_latency_ms": round(float(np.median(latencies)), 2),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "iqr_latency_ms": round(float(np.percentile(latencies, 75) - np.percentile(latencies, 25)), 2),
            "mean_latency_savings_pct": round(float(np.mean(savings_vs_k16)), 2),
            "mean_speedup_vs_k16": round(float(np.mean(speedup_vs_k16)), 3),
            "mean_speedup_vs_i15": round(float(np.mean(speedup_vs_i15)), 3),
            "mean_speedup_vs_i14": round(float(np.mean(speedup_vs_i14)), 3),
            "matched_latency_savings_vs_i15_pct": round(float(np.mean(savings_vs_i15)), 2),
            "mean_peak_vram_mb": round(float(np.mean(vram_list)), 1),
        }

    json_path = os.path.join(output_dir, "v08_timing_benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved benchmark summary: {json_path}")
    return summary


if __name__ == "__main__":
    benchmark_v08_efficiency()
