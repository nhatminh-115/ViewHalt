"""Smoke test script for DVLT inference on RTX 5070 Laptop GPU (8GB VRAM).

Tests feasibility under V0 constraints:
- Input resolution: 504x504
- Sequence lengths: S ∈ {2, 4}
- Recurrent step counts: K ∈ {8, 12, 16}
- Precision: bf16 (via Accelerator / autocast)
- Measures peak VRAM and wall-clock execution time
"""

import json
import os
import time
import torch
from accelerate import Accelerator

from dvlt.common.constants import DataField, PredictionField
from dvlt.model.dvlt.model import DVLT


def run_smoke_test():
    print("=" * 60)
    print("Starting DVLT V0 Feasibility Smoke Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required.")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Device: {gpu_name} (Total VRAM: {total_vram_gb:.2f} GB)")

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    print("Instantiating DVLT model (img_size=504, depth_head_type='conv')...")
    model = DVLT(img_size=504, depth_head_type="conv")
    print("Loading pretrained weights from nvidia/dvlt...")
    model.load_pretrained("nvidia/dvlt", strict=True)
    model.setup_test(accelerator)

    results = {
        "gpu": gpu_name,
        "total_vram_gb": total_vram_gb,
        "resolution": [504, 504],
        "runs": []
    }

    # Warmup with dummy batch (2 views, K=8)
    print("\nWarming up model (2 views, K=8)...")
    warmup_images = torch.rand(1, 2, 3, 504, 504, device=device)
    model.inference_steps = 8
    model.model.inference_steps = 8
    with torch.no_grad(), accelerator.autocast():
        _ = model.predict({DataField.IMAGES: warmup_images}, accelerator)
    torch.cuda.synchronize()
    print("Warmup complete.")

    view_counts = [2, 4]
    step_counts = [8, 12, 16]

    for S in view_counts:
        print(f"\n--- Testing Sequence Length S={S} views ---")
        dummy_images = torch.rand(1, S, 3, 504, 504, device=device)
        batch = {DataField.IMAGES: dummy_images}

        for K in step_counts:
            # Set inference steps
            model.inference_steps = K
            model.model.inference_steps = K

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            start_time = time.perf_counter()
            with torch.no_grad(), accelerator.autocast():
                predictions = model.predict(batch, accelerator)
            torch.cuda.synchronize()
            elapsed_sec = time.perf_counter() - start_time

            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_vram_gb = peak_vram_mb / 1024

            cameras = predictions[PredictionField.CAMERAS][0]
            depths = predictions[PredictionField.DEPTHS][0]
            world_pts = predictions[PredictionField.WORLD_POINTS][0]

            print(
                f"[S={S} views, K={K:2d} steps] "
                f"Peak VRAM: {peak_vram_mb:.1f} MiB ({peak_vram_gb:.2f} GB) | "
                f"Wall-clock: {elapsed_sec * 1000:.1f} ms | "
                f"Depths: {tuple(depths.shape)} | Points: {tuple(world_pts.shape)}"
            )

            results["runs"].append({
                "views": S,
                "steps_K": K,
                "peak_vram_mb": round(peak_vram_mb, 2),
                "peak_vram_gb": round(peak_vram_gb, 3),
                "elapsed_sec": round(elapsed_sec, 4),
                "depths_shape": list(depths.shape),
                "world_points_shape": list(world_pts.shape),
                "within_8gb": peak_vram_gb < 8.0
            })

    os.makedirs("outputs", exist_ok=True)
    out_file = "outputs/smoke_test_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    all_passed = all(r["within_8gb"] for r in results["runs"])
    print(f"Smoke test summary: All configurations within 8 GB VRAM? {all_passed}")
    print(f"Results saved to {out_file}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run_smoke_test()
