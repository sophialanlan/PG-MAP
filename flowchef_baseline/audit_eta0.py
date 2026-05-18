"""Bitwise audit: at eta_z=0 the FlowChef pipeline must reproduce flow_baseline.

The FlowChef refine_step at eta_z=0 returns z_var = z_iter (no update applied).
The outer sampler is the same _sample_with_callback used by generate_sd3_baseline.
Therefore the FlowChef output at eta_z=0 must match the existing flow_baseline
images at 0/255 max abs pixel deviation.

Runs on n=20 PartiPrompts (same prompts the existing PG-MAP baseline used) and
reports per-image max abs pixel deviation.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
MAP_PGAC = HERE.parent
sys.path.insert(0, str(MAP_PGAC))
sys.path.insert(0, str(HERE))

from pgmap_eval import load_parti_prompts
from pgmap_flow_sd3 import load_sd3_models
from flowchef_sd3 import generate_sd3_flowchef


def main():
    n = 20
    seed = 123
    # CRITICAL: load_parti_prompts(seed, k) is NOT prefix-stable across k.
    # The existing baseline at flow_pgmap_sd35_n1632_seed123 was generated with
    # load_parti_prompts(123, 1632); we must use the same k=1632 list and
    # truncate, otherwise prompts at the same i differ.
    prompts_full = load_parti_prompts(seed, 1632)
    prompts = prompts_full[:n]
    seeds = [seed + i for i in range(len(prompts))]
    print(f"Audit on {len(prompts)} prompts (seed={seed})")

    print("Loading SD3.5-medium ...")
    models = load_sd3_models(
        "stabilityai/stable-diffusion-3.5-medium",
        device="cuda", dtype=torch.float16,
    )
    from pgmap_reward import FrozenRewardModel
    print("Loading PickScore (only for FlowChef API; eta=0 means it's never invoked) ...")
    reward_model = FrozenRewardModel("pickscore", device="cuda")

    baseline_dir = Path(
        os.environ.get(
            "PG_MAP_SHARED_FM_BASELINE",
            str(Path(__file__).resolve().parent.parent / "eval_results" / "table2_fm" / "baseline" / "images"),
        )
    )
    out_dir = HERE / "audit_eta0_outputs"
    out_dir.mkdir(exist_ok=True)
    print(f"Reference baseline: {baseline_dir}")

    diffs = []
    for i, (p, s) in enumerate(tqdm(list(zip(prompts, seeds)))):
        # Only need to test always-on (most exposed gating)
        img_test = generate_sd3_flowchef(
            p, neg_prompt="blurry, low quality", models=models,
            reward_model=reward_model,
            height=1024, width=1024,
            num_steps=28, cfg_scale=7.0, seed=s,
            K=1, eta_z=0.0, rho_Q=0.3, gate_side="alwayson",
        )
        ref_path = baseline_dir / f"{i:05d}.png"
        if not ref_path.exists():
            print(f"[warn] no baseline at {ref_path}, skipping")
            continue
        ref = Image.open(ref_path).convert("RGB")
        diff = np.abs(np.array(img_test, dtype=np.int32) - np.array(ref, dtype=np.int32))
        max_abs = int(diff.max())
        mean_abs = float(diff.mean())
        diffs.append(max_abs)
        if max_abs > 0:
            print(f"  [{i:02d}] max_abs={max_abs}/255  mean_abs={mean_abs:.4f}  prompt={p[:60]!r}")
            img_test.save(out_dir / f"test_{i:05d}.png")

    print()
    print("=" * 78)
    print(f"Audit summary on n={len(diffs)}")
    print(f"  max_abs_pixel_deviation_max  = {max(diffs) if diffs else 'N/A'}")
    print(f"  max_abs_pixel_deviation_mean = {np.mean(diffs) if diffs else 'N/A':.3f}")
    print(f"  perfect (0/255) count        = {sum(1 for d in diffs if d == 0)}/{len(diffs)}")
    print("=" * 78)
    if diffs and max(diffs) == 0:
        print("✅ BITWISE-AUDIT PASS: FlowChef at eta_z=0 == baseline")
    elif diffs and max(diffs) <= 1:
        print("✅ NEAR-AUDIT PASS (≤1/255 = quantization noise) — acceptable")
    else:
        print("❌ AUDIT FAIL — investigate before running the headline n=1632")


if __name__ == "__main__":
    main()
