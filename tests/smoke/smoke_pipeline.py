#!/usr/bin/env python3
"""Smoke test for the three PG-MAP diffusers pipeline subclasses.

Verifies, on a single GPU:

  1. ``PGMAPStableDiffusionPipeline``       — class is loadable, inherits from
     ``StableDiffusionPipeline``, vanilla pass-through generates a 512x512 PIL
     image, MAP-c PG-MAP path generates a 512x512 PIL image.
  2. ``PGMAPStableDiffusionXLPipeline``     — class loads + vanilla pass-through
     produces a 1024x1024 PIL image (PG-MAP path tested by reproduce_table1_sdxl.sh).
  3. ``PGMAPStableDiffusion3Pipeline``      — class hierarchy only (loading SD3.5
     requires accepting the Stability AI Community License).

Usage::

    python tests/smoke/smoke_pipeline.py [--skip-sdxl] [--skip-sd3]

Exit code 0 on success; nonzero with a diagnostic on any failure.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def fail(msg: str) -> None:
    print(f"[PIPELINE SMOKE] FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def smoke_sd15():
    """Test PGMAPStableDiffusionPipeline."""
    import torch
    from dataclasses import replace
    from diffusers import StableDiffusionPipeline

    from pgmap.pipelines import PGMAPStableDiffusionPipeline
    from pgmap_config import sd15_defaults

    assert issubclass(PGMAPStableDiffusionPipeline, StableDiffusionPipeline), (
        "PGMAPStableDiffusionPipeline must subclass StableDiffusionPipeline"
    )
    print("[1/3] SD 1.5 — class hierarchy OK")

    print("      loading SD 1.5 (cached HF model)...")
    t0 = time.time()
    pipe = PGMAPStableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    print(f"      loaded in {time.time() - t0:.1f}s")

    # Vanilla pass-through
    print("      vanilla pass-through (8 steps)...")
    t0 = time.time()
    img1 = pipe(
        "a cinematic photo of a red panda astronaut",
        num_inference_steps=8,
        guidance_scale=7.5,
        height=512, width=512,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    assert img1.size == (512, 512) and img1.mode == "RGB", f"unexpected: {img1.size}/{img1.mode}"
    print(f"      vanilla generated in {time.time() - t0:.1f}s")

    # PG-MAP MAP-c (no reward backward — fast)
    print("      PG-MAP MAP-c path (8 steps, no reward)...")
    cfg = sd15_defaults()
    cfg = replace(cfg, num_steps=8, seed=42, optimize_z=False, use_reward=False)
    cfg.refinement.K = 1
    t0 = time.time()
    img2 = pipe(
        "a cinematic photo of a red panda astronaut",
        pg_map_config=cfg,
        num_inference_steps=8,
        guidance_scale=7.5,
    ).images[0]
    assert img2.size == (512, 512) and img2.mode == "RGB"
    print(f"      MAP-c generated in {time.time() - t0:.1f}s")
    print("[1/3] SD 1.5 PASSED")
    del pipe
    torch.cuda.empty_cache()


def smoke_sdxl():
    """Test PGMAPStableDiffusionXLPipeline (vanilla pass-through only)."""
    import torch
    from diffusers import StableDiffusionXLPipeline

    from pgmap.pipelines import PGMAPStableDiffusionXLPipeline

    assert issubclass(PGMAPStableDiffusionXLPipeline, StableDiffusionXLPipeline)
    print("[2/3] SDXL — class hierarchy OK")

    print("      loading SDXL (cached HF model)...")
    t0 = time.time()
    pipe = PGMAPStableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    print(f"      loaded in {time.time() - t0:.1f}s")

    print("      vanilla pass-through (4 steps, 1024x1024)...")
    t0 = time.time()
    img = pipe(
        "a phoenix rising from ashes",
        num_inference_steps=4,
        guidance_scale=5.0,
        height=1024, width=1024,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    assert img.size == (1024, 1024) and img.mode == "RGB"
    print(f"      vanilla generated in {time.time() - t0:.1f}s")
    print("[2/3] SDXL PASSED")
    print("      (PG-MAP path on SDXL: tested by scripts/reproduce_table1_sdxl.sh)")
    del pipe
    torch.cuda.empty_cache()


def smoke_sd3():
    """Test PGMAPStableDiffusion3Pipeline (class hierarchy only)."""
    from diffusers import StableDiffusion3Pipeline

    from pgmap.pipelines import PGMAPStableDiffusion3Pipeline

    assert issubclass(PGMAPStableDiffusion3Pipeline, StableDiffusion3Pipeline)
    print("[3/3] SD3.5 — class hierarchy OK")
    print("      (loading SD3.5 requires Stability AI Community License acceptance;")
    print("       end-to-end generation is exercised by scripts/reproduce_table2_fm.sh)")
    print("[3/3] SD3.5 PASSED (class check only)")


def main():
    ap = argparse.ArgumentParser(description="PG-MAP pipeline smoke tests")
    ap.add_argument("--skip-sdxl", action="store_true", help="Skip the SDXL test (saves ~30s).")
    ap.add_argument("--skip-sd3",  action="store_true", help="Skip the SD3.5 class check.")
    args = ap.parse_args()

    smoke_sd15()
    if not args.skip_sdxl:
        smoke_sdxl()
    if not args.skip_sd3:
        smoke_sd3()

    print("\n[PIPELINE SMOKE] all checks passed")


if __name__ == "__main__":
    main()
