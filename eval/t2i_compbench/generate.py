#!/usr/bin/env python3
"""Generate images for T2I-CompBench++ evaluation using PG-MAP.

Produces PNGs in the format the T2I-CompBench eval scripts expect:

    out_dir/<category>/samples/<prompt-text>_NNNNNN.png

where NNNNNN is a 6-digit zero-padded index that matches the prompt's
line number in the val txt file. One image per prompt by default
(the typical pattern in the T2I-CompBench paper); set --n_per_prompt
higher for variance estimation.

Usage::

    python eval/t2i_compbench/generate.py \\
        --method pgmap_K1 \\
        --backbone sdxl \\
        --categories color shape texture \\
        --out_dir eval_results/t2i_compbench/sdxl_pgmap_K1/

Methods:
    baseline_sdxl  — vanilla SDXL (no PG-MAP), reference run.
    mapc           — Conditioning-only MAP refinement (optimize_c, no z, no reward).
                     Paper's predicted attribute-binding winner (paper §3:
                     "strongest on attribute-binding and short / typography prompts").
    pgmap_K1       — SDXL PG-MAP, K_inner=1 (~5x faster than K=2, ~25% smaller win-rates).
    pgmap_K2       — SDXL PG-MAP, paper default K_inner=2.
    pgmap_K2_tcfg  — SDXL Tuned-CFG (w=7.5) + PG-MAP K_inner=2 (highest HPS/Aesthetic).
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

# Allow running as a script from anywhere
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def _slugify(prompt: str, maxlen: int = 200) -> str:
    """T2I-CompBench filename convention: keep the prompt text mostly intact,
    strip leading/trailing whitespace, replace pathological chars only."""
    s = prompt.strip().rstrip(".")
    # Filesystem-illegal chars
    s = re.sub(r'[/\\\0]', '_', s)
    if len(s) > maxlen:
        s = s[:maxlen]
    return s


def load_prompts(category: str) -> list[str]:
    fp = PROMPT_DIR / f"{category}.txt"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp} — copy from upstream T2I-CompBench/examples/dataset/")
    return [ln.strip() for ln in fp.read_text().splitlines() if ln.strip()]


def build_config(method: str, backbone: str, seed: int):
    """Build PGMAPConfig for the requested method."""
    from pgmap_config import (
        PriorConfig, RefinementConfig, RewardConfig,
        baseline_config, sd15_defaults, sdxl_defaults,
    )

    if backbone == "sd15":
        base = sd15_defaults
    elif backbone == "sdxl":
        base = sdxl_defaults
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    if method.startswith("baseline"):
        cfg = baseline_config(backbone)
        cfg.seed = seed
        return cfg

    cfg = base()
    cfg.seed = seed
    cfg.prior = PriorConfig(sigma_c=1.0, gamma=1.0)

    if method == "mapc":
        # Conditioning-only MAP, no reward — paper's attribute-binding pick.
        # K=1 to match the cost of pgmap_K1 (single inner step).
        cfg.refinement = RefinementConfig(K=1, eta_c=1e-3, eta_z=0.0)
        cfg.reward = RewardConfig(lambda_reward=0.0, rho_Q=0.0, grad_norm_strategy="unit")
        cfg.optimize_c = True
        cfg.optimize_z = False
        cfg.use_reward = False
    elif method == "pgmap_K1":
        cfg.refinement = RefinementConfig(K=1, eta_c=1e-3, eta_z=5e-3)
        cfg.reward = RewardConfig(lambda_reward=0.1, rho_Q=0.3, grad_norm_strategy="unit")
    elif method == "pgmap_K2":
        cfg.refinement = RefinementConfig(K=2, eta_c=1e-3, eta_z=5e-3)
        cfg.reward = RewardConfig(lambda_reward=0.1, rho_Q=0.3, grad_norm_strategy="unit")
    elif method == "pgmap_K2_tcfg":
        cfg.refinement = RefinementConfig(K=2, eta_c=1e-3, eta_z=5e-3)
        cfg.reward = RewardConfig(lambda_reward=0.1, rho_Q=0.3, grad_norm_strategy="unit")
        cfg.guidance_scale = 7.5     # Tuned-CFG (paper §3 row 8)
    else:
        raise ValueError(f"Unknown method: {method}")
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Generate images for T2I-CompBench++ eval")
    ap.add_argument("--method", required=True,
                    choices=["baseline_sdxl", "mapc", "pgmap_K1", "pgmap_K2", "pgmap_K2_tcfg"])
    ap.add_argument("--backbone", default="sdxl", choices=["sd15", "sdxl"])
    ap.add_argument("--categories", nargs="+", required=True,
                    choices=["color", "shape", "texture", "spatial",
                              "non_spatial", "complex"])
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_per_prompt", type=int, default=1,
                    help="Images per prompt (T2I-CompBench paper uses 10; we default to 1).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Truncate to first N prompts per category (smoke test).")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip prompts whose PNGs already exist on disk.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[t2i_compbench] method={args.method}  backbone={args.backbone}  "
          f"categories={args.categories}  out_dir={args.out_dir}")

    # Lazy model load (only after argparse so smoke runs / --help are instant)
    print("[load] PG-MAP pipeline + reward model...")
    if args.backbone == "sd15":
        from pgmap_sd15 import generate_sd15_baseline, generate_sd15_pgmap, load_sd15_models
        models = load_sd15_models("runwayml/stable-diffusion-v1-5", dtype=torch.float16)
        gen_pgmap = generate_sd15_pgmap
        gen_baseline = generate_sd15_baseline
    else:
        from pgmap_sdxl import generate_sdxl_baseline, generate_sdxl_pgmap, load_sdxl_models
        models = load_sdxl_models("stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.float16)
        gen_pgmap = generate_sdxl_pgmap
        gen_baseline = generate_sdxl_baseline

    needs_reward = "pgmap" in args.method
    reward = None
    if needs_reward:
        from pgmap_reward import FrozenRewardModel
        reward = FrozenRewardModel("pickscore", device="cuda")

    t_all = time.time()
    total_imgs = 0

    for cat in args.categories:
        prompts = load_prompts(cat)
        if args.limit is not None:
            prompts = prompts[: args.limit]
        cat_dir = args.out_dir / cat / "samples"
        cat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== [{cat}]  {len(prompts)} prompts  -> {cat_dir} ===")
        t_cat = time.time()
        for q_id, prompt in enumerate(prompts):
            for rep in range(args.n_per_prompt):
                # Filename convention: <prompt-slug>_NNNNNN.png with NNNNNN = q_id (matches T2I-CompBench)
                # When n_per_prompt > 1, replicate at different seeds (q_id stays same).
                fname = f"{_slugify(prompt)}_{q_id:06d}.png"
                fpath = cat_dir / fname
                if args.skip_existing and fpath.exists():
                    continue

                seed_i = args.seed + q_id * args.n_per_prompt + rep
                cfg = build_config(args.method, args.backbone, seed_i)

                if args.method.startswith("baseline"):
                    img, _ = gen_baseline(prompt, models=models, config=cfg)
                else:
                    img, _ = gen_pgmap(prompt, models=models, config=cfg, reward_model=reward)
                img.save(fpath)
                total_imgs += 1
                if (q_id % 20 == 0) and rep == 0:
                    eta = (time.time() - t_cat) / max(q_id + 1, 1) * (len(prompts) - q_id - 1)
                    print(f"  [{q_id+1}/{len(prompts)}]  ETA {eta/60:.1f} min")

        print(f"=== [{cat}] done in {(time.time()-t_cat)/60:.1f} min ===")

    print(f"\n[t2i_compbench] all done. {total_imgs} images in {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
