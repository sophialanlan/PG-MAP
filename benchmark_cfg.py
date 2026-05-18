#!/usr/bin/env python3
"""
Tuned-CFG Sweep Baseline (Step 2)
===================================

Sweeps CFG guidance_scale w ∈ {3, 5, 7.5, 10, 12, 15} on the validation
split (n=490, 30% of PartiPrompts 1632), picks the best w per backbone per
metric, then runs the full test split (n=1142) at the chosen w.

Evaluation metrics: HPSv2, PickScore, ImageReward, CLIPScore.
Seeds: {0, 1, 2} cycled per prompt (same protocol as PG-MAP).

Usage:
    # Val sweep (SDXL)
    python benchmark_cfg.py --backbone sdxl --phase val \
        --out_dir eval_results/cfg_sweep/sdxl

    # Full test at chosen w
    python benchmark_cfg.py --backbone sdxl --phase test --guidance 7.5 \
        --out_dir eval_results/cfg_sweep/sdxl

    # SD1.5
    python benchmark_cfg.py --backbone sd15 --phase val \
        --out_dir eval_results/cfg_sweep/sd15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

# ── env ─────────────────────────────────────────────────────────────────────
_default_cache = os.path.expanduser("~/.cache/pgmap")
os.environ.setdefault("HF_HOME",            f"{_default_cache}/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE",  f"{_default_cache}/hf_home/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_default_cache}/hf_home/transformers")
os.environ.setdefault("HPS_CACHE",          f"{_default_cache}/hps_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── constants ────────────────────────────────────────────────────────────────
CFG_SWEEP = [3.0, 5.0, 7.5, 10.0, 12.0, 15.0]
SEEDS = [0, 1, 2]
VAL_FRAC = 0.30   # 30 % of PartiPrompts → validation split
PARTI_SEED = 123  # master seed for prompt ordering


# ── prompt loading ───────────────────────────────────────────────────────────

def _clean(s: str) -> str:
    return " ".join(s.replace("<|endoftext|>", " ").split()).strip()


def load_parti_prompts_split(master_seed: int = PARTI_SEED):
    """Return (val_prompts, test_prompts) from PartiPrompts 1632."""
    from datasets import load_dataset
    ds = load_dataset("nateraw/parti-prompts", split="train")
    key = "Prompt" if "Prompt" in ds.features else list(ds.features.keys())[0]
    all_prompts = [_clean(x) for x in ds[key] if isinstance(x, str) and x.strip()]

    rng = np.random.default_rng(master_seed)
    idx = rng.permutation(len(all_prompts))
    n_val = int(len(all_prompts) * VAL_FRAC)
    val_idx   = idx[:n_val]
    test_idx  = idx[n_val:]
    return ([all_prompts[i] for i in val_idx],
            [all_prompts[i] for i in test_idx])


# ── model loading ────────────────────────────────────────────────────────────

def load_models(backbone: str, device: str, dtype: torch.dtype):
    if backbone == "sdxl":
        from pgmap_sdxl import load_sdxl_models
        return load_sdxl_models("stabilityai/stable-diffusion-xl-base-1.0", device, dtype)
    else:
        from pgmap_sd15 import load_sd15_models
        return load_sd15_models("runwayml/stable-diffusion-v1-5", device, dtype)


# ── generation ───────────────────────────────────────────────────────────────

def generate_one(
    prompt: str,
    models,
    backbone: str,
    guidance: float,
    seed: int,
    num_steps: int,
    negative_prompt: str = "blurry, low quality",
) -> Image.Image:
    """Generate a single image with standard DDIM+CFG (no PG-MAP)."""
    from pgmap_config import baseline_config

    cfg = baseline_config(backbone)
    cfg.num_steps = num_steps
    cfg.guidance_scale = guidance
    cfg.seed = seed

    if backbone == "sdxl":
        from pgmap_sdxl import generate_sdxl_pgmap
        img, _ = generate_sdxl_pgmap(prompt, negative_prompt, models=models, config=cfg)
    else:
        from pgmap_sd15 import generate_sd15_pgmap
        img, _ = generate_sd15_pgmap(prompt, negative_prompt, models=models, config=cfg)

    return img


def generate_split(
    prompts: List[str],
    models,
    backbone: str,
    guidance: float,
    out_dir: str,
    num_steps: int,
    negative_prompt: str = "blurry, low quality",
    seeds: Optional[List[int]] = None,
    skip_existing: bool = True,
) -> List[str]:
    """Generate images for all prompts, saving to out_dir/NNNNN.png.

    Returns list of saved file paths (in prompt order).
    """
    os.makedirs(out_dir, exist_ok=True)
    if seeds is None:
        seeds = [SEEDS[i % len(SEEDS)] for i in range(len(prompts))]

    paths = []
    for i, (p, s) in enumerate(tqdm(list(zip(prompts, seeds)), desc=f"cfg={guidance}")):
        path = os.path.join(out_dir, f"{i:05d}.png")
        if not (skip_existing and os.path.exists(path)):
            img = generate_one(p, models, backbone, guidance, s, num_steps, negative_prompt)
            img.save(path)
        paths.append(path)
    return paths


# ── scoring ──────────────────────────────────────────────────────────────────

def load_scorers(device: str) -> Dict:
    """Load all four evaluation scorers."""
    _repo = os.path.dirname(os.path.abspath(__file__))
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    scorers = {}
    print("  Loading PickScore...")
    from utils.pickscore_utils import Selector as PSSelector
    scorers["pickscore"] = PSSelector(device)
    print("  Loading HPSv2...")
    from utils.hps_utils import Selector as HPSSelector
    scorers["hps"] = HPSSelector(device)
    print("  Loading CLIPScore...")
    from utils.clip_utils import Selector as CLIPSelector
    scorers["clip"] = CLIPSelector(device)
    print("  Loading Aesthetic...")
    from utils.aes_utils import Selector as AesSelector
    scorers["aes"] = AesSelector(device)
    return scorers


def score_images(
    img_paths: List[str],
    prompts: List[str],
    scorers: Dict,
) -> Dict[str, List[float]]:
    """Score a list of images, returning per-metric lists."""
    results: Dict[str, List[float]] = {k: [] for k in scorers}
    for path, prompt in tqdm(list(zip(img_paths, prompts)), desc="Scoring"):
        img = Image.open(path).convert("RGB")
        for name, sel in scorers.items():
            # Selector.score expects a list of images and a prompt
            scores = sel.score([img], prompt)
            results[name].append(float(scores[0]))
    return results


def aggregate_scores(scores: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: float(np.mean(v)) for k, v in scores.items()}


# ── val sweep ────────────────────────────────────────────────────────────────

def run_val_sweep(args):
    """Sweep CFG on the validation split and find best w per metric."""
    val_prompts, _ = load_parti_prompts_split(PARTI_SEED)
    print(f"Val split: {len(val_prompts)} prompts")

    device = args.device
    dtype = torch.float16 if (args.dtype == "fp16" and device == "cuda") else torch.float32
    models = load_models(args.backbone, device, dtype)

    # Phase 1: generate at each w
    val_img_dirs: Dict[float, str] = {}
    for w in args.sweep:
        w_dir = os.path.join(args.out_dir, "val", f"cfg_{w:.1f}", "images")
        generate_split(val_prompts, models, args.backbone, w, w_dir,
                       args.steps, args.negative_prompt)
        val_img_dirs[w] = w_dir

    # Phase 2: free generation models, load scorers
    import gc
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scorers = load_scorers(args.score_device)

    sweep_results: Dict[float, Dict[str, float]] = {}
    for w, w_dir in val_img_dirs.items():
        paths = [os.path.join(w_dir, f"{i:05d}.png") for i in range(len(val_prompts))]
        raw = score_images(paths, val_prompts, scorers)
        agg = aggregate_scores(raw)
        sweep_results[w] = agg
        print(f"  w={w:.1f}: {agg}")

    # Save sweep table
    sweep_path = os.path.join(args.out_dir, "val", "sweep_results.json")
    os.makedirs(os.path.dirname(sweep_path), exist_ok=True)
    with open(sweep_path, "w") as f:
        json.dump({str(w): v for w, v in sweep_results.items()}, f, indent=2)
    print(f"Saved sweep results → {sweep_path}")

    # Best w per metric
    metrics = list(next(iter(sweep_results.values())).keys())
    best = {}
    for m in metrics:
        best_w = max(sweep_results, key=lambda w: sweep_results[w][m])
        best[m] = {"w": best_w, "score": sweep_results[best_w][m]}
    best_path = os.path.join(args.out_dir, "val", "best_w.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best w per metric → {best_path}")
    for m, d in best.items():
        print(f"  {m}: best w={d['w']:.1f}, score={d['score']:.4f}")

    return best


# ── test run ─────────────────────────────────────────────────────────────────

def run_test(args):
    """Generate and score the full test split at a fixed guidance_scale."""
    _, test_prompts = load_parti_prompts_split(PARTI_SEED)
    print(f"Test split: {len(test_prompts)} prompts  (w={args.guidance})")

    device = args.device
    dtype = torch.float16 if (args.dtype == "fp16" and device == "cuda") else torch.float32
    models = load_models(args.backbone, device, dtype)

    test_dir = os.path.join(args.out_dir, "test", f"cfg_{args.guidance:.1f}", "images")
    paths = generate_split(test_prompts, models, args.backbone, args.guidance,
                           test_dir, args.steps, args.negative_prompt)

    import gc
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scorers = load_scorers(args.score_device)
    raw = score_images(paths, test_prompts, scorers)
    agg = aggregate_scores(raw)

    out = {
        "backbone": args.backbone,
        "guidance": args.guidance,
        "n_test": len(test_prompts),
        "scores": agg,
        "per_prompt": {k: v for k, v in raw.items()},
    }
    result_path = os.path.join(args.out_dir, "test", f"cfg_{args.guidance:.1f}", "results.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Test results → {result_path}")
    print(json.dumps(agg, indent=2))
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Tuned-CFG sweep baseline")
    ap.add_argument("--backbone", choices=["sd15", "sdxl"], default="sdxl")
    ap.add_argument("--phase", choices=["val", "test", "both"], default="val",
                    help="val: sweep on validation split; test: score at --guidance; both: val then test at best pickscore w")
    ap.add_argument("--guidance", type=float, default=7.5,
                    help="Fixed CFG scale for --phase test")
    ap.add_argument("--sweep", type=float, nargs="+", default=CFG_SWEEP,
                    help="CFG values to sweep (val phase)")
    ap.add_argument("--out_dir", type=str, default="eval_results/cfg_sweep")
    ap.add_argument("--steps", type=int, default=None,
                    help="DDIM steps (default: 30 for sd15, 50 for sdxl)")
    ap.add_argument("--negative_prompt", type=str, default="blurry, low quality")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--score_device", type=str, default="cuda")
    args = ap.parse_args()

    if args.steps is None:
        args.steps = 30 if args.backbone == "sd15" else 50

    os.makedirs(args.out_dir, exist_ok=True)

    if args.phase == "val":
        run_val_sweep(args)
    elif args.phase == "test":
        run_test(args)
    else:  # both
        best = run_val_sweep(args)
        best_w = best.get("pickscore", {}).get("w", args.guidance)
        args.guidance = best_w
        print(f"\n--- Running test split at best PickScore w={best_w} ---")
        run_test(args)


if __name__ == "__main__":
    main()
