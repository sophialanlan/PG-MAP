#!/usr/bin/env python3
"""SDXL: MAP-cz (lambda=0) + closed-form damped Newton inner-loop step.

This is the *new recommended default* deployment per the user's revised
strategy: drop the reward term entirely (lambda=0 -> MAP-cz) and use the
analytical Newton preconditioner instead of K=2 SGD with hand-tuned
eta_z=0.005. Compares against baseline only (since the original MAP-cz
+ Newton has no n=1632 baseline yet).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Dict

import torch
from PIL import Image
from tqdm.auto import tqdm

from pgmap_config import PGMAPConfig, baseline_config, sdxl_defaults, RefinementConfig
from pgmap_eval import load_parti_prompts, _load_utils_scorers, score_method_pair, _aggregate
from pgmap_reward import FrozenRewardModel
from pgmap_sdxl import generate_sdxl_baseline, load_sdxl_models
from pgmap_sdxl_variants import generate_sdxl_pgmap_newton


def mapcz_newton_config() -> PGMAPConfig:
    """MAP-cz (lambda=0) + closed-form damped Newton step, K=1, eta_0=0.1.
    No reward, joint (c, z) optimization. The Newton preconditioner cancels
    the 1/beta_t schedule coupling so eta_0=0.1 is safe.
    """
    cfg = sdxl_defaults()
    cfg.optimize_c = True
    cfg.optimize_z = True
    cfg.use_reward = False
    cfg.reward.lambda_reward = 0.0
    cfg.reward.rho_Q = 0.3
    cfg.reward.grad_norm_strategy = "unit"
    cfg.prior.gamma = 1.0
    cfg.rho = 0.5
    cfg.refinement = RefinementConfig(
        K=1, eta_c=1e-3, eta_z=0.1,
        optimizer="sgd", gauss_seidel=False,
    )
    return cfg


def gen_baseline(prompts, seeds, models, out_dir, neg):
    cfg = baseline_config("sdxl")
    cfg.num_steps = 50; cfg.guidance_scale = 5.0; cfg.height = cfg.width = 1024
    img_dir = os.path.join(out_dir, "baseline", "images"); os.makedirs(img_dir, exist_ok=True)
    print(f"\n[GEN] baseline -> {img_dir}")
    for i, (p, s) in enumerate(tqdm(list(zip(prompts, seeds)), desc="baseline")):
        path = os.path.join(img_dir, f"{i:05d}.png")
        if os.path.exists(path): continue
        cfg.seed = s
        img, _ = generate_sdxl_baseline(p, neg, models=models, config=cfg)
        img.save(path)
    return img_dir


def gen_method(name, gen_fn, cfg, prompts, seeds, models, reward_model, out_dir, neg):
    img_dir = os.path.join(out_dir, name, "images"); os.makedirs(img_dir, exist_ok=True)
    print(f"\n[GEN] {name} -> {img_dir}")
    for i, (p, s) in enumerate(tqdm(list(zip(prompts, seeds)), desc=name)):
        path = os.path.join(img_dir, f"{i:05d}.png")
        if os.path.exists(path): continue
        cfg.seed = s
        img, _ = gen_fn(p, neg, models=models, config=cfg, reward_model=reward_model)
        img.save(path)
    return img_dir


def score_all(out_dir, baseline_dir, method_dirs, prompts, seeds):
    print("\n[SCORE] Loading scorers...")
    scorers = _load_utils_scorers(score_device="cuda",
        enable_pickscore=True, enable_aes=True, enable_hps=True, enable_clip=True,
    )
    summaries = {}
    for name, mdir in method_dirs.items():
        out = os.path.join(out_dir, name); os.makedirs(out, exist_ok=True)
        print(f"\n[SCORE] {name} vs baseline")
        rows = score_method_pair(baseline_dir, mdir, prompts, scorers, out, seeds)
        s = _aggregate(rows)
        summaries[name] = s
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(s, f, indent=2)
    with open(os.path.join(out_dir, "all_summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1632)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--negative_prompt", default="blurry, low quality")
    ap.add_argument("--skip_gen", action="store_true")
    ap.add_argument("--skip_score", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg_log = vars(args).copy()
    cfg_log["torch"] = torch.__version__
    if torch.cuda.is_available():
        cfg_log["gpu"] = torch.cuda.get_device_name(0)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg_log, f, indent=2)

    prompts = load_parti_prompts(args.seed, args.n)
    seeds = [args.seed + i for i in range(len(prompts))]
    with open(os.path.join(args.out_dir, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"[INFO] Loaded {len(prompts)} prompts (seed {args.seed})")

    method_dirs: Dict[str, str] = {}
    if not args.skip_gen:
        print("[INFO] Loading SDXL")
        models = load_sdxl_models("stabilityai/stable-diffusion-xl-base-1.0",
                                   device="cuda", dtype=torch.float16)
        # No reward model needed since lambda=0, but the variant pipeline
        # still expects one (it just doesn't get called). Pass None.
        reward_model = None

        baseline_dir = gen_baseline(prompts, seeds, models, args.out_dir, args.negative_prompt)
        method_dirs["mapcz_newton"] = gen_method(
            "mapcz_newton", generate_sdxl_pgmap_newton, mapcz_newton_config(),
            prompts, seeds, models, reward_model, args.out_dir, args.negative_prompt,
        )

        del models, reward_model
        gc.collect(); torch.cuda.empty_cache()
    else:
        baseline_dir = os.path.join(args.out_dir, "baseline", "images")
        for n in ["mapcz_newton"]:
            d = os.path.join(args.out_dir, n, "images")
            if os.path.isdir(d): method_dirs[n] = d

    if not args.skip_score:
        score_all(args.out_dir, baseline_dir, method_dirs, prompts, seeds)
    print(f"\n[DONE] Results in {args.out_dir}")


if __name__ == "__main__":
    main()
