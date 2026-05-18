#!/usr/bin/env python3
"""SD1.5 driver: baseline + newB_newton + (optional) pgmap_reference on
n PartiPrompts. Mirrors run_variants2_experiment.py for SDXL but switches to
SD1.5 backbone and uses pgmap_sd15_variants.generate_sd15_pgmap_newton.
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

from pgmap_config import (
    PGMAPConfig, baseline_config, sd15_defaults, RefinementConfig,
)
from pgmap_eval import load_parti_prompts, _load_utils_scorers, score_method_pair, _aggregate
from pgmap_reward import FrozenRewardModel
from pgmap_sd15 import load_sd15_models, generate_sd15_baseline, generate_sd15_pgmap
from pgmap_sd15_variants import generate_sd15_pgmap_newton


def _shared_base() -> PGMAPConfig:
    cfg = sd15_defaults()
    cfg.optimize_c = True
    cfg.optimize_z = True
    cfg.use_reward = True
    cfg.reward.lambda_reward = 0.1
    cfg.reward.rho_Q = 0.3
    cfg.reward.grad_norm_strategy = "unit"
    cfg.prior.gamma = 0.5
    cfg.rho = 0.4
    return cfg


def newB_config_newton() -> PGMAPConfig:
    """B2: closed-form damped Newton, K=1 enforced internally.
    eta_z=0.1 is the "soft damping" inverse trust radius, identical to the
    SDXL configuration so behaviour is comparable across backbones.
    """
    cfg = _shared_base()
    cfg.refinement = RefinementConfig(
        K=1, eta_c=1e-3, eta_z=0.1,
        optimizer="sgd", gauss_seidel=False,
    )
    return cfg


def reference_pgmap_config() -> PGMAPConfig:
    """SD1.5 paper default."""
    cfg = _shared_base()
    cfg.refinement = RefinementConfig(
        K=1, eta_c=1e-4, eta_z=0.005,
        optimizer="sgd", gauss_seidel=False,
    )
    return cfg


def gen_baseline(prompts, seeds, models, out_dir, neg):
    cfg = baseline_config("sd15")
    img_dir = os.path.join(out_dir, "baseline", "images"); os.makedirs(img_dir, exist_ok=True)
    print(f"\n[GEN] baseline -> {img_dir}")
    for i, (p, s) in enumerate(tqdm(list(zip(prompts, seeds)), desc="baseline")):
        path = os.path.join(img_dir, f"{i:05d}.png")
        if os.path.exists(path): continue
        cfg.seed = s
        img, _ = generate_sd15_baseline(p, neg, models=models, config=cfg)
        img.save(path)
    return img_dir


def _gen_loop(name, gen_fn, cfg, prompts, seeds, models, reward_model, out_dir, neg):
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
    scorers = _load_utils_scorers(
        score_device="cuda",
        enable_pickscore=True, enable_aes=True,
        enable_hps=True, enable_clip=True,
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
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--negative_prompt", default="blurry, low quality")
    ap.add_argument("--include_reference_pgmap", action="store_true")
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
    print(f"[INFO] Loaded {len(prompts)} prompts")

    method_dirs: Dict[str, str] = {}
    if not args.skip_gen:
        print("[INFO] Loading SD1.5 models")
        models = load_sd15_models(
            "runwayml/stable-diffusion-v1-5",
            device="cuda", dtype=torch.float16,
        )
        print("[INFO] Loading PickScore reward model")
        reward_model = FrozenRewardModel("pickscore", device="cuda")

        baseline_dir = gen_baseline(prompts, seeds, models, args.out_dir, args.negative_prompt)
        method_dirs["newB_newton"] = _gen_loop(
            "newB_newton", generate_sd15_pgmap_newton, newB_config_newton(),
            prompts, seeds, models, reward_model, args.out_dir, args.negative_prompt,
        )
        if args.include_reference_pgmap:
            method_dirs["pgmap_reference"] = _gen_loop(
                "pgmap_reference", generate_sd15_pgmap, reference_pgmap_config(),
                prompts, seeds, models, reward_model, args.out_dir, args.negative_prompt,
            )

        del models, reward_model
        gc.collect(); torch.cuda.empty_cache()
    else:
        baseline_dir = os.path.join(args.out_dir, "baseline", "images")
        for name in ["newB_newton"] + (["pgmap_reference"] if args.include_reference_pgmap else []):
            d = os.path.join(args.out_dir, name, "images")
            if os.path.isdir(d):
                method_dirs[name] = d

    if not args.skip_score:
        score_all(args.out_dir, baseline_dir, method_dirs, prompts, seeds)
    print(f"\n[DONE] Results in {args.out_dir}")


if __name__ == "__main__":
    main()
