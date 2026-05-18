#!/usr/bin/env python3
"""
FM-MAP-cz variants experiment on SD3.5-medium.

Six methods compared on the same 100 PartiPrompts:
  baseline           = standard SD3 rectified-flow Euler sampler
  ug_fm_data         = UG-FM with data-side gate (paper's 91.9% PS reference)
  pgmap_fm_noise     = paper's full PG-MAP-FM, noise-side gate (the failing one)
  fm_v1_c_only       = c-only PG-MAP-FM (drop z-optimization), noise-side gate
  fm_v2_amp_compen   = amplification-compensating eta_z schedule, noise-side gate
  fm_v3_trust_region = trust-region proximal with t-aware tau, noise-side gate

All variants use noise-side gate (where the paper's PG-MAP-FM fails) so success
here is the strongest test that we have recovered MAP_cz behavior on FM.

Usage:
  python run_fm_variants_experiment.py --n 100 --out_dir fm_variants_run_n100
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Dict, List

import torch
from PIL import Image
from tqdm.auto import tqdm

from pgmap_eval import load_parti_prompts, _load_utils_scorers, score_method_pair, _aggregate
from pgmap_reward import FrozenRewardModel
from pgmap_flow_sd3 import (
    SD3FlowModels, load_sd3_models,
    generate_sd3_baseline, generate_sd3_ug_flow, generate_sd3_pgmap_flow,
)
from pgmap_flow_variants import generate_sd3_fm_v2, generate_sd3_fm_v3


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------

def gen_one(method: str, prompt: str, neg_prompt: str,
            models: SD3FlowModels, reward_model, seed: int):
    if method == "baseline":
        return generate_sd3_baseline(
            prompt, neg_prompt,
            models=models, height=1024, width=1024,
            num_steps=28, cfg_scale=7.0, seed=seed,
        )
    if method == "ug_fm_data":
        return generate_sd3_ug_flow(
            prompt, neg_prompt,
            models=models, reward_model=reward_model,
            height=1024, width=1024,
            num_steps=28, cfg_scale=7.0, seed=seed,
            K_ug=4, eta_z=0.1, rho_Q=0.3, gate_side="data",
        )
    if method == "pgmap_fm_noise":
        return generate_sd3_pgmap_flow(
            prompt, neg_prompt,
            models=models, reward_model=reward_model,
            height=1024, width=1024,
            num_steps=28, cfg_scale=7.0, seed=seed,
            K=2, eta_c=1e-3, eta_z=5e-3,
            sigma_c=1.0, gamma=1.0, sigma_flow=0.1,
            lambda_reward=0.05,
            rho=0.5, rho_Q=0.3,
            optimize_c=True, optimize_z=True, use_reward=True,
            gate_side="noise",
        )
    if method == "fm_v1_c_only":
        # c-only: drop z-optimization entirely
        return generate_sd3_pgmap_flow(
            prompt, neg_prompt,
            models=models, reward_model=reward_model,
            height=1024, width=1024,
            num_steps=28, cfg_scale=7.0, seed=seed,
            K=2, eta_c=1e-3, eta_z=0.0,
            sigma_c=1.0, gamma=1.0, sigma_flow=0.1,
            lambda_reward=0.05,
            rho=0.5, rho_Q=0.3,
            optimize_c=True, optimize_z=False, use_reward=True,
            gate_side="noise",
        )
    if method == "fm_v2_amp_compen":
        return generate_sd3_fm_v2(
            prompt, neg_prompt,
            models=models, reward_model=reward_model,
            height=1024, width=1024,
            num_steps=28, cfg_scale=7.0, seed=seed,
            K=2, eta_c=1e-3, eta_z=0.5,    # base eta is large; gets divided
            sigma_c=1.0, gamma=1.0, sigma_flow=0.1,
            lambda_reward=0.05,
            rho=0.5, rho_Q=0.3,
            optimize_c=True, optimize_z=True, use_reward=True,
            gate_side="noise", A_amp=1.05,
        )
    if method == "fm_v3_trust_region":
        return generate_sd3_fm_v3(
            prompt, neg_prompt,
            models=models, reward_model=reward_model,
            height=1024, width=1024,
            num_steps=28, cfg_scale=7.0, seed=seed,
            K=2, eta_c=1e-3, eta_z=0.1,
            sigma_c=1.0, gamma=1.0, sigma_flow=0.1,
            lambda_reward=0.05,
            rho=0.5, rho_Q=0.3,
            optimize_c=True, optimize_z=True, use_reward=True,
            gate_side="noise", tau_0=1.0,
        )
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def gen_loop(method: str, prompts, seeds, models, reward_model,
             out_dir: str, neg_prompt: str):
    img_dir = os.path.join(out_dir, method, "images")
    os.makedirs(img_dir, exist_ok=True)
    print(f"\n[GEN] {method} -> {img_dir}")
    for i, (p, s) in enumerate(tqdm(list(zip(prompts, seeds)), desc=method)):
        path = os.path.join(img_dir, f"{i:05d}.png")
        if os.path.exists(path):
            continue
        img = gen_one(method, p, neg_prompt, models, reward_model, s)
        img.save(path)
    return img_dir


def score_all(out_dir: str, baseline_dir: str, method_dirs: Dict[str, str],
              prompts: List[str], seeds: List[int]):
    print("\n[SCORE] Loading scorers...")
    scorers = _load_utils_scorers(
        score_device="cuda",
        enable_pickscore=True, enable_aes=True,
        enable_hps=True, enable_clip=True,
    )
    summaries = {}
    for name, mdir in method_dirs.items():
        out = os.path.join(out_dir, name)
        os.makedirs(out, exist_ok=True)
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
    ap.add_argument("--methods", nargs="+",
                    default=["ug_fm_data", "pgmap_fm_noise",
                             "fm_v1_c_only", "fm_v2_amp_compen", "fm_v3_trust_region"],
                    help="Method names to run (baseline always runs).")
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
        print("[INFO] Loading SD3.5-medium (FM backbone)")
        models = load_sd3_models("stabilityai/stable-diffusion-3.5-medium",
                                 device="cuda", dtype=torch.bfloat16)
        print("[INFO] Loading PickScore reward")
        reward_model = FrozenRewardModel("pickscore", device="cuda")

        baseline_dir = gen_loop("baseline", prompts, seeds, models, reward_model,
                                 args.out_dir, args.negative_prompt)
        for m in args.methods:
            method_dirs[m] = gen_loop(m, prompts, seeds, models, reward_model,
                                       args.out_dir, args.negative_prompt)

        del models, reward_model
        gc.collect(); torch.cuda.empty_cache()
    else:
        baseline_dir = os.path.join(args.out_dir, "baseline", "images")
        for m in args.methods:
            d = os.path.join(args.out_dir, m, "images")
            if os.path.isdir(d):
                method_dirs[m] = d

    if not args.skip_score:
        score_all(args.out_dir, baseline_dir, method_dirs, prompts, seeds)
    print(f"\n[DONE] Results in {args.out_dir}")


if __name__ == "__main__":
    main()
