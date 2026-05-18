"""FlowChef vs PG-MAP UG-FM eval driver on SD3.5-medium.

Reuses PG-MAP's PartiPrompts loader, scoring helpers, and FM baseline
images. Generates ONLY the FlowChef variant (the baseline is symlinked
from PG-MAP's pre-computed n=1632 baseline directory at the same seed).

Output layout matches PG-MAP's flow_pgmap_eval.py:
    <out_root>/flow_baseline/images/00000.png ... (symlink)
    <out_root>/flow_flowchef/images/00000.png ... (this run)
    <out_root>/flow_flowchef/scores.jsonl
    <out_root>/flow_flowchef/scores_summary.json
    <out_root>/flow_flowchef/bootstrap.json
    <out_root>/flow_flowchef/wilcoxon.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
MAP_PGAC = HERE.parent
sys.path.insert(0, str(MAP_PGAC))
sys.path.insert(0, str(HERE))

from pgmap_eval import (
    load_parti_prompts,
    score_method_pair,
    compute_all_statistics,
    _load_utils_scorers,
)
from pgmap_flow_sd3 import load_sd3_models, generate_sd3_baseline
from flowchef_sd3 import generate_sd3_flowchef


# Optional: symlink baseline images from another directory instead of regenerating.
# Override via PG_MAP_SHARED_FM_BASELINE env var (set to absolute path) or pass
# --regenerate_baseline to disable symlinking entirely.
SHARED_BASELINE = os.environ.get(
    "PG_MAP_SHARED_FM_BASELINE",
    str(Path(__file__).resolve().parent.parent / "eval_results" / "table2_fm" / "baseline" / "images"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_prompts", type=int, default=1632)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--neg_prompt", type=str, default="blurry, low quality")
    # FlowChef hyperparams
    ap.add_argument("--K", type=int, default=1,
                    help="FlowChef inner-ascent steps. K=1 is the FlowChef released "
                         "config; K=4 is NFE-matched to PG-MAP UG-FM.")
    ap.add_argument("--eta_z", type=float, default=0.1)
    ap.add_argument("--rho_Q", type=float, default=0.3,
                    help="Reward window fraction (data/noise gate only).")
    ap.add_argument("--gate_side", type=str, default="alwayson",
                    choices=["alwayson", "data", "noise"])
    ap.add_argument("--reward_model", type=str, default="pickscore")
    # Run-control
    ap.add_argument("--regenerate_baseline", action="store_true",
                    help="Generate baseline locally instead of symlinking.")
    ap.add_argument("--no_score", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Prompts (CRITICAL: load_parti_prompts is not prefix-stable across n;
    #               use n=1632 then truncate so we align with shared baseline) ----
    full = load_parti_prompts(args.seed, 1632)
    prompts = full[:args.num_prompts]
    seeds = [args.seed + i for i in range(len(prompts))]
    print(f"Loaded {len(prompts)} prompts (seed={args.seed}, prefix of n=1632).")

    # ---- Symlink (or generate) baseline ----
    base_dir = out_root / "flow_baseline" / "images"
    base_dir.mkdir(parents=True, exist_ok=True)
    shared = Path(SHARED_BASELINE)
    if (not args.regenerate_baseline) and shared.exists():
        n_linked = 0
        for i in range(len(prompts)):
            src = shared / f"{i:05d}.png"
            dst = base_dir / f"{i:05d}.png"
            if src.exists() and not dst.exists():
                os.symlink(src, dst)
                n_linked += 1
        print(f"Symlinked {n_linked} baseline images from {shared}")

    # ---- FlowChef variant ----
    fc_dir = out_root / "flow_flowchef" / "images"
    fc_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(fc_dir.glob("*.png"))
    n_existing = len(existing)
    if n_existing >= len(prompts):
        print(f"[skip] flowchef already has {n_existing}/{len(prompts)} images")
    else:
        print(f"Loading SD3.5-medium ...")
        device = args.device
        dtype = torch.float16 if (args.dtype == "fp16" and device == "cuda") else torch.float32
        models = load_sd3_models(
            "stabilityai/stable-diffusion-3.5-medium", device=device, dtype=dtype
        )
        from pgmap_reward import FrozenRewardModel
        print(f"Loading reward model: {args.reward_model}...")
        reward_model = FrozenRewardModel(args.reward_model, device=device)

        # Optionally generate baselines that the symlink missed
        if args.regenerate_baseline or not shared.exists():
            print("Generating baselines locally...")
            for i, (p, s) in enumerate(tqdm(list(zip(prompts, seeds)), desc="baseline")):
                target = base_dir / f"{i:05d}.png"
                if target.exists():
                    continue
                img = generate_sd3_baseline(
                    p, neg_prompt=args.neg_prompt, models=models,
                    height=args.height, width=args.width,
                    num_steps=args.steps, cfg_scale=args.cfg, seed=s,
                )
                img.save(target)

        print(f"Generating FlowChef (K={args.K}, eta_z={args.eta_z}, gate={args.gate_side}) ...")
        for i, (p, s) in enumerate(tqdm(list(zip(prompts, seeds)), desc="flowchef")):
            target = fc_dir / f"{i:05d}.png"
            if target.exists():
                continue
            img = generate_sd3_flowchef(
                p, neg_prompt=args.neg_prompt, models=models,
                reward_model=reward_model,
                height=args.height, width=args.width,
                num_steps=args.steps, cfg_scale=args.cfg, seed=s,
                K=args.K, eta_z=args.eta_z, rho_Q=args.rho_Q,
                gate_side=args.gate_side,
            )
            img.save(target)
            if torch.cuda.is_available() and (i + 1) % 25 == 0:
                torch.cuda.empty_cache()

        del models, reward_model
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Score ----
    if not args.no_score:
        print("\nLoading scorers...")
        scorers = _load_utils_scorers(
            score_device=args.device,
            enable_pickscore=True, enable_aes=True,
            enable_hps=True, enable_clip=True,
        )
        v_dir = out_root / "flow_flowchef"
        rows = score_method_pair(
            str(base_dir), str(fc_dir), prompts, scorers, str(v_dir), seeds,
        )
        compute_all_statistics(rows, str(v_dir))
        sm = json.loads((v_dir / "scores_summary.json").read_text())
        print("\n" + "=" * 78)
        print(f"FlowChef on SD3.5-medium — n={len(prompts)} | K={args.K} eta_z={args.eta_z} gate={args.gate_side}")
        print("=" * 78)
        for m in ["pickscore", "hps", "clip", "aes"]:
            d = sm.get(m, {})
            print(f"  {m:9s}: WR={d.get('win_rate', 0)*100:.1f}%  "
                  f"mean_delta={d.get('mean_delta', 0):+.5f}  n={d.get('n', 0)}")
        print("=" * 78)


if __name__ == "__main__":
    main()
