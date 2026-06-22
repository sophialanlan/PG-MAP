#!/usr/bin/env python3
"""
Universal Guidance (UG) Baseline (Step 3)
==========================================

Implements the Universal Guided Diffusion approach (Bansal et al., 2023)
with PickScore as the guidance function, matched to PG-MAP's protocol:

  * Reward activated only during the first rho_Q=0.3 fraction of steps.
  * Loss: L = -Q(x̂₀(z_t), y)  where Q = PickScore.
  * Guidance update: z_t ← z_t - η_z · ∇_{z_t} L   (latent-space gradient).
  * Only z_t is updated (no c optimisation — UG is pure latent guidance).
  * NFE-matched to PG-MAP: same total number of UNet forwards.

Key design choices vs. vanilla UG:
  - We skip VAE decode in the gradient step and instead use a low-res
    (28×28 latent) decode to match the fast-decode trick in PG-MAP.
  - We apply the gradient update BEFORE the DDIM step (same order as PG-MAP).
  - The outer DDIM step uses CFG (batched uncond+cond) to match PG-MAP's
    inference quality.

NFE counting:
  PG-MAP (K=2, rho=0.5, rho_Q=0.3):
    refine steps with reward   : 15 × (2 inner_cond + 1 outer_cfg)   = 45
    refine steps without reward: 10 × (2 inner_cond + 1 outer_cfg)   = 30
    pure DDIM steps            : 25 × 1 outer_cfg                    = 25
    Total                      : 100 UNet calls

  UG (this file) with K_ug inner steps tuned to match:
    reward steps   : rho_Q*T × (K_ug + 1 outer_cfg)
    non-reward steps:         T × 1 outer_cfg
    To match 100:  15*(K_ug+1) + 35 = 100  →  K_ug = 4

    Set K_ug=4 for NFE matching when running against PG-MAP (K=2, T=50).

Usage:
    # Tune η_z on validation split (3 LR values)
    python benchmark_ug.py --backbone sdxl --phase tune \
        --out_dir eval_results/ug/sdxl

    # Full test split at chosen LR
    python benchmark_ug.py --backbone sdxl --phase test --eta_z 0.02 \
        --out_dir eval_results/ug/sdxl

    # SD1.5
    python benchmark_ug.py --backbone sd15 --phase tune \
        --out_dir eval_results/ug/sd15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
SEEDS = [0, 1, 2]
VAL_FRAC = 0.30
PARTI_SEED = 123
ETA_Z_SWEEP = [1e-3, 1e-2, 1e-1]   # latent LR candidates for tuning
REWARD_DECODE_SIZE = 224            # latent decoded at 28×28 → 224×224 for reward


# ── prompt loading ───────────────────────────────────────────────────────────

def _clean(s: str) -> str:
    return " ".join(s.replace("<|endoftext|>", " ").split()).strip()


def load_parti_prompts_split(master_seed: int = PARTI_SEED):
    from datasets import load_dataset
    ds = load_dataset("nateraw/parti-prompts", split="train")
    key = "Prompt" if "Prompt" in ds.features else list(ds.features.keys())[0]
    all_prompts = [_clean(x) for x in ds[key] if isinstance(x, str) and x.strip()]
    rng = np.random.default_rng(master_seed)
    idx = rng.permutation(len(all_prompts))
    n_val = int(len(all_prompts) * VAL_FRAC)
    return ([all_prompts[i] for i in idx[:n_val]],
            [all_prompts[i] for i in idx[n_val:]])


# ── model loading ────────────────────────────────────────────────────────────

def load_models(backbone: str, device: str, dtype: torch.dtype):
    if backbone == "sdxl":
        from pgmap_sdxl import load_sdxl_models
        return load_sdxl_models("stabilityai/stable-diffusion-xl-base-1.0", device, dtype)
    from pgmap_sd15 import load_sd15_models
    return load_sd15_models("stable-diffusion-v1-5/stable-diffusion-v1-5", device, dtype)


def load_reward_model(device: str):
    from pgmap_reward import FrozenRewardModel
    return FrozenRewardModel("pickscore", device=device)


# ── core helpers ─────────────────────────────────────────────────────────────

def _decode_for_reward(vae, z0_hat: torch.Tensor, target_hw: int = REWARD_DECODE_SIZE) -> torch.Tensor:
    """Low-res differentiable VAE decode for reward scoring."""
    scale = getattr(vae.config, "scaling_factor", 0.18215)
    lat = target_hw // 8
    z = F.interpolate(z0_hat.float(), size=(lat, lat), mode="bilinear", align_corners=False)
    vae_dtype = next(vae.parameters()).dtype
    imgs = vae.decode((z / scale).to(vae_dtype)).sample
    return (imgs / 2 + 0.5).clamp(0, 1)


def _decode_latents(vae, latents: torch.Tensor) -> List[Image.Image]:
    scale = getattr(vae.config, "scaling_factor", 0.18215)
    vae_dtype = next(vae.parameters()).dtype
    imgs = vae.decode((latents / scale).to(vae_dtype)).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray((im * 255).round().astype("uint8")) for im in imgs]


def _cfg_eps_sdxl(unet, sched, z_t, t_train, c_u, c0, pooled_u, pooled0, time_ids, gs):
    """Batched CFG eps for SDXL (single 2B UNet call)."""
    B = z_t.shape[0]
    z_in = torch.cat([z_t, z_t], dim=0)
    t_in = t_train.repeat(2)
    c_in = torch.cat([c_u, c0], dim=0)
    dt = next(unet.parameters()).dtype
    pooled_in = torch.cat([pooled_u, pooled0], dim=0).to(dt)
    z_sc = sched.scale_model_input(z_in, t_train[0])
    out = unet(
        z_sc.to(dt), t_in,
        encoder_hidden_states=c_in.to(dt),
        added_cond_kwargs={"text_embeds": pooled_in,
                           "time_ids": time_ids.repeat(2, 1).to(dt)},
    ).sample
    eps_u, eps_c = out[:B], out[B:]
    return (eps_u + gs * (eps_c - eps_u)).to(z_t.dtype)


def _cfg_eps_sd15(unet, sched, z_t, t_train, c_u, c0, gs):
    """Batched CFG eps for SD1.5."""
    B = z_t.shape[0]
    z_in = torch.cat([z_t, z_t], dim=0)
    t_in = t_train.repeat(2)
    c_in = torch.cat([c_u, c0], dim=0)
    dt = next(unet.parameters()).dtype
    z_sc = sched.scale_model_input(z_in, t_train[0])
    out = unet(z_sc.to(dt), t_in, encoder_hidden_states=c_in.to(dt)).sample
    eps_u, eps_c = out[:B], out[B:]
    return (eps_u + gs * (eps_c - eps_u)).to(z_t.dtype)


def _z0_hat_from_eps(sched, z_t: torch.Tensor, eps: torch.Tensor, t_train: torch.Tensor) -> torch.Tensor:
    ac = sched.alphas_cumprod.to(device=z_t.device, dtype=torch.float32)
    ab_t = ac[t_train].view(-1, 1, 1, 1)
    return (z_t.float() - (1.0 - ab_t).sqrt() * eps.float()) / ab_t.sqrt()


# ── Universal Guidance generation ────────────────────────────────────────────

def generate_ug_sdxl(
    prompt: str,
    negative_prompt: str,
    models,
    reward_model,
    *,
    guidance_scale: float = 5.0,
    num_steps: int = 50,
    rho_Q: float = 0.3,
    K_ug: int = 4,
    eta_z: float = 0.02,
    seed: int = 0,
    height: int = 1024,
    width: int = 1024,
) -> Image.Image:
    """Universal Guidance generation for SDXL.

    At each reward-active step:
      1. Compute eps_cond with grad tracking.
      2. Estimate x̂₀ from eps_cond.
      3. Compute reward gradient ∇_{z_t} (-Q(x̂₀)).
      4. Apply K_ug gradient steps to z_t.
      5. Standard CFG DDIM step with updated z_t.
    """
    from pgmap_sdxl import encode_prompt_sdxl, make_sdxl_time_ids, decode_latents as _dec
    from diffusers import DDIMScheduler

    dev = models.device
    dtype = models.dtype
    unet = models.unet
    vae = models.vae
    sched = models.sched

    sched.set_timesteps(num_steps, device=dev)
    timesteps = sched.timesteps
    T = len(timesteps)
    reward_steps = max(1, int(rho_Q * T))

    c0, pooled0 = encode_prompt_sdxl(
        models.tokenizer_1, models.tokenizer_2,
        models.text_encoder_1, models.text_encoder_2, prompt, dev,
    )
    c_u, pooled_u = encode_prompt_sdxl(
        models.tokenizer_1, models.tokenizer_2,
        models.text_encoder_1, models.text_encoder_2, negative_prompt, dev,
    )
    time_ids = make_sdxl_time_ids(dev, dtype, original_size=(height, width),
                                   target_size=(height, width), batch_size=1)

    g = torch.Generator(device=dev).manual_seed(seed)
    z_t = torch.randn((1, unet.config.in_channels, height // 8, width // 8),
                      generator=g, device=dev, dtype=dtype)
    z_t = z_t * sched.init_noise_sigma

    reward_model.precompute_text_features(prompt)

    for step_i, t_val in enumerate(timesteps):
        t_train = torch.full((1,), int(t_val.item()), device=dev, dtype=torch.long)

        if step_i < reward_steps:
            # Universal Guidance: K_ug latent gradient steps
            for _ in range(K_ug):
                z_var = z_t.detach().clone().float().requires_grad_(True)
                dt = next(unet.parameters()).dtype
                z_sc = sched.scale_model_input(z_var.to(dt), t_train[0])
                eps_c = unet(
                    z_sc, t_train,
                    encoder_hidden_states=c0.to(dt),
                    added_cond_kwargs={
                        "text_embeds": pooled0.to(dt),
                        "time_ids": time_ids.to(dt),
                    },
                ).sample.float()
                z0_hat = _z0_hat_from_eps(sched, z_var, eps_c, t_train)
                pixels = _decode_for_reward(vae, z0_hat)
                Q = reward_model.score(pixels, prompt).mean()
                grad = torch.autograd.grad(-Q, z_var)[0]
                with torch.no_grad():
                    z_t = (z_var - eta_z * grad).to(dtype)

        with torch.no_grad():
            eps = _cfg_eps_sdxl(unet, sched, z_t, t_train,
                                 c_u, c0, pooled_u, pooled0, time_ids,
                                 guidance_scale)
            z_t = sched.step(eps, int(t_train[0].item()), z_t).prev_sample

    return _dec(vae, z_t)[0]


def generate_ug_sd15(
    prompt: str,
    negative_prompt: str,
    models,
    reward_model,
    *,
    guidance_scale: float = 7.5,
    num_steps: int = 30,
    rho_Q: float = 0.3,
    K_ug: int = 4,
    eta_z: float = 0.02,
    seed: int = 0,
    height: int = 512,
    width: int = 512,
) -> Image.Image:
    """Universal Guidance generation for SD1.5."""
    from pgmap_sd15 import encode_prompt, decode_latents as _dec

    dev = models.device
    dtype = models.dtype
    unet = models.unet
    vae = models.vae
    sched = models.sched

    sched.set_timesteps(num_steps, device=dev)
    timesteps = sched.timesteps
    T = len(timesteps)
    reward_steps = max(1, int(rho_Q * T))

    c0 = encode_prompt(models.tokenizer, models.text_encoder, prompt, dev)
    c_u = encode_prompt(models.tokenizer, models.text_encoder, negative_prompt, dev)

    g = torch.Generator(device=dev).manual_seed(seed)
    z_t = torch.randn((1, unet.config.in_channels, height // 8, width // 8),
                      generator=g, device=dev, dtype=dtype)
    z_t = z_t * sched.init_noise_sigma

    reward_model.precompute_text_features(prompt)

    for step_i, t_val in enumerate(timesteps):
        t_train = torch.full((1,), int(t_val.item()), device=dev, dtype=torch.long)

        if step_i < reward_steps:
            for _ in range(K_ug):
                z_var = z_t.detach().clone().float().requires_grad_(True)
                dt = next(unet.parameters()).dtype
                z_sc = sched.scale_model_input(z_var.to(dt), t_train[0])
                eps_c = unet(z_sc, t_train,
                             encoder_hidden_states=c0.to(dt)).sample.float()
                z0_hat = _z0_hat_from_eps(sched, z_var, eps_c, t_train)
                pixels = _decode_for_reward(vae, z0_hat)
                Q = reward_model.score(pixels, prompt).mean()
                grad = torch.autograd.grad(-Q, z_var)[0]
                with torch.no_grad():
                    z_t = (z_var - eta_z * grad).to(dtype)

        with torch.no_grad():
            eps = _cfg_eps_sd15(unet, sched, z_t, t_train, c_u, c0, guidance_scale)
            z_t = sched.step(eps, int(t_train[0].item()), z_t).prev_sample

    return _dec(vae, z_t)[0]


def generate_ug(prompt, neg, models, reward_model, backbone, args, seed):
    kw = dict(
        guidance_scale=args.guidance,
        num_steps=args.steps,
        rho_Q=args.rho_Q,
        K_ug=args.K_ug,
        eta_z=args.eta_z,
        seed=seed,
    )
    if backbone == "sdxl":
        kw.update(height=1024, width=1024)
        return generate_ug_sdxl(prompt, neg, models, reward_model, **kw)
    kw.update(height=512, width=512)
    return generate_ug_sd15(prompt, neg, models, reward_model, **kw)


# ── generation over a split ──────────────────────────────────────────────────

def generate_split(
    prompts: List[str],
    models,
    reward_model,
    backbone: str,
    out_dir: str,
    args,
    seeds: Optional[List[int]] = None,
    skip_existing: bool = True,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    if seeds is None:
        seeds = [SEEDS[i % len(SEEDS)] for i in range(len(prompts))]
    paths = []
    for i, (p, s) in enumerate(tqdm(list(zip(prompts, seeds)), desc=f"UG eta={args.eta_z}")):
        path = os.path.join(out_dir, f"{i:05d}.png")
        if not (skip_existing and os.path.exists(path)):
            img = generate_ug(p, args.negative_prompt, models, reward_model, backbone, args, s)
            img.save(path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        paths.append(path)
    return paths


# ── scoring ──────────────────────────────────────────────────────────────────

def load_scorers(device: str) -> Dict:
    _repo = os.path.dirname(os.path.abspath(__file__))
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    scorers = {}
    print("  Loading pickscore...")
    from utils.pickscore_utils import Selector as PSSelector
    scorers["pickscore"] = PSSelector(device)
    print("  Loading hps...")
    from utils.hps_utils import Selector as HPSSelector
    scorers["hps"] = HPSSelector(device)
    print("  Loading clip...")
    from utils.clip_utils import Selector as CLIPSelector
    scorers["clip"] = CLIPSelector(device)
    print("  Loading aes...")
    from utils.aes_utils import Selector as AESSelector
    scorers["aes"] = AESSelector(device)
    return scorers


def score_images(paths: List[str], prompts: List[str], scorers: Dict) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {k: [] for k in scorers}
    for path, prompt in tqdm(list(zip(paths, prompts)), desc="Scoring"):
        img = Image.open(path).convert("RGB")
        for name, sel in scorers.items():
            results[name].append(float(sel.score([img], prompt)[0]))
    return results


def aggregate(scores: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: float(np.mean(v)) for k, v in scores.items()}


# ── tuning phase ─────────────────────────────────────────────────────────────

def run_tune(args):
    """Sweep eta_z on validation split and pick the best value per metric."""
    val_prompts, _ = load_parti_prompts_split()
    print(f"Val split: {len(val_prompts)} prompts")

    device = args.device
    dtype = torch.float16 if (args.dtype == "fp16" and device == "cuda") else torch.float32
    models = load_models(args.backbone, device, dtype)
    reward_model = load_reward_model(device)

    lr_dirs: Dict[float, str] = {}
    for eta in args.sweep:
        args.eta_z = eta
        out_dir = os.path.join(args.out_dir, "tune", f"eta_{eta:.0e}", "images")
        generate_split(val_prompts, models, reward_model, args.backbone, out_dir, args)
        lr_dirs[eta] = out_dir

    del models, reward_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scorers = load_scorers(args.score_device)
    results: Dict[float, Dict[str, float]] = {}
    for eta, d in lr_dirs.items():
        paths = [os.path.join(d, f"{i:05d}.png") for i in range(len(val_prompts))]
        results[eta] = aggregate(score_images(paths, val_prompts, scorers))
        print(f"  eta={eta}: {results[eta]}")

    tune_path = os.path.join(args.out_dir, "tune", "tune_results.json")
    os.makedirs(os.path.dirname(tune_path), exist_ok=True)
    with open(tune_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    best = {}
    for m in list(next(iter(results.values())).keys()):
        best_eta = max(results, key=lambda e: results[e][m])
        best[m] = {"eta_z": best_eta, "score": results[best_eta][m]}
    best_path = os.path.join(args.out_dir, "tune", "best_eta.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best eta per metric → {best_path}")
    for m, d in best.items():
        print(f"  {m}: best eta={d['eta_z']}, score={d['score']:.4f}")
    return best


# ── test phase ───────────────────────────────────────────────────────────────

def run_test(args):
    _, test_prompts = load_parti_prompts_split()
    print(f"Test split: {len(test_prompts)} prompts  eta_z={args.eta_z}")

    device = args.device
    dtype = torch.float16 if (args.dtype == "fp16" and device == "cuda") else torch.float32
    models = load_models(args.backbone, device, dtype)
    reward_model = load_reward_model(device)

    out_dir = os.path.join(args.out_dir, "test", f"eta_{args.eta_z:.0e}", "images")
    paths = generate_split(test_prompts, models, reward_model, args.backbone, out_dir, args)

    del models, reward_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scorers = load_scorers(args.score_device)
    raw = score_images(paths, test_prompts, scorers)
    agg = aggregate(raw)

    result = {
        "backbone": args.backbone,
        "eta_z": args.eta_z,
        "K_ug": args.K_ug,
        "rho_Q": args.rho_Q,
        "n_test": len(test_prompts),
        "scores": agg,
    }
    rpath = os.path.join(args.out_dir, "test", f"eta_{args.eta_z:.0e}", "results.json")
    os.makedirs(os.path.dirname(rpath), exist_ok=True)
    with open(rpath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Test results → {rpath}")
    print(json.dumps(agg, indent=2))
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Universal Guidance baseline")
    ap.add_argument("--backbone", choices=["sd15", "sdxl"], default="sdxl")
    ap.add_argument("--phase", choices=["tune", "test", "both"], default="tune")
    ap.add_argument("--eta_z", type=float, default=0.02,
                    help="Latent learning rate for UG gradient step")
    ap.add_argument("--sweep", type=float, nargs="+", default=ETA_Z_SWEEP,
                    help="eta_z values to sweep in tune phase")
    ap.add_argument("--K_ug", type=int, default=4,
                    help="Inner gradient steps per reward-active denoising step "
                         "(set to match PG-MAP NFE: K_ug=4 for PG-MAP K=2 T=50)")
    ap.add_argument("--rho_Q", type=float, default=0.3,
                    help="Fraction of steps where reward is active (same as PG-MAP)")
    ap.add_argument("--guidance", type=float, default=None,
                    help="CFG scale (default: 7.5 for sd15, 5.0 for sdxl)")
    ap.add_argument("--steps", type=int, default=None,
                    help="DDIM steps (default: 30 for sd15, 50 for sdxl)")
    ap.add_argument("--negative_prompt", type=str, default="blurry, low quality")
    ap.add_argument("--out_dir", type=str, default="eval_results/ug")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--score_device", type=str, default="cuda")
    args = ap.parse_args()

    if args.steps is None:
        args.steps = 30 if args.backbone == "sd15" else 50
    if args.guidance is None:
        args.guidance = 7.5 if args.backbone == "sd15" else 5.0

    os.makedirs(args.out_dir, exist_ok=True)

    if args.phase == "tune":
        run_tune(args)
    elif args.phase == "test":
        run_test(args)
    else:  # both
        best = run_tune(args)
        args.eta_z = best.get("pickscore", {}).get("eta_z", args.eta_z)
        print(f"\n--- Test at best PickScore eta_z={args.eta_z} ---")
        run_test(args)


if __name__ == "__main__":
    main()
