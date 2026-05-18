#!/usr/bin/env python3
"""
PG-MAP Full Evaluation Pipeline
==================================

CLI entry point for running the complete PartiPrompts evaluation.
Supports all methods, ablation sweeps, statistical testing, and
paper-ready output generation.

Evaluation workflow:
    1. Load models (UNet, VAE, text encoder, scheduler, reward model, scorers)
    2. Load PartiPrompts benchmark
    3. For each method: generate images, score pairs, compute statistics
    4. Output scores.jsonl, scores_summary.json, wilcoxon.json, bootstrap.json
    5. Generate LaTeX tables

Methods (controlled via --methods flag):
    baseline     Standard DDIM+CFG (no refinement)
    mapc         MAP-c (conditioning-only MAP)
    reward_z     Reward-z (latent-only + reward)
    joint_cz     Joint (c,z) without reward
    pgmap        PG-MAP (full method)
    ae           Attend-and-Excite (SD1.5 only)

Example usage:
    # Full evaluation with SD1.5
    python pgmap_eval.py --backbone sd15 --num_prompts 1632 --seed 123 \\
        --methods baseline mapc pgmap --score --out_dir eval_results/sd15

    # Quick sanity check
    python pgmap_eval.py --backbone sd15 --num_prompts 8 --seed 42 \\
        --methods baseline pgmap --score --out_dir eval_quick

    # Ablation sweep
    python pgmap_eval.py --backbone sd15 --num_prompts 200 --seed 123 \\
        --ablation lambda_sweep --out_dir eval_results/ablation/lambda_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from pgmap_config import (
    PGMAPConfig,
    PriorConfig,
    RewardConfig,
    RefinementConfig,
    SemanticConfig,
    SDPGMAPConfig,
    baseline_config,
    joint_cz_config,
    mapc_config,
    reward_z_config,
    sd15_defaults,
    sdxl_defaults,
    sdpgmap_sd15_defaults,
    sdpgmap_sdxl_defaults,
)
from pgmap_reward import FrozenRewardModel


# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------
# Prompt loading
# -----------------------------------------------------------------------

def _clean_prompt(s: str) -> str:
    s = s.replace("<|endoftext|>", " ")
    return " ".join(s.split()).strip()


def load_parti_prompts(seed: int, n: int) -> List[str]:
    """Load n prompts from the PartiPrompts benchmark.

    Uses the HuggingFace datasets library to load nateraw/parti-prompts.
    Prompts are shuffled with a fixed seed for reproducibility.

    Args:
        seed: Random seed for prompt selection.
        n:    Number of prompts to load (use -1 for all ~1632).

    Returns:
        List of cleaned prompt strings.
    """
    from datasets import load_dataset
    ds = load_dataset("nateraw/parti-prompts", split="train")
    key = "Prompt" if "Prompt" in ds.features else list(ds.features.keys())[0]

    all_prompts = [_clean_prompt(x) for x in ds[key] if isinstance(x, str) and x.strip()]
    if not all_prompts:
        raise RuntimeError("No prompts found in dataset.")

    if n <= 0 or n >= len(all_prompts):
        n = len(all_prompts)

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(all_prompts), size=n, replace=False)
    return [all_prompts[i] for i in sel]


# -----------------------------------------------------------------------
# Method configurations
# -----------------------------------------------------------------------

def get_method_config(method: str, backbone: str, args) -> PGMAPConfig:
    """Create a PGMAPConfig for the given method name.

    Applies CLI overrides (steps, guidance, learning rates, etc.)
    on top of the method-specific defaults.

    Args:
        method:   One of: baseline, mapc, reward_z, joint_cz, pgmap.
        backbone: "sd15" or "sdxl".
        args:     Parsed CLI arguments for hyperparameter overrides.

    Returns:
        Configured PGMAPConfig.
    """
    config_map = {
        "baseline": baseline_config,
        "mapc": mapc_config,
        "reward_z": reward_z_config,
        "joint_cz": joint_cz_config,
        "pgmap": sd15_defaults if backbone == "sd15" else sdxl_defaults,
    }

    # sdpgmap returns SDPGMAPConfig, handled separately in generate_image
    if method == "sdpgmap":
        base = sdpgmap_sd15_defaults() if backbone == "sd15" else sdpgmap_sdxl_defaults()
        # Apply CLI overrides to the inner PGMAPConfig
        p = base.pgmap
        p.num_steps = args.steps
        p.guidance_scale = args.guidance
        p.height = args.height
        p.width = args.width
        p.rho = args.rho
        p.refinement.K = args.K
        p.refinement.eta_c = args.eta_c
        p.refinement.eta_z = args.eta_z
        p.prior.sigma_c = args.sigma_c
        p.prior.gamma = args.gamma
        p.reward.lambda_reward = args.lambda_reward
        p.reward.rho_Q = args.rho_Q
        p.reward.model_name = args.reward_model
        p.reward.grad_norm_strategy = args.grad_norm_strategy
        p.reward.lambda_ramp = getattr(args, "lambda_ramp", False)
        p.refinement.optimizer = getattr(args, "optimizer", "sgd")
        # SD-specific overrides
        base.semantic.k_early  = getattr(args, "k_early",  base.semantic.k_early)
        base.semantic.k_mid    = getattr(args, "k_mid",    base.semantic.k_mid)
        base.semantic.k_late   = getattr(args, "k_late",   base.semantic.k_late)
        base.semantic.k_c      = getattr(args, "k_c",      base.semantic.k_c)
        base.semantic.alpha_sem         = getattr(args, "alpha_sem",         base.semantic.alpha_sem)
        base.semantic.beta_nonsem       = getattr(args, "beta_nonsem",       base.semantic.beta_nonsem)
        base.semantic.cos_gate_threshold = getattr(args, "cos_gate_threshold", base.semantic.cos_gate_threshold)
        base.semantic.proj_scale         = getattr(args, "proj_scale",         base.semantic.proj_scale)
        return base  # NOTE: returns SDPGMAPConfig, not PGMAPConfig

    if method in config_map:
        cfg = config_map[method](backbone) if method != "pgmap" else config_map[method]()
    else:
        cfg = sd15_defaults() if backbone == "sd15" else sdxl_defaults()

    # Apply CLI overrides
    cfg.num_steps = args.steps
    cfg.guidance_scale = args.guidance
    cfg.height = args.height
    cfg.width = args.width
    cfg.rho = args.rho

    cfg.refinement.K = args.K
    cfg.refinement.eta_c = args.eta_c
    cfg.refinement.eta_z = args.eta_z

    cfg.prior.sigma_c = args.sigma_c
    cfg.prior.gamma = args.gamma

    cfg.reward.lambda_reward = args.lambda_reward
    cfg.reward.rho_Q = args.rho_Q
    cfg.reward.model_name = args.reward_model
    cfg.reward.grad_norm_strategy = args.grad_norm_strategy
    cfg.reward.lambda_ramp = getattr(args, "lambda_ramp", False)
    cfg.refinement.optimizer = getattr(args, "optimizer", "sgd")

    return cfg


# -----------------------------------------------------------------------
# Image generation dispatch
# -----------------------------------------------------------------------

def generate_image(
    method: str,
    prompt: str,
    negative_prompt: str,
    config: PGMAPConfig,
    models: Any,
    reward_model: Optional[FrozenRewardModel],
    backbone: str,
    ae_pipeline: Any = None,
    subspace_bank: Any = None,
) -> Image.Image:
    """Generate a single image for the given method.

    Dispatches to the appropriate pipeline based on method and backbone.

    Args:
        method:          Method name.
        prompt:          Text prompt.
        negative_prompt: Negative prompt for CFG.
        config:          PGMAPConfig with seed already set.
        models:          Model bundle (SD15Models or SDXLModels).
        reward_model:    FrozenRewardModel (or None for methods without reward).
        backbone:        "sd15" or "sdxl".
        ae_pipeline:     Pre-loaded Attend-and-Excite pipeline (SD1.5 only).

    Returns:
        Generated PIL Image.
    """
    if method == "ae":
        return _generate_attend_and_excite(
            ae_pipeline, prompt, negative_prompt, config
        )

    # Semantic Direction-Constrained PG-MAP
    if method == "sdpgmap":
        if backbone == "sd15":
            from pgmap_sd15_sd import generate_sd15_sdpgmap
            img, _ = generate_sd15_sdpgmap(
                prompt, negative_prompt,
                models=models, cfg=config,
                reward_model=reward_model if config.pgmap.use_reward else None,
                subspace_bank=subspace_bank,
            )
        else:
            from pgmap_sdxl_sd import generate_sdxl_sdpgmap
            img, _ = generate_sdxl_sdpgmap(
                prompt, negative_prompt,
                models=models, cfg=config,
                reward_model=reward_model if config.pgmap.use_reward else None,
                subspace_bank=subspace_bank,
            )
        return img

    if backbone == "sd15":
        from pgmap_sd15 import generate_sd15_pgmap
        img, _ = generate_sd15_pgmap(
            prompt, negative_prompt,
            models=models, config=config,
            reward_model=reward_model if config.use_reward else None,
        )
    else:
        from pgmap_sdxl import generate_sdxl_pgmap
        img, _ = generate_sdxl_pgmap(
            prompt, negative_prompt,
            models=models, config=config,
            reward_model=reward_model if config.use_reward else None,
        )

    return img


# -----------------------------------------------------------------------
# Attend-and-Excite baseline
# -----------------------------------------------------------------------

def load_attend_and_excite(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load the Attend-and-Excite pipeline from diffusers.

    Only available for SD1.5. Uses spaCy for automatic noun detection.

    Returns:
        (pipeline, nlp_model) tuple.
    """
    from diffusers import StableDiffusionAttendAndExcitePipeline

    pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
        model_id, torch_dtype=dtype,
    ).to(device)

    # Load spaCy for noun extraction
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        nlp = None
        print("[WARN] spaCy not available; A&E will use fallback token indices.")

    return pipe, nlp


def _get_noun_indices(prompt: str, tokenizer, nlp=None) -> List[int]:
    """Extract noun token indices for Attend-and-Excite.

    Uses spaCy POS tagging when available, otherwise falls back to
    selecting every other content word.

    Args:
        prompt:    Text prompt.
        tokenizer: CLIP tokenizer from the pipeline.
        nlp:       spaCy language model (optional).

    Returns:
        List of 1-indexed token positions for subject nouns.
    """
    if nlp is not None:
        doc = nlp(prompt)
        nouns = {tok.text.lower() for tok in doc if tok.pos_ in ("NOUN", "PROPN")}
        tokens = tokenizer.tokenize(prompt)
        indices = [
            i + 1  # +1 for BOS token
            for i, tok in enumerate(tokens)
            if tok.replace("</w>", "").lower() in nouns
        ]
        return indices if indices else [1]

    # Fallback: use first 3 non-stopword tokens
    tokens = tokenizer.tokenize(prompt)
    stopwords = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "is", "with"}
    indices = [
        i + 1
        for i, tok in enumerate(tokens)
        if tok.replace("</w>", "").lower() not in stopwords
    ]
    return indices[:3] if indices else [1]


def _generate_attend_and_excite(
    pipe_and_nlp: Tuple,
    prompt: str,
    negative_prompt: str,
    config: PGMAPConfig,
) -> Image.Image:
    """Generate an image using Attend-and-Excite.

    Args:
        pipe_and_nlp: (pipeline, nlp_model) from load_attend_and_excite().
        prompt:       Text prompt.
        negative_prompt: Negative prompt.
        config:       PGMAPConfig (uses seed, steps, guidance, height, width).

    Returns:
        Generated PIL Image.
    """
    pipe, nlp = pipe_and_nlp
    token_indices = _get_noun_indices(prompt, pipe.tokenizer, nlp)

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            token_indices=[token_indices],
            num_inference_steps=config.num_steps,
            guidance_scale=config.guidance_scale,
            height=config.height,
            width=config.width,
            generator=torch.Generator(device=pipe.device).manual_seed(config.seed),
            max_iter_to_alter=int(0.8 * config.num_steps),
        )
        return result.images[0]
    except Exception as e:
        print(f"[WARN] A&E failed for '{prompt[:50]}...': {e}")
        # Fallback to standard generation
        from diffusers import StableDiffusionPipeline
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=config.num_steps,
            guidance_scale=config.guidance_scale,
            height=config.height,
            width=config.width,
            generator=torch.Generator(device=pipe.device).manual_seed(config.seed),
        )
        return result.images[0]


# -----------------------------------------------------------------------
# Scoring pipeline
# -----------------------------------------------------------------------

def _load_utils_scorers(score_device: str,
                        enable_pickscore: bool,
                        enable_aes: bool,
                        enable_hps: bool,
                        enable_clip: bool) -> Dict[str, Any]:
    """Load scorers directly from utils/, one at a time to minimize peak VRAM."""
    import sys
    _utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
    if _utils_dir not in sys.path:
        sys.path.insert(0, _utils_dir)

    scorers = {}
    if enable_pickscore:
        print("  Loading PickScore scorer...")
        from utils.pickscore_utils import Selector as PSSelector
        scorers["pickscore"] = PSSelector(score_device)
    if enable_aes:
        print("  Loading Aesthetic scorer...")
        from utils.aes_utils import Selector as AesSelector
        scorers["aes"] = AesSelector(score_device)
    if enable_hps:
        print("  Loading HPS scorer...")
        from utils.hps_utils import Selector as HPSSelector
        scorers["hps"] = HPSSelector(score_device)
    if enable_clip:
        print("  Loading CLIP scorer...")
        from utils.clip_utils import Selector as CLIPSelector
        scorers["clip"] = CLIPSelector(score_device)
    return scorers


def score_method_pair(
    ref_img_dir: str,
    method_img_dir: str,
    prompts: List[str],
    scorers: Dict[str, Any],
    out_dir: str,
    seeds: List[int],
) -> List[Dict]:
    """Score all image pairs from disk and write scores.jsonl.

    Loads images from saved PNG files rather than keeping them in memory,
    so generation models can remain on GPU without conflicting with scorers.

    Args:
        ref_img_dir:    Directory of baseline images (00000.png, 00001.png, ...).
        method_img_dir: Directory of method images (same naming).
        prompts:        Text prompts.
        scorers:        Dict of {metric_name: Selector} from utils/.
        out_dir:        Output directory for scores.jsonl.
        seeds:          Generation seeds (for logging).

    Returns:
        List of score row dicts.
    """
    score_path = os.path.join(out_dir, "scores.jsonl")
    rows = []

    with open(score_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(len(prompts)), desc="Scoring"):
            ref_img = Image.open(os.path.join(ref_img_dir, f"{i:05d}.png")).convert("RGB")
            tr_img  = Image.open(os.path.join(method_img_dir, f"{i:05d}.png")).convert("RGB")
            ims = [ref_img, tr_img]

            metrics: Dict[str, Any] = {}
            for name, sel in scorers.items():
                s = sel.score(ims, prompts[i])
                metrics[name] = {
                    "ref":     float(s[0]),
                    "trained": float(s[1]),
                    "delta":   float(s[1]) - float(s[0]),
                }

            row = {
                "i": i,
                "seed": int(seeds[i]),
                "prompt": prompts[i],
                "metrics": metrics,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return rows


def _aggregate(scores_rows: List[Dict]) -> Dict:
    """Aggregate per-row scores into summary statistics."""
    metrics: Dict[str, Any] = {}
    for row in scores_rows:
        for k, v in row.get("metrics", {}).items():
            metrics.setdefault(k, {"ref": [], "trained": [], "delta": [], "win": 0, "n": 0})
            metrics[k]["ref"].append(v["ref"])
            metrics[k]["trained"].append(v["trained"])
            metrics[k]["delta"].append(v["delta"])
            metrics[k]["win"] += 1 if v["delta"] > 0 else 0
            metrics[k]["n"] += 1

    summary = {}
    for k, d in metrics.items():
        ref = np.array(d["ref"], dtype=np.float64)
        tr  = np.array(d["trained"], dtype=np.float64)
        de  = np.array(d["delta"],   dtype=np.float64)
        summary[k] = {
            "mean_ref":     float(ref.mean()) if len(ref) else None,
            "mean_trained": float(tr.mean())  if len(tr)  else None,
            "mean_delta":   float(de.mean())  if len(de)  else None,
            "win_rate":     float(d["win"] / max(d["n"], 1)),
            "n":            int(d["n"]),
        }
    return summary


# -----------------------------------------------------------------------
# Statistical testing
# -----------------------------------------------------------------------

def compute_wilcoxon(rows: List[Dict], metric: str = "pickscore") -> Dict:
    """Run paired Wilcoxon signed-rank test.

    Tests H0: median(PG-MAP score - baseline score) = 0.
    Alternative: PG-MAP > baseline (one-sided).

    Args:
        rows:   Score rows from score_method_pair().
        metric: Which metric to test (pickscore, hps, aes, clip).

    Returns:
        Dict with statistic, p_value, significance flags, etc.
    """
    from scipy.stats import wilcoxon

    ref_scores = []
    tr_scores = []
    for row in rows:
        m = row.get("metrics", {}).get(metric, {})
        if "ref" in m and "trained" in m:
            ref_scores.append(m["ref"])
            tr_scores.append(m["trained"])

    if len(ref_scores) < 10:
        return {"statistic": None, "p_value": 1.0, "n": len(ref_scores), "note": "too_few_samples"}

    diffs = np.array(tr_scores) - np.array(ref_scores)
    nonzero_mask = diffs != 0

    if nonzero_mask.sum() < 10:
        return {"statistic": None, "p_value": 1.0, "n": len(diffs), "note": "too_few_nonzero"}

    stat, p_val = wilcoxon(
        diffs[nonzero_mask], alternative="greater"
    )
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "n": int(len(diffs)),
        "n_nonzero": int(nonzero_mask.sum()),
        "significant_at_001": bool(p_val < 0.001),
        "significant_at_005": bool(p_val < 0.005),
        "significant_at_05": bool(p_val < 0.05),
    }


def bootstrap_win_rate_ci(
    rows: List[Dict],
    metric: str = "pickscore",
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Compute bootstrap 95% CI for win rate.

    Win rate = fraction of prompts where method > baseline.

    Args:
        rows:             Score rows.
        metric:           Metric name.
        n_resamples:      Number of bootstrap resamples.
        confidence_level: CI level (default 0.95).
        seed:             Random seed for bootstrap.

    Returns:
        Dict with win_rate, ci_low, ci_high, n.
    """
    from scipy.stats import bootstrap

    wins = []
    for row in rows:
        m = row.get("metrics", {}).get(metric, {})
        if "delta" in m:
            wins.append(1.0 if m["delta"] > 0 else 0.0)

    if not wins:
        return {"win_rate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    wins_arr = np.array(wins)
    win_rate = float(wins_arr.mean())

    if len(wins_arr) < 10:
        return {"win_rate": win_rate, "ci_low": win_rate, "ci_high": win_rate, "n": len(wins)}

    res = bootstrap(
        (wins_arr,),
        statistic=np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=np.random.default_rng(seed),
        method="percentile",
    )
    return {
        "win_rate": win_rate,
        "ci_low": float(res.confidence_interval.low),
        "ci_high": float(res.confidence_interval.high),
        "n": len(wins),
    }


def compute_all_statistics(
    rows: List[Dict],
    out_dir: str,
    metrics: List[str] = None,
):
    """Compute and save all statistical tests.

    Runs Wilcoxon and bootstrap CI for each metric, saves results to
    wilcoxon.json and bootstrap.json alongside the score summary.

    Args:
        rows:    Score rows.
        out_dir: Output directory.
        metrics: List of metric names. Auto-detected if None.
    """
    if metrics is None:
        # Detect available metrics from first row
        first_metrics = rows[0].get("metrics", {}) if rows else {}
        metrics = list(first_metrics.keys())

    # Summary
    summary = _aggregate(rows)
    with open(os.path.join(out_dir, "scores_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Wilcoxon
    wilcoxon_results = {}
    for m in metrics:
        wilcoxon_results[m] = compute_wilcoxon(rows, m)
    with open(os.path.join(out_dir, "wilcoxon.json"), "w") as f:
        json.dump(wilcoxon_results, f, indent=2)

    # Bootstrap CIs
    bootstrap_results = {}
    for m in metrics:
        bootstrap_results[m] = bootstrap_win_rate_ci(rows, m)
    with open(os.path.join(out_dir, "bootstrap.json"), "w") as f:
        json.dump(bootstrap_results, f, indent=2)


# -----------------------------------------------------------------------
# Wall-clock timing
# -----------------------------------------------------------------------

def time_generation(
    method: str,
    gen_fn: Callable,
    prompt: str,
    negative_prompt: str,
    config: PGMAPConfig,
    models: Any,
    reward_model: Optional[FrozenRewardModel],
    backbone: str,
    n_trials: int = 50,
    warmup: int = 3,
    ae_pipeline: Any = None,
) -> Dict:
    """Time a generation function over n_trials (excluding warmup).

    Args:
        method:          Method name passed to gen_fn (e.g. "pgmap", "baseline").
        gen_fn:          The generate_image function.
        prompt:          Test prompt.
        negative_prompt: Negative prompt.
        config:          PGMAPConfig.
        models:          Model bundle.
        reward_model:    Reward model.
        backbone:        "sd15" or "sdxl".
        n_trials:        Number of timed trials.
        warmup:          Number of warmup runs to skip.

    Returns:
        Dict with mean_sec, std_sec, median_sec, n_trials.
    """
    for _ in range(warmup):
        gen_fn(method, prompt, negative_prompt, config, models, reward_model, backbone, ae_pipeline=ae_pipeline)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(n_trials):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen_fn(method, prompt, negative_prompt, config, models, reward_model, backbone, ae_pipeline=ae_pipeline)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "mean_sec": float(np.mean(times)),
        "std_sec": float(np.std(times)),
        "median_sec": float(np.median(times)),
        "n_trials": n_trials,
    }


# -----------------------------------------------------------------------
# Ablation sweeps
# -----------------------------------------------------------------------

ABLATION_CONFIGS = {
    "eta_c_sweep": {
        "variable": "eta_c",
        "values": [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        "default_idx": 2,  # 1e-4 current default
    },
    "lambda_sweep": {
        "variable": "lambda_reward",
        "values": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
        "default_idx": 3,  # 0.1 is the paper default
    },
    "gamma_sweep": {
        "variable": "gamma",
        "values": [0.0, 0.1, 0.3, 0.5, 1.0],
        "default_idx": 3,  # 0.5
    },
    "K_sweep": {
        "variable": "K",
        "values": [1, 2, 3, 5],
        "default_idx": 0,  # K=1
    },
    "reward_model_sweep": {
        "variable": "reward_model",
        "values": ["pickscore", "hps", "clip"],
        "default_idx": 0,  # pickscore
    },
    "optimizer_sweep": {
        "variable": "optimizer",
        "values": ["sgd_K1", "adam_K1", "adam_K2", "adam_K3"],
        "default_idx": 0,  # sgd K=1 (paper default)
    },
}


def run_ablation(
    ablation_name: str,
    backbone: str,
    prompts: List[str],
    seeds: List[int],
    models: Any,
    reward_model: Optional[FrozenRewardModel],
    args,
    out_dir: str,
):
    """Run a single ablation sweep.

    Generates images for each value of the swept variable (all other
    params at defaults) and scores against the baseline.

    Args:
        ablation_name: Key into ABLATION_CONFIGS.
        backbone:      "sd15" or "sdxl".
        prompts:       List of prompts (typically 200 for ablations).
        seeds:         Generation seeds.
        models:        Model bundle.
        reward_model:  FrozenRewardModel.
        args:          CLI arguments.
        out_dir:       Output directory for this ablation.
    """
    abl = ABLATION_CONFIGS[ablation_name]
    variable = abl["variable"]
    values = abl["values"]

    print(f"\n{'='*60}")
    print(f"Ablation: {ablation_name} ({variable})")
    print(f"Values: {values}")
    print(f"{'='*60}\n")

    # Generate baseline images once
    base_cfg = baseline_config(backbone)
    base_cfg.num_steps = args.steps
    base_cfg.guidance_scale = args.guidance
    base_cfg.height = args.height
    base_cfg.width = args.width

    baseline_dir = os.path.join(out_dir, "baseline", "images")
    os.makedirs(baseline_dir, exist_ok=True)

    for i, (p, s) in enumerate(tqdm(list(zip(prompts, seeds)), desc="Baseline")):
        base_cfg.seed = s
        img = generate_image("baseline", p, args.negative_prompt, base_cfg,
                             models, None, backbone)
        img.save(os.path.join(baseline_dir, f"{i:05d}.png"))

    # Sweep each value — Phase 1: generate all
    val_img_dirs: Dict[str, tuple] = {}
    for val in values:
        val_name = f"{variable}_{val}"
        val_dir = os.path.join(out_dir, val_name)
        method_img_dir = os.path.join(val_dir, "images")
        os.makedirs(method_img_dir, exist_ok=True)

        cfg = get_method_config("pgmap", backbone, args)

        # Apply the swept variable
        if variable == "eta_c":
            cfg.refinement.eta_c = val
        elif variable == "lambda_reward":
            cfg.reward.lambda_reward = val
        elif variable == "gamma":
            cfg.prior.gamma = val
        elif variable == "K":
            cfg.refinement.K = val
        elif variable == "reward_model":
            cfg.reward.model_name = val
        elif variable == "optimizer":
            # val is like "sgd_K1", "adam_K2"
            parts = val.split("_K")
            opt   = parts[0]                       # "sgd" or "adam"
            k     = int(parts[1]) if len(parts) > 1 else 1
            cfg.refinement.optimizer = opt
            cfg.refinement.K = k

        for i, (p, s) in enumerate(tqdm(list(zip(prompts, seeds)), desc=f"{val_name}")):
            cfg.seed = s
            rm = reward_model
            # For reward model sweep, need to reload the right model
            if variable == "reward_model" and val != args.reward_model:
                rm = FrozenRewardModel(val, device=str(models.device))

            img = generate_image("pgmap", p, args.negative_prompt, cfg,
                                 models, rm, backbone)
            img.save(os.path.join(method_img_dir, f"{i:05d}.png"))

        val_img_dirs[val_name] = (val_dir, method_img_dir)

    # Phase 2: free generation models, load scorers, score from disk
    if args.score and val_img_dirs:
        import gc
        print("\n--- Freeing generation models from GPU ---")
        del models, reward_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Loading scorers from utils/...")
        scorers = _load_utils_scorers(
            score_device=args.score_device,
            enable_pickscore=not args.no_pickscore,
            enable_aes=not args.no_aes,
            enable_hps=not args.no_hps,
            enable_clip=not args.no_clip,
        )
        for val_name, (val_dir, method_img_dir) in val_img_dirs.items():
            rows = score_method_pair(baseline_dir, method_img_dir, prompts, scorers, val_dir, seeds)
            compute_all_statistics(rows, val_dir)
            print(f"[OK] {val_name}: scores saved to {val_dir}")


# -----------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="PG-MAP Full Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full SD1.5 evaluation
  python pgmap_eval.py --backbone sd15 --num_prompts 1632 --seed 123 \\
      --methods baseline mapc pgmap --score --out_dir eval_results/sd15

  # Quick test
  python pgmap_eval.py --backbone sd15 --num_prompts 8 --methods baseline pgmap

  # Ablation sweep
  python pgmap_eval.py --backbone sd15 --num_prompts 200 \\
      --ablation lambda_sweep --out_dir eval_results/ablation/lambda_sweep
        """,
    )

    # --- Output ---
    ap.add_argument("--out_dir", type=str, default="eval_pgmap",
                    help="Output directory for all results")

    # --- Model ---
    ap.add_argument("--backbone", choices=["sd15", "sdxl"], default="sd15",
                    help="Backbone model: sd15 or sdxl")
    ap.add_argument("--model_id", type=str, default=None,
                    help="HuggingFace model ID (auto-detected from backbone if None)")

    # --- Prompts ---
    ap.add_argument("--num_prompts", type=int, default=64,
                    help="Number of PartiPrompts to evaluate (-1 for all)")
    ap.add_argument("--prompt_file", type=str, default=None,
                    help="JSON file with custom prompts (list of strings). "
                         "Overrides --num_prompts and PartiPrompts loading.")
    ap.add_argument("--seed", type=int, default=123,
                    help="Master seed for prompt selection and generation")

    # --- Device ---
    ap.add_argument("--device", type=str, default="cuda",
                    help="Device for generation (cuda or cpu)")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"],
                    help="Model precision")

    # --- Methods ---
    ap.add_argument("--methods", nargs="+",
                    default=["baseline", "pgmap"],
                    choices=["baseline", "mapc", "reward_z", "joint_cz", "pgmap", "ae", "sdpgmap"],
                    help="Methods to evaluate")

    # --- Generation hyperparameters ---
    ap.add_argument("--steps", type=int, default=None,
                    help="DDIM steps (default: 30 for sd15, 50 for sdxl)")
    ap.add_argument("--guidance", type=float, default=None,
                    help="CFG scale (default: 7.5 for sd15, 5.0 for sdxl)")
    ap.add_argument("--height", type=int, default=None,
                    help="Image height (default: 512 for sd15, 1024 for sdxl)")
    ap.add_argument("--width", type=int, default=None,
                    help="Image width (default: 512 for sd15, 1024 for sdxl)")
    ap.add_argument("--negative_prompt", type=str, default="blurry, low quality",
                    help="Negative prompt for CFG")

    # --- PG-MAP hyperparameters ---
    ap.add_argument("--rho", type=float, default=0.4,
                    help="Fraction of steps for refinement")
    ap.add_argument("--rho_Q", type=float, default=0.3,
                    help="Fraction of steps for reward")
    ap.add_argument("--K", type=int, default=1,
                    help="Inner gradient steps per denoising step")
    ap.add_argument("--eta_c", type=float, default=1e-4,
                    help="Learning rate for conditioning c")
    ap.add_argument("--eta_z", type=float, default=0.005,
                    help="Learning rate for latent z_t")
    ap.add_argument("--sigma_c", type=float, default=1.0,
                    help="Conditioning prior std")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="Latent prior scale factor")
    ap.add_argument("--lambda_reward", type=float, default=0.1,
                    help="Reward weight")
    ap.add_argument("--reward_model", type=str, default="pickscore",
                    choices=["pickscore", "hps", "clip", "aesthetic", "imagereward"],
                    help="Reward model to use")
    ap.add_argument("--grad_norm_strategy", type=str, default="unit",
                    choices=["unit", "adaptive", "raw"],
                    help="Gradient normalization strategy for reward")
    ap.add_argument("--optimizer", type=str, default="sgd",
                    choices=["sgd", "adam"],
                    help="Inner-loop optimizer: sgd (original) or adam (with persistent momentum)")
    ap.add_argument("--lambda_ramp", action="store_true",
                    help="Ramp lambda from 0 → lambda_reward over reward-active steps")

    # --- SD-PG-MAP (semantic direction-constrained) hyperparameters ---
    ap.add_argument("--k_early", type=int, default=8,
                    help="[sdpgmap] Semantic subspace components for early stage")
    ap.add_argument("--k_mid", type=int, default=16,
                    help="[sdpgmap] Semantic subspace components for middle stage")
    ap.add_argument("--k_late", type=int, default=32,
                    help="[sdpgmap] Semantic subspace components for late stage")
    ap.add_argument("--k_c", type=int, default=16,
                    help="[sdpgmap] Semantic subspace components for c embedding")
    ap.add_argument("--alpha_sem", type=float, default=2.0,
                    help="[sdpgmap] Anisotropic prior variance along semantic dirs")
    ap.add_argument("--beta_nonsem", type=float, default=0.5,
                    help="[sdpgmap] Anisotropic prior variance perpendicular to subspace")
    ap.add_argument("--cos_gate_threshold", type=float, default=0.1,
                    help="[sdpgmap] Min cosine-similarity for projection gate (0=always project)")
    ap.add_argument("--proj_scale", type=float, default=0.3,
                    help="[sdpgmap] Scale factor for projected gradient (1.0=no scaling)")
    ap.add_argument("--subspace_path", type=str, default=None,
                    help="[sdpgmap] Path to precomputed subspace directory (offline mode). "
                         "If not given, falls back to online per-image estimation.")

    # --- Scoring ---
    ap.add_argument("--score", action="store_true",
                    help="Run automatic scoring after generation")
    ap.add_argument("--score_device", type=str, default="cuda",
                    help="Device for scoring models")
    ap.add_argument("--no_pickscore", action="store_true")
    ap.add_argument("--no_aes", action="store_true")
    ap.add_argument("--no_hps", action="store_true")
    ap.add_argument("--no_clip", action="store_true")

    # --- Ablation ---
    ap.add_argument("--ablation", type=str, default=None,
                    choices=list(ABLATION_CONFIGS.keys()),
                    help="Run ablation sweep instead of standard eval")

    # --- Timing ---
    ap.add_argument("--timing", action="store_true",
                    help="Run wall-clock timing benchmark")
    ap.add_argument("--timing_trials", type=int, default=50,
                    help="Number of timing trials")

    args = ap.parse_args()

    # --- Apply backbone defaults ---
    if args.model_id is None:
        args.model_id = (
            "runwayml/stable-diffusion-v1-5" if args.backbone == "sd15"
            else "stabilityai/stable-diffusion-xl-base-1.0"
        )
    if args.steps is None:
        args.steps = 30 if args.backbone == "sd15" else 50
    if args.guidance is None:
        args.guidance = 7.5 if args.backbone == "sd15" else 5.0
    if args.height is None:
        args.height = 512 if args.backbone == "sd15" else 1024
    if args.width is None:
        args.width = 512 if args.backbone == "sd15" else 1024

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Save config
    config_log = {k: v for k, v in vars(args).items()}
    config_log["torch_version"] = torch.__version__
    config_log["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        config_log["gpu_name"] = torch.cuda.get_device_name(0)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config_log, f, indent=2)

    device = args.device
    dtype = torch.float16 if (args.dtype == "fp16" and device == "cuda") else torch.float32

    # --- Load prompts ---
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        print(f"Loading {args.num_prompts} prompts from PartiPrompts...")
        prompts = load_parti_prompts(args.seed, args.num_prompts)
        print(f"Loaded {len(prompts)} prompts.")
    seeds = [args.seed + i for i in range(len(prompts))]

    # --- Unzip utils if needed ---
    utils_dir = os.path.join(os.path.dirname(__file__), "utils")
    utils_zip = os.path.join(os.path.dirname(__file__), "utils.zip")
    if not os.path.isdir(utils_dir) and os.path.isfile(utils_zip):
        import zipfile
        print("Unzipping utils.zip...")
        with zipfile.ZipFile(utils_zip, "r") as z:
            z.extractall(os.path.dirname(__file__))

    # --- Load models ---
    print(f"Loading {args.backbone.upper()} models from {args.model_id}...")
    if args.backbone == "sd15":
        from pgmap_sd15 import load_sd15_models
        models = load_sd15_models(args.model_id, device, dtype)
    else:
        from pgmap_sdxl import load_sdxl_models
        models = load_sdxl_models(args.model_id, device, dtype)

    # --- Load reward model ---
    needs_reward = any(m in ["pgmap", "reward_z", "sdpgmap"] for m in args.methods) or args.ablation
    reward_model = None
    if needs_reward:
        print(f"Loading reward model: {args.reward_model}...")
        reward_model = FrozenRewardModel(args.reward_model, device=device)

    # --- Load precomputed subspace bank (offline SD-PG-MAP) ---
    subspace_bank = None
    if "sdpgmap" in args.methods and args.subspace_path is not None:
        from pgmap_core_sd import PrecomputedSubspaceBank
        print(f"Loading precomputed subspace bank from: {args.subspace_path}")
        subspace_bank = PrecomputedSubspaceBank(args.subspace_path)
        print(f"  Bank loaded (metadata: {subspace_bank.metadata.get('n_calib', '?')} calib prompts)")

    # --- Load A&E pipeline if needed ---
    ae_pipeline = None
    if "ae" in args.methods and args.backbone == "sd15":
        print("Loading Attend-and-Excite pipeline...")
        ae_pipeline = load_attend_and_excite(args.model_id, device, dtype)

    # --- Ablation mode ---
    if args.ablation:
        run_ablation(
            args.ablation, args.backbone, prompts, seeds,
            models, reward_model, args, args.out_dir,
        )
        print(f"\n[DONE] Ablation results saved to: {args.out_dir}")
        return

    # ===================================================================
    # PHASE 1: Generate all images (all methods), save to disk
    # Generation models stay on GPU; no scorers loaded yet.
    # ===================================================================

    # Step 1: Generate baseline images
    baseline_img_dir = None
    if "baseline" in args.methods or args.score:
        base_cfg = get_method_config("baseline", args.backbone, args)
        baseline_dir = os.path.join(args.out_dir, "baseline")
        baseline_img_dir = os.path.join(baseline_dir, "images")
        os.makedirs(baseline_img_dir, exist_ok=True)

        print("\n--- Generating: baseline ---")
        for i, (p, s) in enumerate(tqdm(list(zip(prompts, seeds)), desc="Baseline")):
            img_path = os.path.join(baseline_img_dir, f"{i:05d}.png")
            if os.path.exists(img_path):
                continue
            base_cfg.seed = s
            img = generate_image("baseline", p, args.negative_prompt, base_cfg,
                                 models, None, args.backbone)
            img.save(img_path)

    # Step 2: Generate images for each method
    method_img_dirs: Dict[str, str] = {}
    for method in args.methods:
        if method == "baseline":
            continue

        method_cfg = get_method_config(method, args.backbone, args)
        method_dir = os.path.join(args.out_dir, method)
        method_img_dir = os.path.join(method_dir, "images")
        os.makedirs(method_img_dir, exist_ok=True)
        method_img_dirs[method] = method_img_dir

        print(f"\n--- Generating: {method} ---")
        for i, (p, s) in enumerate(tqdm(list(zip(prompts, seeds)), desc=method)):
            img_path = os.path.join(method_img_dir, f"{i:05d}.png")
            if os.path.exists(img_path):
                continue
            method_cfg.seed = s
            img = generate_image(
                method, p, args.negative_prompt, method_cfg,
                models, reward_model, args.backbone,
                ae_pipeline=ae_pipeline,
                subspace_bank=subspace_bank,
            )
            img.save(img_path)

    # --- Timing ---
    if args.timing:
        print("\n--- Running timing benchmark ---")
        timing_results = {}
        for method in args.methods:
            method_cfg = get_method_config(method, args.backbone, args)
            method_cfg.seed = args.seed
            timing = time_generation(
                method, generate_image, prompts[0], args.negative_prompt, method_cfg,
                models, reward_model, args.backbone,
                n_trials=args.timing_trials, warmup=3,
                ae_pipeline=ae_pipeline,
            )
            timing_results[method] = timing
            print(f"  {method}: {timing['mean_sec']:.2f}s +/- {timing['std_sec']:.2f}s")

        with open(os.path.join(args.out_dir, "timing.json"), "w") as f:
            json.dump(timing_results, f, indent=2)

    # ===================================================================
    # PHASE 2: Free generation models, load scorers, score from disk
    # ===================================================================
    if args.score and baseline_img_dir is not None and method_img_dirs:
        import gc
        print("\n--- Freeing generation models from GPU ---")
        del models, reward_model
        if ae_pipeline is not None:
            del ae_pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Loading scorers from utils/...")
        scorers = _load_utils_scorers(
            score_device=args.score_device,
            enable_pickscore=not args.no_pickscore,
            enable_aes=not args.no_aes,
            enable_hps=not args.no_hps,
            enable_clip=not args.no_clip,
        )

        for method, method_img_dir in method_img_dirs.items():
            method_dir = os.path.join(args.out_dir, method)
            print(f"\n--- Scoring: {method} vs baseline ---")
            rows = score_method_pair(
                baseline_img_dir, method_img_dir, prompts, scorers, method_dir, seeds,
            )
            compute_all_statistics(rows, method_dir)
            print(f"[OK] {method}: statistics saved to {method_dir}")

    print(f"\n[DONE] All results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
