"""PG-MAP ComfyUI nodes.

Implementation notes
--------------------

Why we don't consume ComfyUI's MODEL type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PG-MAP needs a differentiable forward through the UNet/MMDiT plus access to
the VAE decode in the same graph. ComfyUI wraps models via its
``ModelPatcher`` abstraction which (a) keeps weights on CPU + lazy-paged to
GPU, and (b) interposes its own forward dispatch. Reaching the underlying
``diffusers`` UNet is fragile across ComfyUI versions, so this node set
takes the simpler, more portable path: each ``PGMAPSampler`` invocation
loads a self-contained ``diffusers`` pipeline (cached across runs in
process memory) and uses that for generation.

Cost: one extra UNet copy in VRAM when a vanilla ComfyUI pipeline is
already loaded. Benefit: behaves identically to the PyPI / HF Hub flow,
deterministic across ComfyUI updates.

Outputs
~~~~~~~

The sampler outputs ComfyUI's ``IMAGE`` tensor (B,H,W,3 in [0,1]) rather
than ``LATENT``; we already decode with the VAE during PG-MAP refinement,
so re-encoding back to latent space would be wasteful and introduce drift.
"""
from __future__ import annotations

import gc
from typing import Optional

import torch
import numpy as np

# Process-level cache of loaded pipelines (one per backbone).
_PIPELINE_CACHE: dict = {}


def _load_pipeline(backbone: str, device: str = "cuda"):
    """Load (or return cached) PG-MAP pipeline for `backbone`."""
    key = (backbone, device)
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    from diffusers import DiffusionPipeline

    spec = {
        "sd15": ("stable-diffusion-v1-5/stable-diffusion-v1-5", "sophialan/pg-map-sd15", {}),
        "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", "sophialan/pg-map-sdxl",
                  {"variant": "fp16"}),
        "sd3":  ("stabilityai/stable-diffusion-3.5-medium", "sophialan/pg-map-sd3", {}),
    }[backbone]
    model_id, custom_pipe, extra = spec

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        custom_pipeline=custom_pipe,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        **extra,
    ).to(device)
    _PIPELINE_CACHE[key] = pipe
    return pipe


def _free_pipeline_cache():
    """Free all cached pipelines (used by maintenance / OOM recovery)."""
    global _PIPELINE_CACHE
    for pipe in _PIPELINE_CACHE.values():
        del pipe
    _PIPELINE_CACHE = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pil_to_comfy_image(pil_imgs):
    """Convert a list of PIL.Image to a ComfyUI IMAGE tensor (B,H,W,3) in [0,1]."""
    arr = np.stack([np.asarray(img.convert("RGB")) for img in pil_imgs], axis=0)
    arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr)


# =============================================================================
# Reward loader node
# =============================================================================

class PGMAPRewardLoader:
    """Load a frozen preference reward model.

    Outputs a ``PGMAP_REWARD`` socket that downstream nodes (currently the
    ``PG-MAP Sampler``) accept. The reward model parameters are frozen at
    load time; this node is a pure "loader" with no inference of its own.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reward_model": (["pickscore", "hps", "clip", "aesthetic", "imagereward"],
                                  {"default": "pickscore"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("PGMAP_REWARD",)
    RETURN_NAMES = ("reward",)
    FUNCTION = "load"
    CATEGORY = "PG-MAP"

    def load(self, reward_model, device):
        from pgmap_reward import FrozenRewardModel
        rm = FrozenRewardModel(reward_model, device=device)
        return (rm,)


# =============================================================================
# Config builder node
# =============================================================================

class PGMAPConfigBuilder:
    """Build a PGMAPConfig from per-hyperparameter sliders.

    Defaults reflect the paper's recommended row for the chosen backbone:
        SDXL:   λ=0.10, η_z=0.005, K=2, ρ=0.5, ρ_Q=0.3
        SD 1.5: λ=0.05, η_z=0.005, K=2, ρ=0.4, ρ_Q=0.3

    Outputs a ``PGMAP_CONFIG`` socket that the ``PG-MAP Sampler`` accepts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backbone": (["sd15", "sdxl", "sd3"], {"default": "sdxl"}),
                "lambda_reward": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.5, "step": 0.005}),
                "eta_z":         ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.5, "step": 0.001}),
                "eta_c":         ("FLOAT", {"default": 1e-3,  "min": 0.0, "max": 0.1, "step": 1e-4}),
                "K":             ("INT",   {"default": 2, "min": 1, "max": 6, "step": 1}),
                "rho":           ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05}),
                "rho_Q":         ("FLOAT", {"default": 0.3, "min": 0.0,  "max": 1.0, "step": 0.05}),
                "sigma_c":       ("FLOAT", {"default": 1.0, "min": 0.1,  "max": 5.0, "step": 0.1}),
                "gamma":         ("FLOAT", {"default": 1.0, "min": 0.1,  "max": 5.0, "step": 0.1}),
                "optimize_c":    ("BOOLEAN", {"default": True}),
                "optimize_z":    ("BOOLEAN", {"default": True}),
                "use_reward":    ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("PGMAP_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build"
    CATEGORY = "PG-MAP"

    def build(self, backbone, lambda_reward, eta_z, eta_c, K, rho, rho_Q,
              sigma_c, gamma, optimize_c, optimize_z, use_reward):
        from dataclasses import replace
        from pgmap import sd15_defaults, sdxl_defaults

        # Pick the right preset; SD3 uses sdxl_defaults as a starting point.
        presets = {"sd15": sd15_defaults, "sdxl": sdxl_defaults, "sd3": sdxl_defaults}
        cfg = presets[backbone]()
        cfg = replace(cfg, rho=float(rho),
                      optimize_c=bool(optimize_c),
                      optimize_z=bool(optimize_z),
                      use_reward=bool(use_reward))
        cfg.refinement.K = int(K)
        cfg.refinement.eta_c = float(eta_c)
        cfg.refinement.eta_z = float(eta_z)
        cfg.reward.lambda_reward = float(lambda_reward)
        cfg.reward.rho_Q = float(rho_Q)
        cfg.prior.sigma_c = float(sigma_c)
        cfg.prior.gamma = float(gamma)
        return (cfg,)


# =============================================================================
# Sampler node
# =============================================================================

class PGMAPSampler:
    """Run PG-MAP refinement and return a ComfyUI IMAGE.

    Loads (or reuses cached) the official PG-MAP custom pipeline for the
    selected backbone from the HuggingFace Hub
    (``sophialan/pg-map-{sd15,sdxl,sd3}``).

    If ``config`` and ``reward`` are both unset / disabled, this node
    behaves identically to a vanilla diffusers sampler on the same backbone.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backbone": (["sd15", "sdxl", "sd3"], {"default": "sdxl"}),
                "prompt":   ("STRING", {"default": "a phoenix rising from ashes",
                                         "multiline": True}),
                "negative_prompt": ("STRING", {"default": "blurry, low quality",
                                                "multiline": True}),
                "seed":     ("INT",   {"default": 42, "min": 0, "max": 2**31 - 1, "step": 1}),
                "steps":    ("INT",   {"default": 50, "min": 8, "max": 100, "step": 1}),
                "cfg_scale":("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "width":    ("INT",   {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height":   ("INT",   {"default": 1024, "min": 256, "max": 2048, "step": 64}),
            },
            "optional": {
                "config": ("PGMAP_CONFIG",),
                "reward": ("PGMAP_REWARD",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "PG-MAP"

    def sample(self, backbone, prompt, negative_prompt, seed, steps, cfg_scale,
               width, height, config=None, reward=None):
        from dataclasses import replace

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = _load_pipeline(backbone, device=device)

        # Update the user-controllable fields on the config if provided.
        pg_cfg = None
        if config is not None:
            pg_cfg = replace(
                config,
                seed=int(seed),
                num_steps=int(steps),
                guidance_scale=float(cfg_scale),
                height=int(height),
                width=int(width),
            )

        gen = torch.Generator(device=device).manual_seed(int(seed))

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg_scale),
            generator=gen,
        )
        # Only the SD/SDXL pipelines take height/width; SD3 reads them differently.
        if backbone in ("sd15", "sdxl"):
            kwargs.update(height=int(height), width=int(width))
        else:
            kwargs.update(height=int(height), width=int(width))   # SD3 also accepts H/W

        if pg_cfg is not None:
            kwargs["pg_map_config"] = pg_cfg
        if reward is not None:
            kwargs["reward_model"] = reward

        out = pipe(**kwargs)
        return (_pil_to_comfy_image(out.images),)
