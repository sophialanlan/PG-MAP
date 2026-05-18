"""
PG-MAP Pipeline for Stable Diffusion 1.5
==========================================

Full generation pipeline implementing Algorithm 1 from the paper
using SD1.5 as the backbone. Supports all ablation modes:

    PG-MAP (full):  optimize_c=True,  optimize_z=True,  use_reward=True
    MAP-c:          optimize_c=True,  optimize_z=False, use_reward=False
    Reward-z:       optimize_c=False, optimize_z=True,  use_reward=True
    Joint (c,z):    optimize_c=True,  optimize_z=True,  use_reward=False
    Baseline:       optimize_c=False, optimize_z=False, use_reward=False

Preserves all existing helper functions (PatchSchedule, encode_prompt,
predict_eps, decode_latents, etc.) from the original codebase.

Usage:
    from pgmap_sd15 import generate_sd15_pgmap, load_sd15_models
    from pgmap_config import sd15_defaults
    from pgmap_reward import FrozenRewardModel

    models = load_sd15_models("runwayml/stable-diffusion-v1-5")
    reward = FrozenRewardModel("pickscore", device="cuda")
    config = sd15_defaults()

    image, logs = generate_sd15_pgmap(
        "a cat wearing a top hat",
        models=models,
        config=config,
        reward_model=reward,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from pgmap_config import PGMAPConfig, sd15_defaults
from pgmap_core import pgmap_refine_step, _gather_alphas_cumprod, cfg_eps
from pgmap_reward import FrozenRewardModel

# PatchSchedule was extracted to a standalone module after the original
# eval_sd15_map40_clip.py was removed from the repo.
from pgmap_patch_schedule import PatchSchedule


# -----------------------------------------------------------------------
# Model bundle
# -----------------------------------------------------------------------

@dataclass
class SD15Models:
    """Container for all SD1.5 model components."""
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    sched: DDIMScheduler
    device: torch.device
    dtype: torch.dtype


def load_sd15_models(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> SD15Models:
    """Load all SD1.5 model components.

    Args:
        model_id: HuggingFace model identifier.
        device:   Target device ("cuda" or "cpu"). Auto-detected if None.
        dtype:    Model precision (float16 recommended for GPU).

    Returns:
        SD15Models bundle with all components loaded and frozen.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder"
    ).to(dev, dtype=dtype).eval()
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae"
    ).to(dev, dtype=torch.float32).eval()  # VAE in fp32 for stability
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet"
    ).to(dev, dtype=dtype).eval()
    sched = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze all parameters
    for m in [text_encoder, vae, unet]:
        for p in m.parameters():
            p.requires_grad_(False)

    return SD15Models(tokenizer, text_encoder, vae, unet, sched, dev, dtype)


# -----------------------------------------------------------------------
# Prompt encoding
# -----------------------------------------------------------------------

@torch.no_grad()
def encode_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    device: torch.device,
    max_length: int = 77,
) -> torch.Tensor:
    """Encode a text prompt into CLIP embeddings.

    Returns:
        (1, 77, 768) tensor of text embeddings.
    """
    tokens = tokenizer(
        prompt, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    use_mask = getattr(text_encoder.config, "use_attention_mask", False)
    kwargs = {"attention_mask": tokens.attention_mask.to(device)} if use_mask else {}
    out = text_encoder(input_ids=tokens.input_ids.to(device), **kwargs)
    return out.last_hidden_state


# -----------------------------------------------------------------------
# Decoding
# -----------------------------------------------------------------------

@torch.no_grad()
def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> List[Image.Image]:
    """Decode latents to PIL images."""
    latents = latents / 0.18215
    latents_f32 = latents.to(dtype=next(vae.parameters()).dtype)
    imgs = vae.decode(latents_f32).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray((im * 255).round().astype("uint8")) for im in imgs]


# -----------------------------------------------------------------------
# Main generation function
# -----------------------------------------------------------------------

def generate_sd15_pgmap(
    prompt: str,
    negative_prompt: str = "",
    *,
    models: SD15Models,
    config: Optional[PGMAPConfig] = None,
    reward_model: Optional[FrozenRewardModel] = None,
) -> Tuple[Image.Image, Dict]:
    """Generate an image using PG-MAP with SD1.5.

    Implements Algorithm 1 from the paper:
        For each denoising step t:
            1. If in refinement phase, run pgmap_refine_step on (c, z_t)
            2. Apply CFG + DDIM update with the refined (c, z_t)

    Args:
        prompt:          Text prompt for generation.
        negative_prompt: Negative prompt for CFG.
        models:          SD15Models bundle from load_sd15_models().
        config:          PGMAPConfig. Uses sd15_defaults() if None.
        reward_model:    Optional FrozenRewardModel for preference guidance.

    Returns:
        (image, logs) where:
            - image: PIL Image of the generated result.
            - logs: Dict with optional trajectory data (c_traj, timesteps).
    """
    if config is None:
        config = sd15_defaults()

    dev = models.device
    dtype = models.dtype
    unet = models.unet
    vae = models.vae
    sched = models.sched

    # --- Setup scheduler ---
    sched.set_timesteps(config.num_steps, device=dev)
    timesteps = sched.timesteps
    num_steps = timesteps.shape[0]

    # --- Encode prompts ---
    c0 = encode_prompt(models.tokenizer, models.text_encoder, prompt, dev)
    c_u = encode_prompt(models.tokenizer, models.text_encoder, negative_prompt, dev)
    B = 1
    c0 = c0.repeat(B, 1, 1)
    c_u = c_u.repeat(B, 1, 1)

    # --- Setup patch schedule ---
    patch_sched = PatchSchedule(
        mode=config.patch_mode,
        patches=config.external_patches,
        add_to_c0=config.patch_add_to_c0,
        K=config.patch_K,
        patch_scale=config.patch_scale,
        seed=config.patch_seed,
    )

    # --- Initialize latents ---
    g = torch.Generator(device=dev).manual_seed(config.seed)
    h = config.height // 8
    w = config.width // 8
    z_t = torch.randn(
        (B, unet.config.in_channels, h, w),
        generator=g, device=dev, dtype=dtype,
    )
    z_t = z_t * sched.init_noise_sigma

    # --- Precompute reward text features ---
    if reward_model is not None and config.use_reward:
        reward_model.precompute_text_features(prompt)

    # --- Compute refinement phase boundaries ---
    refine_steps = max(1, int(config.rho * num_steps))
    reward_steps = max(1, int(config.reward.rho_Q * num_steps))

    # --- Tracking ---
    c_t = c0.clone()
    c_traj: List[torch.Tensor] = [] if config.save_c_traj else None
    mid_imgs: List[Tuple[int, Image.Image]] = []
    logs: Dict = {}
    _adam_state_c = None  # persistent Adam state for c across denoising steps

    # --- Denoising loop (Algorithm 1) ---
    for step_i, t_val in enumerate(timesteps):
        t_train = torch.full((B,), int(t_val.item()), device=dev, dtype=torch.long)
        prev_t = timesteps[step_i + 1] if (step_i + 1) < num_steps else torch.tensor(-1, device=dev)
        t_prev = torch.full((B,), int(prev_t.item()), device=dev, dtype=torch.long)

        in_refine_phase = (step_i < refine_steps)
        reward_active = (step_i < reward_steps) and config.use_reward

        if in_refine_phase and (config.optimize_c or config.optimize_z):
            mu_t = patch_sched.tau(c0, step_i=step_i, num_steps=num_steps)
            mu_t = mu_t.to(device=dev, dtype=torch.float32)
            alpha_bar_t = _gather_alphas_cumprod(sched, t_train, z_t).float()
            # step_frac: 0 at first reward step → 1 at last reward step
            step_frac = step_i / max(reward_steps - 1, 1) if reward_active else 1.0

            with torch.enable_grad():
                c_t, z_t, eps_fused, _adam_state_c = pgmap_refine_step(
                    unet, sched, vae,
                    z_t_init=z_t, t_train=t_train, t_prev=t_prev,
                    c_init=c_t, mu_t=mu_t,
                    prior=config.prior, ref_cfg=config.refinement,
                    alpha_bar_t=alpha_bar_t,
                    reward_model=reward_model if reward_active else None,
                    reward_config=config.reward if reward_active else None,
                    prompt=prompt,
                    optimize_c=config.optimize_c,
                    optimize_z=config.optimize_z,
                    use_reward=reward_active,
                    step_frac=step_frac,
                    fuse_cfg=True,
                    c_uncond=c_u,
                    guidance_scale=config.guidance_scale,
                    reward_decode_size=224,
                    adam_state_c=_adam_state_c,
                )
        else:
            c_t = c_t.detach()
            z_t = z_t.detach()
            eps_fused = None

        # Save conditioning trajectory
        if c_traj is not None:
            c_traj.append(c_t.detach().to(torch.float32))

        # --- CFG + DDIM update ---
        # Skip outer cfg_eps call when fuse_cfg provided eps from the inner loop.
        with torch.no_grad():
            if eps_fused is not None:
                eps = eps_fused
            else:
                eps = cfg_eps(unet, sched, z_t, t_train, c_u, c_t, config.guidance_scale)
            z_t = sched.step(eps, int(t_train[0].item()), z_t).prev_sample

        # Save intermediate images
        if config.save_progress:
            if step_i % config.save_every == 0 or step_i == num_steps - 1:
                mid_img = decode_latents(vae, z_t.detach())[0]
                mid_imgs.append((step_i, mid_img))

    # --- Final decode ---
    image = decode_latents(vae, z_t)[0]

    # --- Build logs ---
    if c_traj is not None:
        logs["c_traj"] = torch.stack(c_traj, dim=0)
    logs["timesteps"] = timesteps.detach().cpu()
    if mid_imgs:
        logs["mid_imgs"] = mid_imgs

    return image, logs


# -----------------------------------------------------------------------
# Convenience: baseline generation (no refinement)
# -----------------------------------------------------------------------

def generate_sd15_baseline(
    prompt: str,
    negative_prompt: str = "",
    *,
    models: SD15Models,
    config: Optional[PGMAPConfig] = None,
) -> Tuple[Image.Image, Dict]:
    """Generate a standard DDIM+CFG baseline image (no refinement).

    This is equivalent to calling generate_sd15_pgmap with all
    optimization flags disabled.
    """
    from pgmap_config import baseline_config
    if config is None:
        config = baseline_config("sd15")
    else:
        config.optimize_c = False
        config.optimize_z = False
        config.use_reward = False

    return generate_sd15_pgmap(
        prompt, negative_prompt,
        models=models, config=config, reward_model=None,
    )
