"""
PG-MAP Pipeline for Stable Diffusion XL
==========================================

Full generation pipeline implementing Algorithm 1 using SDXL as the backbone.
Mirrors pgmap_sd15.py but handles SDXL-specific details:

    - Dual text encoders (CLIP ViT-L + CLIP ViT-G)
    - Pooled embeddings passed as added_cond_kwargs
    - time_ids for original/crop/target size conditioning
    - Only the token-level embedding c is optimized, NOT the pooled embedding
    - VAE in fp32, UNet in fp16

Usage:
    from pgmap_sdxl import generate_sdxl_pgmap, load_sdxl_models
    from pgmap_config import sdxl_defaults
    from pgmap_reward import FrozenRewardModel

    models = load_sdxl_models("stabilityai/stable-diffusion-xl-base-1.0")
    reward = FrozenRewardModel("pickscore", device="cuda")
    config = sdxl_defaults()

    image, logs = generate_sdxl_pgmap(
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

from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from pgmap_config import PGMAPConfig, sdxl_defaults
from pgmap_core import pgmap_refine_step, _gather_alphas_cumprod, cfg_eps
from pgmap_reward import FrozenRewardModel

# Use the standalone PatchSchedule module (cleaner than depending on the
# legacy eval_sdxl_map40_clip_v1.py file).
from pgmap_patch_schedule import PatchSchedule


# -----------------------------------------------------------------------
# Model bundle
# -----------------------------------------------------------------------

@dataclass
class SDXLModels:
    """Container for all SDXL model components."""
    tokenizer_1: CLIPTokenizer
    tokenizer_2: CLIPTokenizer
    text_encoder_1: CLIPTextModel
    text_encoder_2: CLIPTextModelWithProjection
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    sched: DDIMScheduler
    device: torch.device
    dtype: torch.dtype


def load_sdxl_models(
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> SDXLModels:
    """Load all SDXL model components.

    Uses the diffusers StableDiffusionXLPipeline for loading, then
    extracts individual components. VAE is kept in fp32 for stability.

    Args:
        model_id: HuggingFace model identifier.
        device:   Target device. Auto-detected if None.
        dtype:    UNet/text encoder precision.

    Returns:
        SDXLModels bundle with all components loaded and frozen.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype if dev.type == "cuda" else torch.float32,
        variant="fp16" if (dtype == torch.float16 and dev.type == "cuda") else None,
    )

    sched = DDIMScheduler.from_config(pipe.scheduler.config)

    tokenizer_1 = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    text_encoder_1 = pipe.text_encoder.to(dev, dtype=dtype).eval()
    text_encoder_2 = pipe.text_encoder_2.to(dev, dtype=dtype).eval()
    vae = pipe.vae.to(dev, dtype=torch.float32).eval()  # fp32 for stability
    unet = pipe.unet.to(dev, dtype=dtype).eval()

    # Freeze all parameters
    for m in [text_encoder_1, text_encoder_2, vae, unet]:
        for p in m.parameters():
            p.requires_grad_(False)

    return SDXLModels(
        tokenizer_1, tokenizer_2,
        text_encoder_1, text_encoder_2,
        vae, unet, sched, dev, dtype,
    )


# -----------------------------------------------------------------------
# SDXL prompt encoding
# -----------------------------------------------------------------------

@torch.no_grad()
def encode_prompt_sdxl(
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    prompt: str,
    device: torch.device,
    max_length: int = 77,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode prompt with both SDXL text encoders.

    Returns:
        (prompt_embeds, pooled_embeds) where:
            prompt_embeds: (1, 77, 2048) concatenated hidden states.
            pooled_embeds: (1, 1280) from encoder 2.
    """
    # Encoder 1: CLIP ViT-L
    t1 = tokenizer_1(
        prompt, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    out1 = text_encoder_1(
        input_ids=t1.input_ids.to(device),
        attention_mask=t1.attention_mask.to(device),
    )
    emb1 = out1.last_hidden_state  # (1, 77, 768)

    # Encoder 2: CLIP ViT-G with projection
    t2 = tokenizer_2(
        prompt, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    out2 = text_encoder_2(
        input_ids=t2.input_ids.to(device),
        attention_mask=t2.attention_mask.to(device),
    )
    emb2 = out2.last_hidden_state  # (1, 77, 1280)

    # Pooled embedding from encoder 2
    pooled = None
    if hasattr(out2, "text_embeds") and out2.text_embeds is not None:
        pooled = out2.text_embeds
    elif hasattr(out2, "pooler_output") and out2.pooler_output is not None:
        pooled = out2.pooler_output
    else:
        pooled = emb2[:, 0]  # CLS token fallback

    # Concatenate: (1, 77, 768) + (1, 77, 1280) -> (1, 77, 2048)
    prompt_embeds = torch.cat([emb1, emb2], dim=-1)

    return prompt_embeds, pooled


def make_sdxl_time_ids(
    device: torch.device,
    dtype: torch.dtype,
    *,
    original_size: Tuple[int, int],
    crop_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Tuple[int, int],
    batch_size: int = 1,
) -> torch.Tensor:
    """Create SDXL time_ids tensor for size conditioning.

    Returns:
        (B, 6) tensor: [orig_h, orig_w, crop_y, crop_x, tgt_h, tgt_w].
    """
    oh, ow = original_size
    cy, cx = crop_coords_top_left
    th, tw = target_size
    time_ids = torch.tensor([[oh, ow, cy, cx, th, tw]], device=device, dtype=dtype)
    return time_ids.repeat(batch_size, 1)


# -----------------------------------------------------------------------
# Decoding
# -----------------------------------------------------------------------

@torch.no_grad()
def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> List[Image.Image]:
    """Decode latents to PIL images (SDXL scaling factor)."""
    scale = getattr(vae.config, "scaling_factor", 0.18215)
    p = next(vae.parameters())
    latents = (latents / scale).to(device=p.device, dtype=p.dtype)
    imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray((im * 255).round().astype("uint8")) for im in imgs]


# -----------------------------------------------------------------------
# SDXL noise prediction (for the outer CFG loop)
# -----------------------------------------------------------------------

def predict_eps_sdxl(
    unet: UNet2DConditionModel,
    sched: DDIMScheduler,
    z_t: torch.Tensor,
    t_train: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    time_ids: torch.Tensor,
) -> torch.Tensor:
    """UNet noise prediction for SDXL with added_cond_kwargs."""
    z_in = sched.scale_model_input(z_t, t_train)
    return unet(
        z_in, t_train,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={
            "text_embeds": pooled_embeds,
            "time_ids": time_ids,
        },
    ).sample


# -----------------------------------------------------------------------
# Main generation function
# -----------------------------------------------------------------------

def generate_sdxl_pgmap(
    prompt: str,
    negative_prompt: str = "",
    *,
    models: SDXLModels,
    config: Optional[PGMAPConfig] = None,
    reward_model: Optional[FrozenRewardModel] = None,
) -> Tuple[Image.Image, Dict]:
    """Generate an image using PG-MAP with SDXL.

    Implements Algorithm 1, SDXL variant. Key difference from SD1.5:
    only the token-level embedding c is optimized; pooled_embeds and
    time_ids are kept frozen (as specified in paper Sec. 3.5).

    Args:
        prompt:          Text prompt for generation.
        negative_prompt: Negative prompt for CFG.
        models:          SDXLModels bundle from load_sdxl_models().
        config:          PGMAPConfig. Uses sdxl_defaults() if None.
        reward_model:    Optional FrozenRewardModel for preference guidance.

    Returns:
        (image, logs) where:
            - image: PIL Image of the generated result.
            - logs: Dict with optional trajectory data.
    """
    if config is None:
        config = sdxl_defaults()

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
    c0, pooled0 = encode_prompt_sdxl(
        models.tokenizer_1, models.tokenizer_2,
        models.text_encoder_1, models.text_encoder_2,
        prompt, dev,
    )
    c_u, pooled_u = encode_prompt_sdxl(
        models.tokenizer_1, models.tokenizer_2,
        models.text_encoder_1, models.text_encoder_2,
        negative_prompt, dev,
    )

    B = 1
    c0 = c0.repeat(B, 1, 1)
    c_u = c_u.repeat(B, 1, 1)
    pooled0 = pooled0.repeat(B, 1)
    pooled_u = pooled_u.repeat(B, 1)

    # --- Time IDs ---
    time_ids = make_sdxl_time_ids(
        dev, dtype,
        original_size=(config.height, config.width),
        target_size=(config.height, config.width),
        batch_size=B,
    )

    # --- Patch schedule ---
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

    # --- Phase boundaries ---
    refine_steps = max(1, int(config.rho * num_steps))
    reward_steps = max(1, int(config.reward.rho_Q * num_steps))

    # --- Tracking ---
    c_t = c0.clone()
    c_traj: List[torch.Tensor] = [] if config.save_c_traj else None
    mid_imgs: List[Tuple[int, Image.Image]] = []
    logs: Dict = {}
    _adam_state_c = None  # persistent Adam state for c across denoising steps

    # --- Denoising loop ---
    for step_i, t_val in enumerate(timesteps):
        t_train = torch.full((B,), int(t_val.item()), device=dev, dtype=torch.long)
        prev_t = timesteps[step_i + 1] if (step_i + 1) < num_steps else torch.tensor(-1, device=dev)
        t_prev = torch.full((B,), int(prev_t.item()), device=dev, dtype=torch.long)

        in_refine_phase = (step_i < refine_steps)
        reward_active = (step_i < reward_steps) and config.use_reward

        if in_refine_phase and (config.optimize_c or config.optimize_z):
            mu_t = patch_sched.tau(c0, step_i=step_i, num_steps=num_steps)
            mu_t = mu_t.to(device=dev, dtype=dtype)
            alpha_bar_t = _gather_alphas_cumprod(sched, t_train, z_t)
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
                    pooled=pooled0,
                    time_ids=time_ids,
                    step_frac=step_frac,
                    fuse_cfg=True,
                    c_uncond=c_u,
                    pooled_uncond=pooled_u,
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
        # When fuse_cfg delivered eps_fused from the inner loop, skip the
        # redundant outer cfg_eps call (saves one full UNet forward per refine step).
        with torch.no_grad():
            if eps_fused is not None:
                eps = eps_fused
            else:
                eps = cfg_eps(
                    unet, sched, z_t, t_train,
                    c_u, c_t, config.guidance_scale,
                    pooled_uncond=pooled_u, pooled_cond=pooled0, time_ids=time_ids,
                )
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
# Convenience: baseline generation
# -----------------------------------------------------------------------

def generate_sdxl_baseline(
    prompt: str,
    negative_prompt: str = "",
    *,
    models: SDXLModels,
    config: Optional[PGMAPConfig] = None,
) -> Tuple[Image.Image, Dict]:
    """Generate a standard DDIM+CFG baseline image with SDXL."""
    from pgmap_config import baseline_config
    if config is None:
        config = baseline_config("sdxl")
    else:
        config.optimize_c = False
        config.optimize_z = False
        config.use_reward = False

    return generate_sdxl_pgmap(
        prompt, negative_prompt,
        models=models, config=config, reward_model=None,
    )
