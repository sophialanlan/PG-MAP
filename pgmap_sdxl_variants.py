"""
SDXL generation pipeline that dispatches to the variant refine step.

This is a thin clone of generate_sdxl_pgmap from pgmap_sdxl.py, modified to
call pgmap_refine_step_variant instead of pgmap_refine_step. All other logic
(prompt encoding, scheduler setup, CFG, decoding) is reused from pgmap_sdxl.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from pgmap_config import PGMAPConfig, sdxl_defaults
from pgmap_core import _gather_alphas_cumprod, cfg_eps
from pgmap_reward import FrozenRewardModel
from pgmap_variants import (
    pgmap_refine_step_variant,
    pgmap_refine_step_newton,
    pgmap_refine_step_eps2,
)
from pgmap_sdxl import (
    SDXLModels,
    encode_prompt_sdxl,
    make_sdxl_time_ids,
    decode_latents,
)
from pgmap_patch_schedule import PatchSchedule


def generate_sdxl_pgmap_variant(
    prompt: str,
    negative_prompt: str = "",
    *,
    models: SDXLModels,
    config: Optional[PGMAPConfig] = None,
    reward_model: Optional[FrozenRewardModel] = None,
    variant: str = "A",
    trust_radius: float = 1.0,
) -> Tuple[Image.Image, Dict]:
    """SDXL PG-MAP with one of the three gradient-shape variants.

    Args:
        variant:       'A' / 'B' / 'C' (see pgmap_variants for definitions).
        trust_radius:  L2 trust radius for variant C (ignored for A/B).
    """
    if config is None:
        config = sdxl_defaults()

    dev = models.device
    dtype = models.dtype
    unet = models.unet
    vae = models.vae
    sched = models.sched

    sched.set_timesteps(config.num_steps, device=dev)
    timesteps = sched.timesteps
    num_steps = timesteps.shape[0]

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
    time_ids = make_sdxl_time_ids(
        dev, dtype,
        original_size=(config.height, config.width),
        target_size=(config.height, config.width),
        batch_size=B,
    )
    patch_sched = PatchSchedule(
        mode=config.patch_mode,
        patches=config.external_patches,
        add_to_c0=config.patch_add_to_c0,
        K=config.patch_K,
        patch_scale=config.patch_scale,
        seed=config.patch_seed,
    )

    g = torch.Generator(device=dev).manual_seed(config.seed)
    h = config.height // 8
    w = config.width // 8
    z_t = torch.randn(
        (B, unet.config.in_channels, h, w),
        generator=g, device=dev, dtype=dtype,
    )
    z_t = z_t * sched.init_noise_sigma

    if reward_model is not None and config.use_reward:
        reward_model.precompute_text_features(prompt)

    refine_steps = max(1, int(config.rho * num_steps))
    reward_steps = max(1, int(config.reward.rho_Q * num_steps))

    c_t = c0.clone()
    logs: Dict = {}

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
                c_t, z_t, eps_fused = pgmap_refine_step_variant(
                    unet, sched, vae,
                    z_t_init=z_t, t_train=t_train, t_prev=t_prev,
                    c_init=c_t, mu_t=mu_t,
                    prior=config.prior, ref_cfg=config.refinement,
                    alpha_bar_t=alpha_bar_t,
                    variant=variant,
                    trust_radius=trust_radius,
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
                )
        else:
            c_t = c_t.detach()
            z_t = z_t.detach()
            eps_fused = None

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

    image = decode_latents(vae, z_t)[0]
    logs["timesteps"] = timesteps.detach().cpu()
    return image, logs


# ---------------------------------------------------------------------------
# Variants for the second round (directions B2, C2)
# Direction A2 (Adam) reuses generate_sdxl_pgmap from pgmap_sdxl with an Adam
# config, no new pipeline needed.
# ---------------------------------------------------------------------------

def _make_pipeline_for_refine_step(refine_step_fn, **extra_kwargs):
    """Build a pipeline that calls a custom refine-step function.
    All other denoising machinery is identical to generate_sdxl_pgmap_variant.
    """
    def gen(prompt, negative_prompt="", *, models, config=None, reward_model=None):
        if config is None:
            config = sdxl_defaults()
        dev = models.device
        dtype = models.dtype
        unet = models.unet
        vae = models.vae
        sched = models.sched
        sched.set_timesteps(config.num_steps, device=dev)
        timesteps = sched.timesteps
        num_steps = timesteps.shape[0]

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
        c0 = c0.repeat(B, 1, 1); c_u = c_u.repeat(B, 1, 1)
        pooled0 = pooled0.repeat(B, 1); pooled_u = pooled_u.repeat(B, 1)
        time_ids = make_sdxl_time_ids(
            dev, dtype,
            original_size=(config.height, config.width),
            target_size=(config.height, config.width),
            batch_size=B,
        )
        patch_sched = PatchSchedule(
            mode=config.patch_mode, patches=config.external_patches,
            add_to_c0=config.patch_add_to_c0,
            K=config.patch_K, patch_scale=config.patch_scale, seed=config.patch_seed,
        )
        g = torch.Generator(device=dev).manual_seed(config.seed)
        h, w = config.height // 8, config.width // 8
        z_t = torch.randn(
            (B, unet.config.in_channels, h, w),
            generator=g, device=dev, dtype=dtype,
        ) * sched.init_noise_sigma

        if reward_model is not None and config.use_reward:
            reward_model.precompute_text_features(prompt)

        refine_steps = max(1, int(config.rho * num_steps))
        reward_steps = max(1, int(config.reward.rho_Q * num_steps))
        c_t = c0.clone()

        for step_i, t_val in enumerate(timesteps):
            t_train = torch.full((B,), int(t_val.item()), device=dev, dtype=torch.long)
            prev_t = timesteps[step_i + 1] if (step_i + 1) < num_steps else torch.tensor(-1, device=dev)
            t_prev = torch.full((B,), int(prev_t.item()), device=dev, dtype=torch.long)
            in_refine_phase = (step_i < refine_steps)
            reward_active = (step_i < reward_steps) and config.use_reward

            if in_refine_phase and (config.optimize_c or config.optimize_z):
                mu_t = patch_sched.tau(c0, step_i=step_i, num_steps=num_steps).to(device=dev, dtype=dtype)
                alpha_bar_t = _gather_alphas_cumprod(sched, t_train, z_t)
                step_frac = step_i / max(reward_steps - 1, 1) if reward_active else 1.0
                with torch.enable_grad():
                    c_t, z_t, eps_fused = refine_step_fn(
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
                        pooled=pooled0, time_ids=time_ids,
                        step_frac=step_frac,
                        fuse_cfg=True,
                        c_uncond=c_u, pooled_uncond=pooled_u,
                        guidance_scale=config.guidance_scale,
                        reward_decode_size=224,
                        **extra_kwargs,
                    )
            else:
                c_t = c_t.detach(); z_t = z_t.detach(); eps_fused = None

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

        return decode_latents(vae, z_t)[0], {"timesteps": timesteps.detach().cpu()}
    return gen


generate_sdxl_pgmap_newton = _make_pipeline_for_refine_step(pgmap_refine_step_newton)
generate_sdxl_pgmap_eps2   = _make_pipeline_for_refine_step(pgmap_refine_step_eps2)
