"""
SD1.5 pipeline that dispatches to the round-2 variant refine steps
(currently exposes Newton; can be extended for eps2 the same way as SDXL).

Mirrors pgmap_sdxl_variants.py for SD1.5: same logic but without pooled
embeddings and time_ids (SD1.5 uses a single CLIP encoder).
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict

import torch
from PIL import Image

from pgmap_config import PGMAPConfig, sd15_defaults
from pgmap_core import _gather_alphas_cumprod, cfg_eps
from pgmap_reward import FrozenRewardModel
from pgmap_variants import pgmap_refine_step_newton, pgmap_refine_step_eps2
from pgmap_sd15 import (
    SD15Models,
    encode_prompt,
    decode_latents,
)
from pgmap_patch_schedule import PatchSchedule


def _make_sd15_pipeline_for_refine_step(refine_step_fn, **extra_kwargs):
    def gen(prompt: str, negative_prompt: str = "",
            *, models: SD15Models,
            config: Optional[PGMAPConfig] = None,
            reward_model: Optional[FrozenRewardModel] = None) -> Tuple[Image.Image, Dict]:
        if config is None:
            config = sd15_defaults()

        dev = models.device
        dtype = models.dtype
        unet = models.unet
        vae = models.vae
        sched = models.sched

        sched.set_timesteps(config.num_steps, device=dev)
        timesteps = sched.timesteps
        num_steps = timesteps.shape[0]

        c0 = encode_prompt(models.tokenizer, models.text_encoder, prompt, dev)
        c_u = encode_prompt(models.tokenizer, models.text_encoder, negative_prompt, dev)
        B = 1
        c0 = c0.repeat(B, 1, 1)
        c_u = c_u.repeat(B, 1, 1)

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
                mu_t = patch_sched.tau(c0, step_i=step_i, num_steps=num_steps).to(device=dev, dtype=torch.float32)
                alpha_bar_t = _gather_alphas_cumprod(sched, t_train, z_t).float()
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
                        # SD1.5: no pooled / no time_ids
                        pooled=None, time_ids=None,
                        step_frac=step_frac,
                        fuse_cfg=True,
                        c_uncond=c_u,
                        pooled_uncond=None,
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
                    eps = cfg_eps(unet, sched, z_t, t_train,
                                   c_u, c_t, config.guidance_scale)
                z_t = sched.step(eps, int(t_train[0].item()), z_t).prev_sample

        return decode_latents(vae, z_t)[0], {"timesteps": timesteps.detach().cpu()}
    return gen


generate_sd15_pgmap_newton = _make_sd15_pipeline_for_refine_step(pgmap_refine_step_newton)
generate_sd15_pgmap_eps2   = _make_sd15_pipeline_for_refine_step(pgmap_refine_step_eps2)
