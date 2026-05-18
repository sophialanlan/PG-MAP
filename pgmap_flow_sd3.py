"""SD3 / SD3.5 generation pipelines: standard FM baseline, UG-FM, PG-MAP-FM.

Standard rectified-flow Euler is run via ``FlowMatchEulerDiscreteScheduler``
(the same scheduler diffusers uses internally for ``StableDiffusion3Pipeline``).
We expose the per-timestep latent so we can drop in PG-MAP refinement before
the Euler update without modifying the official sampling logic.

Three generators are provided:

    generate_sd3_baseline(prompt, ...)          → standard FM (no PG-MAP)
    generate_sd3_ug_flow(prompt, reward, ...)   → UG-FM (reward-only)
    generate_sd3_pgmap_flow(prompt, reward, ...) → PG-MAP-FM (full)

All three share the same backbone and scheduler; only the per-step latent
update differs. The "baseline" matches the reference SD3 sampler exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from PIL import Image

from pgmap_flow_core import (
    FlowRefinementConfig, FlowPriorConfig, FlowRewardConfig,
    pgmap_flow_refine_step, ug_flow_refine_step, flow_endpoint_estimate,
)


@dataclass
class SD3FlowModels:
    """Bundle of SD3 components needed for explicit FM sampling."""
    pipe: object                # diffusers StableDiffusion3Pipeline
    transformer: object         # SD3Transformer2DModel
    vae: object                 # AutoencoderKL
    scheduler: object           # FlowMatchEulerDiscreteScheduler
    device: torch.device
    dtype: torch.dtype


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def load_sd3_models(
    model_id: str = "stabilityai/stable-diffusion-3.5-medium",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> SD3FlowModels:
    from diffusers import StableDiffusion3Pipeline
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    # Standard rectified-flow scheduler (verbatim FM, not distilled).
    sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    return SD3FlowModels(
        pipe=pipe,
        transformer=pipe.transformer,
        vae=pipe.vae,
        scheduler=sched,
        device=torch.device(device),
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_prompt(models: SD3FlowModels, prompt: str, neg_prompt: str
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pipe = models.pipe
    (prompt_embeds, neg_prompt_embeds,
     pooled_prompt_embeds, neg_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, prompt_3=prompt,
        negative_prompt=neg_prompt, negative_prompt_2=neg_prompt, negative_prompt_3=neg_prompt,
        device=models.device, num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    return prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds


def _make_velocity_predictor(models: SD3FlowModels,
                             pooled_cond: torch.Tensor,
                             neg_prompt_embeds: torch.Tensor,
                             neg_pooled: torch.Tensor,
                             cfg_scale: float):
    """Returns ``predict_velocity(z, t_scalar, c)`` that runs CFG-extrapolated
    velocity prediction through the SD3 transformer with grad enabled.

    ``c`` is the (potentially refined) conditional prompt-embedding tensor.
    Pooled and unconditional embeddings are held fixed.
    """
    transformer = models.transformer

    def predict_velocity(z: torch.Tensor, t_now: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # SD3 expects timestep in the same units the scheduler uses (fp32).
        # Do NOT cast to z.dtype (fp16) — the diffusers pipeline passes
        # scheduler.timesteps (fp32) verbatim to the transformer, and
        # downcasting here causes the manual sampling loop to diverge from
        # the official pipeline.
        if not torch.is_tensor(t_now):
            t_now = torch.tensor([t_now], device=z.device, dtype=torch.float32)
        if t_now.ndim == 0:
            t_now = t_now[None]
        t_batch = t_now.expand(z.shape[0]).to(device=z.device)

        # Concatenate cond / uncond pairs for CFG.
        z_in = torch.cat([z, z], dim=0)
        c_in = torch.cat([neg_prompt_embeds, c], dim=0)
        p_in = torch.cat([neg_pooled, pooled_cond], dim=0)
        t_in = torch.cat([t_batch, t_batch], dim=0)

        v_full = transformer(
            hidden_states=z_in,
            encoder_hidden_states=c_in,
            pooled_projections=p_in,
            timestep=t_in,
            return_dict=False,
        )[0]
        v_uncond, v_cond = v_full.chunk(2, dim=0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    return predict_velocity


@torch.no_grad()
def _decode_to_image(models: SD3FlowModels, z0: torch.Tensor) -> Image.Image:
    vae = models.vae
    z = z0 / vae.config.scaling_factor + vae.config.shift_factor
    img = vae.decode(z.to(models.dtype), return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img.cpu().permute(0, 2, 3, 1).float().numpy()
    img = (img[0] * 255).round().astype("uint8")
    return Image.fromarray(img)


def _decode_endpoint_grad(models: SD3FlowModels, z_endpoint: torch.Tensor) -> torch.Tensor:
    """Differentiable VAE decode for the endpoint estimate (for reward grad)."""
    vae = models.vae
    z = z_endpoint / vae.config.scaling_factor + vae.config.shift_factor
    img = vae.decode(z.to(models.dtype), return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1)
    return img  # [B, 3, H, W], in [0, 1]


# ---------------------------------------------------------------------------
# Sampling loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sd3_baseline(
    prompt: str, neg_prompt: str = "blurry, low quality",
    *,
    models: SD3FlowModels,
    height: int = 1024, width: int = 1024,
    num_steps: int = 28, cfg_scale: float = 7.0,
    seed: int = 42,
) -> Image.Image:
    """Standard SD3 rectified-flow Euler sampler. Verbatim FM baseline, no
    PG-MAP refinement. Goes through the diffusers pipeline so we know we
    match the reference implementation exactly.
    """
    pipe = models.pipe
    g = torch.Generator(device=models.device).manual_seed(int(seed))
    out = pipe(
        prompt=prompt, negative_prompt=neg_prompt,
        height=height, width=width,
        num_inference_steps=num_steps, guidance_scale=cfg_scale,
        generator=g,
    )
    return out.images[0]


def _sample_with_callback(
    models: SD3FlowModels,
    prompt: str, neg_prompt: str,
    height: int, width: int,
    num_steps: int, cfg_scale: float,
    seed: int,
    refine_fn,
) -> Image.Image:
    """Run the SD3 rectified-flow Euler sampler manually so we can intervene
    at each timestep via ``refine_fn(z_t, t_now, t_next, sigma_t, sigma_next, c, c_anchor)``
    which returns possibly-refined ``(z_t_new, c_new)`` and the velocity to
    use in the Euler update.
    """
    sched = models.scheduler
    # SD3 uses a resolution-dependent shift `mu` to concentrate timesteps
    # near the data side. Without it, set_timesteps gives uniform sigmas,
    # which produces a different trajectory than the official pipeline.
    # (Verified by audit: identity-refine deviation drops from ~3.7 to ~0
    # mean abs pixel diff after this fix.)
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
        calculate_shift,
    )
    patch_size = getattr(models.transformer.config, "patch_size", 2)
    latent_h_for_sched = height // models.pipe.vae_scale_factor
    latent_w_for_sched = width // models.pipe.vae_scale_factor
    image_seq_len = (latent_h_for_sched // patch_size) * (latent_w_for_sched // patch_size)
    mu = calculate_shift(
        image_seq_len,
        base_seq_len=sched.config.get("base_image_seq_len", 256),
        max_seq_len=sched.config.get("max_image_seq_len", 4096),
        base_shift=sched.config.get("base_shift", 0.5),
        max_shift=sched.config.get("max_shift", 1.15),
    )
    sched.set_timesteps(num_steps, device=models.device, mu=mu)

    # Encode prompts (gradient-enabled for c, fixed for unconditional + pooled).
    prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = _encode_prompt(
        models, prompt, neg_prompt,
    )
    c_anchor = prompt_embeds.detach().clone()

    predict_velocity = _make_velocity_predictor(
        models, pooled_cond=pooled.detach(),
        neg_prompt_embeds=neg_prompt_embeds.detach(),
        neg_pooled=neg_pooled.detach(),
        cfg_scale=cfg_scale,
    )

    # Initial latent.
    g = torch.Generator(device=models.device).manual_seed(int(seed))
    latent_h = height // models.pipe.vae_scale_factor
    latent_w = width // models.pipe.vae_scale_factor
    z = torch.randn(
        (1, models.transformer.config.in_channels, latent_h, latent_w),
        generator=g, device=models.device, dtype=models.dtype,
    )

    for i, t in enumerate(sched.timesteps):
        # Continuous t_now in [0, 1] from the scheduler's sigma schedule.
        sigma_now = sched.sigmas[i].item()
        sigma_next = sched.sigmas[i + 1].item() if (i + 1) < len(sched.sigmas) else 0.0
        # Rectified-flow convention: t = 1 - sigma (start at t=0 for noise,
        # end at t=1 for data). The scheduler stores sigmas decreasing from 1
        # toward 0, so 1 - sigma gives the FM continuous time.
        t_cont = max(0.0, 1.0 - sigma_now)
        dt = sigma_now - sigma_next   # >= 0; standard Euler step magnitude

        c_used = c_anchor
        with torch.no_grad():
            v_default = predict_velocity(z, t, c_anchor)

        # Custom refine (returns possibly-refined z, c, and the velocity to use).
        z_new, c_new, v_use = refine_fn(
            z=z, t=t, t_cont=t_cont, dt=dt,
            c_anchor=c_anchor, predict_velocity=predict_velocity,
            v_default=v_default,
        )
        # Standard Euler update with the (possibly refined) velocity.
        # diffusers' FlowMatchEulerDiscreteScheduler.step does exactly:
        #     prev_sample = sample + dt * model_output
        # with dt = sigma_next - sigma_now. We use that to stay verbatim.
        with torch.no_grad():
            z = sched.step(v_use.to(z.dtype), t, z_new.to(z.dtype),
                           return_dict=False)[0]

    return _decode_to_image(models, z)


def generate_sd3_ug_flow(
    prompt: str, neg_prompt: str = "blurry, low quality",
    *,
    models: SD3FlowModels,
    reward_model,                    # FrozenRewardModel-compatible
    height: int = 1024, width: int = 1024,
    num_steps: int = 28, cfg_scale: float = 7.0,
    seed: int = 42,
    K_ug: int = 4, eta_z: float = 0.1,
    rho_Q: float = 0.3,              # reward active during ρ_Q fraction
    gate_side: str = "noise",        # "noise"|"data"|"both"
    lambda_anneal_pow: float = 0.0,  # if >0, scale eta_z by t_cont^pow inside window
) -> Image.Image:
    """Universal Guidance (NFE-matched analogue of PG-MAP-FM): pure
    latent-reward gradient, no MAP regularization. Used as the ablation
    that isolates "what does MAP regularization buy on top of UG?"

    gate_side: "data" = last rho_Q fraction (data side, t_cont>=1-rho_Q;
    91.9% PS configuration); "noise" = first rho_Q fraction (t_cont<rho_Q;
    the visible-images mode); "both" activates on both ends of the trajectory.
    lambda_anneal_pow > 0 multiplies eta_z by (t_cont)^pow inside the active
    window, ramping reward strength toward the data side where the FM
    endpoint estimate is most accurate. 0 keeps eta_z constant.
    """
    n_active = max(1, int(round(num_steps * rho_Q)))
    gate_side = str(gate_side).lower()
    if gate_side not in ("noise", "data", "both"):
        raise ValueError(f"gate_side must be 'noise'|'data'|'both', got {gate_side!r}")
    anneal_p = float(lambda_anneal_pow)

    def refine_fn(z, t, t_cont, dt, c_anchor, predict_velocity, v_default):
        if gate_side == "data":
            active = (t_cont >= 1.0 - float(rho_Q))
        elif gate_side == "noise":
            active = (t_cont < float(rho_Q))
        else:  # both
            active = (t_cont < float(rho_Q)) or (t_cont >= 1.0 - float(rho_Q))
        if not active:
            return z, c_anchor, v_default

        eta_z_eff = float(eta_z)
        if anneal_p > 0.0:
            eta_z_eff = eta_z_eff * (max(float(t_cont), 0.0) ** anneal_p)

        def reward_fn(x1):
            return reward_model.score(x1, prompt)

        def decode_endpoint(z_endpoint):
            return _decode_endpoint_grad(models, z_endpoint)

        z_new = ug_flow_refine_step(
            predict_velocity=predict_velocity,
            z_t=z, t_for_model=t, t_cont=t_cont,
            c_anchor=c_anchor,
            decode_endpoint=decode_endpoint, reward_fn=reward_fn,
            K=K_ug, eta_z=eta_z_eff,
        )
        with torch.no_grad():
            v_use = predict_velocity(z_new, t, c_anchor)
        return z_new, c_anchor, v_use

    return _sample_with_callback(
        models, prompt, neg_prompt,
        height=height, width=width,
        num_steps=num_steps, cfg_scale=cfg_scale, seed=seed,
        refine_fn=refine_fn,
    )


def generate_sd3_pgmap_flow(
    prompt: str, neg_prompt: str = "blurry, low quality",
    *,
    models: SD3FlowModels,
    reward_model,
    height: int = 1024, width: int = 1024,
    num_steps: int = 28, cfg_scale: float = 7.0,
    seed: int = 42,
    K: int = 2, eta_c: float = 1e-3, eta_z: float = 5e-3,
    sigma_c: float = 1.0, gamma: float = 1.0, sigma_flow: float = 0.1,
    lambda_reward: float = 0.05,
    rho: float = 0.5, rho_Q: float = 0.3,
    optimize_c: bool = True, optimize_z: bool = True, use_reward: bool = True,
    use_likelihood: bool = True,
    use_prior_c: bool = True,
    use_prior_z: bool = True,
    gate_side: str = "noise",
    grad_norm_strategy: str = "unit",
    split_grads: bool = False,
) -> Image.Image:
    """Full PG-MAP-FM (per FlowMatching.tex): joint (c, z_t) optimization
    with flow consistency + Gaussian priors + reward, restricted to the
    rho fraction of denoising steps determined by gate_side.

    Setting optimize_c=False, use_reward=True → Reward-z (FM) ablation.
    Setting optimize_z=False, use_reward=False → MAP-c (FM) ablation.
    Setting use_likelihood=False keeps priors+reward without the flow
    consistency residual; this isolates whether the latent prior contributes
    over Universal Guidance.
    gate_side="data" mirrors the n=1632 UG-FM configuration; "noise" is the
    current default.
    grad_norm_strategy: "unit" (default; whole-gradient unit-norm) or
    "raw" (no normalization). Unit-norm is the historical default.
    """
    ref_cfg = FlowRefinementConfig(K=K, eta_c=eta_c, eta_z=eta_z)
    prior   = FlowPriorConfig(sigma_c=sigma_c, gamma=gamma, sigma_flow=sigma_flow)
    rcfg    = FlowRewardConfig(lambda_reward=lambda_reward,
                                grad_norm_strategy=grad_norm_strategy,
                                split_grads=split_grads)
    gate_side = str(gate_side).lower()
    if gate_side not in ("noise", "data", "both"):
        raise ValueError(f"gate_side must be 'noise'|'data'|'both', got {gate_side!r}")

    def refine_fn(z, t, t_cont, dt, c_anchor, predict_velocity, v_default):
        if gate_side == "data":
            in_refine_window = (t_cont >= 1.0 - float(rho))
            reward_active = (t_cont >= 1.0 - float(rho_Q))
        elif gate_side == "noise":
            in_refine_window = (t_cont < float(rho))
            reward_active = (t_cont < float(rho_Q))
        else:  # both
            in_refine_window = (t_cont < float(rho)) or (t_cont >= 1.0 - float(rho))
            reward_active = (t_cont < float(rho_Q)) or (t_cont >= 1.0 - float(rho_Q))
        if not in_refine_window:
            return z, c_anchor, v_default

        def reward_fn(x1):
            return reward_model.score(x1, prompt)

        def decode_endpoint(z_endpoint):
            return _decode_endpoint_grad(models, z_endpoint)

        c_star, z_star, v_star = pgmap_flow_refine_step(
            predict_velocity=predict_velocity,
            z_t=z, t_for_model=t, t_cont=t_cont, dt=dt,
            c_anchor=c_anchor, z_t_anchor=z,
            ref_cfg=ref_cfg, prior=prior, reward_cfg=rcfg,
            decode_endpoint=decode_endpoint, reward_fn=reward_fn,
            optimize_c=optimize_c, optimize_z=optimize_z,
            use_reward=use_reward, reward_active=reward_active,
            use_likelihood=use_likelihood,
            use_prior_c=use_prior_c, use_prior_z=use_prior_z,
        )
        return z_star, c_star, v_star

    return _sample_with_callback(
        models, prompt, neg_prompt,
        height=height, width=width,
        num_steps=num_steps, cfg_scale=cfg_scale, seed=seed,
        refine_fn=refine_fn,
    )
