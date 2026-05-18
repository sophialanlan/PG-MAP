"""
PG-MAP Core: Objective Function and Inner Gradient Loop
=========================================================

Implements Algorithm 1 from the paper. This is the main algorithmic file
that computes the PG-MAP objective and runs the inner-loop gradient ascent
on (c, z_t) at each denoising step.

The objective J_t(c, z_t) has four terms:
    1. Forward-consistency:  -1/(2*beta_t) * ||z_t - sqrt(alpha_t) * z_prev_hat||^2
    2. Conditioning prior:   -1/(2*sigma_c^2) * ||c - mu_t||^2
    3. Latent prior:         -1/(2*sigma_z^2) * ||z_t - z_t_ddim||^2
       where sigma_z(t) = gamma * sqrt(1 - alpha_bar_t)   [schedule-adaptive]
    4. Preference reward:    lambda * Q(x_hat_0, prompt)

Key optimization (vs original): a single UNet forward pass computes eps,
z0_hat, and z_prev_hat together. The split-gradient path then issues two
backward passes (retain_graph=True for the first) rather than two separate
UNet forwards. This halves UNet evaluations in the inner loop.

CFG in the outer denoising loop is also batched into a single UNet call.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from pgmap_config import PriorConfig, RewardConfig, RefinementConfig


# -----------------------------------------------------------------------
# Helper: gather scheduler quantities
# -----------------------------------------------------------------------

def _gather_alphas_cumprod(sched, t_idx: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Gather cumulative alpha values, broadcasting to (B,1,1,1)."""
    ac = sched.alphas_cumprod.to(device=like.device, dtype=like.dtype)[t_idx]
    return ac.view(-1, 1, 1, 1)


def _gather_alphas(sched, t_idx: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Gather per-step alpha values as (B,)."""
    return sched.alphas.to(device=like.device, dtype=like.dtype)[t_idx]


# -----------------------------------------------------------------------
# Fused UNet forward pass
# -----------------------------------------------------------------------

def _fwd(
    unet,
    sched,
    z_t: torch.Tensor,
    t_train: torch.Tensor,
    t_prev: torch.Tensor,
    c: torch.Tensor,
    pooled: Optional[torch.Tensor] = None,
    time_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single UNet forward → (eps, z0_hat, z_prev_hat).

    Computes eps once and derives both the Tweedie clean estimate (z0_hat)
    and the DDIM previous-step proposal (z_prev_hat) from it, avoiding any
    redundant UNet calls that existed in the original _z0_pred /
    _ddim_proposal_prev pair.

    All outputs are in float32 for gradient stability.
    """
    z_in = sched.scale_model_input(z_t, t_train)

    unet_dtype = next(unet.parameters()).dtype
    kwargs: dict = {"encoder_hidden_states": c.to(unet_dtype)}
    if pooled is not None and time_ids is not None:
        kwargs["added_cond_kwargs"] = {
            "text_embeds": pooled.to(unet_dtype),
            "time_ids": time_ids.to(unet_dtype),
        }

    eps = unet(z_in.to(unet_dtype), t_train, **kwargs).sample.float()

    ac = sched.alphas_cumprod.to(device=z_t.device, dtype=torch.float32)
    ab_t = ac[t_train].view(-1, 1, 1, 1)

    # Tweedie estimate
    z0_hat = (z_t.float() - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()

    # DDIM proposal for previous step
    ab_prev = torch.where(
        t_prev >= 0,
        ac[t_prev.clamp(min=0)],
        torch.ones_like(t_prev, dtype=torch.float32, device=z_t.device),
    ).view(-1, 1, 1, 1)
    z_prev_hat = ab_prev.sqrt() * z0_hat + (1.0 - ab_prev).sqrt() * eps

    return eps, z0_hat, z_prev_hat


# -----------------------------------------------------------------------
# CFG batched UNet call (for outer denoising loop)
# -----------------------------------------------------------------------

def cfg_eps(
    unet,
    sched,
    z_t: torch.Tensor,
    t_train: torch.Tensor,
    c_uncond: torch.Tensor,
    c_cond: torch.Tensor,
    guidance_scale: float,
    pooled_uncond: Optional[torch.Tensor] = None,
    pooled_cond: Optional[torch.Tensor] = None,
    time_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched CFG noise prediction in a single UNet call.

    Concatenates [uncond, cond] along the batch dimension so the UNet
    runs once instead of twice. Equivalent to the original two-call pattern
    but ~2x faster for CFG overhead.
    """
    B = z_t.shape[0]
    z_in = torch.cat([z_t, z_t], dim=0)         # (2B, C, H, W)
    t_in = t_train.repeat(2)                      # (2B,)
    c_in = torch.cat([c_uncond, c_cond], dim=0)  # (2B, L, D)

    unet_dtype = next(unet.parameters()).dtype
    kwargs: dict = {"encoder_hidden_states": c_in.to(unet_dtype)}

    if pooled_uncond is not None and pooled_cond is not None and time_ids is not None:
        pooled_in = torch.cat([pooled_uncond, pooled_cond], dim=0)
        time_in = time_ids.repeat(2, 1)
        kwargs["added_cond_kwargs"] = {
            "text_embeds": pooled_in.to(unet_dtype),
            "time_ids": time_in.to(unet_dtype),
        }

    z_in_scaled = sched.scale_model_input(z_in, t_train[0])
    out = unet(z_in_scaled.to(unet_dtype), t_in, **kwargs).sample

    eps_u, eps_c = out[:B], out[B:]
    return (eps_u + guidance_scale * (eps_c - eps_u)).to(z_t.dtype)


# -----------------------------------------------------------------------
# Differentiable VAE decode for reward scoring
# -----------------------------------------------------------------------

def _decode_to_pixels(vae, z0_hat: torch.Tensor, target_hw: int = 0) -> torch.Tensor:
    """Differentiable VAE decode: z0_hat -> pixel values in [0, 1].

    Args:
        vae:       The VAE decoder.
        z0_hat:    Clean latent estimate (B, C, H, W), with gradient.
        target_hw: If > 0, downsample the latent to (target_hw//8, target_hw//8)
                   before decoding.  Reward models resize to 224 anyway, so
                   setting target_hw=224 decodes at 28×28 latents instead of the
                   full 128×128 (SDXL) — roughly 20× faster VAE forward while
                   the 224×224 reward-model input quality is essentially unchanged.
                   Use 0 (default) to keep the full resolution for final output.
    """
    scale = getattr(vae.config, "scaling_factor", 0.18215)
    z = z0_hat.float() / scale
    if target_hw > 0:
        lat = target_hw // 8  # e.g. 224//8 = 28
        z = F.interpolate(z, size=(lat, lat), mode="bilinear", align_corners=False)
    vae_dtype = next(vae.parameters()).dtype
    imgs = vae.decode(z.to(dtype=vae_dtype)).sample
    return (imgs / 2 + 0.5).clamp(0, 1)


# -----------------------------------------------------------------------
# Fused cond+uncond UNet forward (for inner-step CFG fusion)
# -----------------------------------------------------------------------

def _fwd_fused_cfg(
    unet,
    sched,
    z_var: torch.Tensor,
    t_train: torch.Tensor,
    t_prev: torch.Tensor,
    c_var: torch.Tensor,
    c_uncond: torch.Tensor,
    pooled: Optional[torch.Tensor] = None,
    pooled_uncond: Optional[torch.Tensor] = None,
    time_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched cond+uncond UNet forward for inner-step CFG fusion.

    Runs UNet on [z_uncond_detached, z_cond] × [c_uncond_detached, c_var] in
    one 2B forward pass.  Gradients flow only through the conditioned half
    (since uncond inputs are detached), so this is gradient-equivalent to a
    standard cond-only _fwd call while also delivering eps_uncond for CFG.

    Returns:
        eps_u:      (B,) uncond noise prediction — detached, no gradient.
        eps_c:      (B,) cond noise prediction  — carries gradients.
        z0_hat:     Tweedie clean estimate from eps_c.
        z_prev_hat: DDIM prev-step proposal from eps_c.
    """
    B = z_var.shape[0]

    # Uncond inputs are fully detached — grads flow only via the cond half.
    z_batch = torch.cat([z_var.detach(), z_var], dim=0)   # (2B, C, H, W)
    c_batch = torch.cat([c_uncond.float().detach(), c_var], dim=0)  # (2B, L, D)

    unet_dtype = next(unet.parameters()).dtype
    kwargs: dict = {"encoder_hidden_states": c_batch.to(unet_dtype)}
    if pooled is not None and pooled_uncond is not None and time_ids is not None:
        p_batch = torch.cat([pooled_uncond.to(unet_dtype), pooled.to(unet_dtype)], dim=0)
        kwargs["added_cond_kwargs"] = {
            "text_embeds": p_batch,
            "time_ids": time_ids.repeat(2, 1).to(unet_dtype),
        }

    z_in = sched.scale_model_input(z_batch, t_train[0])
    t_in = t_train.repeat(2)
    eps_batch = unet(z_in.to(unet_dtype), t_in, **kwargs).sample.float()

    eps_u = eps_batch[:B].detach()   # no gradient needed for uncond
    eps_c = eps_batch[B:]            # gradients flow through this

    ac = sched.alphas_cumprod.to(device=z_var.device, dtype=torch.float32)
    ab_t = ac[t_train].view(-1, 1, 1, 1)
    z0_hat = (z_var.float() - (1.0 - ab_t).sqrt() * eps_c) / ab_t.sqrt()

    ab_prev = torch.where(
        t_prev >= 0,
        ac[t_prev.clamp(min=0)],
        torch.ones_like(t_prev, dtype=torch.float32, device=z_var.device),
    ).view(-1, 1, 1, 1)
    z_prev_hat = ab_prev.sqrt() * z0_hat + (1.0 - ab_prev).sqrt() * eps_c

    return eps_u, eps_c, z0_hat, z_prev_hat


# -----------------------------------------------------------------------
# PG-MAP non-reward objective terms (differentiable, no UNet call)
# -----------------------------------------------------------------------

def _obj_terms(
    z_t: torch.Tensor,
    t_train: torch.Tensor,
    c: torch.Tensor,
    mu_t: torch.Tensor,
    z_t_anchor: torch.Tensor,
    prior: PriorConfig,
    alpha_bar_t: torch.Tensor,
    alpha_t: torch.Tensor,
    z_prev_hat: torch.Tensor,
) -> torch.Tensor:
    """Compute consistency + prior terms from pre-computed z_prev_hat.

    This is called with an already-computed z_prev_hat (from _fwd), so it
    adds no UNet calls. Returns (B,) tensor.
    """
    z_t_f = z_t.float()

    # Term 1: Forward consistency
    beta_t = (1.0 - alpha_t).clamp(min=1e-12).view(-1, 1, 1, 1)
    alpha_t_b = alpha_t.view(-1, 1, 1, 1)
    resid = z_t_f - alpha_t_b.sqrt() * z_prev_hat
    ll = -0.5 * (resid * resid / beta_t).flatten(1).sum(dim=-1)

    # Term 2: Conditioning prior
    diff_c = c.float() - mu_t.float()
    prior_c = -0.5 / (prior.sigma_c ** 2) * (diff_c * diff_c).flatten(1).sum(dim=-1)

    # Term 3: Schedule-adaptive latent prior
    sigma_z = prior.gamma * (1.0 - alpha_bar_t.view(-1)).clamp(min=1e-12).sqrt()
    diff_z = z_t_f - z_t_anchor.float()
    prior_z = -0.5 / (sigma_z ** 2).clamp(min=1e-12).view(-1, 1, 1, 1) * (diff_z * diff_z)
    prior_z = prior_z.flatten(1).sum(dim=-1)

    return ll + prior_c + prior_z


# -----------------------------------------------------------------------
# PG-MAP Inner Gradient Loop
# -----------------------------------------------------------------------

def pgmap_refine_step(
    unet,
    sched,
    vae,
    z_t_init: torch.Tensor,
    t_train: torch.Tensor,
    t_prev: torch.Tensor,
    c_init: torch.Tensor,
    mu_t: torch.Tensor,
    prior: PriorConfig,
    ref_cfg: RefinementConfig,
    alpha_bar_t: torch.Tensor,
    reward_model=None,
    reward_config: Optional[RewardConfig] = None,
    prompt: str = "",
    optimize_c: bool = True,
    optimize_z: bool = True,
    use_reward: bool = True,
    pooled: Optional[torch.Tensor] = None,
    time_ids: Optional[torch.Tensor] = None,
    step_frac: float = 1.0,
    # --- CFG fusion: fuse last inner step with outer CFG call ---
    fuse_cfg: bool = False,
    c_uncond: Optional[torch.Tensor] = None,
    pooled_uncond: Optional[torch.Tensor] = None,
    guidance_scale: float = 7.5,
    # --- Low-res VAE decode for reward (0 = full resolution) ---
    reward_decode_size: int = 224,
    # --- Persistent Adam state for c across denoising steps ---
    # Pass the dict returned by a previous call to reuse c momentum.
    # Only meaningful when optimizer="adam" and optimize_c=True.
    # z momentum is always reset because z lives in a different noise space each step.
    adam_state_c: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
    """Run K inner gradient ascent steps on (c, z_t).

    One UNet forward pass per inner step (down from two in the original).
    For the split-gradient path the computation graph is retained after the
    first backward so the second backward (reward only) reuses it without
    issuing a new UNet forward.

    CFG fusion (fuse_cfg=True):
        The final inner step is run as a batched cond+uncond forward
        (_fwd_fused_cfg) rather than a plain cond-only _fwd.  This delivers
        eps_uncond for free, so the caller can skip the usual outer cfg_eps
        call entirely — saving one full UNet forward per refine step.
        The returned eps_fused is a first-order approximation: it is computed
        at (z_var, c_var) rather than the post-update (z_t, c_t), which
        introduces O(eta * ||grad||) error (negligible for eta_c=1e-4 and
        eta_z=5e-3).

    Low-res VAE decode (reward_decode_size > 0):
        When reward is active, z0_hat is downsampled to
        (reward_decode_size//8)^2 latent before VAE decoding.  Since reward
        models (PickScore, CLIP) resize to 224 anyway, decoding at 28×28
        latents instead of 128×128 (SDXL) cuts VAE cost ≈20× with negligible
        quality impact on the reward signal.

    Args:
        step_frac:          Fraction of reward-active window elapsed.
        fuse_cfg:           Enable fused inner+outer CFG (requires c_uncond).
        c_uncond:           Unconditional text embedding for fusion.
        pooled_uncond:      Unconditional pooled embedding (SDXL only).
        guidance_scale:     CFG scale used when building fused eps.
        reward_decode_size: Latent height/width × 8 for VAE decode.  0 = full.

    Returns:
        (c_refined, z_t_refined, eps_fused, adam_state_c_out):
            - c_refined, z_t_refined: Both detached, in original dtype.
            - eps_fused: CFG noise estimate for the DDIM update when
              fuse_cfg=True and fusion was applied; None otherwise.
            - adam_state_c_out: dict with keys "m1", "m2", "step" for c,
              or None when not using Adam. Pass back as adam_state_c on the
              next denoising step to carry momentum across timesteps.
    """
    if not optimize_c and not optimize_z:
        return c_init.detach(), z_t_init.detach(), None, None

    orig_dtype = z_t_init.dtype
    c = c_init.detach().clone().float()
    z_t = z_t_init.detach().clone().float()
    z_t_anchor = z_t_init.detach().float()

    alpha_t = _gather_alphas(sched, t_train, z_t)  # (B,)

    active_reward = (
        use_reward
        and reward_model is not None
        and reward_config is not None
        and reward_config.lambda_reward > 0
    )
    use_split_grads = (
        active_reward
        and reward_config.grad_norm_strategy in ("unit", "adaptive")
    )

    # CFG fusion: fuse the last inner step with the outer CFG call when the
    # uncond embedding is available.  Works for both reward and non-reward steps.
    _do_fuse = fuse_cfg and (c_uncond is not None)
    _K = ref_cfg.K

    # Effective lambda: optionally ramp from 0 → lambda_reward over reward steps
    if active_reward and getattr(reward_config, 'lambda_ramp', False):
        effective_lambda = reward_config.lambda_reward * step_frac
    else:
        effective_lambda = reward_config.lambda_reward if active_reward else 0.0

    # Adam state for c: optionally persistent across denoising steps.
    # z momentum is always reset — z lives in a different noise space each step.
    use_adam = (ref_cfg.optimizer == "adam")
    if use_adam:
        b1, b2, aeps = ref_cfg.adam_beta1, ref_cfg.adam_beta2, ref_cfg.adam_eps
        if optimize_c and adam_state_c is not None:
            m1_c  = adam_state_c["m1"].clone()
            m2_c  = adam_state_c["m2"].clone()
            _c_step_offset = adam_state_c["step"]   # bias-correction offset
        else:
            m1_c  = torch.zeros_like(c) if optimize_c else None
            m2_c  = torch.zeros_like(c) if optimize_c else None
            _c_step_offset = 0
        m1_z = torch.zeros_like(z_t) if optimize_z else None
        m2_z = torch.zeros_like(z_t) if optimize_z else None

    eps_fused: Optional[torch.Tensor] = None  # returned when fuse_cfg applies
    # GS: update c first, re-forward at (c_new, z) to get z gradient
    use_gs = getattr(ref_cfg, 'gauss_seidel', False) and optimize_c and optimize_z

    for k in range(_K):
        is_last_step = (k == _K - 1)
        c_step_idx = _c_step_offset + k + 1 if use_adam else k + 1
        z_step_idx = k + 1

        # ── Gauss-Seidel branch: two forwards per step ───────────────
        if use_gs:
            # ---- Step 1: ∇_c at (c_var, z_fixed) → update c --------
            c_var1 = c.detach().clone().requires_grad_(True)
            z_fixed = z_t.detach()
            _, z0h1, zph1 = _fwd(unet, sched, z_fixed, t_train, t_prev,
                                  c_var1, pooled, time_ids)
            if use_split_grads:
                om1 = _obj_terms(z_fixed, t_train, c_var1, mu_t, z_t_anchor,
                                  prior, alpha_bar_t, alpha_t, zph1).mean()
                (gc_main,) = torch.autograd.grad(-om1, [c_var1],
                                                 retain_graph=True, create_graph=False)
                px1 = _decode_to_pixels(vae, z0h1, reward_decode_size)
                Q1  = reward_model.score(px1, prompt).mean()
                (gc_rew,) = torch.autograd.grad(-Q1, [c_var1],
                                                retain_graph=False, create_graph=False)
                (grad_c,) = _combine_grads((gc_main,), (gc_rew,),
                                           reward_config.grad_norm_strategy, effective_lambda)
            elif active_reward:
                px1  = _decode_to_pixels(vae, z0h1, reward_decode_size)
                Q1   = reward_model.score(px1, prompt)
                obj1 = _obj_terms(z_fixed, t_train, c_var1, mu_t, z_t_anchor,
                                   prior, alpha_bar_t, alpha_t, zph1) + effective_lambda * Q1
                (grad_c,) = torch.autograd.grad(-obj1.mean(), [c_var1],
                                                retain_graph=False, create_graph=False)
            else:
                obj1 = _obj_terms(z_fixed, t_train, c_var1, mu_t, z_t_anchor,
                                   prior, alpha_bar_t, alpha_t, zph1)
                (grad_c,) = torch.autograd.grad(-obj1.mean(), [c_var1],
                                                retain_graph=False, create_graph=False)

            with torch.no_grad():
                if use_adam:
                    m1_c = b1 * m1_c + (1 - b1) * grad_c
                    m2_c = b2 * m2_c + (1 - b2) * grad_c.pow(2)
                    mh1  = m1_c / (1 - b1 ** c_step_idx)
                    mh2  = m2_c / (1 - b2 ** c_step_idx)
                    c = c_var1.detach() - ref_cfg.eta_c * mh1 / (mh2.sqrt() + aeps)
                else:
                    c = c_var1.detach() - ref_cfg.eta_c * grad_c
                if ref_cfg.clamp_c_norm is not None:
                    n = c.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    c = (c.flatten(1) * (ref_cfg.clamp_c_norm / n).clamp(max=1.0)).view_as(c)

            # ---- Step 2: ∇_z at (c_new, z_var) → update z ----------
            c2    = c.detach()
            z_var2 = z_t.detach().clone().requires_grad_(True)
            if _do_fuse and is_last_step:
                eu2, ec2, z0h2, zph2 = _fwd_fused_cfg(
                    unet, sched, z_var2, t_train, t_prev,
                    c2, c_uncond, pooled, pooled_uncond, time_ids)
                eps_fused = (eu2 + guidance_scale * (ec2.detach() - eu2)).to(orig_dtype)
            else:
                _, z0h2, zph2 = _fwd(unet, sched, z_var2, t_train, t_prev,
                                      c2, pooled, time_ids)
            if use_split_grads:
                om2 = _obj_terms(z_var2, t_train, c2, mu_t, z_t_anchor,
                                  prior, alpha_bar_t, alpha_t, zph2).mean()
                (gz_main,) = torch.autograd.grad(-om2, [z_var2],
                                                 retain_graph=True, create_graph=False)
                px2 = _decode_to_pixels(vae, z0h2, reward_decode_size)
                Q2  = reward_model.score(px2, prompt).mean()
                (gz_rew,) = torch.autograd.grad(-Q2, [z_var2],
                                                retain_graph=False, create_graph=False)
                (grad_z,) = _combine_grads((gz_main,), (gz_rew,),
                                           reward_config.grad_norm_strategy, effective_lambda)
            elif active_reward:
                px2  = _decode_to_pixels(vae, z0h2, reward_decode_size)
                Q2   = reward_model.score(px2, prompt)
                obj2 = _obj_terms(z_var2, t_train, c2, mu_t, z_t_anchor,
                                   prior, alpha_bar_t, alpha_t, zph2) + effective_lambda * Q2
                (grad_z,) = torch.autograd.grad(-obj2.mean(), [z_var2],
                                                retain_graph=False, create_graph=False)
            else:
                obj2 = _obj_terms(z_var2, t_train, c2, mu_t, z_t_anchor,
                                   prior, alpha_bar_t, alpha_t, zph2)
                (grad_z,) = torch.autograd.grad(-obj2.mean(), [z_var2],
                                                retain_graph=False, create_graph=False)

            with torch.no_grad():
                if use_adam:
                    m1_z = b1 * m1_z + (1 - b1) * grad_z
                    m2_z = b2 * m2_z + (1 - b2) * grad_z.pow(2)
                    mh1z = m1_z / (1 - b1 ** z_step_idx)
                    mh2z = m2_z / (1 - b2 ** z_step_idx)
                    z_t = z_var2.detach() - ref_cfg.eta_z * mh1z / (mh2z.sqrt() + aeps)
                else:
                    z_t = z_var2.detach() - ref_cfg.eta_z * grad_z
                if ref_cfg.clamp_z_delta is not None:
                    delta = z_t - z_t_anchor
                    n = delta.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    z_t = z_t_anchor + (delta.flatten(1) * (ref_cfg.clamp_z_delta / n).clamp(max=1.0)).view_as(z_t)
            continue  # skip Jacobi path below

        # ── Jacobi branch (original) ─────────────────────────────────
        c_var = c.detach().clone().requires_grad_(optimize_c)
        z_var = z_t.detach().clone().requires_grad_(optimize_z)
        opt_vars = [v for cond, v in ((optimize_c, c_var), (optimize_z, z_var)) if cond]

        # ── UNet forward ────────────────────────────────────────────────
        if _do_fuse and is_last_step:
            # Fused cond+uncond: delivers eps_u for CFG while computing the
            # same gradient-carrying eps_c as a plain _fwd would.
            eps_u, eps_c, z0_hat, z_prev_hat = _fwd_fused_cfg(
                unet, sched, z_var, t_train, t_prev, c_var, c_uncond,
                pooled, pooled_uncond, time_ids,
            )
            # Pre-compute fused CFG eps at (z_var, c_var) — slight approximation
            # vs. post-update (z_t, c_t), valid for small learning rates.
            eps_fused = (eps_u + guidance_scale * (eps_c.detach() - eps_u)).to(orig_dtype)
        else:
            _, z0_hat, z_prev_hat = _fwd(
                unet, sched, z_var, t_train, t_prev, c_var, pooled, time_ids
            )

        # ── Gradient computation ─────────────────────────────────────────
        if use_split_grads:
            # Consistency + prior (backward 1, keep graph for reward)
            obj_main = _obj_terms(
                z_var, t_train, c_var, mu_t, z_t_anchor,
                prior, alpha_bar_t, alpha_t, z_prev_hat,
            ).mean()
            grads_main = torch.autograd.grad(
                -obj_main, opt_vars, retain_graph=True, create_graph=False,
            )

            # Reward only (backward 2, reuses existing graph — no new UNet call).
            # Use low-res VAE decode: ~20× faster for SDXL; reward model resizes
            # to 224 anyway so quality impact on the reward signal is negligible.
            pixels = _decode_to_pixels(vae, z0_hat, target_hw=reward_decode_size)
            Q = reward_model.score(pixels, prompt).mean()
            grads_reward = torch.autograd.grad(
                -Q, opt_vars, retain_graph=False, create_graph=False,
            )

            combined_grads = _combine_grads(
                grads_main, grads_reward,
                reward_config.grad_norm_strategy,
                effective_lambda,
            )

        elif active_reward:
            # Single backward with combined objective
            pixels = _decode_to_pixels(vae, z0_hat, target_hw=reward_decode_size)
            Q = reward_model.score(pixels, prompt)
            obj = _obj_terms(
                z_var, t_train, c_var, mu_t, z_t_anchor,
                prior, alpha_bar_t, alpha_t, z_prev_hat,
            ) + effective_lambda * Q
            combined_grads = torch.autograd.grad(
                -obj.mean(), opt_vars, retain_graph=False, create_graph=False,
            )

        else:
            obj = _obj_terms(
                z_var, t_train, c_var, mu_t, z_t_anchor,
                prior, alpha_bar_t, alpha_t, z_prev_hat,
            )
            combined_grads = torch.autograd.grad(
                -obj.mean(), opt_vars, retain_graph=False, create_graph=False,
            )

        # ── Parameter update (SGD or Adam) ───────────────────────────────
        with torch.no_grad():
            grad_iter = iter(combined_grads)

            if optimize_c:
                grad_c = next(grad_iter)
                if use_adam:
                    m1_c = b1 * m1_c + (1 - b1) * grad_c
                    m2_c = b2 * m2_c + (1 - b2) * grad_c.pow(2)
                    m1_hat = m1_c / (1 - b1 ** c_step_idx)
                    m2_hat = m2_c / (1 - b2 ** c_step_idx)
                    c = c_var.detach() - ref_cfg.eta_c * m1_hat / (m2_hat.sqrt() + aeps)
                else:
                    c = c_var.detach() - ref_cfg.eta_c * grad_c
                if ref_cfg.clamp_c_norm is not None:
                    n = c.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    c = (c.flatten(1) * (ref_cfg.clamp_c_norm / n).clamp(max=1.0)).view_as(c)

            if optimize_z:
                grad_z = next(grad_iter)
                if use_adam:
                    m1_z = b1 * m1_z + (1 - b1) * grad_z
                    m2_z = b2 * m2_z + (1 - b2) * grad_z.pow(2)
                    m1_hat = m1_z / (1 - b1 ** z_step_idx)
                    m2_hat = m2_z / (1 - b2 ** z_step_idx)
                    z_t = z_var.detach() - ref_cfg.eta_z * m1_hat / (m2_hat.sqrt() + aeps)
                else:
                    z_t = z_var.detach() - ref_cfg.eta_z * grad_z
                if ref_cfg.clamp_z_delta is not None:
                    delta = z_t - z_t_anchor
                    n = delta.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    z_t = z_t_anchor + (
                        delta.flatten(1) * (ref_cfg.clamp_z_delta / n).clamp(max=1.0)
                    ).view_as(z_t)

    # Build output Adam state for c (to optionally pass to the next denoising step)
    out_adam_state_c: Optional[dict] = None
    if use_adam and optimize_c and m1_c is not None:
        out_adam_state_c = {
            "m1":   m1_c.detach(),
            "m2":   m2_c.detach(),
            "step": _c_step_offset + _K,
        }

    return c.detach().to(orig_dtype), z_t.detach().to(orig_dtype), eps_fused, out_adam_state_c


# -----------------------------------------------------------------------
# Gradient combination strategies
# -----------------------------------------------------------------------

def _combine_grads(
    grads_main: Tuple[torch.Tensor, ...],
    grads_reward: Tuple[torch.Tensor, ...],
    strategy: str,
    lambda_reward: float,
) -> Tuple[torch.Tensor, ...]:
    combined = []
    for g_main, g_reward in zip(grads_main, grads_reward):
        if strategy == "unit":
            g_r_flat = g_reward.flatten(1)
            g_r_norm = g_r_flat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            g_r_unit = (g_r_flat / g_r_norm).view_as(g_reward)
            combined.append(g_main + lambda_reward * g_r_unit)
        elif strategy == "adaptive":
            g_m_norm = g_main.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
            g_r_norm = g_reward.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
            w_eff = (lambda_reward * g_m_norm / g_r_norm).clamp(max=5.0)
            combined.append(g_main + w_eff.view(-1, *([1] * (g_reward.dim() - 1))) * g_reward)
        else:
            combined.append(g_main + lambda_reward * g_reward)
    return tuple(combined)


# -----------------------------------------------------------------------
# Legacy aliases (kept for any external callers)
# -----------------------------------------------------------------------

def _predict_eps(unet, sched, z_t, t_train, c, pooled=None, time_ids=None):
    """Kept for backward compatibility. Prefer _fwd()."""
    t_prev = torch.full_like(t_train, -1)
    eps, _, _ = _fwd(unet, sched, z_t, t_train, t_prev, c, pooled, time_ids)
    return eps


def _z0_pred(unet, sched, z_t, t_train, c, pooled=None, time_ids=None):
    """Kept for backward compatibility. Prefer _fwd()."""
    t_prev = torch.full_like(t_train, -1)
    _, z0_hat, _ = _fwd(unet, sched, z_t, t_train, t_prev, c, pooled, time_ids)
    return z0_hat
