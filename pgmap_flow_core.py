"""PG-MAP for Flow Matching / Rectified Flow models.

Implements the FM-adapted version of PG-MAP per ``FlowMatching.tex``:

    J^FM_t(c, z_t) =
        - 1/(2 σ_flow²) || z_{t+Δt}^{ref} - (z_t + Δt · v_θ(z_t, t, c)) ||²   (flow consistency)
        - 1/(2 σ_c²)    || c - c_0 ||²                                        (cond. prior)
        - 1/(2 σ_z(t)²) || z_t - z_t^{base} ||²                               (latent prior)
        + λ Q( D(x̂_1(z_t, c)), y )                                            (reward)

with endpoint estimate (analogue of Tweedie):
    x̂_1 = z_t + (1 - t) · v_θ(z_t, t, c)

and schedule-adaptive latent-prior variance:
    σ_z(t) = γ · (1 - t)        (rectified-flow convention; tighter near t=1)

Three variants are recoverable as special cases:
    MAP-c          : optimize_c=True,  optimize_z=False, use_reward=False
    Reward-z (UG)  : optimize_c=False, optimize_z=True,  use_reward=True,
                     and likelihood/prior weights → 0 to recover plain UG-FM
    PG-MAP-FM      : optimize_c=True,  optimize_z=True,  use_reward=True

The model is treated as a black-box velocity predictor:
    v_θ(z_t, t, c, **extra) -> ε_t   (same shape as z_t)

so the same routine runs against SD3, SD3.5, Flux, AuraFlow, etc. The
caller only has to supply ``predict_velocity`` and ``decode_endpoint``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Configuration dataclasses (mirror pgmap_config.py for the diffusion case)
# ---------------------------------------------------------------------------

@dataclass
class FlowRefinementConfig:
    """Inner-loop optimizer settings for FM PG-MAP."""
    K: int = 2                  # joint gradient-ascent steps
    eta_c: float = 1e-3         # conditioning step size
    eta_z: float = 5e-3         # latent step size


@dataclass
class FlowPriorConfig:
    """Prior strengths for FM PG-MAP."""
    sigma_c: float = 1.0        # conditioning prior σ_c
    gamma: float = 1.0          # latent prior scale: σ_z(t) = γ · (1 - t)
    sigma_flow: float = 0.1     # flow-consistency residual scale σ_flow


@dataclass
class FlowRewardConfig:
    """Preference reward settings."""
    lambda_reward: float = 0.05
    grad_norm_strategy: str = "unit"   # "unit" | "raw" | "adaptive"
    # When True, split the gradient computation into a separate
    # ∂(likelihood+prior)/∂vars and ∂reward/∂vars pass and combine via the
    # selected strategy. This mirrors the DDPM core's "split-gradient" path.
    # When False, the FM legacy whole-gradient unit-norm is used (which can
    # let the likelihood term dominate the combined direction at small lambda).
    split_grads: bool = False


# ---------------------------------------------------------------------------
# Objective terms (separate so they're individually patchable for ablations)
# ---------------------------------------------------------------------------

def flow_consistency_residual(
    z_t: torch.Tensor,
    z_next_ref: torch.Tensor,
    v_pred: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """One-step Euler consistency residual:
        r = z_next_ref - (z_t - dt · v_pred)

    Diffusers' FlowMatchEulerDiscreteScheduler stores the transformer output
    as ``v_pred = z_0 - x_1 = -v_FM`` and steps via
    ``z_new = z + (sigma_next - sigma) · v_pred = z - dt · v_pred`` with
    ``dt = sigma - sigma_next > 0``. So the predicted next-state from
    ``(z_t, c)`` is ``z_t - dt · v_pred``.
    """
    return z_next_ref - (z_t - dt * v_pred)


def flow_endpoint_estimate(
    z_t: torch.Tensor,
    v_pred: torch.Tensor,
    t_now: float,
) -> torch.Tensor:
    """FM endpoint estimate (Tweedie analogue) under the diffusers convention
    ``v_pred = z_0 - x_1 = -v_FM``:

        x̂_1 = z_t - (1 - t) · v_pred

    For the linear FM interpolant ``z_t = (1-t)·z_0 + t·x_1`` and
    ``v_FM = x_1 - z_0``, the blueprint formula is ``x̂_1 = z_t + (1-t)·v_FM``;
    diffusers SD3/AuraFlow output ``-v_FM`` so the sign is inverted here.
    """
    return z_t - (1.0 - t_now) * v_pred


def _obj_terms_flow(
    z_t: torch.Tensor,
    c: torch.Tensor,
    v_pred: torch.Tensor,
    z_t_anchor: torch.Tensor,
    z_next_ref: torch.Tensor,
    mu_c: torch.Tensor,
    t_now: float,
    dt: float,
    prior: FlowPriorConfig,
    use_likelihood: bool = True,
    use_prior_c: bool = True,
    use_prior_z: bool = True,
) -> torch.Tensor:
    """Per-batch-element scalar objective (without the reward term).

    When all three flags are False, returns a detached zero tensor (no graph
    connection): a graph-connected ``z.sum()*0`` zero introduces a redundant
    backward path that, when combined with a separately-graph-connected
    reward term and a unit-norm gradient step at large eta_z (≥0.1), can
    drive z off-manifold producing NaN latents in fp16. The detached-zero
    path keeps obj_main + lam*r_val mathematically equivalent to lam*r_val
    while routing the autograd graph cleanly through the reward only.
    """
    if not (use_likelihood or use_prior_c or use_prior_z):
        return torch.zeros(z_t.shape[0], device=z_t.device, dtype=torch.float32)

    z_t_f = z_t.float()
    out = z_t_f.flatten(1).sum(dim=-1) * 0.0  # graph-connected zero
    if c.requires_grad:
        out = out + c.float().flatten(1).sum(dim=-1) * 0.0

    if use_likelihood:
        r = flow_consistency_residual(z_t_f, z_next_ref.float(),
                                       v_pred.float(), dt)
        sigma_flow = max(float(prior.sigma_flow), 1e-12)
        out = out - 0.5 / (sigma_flow ** 2) * (r * r).flatten(1).sum(dim=-1)

    if use_prior_c:
        diff_c = c.float() - mu_c.float()
        sigma_c = max(float(prior.sigma_c), 1e-12)
        out = out - 0.5 / (sigma_c ** 2) * (diff_c * diff_c).flatten(1).sum(dim=-1)

    if use_prior_z:
        # Rectified-flow schedule: tighter trust region as t → 1
        sigma_z = max(float(prior.gamma) * (1.0 - float(t_now)), 1e-12)
        diff_z = z_t_f - z_t_anchor.float()
        out = out - 0.5 / (sigma_z ** 2) * (diff_z * diff_z).flatten(1).sum(dim=-1)

    return out


# ---------------------------------------------------------------------------
# Inner-loop refine step
# ---------------------------------------------------------------------------

def pgmap_flow_refine_step(
    *,
    predict_velocity: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    z_t: torch.Tensor,
    t_for_model: torch.Tensor,            # scheduler-discrete timestep (passed verbatim
                                          # to the transformer; whatever units it uses)
    t_cont: float,                        # continuous time ∈ [0, 1] for the MAP math
                                          # (rectified-flow convention: 0 = noise, 1 = data)
    dt: float,                            # forward Euler step Δt (sigma_now - sigma_next)
    c_anchor: torch.Tensor,               # original conditioning embedding c_0
    z_t_anchor: torch.Tensor,             # unperturbed latent at t (= z_t at entry)
    ref_cfg: FlowRefinementConfig,
    prior: FlowPriorConfig,
    reward_cfg: FlowRewardConfig,
    decode_endpoint: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    reward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    optimize_c: bool = True,
    optimize_z: bool = True,
    use_reward: bool = True,
    reward_active: bool = True,
    use_likelihood: bool = True,
    use_prior_c: bool = True,
    use_prior_z: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Joint K-step gradient ascent on (c, z_t) under the FM MAP objective.

    Two distinct time arguments are required because flow-matching schedulers
    use a discrete timestep (e.g., a float in [0, num_train_timesteps]) for
    the model call but the MAP objective is written in continuous t ∈ [0, 1].
    Caller must pass:
        t_for_model : whatever the predict_velocity callable expects
        t_cont      : continuous time, used for σ_z(t) = γ(1-t) and endpoint x̂_1
    """
    K = max(int(ref_cfg.K), 1)
    eta_c = float(ref_cfg.eta_c)
    eta_z = float(ref_cfg.eta_z)
    lam   = float(reward_cfg.lambda_reward) if (use_reward and reward_active) else 0.0
    norm_strategy = reward_cfg.grad_norm_strategy
    split_grads = bool(getattr(reward_cfg, "split_grads", False)) and (lam > 0.0) \
                  and decode_endpoint is not None and reward_fn is not None
    t_cont_f = float(t_cont)

    # Reference next-state: one Euler step from the unperturbed latent and
    # original conditioning. This is the "observation" the consistency term
    # tries to explain. Sign matches the diffusers Euler step
    # ``z + (sigma_next - sigma) · v = z - dt · v`` with dt > 0.
    with torch.no_grad():
        v_ref = predict_velocity(z_t_anchor, t_for_model, c_anchor).detach()
        z_next_ref = z_t_anchor - dt * v_ref

    # Initialize optimization variables.
    c_var = c_anchor.clone()
    z_var = z_t_anchor.clone()
    if optimize_c:
        c_var = c_var.requires_grad_(True)
    if optimize_z:
        z_var = z_var.requires_grad_(True)

    if not (optimize_c or optimize_z):
        return c_var.detach(), z_var.detach(), v_ref

    for _k in range(K):
        # Make sure the leaves are detached so a fresh graph is built each step.
        c_iter = c_var.detach().requires_grad_(optimize_c) if optimize_c else c_anchor
        z_iter = z_var.detach().requires_grad_(optimize_z) if optimize_z else z_t_anchor

        v = predict_velocity(z_iter, t_for_model, c_iter)

        obj_main = _obj_terms_flow(
            z_t=z_iter, c=c_iter, v_pred=v,
            z_t_anchor=z_t_anchor, z_next_ref=z_next_ref,
            mu_c=c_anchor, t_now=t_cont_f,
            dt=dt, prior=prior,
            use_likelihood=use_likelihood,
            use_prior_c=(use_prior_c and optimize_c),
            use_prior_z=(use_prior_z and optimize_z),
        )

        opt_vars = []
        if optimize_c:
            opt_vars.append(c_iter)
        if optimize_z:
            opt_vars.append(z_iter)

        if split_grads:
            # ── Split-gradient path (DDPM-equivalent) ────────────────
            # Two backward passes over the same UNet forward graph.
            # Pass 1: ∂(likelihood + priors)/∂vars
            grads_main = torch.autograd.grad(
                -obj_main.mean(), opt_vars,
                retain_graph=True, create_graph=False,
            )
            # Pass 2: ∂(reward)/∂vars, reusing the existing graph.
            x1_hat = decode_endpoint(flow_endpoint_estimate(z_iter, v, t_cont_f))
            r_val = reward_fn(x1_hat).mean()
            grads_reward = torch.autograd.grad(
                -r_val, opt_vars,
                retain_graph=False, create_graph=False,
            )

            # Combine per the selected strategy.
            grads = []
            for g_main, g_reward in zip(grads_main, grads_reward):
                if norm_strategy == "unit":
                    g_r_flat = g_reward.flatten(1)
                    g_r_norm = g_r_flat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    g_r_unit = (g_r_flat / g_r_norm).view_as(g_reward)
                    grads.append(g_main + lam * g_r_unit)
                elif norm_strategy == "adaptive":
                    g_m_norm = g_main.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    g_r_norm = g_reward.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    w_eff = (lam * g_m_norm / g_r_norm).clamp(max=5.0)
                    w_eff = w_eff.view(-1, *([1] * (g_reward.dim() - 1)))
                    grads.append(g_main + w_eff * g_reward)
                else:  # raw
                    grads.append(g_main + lam * g_reward)
        else:
            # ── Whole-gradient legacy path ───────────────────────────
            obj = obj_main
            if lam > 0.0 and decode_endpoint is not None and reward_fn is not None:
                x1_hat = decode_endpoint(flow_endpoint_estimate(z_iter, v, t_cont_f))
                r_val = reward_fn(x1_hat)
                obj = obj + lam * r_val

            grads = torch.autograd.grad(
                -obj.mean(), opt_vars,
                retain_graph=False, create_graph=False,
            )

        with torch.no_grad():
            grad_iter = iter(grads)
            if optimize_c:
                gc = next(grad_iter)
                if norm_strategy == "unit" and not split_grads:
                    n = gc.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    gc = (gc.flatten(1) / n).view_as(gc)
                c_var = (c_iter.detach() - eta_c * gc).detach()
            if optimize_z:
                gz = next(grad_iter)
                if norm_strategy == "unit" and not split_grads:
                    n = gz.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    gz = (gz.flatten(1) / n).view_as(gz)
                z_var = (z_iter.detach() - eta_z * gz).detach()

    # Final velocity at (c*, z_t*) for the outer Euler step.
    with torch.no_grad():
        v_star = predict_velocity(z_var.detach(), t_for_model, c_var.detach()).detach()

    return c_var.detach(), z_var.detach(), v_star


# ---------------------------------------------------------------------------
# Pure-UG-FM reference (drops likelihood and prior; reward gradient only)
# ---------------------------------------------------------------------------

def ug_flow_refine_step(
    *,
    predict_velocity: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    z_t: torch.Tensor,
    t_for_model: torch.Tensor,
    t_cont: float,
    c_anchor: torch.Tensor,
    decode_endpoint: Callable[[torch.Tensor], torch.Tensor],
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    K: int = 4,
    eta_z: float = 0.1,
) -> torch.Tensor:
    """Universal Guidance for flow matching: pure latent reward gradient,
    no MAP regularization. NFE-matched to PG-MAP-FM (K_UG = 4 inner steps).
    Endpoint estimate uses continuous time t_cont ∈ [0, 1].
    """
    t_cont_f = float(t_cont)
    z_var = z_t.detach()
    for _k in range(K):
        z_iter = z_var.detach().requires_grad_(True)
        v = predict_velocity(z_iter, t_for_model, c_anchor)
        x1_hat = decode_endpoint(flow_endpoint_estimate(z_iter, v, t_cont_f))
        r_val = reward_fn(x1_hat)
        (g_z,) = torch.autograd.grad(r_val.mean(), z_iter,
                                      retain_graph=False, create_graph=False)
        n = g_z.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
        g_z = (g_z.flatten(1) / n).view_as(g_z)
        z_var = (z_iter.detach() + eta_z * g_z).detach()
    return z_var
