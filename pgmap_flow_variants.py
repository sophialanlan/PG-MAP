"""
FM-MAP-cz variants designed to mitigate the Euler step Jacobian amplification
problem on flow-matching backbones (SD3.5-medium).

Background (from MAP_neurips2026.tex M2 analysis):
    A perturbation delta_z injected at scheduler step k_0 propagates to the
    endpoint as
        delta_z^(K) ~ prod_{j=k_0}^{K-1}(I + dt_j * d v_theta / d z) * delta_z^(k_0)
    On DDPM the per-step SDE noise injection wipes out accumulation; on FM
    the deterministic ODE preserves and *multiplicatively amplifies* it.
    The paper's full PG-MAP-FM (joint c + z) thus collapses to ~52% PickScore
    while UG-FM (no MAP regularization, data-side reward only) reaches
    91.9% PickScore. The framework story falls apart because the only
    z-perturbation that survives Euler amplification is one applied near
    the data side -- which UG already exploits without MAP.

Three variants in this file try to recover MAP_cz_FM from this collapse:

  FM_V1 ("c-only"): drop z-optimization entirely. c perturbations affect the
    model output through v_theta(z, t, c) but do NOT directly accumulate in
    z-space, so they bypass the amplification problem by construction.
    Implementation: just call generate_sd3_pgmap_flow with optimize_z=False.
    No new code in this module -- handled in the runner.

  FM_V2 ("amplification-compensating eta_z schedule"): scale eta_z inversely
    with the expected residual-step amplification budget. At scheduler step
    k of K total, remaining propagation is (K-k) Euler steps. With assumed
    per-step amplification A ~ 1.05, compensate via
        eta_z_eff(k) = eta_z * (1 / A^(K-k))
    so that the *effective* z-perturbation reaching the endpoint is roughly
    eta_z-magnitude regardless of where in the trajectory it was injected.
    This is a "schedule-aware step size" analogous to Variant A on DDPM.

  FM_V3 ("trust-region proximal with t-aware radius"): cap the per-step
    z-displacement magnitude using a smooth proximal step rule
        z_new = z + (eta_z / (1 + eta_z * ||g||/tau(t))) * g
    with trust radius tau(t) = tau_0 * t (loose at data-side, tight at
    noise-side). When ||g|| is small the step is ~eta_z * g (vanilla);
    when ||g|| is large the step saturates to tau(t) * g/||g||, giving a
    hard cap that scales with t. Analogous to Variant C on DDPM.

All three variants are evaluated in the noise-side gate window (t < rho)
where the paper's reference PG-MAP-FM fails, since the data-side gate is
where UG-FM already wins without MAP regularization. The point of these
variants is to recover MAP_cz behavior on the noise-side, where amplification
is the deepest -- so success here is the most stringent test.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from pgmap_flow_core import (
    FlowRefinementConfig,
    FlowPriorConfig,
    FlowRewardConfig,
    flow_endpoint_estimate,
    pgmap_flow_refine_step,
    _obj_terms_flow,
    flow_consistency_residual,
)


# ---------------------------------------------------------------------------
# FM_V2: amplification-compensating eta_z schedule
# ---------------------------------------------------------------------------

def make_v2_refine_fn(
    *,
    models,
    reward_model,
    prompt: str,
    K: int,
    eta_c: float,
    eta_z_base: float,
    sigma_c: float,
    gamma: float,
    sigma_flow: float,
    lambda_reward: float,
    rho: float,
    rho_Q: float,
    optimize_c: bool,
    optimize_z: bool,
    use_reward: bool,
    grad_norm_strategy: str,
    split_grads: bool,
    gate_side: str,
    A_amp: float = 1.05,
    num_steps: int = 28,
    decode_endpoint_grad_fn=None,
):
    """Build a refine_fn that scales eta_z by 1/A^(K_remaining(t)).

    K_remaining is computed from t_cont via the linear-shift scheduler
    approximation: at t_cont, the fraction of remaining trajectory is
    (1 - t_cont), so K_remaining ~ ceil((1 - t_cont) * num_steps).
    """
    from pgmap_flow_sd3 import _decode_endpoint_grad   # local import to avoid cycle

    prior = FlowPriorConfig(sigma_c=sigma_c, gamma=gamma, sigma_flow=sigma_flow)
    rcfg  = FlowRewardConfig(lambda_reward=lambda_reward,
                              grad_norm_strategy=grad_norm_strategy,
                              split_grads=split_grads)

    def refine_fn(z, t, t_cont, dt, c_anchor, predict_velocity, v_default):
        if gate_side == "data":
            in_window = (t_cont >= 1.0 - float(rho))
            reward_active = (t_cont >= 1.0 - float(rho_Q))
        elif gate_side == "noise":
            in_window = (t_cont < float(rho))
            reward_active = (t_cont < float(rho_Q))
        else:
            in_window = (t_cont < float(rho)) or (t_cont >= 1.0 - float(rho))
            reward_active = (t_cont < float(rho_Q)) or (t_cont >= 1.0 - float(rho_Q))
        if not in_window:
            return z, c_anchor, v_default

        # Amplification compensation: residual Euler steps from t_cont -> 1.
        # Use ceil((1 - t_cont) * num_steps).
        import math
        k_remaining = max(1, math.ceil((1.0 - float(t_cont)) * num_steps))
        eta_z_eff = float(eta_z_base) / (float(A_amp) ** k_remaining)

        ref_cfg_eff = FlowRefinementConfig(K=K, eta_c=eta_c, eta_z=eta_z_eff)

        def reward_fn(x1):
            return reward_model.score(x1, prompt)

        def decode_endpoint(z_endpoint):
            return _decode_endpoint_grad(models, z_endpoint)

        c_star, z_star, v_star = pgmap_flow_refine_step(
            predict_velocity=predict_velocity,
            z_t=z, t_for_model=t, t_cont=t_cont, dt=dt,
            c_anchor=c_anchor, z_t_anchor=z,
            ref_cfg=ref_cfg_eff, prior=prior, reward_cfg=rcfg,
            decode_endpoint=decode_endpoint, reward_fn=reward_fn,
            optimize_c=optimize_c, optimize_z=optimize_z,
            use_reward=use_reward, reward_active=reward_active,
            use_likelihood=True, use_prior_c=True, use_prior_z=True,
        )
        return z_star, c_star, v_star

    return refine_fn


# ---------------------------------------------------------------------------
# FM_V3: trust-region proximal step with t-aware radius
# ---------------------------------------------------------------------------

def pgmap_flow_v3_refine_step(
    *,
    predict_velocity: Callable,
    z_t: torch.Tensor,
    t_for_model: torch.Tensor,
    t_cont: float,
    dt: float,
    c_anchor: torch.Tensor,
    z_t_anchor: torch.Tensor,
    ref_cfg: FlowRefinementConfig,
    prior: FlowPriorConfig,
    reward_cfg: FlowRewardConfig,
    decode_endpoint: Optional[Callable] = None,
    reward_fn: Optional[Callable] = None,
    optimize_c: bool = True,
    optimize_z: bool = True,
    use_reward: bool = True,
    reward_active: bool = True,
    tau_0: float = 1.0,
):
    """K-step ascent with trust-region proximal update on z (only).

    For each inner step k:
      - Compute combined gradient g for (c, z) jointly under the FM MAP loss
      - c update: vanilla SGD with eta_c
      - z update (PROXIMAL):
            tau_t = tau_0 * t_cont
            z_new = z - (eta_z / (1 + eta_z * ||g_z||/tau_t)) * g_z
        which caps the per-step z displacement to ~tau_t when ||g_z|| is large.

    The smooth proximal cap means at noise-side (t_cont small), tau_t is small,
    so the step is heavily damped regardless of the raw gradient magnitude.
    At data-side (t_cont -> 1), tau_t = tau_0, the cap is loose, behaving like
    vanilla SGD.
    """
    K = max(int(ref_cfg.K), 1)
    eta_c = float(ref_cfg.eta_c)
    eta_z = float(ref_cfg.eta_z)
    lam = float(reward_cfg.lambda_reward) if (use_reward and reward_active) else 0.0
    norm_strategy = reward_cfg.grad_norm_strategy
    t_cont_f = float(t_cont)

    with torch.no_grad():
        v_ref = predict_velocity(z_t_anchor, t_for_model, c_anchor).detach()
        z_next_ref = z_t_anchor - dt * v_ref

    c_var = c_anchor.clone()
    z_var = z_t_anchor.clone()
    if optimize_c:
        c_var = c_var.requires_grad_(True)
    if optimize_z:
        z_var = z_var.requires_grad_(True)
    if not (optimize_c or optimize_z):
        return c_var.detach(), z_var.detach(), v_ref

    tau_t = max(float(tau_0) * t_cont_f, 1e-6)

    for _k in range(K):
        c_iter = c_var.detach().requires_grad_(optimize_c) if optimize_c else c_anchor
        z_iter = z_var.detach().requires_grad_(optimize_z) if optimize_z else z_t_anchor

        v = predict_velocity(z_iter, t_for_model, c_iter)
        obj_main = _obj_terms_flow(
            z_t=z_iter, c=c_iter, v_pred=v,
            z_t_anchor=z_t_anchor, z_next_ref=z_next_ref,
            mu_c=c_anchor, t_now=t_cont_f, dt=dt, prior=prior,
            use_likelihood=True,
            use_prior_c=optimize_c,
            use_prior_z=optimize_z,
        )
        opt_vars = []
        if optimize_c: opt_vars.append(c_iter)
        if optimize_z: opt_vars.append(z_iter)

        obj = obj_main
        if lam > 0.0 and decode_endpoint is not None and reward_fn is not None:
            x1_hat = decode_endpoint(flow_endpoint_estimate(z_iter, v, t_cont_f))
            r_val = reward_fn(x1_hat)
            obj = obj + lam * r_val
        grads = torch.autograd.grad(
            -obj.mean(), opt_vars, retain_graph=False, create_graph=False,
        )

        with torch.no_grad():
            grad_iter = iter(grads)
            if optimize_c:
                gc = next(grad_iter)
                if norm_strategy == "unit":
                    n = gc.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    gc = (gc.flatten(1) / n).view_as(gc)
                c_var = (c_iter.detach() - eta_c * gc).detach()
            if optimize_z:
                gz = next(grad_iter)
                if norm_strategy == "unit":
                    n = gz.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    gz = (gz.flatten(1) / n).view_as(gz)
                # Smooth proximal step on z (key difference vs reference).
                gz_flat = gz.flatten(1)
                gz_norm = gz_flat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                factor = eta_z / (1.0 + eta_z * gz_norm / tau_t)
                while factor.dim() < gz.dim():
                    factor = factor.unsqueeze(-1)
                z_var = (z_iter.detach() - factor * gz).detach()

    with torch.no_grad():
        v_star = predict_velocity(z_var.detach(), t_for_model, c_var.detach()).detach()
    return c_var.detach(), z_var.detach(), v_star


def make_v3_refine_fn(
    *,
    models,
    reward_model,
    prompt: str,
    K: int,
    eta_c: float,
    eta_z: float,
    sigma_c: float,
    gamma: float,
    sigma_flow: float,
    lambda_reward: float,
    rho: float,
    rho_Q: float,
    optimize_c: bool,
    optimize_z: bool,
    use_reward: bool,
    grad_norm_strategy: str,
    split_grads: bool,
    gate_side: str,
    tau_0: float = 1.0,
):
    from pgmap_flow_sd3 import _decode_endpoint_grad

    ref_cfg = FlowRefinementConfig(K=K, eta_c=eta_c, eta_z=eta_z)
    prior = FlowPriorConfig(sigma_c=sigma_c, gamma=gamma, sigma_flow=sigma_flow)
    rcfg = FlowRewardConfig(lambda_reward=lambda_reward,
                             grad_norm_strategy=grad_norm_strategy,
                             split_grads=split_grads)

    def refine_fn(z, t, t_cont, dt, c_anchor, predict_velocity, v_default):
        if gate_side == "data":
            in_window = (t_cont >= 1.0 - float(rho))
            reward_active = (t_cont >= 1.0 - float(rho_Q))
        elif gate_side == "noise":
            in_window = (t_cont < float(rho))
            reward_active = (t_cont < float(rho_Q))
        else:
            in_window = (t_cont < float(rho)) or (t_cont >= 1.0 - float(rho))
            reward_active = (t_cont < float(rho_Q)) or (t_cont >= 1.0 - float(rho_Q))
        if not in_window:
            return z, c_anchor, v_default

        def reward_fn(x1):
            return reward_model.score(x1, prompt)

        def decode_endpoint(z_endpoint):
            return _decode_endpoint_grad(models, z_endpoint)

        c_star, z_star, v_star = pgmap_flow_v3_refine_step(
            predict_velocity=predict_velocity,
            z_t=z, t_for_model=t, t_cont=t_cont, dt=dt,
            c_anchor=c_anchor, z_t_anchor=z,
            ref_cfg=ref_cfg, prior=prior, reward_cfg=rcfg,
            decode_endpoint=decode_endpoint, reward_fn=reward_fn,
            optimize_c=optimize_c, optimize_z=optimize_z,
            use_reward=use_reward, reward_active=reward_active,
            tau_0=tau_0,
        )
        return z_star, c_star, v_star

    return refine_fn


# ---------------------------------------------------------------------------
# Top-level dispatch generators (mirror generate_sd3_pgmap_flow signature)
# ---------------------------------------------------------------------------

def generate_sd3_fm_v2(
    prompt: str, neg_prompt: str = "blurry, low quality",
    *,
    models, reward_model,
    height: int = 1024, width: int = 1024,
    num_steps: int = 28, cfg_scale: float = 7.0,
    seed: int = 42,
    K: int = 2, eta_c: float = 1e-3, eta_z: float = 0.5,
    sigma_c: float = 1.0, gamma: float = 1.0, sigma_flow: float = 0.1,
    lambda_reward: float = 0.05,
    rho: float = 0.5, rho_Q: float = 0.3,
    optimize_c: bool = True, optimize_z: bool = True, use_reward: bool = True,
    gate_side: str = "noise",
    grad_norm_strategy: str = "unit",
    split_grads: bool = False,
    A_amp: float = 1.05,
):
    """FM-V2: amplification-compensating eta_z schedule.

    eta_z is the *base* step size; the actual per-step value is
    eta_z / A_amp^(remaining_euler_steps). Default eta_z=0.5 is large because
    it gets divided by A_amp^k_rem where k_rem can be up to num_steps.
    """
    from pgmap_flow_sd3 import _sample_with_callback
    refine_fn = make_v2_refine_fn(
        models=models, reward_model=reward_model, prompt=prompt,
        K=K, eta_c=eta_c, eta_z_base=eta_z,
        sigma_c=sigma_c, gamma=gamma, sigma_flow=sigma_flow,
        lambda_reward=lambda_reward, rho=rho, rho_Q=rho_Q,
        optimize_c=optimize_c, optimize_z=optimize_z, use_reward=use_reward,
        grad_norm_strategy=grad_norm_strategy, split_grads=split_grads,
        gate_side=gate_side, A_amp=A_amp, num_steps=num_steps,
    )
    return _sample_with_callback(
        models, prompt, neg_prompt,
        height=height, width=width,
        num_steps=num_steps, cfg_scale=cfg_scale, seed=seed,
        refine_fn=refine_fn,
    )


def generate_sd3_fm_v3(
    prompt: str, neg_prompt: str = "blurry, low quality",
    *,
    models, reward_model,
    height: int = 1024, width: int = 1024,
    num_steps: int = 28, cfg_scale: float = 7.0,
    seed: int = 42,
    K: int = 2, eta_c: float = 1e-3, eta_z: float = 0.1,
    sigma_c: float = 1.0, gamma: float = 1.0, sigma_flow: float = 0.1,
    lambda_reward: float = 0.05,
    rho: float = 0.5, rho_Q: float = 0.3,
    optimize_c: bool = True, optimize_z: bool = True, use_reward: bool = True,
    gate_side: str = "noise",
    grad_norm_strategy: str = "unit",
    split_grads: bool = False,
    tau_0: float = 1.0,
):
    """FM-V3: trust-region proximal step with t-aware trust radius."""
    from pgmap_flow_sd3 import _sample_with_callback
    refine_fn = make_v3_refine_fn(
        models=models, reward_model=reward_model, prompt=prompt,
        K=K, eta_c=eta_c, eta_z=eta_z,
        sigma_c=sigma_c, gamma=gamma, sigma_flow=sigma_flow,
        lambda_reward=lambda_reward, rho=rho, rho_Q=rho_Q,
        optimize_c=optimize_c, optimize_z=optimize_z, use_reward=use_reward,
        grad_norm_strategy=grad_norm_strategy, split_grads=split_grads,
        gate_side=gate_side, tau_0=tau_0,
    )
    return _sample_with_callback(
        models, prompt, neg_prompt,
        height=height, width=width,
        num_steps=num_steps, cfg_scale=cfg_scale, seed=seed,
        refine_fn=refine_fn,
    )
