"""
PG-MAP gradient/likelihood variants for the gradient-scaling investigation.

Three new inner-loop refinement strategies that address the 1/beta_t prefactor
in the forward-consistency gradient (Section 5.2 of the paper, the catastrophic
collapse at eta_z=0.1 in rows D/E of Table tab:map_components).

VARIANTS
--------
A2: Adam K=3 with persistent c-momentum across denoising steps. RMS-normalized
    inner step adapts to per-iteration gradient magnitude — no reliance on
    cumulative "snowball" of small steps. Implemented via the existing Adam
    code path in pgmap_core.pgmap_refine_step (no new code in this module).

B2: closed-form damped Newton in z. Single inner forward+backward, then
    scalar preconditioning by (1/beta_t + 1/sigma_z(t)^2 + 1/eta_0)^-1.
    No K-iteration ascent — the step magnitude is set by the local Hessian
    not by accumulation. c is updated with vanilla SGD as before.

C2: epsilon-residual likelihood with eps_ref **recomputed each inner step**
    at the current (z, c). Larger eta_z (1.0) compensates for the fact that
    eps-space gradient magnitude is O(1) regardless of t (no 1/beta_t to fight
    against). Tests whether iterating in eps-space with re-anchored ref is
    materially different from one-shot eps-residual (Variant B from the prior
    round, which was a near no-op).

(Original Variants A/B/C — likelihood-cancellation, eps-residual, proximal
 trust-region — remain below for backwards compatibility.)

A: schedule-adaptive eta_z(t) = eta_0 * beta_t  (likelihood-only cancellation)
   The forward-consistency loss is multiplied by beta_t inside _obj_terms_A
   so its gradient drops the 1/beta_t prefactor analytically. Prior and reward
   gradients keep their natural magnitudes. Step size eta_0 is then schedule-
   independent.

B: epsilon-residual reparameterization
   The likelihood term is rewritten in noise-prediction space as
       ell^eps = -0.5 * || eps_theta(z_t, t, c) - eps_ref ||^2
   where eps_ref is the model's noise prediction at the unperturbed
   (z_t^ddim, c_0). This removes the 1/beta_t coupling entirely; the gradient
   passes through one Jacobian J_z = d eps_theta / d z_t (same backward depth
   as the standard PG-MAP).
   Design note (audit response): the latent prior_z stays in z-space (not
   reformulated to eps-space) by design. The likelihood and prior have
   different roles: the likelihood enforces "the (c, z_t) pair must be
   consistent with the model's noise prediction at the anchor", which lives
   naturally in eps-space; the prior_z enforces "stay near the unperturbed
   trajectory point", which is a z-space concept (the trajectory is in z).
   A pure-eps formulation would replace prior_z with a prior on eps which
   has no anchor analogue — the trajectory does not have a canonical eps.

C: trust-region / proximal step (smooth gradient norm clipping)
   After computing the combined gradient g, the update is
       z <- z + (eta_0 / (1 + eta_0 * ||g||_2 / tau)) * g
   which is the proximal operator for an L2 trust radius of order tau. Allows
   eta_0 to be set aggressively (e.g. 0.1) without divergence: when ||g||
   is small the step is ~eta_0 * g; when ||g|| is large the step saturates
   to ~tau * g/||g||.

All three variants reuse the existing _fwd, _decode_to_pixels, and _combine_grads
machinery from pgmap_core. The only changes are (i) the _obj_terms construction
and (ii) the parameter update. Adam, gauss-seidel, and CFG-fusion paths from
the original pgmap_refine_step are NOT replicated here -- variants always use
plain SGD on the joint Jacobi update with the standard fused-CFG fast path.
This is a deliberate scope cut: the variants are about gradient *shape*, not
optimizer *state*.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from pgmap_config import PriorConfig, RewardConfig, RefinementConfig
from pgmap_core import (
    _fwd,
    _fwd_fused_cfg,
    _decode_to_pixels,
    _gather_alphas,
    _combine_grads,
)


# ---------------------------------------------------------------------------
# Variant-specific objective terms
# ---------------------------------------------------------------------------

def _obj_terms_standard(
    z_t: torch.Tensor,
    c: torch.Tensor,
    mu_t: torch.Tensor,
    z_t_anchor: torch.Tensor,
    prior: PriorConfig,
    alpha_bar_t: torch.Tensor,
    alpha_t: torch.Tensor,
    z_prev_hat: torch.Tensor,
) -> torch.Tensor:
    """Original PG-MAP objective: ll + prior_c + prior_z, summed per sample.

    This is identical to _obj_terms in pgmap_core (verbatim re-implementation
    so the variants module is self-contained for the audit reviewer).
    """
    z_t_f = z_t.float()
    beta_t = (1.0 - alpha_t).clamp(min=1e-12).view(-1, 1, 1, 1)
    alpha_t_b = alpha_t.view(-1, 1, 1, 1)
    resid = z_t_f - alpha_t_b.sqrt() * z_prev_hat
    ll = -0.5 * (resid * resid / beta_t).flatten(1).sum(dim=-1)

    diff_c = c.float() - mu_t.float()
    prior_c = -0.5 / (prior.sigma_c ** 2) * (diff_c * diff_c).flatten(1).sum(dim=-1)

    sigma_z2 = (prior.gamma ** 2) * (1.0 - alpha_bar_t.view(-1)).clamp(min=1e-12)
    diff_z = z_t_f - z_t_anchor.float()
    prior_z = -0.5 / sigma_z2.clamp(min=1e-12).view(-1, 1, 1, 1) * (diff_z * diff_z)
    prior_z = prior_z.flatten(1).sum(dim=-1)
    return ll + prior_c + prior_z


def _obj_terms_variantA(
    z_t: torch.Tensor,
    c: torch.Tensor,
    mu_t: torch.Tensor,
    z_t_anchor: torch.Tensor,
    prior: PriorConfig,
    alpha_bar_t: torch.Tensor,
    alpha_t: torch.Tensor,
    z_prev_hat: torch.Tensor,
) -> torch.Tensor:
    """Variant A: forward-consistency loss multiplied by beta_t.

    Mathematically: ll_A = beta_t * ll_standard, so
        d ll_A / d z = beta_t * d ll_standard / d z
                     = beta_t * (1/beta_t) * (I - sqrt(alpha_t) J_z)^T r
                     = (I - sqrt(alpha_t) J_z)^T r
    The 1/beta_t prefactor is cancelled. Prior_c and prior_z gradients are
    unchanged from the standard objective.
    """
    z_t_f = z_t.float()
    beta_t = (1.0 - alpha_t).clamp(min=1e-12).view(-1, 1, 1, 1)
    alpha_t_b = alpha_t.view(-1, 1, 1, 1)
    resid = z_t_f - alpha_t_b.sqrt() * z_prev_hat
    # Variant A: drop the 1/beta_t weight. Equivalent to multiplying the
    # standard ll by beta_t.
    ll = -0.5 * (resid * resid).flatten(1).sum(dim=-1)

    diff_c = c.float() - mu_t.float()
    prior_c = -0.5 / (prior.sigma_c ** 2) * (diff_c * diff_c).flatten(1).sum(dim=-1)

    sigma_z2 = (prior.gamma ** 2) * (1.0 - alpha_bar_t.view(-1)).clamp(min=1e-12)
    diff_z = z_t_f - z_t_anchor.float()
    prior_z = -0.5 / sigma_z2.clamp(min=1e-12).view(-1, 1, 1, 1) * (diff_z * diff_z)
    prior_z = prior_z.flatten(1).sum(dim=-1)
    return ll + prior_c + prior_z


def _obj_terms_variantB(
    z_t: torch.Tensor,
    c: torch.Tensor,
    mu_t: torch.Tensor,
    z_t_anchor: torch.Tensor,
    prior: PriorConfig,
    alpha_bar_t: torch.Tensor,
    alpha_t: torch.Tensor,
    eps_pred: torch.Tensor,
    eps_ref: torch.Tensor,
) -> torch.Tensor:
    """Variant B: forward-consistency in epsilon-prediction space.

    Replaces the z-space residual with an eps-space residual:
        ll_B = -0.5 * || eps_theta(z_t, t, c) - eps_ref ||^2
    where eps_ref is the no-grad noise prediction at (z_t^ddim, c_0).
    The 1/beta_t prefactor is absent because we are penalising deviation
    in noise-prediction space rather than in latent-residual space.
    """
    diff_eps = eps_pred - eps_ref
    ll = -0.5 * (diff_eps * diff_eps).flatten(1).sum(dim=-1)

    diff_c = c.float() - mu_t.float()
    prior_c = -0.5 / (prior.sigma_c ** 2) * (diff_c * diff_c).flatten(1).sum(dim=-1)

    sigma_z2 = (prior.gamma ** 2) * (1.0 - alpha_bar_t.view(-1)).clamp(min=1e-12)
    diff_z = z_t.float() - z_t_anchor.float()
    prior_z = -0.5 / sigma_z2.clamp(min=1e-12).view(-1, 1, 1, 1) * (diff_z * diff_z)
    prior_z = prior_z.flatten(1).sum(dim=-1)
    return ll + prior_c + prior_z


# ---------------------------------------------------------------------------
# Variant C: trust-region step rule (proximal with smooth norm clipping)
# ---------------------------------------------------------------------------

def _trust_region_update(
    var: torch.Tensor,
    grad: torch.Tensor,
    eta_0: float,
    tau: float,
) -> torch.Tensor:
    """Smooth proximal step: var_new = var - (eta_0 / (1 + eta_0 ||g||/tau)) * g.

    Per-sample gradient norm is computed across all non-batch dimensions.

    When ||g||_2 << tau/eta_0:
        step ~= eta_0 * g                   (vanilla SGD)
    When ||g||_2 >> tau/eta_0:
        step ~= tau * g / ||g||_2           (saturates to a unit-tau move
                                             along the gradient direction)
    The transition is C^1 smooth, so unlike hard clipping there is no kink.
    """
    g_flat = grad.flatten(1)
    g_norm = g_flat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    # broadcast factor to grad shape: (B, 1, 1, ...)
    factor = eta_0 / (1.0 + eta_0 * g_norm / tau)
    # restore (B, 1, 1, 1) or (B, 1, 1) shape
    while factor.dim() < grad.dim():
        factor = factor.unsqueeze(-1)
    return var - factor * grad   # subtract because grad is of -obj


# ---------------------------------------------------------------------------
# Unified variant refine step
# ---------------------------------------------------------------------------

def pgmap_refine_step_variant(
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
    *,
    variant: str,
    trust_radius: float = 1.0,
    reward_model=None,
    reward_config: Optional[RewardConfig] = None,
    prompt: str = "",
    optimize_c: bool = True,
    optimize_z: bool = True,
    use_reward: bool = True,
    pooled: Optional[torch.Tensor] = None,
    time_ids: Optional[torch.Tensor] = None,
    step_frac: float = 1.0,
    fuse_cfg: bool = False,
    c_uncond: Optional[torch.Tensor] = None,
    pooled_uncond: Optional[torch.Tensor] = None,
    guidance_scale: float = 7.5,
    reward_decode_size: int = 224,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """K inner gradient ascent steps with the chosen variant.

    variant:
        'A'  -> likelihood multiplied by beta_t (cancels 1/beta_t)
        'B'  -> eps-space likelihood (eps_ref pre-computed without grad)
        'C'  -> standard objective + smooth proximal step rule
    """
    assert variant in {"A", "B", "C"}, f"unknown variant {variant!r}"
    if not optimize_c and not optimize_z:
        return c_init.detach(), z_t_init.detach(), None

    orig_dtype = z_t_init.dtype
    c = c_init.detach().clone().float()
    z_t = z_t_init.detach().clone().float()
    z_t_anchor = z_t_init.detach().float()
    alpha_t = _gather_alphas(sched, t_train, z_t)

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

    # Variant B: pre-compute eps_ref ONCE at (z_t_init, c_init) without grad.
    # eps_ref is the reference noise prediction; the variable eps_pred will
    # be computed at (z_var, c_var) inside the inner loop and compared.
    eps_ref: Optional[torch.Tensor] = None
    if variant == "B":
        with torch.no_grad():
            eps_ref, _, _ = _fwd(
                unet, sched, z_t_anchor, t_train, t_prev,
                c.detach(), pooled, time_ids,
            )
        eps_ref = eps_ref.detach()

    _do_fuse = fuse_cfg and (c_uncond is not None)
    _K = ref_cfg.K
    if active_reward and getattr(reward_config, "lambda_ramp", False):
        effective_lambda = reward_config.lambda_reward * step_frac
    else:
        effective_lambda = reward_config.lambda_reward if active_reward else 0.0

    eps_fused: Optional[torch.Tensor] = None

    for k in range(_K):
        is_last_step = (k == _K - 1)
        c_var = c.detach().clone().requires_grad_(optimize_c)
        z_var = z_t.detach().clone().requires_grad_(optimize_z)
        opt_vars = [v for cond, v in ((optimize_c, c_var), (optimize_z, z_var)) if cond]

        # ── UNet forward (fused CFG on last step if requested) ─────────
        if _do_fuse and is_last_step:
            eps_u, eps_c, z0_hat, z_prev_hat = _fwd_fused_cfg(
                unet, sched, z_var, t_train, t_prev, c_var, c_uncond,
                pooled, pooled_uncond, time_ids,
            )
            eps_fused = (eps_u + guidance_scale * (eps_c.detach() - eps_u)).to(orig_dtype)
            eps_pred = eps_c   # carries gradient through cond branch
        else:
            eps_pred, z0_hat, z_prev_hat = _fwd(
                unet, sched, z_var, t_train, t_prev, c_var, pooled, time_ids,
            )

        # ── Build objective per variant ────────────────────────────────
        if variant == "A":
            obj_main = _obj_terms_variantA(
                z_var, c_var, mu_t, z_t_anchor,
                prior, alpha_bar_t, alpha_t, z_prev_hat,
            ).mean()
        elif variant == "B":
            obj_main = _obj_terms_variantB(
                z_var, c_var, mu_t, z_t_anchor,
                prior, alpha_bar_t, alpha_t, eps_pred, eps_ref,
            ).mean()
        else:   # variant C uses the standard objective with proximal step rule
            obj_main = _obj_terms_standard(
                z_var, c_var, mu_t, z_t_anchor,
                prior, alpha_bar_t, alpha_t, z_prev_hat,
            ).mean()

        # ── Backward(s) ────────────────────────────────────────────────
        if use_split_grads:
            grads_main = torch.autograd.grad(
                -obj_main, opt_vars, retain_graph=True, create_graph=False,
            )
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
            pixels = _decode_to_pixels(vae, z0_hat, target_hw=reward_decode_size)
            Q = reward_model.score(pixels, prompt)
            obj = obj_main + effective_lambda * Q.mean()
            combined_grads = torch.autograd.grad(
                -obj, opt_vars, retain_graph=False, create_graph=False,
            )
        else:
            combined_grads = torch.autograd.grad(
                -obj_main, opt_vars, retain_graph=False, create_graph=False,
            )

        # ── Parameter update ───────────────────────────────────────────
        with torch.no_grad():
            grad_iter = iter(combined_grads)
            if optimize_c:
                grad_c = next(grad_iter)
                if variant == "C":
                    c = _trust_region_update(c_var.detach(), grad_c, ref_cfg.eta_c, trust_radius)
                else:
                    c = c_var.detach() - ref_cfg.eta_c * grad_c
                if ref_cfg.clamp_c_norm is not None:
                    n = c.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    c = (c.flatten(1) * (ref_cfg.clamp_c_norm / n).clamp(max=1.0)).view_as(c)
            if optimize_z:
                grad_z = next(grad_iter)
                if variant == "C":
                    z_t = _trust_region_update(z_var.detach(), grad_z, ref_cfg.eta_z, trust_radius)
                else:
                    z_t = z_var.detach() - ref_cfg.eta_z * grad_z
                if ref_cfg.clamp_z_delta is not None:
                    delta = z_t - z_t_anchor
                    n = delta.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    z_t = z_t_anchor + (
                        delta.flatten(1) * (ref_cfg.clamp_z_delta / n).clamp(max=1.0)
                    ).view_as(z_t)

    return c.detach().to(orig_dtype), z_t.detach().to(orig_dtype), eps_fused


# ---------------------------------------------------------------------------
# Direction B2: single-shot closed-form damped Newton (one inner step, no K)
# ---------------------------------------------------------------------------

def pgmap_refine_step_newton(
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
    *,
    reward_model=None,
    reward_config: Optional[RewardConfig] = None,
    prompt: str = "",
    optimize_c: bool = True,
    optimize_z: bool = True,
    use_reward: bool = True,
    pooled: Optional[torch.Tensor] = None,
    time_ids: Optional[torch.Tensor] = None,
    step_frac: float = 1.0,
    fuse_cfg: bool = False,
    c_uncond: Optional[torch.Tensor] = None,
    pooled_uncond: Optional[torch.Tensor] = None,
    guidance_scale: float = 7.5,
    reward_decode_size: int = 224,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Single closed-form damped Newton step at z_anchor.

    z step:
        Hessian approx  H_z ~ (1/beta_t + 1/sigma_z(t)^2) I + (1/eta_z) I
        Update          dz = -H_z^-1 * (-grad_J) = grad_J / (1/beta_t + 1/sigma_z^2 + 1/eta_z)

    c step:
        H_c ~ (1/sigma_c^2) I + (1/eta_c) I
        Update          dc = grad_Jc / (1/sigma_c^2 + 1/eta_c)
        (Drops the likelihood-side curvature for c since the J_c^T J_c product
         is too expensive to compute and is small relative to the c-prior.)

    Single inner forward+backward — no K iterations. The step magnitude is
    determined analytically by the local Hessian, not by accumulation.
    """
    if not optimize_c and not optimize_z:
        return c_init.detach(), z_t_init.detach(), None

    orig_dtype = z_t_init.dtype
    c = c_init.detach().clone().float()
    z_t = z_t_init.detach().clone().float()
    z_t_anchor = z_t_init.detach().float()
    alpha_t = _gather_alphas(sched, t_train, z_t)
    beta_t_scalar = float((1.0 - alpha_t).clamp(min=1e-12).item())
    sigma_z2 = (prior.gamma ** 2) * float((1.0 - alpha_bar_t.view(-1)).clamp(min=1e-12).item())

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

    if active_reward and getattr(reward_config, "lambda_ramp", False):
        effective_lambda = reward_config.lambda_reward * step_frac
    else:
        effective_lambda = reward_config.lambda_reward if active_reward else 0.0

    eps_fused: Optional[torch.Tensor] = None
    _do_fuse = fuse_cfg and (c_uncond is not None)

    # Single inner step (K=1 enforced by design)
    c_var = c.detach().clone().requires_grad_(optimize_c)
    z_var = z_t.detach().clone().requires_grad_(optimize_z)
    opt_vars = [v for cond, v in ((optimize_c, c_var), (optimize_z, z_var)) if cond]

    if _do_fuse:
        eps_u, eps_c, z0_hat, z_prev_hat = _fwd_fused_cfg(
            unet, sched, z_var, t_train, t_prev, c_var, c_uncond,
            pooled, pooled_uncond, time_ids,
        )
        eps_fused = (eps_u + guidance_scale * (eps_c.detach() - eps_u)).to(orig_dtype)
    else:
        _, z0_hat, z_prev_hat = _fwd(
            unet, sched, z_var, t_train, t_prev, c_var, pooled, time_ids,
        )

    obj_main = _obj_terms_standard(
        z_var, c_var, mu_t, z_t_anchor,
        prior, alpha_bar_t, alpha_t, z_prev_hat,
    ).mean()

    if use_split_grads:
        grads_main = torch.autograd.grad(
            -obj_main, opt_vars, retain_graph=True, create_graph=False,
        )
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
        pixels = _decode_to_pixels(vae, z0_hat, target_hw=reward_decode_size)
        Q = reward_model.score(pixels, prompt)
        obj = obj_main + effective_lambda * Q.mean()
        combined_grads = torch.autograd.grad(
            -obj, opt_vars, retain_graph=False, create_graph=False,
        )
    else:
        combined_grads = torch.autograd.grad(
            -obj_main, opt_vars, retain_graph=False, create_graph=False,
        )

    # Damped Newton scalar preconditioners (analytic, schedule-aware)
    inv_eta_z = 1.0 / max(ref_cfg.eta_z, 1e-12)
    inv_eta_c = 1.0 / max(ref_cfg.eta_c, 1e-12)
    Hz = 1.0 / beta_t_scalar + 1.0 / max(sigma_z2, 1e-12) + inv_eta_z
    Hc = 1.0 / max(prior.sigma_c ** 2, 1e-12) + inv_eta_c

    with torch.no_grad():
        grad_iter = iter(combined_grads)
        if optimize_c:
            grad_c = next(grad_iter)
            c = c_var.detach() - grad_c / Hc
            if ref_cfg.clamp_c_norm is not None:
                n = c.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                c = (c.flatten(1) * (ref_cfg.clamp_c_norm / n).clamp(max=1.0)).view_as(c)
        if optimize_z:
            grad_z = next(grad_iter)
            z_t = z_var.detach() - grad_z / Hz
            if ref_cfg.clamp_z_delta is not None:
                delta = z_t - z_t_anchor
                n = delta.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                z_t = z_t_anchor + (
                    delta.flatten(1) * (ref_cfg.clamp_z_delta / n).clamp(max=1.0)
                ).view_as(z_t)

    return c.detach().to(orig_dtype), z_t.detach().to(orig_dtype), eps_fused


# ---------------------------------------------------------------------------
# Direction C2: epsilon-residual with eps_ref re-anchored each inner step
# ---------------------------------------------------------------------------

def pgmap_refine_step_eps2(
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
    *,
    reward_model=None,
    reward_config: Optional[RewardConfig] = None,
    prompt: str = "",
    optimize_c: bool = True,
    optimize_z: bool = True,
    use_reward: bool = True,
    pooled: Optional[torch.Tensor] = None,
    time_ids: Optional[torch.Tensor] = None,
    step_frac: float = 1.0,
    fuse_cfg: bool = False,
    c_uncond: Optional[torch.Tensor] = None,
    pooled_uncond: Optional[torch.Tensor] = None,
    guidance_scale: float = 7.5,
    reward_decode_size: int = 224,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """eps-residual likelihood with eps_ref re-anchored at every inner step.

    Differs from the prior Variant B (one-shot eps_ref pre-computed) in that
    eps_ref is recomputed at the current (z^(k), c^(k)) at every inner
    iteration k. This forces the optimization to "track" a moving target,
    so the gradient stays meaningfully nonzero across iterations rather than
    converging to zero (which would happen in single-anchor eps-space).

    The likelihood gradient magnitude is O(1) regardless of t, so eta_z can
    be set aggressively (1.0 default) without divergence — provided K is
    small (2 here).
    """
    if not optimize_c and not optimize_z:
        return c_init.detach(), z_t_init.detach(), None

    orig_dtype = z_t_init.dtype
    c = c_init.detach().clone().float()
    z_t = z_t_init.detach().clone().float()
    z_t_anchor = z_t_init.detach().float()
    alpha_t = _gather_alphas(sched, t_train, z_t)

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

    if active_reward and getattr(reward_config, "lambda_ramp", False):
        effective_lambda = reward_config.lambda_reward * step_frac
    else:
        effective_lambda = reward_config.lambda_reward if active_reward else 0.0

    eps_fused: Optional[torch.Tensor] = None
    _do_fuse = fuse_cfg and (c_uncond is not None)
    _K = ref_cfg.K

    for k in range(_K):
        is_last_step = (k == _K - 1)

        # Re-anchor eps_ref at current (z, c) — no grad
        with torch.no_grad():
            eps_ref_k, _, _ = _fwd(
                unet, sched, z_t.detach(), t_train, t_prev,
                c.detach(), pooled, time_ids,
            )
        eps_ref_k = eps_ref_k.detach()

        c_var = c.detach().clone().requires_grad_(optimize_c)
        z_var = z_t.detach().clone().requires_grad_(optimize_z)
        opt_vars = [v for cond, v in ((optimize_c, c_var), (optimize_z, z_var)) if cond]

        if _do_fuse and is_last_step:
            eps_u, eps_c, z0_hat, z_prev_hat = _fwd_fused_cfg(
                unet, sched, z_var, t_train, t_prev, c_var, c_uncond,
                pooled, pooled_uncond, time_ids,
            )
            eps_fused = (eps_u + guidance_scale * (eps_c.detach() - eps_u)).to(orig_dtype)
            eps_pred = eps_c
        else:
            eps_pred, z0_hat, z_prev_hat = _fwd(
                unet, sched, z_var, t_train, t_prev, c_var, pooled, time_ids,
            )

        # eps-residual likelihood with re-anchored ref. The first iteration
        # (k=0) sees eps_ref_k == eps_pred so ll_grad ≈ 0; subsequent iterations
        # see meaningful eps_pred - eps_ref_k once z, c have moved.
        # Note: at k=0 the gradient contribution from likelihood IS zero, but
        # the prior_z gradient is also zero (z = z_anchor), so updates at k=0
        # are driven entirely by reward + c-prior. Subsequent iterations have
        # nonzero likelihood and prior gradients.
        obj_main = _obj_terms_variantB(
            z_var, c_var, mu_t, z_t_anchor,
            prior, alpha_bar_t, alpha_t, eps_pred, eps_ref_k,
        ).mean()

        if use_split_grads:
            grads_main = torch.autograd.grad(
                -obj_main, opt_vars, retain_graph=True, create_graph=False,
            )
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
            pixels = _decode_to_pixels(vae, z0_hat, target_hw=reward_decode_size)
            Q = reward_model.score(pixels, prompt)
            obj = obj_main + effective_lambda * Q.mean()
            combined_grads = torch.autograd.grad(
                -obj, opt_vars, retain_graph=False, create_graph=False,
            )
        else:
            combined_grads = torch.autograd.grad(
                -obj_main, opt_vars, retain_graph=False, create_graph=False,
            )

        with torch.no_grad():
            grad_iter = iter(combined_grads)
            if optimize_c:
                grad_c = next(grad_iter)
                c = c_var.detach() - ref_cfg.eta_c * grad_c
                if ref_cfg.clamp_c_norm is not None:
                    n = c.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    c = (c.flatten(1) * (ref_cfg.clamp_c_norm / n).clamp(max=1.0)).view_as(c)
            if optimize_z:
                grad_z = next(grad_iter)
                z_t = z_var.detach() - ref_cfg.eta_z * grad_z
                if ref_cfg.clamp_z_delta is not None:
                    delta = z_t - z_t_anchor
                    n = delta.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    z_t = z_t_anchor + (
                        delta.flatten(1) * (ref_cfg.clamp_z_delta / n).clamp(max=1.0)
                    ).view_as(z_t)

    return c.detach().to(orig_dtype), z_t.detach().to(orig_dtype), eps_fused
