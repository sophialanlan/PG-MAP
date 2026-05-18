"""FlowChef gradient-skipping core.

Adaptation of FlowChef (arXiv:2412.00100, ICCV 2025) for PickScore-driven
T2I on SD3.5-medium. The DISTINGUISHING algorithmic axis between FlowChef
and PG-MAP's UG-FM is **gradient skipping**:

    UG-FM:   g = ∂_{z_t} Q(D( z_t - (1-t) v_θ(z_t,t,c) ))
             — full backprop through v_θ AND the VAE D
             — captures the full Jacobian (I - (1-t) ∂_z v_θ) ∘ ∂x decoder

    FlowChef: g = ∂_{z_t} Q(D( z_t - (1-t) v_θ.detach() ))
             — backprop only through the linear endpoint formula and the VAE
             — Jacobian of v_θ w.r.t. z_t is treated as ZERO (stop-grad)
             — this is FlowChef's released schema (per their codebase, the
               velocity field forward is wrapped in torch.no_grad())

Original FlowChef use-cases (Flux/InstaFlow, inverse problems / PIE-Bench
editing) used a reference-image L2 loss; we swap that for the frozen
PickScore reward Q(x,y) on the FM endpoint estimate, identical to PG-MAP's
UG-FM objective so the comparison isolates the gradient-skipping axis only.

Sign convention: PickScore is a reward to MAXIMIZE, so we ascend
z_t <- z_t + eta * g (matches PG-MAP UG-FM); the user-facing description
in the spec ("z_t <- z_t - eta") was loss-minimization framing.

NFE accounting (per outer denoising step):
    UG-FM K=4:    1 main v_θ forward + 4 × (1 v_θ forward + reward backward
                  through v_θ + VAE) ≈ 5 effective forwards + heavy backward
    FlowChef K=1: 1 main v_θ forward + 1 × (1 v_θ no_grad forward + reward
                  backward through VAE only) ≈ 2 forwards + light backward

We therefore report FlowChef at MULTIPLE K values to give an NFE-matched
column in the comparison: K=1 (the FlowChef released config), and K=4
(NFE-matched to UG-FM). Both run.
"""
from __future__ import annotations
from typing import Callable
import torch


def flow_endpoint_estimate_local(z_t: torch.Tensor, v: torch.Tensor, t_cont: float) -> torch.Tensor:
    """FM endpoint estimate (diffusers sign): x_hat_1 = z_t - (1-t) v_pred.

    Local copy so we don't pull the autograd graph from pgmap_flow_core's
    full-backprop version; the contract is the same.
    """
    return z_t - (1.0 - float(t_cont)) * v


def flowchef_refine_step(
    *,
    predict_velocity: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    z_t: torch.Tensor,
    t_for_model: torch.Tensor,
    t_cont: float,
    c_anchor: torch.Tensor,
    decode_endpoint: Callable[[torch.Tensor], torch.Tensor],
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    K: int = 1,
    eta_z: float = 0.1,
) -> torch.Tensor:
    """One outer denoising step's per-step latent refinement, FlowChef-style.

    K=1 mirrors the FlowChef released per-step ascent. K>1 is provided for
    NFE-matched comparison against PG-MAP's UG-FM (which uses K=4).

    Gradient flow:
      - v = predict_velocity(z, ...)         within torch.no_grad()
      - x_hat = z + (-(1-t)) * v             gradient flows only through z
                                             (v is detached)
      - r = reward_fn(decode_endpoint(x_hat))
      - g_z = autograd.grad(r, z)            no Jacobian-of-v contribution
      - z <- z + eta * g_z / ||g_z||         unit-norm ascent (matches UG-FM
                                             gradient normalization for fair
                                             comparison; FlowChef's released
                                             code also normalizes per-step)
    """
    t_cont_f = float(t_cont)
    z_var = z_t.detach()
    for _k in range(K):
        z_iter = z_var.detach().requires_grad_(True)

        # *** GRADIENT SKIPPING ***: v_theta forward is a stop-gradient.
        with torch.no_grad():
            v_no_grad = predict_velocity(z_iter, t_for_model, c_anchor)
        # By detaching after no_grad we make doubly sure no graph leaks in.
        v = v_no_grad.detach()

        # Linear endpoint formula: gradient w.r.t. z_iter is identity.
        x1_hat_latent = flow_endpoint_estimate_local(z_iter, v, t_cont_f)
        x1_hat = decode_endpoint(x1_hat_latent)
        r_val = reward_fn(x1_hat)

        (g_z,) = torch.autograd.grad(
            r_val.mean(), z_iter,
            retain_graph=False, create_graph=False,
        )
        n = g_z.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-12)
        g_z = (g_z.flatten(1) / n).view_as(g_z)
        z_var = (z_iter.detach() + eta_z * g_z).detach()

    return z_var
