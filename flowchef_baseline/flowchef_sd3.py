"""SD3.5-medium FlowChef generator.

Mirrors pgmap_flow_sd3.generate_sd3_ug_flow but swaps the inner refinement
loop from the full-backprop ug_flow_refine_step (PG-MAP UG-FM) to the
FlowChef gradient-skipping flowchef_refine_step.

Everything else (sampler, scheduler, gating window, normalization, decoder)
matches PG-MAP's UG-FM byte-for-byte so the head-to-head comparison
isolates ONLY the gradient-skipping algorithmic axis.

Bitwise-audit guarantee at eta_z=0:
  At eta_z=0 the inner ascent is the identity update and predict_velocity
  is called within torch.no_grad(); the outer SD3.5 sampler is the same
  _sample_with_callback used by the baseline. Therefore the audit script
  audit_eta0.py should produce 0/255 max abs deviation against the existing
  flow_baseline images.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
from PIL import Image

# Pull in PG-MAP's existing SD3 plumbing (loader, decoder, sampler).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pgmap_flow_sd3 import (
    SD3FlowModels, load_sd3_models, _decode_endpoint_grad, _sample_with_callback,
)
from flowchef_core import flowchef_refine_step


def generate_sd3_flowchef(
    prompt: str, neg_prompt: str = "blurry, low quality",
    *,
    models: SD3FlowModels,
    reward_model,
    height: int = 1024, width: int = 1024,
    num_steps: int = 28, cfg_scale: float = 7.0,
    seed: int = 42,
    K: int = 1, eta_z: float = 0.1,
    rho_Q: float = 0.3,
    gate_side: str = "alwayson",     # "alwayson"|"data"|"noise"
) -> Image.Image:
    """FlowChef adaptation: gradient-skipping latent reward step on SD3.5-medium.

    gate_side options:
      - "alwayson" : refine at EVERY denoising step (FlowChef's released default)
      - "data"     : refine in the last rho_Q fraction (data side, t_cont >= 1-rho_Q)
                     — matches PG-MAP UG-FM's headline 91.9% PS configuration so
                       we can isolate gradient-skipping vs full-backprop
      - "noise"    : refine in the first rho_Q fraction (t_cont < rho_Q)
    """
    gate_side = str(gate_side).lower()
    if gate_side not in ("alwayson", "data", "noise"):
        raise ValueError(f"gate_side must be 'alwayson'|'data'|'noise', got {gate_side!r}")

    def refine_fn(z, t, t_cont, dt, c_anchor, predict_velocity, v_default):
        if gate_side == "alwayson":
            active = True
        elif gate_side == "data":
            active = (t_cont >= 1.0 - float(rho_Q))
        else:  # noise
            active = (t_cont < float(rho_Q))
        if not active:
            return z, c_anchor, v_default

        def reward_fn(x1):
            return reward_model.score(x1, prompt)

        def decode_endpoint(z_endpoint):
            return _decode_endpoint_grad(models, z_endpoint)

        z_new = flowchef_refine_step(
            predict_velocity=predict_velocity,
            z_t=z, t_for_model=t, t_cont=t_cont,
            c_anchor=c_anchor,
            decode_endpoint=decode_endpoint, reward_fn=reward_fn,
            K=K, eta_z=eta_z,
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
