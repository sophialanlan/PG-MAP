"""PG-MAP ComfyUI custom node — entry point.

Install: clone this repo into ``ComfyUI/custom_nodes/`` (or the ``comfyui/``
subfolder of this repo) and restart ComfyUI. Three new nodes will appear
under the ``PG-MAP`` category:

    PG-MAP Sampler           — generate with per-step (c, z_t) refinement
    PG-MAP Config Builder    — build a PGMAPConfig from sliders
    PG-MAP Reward Loader     — load PickScore / HPS / Aesthetic / CLIP / ImageReward

The nodes load their own diffusers pipeline (independent of ComfyUI's model
loader) because PG-MAP relies on a differentiable forward path through the
denoiser that ComfyUI's wrapped MODEL type does not expose. This costs
roughly one extra UNet copy in VRAM when a vanilla ComfyUI pipeline is
loaded alongside.

Requires: ``pip install pg-map>=1.3.0`` inside the ComfyUI Python environment.

See ``comfyui/README.md`` for the full install walkthrough and the
``comfyui/workflows/pgmap_sdxl_basic.json`` sample workflow.
"""
from __future__ import annotations

from .nodes import (
    PGMAPConfigBuilder,
    PGMAPRewardLoader,
    PGMAPSampler,
)

NODE_CLASS_MAPPINGS = {
    "PGMAPSampler":        PGMAPSampler,
    "PGMAPConfigBuilder":  PGMAPConfigBuilder,
    "PGMAPRewardLoader":   PGMAPRewardLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PGMAPSampler":        "PG-MAP Sampler",
    "PGMAPConfigBuilder":  "PG-MAP Config Builder",
    "PGMAPRewardLoader":   "PG-MAP Reward Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
