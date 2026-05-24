"""PG-MAP custom diffusers pipelines (Phase B).

Drop-in replacements for the standard diffusers pipelines that run
PG-MAP inference-time alignment per denoising step.

    from pgmap.pipelines import (
        PGMAPStableDiffusionPipeline,
        PGMAPStableDiffusionXLPipeline,
        PGMAPStableDiffusion3Pipeline,
    )

All three accept the same constructor arguments as their parent classes
(``StableDiffusionPipeline`` / ``StableDiffusionXLPipeline`` /
``StableDiffusion3Pipeline``) and add two PG-MAP-specific keyword
arguments to ``__call__``:

    pg_map_config: PGMAPConfig | None
        The full hyperparameter surface for the per-step refinement.
        If ``None``, the pipeline falls through to the vanilla
        ``super().__call__()`` so the subclass is *also* a perfectly
        valid drop-in for the standard pipeline.

    reward_model: RewardModel | None
        Any object implementing the ``RewardModel`` protocol
        (``score(pixel_values, prompt) -> Tensor[B]``).
        If ``None`` and ``pg_map_config`` enables the reward term
        (``use_reward=True``, ``lambda_reward>0``), a ``FrozenRewardModel``
        is auto-instantiated with the model name from
        ``pg_map_config.reward.model_name``.

For the HuggingFace community-pipeline registry, single-file mirrors of
these classes will be published under ``sophialan/pg-map-{sd15,sdxl,sd3}``
in Phase C so users can do::

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        custom_pipeline="sophialan/pg-map-sdxl",
        torch_dtype=torch.float16,
    )
"""
from __future__ import annotations

from pgmap.pipelines.sd15 import PGMAPStableDiffusionPipeline
from pgmap.pipelines.sdxl import PGMAPStableDiffusionXLPipeline
from pgmap.pipelines.sd3 import PGMAPStableDiffusion3Pipeline

__all__ = [
    "PGMAPStableDiffusionPipeline",
    "PGMAPStableDiffusionXLPipeline",
    "PGMAPStableDiffusion3Pipeline",
]
