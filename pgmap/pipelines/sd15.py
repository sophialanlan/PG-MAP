"""PG-MAP custom pipeline for Stable Diffusion 1.5.

Subclass of :class:`diffusers.StableDiffusionPipeline` that inserts the
PG-MAP per-step refinement of (c, z_t) before each scheduler step. When
called without ``pg_map_config`` and ``reward_model`` it falls through to
the vanilla pipeline, so the class is a strict superset of the parent.

Usage::

    from diffusers import DiffusionPipeline
    from pgmap.pipelines import PGMAPStableDiffusionPipeline
    from pgmap import sd15_defaults, FrozenRewardModel

    # Option A: instantiate the subclass directly
    pipe = PGMAPStableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to("cuda")

    # Option B: HF community-pipeline registry (Phase C — pending publish)
    # pipe = DiffusionPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5",
    #     custom_pipeline="sophialan/pg-map-sd15",
    #     torch_dtype=torch.float16,
    # )

    cfg = sd15_defaults()             # paper defaults
    cfg.seed = 42

    reward = FrozenRewardModel("pickscore", device="cuda")
    image = pipe(
        "a phoenix rising from ashes, vivid orange and red feathers",
        pg_map_config=cfg,
        reward_model=reward,
        num_inference_steps=cfg.num_steps,
        guidance_scale=cfg.guidance_scale,
    ).images[0]
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Union

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


class PGMAPStableDiffusionPipeline(StableDiffusionPipeline):
    """SD 1.5 pipeline with PG-MAP per-step (c, z_t) refinement."""

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # --- PG-MAP-specific kwargs ---
        pg_map_config=None,
        reward_model=None,
        # --- Pass-through diffusers kwargs ---
        output_type: str = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        """Generate image(s) with PG-MAP refinement.

        Args:
            prompt: Text prompt or list of prompts (one image per prompt).
            height, width: Image resolution. Default 512 for SD 1.5.
            num_inference_steps: DDIM steps. Paper default 30.
            guidance_scale: CFG scale. Paper default 7.5.
            negative_prompt: Negative prompt for CFG.
            generator: torch.Generator(s) for reproducibility.
            pg_map_config: ``PGMAPConfig``. If ``None``, falls through to the
                vanilla ``StableDiffusionPipeline.__call__()`` (no PG-MAP).
            reward_model: ``RewardModel``-protocol object. If ``None`` and
                the config enables the reward term, a ``FrozenRewardModel``
                is auto-instantiated.
            output_type, return_dict: Standard diffusers output conventions.

        Returns:
            ``StableDiffusionPipelineOutput`` when ``return_dict=True``,
            else a tuple ``(images, has_nsfw_concept=None)``.
        """
        # --- Pass-through to the vanilla pipeline when PG-MAP is not requested ---
        if pg_map_config is None and reward_model is None:
            return super().__call__(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator,
                output_type=output_type,
                return_dict=return_dict,
                **kwargs,
            )

        # --- PG-MAP path: delegate to the reference inner-loop implementation ---
        # Local imports keep the parent pipeline importable without pgmap_*
        # files on the path (e.g. when only the custom_pipeline file is shipped).
        from pgmap_config import PGMAPConfig, sd15_defaults
        from pgmap_reward import FrozenRewardModel
        from pgmap_sd15 import SD15Models, generate_sd15_pgmap

        if pg_map_config is None:
            pg_map_config = sd15_defaults()
        elif not isinstance(pg_map_config, PGMAPConfig):
            raise TypeError(
                f"pg_map_config must be a PGMAPConfig, got {type(pg_map_config).__name__}"
            )

        # Sync diffusers-level args into the PG-MAP config (only when caller
        # set them explicitly — None means "use the config's value").
        cfg = pg_map_config
        if height is not None:
            cfg = replace(cfg, height=int(height))
        if width is not None:
            cfg = replace(cfg, width=int(width))
        if num_inference_steps is not None and num_inference_steps != cfg.num_steps:
            cfg = replace(cfg, num_steps=int(num_inference_steps))
        if guidance_scale is not None and guidance_scale != cfg.guidance_scale:
            cfg = replace(cfg, guidance_scale=float(guidance_scale))
        # Forward the generator's seed if the user passed one.
        if generator is not None:
            gen0 = generator[0] if isinstance(generator, list) else generator
            cfg = replace(cfg, seed=int(gen0.initial_seed()))

        # Auto-instantiate the reward model if the config asks for it.
        if reward_model is None and getattr(cfg, "use_reward", False) and cfg.reward.lambda_reward > 0:
            reward_model = FrozenRewardModel(
                cfg.reward.model_name,
                model_id=cfg.reward.model_id,
                device=str(self._execution_device),
            )

        # Bundle pipe components into the SD15Models dataclass that the
        # reference inner-loop expects. The components are shared by
        # reference, so no extra GPU memory is allocated.
        models = SD15Models(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            unet=self.unet,
            vae=self.vae,
            sched=self.scheduler,
            device=self._execution_device,
            dtype=self.unet.dtype,
        )

        # Run PG-MAP per prompt (the reference impl handles batch=1 only;
        # we loop over batched prompts to preserve seed-determinism).
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        neg = negative_prompt if isinstance(negative_prompt, str) else (
            "" if negative_prompt is None else (negative_prompt[0] if negative_prompt else "")
        )

        images = []
        for i, p in enumerate(prompts):
            seed_i = cfg.seed + i if isinstance(prompts, list) and len(prompts) > 1 else cfg.seed
            cfg_i = replace(cfg, seed=seed_i)
            img, _logs = generate_sd15_pgmap(
                p,
                negative_prompt=neg,
                models=models,
                config=cfg_i,
                reward_model=reward_model,
            )
            images.append(img)

        # Honour the standard diffusers output_type contract.
        if output_type == "latent":
            raise NotImplementedError(
                "output_type='latent' is not supported in the PG-MAP path; "
                "PG-MAP returns decoded PIL images. Use the vanilla pipeline "
                "(no pg_map_config) for latent outputs."
            )
        if output_type == "np":
            import numpy as np
            images = [np.array(img) for img in images]

        if not return_dict:
            return (images, None)
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)
