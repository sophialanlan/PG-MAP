"""PG-MAP custom pipeline for SDXL.

Subclass of :class:`diffusers.StableDiffusionXLPipeline` that inserts the
PG-MAP per-step refinement of (c, z_t) before each scheduler step. On SDXL
only the token-level c is optimised; pooled embeddings and time_ids are
kept frozen (paper §3.5).
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Union

import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput


class PGMAPStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    """SDXL pipeline with PG-MAP per-step (c, z_t) refinement."""

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
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
        """Generate image(s) with PG-MAP refinement on SDXL.

        See ``PGMAPStableDiffusionPipeline.__call__`` for argument semantics;
        the SDXL pipeline accepts the same arguments with SDXL defaults
        (50 DDIM steps, $w=5.0$, 1024² resolution).
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

        # --- PG-MAP path ---
        from pgmap_config import PGMAPConfig, sdxl_defaults
        from pgmap_reward import FrozenRewardModel
        from pgmap_sdxl import SDXLModels, generate_sdxl_pgmap

        if pg_map_config is None:
            pg_map_config = sdxl_defaults()
        elif not isinstance(pg_map_config, PGMAPConfig):
            raise TypeError(
                f"pg_map_config must be a PGMAPConfig, got {type(pg_map_config).__name__}"
            )

        cfg = pg_map_config
        if height is not None:
            cfg = replace(cfg, height=int(height))
        if width is not None:
            cfg = replace(cfg, width=int(width))
        if num_inference_steps is not None and num_inference_steps != cfg.num_steps:
            cfg = replace(cfg, num_steps=int(num_inference_steps))
        if guidance_scale is not None and guidance_scale != cfg.guidance_scale:
            cfg = replace(cfg, guidance_scale=float(guidance_scale))
        if generator is not None:
            gen0 = generator[0] if isinstance(generator, list) else generator
            cfg = replace(cfg, seed=int(gen0.initial_seed()))

        if reward_model is None and getattr(cfg, "use_reward", False) and cfg.reward.lambda_reward > 0:
            reward_model = FrozenRewardModel(
                cfg.reward.model_name,
                model_id=cfg.reward.model_id,
                device=str(self._execution_device),
            )

        # SDXL bundle — both tokenizers + both text encoders, dual-stream.
        models = SDXLModels(
            tokenizer_1=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            text_encoder_1=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            unet=self.unet,
            vae=self.vae,
            sched=self.scheduler,
            device=self._execution_device,
            dtype=self.unet.dtype,
        )

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        neg = negative_prompt if isinstance(negative_prompt, str) else (
            "" if negative_prompt is None else (negative_prompt[0] if negative_prompt else "")
        )

        images = []
        for i, p in enumerate(prompts):
            seed_i = cfg.seed + i if len(prompts) > 1 else cfg.seed
            cfg_i = replace(cfg, seed=seed_i)
            img, _logs = generate_sdxl_pgmap(
                p,
                negative_prompt=neg,
                models=models,
                config=cfg_i,
                reward_model=reward_model,
            )
            images.append(img)

        if output_type == "latent":
            raise NotImplementedError(
                "output_type='latent' is not supported in the PG-MAP path."
            )
        if output_type == "np":
            import numpy as np
            images = [np.array(img) for img in images]

        if not return_dict:
            return (images,)
        return StableDiffusionXLPipelineOutput(images=images)
