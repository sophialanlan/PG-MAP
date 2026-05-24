"""PG-MAP custom pipeline for Stable Diffusion 3.5-medium.

Subclass of :class:`diffusers.StableDiffusion3Pipeline` that inserts the
flow-matching reduction of PG-MAP per Euler step.

Dispatch:
  - ``pg_map_config.optimize_c and pg_map_config.optimize_z`` -> full PG-MAP-FM
    (``generate_sd3_pgmap_flow``).
  - ``pg_map_config.optimize_z and not optimize_c`` -> **UG-FM** (default,
    the paper's 91.9% PickScore / 75.7% HPS row, ``generate_sd3_ug_flow``).
  - ``pg_map_config is None and reward_model is None`` -> vanilla
    rectified-flow Euler via ``super().__call__()``.

UG-FM is the default because, per paper §3.2, on flow matching the optimal
active set collapses to ``{z_t}`` only (data-side gate). Setting
``optimize_c=True`` enables the more expensive joint variant.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Union

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3PipelineOutput


class PGMAPStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    """SD3.5-medium pipeline with PG-MAP (UG-FM by default) flow-matching refinement."""

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = "blurry, low quality",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # --- PG-MAP-specific kwargs ---
        pg_map_config=None,
        reward_model=None,
        # --- Pass-through diffusers kwargs ---
        output_type: str = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        """Generate image(s) on SD3.5-medium with PG-MAP / UG-FM refinement.

        UG-FM (the default when ``pg_map_config.optimize_c=False``) is the
        91.9% PickScore configuration: data-side gate, K_UG=4, eta_z=0.1,
        full backprop through the velocity prediction.
        """
        # --- Pass-through to vanilla SD3.5 pipeline ---
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

        from pgmap_flow_sd3 import (
            SD3FlowModels,
            generate_sd3_baseline,
            generate_sd3_pgmap_flow,
            generate_sd3_ug_flow,
        )
        from pgmap_reward import FrozenRewardModel

        # Build the SD3FlowModels bundle around `self`. The pipe field
        # reuses self so any encode/decode helper paths stay consistent.
        models = SD3FlowModels(
            pipe=self,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )

        # Dispatch decision: full joint (c, z_t) vs UG-FM (latent only).
        # Defaults match the paper's UG-FM headline (data-side gate, K=4, eta=0.1).
        optimize_c = bool(getattr(pg_map_config, "optimize_c", False)) if pg_map_config else False
        optimize_z = bool(getattr(pg_map_config, "optimize_z", True))  if pg_map_config else True
        use_reward = bool(getattr(pg_map_config, "use_reward",  True)) if pg_map_config else True

        # Resolve the seed
        if generator is not None:
            gen0 = generator[0] if isinstance(generator, list) else generator
            seed = int(gen0.initial_seed())
        elif pg_map_config is not None and getattr(pg_map_config, "seed", None) is not None:
            seed = int(pg_map_config.seed)
        else:
            seed = 42

        # Auto-instantiate reward model if needed
        if reward_model is None and use_reward:
            reward_name = "pickscore"
            if pg_map_config is not None and hasattr(pg_map_config, "reward"):
                reward_name = pg_map_config.reward.model_name
            reward_model = FrozenRewardModel(
                reward_name, device=str(self._execution_device),
            )

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        neg = negative_prompt if isinstance(negative_prompt, str) else (
            "blurry, low quality" if negative_prompt is None else (negative_prompt[0] if negative_prompt else "")
        )

        images = []
        for i, p in enumerate(prompts):
            seed_i = seed + i if len(prompts) > 1 else seed

            if not optimize_z and not optimize_c:
                # Vanilla baseline (rare path; usually super().__call__ above)
                img = generate_sd3_baseline(
                    p, neg_prompt=neg, models=models,
                    height=height, width=width,
                    num_steps=num_inference_steps, cfg_scale=guidance_scale,
                    seed=seed_i,
                )
            elif optimize_c and optimize_z:
                # Full PG-MAP-FM (joint c + z_t with consistency + reward + priors).
                cfg = pg_map_config
                img = generate_sd3_pgmap_flow(
                    p, neg_prompt=neg, models=models, reward_model=reward_model,
                    height=height, width=width,
                    num_steps=num_inference_steps, cfg_scale=guidance_scale,
                    seed=seed_i,
                    K=cfg.refinement.K,
                    eta_c=cfg.refinement.eta_c,
                    eta_z=cfg.refinement.eta_z,
                    sigma_c=cfg.prior.sigma_c,
                    gamma=cfg.prior.gamma,
                    lambda_reward=cfg.reward.lambda_reward,
                    rho=cfg.rho,
                    rho_Q=cfg.reward.rho_Q,
                    optimize_c=True,
                    optimize_z=True,
                    use_reward=use_reward,
                    grad_norm_strategy=cfg.reward.grad_norm_strategy,
                )
            else:
                # UG-FM headline (z_t-only, data-side gate, K=4, eta_z=0.1).
                cfg = pg_map_config
                K_ug   = cfg.refinement.K          if cfg is not None else 4
                eta_z  = cfg.refinement.eta_z      if cfg is not None else 0.1
                rho_Q  = cfg.reward.rho_Q          if cfg is not None else 0.3
                # FM headline uses K_ug=4 / eta_z=0.1; respect cfg if it
                # explicitly overrides, otherwise use the paper defaults.
                if cfg is not None:
                    if K_ug <= 0:
                        K_ug = 4
                    if eta_z <= 0:
                        eta_z = 0.1
                img = generate_sd3_ug_flow(
                    p, neg_prompt=neg, models=models, reward_model=reward_model,
                    height=height, width=width,
                    num_steps=num_inference_steps, cfg_scale=guidance_scale,
                    seed=seed_i,
                    K_ug=K_ug, eta_z=eta_z, rho_Q=rho_Q,
                    gate_side="data",
                )
            images.append(img)

        if output_type == "latent":
            raise NotImplementedError(
                "output_type='latent' is not supported in the PG-MAP-FM path."
            )
        if output_type == "np":
            import numpy as np
            images = [np.array(img) for img in images]

        if not return_dict:
            return (images,)
        return StableDiffusion3PipelineOutput(images=images)
