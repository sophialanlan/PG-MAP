# Changelog

All notable changes to the public PG-MAP release.

## [v1.2.0] — 2026-05-18

**Phase B: diffusers ``DiffusionPipeline`` subclasses.** PG-MAP now ships as a drop-in replacement for the standard diffusers pipelines for all three backbones, with a single ``pg_map_config`` kwarg controlling the per-step refinement.

### Added

- **`pgmap.pipelines.PGMAPStableDiffusionPipeline`** — subclass of `StableDiffusionPipeline`. Drop-in replacement; passing `pg_map_config=None` falls through to vanilla SD 1.5.
- **`pgmap.pipelines.PGMAPStableDiffusionXLPipeline`** — subclass of `StableDiffusionXLPipeline`. SDXL with token-level $c$ refinement (pooled embeddings + time_ids kept frozen per paper §3.5).
- **`pgmap.pipelines.PGMAPStableDiffusion3Pipeline`** — subclass of `StableDiffusion3Pipeline`. Dispatches to UG-FM (default; the 91.9% PickScore / 75.7% HPS row) or full PG-MAP-FM (when `pg_map_config.optimize_c=True`).
- **Lazy import** of the heavy pipeline classes via `pgmap.__getattr__` — `import pgmap` stays fast for users who only need configs/reward.
- **GPU smoke test** at [tests/smoke/smoke_pipeline.py](tests/smoke/smoke_pipeline.py) — verifies SD 1.5 vanilla + MAP-c paths, SDXL vanilla path, and SD3.5 class hierarchy in under 90 seconds on a single GPU.

### Usage

```python
from pgmap.pipelines import PGMAPStableDiffusionPipeline
from pgmap import sd15_defaults, FrozenRewardModel
import torch

pipe = PGMAPStableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

cfg = sd15_defaults()
reward = FrozenRewardModel("pickscore", device="cuda")

image = pipe(
    "a phoenix rising from ashes, vivid orange and red feathers",
    pg_map_config=cfg,
    reward_model=reward,
).images[0]
```

Or, omit the PG-MAP args and the pipeline behaves identically to vanilla `StableDiffusionPipeline`:

```python
image = pipe("a phoenix rising from ashes").images[0]   # vanilla SD 1.5
```

### Maintained / unchanged

- The v1.0 functional API (`generate_sd15_pgmap`, `generate_sdxl_pgmap`, `generate_sd3_*`) and the eval CLI are unchanged. The pipeline subclasses wrap these helpers internally — both APIs share the same inner loop.
- All paper-table reproduction scripts (`scripts/reproduce_table*.sh`) continue to work unchanged.

### Deferred to v1.3 (Phase C)

- Push `sophialan/pg-map-{sd15,sdxl,sd3}` custom-pipeline repos to HuggingFace Hub.
- Gradio Space + Colab one-click notebook.

## [v1.1.0] — 2026-05-18

**Foundation for library-shaped distribution.** No behavior change on default settings; all v1.0 reproduction scripts still work unchanged. Lands the prerequisites for Phase B (diffusers pipeline subclasses) and Phase C (HuggingFace Hub push).

### Added

- **`pyproject.toml`** — `pip install -e .` now works; `pg-map` is installable from a local checkout and ready for PyPI publication. Console script `pgmap-eval` exposed as a proper entry point.
- **`pgmap/` package facade** — single-namespace re-exports of the flat research modules. Users can now write `from pgmap import PGMAPConfig, FrozenRewardModel, sd15_defaults` instead of the per-module imports. The flat `pgmap_*.py` modules remain importable, so v1.0 callers do not break.
- **`RewardModel` Protocol** in [pgmap_reward.py](pgmap_reward.py) — structural typing protocol with the `.score(pixel_values, prompt) -> Tensor[B]` signature. External rewards no longer need to subclass `FrozenRewardModel`; implementing the protocol is enough. Verified with `isinstance(my_reward, RewardModel)` at runtime.
- **`--mixed_precision {no,fp16,bf16}`** flag in `pgmap_eval.py` — accelerate-style alias for `--dtype`. Plugs cleanly into HuggingFace `accelerate launch` configs.
- **`--gradient_checkpointing`** flag — enables UNet/MMDiT gradient checkpointing for the reward backward. Required for SDXL @ 1024² with $K_\text{inner} > 2$ on 24 GB cards.
- **`bf16`** as a supported precision (in addition to `fp16` and `fp32`). More stable on Ampere+/H100; recommended when stress-testing PG-MAP at higher $K$.

### Maintained / unchanged

- All v1.0 reproduction scripts ([scripts/reproduce_table*.sh](scripts/)) and SLURM templates run without modification.
- Default precision remains `fp16` (the paper's reference); existing numbers do not move.
- The 1.9 GB HPS-v2 checkpoint remains unbundled (download via [scripts/download_hps.sh](scripts/download_hps.sh)).

### Deferred to v1.2 (Phase B)

- Diffusers `DiffusionPipeline` subclasses (`PGMAPStableDiffusionPipeline` / `PGMAPStableDiffusionXLPipeline` / `PGMAPStableDiffusion3Pipeline`) with `__call__` overrides accepting `pg_map_config=PGMAPConfig(...)`. Will register under the HuggingFace community-pipelines registry as `<sophialan>/pg-map`.
- Speedups touching the hot path (Tweedie cache for $K_\text{inner} > 1$, `effective_lambda > eps` gate when `lambda_ramp` is active). Deferred so they can land with bench validation rather than in a foundation patch.

## [v1.0-neurips2026] — 2026-05-18

Initial public release accompanying the NeurIPS 2026 paper *"PG-MAP: Joint MAP Optimization for Inference-Time Alignment of Diffusion and Flow-Matching Models"*.

### Included

- Reference implementation of PG-MAP and all variants:
  - **Diffusion side** (SD 1.5, SDXL): MAP-$c$, Reward-$z$, MAP-$cz$ ($\lambda{=}0$), PG-MAP (default), Tuned-CFG$+$PG-MAP.
  - **Flow-matching side** (SD3.5-medium): UG-FM (data-side, $K{=}4$, $\eta_z{=}0.1$).
- Per-row YAML configs covering every paper table cell.
- One-command reproduction scripts: [scripts/reproduce_table1_sd15.sh](scripts/reproduce_table1_sd15.sh), [scripts/reproduce_table1_sdxl.sh](scripts/reproduce_table1_sdxl.sh), [scripts/reproduce_table2_fm.sh](scripts/reproduce_table2_fm.sh), [scripts/reproduce_table4_crr.sh](scripts/reproduce_table4_crr.sh).
- Smoke tests (≤ 10 min per backbone) at [tests/smoke/](tests/smoke/).
- FlowChef head-to-head adapter at [flowchef_baseline/](flowchef_baseline/).
- Post-hoc analyses: win-rate, Wilcoxon $p$, bootstrap CI, BLIP-VQA audit, CRR-MAP oracle routing.

### Reproducibility notes

- All tables use **seed `123`** and the same PartiPrompts ($n{=}1632$) ordering.
- Generation is deterministic on fixed hardware (RTX PRO 6000 Blackwell, fp16).
- Cross-GPU (A100 / H100 / RTX 6000) drift is within bootstrap CI half-width.
- The HPS-v2 scoring checkpoint (1.9 GB) is **not bundled**; download via [scripts/download_hps.sh](scripts/download_hps.sh).
