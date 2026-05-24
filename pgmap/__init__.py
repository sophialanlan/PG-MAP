"""PG-MAP — Preference-Guided Adaptive MAP.

Public API facade that re-exports the most common symbols under a single
namespace so users can write::

    from pgmap import (
        PGMAPConfig,
        FrozenRewardModel,
        RewardModel,
        sd15_defaults,
        sdxl_defaults,
    )

instead of the per-module imports the research-shape code uses internally::

    from pgmap_config import PGMAPConfig
    from pgmap_reward import FrozenRewardModel
    ...

Both styles work — the flat modules remain installed at the top level so
existing callers (research scripts, the eval CLI, the reproduce_*.sh scripts)
do not break.

Phase B will add ``PGMAPStableDiffusionPipeline`` etc. as proper
``diffusers.DiffusionPipeline`` subclasses. The version here is the v1.1
foundation: configs, reward protocol, and re-exports.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
from pgmap_config import (
    PGMAPConfig,
    PriorConfig,
    RefinementConfig,
    RewardConfig,
    SDPGMAPConfig,
    SemanticConfig,
    baseline_config,
    joint_cz_config,
    mapc_config,
    reward_z_config,
    sd15_defaults,
    sdpgmap_sd15_defaults,
    sdpgmap_sdxl_defaults,
    sdxl_defaults,
)

# ---------------------------------------------------------------------------
# Reward stack (Protocol + reference frozen-network implementation)
# ---------------------------------------------------------------------------
from pgmap_reward import FrozenRewardModel, RewardModel

# ---------------------------------------------------------------------------
# Core inner-loop refinement step (Algorithm 1 in the paper)
# ---------------------------------------------------------------------------
from pgmap_core import pgmap_refine_step

# ---------------------------------------------------------------------------
# Backbone-specific procedural pipelines (current API; subclassed Pipelines
# arrive in v1.2 under Phase B without breaking these).
# ---------------------------------------------------------------------------
from pgmap_sd15 import SD15Models, generate_sd15_pgmap, load_sd15_models
from pgmap_sdxl import SDXLModels, generate_sdxl_pgmap, load_sdxl_models
from pgmap_flow_sd3 import (
    SD3FlowModels,
    generate_sd3_baseline,
    generate_sd3_pgmap_flow,
    generate_sd3_ug_flow,
    load_sd3_models,
)

# ---------------------------------------------------------------------------
# Diffusers pipeline subclasses (Phase B, v1.2+).
# Heavyweight diffusers imports are deferred to first use so `import pgmap`
# stays fast for users who only want config dataclasses or the reward model.
# ---------------------------------------------------------------------------
def __getattr__(name):
    if name in ("PGMAPStableDiffusionPipeline",
                "PGMAPStableDiffusionXLPipeline",
                "PGMAPStableDiffusion3Pipeline"):
        from pgmap.pipelines import (
            PGMAPStableDiffusionPipeline,
            PGMAPStableDiffusionXLPipeline,
            PGMAPStableDiffusion3Pipeline,
        )
        cls = {
            "PGMAPStableDiffusionPipeline":   PGMAPStableDiffusionPipeline,
            "PGMAPStableDiffusionXLPipeline": PGMAPStableDiffusionXLPipeline,
            "PGMAPStableDiffusion3Pipeline":  PGMAPStableDiffusion3Pipeline,
        }[name]
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "1.5.0"

__all__ = [
    # config
    "PGMAPConfig",
    "PriorConfig",
    "RefinementConfig",
    "RewardConfig",
    "SDPGMAPConfig",
    "SemanticConfig",
    "baseline_config",
    "joint_cz_config",
    "mapc_config",
    "reward_z_config",
    "sd15_defaults",
    "sdpgmap_sd15_defaults",
    "sdpgmap_sdxl_defaults",
    "sdxl_defaults",
    # reward
    "FrozenRewardModel",
    "RewardModel",
    # core
    "pgmap_refine_step",
    # backbones
    "SD15Models",
    "SDXLModels",
    "SD3FlowModels",
    "generate_sd15_pgmap",
    "generate_sdxl_pgmap",
    "generate_sd3_baseline",
    "generate_sd3_pgmap_flow",
    "generate_sd3_ug_flow",
    "load_sd15_models",
    "load_sdxl_models",
    "load_sd3_models",
    # diffusers pipeline subclasses (lazy-loaded via __getattr__)
    "PGMAPStableDiffusionPipeline",
    "PGMAPStableDiffusionXLPipeline",
    "PGMAPStableDiffusion3Pipeline",
    # meta
    "__version__",
]
