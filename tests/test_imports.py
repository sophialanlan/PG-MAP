"""Lightweight import-smoke test — runs without a GPU.

Verifies every top-level module imports cleanly with no syntax errors and
without triggering an actual model download. Useful for CI on CPU runners.

    python -m pytest tests/test_imports.py -v
"""
from __future__ import annotations

import importlib
import sys


MODULES = [
    "pgmap_config",
    "pgmap_core",
    "pgmap_reward",
    "pgmap_sd15",
    "pgmap_sdxl",
    "pgmap_flow_core",
    "pgmap_flow_sd3",
    "pgmap_variants",
    "pgmap_sd15_variants",
    "pgmap_sdxl_variants",
    "pgmap_flow_variants",
    "pgmap_patch_schedule",
    "pgmap_eval",
    "validate_criteria",
]


def test_modules_importable():
    for name in MODULES:
        try:
            importlib.import_module(name)
        except ImportError as e:
            raise AssertionError(f"failed to import {name}: {e}")


def test_preset_configs():
    from pgmap_config import (
        sd15_defaults, sdxl_defaults,
        baseline_config, mapc_config, reward_z_config, joint_cz_config,
    )
    cfg = sd15_defaults()
    assert cfg.refinement.K >= 1
    assert cfg.prior.gamma > 0
    assert cfg.reward.lambda_reward >= 0

    cfg = sdxl_defaults()
    assert cfg.refinement.K >= 1


if __name__ == "__main__":
    test_modules_importable()
    test_preset_configs()
    print("imports OK")
