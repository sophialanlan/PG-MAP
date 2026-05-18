"""PatchSchedule — per-step conditioning anchor mu_t provider.

Extracted from the deprecated eval_sd15_map40_clip.py / eval_sdxl_map40_clip_v1.py
into a standalone module so pgmap_sd15.py and pgmap_sdxl.py can import it
after the originals were removed from the repo.

Default usage in PG-MAP (mode='none') simply returns the unperturbed c0 at
every step. The 'external' / 'internal' modes are unused in the main paper
results but kept for backwards compatibility with existing config plumbing.
"""
from __future__ import annotations

from typing import Optional

import torch


class PatchSchedule:
    """tau_k(c0) provider for each sampling step.

    Modes:
      - 'none': returns c0 (no anchor perturbation).
      - 'external': uses caller-supplied patches of shape (K, 77, 768)
                    or (K, B, 77, 768).
      - 'internal': random unit-norm directions seeded by ``seed``.
    """

    def __init__(
        self,
        mode: str = "none",
        *,
        patches: Optional[torch.Tensor] = None,
        add_to_c0: bool = True,
        K: int = 4,
        patch_scale: float = 0.05,
        seed: int = 0,
    ):
        if mode not in ("none", "external", "internal"):
            raise ValueError(f"Unknown patch mode: {mode}")
        self.mode = mode
        self.add_to_c0 = add_to_c0
        self.K = K
        self.patch_scale = patch_scale
        self.seed = seed

        if mode == "external":
            if patches is None:
                raise ValueError("mode='external' requires patches tensor")
            if patches.dim() not in (3, 4):
                raise ValueError("patches must be (K,77,768) or (K,B,77,768)")
            self.patches = patches
        else:
            self.patches = None

    def _internal_patches(self, c0: torch.Tensor) -> torch.Tensor:
        g = torch.Generator(device=c0.device)
        g.manual_seed(self.seed)
        P = torch.randn((self.K, 1) + c0.shape[1:],
                        generator=g, device=c0.device, dtype=c0.dtype)
        norm = P.flatten(2).norm(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-12)
        P = (P / norm) * self.patch_scale
        return P

    def tau(self, c0: torch.Tensor, step_i: int, num_steps: int) -> torch.Tensor:
        """Returns mu_t (B, 77, 768) for the given sampling step index."""
        if self.mode == "none":
            return c0

        if self.mode == "internal":
            patches = self._internal_patches(c0)
        else:
            patches = self.patches.to(device=c0.device, dtype=c0.dtype)
            if patches.dim() == 3:
                patches = patches.unsqueeze(1)

        K = patches.shape[0]
        k = max(0, min(K - 1, int(step_i * K / max(num_steps, 1))))
        pk = patches[k]
        if pk.shape[0] == 1:
            pk = pk.expand(c0.shape[0], -1, -1)
        return (c0 + pk) if self.add_to_c0 else pk
