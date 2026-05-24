"""
PG-MAP Configuration Dataclasses
=================================

All configuration for the Preference-Guided Adaptive MAP (PG-MAP) method.

Key equation (paper Eq. 9):
    J_t(c, z_t) = -1/(2*beta_t) ||z_t - sqrt(alpha_t) * f_theta(z_t,t,c)||^2   [forward-consistency]
                  - 1/(2*sigma_c^2) ||c - mu_t||^2                               [conditioning prior]
                  - 1/(2*sigma_z^2) ||z_t - z_t^ddim||^2                          [latent prior]
                  + lambda * Q(x_hat_0(z_t, c), y)                                [preference reward]

Ablation flags allow recovering special cases:
    - optimize_z=False, use_reward=False  =>  MAP-c (conditioning-only MAP)
    - optimize_c=False, use_reward=True   =>  Reward-z (latent-only + reward)
    - optimize_c=True,  optimize_z=True, use_reward=False  =>  Joint-(c,z) without reward
    - optimize_c=False, optimize_z=False, use_reward=False  =>  Standard DDIM+CFG baseline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PriorConfig:
    """Prior variance configuration for conditioning and latent anchors.

    Attributes:
        sigma_c: Standard deviation for the conditioning prior.
                 Smaller => stronger pull toward the anchor mu_t.
        gamma:   Latent prior scale factor. The latent prior std is computed
                 adaptively as sigma_z(t) = gamma * sqrt(1 - alpha_bar_t).
                 This makes the prior weaker at high noise (where z_t is far
                 from the clean signal) and stronger at low noise.
    """
    sigma_c: float = 1.0
    gamma: float = 0.5


@dataclass
class RewardConfig:
    """Configuration for the frozen preference reward model.

    Attributes:
        model_name:         Which reward model to use. One of:
                            "pickscore", "hps", "aesthetic", "clip".
        model_id:           HuggingFace model identifier for loading.
        lambda_reward:      Reward weight (lambda in the paper). Controls
                            the strength of the preference signal relative
                            to the consistency and prior terms.
        rho_Q:              Fraction of denoising steps (from the start)
                            where the reward gradient is active. E.g. 0.3
                            means the reward is used only in the first 30%
                            of steps.
        grad_norm_strategy: How to normalize the reward gradient before
                            combining with the consistency gradient.
                            - "unit":     Unit-normalize the reward gradient,
                                          then scale by lambda. This makes
                                          lambda directly control the step size.
                            - "adaptive": Match reward gradient norm to the
                                          consistency gradient norm (like the
                                          existing clip_adaptive code).
                            - "raw":      No normalization; use raw gradients.
        lambda_ramp:        If True, linearly ramp lambda from 0 → lambda_reward
                            over the reward-active steps. Rationale: early steps
                            have noisy z0_hat and unreliable reward gradients;
                            later steps (cleaner image) deserve stronger reward push.
    """
    model_name: str = "pickscore"
    model_id: str = "yuvalkirstain/PickScore_v1"
    lambda_reward: float = 0.1
    rho_Q: float = 0.3
    grad_norm_strategy: str = "unit"
    lambda_ramp: bool = False


@dataclass
class RefinementConfig:
    """Inner-loop gradient ascent configuration.

    Attributes:
        K:              Number of inner gradient steps per denoising step.
        eta_c:          Learning rate for the conditioning embedding c.
        eta_z:          Learning rate for the latent z_t.
        clamp_c_norm:   If set, clamp the L2 norm of c to this value
                        after each gradient step (safety guardrail).
        clamp_z_delta:  If set, clamp the L2 norm of (z_t - z_t_ddim)
                        to this value after each gradient step.
        optimizer:      Inner-loop optimizer. "sgd" = vanilla gradient descent
                        (original). "adam" = Adam with momentum that persists
                        across the K inner steps, enabling better progress
                        especially for K > 1.
        adam_beta1:     Adam first-moment decay (default 0.9).
        adam_beta2:     Adam second-moment decay (default 0.999).
        adam_eps:       Adam numerical stability constant (default 1e-8).
    """
    K: int = 1
    eta_c: float = 1e-4
    eta_z: float = 0.001
    clamp_c_norm: Optional[float] = None
    clamp_z_delta: Optional[float] = None
    optimizer: str = "sgd"      # "sgd" | "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    gauss_seidel: bool = False  # if True, update c first, then re-forward with new c to update z


@dataclass
class PGMAPConfig:
    """Top-level configuration for PG-MAP generation.

    Groups all sub-configs and provides ablation flags to recover
    special cases of the method.

    Ablation examples:
        Full PG-MAP:   optimize_c=True,  optimize_z=True,  use_reward=True
        MAP-c only:    optimize_c=True,  optimize_z=False, use_reward=False
        Reward-z only: optimize_c=False, optimize_z=True,  use_reward=True
        Joint (c,z):   optimize_c=True,  optimize_z=True,  use_reward=False
        Baseline:      optimize_c=False, optimize_z=False, use_reward=False
    """

    # --- Sampling parameters ---
    num_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    seed: int = 0

    # --- Refinement control ---
    rho: float = 0.4  # fraction of steps to refine (from beginning)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # --- Patch schedule (inherited from existing MAP code) ---
    patch_mode: str = "none"
    patch_add_to_c0: bool = True
    patch_K: int = 4
    patch_scale: float = 0.05
    patch_seed: int = 0
    external_patches: Optional[object] = None  # torch.Tensor, kept as object for serialization

    # --- Ablation flags ---
    optimize_c: bool = True
    optimize_z: bool = True
    use_reward: bool = True

    # --- Logging ---
    save_progress: bool = False
    save_every: int = 10
    save_c_traj: bool = False
    save_z_traj: bool = False     # save per-step (z_before_refine, z_after_refine) pairs


# ---------------------------------------------------------------------------
# Preset configurations for common experimental setups
# ---------------------------------------------------------------------------

def sd15_defaults() -> PGMAPConfig:
    """Default PG-MAP config for Stable Diffusion 1.5 (paper Table 3)."""
    return PGMAPConfig(
        num_steps=30,
        guidance_scale=7.5,
        height=512,
        width=512,
        rho=0.4,
        refinement=RefinementConfig(K=1, eta_c=1e-4, eta_z=0.005),
        prior=PriorConfig(sigma_c=1.0, gamma=0.5),
        reward=RewardConfig(lambda_reward=0.1, rho_Q=0.3),
    )


def sdxl_defaults() -> PGMAPConfig:
    """Default PG-MAP config for SDXL (paper Table 3)."""
    return PGMAPConfig(
        num_steps=50,
        guidance_scale=5.0,
        height=1024,
        width=1024,
        rho=0.5,
        refinement=RefinementConfig(K=1, eta_c=1e-4, eta_z=0.005),
        prior=PriorConfig(sigma_c=1.0, gamma=0.5),
        reward=RewardConfig(lambda_reward=0.1, rho_Q=0.3),
    )


def baseline_config(backbone: str = "sd15") -> PGMAPConfig:
    """Standard DDIM+CFG baseline (no refinement)."""
    cfg = sd15_defaults() if backbone == "sd15" else sdxl_defaults()
    cfg.optimize_c = False
    cfg.optimize_z = False
    cfg.use_reward = False
    return cfg


def mapc_config(backbone: str = "sd15") -> PGMAPConfig:
    """MAP-c: conditioning-only MAP (no latent opt, no reward)."""
    cfg = sd15_defaults() if backbone == "sd15" else sdxl_defaults()
    cfg.optimize_c = True
    cfg.optimize_z = False
    cfg.use_reward = False
    return cfg


def reward_z_config(backbone: str = "sd15") -> PGMAPConfig:
    """Reward-z: latent-only optimization with reward."""
    cfg = sd15_defaults() if backbone == "sd15" else sdxl_defaults()
    cfg.optimize_c = False
    cfg.optimize_z = True
    cfg.use_reward = True
    return cfg


def joint_cz_config(backbone: str = "sd15") -> PGMAPConfig:
    """Joint (c,z) optimization without reward."""
    cfg = sd15_defaults() if backbone == "sd15" else sdxl_defaults()
    cfg.optimize_c = True
    cfg.optimize_z = True
    cfg.use_reward = False
    return cfg


def pgmap_gs_config(backbone: str = "sd15", K: int = 1) -> PGMAPConfig:
    """PG-MAP with Gauss-Seidel inner-loop updates.

    Each inner step costs 2 UNet forwards (vs 1 for Jacobi):
      - Forward 1: at (c, z_fixed)  → ∇_c J → update c
      - Forward 2: at (c_new, z)    → ∇_z J → update z

    The z gradient uses the freshly updated c, capturing c→z coupling that
    Jacobi misses. For NFE-fair comparison: GS K=1 ≈ Jacobi K=2.
    """
    cfg = sd15_defaults() if backbone == "sd15" else sdxl_defaults()
    cfg.optimize_c = True
    cfg.optimize_z = True
    cfg.use_reward = True
    cfg.refinement = RefinementConfig(K=K, eta_c=1e-4, eta_z=5e-3, gauss_seidel=True)
    return cfg


def pgmap_adam_config(backbone: str = "sd15", K: int = 3,
                      eta_z: float = 5e-5, eta_c: float = 1e-5) -> PGMAPConfig:
    """PG-MAP with Adam inner-loop optimizer.

    WHY the smaller default lr vs SGD (eta_z 5e-3 → 5e-5):
    Adam re-initialises its state every denoising step. At inner step k=1 the
    bias-corrected Adam update reduces to signSGD: Δθ ≈ -α · sign(g).  The
    effective per-element step is α, independent of ||g||.  SGD's effective
    step is η·||g||_elem.  Empirically the combined gradient RMS is ~0.01,
    so matching effective step size requires α_Adam ≈ η_SGD · RMS ≈ 5e-3·0.01
    = 5e-5.  Use the diagnostic cell in pgac.ipynb to measure the actual RMS
    and adjust if needed.

    For K ≥ 2 momentum accumulates across inner steps, partially recovering
    Adam's convergence advantage; still use the calibrated lr as the base.
    """
    cfg = sd15_defaults() if backbone == "sd15" else sdxl_defaults()
    cfg.optimize_c = True
    cfg.optimize_z = True
    cfg.use_reward = True
    cfg.refinement = RefinementConfig(
        K=K,
        eta_c=eta_c,
        eta_z=eta_z,
        optimizer="adam",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
    )
    return cfg


# ---------------------------------------------------------------------------
# Semantic Direction-Constrained PG-MAP configuration
# ---------------------------------------------------------------------------

@dataclass
class SemanticConfig:
    """Configuration for Semantic Direction-Constrained PG-MAP.

    Controls the online subspace estimation, timestep-dependent subspace
    dimensionality, and anisotropic prior regularization.

    Attributes:
        k_early:      Number of semantic components in the early stage
                      (t_frac > 0.66 — coarse structure / compositional binding).
        k_mid:        Number of semantic components in the middle stage
                      (0.33 < t_frac ≤ 0.66 — attribute refinement).
        k_late:       Number of semantic components in the late stage
                      (t_frac ≤ 0.33 — fine texture / aesthetic detail).
        k_c:          Number of semantic components for the conditioning
                      embedding c (timestep-independent).
        buffer_size:  Maximum gradient vectors kept in the rolling buffer
                      for online PCA.
        min_samples:  Minimum samples required before the subspace is used.
                      Below this threshold the vanilla PG-MAP update is used.
        alpha_sem:    Relative variance weight along semantic directions
                      (> 1 → more lenient; paper α in Eq. 132).
        beta_nonsem:  Relative variance weight perpendicular to the subspace
                      (< 1 → tighter penalty; paper β in Eq. 132).
    """
    k_early: int = 8
    k_mid: int = 16
    k_late: int = 32
    k_c: int = 16
    buffer_size: int = 30
    min_samples: int = 8
    alpha_sem: float = 2.0
    beta_nonsem: float = 0.5
    cos_gate_threshold: float = 0.1  # min cos-sim for projection to activate; 0 = always project
    proj_scale: float = 0.3          # scale factor applied to projected gradient (fallback unchanged)


def sdpgmap_sd15_defaults() -> "SDPGMAPConfig":
    """Default SD-PG-MAP config for Stable Diffusion 1.5."""
    return SDPGMAPConfig(
        pgmap=sd15_defaults(),
        semantic=SemanticConfig(
            k_early=8, k_mid=16, k_late=32, k_c=16,
            buffer_size=30, min_samples=8,
            alpha_sem=2.0, beta_nonsem=0.5,
            cos_gate_threshold=0.1,
            proj_scale=0.3,
        ),
    )


def sdpgmap_sdxl_defaults() -> "SDPGMAPConfig":
    """Default SD-PG-MAP config for SDXL."""
    return SDPGMAPConfig(
        pgmap=sdxl_defaults(),
        semantic=SemanticConfig(
            k_early=8, k_mid=16, k_late=32, k_c=16,
            buffer_size=30, min_samples=8,
            alpha_sem=2.0, beta_nonsem=0.5,
            cos_gate_threshold=0.1,
            proj_scale=0.3,
        ),
    )


@dataclass
class SDPGMAPConfig:
    """Top-level config for Semantic Direction-Constrained PG-MAP.

    Wraps a standard PGMAPConfig for the base method and adds
    a SemanticConfig for the direction-magnitude decoupling,
    timestep-dependent subspaces, and anisotropic prior.
    """
    pgmap: PGMAPConfig = field(default_factory=sd15_defaults)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
