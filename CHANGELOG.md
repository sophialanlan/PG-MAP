# Changelog

All notable changes to the public PG-MAP release.

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
