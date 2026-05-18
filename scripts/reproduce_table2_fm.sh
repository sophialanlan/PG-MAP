#!/usr/bin/env bash
# =============================================================================
# Reproduce Table 2: SD3.5-medium flow matching headline + FlowChef head-to-head.
#
# Generates 4 rows on PartiPrompts (n=1632, seed=123):
#   1. Baseline (static rectified-flow Euler, 28 steps, cfg=7.0)
#   2. FlowChef (always-on, gradient skipping, eta=1.0)
#   3. FlowChef (gating-matched, data-side, eta=1.0)
#   4. UG-FM (Ours, K=4, eta_z=0.1, data-side, full backprop)
#
# Compute: ~20 h on H200. SD3.5 + reward backward peaks at ~50 GB VRAM.
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_common.sh"

OUT_DIR="${OUT_ROOT}/table2_fm"
mkdir -p "${OUT_DIR}"
banner "Reproducing Table 2 — SD3.5-medium FM — output: ${OUT_DIR}"

# -----------------------------------------------------------------------------
# Rows 1 + 4: baseline (auto) + UG-FM. The FM-variants driver always runs the
# static FM baseline first, then each `--methods` entry against it.
# -----------------------------------------------------------------------------
(cd "${REPO_ROOT}" && "${PYTHON}" run_fm_variants_experiment.py \
    --n 1632 \
    --seed 123 \
    --out_dir "${OUT_DIR}" \
    --methods ug_fm_data)
# baseline is emitted at ${OUT_DIR}/baseline/, UG-FM at ${OUT_DIR}/ug_fm_data/.

# -----------------------------------------------------------------------------
# Rows 2 + 3: FlowChef head-to-head — only differs from UG-FM by `torch.no_grad()`
# around v_theta. Both runs symlink ${OUT_DIR}/baseline images as the shared
# reference so win-rates are directly comparable.
# -----------------------------------------------------------------------------
FLOWCHEF_DIR="${REPO_ROOT}/flowchef_baseline"

(cd "${FLOWCHEF_DIR}" && "${PYTHON}" flowchef_eval.py \
    --num_prompts 1632 \
    --seed 123 \
    --K 1 \
    --eta_z 1.0 \
    --gate_side alwayson \
    --out_root "${OUT_DIR}/flowchef_alwayson")

(cd "${FLOWCHEF_DIR}" && "${PYTHON}" flowchef_eval.py \
    --num_prompts 1632 \
    --seed 123 \
    --K 1 \
    --eta_z 1.0 \
    --gate_side data \
    --out_root "${OUT_DIR}/flowchef_dataside")

# -----------------------------------------------------------------------------
# Win-rates land in each method's scores_summary.json. Aggregate them.
# -----------------------------------------------------------------------------
"${PYTHON}" "${REPO_ROOT}/eval/aggregate_table2.py" \
    --root "${OUT_DIR}" \
    --baseline baseline \
    --methods ug_fm_data flowchef_alwayson/flow_flowchef flowchef_dataside/flow_flowchef \
    --out "${OUT_DIR}/table2_winrates.json"

banner "Done. Headline: ${OUT_DIR}/table2_winrates.json"
echo "Per-method scores: ${OUT_DIR}/<method>/scores_summary.json"
