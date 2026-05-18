#!/usr/bin/env bash
# =============================================================================
# Reproduce Table 4: CRR-MAP oracle-routing diagnostic.
#
# Aggregates the per-prompt scores from Table 1 (already-generated rows) and
# computes the oracle ceiling — the win-rate achieved if a perfect selector
# could dispatch each prompt to its best-scoring variant among
# {MAP-c (f_c), MAP-cz (f_cz), Tuned-CFG+PG-MAP (f_tcfg)}.
#
# Prerequisite: bash scripts/reproduce_table1_sd15.sh AND
#               bash scripts/reproduce_table1_sdxl.sh
# (or at least the mapc / joint_cz / tuned_cfg_pgmap rows of each).
#
# Optional: bash scripts/reproduce_table2_fm.sh  (for the SD3.5 oracle row).
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_common.sh"

OUT_DIR="${OUT_ROOT}/table4_crr"
mkdir -p "${OUT_DIR}"
banner "Reproducing Table 4 — CRR-MAP oracle — output: ${OUT_DIR}"

# --- SD 1.5 + SDXL oracle rows ---
# Reads each pool's scores.jsonl from ${OUT_ROOT}/table1_<backbone>/<method>/
# (see _default_pools() in eval/compute_crr_deployable.py).
"${PYTHON}" "${REPO_ROOT}/eval/compute_crr_deployable.py" \
    --eval_root "${OUT_ROOT}" \
    --backbones sdxl sd15 \
    --out "${OUT_DIR}/crr_deployable.json"

# --- FM oracle row (over UG-FM eta_z regimes) ---
if [ -d "${OUT_ROOT}/table2_fm" ]; then
    "${PYTHON}" "${REPO_ROOT}/eval/compute_crr_fm.py" \
        --table2_dir "${OUT_ROOT}/table2_fm" \
        --out "${OUT_DIR}/crr_fm.json"
fi

banner "Done. CRR-MAP outputs in: ${OUT_DIR}/"
