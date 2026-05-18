#!/usr/bin/env bash
# =============================================================================
# Smoke test: SD3.5-medium UG-FM, 4 prompts, ~5 min on one A100/H200.
#
# Note: requires read access to stabilityai/stable-diffusion-3.5-medium on
# the HuggingFace Hub (login + license acceptance required).
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../../scripts/_common.sh"

OUT_DIR="${OUT_ROOT}/smoke_fm"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

banner "Smoke test — SD3.5 UG-FM — output: ${OUT_DIR}"

(cd "${REPO_ROOT}" && "${PYTHON}" run_fm_variants_experiment.py \
    --n 4 \
    --seed 42 \
    --out_dir "${OUT_DIR}" \
    --methods baseline ug_fm_data \
    --K_ug 4 \
    --eta_z 0.1)

"${PYTHON}" "${SCRIPT_DIR}/_check_smoke.py" "${OUT_DIR}" ug_fm_data baseline
echo "[SMOKE] PASS — all 4 prompts generated and scored"
