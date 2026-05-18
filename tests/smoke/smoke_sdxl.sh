#!/usr/bin/env bash
# =============================================================================
# Smoke test: SDXL PG-MAP, 4 prompts, ~5 min on one A100/H200.
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../../scripts/_common.sh"

OUT_DIR="${OUT_ROOT}/smoke_sdxl"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

banner "Smoke test — SDXL — output: ${OUT_DIR}"

pgmap_eval \
    --backbone sdxl \
    --num_prompts 4 \
    --seed 42 \
    --methods baseline pgmap \
    --score \
    --out_dir "${OUT_DIR}" \
    --steps 50 \
    --guidance 5.0 \
    --rho 0.5 \
    --rho_Q 0.3 \
    --K 2 \
    --eta_c 1.0e-3 \
    --eta_z 0.005 \
    --gamma 1.0 \
    --lambda_reward 0.1 \
    --reward_model pickscore \
    --grad_norm_strategy unit

"${PYTHON}" "${SCRIPT_DIR}/_check_smoke.py" "${OUT_DIR}" pgmap baseline
echo "[SMOKE] PASS — all 4 prompts generated and scored"
