#!/usr/bin/env bash
# =============================================================================
# Smoke test: SD 1.5 PG-MAP, 4 prompts, ~2 min on one A100/H200.
#
# Verifies:
#   1. environment + model loading (HF cache, diffusers, transformers)
#   2. PG-MAP refinement runs without exceptions
#   3. PickScore reward backprop chain is intact (no detached gradients)
#   4. Scoring pipeline returns reasonable metric ranges
#
# Pass criterion: prints "[SMOKE] PASS — all 4 prompts generated and scored".
# Exit code 0 on success, nonzero on any failure.
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../../scripts/_common.sh"

OUT_DIR="${OUT_ROOT}/smoke_sd15"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

banner "Smoke test — SD 1.5 — output: ${OUT_DIR}"

pgmap_eval \
    --backbone sd15 \
    --num_prompts 4 \
    --seed 42 \
    --methods baseline pgmap \
    --score \
    --out_dir "${OUT_DIR}" \
    --steps 30 \
    --guidance 7.5 \
    --rho 0.4 \
    --rho_Q 0.3 \
    --K 2 \
    --eta_c 1.0e-4 \
    --eta_z 0.005 \
    --gamma 1.0 \
    --lambda_reward 0.05 \
    --reward_model pickscore \
    --grad_norm_strategy unit

"${PYTHON}" "${SCRIPT_DIR}/_check_smoke.py" "${OUT_DIR}" pgmap baseline
echo "[SMOKE] PASS — all 4 prompts generated and scored"
