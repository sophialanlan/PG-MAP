#!/usr/bin/env bash
# =============================================================================
# Reproduce Table 1, SDXL panel.
#
# Generates 8 rows on PartiPrompts (n=1632, seed=123):
#   1. Baseline (reference, static DDIM + CFG, w=5.0)
#   2. Tuned-CFG (w*=7.5)
#   3. UG (NFE-matched Universal Guidance, val-tuned eta)
#   4. MAP-c
#   5. Reward-z
#   6. MAP-cz (lambda=0, reward-free)
#   7. PG-MAP (default, our recommended row, w=5.0)
#   8. Tuned-CFG + PG-MAP (w=7.5)
#
# Compute: ~14 h on H200, ~28 h on A100 (1024^2 is 4x slower per pixel).
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_common.sh"

OUT_DIR="${OUT_ROOT}/table1_sdxl"
mkdir -p "${OUT_DIR}"
banner "Reproducing Table 1 — SDXL — output: ${OUT_DIR}"

# Rows 1+4+5+6+7: baseline + variants + PG-MAP (shared baseline reference)
pgmap_eval \
    --backbone sdxl \
    --model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --num_prompts 1632 \
    --seed 123 \
    --methods baseline mapc reward_z joint_cz pgmap \
    --score \
    --out_dir "${OUT_DIR}" \
    --steps 50 \
    --guidance 5.0 \
    --rho 0.5 \
    --rho_Q 0.3 \
    --K 2 \
    --eta_c 1.0e-3 \
    --eta_z 0.005 \
    --sigma_c 1.0 \
    --gamma 1.0 \
    --lambda_reward 0.1 \
    --reward_model pickscore \
    --grad_norm_strategy unit

# Row 8: Tuned-CFG + PG-MAP (w*=7.5)
TCFG_DIR="${OUT_DIR}/tuned_cfg_pgmap"
mkdir -p "${TCFG_DIR}"
if [ ! -e "${TCFG_DIR}/baseline" ]; then
    ln -s "${OUT_DIR}/baseline" "${TCFG_DIR}/baseline"
fi
pgmap_eval \
    --backbone sdxl \
    --num_prompts 1632 \
    --seed 123 \
    --methods pgmap \
    --score \
    --out_dir "${TCFG_DIR}" \
    --steps 50 \
    --guidance 7.5 \
    --rho 0.5 \
    --rho_Q 0.3 \
    --K 2 \
    --eta_c 1.0e-3 \
    --eta_z 0.005 \
    --sigma_c 1.0 \
    --gamma 1.0 \
    --lambda_reward 0.1 \
    --reward_model pickscore \
    --grad_norm_strategy unit

# Rows 2+3: Tuned-CFG and NFE-matched UG (val-tuned eta)
if [ -f "${REPO_ROOT}/benchmark_cfg.py" ]; then
    (cd "${REPO_ROOT}" && "${PYTHON}" benchmark_cfg.py --backbone sdxl \
        --out_dir "${OUT_DIR}/tuned_cfg_only" --device cuda --dtype fp16)
fi
if [ -f "${REPO_ROOT}/benchmark_ug.py" ]; then
    # NFE matching for SDXL: K=2, rho=0.5, rho_Q=0.3, T=50 -> K_ug=3
    (cd "${REPO_ROOT}" && "${PYTHON}" benchmark_ug.py --backbone sdxl \
        --phase tune --K_ug 3 --rho_Q 0.3 \
        --out_dir "${OUT_DIR}/ug" --device cuda --dtype fp16)
    BEST_ETA=$("${PYTHON}" -c "import json; print(json.load(open('${OUT_DIR}/ug/tune/best_eta.json'))['pickscore']['eta_z'])")
    (cd "${REPO_ROOT}" && "${PYTHON}" benchmark_ug.py --backbone sdxl \
        --phase test --eta_z "${BEST_ETA}" --K_ug 3 --rho_Q 0.3 \
        --out_dir "${OUT_DIR}/ug" --device cuda --dtype fp16)
fi

banner "Done. Per-method scores: ${OUT_DIR}/<method>/scores_summary.json"
