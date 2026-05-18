#!/usr/bin/env bash
# =============================================================================
# Reproduce Table 1, Stable Diffusion 1.5 panel.
#
# Generates 8 rows on PartiPrompts (n=1632, seed=123):
#   1. Baseline (reference, static DDIM + CFG)
#   2. Tuned-CFG (w*=12.0)
#   3. UG (NFE-matched Universal Guidance, val-tuned eta)
#   4. MAP-c
#   5. Reward-z
#   6. MAP-cz (lambda=0, reward-free)
#   7. PG-MAP (default, our recommended row)
#   8. Tuned-CFG + PG-MAP
#
# Compute: ~3 h on H200 (or ~6 h on A100). Each row is independent — comment
# out rows to run a subset.
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_common.sh"

OUT_DIR="${OUT_ROOT}/table1_sd15"
mkdir -p "${OUT_DIR}"
banner "Reproducing Table 1 — SD 1.5 — output: ${OUT_DIR}"

# ---------------------------------------------------------------------------
# Row 1 + 4 + 5 + 6 + 7: baseline, mapc, reward_z, joint_cz (MAP-cz), pgmap
# All run together so the *same* baseline images are reused as the win-rate
# reference for every other row.
# ---------------------------------------------------------------------------
pgmap_eval \
    --backbone sd15 \
    --model_id runwayml/stable-diffusion-v1-5 \
    --num_prompts 1632 \
    --seed 123 \
    --methods baseline mapc reward_z joint_cz pgmap \
    --score \
    --out_dir "${OUT_DIR}" \
    --steps 30 \
    --guidance 7.5 \
    --rho 0.4 \
    --rho_Q 0.3 \
    --K 2 \
    --eta_c 1.0e-4 \
    --eta_z 0.005 \
    --sigma_c 1.0 \
    --gamma 1.0 \
    --lambda_reward 0.05 \
    --reward_model pickscore \
    --grad_norm_strategy unit

# ---------------------------------------------------------------------------
# Row 8: Tuned-CFG + PG-MAP (w*=12.0). Reuses the static baseline images so
# the win-rate is computed against the same reference as the other rows.
# ---------------------------------------------------------------------------
TCFG_DIR="${OUT_DIR}/tuned_cfg_pgmap"
mkdir -p "${TCFG_DIR}"
if [ ! -e "${TCFG_DIR}/baseline" ]; then
    ln -s "${OUT_DIR}/baseline" "${TCFG_DIR}/baseline"
fi
pgmap_eval \
    --backbone sd15 \
    --num_prompts 1632 \
    --seed 123 \
    --methods pgmap \
    --score \
    --out_dir "${TCFG_DIR}" \
    --steps 30 \
    --guidance 12.0 \
    --rho 0.4 \
    --rho_Q 0.3 \
    --K 2 \
    --eta_c 1.0e-4 \
    --eta_z 0.005 \
    --sigma_c 1.0 \
    --gamma 1.0 \
    --lambda_reward 0.05 \
    --reward_model pickscore \
    --grad_norm_strategy unit

# ---------------------------------------------------------------------------
# Row 2 + 3: Tuned-CFG only and NFE-matched Universal Guidance.
# These reuse benchmark_cfg.py / benchmark_ug.py respectively, both of which
# match PG-MAP's NFE budget exactly for an apples-to-apples comparison.
# ---------------------------------------------------------------------------
if [ -f "${REPO_ROOT}/benchmark_cfg.py" ]; then
    (cd "${REPO_ROOT}" && "${PYTHON}" benchmark_cfg.py --backbone sd15 \
        --out_dir "${OUT_DIR}/tuned_cfg_only" --device cuda --dtype fp16)
fi
if [ -f "${REPO_ROOT}/benchmark_ug.py" ]; then
    (cd "${REPO_ROOT}" && "${PYTHON}" benchmark_ug.py --backbone sd15 \
        --phase tune --K_ug 3 --rho_Q 0.3 \
        --out_dir "${OUT_DIR}/ug" --device cuda --dtype fp16)
    BEST_ETA=$("${PYTHON}" -c "import json; print(json.load(open('${OUT_DIR}/ug/tune/best_eta.json'))['pickscore']['eta_z'])")
    (cd "${REPO_ROOT}" && "${PYTHON}" benchmark_ug.py --backbone sd15 \
        --phase test --eta_z "${BEST_ETA}" --K_ug 3 --rho_Q 0.3 \
        --out_dir "${OUT_DIR}/ug" --device cuda --dtype fp16)
fi

banner "Done. Per-method scores: ${OUT_DIR}/<method>/scores_summary.json"
