#!/usr/bin/env bash
# =============================================================================
# Reproduce every paper table sequentially.
#
# Wall-clock estimates on a single H200:
#   Table 1 SD 1.5  : ~3 h
#   Table 1 SDXL    : ~14 h
#   Table 2 SD3.5   : ~20 h
#   Table 4 CRR     : ~5 min (post-hoc aggregation only)
#   Total           : ~37 h
#
# For SLURM clusters, prefer scripts/slurm/*.sbatch — those parallelize the
# rows across multiple nodes.
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/reproduce_table1_sd15.sh"
bash "${SCRIPT_DIR}/reproduce_table1_sdxl.sh"
bash "${SCRIPT_DIR}/reproduce_table2_fm.sh"
bash "${SCRIPT_DIR}/reproduce_table4_crr.sh"

echo
echo "All tables reproduced. Per-method scores at eval_results/<table>/<method>/scores_summary.json"
