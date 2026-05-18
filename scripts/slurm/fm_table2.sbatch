#!/usr/bin/env bash
# SLURM batch wrapper for Table 2 SD3.5 FM reproduction.
#SBATCH --job-name=pgmap_table2_fm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/table2_fm_%j.out
#SBATCH --error=logs/table2_fm_%j.err

set -euo pipefail
source "${CONDA_PREFIX_OVERRIDE:-/opt/miniconda3}/etc/profile.d/conda.sh"
conda activate "${PGMAP_ENV:-pgmap}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs
bash scripts/reproduce_table2_fm.sh
