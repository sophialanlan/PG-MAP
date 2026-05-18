#!/usr/bin/env bash
# SLURM batch wrapper for Table 1 SDXL reproduction.
#SBATCH --job-name=pgmap_table1_sdxl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/table1_sdxl_%j.out
#SBATCH --error=logs/table1_sdxl_%j.err

set -euo pipefail
source "${CONDA_PREFIX_OVERRIDE:-/opt/miniconda3}/etc/profile.d/conda.sh"
conda activate "${PGMAP_ENV:-pgmap}"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs
bash scripts/reproduce_table1_sdxl.sh
