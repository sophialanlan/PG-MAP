#!/usr/bin/env bash
# SLURM batch wrapper for Table 1 SD 1.5 reproduction.
# Submit with:  sbatch scripts/slurm/sd15_table1.sbatch
#SBATCH --job-name=pgmap_table1_sd15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/table1_sd15_%j.out
#SBATCH --error=logs/table1_sd15_%j.err

set -euo pipefail

# Load conda — adapt these two lines to your cluster
source "${CONDA_PREFIX_OVERRIDE:-/opt/miniconda3}/etc/profile.d/conda.sh"
conda activate "${PGMAP_ENV:-pgmap}"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs
bash scripts/reproduce_table1_sd15.sh
