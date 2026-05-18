#!/usr/bin/env bash
# =============================================================================
# One-shot environment bootstrap.
#
#   bash scripts/setup_env.sh
#
# Creates a fresh conda env `pgmap` with all dependencies installed.
# Skip and use your own env if you already have torch + diffusers set up.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${1:-pgmap}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi

echo "Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

echo
echo "Done. Activate with:"
echo "    conda activate ${ENV_NAME}"
echo
echo "Optional: download the HPS-v2 reward checkpoint (1.9 GB) for HPS scoring:"
echo "    bash scripts/download_hps.sh"
