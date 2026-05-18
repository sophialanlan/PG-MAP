# Common shell helpers sourced by every reproduce_*.sh script.
# Sets cache env vars, picks a Python, and exposes a `pgmap_eval` shorthand.

set -euo pipefail

# ---- Repo root (resolves regardless of where you invoke the script from) ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

# ---- HuggingFace + HPS caches (override via env if you want elsewhere) ----
: "${PGMAP_CACHE_DIR:=${REPO_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${PGMAP_CACHE_DIR}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PGMAP_CACHE_DIR}/hf_home/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${PGMAP_CACHE_DIR}/hf_home/transformers}"
export HPS_CACHE="${HPS_CACHE:-${PGMAP_CACHE_DIR}/hps_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${HF_HOME}" "${HPS_CACHE}"

# ---- Python ----
: "${PYTHON:=python}"
export PYTHON

# ---- Output root ----
: "${OUT_ROOT:=${REPO_ROOT}/eval_results}"
export OUT_ROOT
mkdir -p "${OUT_ROOT}"

# ---- Helpers ----
banner() {
    echo "=================================================================="
    echo "$*"
    echo "=================================================================="
}

# Run pgmap_eval with the repo path baked in.
pgmap_eval() {
    (cd "${REPO_ROOT}" && "${PYTHON}" pgmap_eval.py "$@")
}
