#!/usr/bin/env bash
# =============================================================================
# Download the HPS-v2 checkpoint to utils/hps/HPS_v2_compressed.pt (1.9 GB).
# Required only for rows that score with --reward_model hps or for
# multi-reward robustness tables.
#
# Source: https://huggingface.co/xswu/HPSv2 (HuggingFace, no auth)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${REPO_ROOT}/utils/hps/HPS_v2_compressed.pt"
URL="https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"

mkdir -p "$(dirname "${TARGET}")"

if [ -f "${TARGET}" ]; then
    echo "Already downloaded: ${TARGET}"
    exit 0
fi

echo "Downloading HPS-v2 checkpoint (1.9 GB) to ${TARGET}..."
if command -v wget >/dev/null 2>&1; then
    wget --show-progress -c -O "${TARGET}.partial" "${URL}"
elif command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar -C - -o "${TARGET}.partial" "${URL}"
else
    echo "Need wget or curl in PATH." >&2
    exit 1
fi
mv "${TARGET}.partial" "${TARGET}"
echo "Done: $(du -h "${TARGET}" | awk '{print $1}')"
