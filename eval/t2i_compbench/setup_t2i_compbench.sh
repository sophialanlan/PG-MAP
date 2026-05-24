#!/usr/bin/env bash
# Clone the upstream T2I-CompBench repo (eval scripts only) and install
# the BLIP-VQA / CLIPScore / UniDet eval dependencies into the current
# conda env. Idempotent.
set -euo pipefail

UPSTREAM_DIR="${PGMAP_T2I_UPSTREAM:-/tmp/pgmap_t2icompbench}"

if [ ! -d "$UPSTREAM_DIR" ]; then
    echo "Cloning T2I-CompBench upstream into $UPSTREAM_DIR ..."
    git clone --depth 1 https://github.com/Karine-Huang/T2I-CompBench.git "$UPSTREAM_DIR"
else
    echo "Upstream already at $UPSTREAM_DIR (delete it to force re-clone)."
fi

echo
echo "Setup complete."
echo "  Upstream eval scripts: $UPSTREAM_DIR"
echo "  Next: run generation via eval/t2i_compbench/generate.py, then BLIP-VQA via run_blip_vqa.sh"
