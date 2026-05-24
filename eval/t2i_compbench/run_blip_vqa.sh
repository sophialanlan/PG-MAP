#!/usr/bin/env bash
# Run BLIP-VQA evaluation (T2I-CompBench's attribute-binding metric) on a
# generated samples directory.
#
# Expects: $1 = directory containing <category>/samples/*.png
# Output:  $1/<category>/annotation_blip/vqa_result.json + a single
#          aggregated score.
#
# Requires: BLIPvqa_eval/ from upstream T2I-CompBench at
#   /tmp/pgmap_t2icompbench/  (cloned by setup_t2i_compbench.sh).
set -euo pipefail

SAMPLES_ROOT="${1:?usage: $0 <samples_root>}"
SAMPLES_ROOT="$(cd "$SAMPLES_ROOT" && pwd)"
UPSTREAM="${PGMAP_T2I_UPSTREAM:-/tmp/pgmap_t2icompbench}"

if [ ! -d "$UPSTREAM" ]; then
    echo "Upstream T2I-CompBench repo not found at $UPSTREAM."
    echo "Run: bash eval/t2i_compbench/setup_t2i_compbench.sh"
    exit 1
fi

for cat in "$SAMPLES_ROOT"/*; do
    [ -d "$cat/samples" ] || continue
    cat_name="$(basename "$cat")"
    echo "=== BLIP-VQA: $cat_name ==="
    (
        cd "$UPSTREAM/BLIPvqa_eval"
        # T2I-CompBench's BLIP_vqa.py is wired to read from --out_dir/samples
        # and write to --out_dir/annotation_blip/vqa_result.json.
        python BLIP_vqa.py --out_dir="$cat" 2>&1 | tail -3
    )
done

echo
echo "[blip-vqa] done. Per-category results at <samples_root>/<cat>/annotation_blip/vqa_result.json"
