#!/usr/bin/env python3
"""Aggregate BLIP-VQA / UniDet / CLIPScore JSON outputs into a single
T2I-CompBench++ leaderboard-style table.

Reads ``<samples_root>/<category>/annotation_blip/vqa_result.json`` (and
similar) and prints a markdown row plus saves to CSV / JSON.

Usage::

    python eval/t2i_compbench/aggregate.py \\
        --samples_root eval_results/t2i_compbench/sdxl_pgmap_K1/ \\
        --method "SDXL + PG-MAP (K=1)" \\
        --out eval_results/t2i_compbench/sdxl_pgmap_K1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


CATEGORY_TO_EVAL = {
    "color":       "annotation_blip/vqa_result.json",
    "shape":       "annotation_blip/vqa_result.json",
    "texture":     "annotation_blip/vqa_result.json",
    "spatial":     "labels/annotation_obj_detection_2d/vqa_result.json",
    "non_spatial": "annotation_clip/vqa_result.json",
    "complex":     "annotation_3_in_1/vqa_result.json",
}


def load_category(samples_root: Path, category: str) -> float | None:
    rel = CATEGORY_TO_EVAL.get(category)
    if rel is None:
        return None
    fp = samples_root / category / rel
    if not fp.exists():
        return None
    rows = json.loads(fp.read_text())
    scores = []
    for r in rows:
        try:
            scores.append(float(r["answer"]))
        except (KeyError, TypeError, ValueError):
            continue
    return mean(scores) if scores else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_root", type=Path, required=True,
                    help="Output dir from eval/t2i_compbench/generate.py")
    ap.add_argument("--method", required=True,
                    help='Row label for the leaderboard (e.g. "SDXL + PG-MAP (K=1)").')
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON path for machine-readable output.")
    args = ap.parse_args()

    row = {"method": args.method, "samples_root": str(args.samples_root)}
    print(f"\n=== T2I-CompBench++ results — {args.method} ===")
    for cat in CATEGORY_TO_EVAL:
        score = load_category(args.samples_root, cat)
        row[cat] = score
        cell = f"{score:.4f}" if score is not None else "—"
        print(f"  {cat:>12}: {cell}")

    # Markdown row for the README
    cells = []
    for cat in ("color", "shape", "texture", "spatial", "non_spatial", "complex"):
        v = row.get(cat)
        cells.append(f"{v:.4f}" if v is not None else "—")
    print("\nMarkdown row (paste into the leaderboard table):")
    print(f"| {args.method} | {' | '.join(cells)} |")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, indent=2))
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
