#!/usr/bin/env python3
"""Validate smoke-test output: image counts, score JSON presence, metric ranges.

Usage:
    python _check_smoke.py <out_dir> <method1> [method2 ...]

Exits 0 on success, nonzero with a diagnostic on any failure.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


METRIC_BOUNDS = {
    # Loose sanity bounds for "did anything obviously break".
    "pickscore": (0.10, 0.40),
    "hps":       (0.10, 0.45),
    "clip":      (0.10, 0.50),
    "aes":       (1.0,  10.0),
}


def fail(msg: str) -> None:
    print(f"[SMOKE] FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def check_method(out_dir: Path, method: str) -> None:
    method_dir = out_dir / method
    if not method_dir.is_dir():
        fail(f"missing method dir: {method_dir}")

    img_dir = method_dir / "images"
    if not img_dir.is_dir():
        fail(f"missing images dir: {img_dir}")
    images = sorted(img_dir.glob("*.png"))
    if not images:
        fail(f"no PNGs in {img_dir}")

    summary_path = method_dir / "scores_summary.json"
    if summary_path.exists():
        with open(summary_path) as fh:
            summary = json.load(fh)
        means = summary.get("means", summary)
        for metric, (lo, hi) in METRIC_BOUNDS.items():
            if metric not in means:
                continue
            v = means[metric]
            if isinstance(v, dict):
                v = v.get("trained", v.get("mean"))
            if v is None:
                continue
            if not (lo <= v <= hi):
                fail(f"{method}/{metric}={v:.3f} outside loose smoke bounds [{lo}, {hi}]")
        print(f"  {method}: {len(images)} images, score summary OK")
    else:
        print(f"  {method}: {len(images)} images (no scores_summary.json — scoring may have been skipped)")


def main() -> None:
    if len(sys.argv) < 3:
        fail("usage: _check_smoke.py <out_dir> <method> [<method> ...]")

    out_dir = Path(sys.argv[1])
    if not out_dir.is_dir():
        fail(f"missing out_dir: {out_dir}")

    for method in sys.argv[2:]:
        check_method(out_dir, method)

    print("[SMOKE] all checks passed")


if __name__ == "__main__":
    main()
