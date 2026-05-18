#!/usr/bin/env python3
"""Aggregate per-method win-rates from a Table 2 (FM) eval directory.

Reads each `<root>/<method>/scores_summary.json` and emits a single JSON
with the headline numbers in the format used by the paper. Generic version
of the legacy `flowchef_baseline/aggregate_results.py` with explicit paths
(no hardcoded scratch directories).

Usage:
    python eval/aggregate_table2.py \\
        --root eval_results/table2_fm \\
        --baseline baseline \\
        --methods ug_fm_data flowchef_alwayson/flow_flowchef flowchef_dataside/flow_flowchef \\
        --out eval_results/table2_fm/table2_winrates.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


METRICS = ("pickscore", "hps", "clip", "aes")


def load_summary(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--baseline", default="baseline",
                    help="Method dir to use as the win-rate reference (just for sanity printing).")
    ap.add_argument("--methods", nargs="+", required=True,
                    help="Method dir names (relative to --root) to aggregate.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    root: Path = args.root
    out: dict = {"root": str(root), "baseline": args.baseline, "rows": {}}

    for m in args.methods:
        summary_path = root / m / "scores_summary.json"
        wil_path     = root / m / "wilcoxon.json"
        boot_path    = root / m / "bootstrap.json"
        summary = load_summary(summary_path)
        if not summary:
            print(f"[warn] missing {summary_path}", file=sys.stderr)
            out["rows"][m] = None
            continue
        wil = load_summary(wil_path)
        boot = load_summary(boot_path)
        entry = {"n": None}
        for metric in METRICS:
            d = summary.get(metric, {})
            entry[metric] = {
                "win_rate": d.get("win_rate"),
                "mean_delta": d.get("mean_delta"),
            }
            if entry["n"] is None:
                entry["n"] = d.get("n")
            entry[f"{metric}_wilcoxon_p"] = wil.get(metric, {}).get("p_value")
            entry[f"{metric}_ci95"]       = boot.get(metric, {}).get("ci_95")
        out["rows"][m] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    # Pretty print
    print(f"\n{'Method':<40} {'PickScore':>10} {'HPS':>10} {'CLIP':>10} {'Aes':>10}")
    print("-" * 84)
    for m, row in out["rows"].items():
        if row is None:
            print(f"{m:<40}  (no data)")
            continue
        cells = [f"{row[k]['win_rate']*100:.1f}%" if row.get(k) and row[k].get("win_rate") is not None else "n/a"
                 for k in METRICS]
        print(f"{m:<40} {cells[0]:>10} {cells[1]:>10} {cells[2]:>10} {cells[3]:>10}")
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
