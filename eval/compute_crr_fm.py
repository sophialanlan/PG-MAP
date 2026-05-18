#!/usr/bin/env python3
"""
CRR-MAP analysis on the FM transport (SD3.5-medium, n=1632, seed 123).

Routing pool — three specializations of the FM-MAP framework's MAP-$z$+reward
specialization (the only MAP-FM specialization that is structurally
applicable; c-optimization off-manifold-collapses on FM per Section 5.5/M2).
The three differ only in the operating regime (gate side, step size):

  f_data           : MAP-z + reward, data-side gate, eta_z = 0.1
                     (paper headline 91.9% PS; low-amplification refinement)
  f_data_high_eta  : MAP-z + reward, data-side gate, eta_z = 0.2
                     (more aggressive data-side step; paper 83.4% PS)
  f_noise          : MAP-z + reward, noise-side gate, eta_z = 0.1
                     (high-amplification structural redirection per M2)

Source files (all at n=1632, seed 123):
  flow_pgmap_sd35_n1632_seed123/flow_ug/   -- UG-FM data-side eta_z=0.1
  flow_ug_eta02_n1632_seed123/flow_ug/      -- UG-FM data-side eta_z=0.2
  flow_v13_noise_z_n1632_seed123/flow_pgmap/-- MAP-z+reward noise-side eta_z=0.1
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
from pathlib import Path

EVAL = Path(os.environ.get("PG_MAP_EVAL_DIR",
            str(Path(__file__).resolve().parent.parent / "eval_results")))
# Default FM pool layout — produced by scripts/reproduce_table2_fm.sh and the
# FM-variants driver. Override via --table2_dir.
DEFAULT_TABLE2 = EVAL / "table2_fm"
POOL_KEYS = ("f_data", "f_data_high_eta", "f_noise")
METRICS = ["pickscore", "hps", "clip", "aes"]


def load(path):
    return sorted([json.loads(l) for l in open(path)], key=lambda r: r["i"])


def main():
    print("=" * 60)
    print("FM CRR-MAP: framework-aligned routing pool on SD3.5-medium")
    print("=" * 60)
    pool = {name: load(p) for name, p in POOL.items()}
    n = len(next(iter(pool.values())))
    # verify alignment
    refs = [r["prompt"] for r in next(iter(pool.values()))]
    for name, rows in pool.items():
        assert len(rows) == n
        for j, r in enumerate(rows):
            assert r["prompt"] == refs[j], f"prompt mismatch at {j}: {name}"
    print(f"\nPool aligned: n={n} prompts, methods={list(pool.keys())}")

    # Per-method win rates vs baseline (each row's own ref)
    standalone = {}
    for name, rows in pool.items():
        standalone[name] = {}
        for k in METRICS:
            wins = sum(1 for r in rows if r["metrics"][k]["delta"] > 0)
            mean_delta = np.mean([r["metrics"][k]["delta"] for r in rows])
            standalone[name][k] = {"win_rate": wins / n, "mean_delta": float(mean_delta)}

    print("\n## Per-method win rates vs baseline (n=1632)\n")
    print("| Method | PickScore | HPS | CLIP | Aesthetic |")
    print("|---|---|---|---|---|")
    for name in pool:
        row = [f"{100*standalone[name][k]['win_rate']:.2f}" for k in METRICS]
        print(f"| {name} | " + " | ".join(row) + " |")

    # Oracle Pareto-sum
    deltas = {name: {k: np.array([r["metrics"][k]["delta"] for r in rows]) for k in METRICS}
              for name, rows in pool.items()}
    z_scored = {}
    for name in pool:
        z_scored[name] = {}
        for k in METRICS:
            d = deltas[name][k]
            mu, sd = d.mean(), d.std()
            z_scored[name][k] = (d - mu) / (sd if sd > 1e-12 else 1.0)
    sums = {name: np.sum([z_scored[name][k] for k in METRICS], axis=0) for name in pool}
    chosen = []
    for i in range(n):
        scores = {name: sums[name][i] for name in pool}
        chosen.append(max(scores, key=scores.get))

    # Oracle aggregate
    oracle = {}
    for k in METRICS:
        wins = sum(1 for i in range(n) if pool[chosen[i]][i]["metrics"][k]["delta"] > 0)
        deltas_oracle = [pool[chosen[i]][i]["metrics"][k]["delta"] for i in range(n)]
        oracle[k] = {"win_rate": wins / n, "mean_delta": float(np.mean(deltas_oracle))}

    print("\n## Oracle (Pareto-sum) win rates vs baseline (n=1632)\n")
    print("| Method | PickScore | HPS | CLIP | Aesthetic |")
    print("|---|---|---|---|---|")
    print(f"| **CRR-FM (oracle)** | " + " | ".join(f"{100*oracle[k]['win_rate']:.2f}" for k in METRICS) + " |")

    # Pareto improvement vs best individual
    print("\n## Pareto improvement (oracle - best individual)\n")
    for k in METRICS:
        best_individual = max(standalone[name][k]['win_rate'] for name in pool)
        gain = oracle[k]['win_rate'] - best_individual
        print(f"  {k}: oracle={100*oracle[k]['win_rate']:.2f}, best_individual={100*best_individual:.2f}, gain=+{100*gain:.2f}pp")

    # Routing distribution
    from collections import Counter
    dist = Counter(chosen)
    print("\n## Oracle routing distribution\n")
    for name in pool:
        c = dist.get(name, 0)
        print(f"  {name}: {c} prompts ({100*c/n:.1f}%)")

    # Save
    out = {
        "n": n,
        "pool": list(pool.keys()),
        "standalone": {name: {k: standalone[name][k] for k in METRICS} for name in pool},
        "oracle": {k: oracle[k] for k in METRICS},
        "routing_distribution": dict(dist),
    }
    with open("crr_fm_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[DONE] saved to crr_fm_results.json")


if __name__ == "__main__":
    main()
