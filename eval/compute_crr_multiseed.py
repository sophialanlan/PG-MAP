#!/usr/bin/env python3
"""
Multi-seed CRR-MAP oracle stability check (n=200, 5 seeds, both backbones).
Uses existing scored data in eval_results/multiseed_5seed_pilot/.

Pool members per seed (the {mapc, joint_cz, pgmap} variant; f_tcfg has no
multi-seed data, so we report this 3-method pool — the paper's main pool
{f_c, f_cz, f_tcfg} replaces 'pgmap' with 'tcfg', but the 'pgmap' rows of
this 3-pool version still let us measure single-seed-vs-population stability
of the oracle Pareto-sum gain.)
"""
from __future__ import annotations
import json, os
import numpy as np
from pathlib import Path

import os as _os
EVAL = Path(_os.environ.get("PG_MAP_EVAL_DIR",
            str(Path(__file__).resolve().parent.parent / "eval_results")))
SEEDS = [42, 123, 456, 789, 2024]
METRICS = ["pickscore", "hps", "clip", "aes"]


def load_scores(path):
    rows = []
    with open(path) as f:
        for line in f: rows.append(json.loads(line.strip()))
    return rows


def per_seed_oracle(backbone, seed):
    """Returns per-method win rates and oracle Pareto-sum win rate for one seed."""
    base = EVAL / "multiseed_5seed_pilot" / backbone / f"seed_{seed}"
    pools = {}
    for m in ["mapc", "joint_cz", "pgmap"]:
        rows = sorted(load_scores(base / m / "scores.jsonl"), key=lambda r: r["i"])
        pools[m] = rows
    n = len(next(iter(pools.values())))
    # Each method's standalone win rate (vs its own ref baseline)
    standalone = {}
    for m, rows in pools.items():
        standalone[m] = {k: float(np.mean([r["metrics"][k]["delta"] > 0 for r in rows])) for k in METRICS}
    # Oracle Pareto-sum: per-prompt argmax of within-method z-scored deltas summed across metrics
    deltas = {m: {k: np.array([r["metrics"][k]["delta"] for r in rows]) for k in METRICS}
              for m, rows in pools.items()}
    z_scored = {}
    for m in pools:
        z_scored[m] = {}
        for k in METRICS:
            d = deltas[m][k]
            mu, sd = d.mean(), d.std()
            z_scored[m][k] = (d - mu) / (sd if sd > 1e-12 else 1.0)
    sums = {m: np.sum([z_scored[m][k] for k in METRICS], axis=0) for m in pools}
    chosen = []
    for i in range(n):
        scores = {m: sums[m][i] for m in pools}
        chosen.append(max(scores, key=scores.get))
    # Oracle win rate per metric: dispatch picks method per prompt; compute win against that method's own ref
    oracle = {}
    for k in METRICS:
        wins = 0
        for i in range(n):
            r = pools[chosen[i]][i]
            wins += 1 if r["metrics"][k]["delta"] > 0 else 0
        oracle[k] = wins / n
    return {"n": n, "standalone": standalone, "oracle": oracle, "dispatch": chosen}


def main():
    print("=" * 60)
    print("Multi-seed CRR-MAP oracle stability (n=200, 5 seeds)")
    print("Pool: {mapc, joint_cz, pgmap}")
    print("=" * 60)
    out = {}
    for backbone in ["sdxl", "sd15"]:
        per_seed = {}
        for s in SEEDS:
            res = per_seed_oracle(backbone, s)
            per_seed[s] = res

        # Aggregate across seeds
        std_methods = ["mapc", "joint_cz", "pgmap"]
        agg_standalone = {m: {k: [] for k in METRICS} for m in std_methods}
        agg_oracle = {k: [] for k in METRICS}
        for s, res in per_seed.items():
            for m in std_methods:
                for k in METRICS:
                    agg_standalone[m][k].append(res["standalone"][m][k])
            for k in METRICS:
                agg_oracle[k].append(res["oracle"][k])

        print(f"\n## {backbone.upper()} (n=200, 5 seeds)\n")
        print("| Method | PickScore | HPS | CLIP | Aesthetic |")
        print("|---|---|---|---|---|")
        for m in std_methods:
            row = [f"{100*np.mean(agg_standalone[m][k]):.2f}±{100*np.std(agg_standalone[m][k]):.2f}" for k in METRICS]
            print(f"| {m} | " + " | ".join(row) + " |")
        oracle_row = [f"{100*np.mean(agg_oracle[k]):.2f}±{100*np.std(agg_oracle[k]):.2f}" for k in METRICS]
        print(f"| **Oracle (Pareto-sum)** | " + " | ".join(oracle_row) + " |")

        # Pareto improvement (oracle - max standalone) per metric, mean across seeds
        print("\n  Pareto improvement (oracle - best individual specialization), mean across 5 seeds:")
        for k in METRICS:
            ind_per_seed = [max(per_seed[s]["standalone"][m][k] for m in std_methods) for s in SEEDS]
            o_per_seed = agg_oracle[k]
            gains = [o - ind for o, ind in zip(o_per_seed, ind_per_seed)]
            print(f"    {k}: +{100*np.mean(gains):.2f}pp ± {100*np.std(gains):.2f}pp (per-seed gains: " +
                  " ".join(f"{100*g:+.2f}" for g in gains) + ")")

        out[backbone] = {
            "n": per_seed[SEEDS[0]]["n"], "seeds": SEEDS,
            "standalone_per_seed": {m: {k: agg_standalone[m][k] for k in METRICS} for m in std_methods},
            "oracle_per_seed": agg_oracle,
            "standalone_mean": {m: {k: float(np.mean(agg_standalone[m][k])) for k in METRICS} for m in std_methods},
            "standalone_std":  {m: {k: float(np.std(agg_standalone[m][k]))  for k in METRICS} for m in std_methods},
            "oracle_mean": {k: float(np.mean(agg_oracle[k])) for k in METRICS},
            "oracle_std":  {k: float(np.std(agg_oracle[k]))  for k in METRICS},
        }
    with open("crr_multiseed_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[DONE] saved to crr_multiseed_results.json")


if __name__ == "__main__":
    main()
