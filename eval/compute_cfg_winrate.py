#!/usr/bin/env python3
"""
Compute win rate and statistics for Tuned-CFG vs SDXL baseline.

Tuned-CFG scores are in eval_results/cfg_sweep/sdxl/test/cfg_7.5/results.json
with per_prompt arrays ordered by test_prompts from load_parti_prompts_split(123).

Baseline per-prompt scores are recovered from full_sdxl_best/pgmap/scores.jsonl
(the 'ref' field, which is the baseline score for each prompt in the full 1632 eval).
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).parent
SCORES_JSONL = ROOT / "eval_results/full_sdxl_best/pgmap/scores.jsonl"
CFG_RESULTS  = ROOT / "eval_results/cfg_sweep/sdxl/test/cfg_7.5/results.json"
METRICS = ["pickscore", "hps", "clip", "aes"]


def load_baseline_by_prompt():
    """Build prompt -> {metric: score} from existing scores.jsonl ref field."""
    mapping = {}
    with open(SCORES_JSONL) as f:
        for line in f:
            row = json.loads(line)
            prompt = row["prompt"]
            mapping[prompt] = {m: row["metrics"][m]["ref"] for m in METRICS}
    return mapping


def load_test_prompts():
    """Reproduce the test split exactly as benchmark_cfg.py does."""
    import numpy as np
    from datasets import load_dataset

    PARTI_SEED = 123
    VAL_FRAC   = 0.30

    ds = load_dataset("nateraw/parti-prompts", split="train")
    key = "Prompt" if "Prompt" in ds.features else list(ds.features.keys())[0]
    all_prompts = [" ".join(x.replace("<|endoftext|>", " ").split()).strip()
                   for x in ds[key] if isinstance(x, str) and x.strip()]

    rng = np.random.default_rng(PARTI_SEED)
    idx = rng.permutation(len(all_prompts))
    n_val = int(len(all_prompts) * VAL_FRAC)
    test_idx = idx[n_val:]
    return [all_prompts[i] for i in test_idx]


def wilcoxon_pvalue(deltas):
    """One-sided Wilcoxon signed-rank test: H1 = CFG > baseline."""
    stat, p_two = stats.wilcoxon(deltas, alternative="greater")
    return float(p_two)


def bootstrap_ci(deltas, n_boot=2000, alpha=0.05):
    rng = np.random.default_rng(0)
    means = [rng.choice(deltas, size=len(deltas), replace=True).mean()
             for _ in range(n_boot)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def main():
    print("Loading test prompts...")
    test_prompts = load_test_prompts()
    print(f"  {len(test_prompts)} test prompts")

    print("Loading baseline scores...")
    baseline_map = load_baseline_by_prompt()

    print("Loading Tuned-CFG per-prompt scores...")
    cfg_data = json.load(open(CFG_RESULTS))
    cfg_pp   = cfg_data["per_prompt"]
    assert all(len(cfg_pp[m]) == len(test_prompts) for m in METRICS), \
        f"Length mismatch: cfg has {len(cfg_pp['pickscore'])}, test_prompts has {len(test_prompts)}"

    # Align: for each test prompt, look up baseline score
    results = {}
    missing = 0
    for metric in METRICS:
        baseline_scores = []
        cfg_scores      = []
        for i, prompt in enumerate(test_prompts):
            if prompt not in baseline_map:
                missing += 1
                continue
            baseline_scores.append(baseline_map[prompt][metric])
            cfg_scores.append(cfg_pp[metric][i])

        baseline_arr = np.array(baseline_scores)
        cfg_arr      = np.array(cfg_scores)
        deltas       = cfg_arr - baseline_arr
        win_rate     = (deltas > 0).mean()
        mean_delta   = deltas.mean()
        p_val        = wilcoxon_pvalue(deltas)
        ci_lo, ci_hi = bootstrap_ci(deltas)

        results[metric] = {
            "n":           len(baseline_scores),
            "baseline_mean": float(baseline_arr.mean()),
            "cfg_mean":    float(cfg_arr.mean()),
            "mean_delta":  float(mean_delta),
            "win_rate":    float(win_rate),
            "wilcoxon_p":  p_val,
            "ci_95":       [ci_lo, ci_hi],
        }

    if missing:
        print(f"  Warning: {missing} prompt-metric pairs not found in baseline (likely minor prompt cleaning diffs)")

    print("\n" + "="*65)
    print(f"Tuned-CFG (w=7.5) vs SDXL Baseline (w=5.0)  [n={results['pickscore']['n']}]")
    print("="*65)
    print(f"{'Metric':<12} {'Baseline':>10} {'CFG w=7.5':>10} {'Δ':>8} {'WinRate':>8} {'p-val':>8}")
    print("-"*65)
    for m in METRICS:
        r = results[m]
        sig = "*" if r["wilcoxon_p"] < 0.05 else " "
        print(f"{m:<12} {r['baseline_mean']:>10.4f} {r['cfg_mean']:>10.4f} "
              f"{r['mean_delta']:>+8.4f} {r['win_rate']:>7.1%}{sig} {r['wilcoxon_p']:>8.4f}")
    print("  * p < 0.05 (one-sided Wilcoxon signed-rank)")

    out_path = ROOT / "eval_results/cfg_sweep/sdxl/test/cfg_7.5/winrate_vs_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
