#!/usr/bin/env python3
"""
Compute win rate for UG vs SDXL baseline.

Scores UG test images (eval_results/ug/sdxl/test/eta_1e-01/images/)
and compares against baseline ref scores from full_sdxl_best/pgmap/scores.jsonl
matched by prompt text.

Note: seeds differ (UG uses 0/1/2 cycle; baseline uses 123+i), so this is
a cross-seed comparison. Win rates remain unbiased estimators but have slightly
higher variance than a perfectly paired test.

Output:
  eval_results/ug_full/sdxl/winrate.json       — full stats per metric
  eval_results/ug_full/sdxl/scores_summary.json — updated with win_rate fields
"""
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats
from tqdm.auto import tqdm

_default_cache = os.path.expanduser("~/.cache/pgmap")
os.environ.setdefault("HF_HOME",            f"{_default_cache}/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE",  f"{_default_cache}/hf_home/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_default_cache}/hf_home/transformers")
os.environ.setdefault("HPS_CACHE",          f"{_default_cache}/hps_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT        = Path(__file__).parent
UG_IMG_DIR  = ROOT / "eval_results/ug/sdxl/test/eta_1e-01/images"
SCORES_JSONL = ROOT / "eval_results/full_sdxl_best/pgmap/scores.jsonl"
OUT_DIR     = ROOT / "eval_results/ug_full/sdxl"
METRICS     = ["pickscore", "hps", "clip", "aes"]
PARTI_SEED  = 123
VAL_FRAC    = 0.30


# ── prompt loading ────────────────────────────────────────────────────────────

def load_test_prompts():
    from datasets import load_dataset
    ds = load_dataset("nateraw/parti-prompts", split="train")
    key = "Prompt" if "Prompt" in ds.features else list(ds.features.keys())[0]
    all_prompts = [" ".join(x.replace("<|endoftext|>", " ").split()).strip()
                   for x in ds[key] if isinstance(x, str) and x.strip()]
    rng = np.random.default_rng(PARTI_SEED)
    idx = rng.permutation(len(all_prompts))
    n_val = int(len(all_prompts) * VAL_FRAC)
    return [all_prompts[i] for i in idx[n_val:]]


# ── baseline loading ──────────────────────────────────────────────────────────

def load_baseline_by_prompt():
    mapping = {}
    with open(SCORES_JSONL) as f:
        for line in f:
            row = json.loads(line)
            mapping[row["prompt"]] = {m: row["metrics"][m]["ref"] for m in METRICS}
    return mapping


# ── UG image scoring ──────────────────────────────────────────────────────────

def load_scorers(device):
    repo = str(ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    scorers = {}
    print("  Loading pickscore...")
    from utils.pickscore_utils import Selector as PS
    scorers["pickscore"] = PS(device)
    print("  Loading hps...")
    from utils.hps_utils import Selector as HPS
    scorers["hps"] = HPS(device)
    print("  Loading clip...")
    from utils.clip_utils import Selector as CLIP
    scorers["clip"] = CLIP(device)
    print("  Loading aes...")
    from utils.aes_utils import Selector as AES
    scorers["aes"] = AES(device)
    return scorers


def score_ug_images(prompts, scorers):
    """Score each UG image; return {metric: [score_0, score_1, ...]}."""
    raw = {m: [] for m in METRICS}
    for i, prompt in enumerate(tqdm(prompts, desc="Scoring UG")):
        img_path = UG_IMG_DIR / f"{i:05d}.png"
        if not img_path.exists():
            for m in METRICS:
                raw[m].append(float("nan"))
            continue
        img = Image.open(img_path).convert("RGB")
        for m, sel in scorers.items():
            raw[m].append(float(sel.score([img], prompt)[0]))
    return raw


# ── statistics ────────────────────────────────────────────────────────────────

def wilcoxon_p(deltas):
    try:
        _, p = stats.wilcoxon(deltas, alternative="greater")
        return float(p)
    except Exception:
        return float("nan")


def bootstrap_ci(deltas, n_boot=2000, alpha=0.05):
    rng = np.random.default_rng(0)
    means = [rng.choice(deltas, size=len(deltas), replace=True).mean()
             for _ in range(n_boot)]
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading test prompts...")
    test_prompts = load_test_prompts()
    print(f"  {len(test_prompts)} test prompts")

    print("Loading baseline scores from scores.jsonl...")
    baseline_map = load_baseline_by_prompt()

    print("Loading scorers...")
    scorers = load_scorers(device)

    print("Scoring UG images...")
    ug_raw = score_ug_images(test_prompts, scorers)

    del scorers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── per-metric stats ───────────────────────────────────────────────────
    results = {}
    missing = 0
    for metric in METRICS:
        baseline_scores, ug_scores = [], []
        for i, prompt in enumerate(test_prompts):
            ug_val = ug_raw[metric][i]
            if np.isnan(ug_val) or prompt not in baseline_map:
                missing += 1
                continue
            baseline_scores.append(baseline_map[prompt][metric])
            ug_scores.append(ug_val)

        b = np.array(baseline_scores)
        u = np.array(ug_scores)
        d = u - b
        win_rate = float((d > 0).mean())
        ci_lo, ci_hi = bootstrap_ci(d)

        results[metric] = {
            "n":             len(baseline_scores),
            "baseline_mean": float(b.mean()),
            "ug_mean":       float(u.mean()),
            "mean_delta":    float(d.mean()),
            "win_rate":      win_rate,
            "wilcoxon_p":    wilcoxon_p(d),
            "ci_95":         [ci_lo, ci_hi],
        }

    if missing:
        print(f"  Warning: {missing} skipped (missing image or prompt mismatch)")

    # ── print table ────────────────────────────────────────────────────────
    n = results["pickscore"]["n"]
    print(f"\n{'='*65}")
    print(f"UG (η★=0.1, K_ug=4) vs SDXL Baseline  [n={n}]")
    print("="*65)
    print(f"{'Metric':<12} {'Baseline':>10} {'UG':>10} {'Δ':>8} {'WinRate':>8} {'p-val':>8}")
    print("-"*65)
    for m in METRICS:
        r = results[m]
        sig = "*" if r["wilcoxon_p"] < 0.05 else " "
        print(f"{m:<12} {r['baseline_mean']:>10.4f} {r['ug_mean']:>10.4f} "
              f"{r['mean_delta']:>+8.4f} {r['win_rate']:>7.1%}{sig} {r['wilcoxon_p']:>8.4f}")
    print("  * p < 0.05 (one-sided Wilcoxon signed-rank)")
    print(f"\n95% CI (PickScore win rate): [{results['pickscore']['ci_95'][0]:.1%}, {results['pickscore']['ci_95'][1]:.1%}]")

    # ── save outputs ───────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    winrate_path = OUT_DIR / "winrate.json"
    with open(winrate_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {winrate_path}")

    # Update scores_summary.json with win_rate and mean_delta
    summary_path = OUT_DIR / "scores_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        for m in METRICS:
            if m in summary and m in results:
                summary[m]["win_rate"]   = results[m]["win_rate"]
                summary[m]["mean_delta"] = results[m]["mean_delta"]
                summary[m]["mean_ref"]   = results[m]["baseline_mean"]
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Updated → {summary_path}")


if __name__ == "__main__":
    main()
