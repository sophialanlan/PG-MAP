#!/usr/bin/env python3
"""
Compute the CRR-MAP *deployable* classifier numbers (Eq. crr_router) and a
*linear-probe* classifier on top of the same routing pool. Fills the gap in
the augmented paper where the deployable classifier is defined but its
empirical win-rates are never reported (only the oracle upper bound is).

Pool members per backbone (matched to paper Tab. crr_results):

  SDXL:
    f_c    = eval_results/full_sdxl_best/mapc/scores.jsonl
    f_cz   = eval_results/full_sdxl_best/joint_cz/scores.jsonl
    f_tcfg = eval_results/sdxl_tcfg_pgmap/pgmap/scores.jsonl

  SD1.5:
    f_c    = eval_results/newB_main_sd15_n1632/mapc/scores.jsonl
    f_cz   = eval_results/newB_main_sd15_n1632/newB_newton/scores.jsonl
             [Note: this is what the existing CRR-MAP analysis used; it is
              the K=1 SGD PG-MAP λ=0.1 reference, equivalent in averaged-metric
              terms to MAP-cz λ=0 per the K=1 vs paper-K=2 audit.]
    f_tcfg = eval_results/sd15_tcfg_pgmap/pgmap/scores.jsonl

Output:
    crr_deployable_results.json  -- aggregated win rates for every routing rule
    Patched .tex Tab :crr_results -- new "(deployable)" / "(linear probe)" rows
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Default eval root resolves to <repo>/eval_results unless --eval_root or
# the PG_MAP_EVAL_DIR env var is set. The default pool paths match the layout
# produced by scripts/reproduce_table1_{sd15,sdxl}.sh and scripts/reproduce_table2_fm.sh.
EVAL_DEFAULT = os.environ.get(
    "PG_MAP_EVAL_DIR",
    str(Path(__file__).resolve().parent.parent / "eval_results"),
)


def _default_pools(eval_root: str) -> Dict[str, Dict[str, str]]:
    return {
        "sdxl": {
            "f_c":    f"{eval_root}/table1_sdxl/mapc/scores.jsonl",
            "f_cz":   f"{eval_root}/table1_sdxl/joint_cz/scores.jsonl",
            "f_tcfg": f"{eval_root}/table1_sdxl/tuned_cfg_pgmap/pgmap/scores.jsonl",
        },
        "sd15": {
            "f_c":    f"{eval_root}/table1_sd15/mapc/scores.jsonl",
            "f_cz":   f"{eval_root}/table1_sd15/joint_cz/scores.jsonl",
            "f_tcfg": f"{eval_root}/table1_sd15/tuned_cfg_pgmap/pgmap/scores.jsonl",
        },
    }
METRICS = ["pickscore", "hps", "clip", "aes"]

# Prototype prompts for CLIP-text routing (Eq. crr_router); copied verbatim
# from app:crr_prototypes in the augmented tex.
PROTOTYPES = {
    "bind": [
        "a red cube on a blue sphere",
        "a green apple inside a yellow basket",
        "a small blue car next to a large white truck",
        "a glass of orange juice with red straws",
        "the word HELLO in big block letters",
        "a stop sign next to a yield sign",
        "two cats and three dogs",
        "a yellow umbrella next to a blue umbrella",
        "a red triangle on top of a green square",
        "an apple, a banana, and a pear",
    ],
    "scene": [
        "a serene mountain landscape at golden hour",
        "an oil painting of a stormy sea with crashing waves",
        "a cyberpunk city street in the rain at night",
        "a misty forest with rays of sunlight piercing the canopy",
        "an aerial view of a coral reef in turquoise water",
        "a rolling field of lavender at sunset",
        "a cozy library with ancient books and a fireplace",
        "an art deco hotel lobby",
        "a quiet beach at dawn with seagulls",
        "a Victorian street scene at dusk",
    ],
    "bal": [
        "a person walking a dog in a park",
        "a chef cooking pasta in a kitchen",
        "a child playing with a toy on a wooden floor",
        "a cat sleeping on a couch",
        "a cup of coffee on a desk",
        "a bicycle leaning against a brick wall",
        "a horse running through a field",
        "a dog catching a frisbee",
        "a woman reading a book",
        "a butterfly on a flower",
    ],
}

# Lexical-override regex (forces routing to f_c)
TYPOGRAPHY_CUES = [
    "the word ", "sign that reads", "sign reading", "letters spelling",
    "text that says", "in big block letters", "spelling out",
    "label reading", "written in",
]


def load_scores(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def align_pools(pool_paths: Dict[str, str]) -> Dict[str, List[dict]]:
    """Load + align by prompt index 'i'. Returns {label: list-of-rows-sorted-by-i}."""
    out = {}
    for label, path in pool_paths.items():
        rows = sorted(load_scores(path), key=lambda r: r["i"])
        out[label] = rows
    # Verify matched length and matched prompts
    n = len(next(iter(out.values())))
    for label, rows in out.items():
        assert len(rows) == n, f"{label}: {len(rows)} vs {n}"
    prompts_ref = [r["prompt"] for r in next(iter(out.values()))]
    for label, rows in out.items():
        for j, r in enumerate(rows):
            assert r["prompt"] == prompts_ref[j], \
                f"prompt mismatch at i={j}: {label} '{r['prompt'][:30]}' vs '{prompts_ref[j][:30]}'"
    return out, prompts_ref


def encode_clip_text(prompts: List[str], device: str = "cuda",
                     model_id: str = "openai/clip-vit-large-patch14",
                     batch_size: int = 64) -> np.ndarray:
    """L2-normalised CLIP-text embeddings."""
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection
    tok = CLIPTokenizer.from_pretrained(model_id)
    model = CLIPTextModelWithProjection.from_pretrained(model_id).to(device).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            t = tok(batch, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            out = model(**t).text_embeds  # (B, 768)
            out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            embs.append(out.cpu().numpy())
    return np.concatenate(embs, axis=0)


def lexical_override_to_fc(prompt: str) -> bool:
    p = prompt.lower()
    if len(p.split()) <= 3:
        return True
    for cue in TYPOGRAPHY_CUES:
        if cue in p:
            return True
    return False


def deployable_route(prompts: List[str], proto_centroids: Dict[str, np.ndarray],
                     embs: np.ndarray) -> List[str]:
    """Apply Eq. crr_router with lexical overrides. Return one of {f_c, f_cz, f_tcfg}."""
    out = []
    centroids = np.stack([proto_centroids["bind"], proto_centroids["scene"], proto_centroids["bal"]], axis=0)
    cls_to_pool = {"bind": "f_c", "scene": "f_tcfg", "bal": "f_cz"}
    for i, p in enumerate(prompts):
        if lexical_override_to_fc(p):
            out.append("f_c")
            continue
        sims = centroids @ embs[i]
        cls = ["bind", "scene", "bal"][int(np.argmax(sims))]
        out.append(cls_to_pool[cls])
    return out


def aggregate_winrate(routing: List[str], pool: Dict[str, List[dict]]) -> Dict[str, dict]:
    """For each prompt, look up the dispatched method's (ref, trained) pair on each
    metric and compute win/lose."""
    n = len(routing)
    summary = {}
    for k in METRICS:
        wins = 0
        deltas = []
        ref_vals = []
        tr_vals = []
        for i in range(n):
            method = routing[i]
            row = pool[method][i]
            r = row["metrics"][k]["ref"]
            t = row["metrics"][k]["trained"]
            wins += 1 if t > r else 0
            deltas.append(t - r)
            ref_vals.append(r)
            tr_vals.append(t)
        deltas = np.array(deltas)
        summary[k] = {
            "win_rate": float(wins / n),
            "mean_delta": float(deltas.mean()),
            "mean_ref": float(np.mean(ref_vals)),
            "mean_trained": float(np.mean(tr_vals)),
            "n": int(n),
        }
    return summary


def oracle_pareto_route(pool: Dict[str, List[dict]]) -> List[str]:
    """Per-prompt argmax over Pareto-sum of within-method z-scored metrics.
    Mirrors the oracle definition in the augmented paper §crr_results."""
    n = len(next(iter(pool.values())))
    pool_list = list(pool.keys())
    # Compute within-method z-scored deltas
    deltas = {m: {k: np.array([row["metrics"][k]["delta"] for row in pool[m]]) for k in METRICS}
              for m in pool_list}
    z_scored = {}
    for m in pool_list:
        z_scored[m] = {}
        for k in METRICS:
            d = deltas[m][k]
            mu, sd = d.mean(), d.std()
            z_scored[m][k] = (d - mu) / (sd if sd > 1e-12 else 1.0)
    # Per-prompt sum
    sums = {m: np.sum([z_scored[m][k] for k in METRICS], axis=0) for m in pool_list}
    chosen = []
    for i in range(n):
        scores = {m: sums[m][i] for m in pool_list}
        chosen.append(max(scores, key=scores.get))
    return chosen


def linear_probe_route(prompts: List[str], embs: np.ndarray, pool: Dict[str, List[dict]],
                       n_folds: int = 5, seed: int = 42) -> tuple:
    """Train a 3-class linear probe with stratified k-fold cross-val. Each fold
    learns argmax of Pareto-sum oracle on the *training* prompts and predicts
    routing on the held-out prompts. Returns the cross-val routing assignment
    (each prompt's prediction came from the fold where it was held out)."""
    n = len(prompts)
    oracle = oracle_pareto_route(pool)
    label_map = {"f_c": 0, "f_cz": 1, "f_tcfg": 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    y = np.array([label_map[lbl] for lbl in oracle])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_size = n // n_folds
    pred = np.zeros(n, dtype=int)
    for f in range(n_folds):
        test_idx = perm[f*fold_size : (f+1)*fold_size if f < n_folds-1 else n]
        test_set = set(test_idx.tolist())
        train_idx = np.array([i for i in perm if i not in test_set])
        # Fit linear probe via torch (multinomial logistic regression, L2 reg.).
        x_tr = torch.tensor(embs[train_idx], dtype=torch.float32, device="cuda")
        y_tr = torch.tensor(y[train_idx], dtype=torch.long, device="cuda")
        x_te = torch.tensor(embs[test_idx], dtype=torch.float32, device="cuda")
        d = embs.shape[1]
        # Initialize logits = 0; standardize input
        with torch.no_grad():
            mu = x_tr.mean(0, keepdim=True); std = x_tr.std(0, keepdim=True).clamp(min=1e-6)
        x_tr_n = (x_tr - mu) / std
        x_te_n = (x_te - mu) / std
        W = torch.zeros(d, 3, device="cuda", requires_grad=True)
        b = torch.zeros(3, device="cuda", requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=200, tolerance_grad=1e-7)
        def closure():
            opt.zero_grad()
            logits = x_tr_n @ W + b
            loss = torch.nn.functional.cross_entropy(logits, y_tr)
            # L2 regularization on W (C=1.0 equivalent: 1/n * sum-loss + 0.5 * ||W||^2)
            loss = loss + 0.5 * 1.0/len(y_tr) * (W*W).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            logits_te = x_te_n @ W + b
            pred_fold = logits_te.argmax(dim=-1).cpu().numpy()
        pred[test_idx] = pred_fold
    routing = [inv_label_map[int(p)] for p in pred]
    accuracy = float((pred == y).mean())
    return routing, accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="crr_deployable_results.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_probe", action="store_true")
    ap.add_argument("--eval_root", default=EVAL_DEFAULT,
                    help="Path to the eval_results directory (default: "
                         "<repo>/eval_results, override with PG_MAP_EVAL_DIR env var).")
    ap.add_argument("--backbones", nargs="+", default=["sdxl", "sd15"],
                    choices=["sdxl", "sd15"],
                    help="Which backbones to run the routing analysis on.")
    args = ap.parse_args()

    pools = _default_pools(args.eval_root)
    print("=" * 60)
    print("CRR-MAP deployable classifier evaluation")
    print(f"  eval_root = {args.eval_root}")
    print("=" * 60)
    results = {}

    for backbone in args.backbones:
        print(f"\n=== Backbone: {backbone.upper()} ===")
        pool, prompts = align_pools(pools[backbone])
        print(f"[{backbone}] aligned {len(prompts)} prompts across {list(pool.keys())}")

        # Per-method win rates (sanity vs paper Tab 1)
        per_method = {}
        for m in pool:
            per_method[m] = aggregate_winrate([m]*len(prompts), pool)
        print(f"[{backbone}] per-method win rates:")
        for m in pool:
            print(f"  {m:>7}: " + ", ".join(f"{k}={100*per_method[m][k]['win_rate']:.2f}" for k in METRICS))

        # Embed prompts with CLIP-L
        print(f"[{backbone}] embedding {len(prompts)} prompts with CLIP-L...")
        embs = encode_clip_text(prompts, device=args.device)

        # Compute prototype centroids
        proto_centroids = {}
        for cls, examples in PROTOTYPES.items():
            proto_embs = encode_clip_text(examples, device=args.device)
            c = proto_embs.mean(axis=0)
            c /= np.linalg.norm(c) + 1e-12
            proto_centroids[cls] = c

        # Deployable router
        deployable_routing = deployable_route(prompts, proto_centroids, embs)
        deploy_summary = aggregate_winrate(deployable_routing, pool)
        from collections import Counter
        deploy_dist = Counter(deployable_routing)
        print(f"[{backbone}] DEPLOYABLE (CLIP-prototype + lexical overrides):")
        print(f"  routing distribution: {dict(deploy_dist)}")
        for k in METRICS:
            print(f"  {k}: win_rate={100*deploy_summary[k]['win_rate']:.2f}%, "
                  f"mean Δ={deploy_summary[k]['mean_delta']:+.5f}")

        # Oracle Pareto-sum (sanity vs the existing paper number)
        oracle_routing = oracle_pareto_route(pool)
        oracle_summary = aggregate_winrate(oracle_routing, pool)
        oracle_dist = Counter(oracle_routing)
        print(f"[{backbone}] ORACLE (Pareto-sum):")
        print(f"  routing distribution: {dict(oracle_dist)}")
        for k in METRICS:
            print(f"  {k}: win_rate={100*oracle_summary[k]['win_rate']:.2f}%, "
                  f"mean Δ={oracle_summary[k]['mean_delta']:+.5f}")

        # Linear probe
        probe_summary = None
        probe_acc = None
        if not args.skip_probe:
            print(f"[{backbone}] training 3-class linear probe (5-fold CV)...")
            probe_routing, probe_acc = linear_probe_route(prompts, embs, pool, n_folds=5)
            probe_summary = aggregate_winrate(probe_routing, pool)
            probe_dist = Counter(probe_routing)
            print(f"[{backbone}] LINEAR PROBE (CLIP-text → 3-class):")
            print(f"  CV accuracy vs oracle Pareto-sum label: {probe_acc:.3f}")
            print(f"  routing distribution: {dict(probe_dist)}")
            for k in METRICS:
                print(f"  {k}: win_rate={100*probe_summary[k]['win_rate']:.2f}%, "
                      f"mean Δ={probe_summary[k]['mean_delta']:+.5f}")

        # How much of oracle gap is closed?
        gaps = {}
        for k in METRICS:
            ind_max = max(per_method[m][k]['win_rate'] for m in pool)
            o = oracle_summary[k]['win_rate']
            d = deploy_summary[k]['win_rate']
            gap_total = o - ind_max
            gap_closed_deploy = (d - ind_max) / gap_total if gap_total > 1e-9 else float("nan")
            gap_closed_probe = (probe_summary[k]['win_rate'] - ind_max) / gap_total if (gap_total > 1e-9 and probe_summary) else float("nan")
            gaps[k] = {
                "best_individual": ind_max,
                "oracle": o,
                "deployable": d,
                "linear_probe": probe_summary[k]['win_rate'] if probe_summary else None,
                "gap_total": gap_total,
                "gap_closed_deployable_pct": 100 * gap_closed_deploy if gap_total > 1e-9 else None,
                "gap_closed_probe_pct": 100 * gap_closed_probe if (gap_total > 1e-9 and probe_summary) else None,
            }

        results[backbone] = {
            "n": len(prompts),
            "per_method_winrate": per_method,
            "deployable": {
                "summary": deploy_summary,
                "routing_distribution": dict(deploy_dist),
            },
            "oracle_pareto": {
                "summary": oracle_summary,
                "routing_distribution": dict(oracle_dist),
            },
            "linear_probe": ({
                "summary": probe_summary,
                "cv_accuracy": probe_acc,
                "routing_distribution": dict(probe_dist),
            } if probe_summary else None),
            "gap_closure": gaps,
        }

        # Free models
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] saved to {args.out}")

    # Markdown table summary
    print("\n" + "=" * 60)
    print("PAPER-READY SUMMARY")
    print("=" * 60)
    for backbone in ["sdxl", "sd15"]:
        r = results[backbone]
        print(f"\n## {backbone.upper()} (n={r['n']})\n")
        print("| Method | PickScore | HPS | CLIP | Aesthetic |")
        print("|---|---|---|---|---|")
        # individual specializations
        for m in ["f_c", "f_cz", "f_tcfg"]:
            row = r["per_method_winrate"][m]
            print(f"| {m} | "
                  + " | ".join(f"{100*row[k]['win_rate']:.2f}" for k in METRICS) + " |")
        # deployable
        s = r["deployable"]["summary"]
        print(f"| **CRR-MAP (deployable)** | "
              + " | ".join(f"{100*s[k]['win_rate']:.2f}" for k in METRICS) + " |")
        # linear probe
        if r["linear_probe"]:
            s = r["linear_probe"]["summary"]
            print(f"| **CRR-MAP (linear probe)** | "
                  + " | ".join(f"{100*s[k]['win_rate']:.2f}" for k in METRICS) + " |")
        # oracle
        s = r["oracle_pareto"]["summary"]
        print(f"| CRR-MAP (oracle, Pareto-sum) | "
              + " | ".join(f"{100*s[k]['win_rate']:.2f}" for k in METRICS) + " |")
        # Gap closure
        print("\nGap closure (% of oracle - best-individual gap recovered by deployable / linear-probe):")
        for k in METRICS:
            g = r["gap_closure"][k]
            d_pct = f"{g['gap_closed_deployable_pct']:.1f}%" if g['gap_closed_deployable_pct'] is not None else "N/A"
            p_pct = f"{g['gap_closed_probe_pct']:.1f}%" if g['gap_closed_probe_pct'] is not None else "N/A"
            print(f"  {k}: deployable={d_pct}, linear_probe={p_pct}, gap_total={100*g['gap_total']:.1f}pp")


if __name__ == "__main__":
    main()
