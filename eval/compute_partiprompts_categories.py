#!/usr/bin/env python3
"""
PartiPrompts Challenge-category win-rate breakdown for the 4 framework
specializations (MAP-c, MAP-cz, PG-MAP, Tuned-CFG+PG-MAP) on SDXL n=1632
seed 123. Uses existing scored data; no GPU needed.

Output: a LaTeX-ready table for appendix `app:crr_categories`.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from datasets import load_dataset


import os
EVAL = Path(os.environ.get("PG_MAP_EVAL_DIR",
            str(Path(__file__).resolve().parent.parent / "eval_results")))
METHODS = {
    "mapc":     EVAL / "table1_sdxl" / "mapc"     / "scores.jsonl",
    "joint_cz": EVAL / "table1_sdxl" / "joint_cz" / "scores.jsonl",
    "pgmap":    EVAL / "table1_sdxl" / "pgmap"    / "scores.jsonl",
    "tcfg_pgm": EVAL / "table1_sdxl" / "tuned_cfg_pgmap" / "pgmap" / "scores.jsonl",
}
PRETTY = {
    "mapc":     r"MAP-$c$",
    "joint_cz": r"MAP-$cz$",
    "pgmap":    r"PG-MAP",
    "tcfg_pgm": r"Tuned-CFG+PG-MAP",
}
METRICS = ["pickscore", "hps", "clip", "aes"]
CHALLENGE_GROUPS = {
    "binding": ["Properties & Positioning", "Quantity"],
    "typography": ["Writing & Symbols"],
    "scene": ["Style & Format", "Imagination", "Perspective"],
    "linguistic": ["Linguistic Structures"],
    "general": ["Basic", "Simple Detail", "Fine-grained Detail", "Complex"],
}
GROUP_ORDER = ["binding", "typography", "scene", "linguistic", "general"]
GROUP_PRETTY = {
    "binding":    "Binding (Prop+Pos / Qty)",
    "typography": "Typography (Writ+Sym)",
    "scene":      "Scene (Style / Imag / Persp)",
    "linguistic": "Linguistic structure",
    "general":    "General (Basic+Detail+Complex)",
}


def main():
    pp = load_dataset("nateraw/parti-prompts", split="train",
                      cache_dir=os.environ.get("HF_HOME"))
    prompt2chal = {r["Prompt"]: r["Challenge"] for r in pp}

    # group challenge labels into our 5 buckets
    chal2group = {}
    for g, chals in CHALLENGE_GROUPS.items():
        for c in chals:
            chal2group[c] = g

    # method -> prompt -> {metric -> delta}
    rows_by_method = {}
    for m, p in METHODS.items():
        rows = sorted([json.loads(l) for l in open(p)], key=lambda r: r["i"])
        rows_by_method[m] = rows
    n = len(rows_by_method["mapc"])
    print(f"Loaded {n} prompts × {len(METHODS)} methods")

    # for each (method, group, metric) -> win rate
    wins = defaultdict(lambda: defaultdict(lambda: {k: [] for k in METRICS}))
    group_counts = defaultdict(int)

    for m, rows in rows_by_method.items():
        for r in rows:
            chal = prompt2chal.get(r["prompt"])
            if chal is None:
                continue
            g = chal2group.get(chal)
            if g is None:
                continue
            if m == "mapc":  # count once
                group_counts[g] += 1
            for k in METRICS:
                wins[m][g][k].append(r["metrics"][k]["delta"] > 0)

    # print summary
    print("\nGroup sizes:")
    for g in GROUP_ORDER:
        print(f"  {GROUP_PRETTY[g]:32s}: {group_counts[g]:4d}")

    print("\nPer-group win rates (vs. baseline):\n")
    print(f"{'group':32s}", end="")
    for k in METRICS:
        for m in METHODS:
            print(f"  {m[:5]+'.'+k[:3]:>10s}", end="")
        print()
        print(" "*32, end="")

    # build LaTeX table
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{PartiPrompts \emph{Challenge}-category breakdown of "
                 r"win rates ($\%$) vs.\ baseline on SDXL ($n{=}1632$, seed 123). "
                 r"Categories are coarse PartiPrompts groupings: \textsc{binding} "
                 r"(\emph{Properties \& Positioning}, \emph{Quantity}), "
                 r"\textsc{typography} (\emph{Writing \& Symbols}), "
                 r"\textsc{scene} (\emph{Style \& Format}, \emph{Imagination}, "
                 r"\emph{Perspective}), \textsc{linguistic} "
                 r"(\emph{Linguistic Structures}), \textsc{general} "
                 r"(\emph{Basic}, \emph{Simple Detail}, "
                 r"\emph{Fine-grained Detail}, \emph{Complex}). \textbf{Bold} "
                 r"marks per-row best across the four specializations on each "
                 r"metric. The case-study dichotomy of "
                 r"Section~\ref{sec:cz_analysis} reproduces at population scale: "
                 r"MAP-$c$ leads on \textsc{typography} CLIPScore and is "
                 r"competitive on \textsc{binding}; Tuned-CFG+PG-MAP leads on "
                 r"\textsc{scene} HPS / Aesthetic; MAP-$cz$ / PG-MAP cluster as "
                 r"the PickScore-best default elsewhere.}")
    lines.append(r"\label{tab:partiprompts_categories}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    cols = "l|" + "cccc|" * len(METRICS)
    lines.append(rf"\begin{{tabular}}{{{cols.rstrip('|')}}}")
    lines.append(r"\toprule")
    head = ["Category (n)"]
    for k in METRICS:
        head.append(rf"\multicolumn{{4}}{{c}}{{{k.upper() if k=='hps' or k=='clip' or k=='aes' else 'PickScore'}}}")
    metric_label = {"pickscore": "PickScore", "hps": "HPS", "clip": "CLIP", "aes": "Aes"}
    head_top = ["Category (n)"]
    for k in METRICS:
        head_top.append(rf"\multicolumn{{4}}{{c}}{{{metric_label[k]}}}")
    lines.append(" & ".join(head_top) + r" \\")
    sub = [""]
    for k in METRICS:
        for m in METHODS:
            sub.append(PRETTY[m].replace("MAP-$cz$","cz").replace("MAP-$c$","c").replace("PG-MAP","pg").replace("Tuned-CFG+PG-MAP","t+pg"))
    # actually use shorter labels
    short = {"mapc":"$c$","joint_cz":"$cz$","pgmap":"pg","tcfg_pgm":"t+pg"}
    sub = [""] + [short[m] for _ in METRICS for m in METHODS]
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")

    for g in GROUP_ORDER:
        cells = [f"{GROUP_PRETTY[g]} ({group_counts[g]})"]
        for k in METRICS:
            row_winrates = {m: 100*np.mean(wins[m][g][k]) for m in METHODS}
            best_m = max(row_winrates, key=row_winrates.get)
            for m in METHODS:
                v = row_winrates[m]
                if m == best_m:
                    cells.append(rf"\textbf{{{v:.1f}}}")
                else:
                    cells.append(f"{v:.1f}")
        lines.append(" & ".join(cells) + r" \\")

    # all-prompts row
    lines.append(r"\midrule")
    all_cells = [f"\\emph{{All}} ({n})"]
    for k in METRICS:
        all_winrates = {}
        for m in METHODS:
            ws = [r["metrics"][k]["delta"] > 0 for r in rows_by_method[m]]
            all_winrates[m] = 100 * np.mean(ws)
        best_m = max(all_winrates, key=all_winrates.get)
        for m in METHODS:
            v = all_winrates[m]
            if m == best_m:
                all_cells.append(rf"\textbf{{{v:.1f}}}")
            else:
                all_cells.append(f"{v:.1f}")
    lines.append(" & ".join(all_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    print("\n=== LaTeX TABLE ===\n")
    print("\n".join(lines))

    # save also as plain JSON
    out = {"groups": {}, "all": {}}
    for g in GROUP_ORDER:
        out["groups"][g] = {m: {k: float(np.mean(wins[m][g][k])) for k in METRICS}
                            for m in METHODS}
        out["groups"][g]["_n"] = group_counts[g]
    for m in METHODS:
        out["all"][m] = {k: float(np.mean([r["metrics"][k]["delta"]>0
                                           for r in rows_by_method[m]])) for k in METRICS}
    with open("partiprompts_category_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[DONE] saved partiprompts_category_results.json")


if __name__ == "__main__":
    main()
