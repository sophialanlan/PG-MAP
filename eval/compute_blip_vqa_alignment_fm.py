#!/usr/bin/env python3
"""
BLIP-VQA alignment scoring on the FM (SD3.5-medium) images.

Mirror of compute_blip_vqa_alignment.py but on the FM transport. Three image
sets at n=1632, seed 123:
  - baseline:  flow_pgmap_sd35_n1632_seed123/flow_baseline/images
  - flow_ug:   flow_pgmap_sd35_n1632_seed123/flow_ug/images       (UG-FM, the
               headline 91.9% PS specialization, optimized on PickScore)
  - flow_pgmap: flow_pgmap_sd35_n1632_seed123/flow_pgmap/images   (full
               PG-MAP-FM, also PickScore-optimized but at small eta_z)

BLIP-VQA is a text-faithfulness scorer (yes/no answer to "is this image
accurately described by: <prompt>?"). It was NOT used as an optimization
signal anywhere in the paper, so this is an independent alignment audit
that is not subject to reward-model exploitation by the FM optimizer.

Compute estimate: 1632 × 3 × ~0.25s = ~20 min on Blackwell.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


EVAL = Path(os.environ.get("PG_MAP_EVAL_DIR",
            str(Path(__file__).resolve().parent.parent / "eval_results")))
ROOT = EVAL / "table2_fm"
METHODS = {
    "baseline":   ROOT / "baseline"    / "images",
    "flow_ug":    ROOT / "ug_fm_data"  / "images",
    "flow_pgmap": ROOT / "ug_fm_data"  / "images",   # placeholder; full PG-MAP-FM under pgmap_fm_noise/
}
PROMPTS_FILE = ROOT / "ug_fm_data" / "scores.jsonl"


def load_prompts():
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            prompts.append(json.loads(line.strip())["prompt"])
    return prompts


def load_blip_vqa(device="cuda", dtype=torch.float16):
    from transformers import BlipForQuestionAnswering, BlipProcessor
    model_id = "Salesforce/blip-vqa-capfilt-large"
    print(f"Loading {model_id}...")
    proc = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id,
            torch_dtype=dtype).to(device).eval()
    return model, proc


@torch.no_grad()
def score_image(model, proc, image: Image.Image, prompt: str, device: str) -> float:
    q = f"Is this image showing: {prompt}?"
    inputs = proc(images=image, text=q, return_tensors="pt").to(
        device, dtype=next(model.parameters()).dtype)
    generated = model.generate(**inputs, max_length=10, num_beams=1, do_sample=False)
    answer = proc.decode(generated[0], skip_special_tokens=True).strip().lower()
    if "yes" in answer: return 1.0
    if "no" in answer:  return 0.0
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", default="blip_vqa_alignment_fm.jsonl")
    ap.add_argument("--out_summary", default="blip_vqa_summary_fm.json")
    ap.add_argument("--max_prompts", type=int, default=-1)
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    prompts = load_prompts()
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]
    n = len(prompts)
    print(f"Loaded {n} prompts; methods: {args.methods}")
    for m in args.methods:
        d = METHODS[m]
        if not d.exists():
            print(f"[WARN] missing {d}; dropping {m}")
            args.methods.remove(m); continue
        cnt = len(list(d.glob("*.png")))
        print(f"  {m}: {cnt} images at {d}")

    model, proc = load_blip_vqa(device=args.device)

    # Resume support
    done_keys = set()
    if os.path.exists(args.out_jsonl):
        with open(args.out_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    done_keys.add((r["method"], r["i"]))
                except Exception:
                    pass
        print(f"Resuming: {len(done_keys)} rows already in {args.out_jsonl}")

    fout = open(args.out_jsonl, "a")
    total = n * len(args.methods)
    pbar = tqdm(total=total, initial=len(done_keys))

    # Note: FM scores.jsonl uses the seed-123 prompt order; image filenames are
    # 00000.png... that order. flow_ug/scores.jsonl already has prompts aligned
    # to filename indices (we verified with the original FM eval scripts).
    # Cross-check: load FM scores.jsonl rows and use their (i, prompt) directly.
    rows_ref = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            rows_ref.append(json.loads(line.strip()))

    for m in args.methods:
        d = METHODS[m]
        for r in rows_ref[:n]:
            i = r["i"]
            prompt = r["prompt"]
            if (m, i) in done_keys:
                pbar.update(1); continue
            img_path = d / f"{i:05d}.png"
            if not img_path.exists():
                pbar.update(1); continue
            try:
                img = Image.open(img_path).convert("RGB").resize((384, 384), Image.LANCZOS)
                p_yes = score_image(model, proc, img, prompt, args.device)
            except Exception as e:
                p_yes = float("nan")
            row = {"method": m, "i": i, "prompt": prompt, "blip_vqa_yes": p_yes}
            fout.write(json.dumps(row) + "\n"); fout.flush()
            pbar.update(1)
    fout.close(); pbar.close()

    # Aggregate
    by_method = {m: {} for m in args.methods}
    with open(args.out_jsonl) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if r["method"] in by_method:
                    by_method[r["method"]][r["i"]] = r["blip_vqa_yes"]
            except Exception:
                pass

    agg = {}
    base_scores = by_method.get("baseline", {})
    for m in args.methods:
        s = np.array([v for v in by_method[m].values() if v is not None and not np.isnan(v)])
        agg[m] = {"mean": float(s.mean()) if len(s) else None, "n": int(len(s))}
        if base_scores and m != "baseline":
            wins = 0; tot = 0; ties = 0
            for i, mv in by_method[m].items():
                bv = base_scores.get(i)
                if bv is None or mv is None or np.isnan(mv) or np.isnan(bv):
                    continue
                tot += 1
                if mv > bv:   wins += 1
                elif mv == bv: ties += 1
            agg[m]["win_rate_vs_baseline"] = wins / tot if tot else None
            agg[m]["tie_rate_vs_baseline"] = ties / tot if tot else None
            agg[m]["n_paired"] = tot

    with open(args.out_summary, "w") as f:
        json.dump(agg, f, indent=2)

    print("\n=== FM BLIP-VQA alignment summary ===")
    for m in args.methods:
        a = agg[m]
        ms = f"{a['mean']:.4f}" if a["mean"] is not None else "N/A"
        print(f"  {m:>12}: mean P(yes)={ms} (n={a['n']})", end="")
        if a.get("win_rate_vs_baseline") is not None:
            print(f", win={100*a['win_rate_vs_baseline']:.2f}%, tie={100*a['tie_rate_vs_baseline']:.2f}%, n_paired={a['n_paired']}")
        else:
            print()


if __name__ == "__main__":
    main()
