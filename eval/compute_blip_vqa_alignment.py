#!/usr/bin/env python3
"""
T2I-CompBench-style alignment score: BLIP-VQA "is this image accurately
described by '{prompt}'?" yes/no probability, on the existing PartiPrompts
SDXL n=1632 images for the 5 main methods.

This adds a 5th, alignment-focused metric beyond PickScore / HPS / CLIP / Aes,
addressing paper Limitation L3 ("CLIPScore at chance for reward-driven variants;
text-faithfulness scorers like ImageReward / T2I-CompBench are better suited").
We use BLIP-VQA as a lightweight T2I-CompBench-style alignment scorer.

Methods scored (using existing scored sets' images directories):
  baseline  : eval_results/full_sdxl_best/baseline/images
  mapc      : eval_results/full_sdxl_best/mapc/images
  joint_cz  : eval_results/full_sdxl_best/joint_cz/images
  pgmap     : eval_results/full_sdxl_best/pgmap/images   (PG-MAP λ=0.05)
  tcfg_pgm  : eval_results/sdxl_tcfg_pgmap/pgmap/images  (Tuned-CFG+PG-MAP)

Output:
  blip_vqa_alignment.jsonl  -- per-(method, prompt) row with yes-prob
  blip_vqa_summary.json     -- aggregate per-method mean + win rate vs baseline

Compute estimate: 1632 prompts × 5 methods × ~0.25s = ~35 min on Blackwell.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

EVAL = Path(os.environ.get("PG_MAP_EVAL_DIR",
            str(Path(__file__).resolve().parent.parent / "eval_results")))
METHODS = {
    "baseline":  EVAL / "table1_sdxl" / "baseline" / "images",
    "mapc":      EVAL / "table1_sdxl" / "mapc"     / "images",
    "joint_cz":  EVAL / "table1_sdxl" / "joint_cz" / "images",
    "pgmap":     EVAL / "table1_sdxl" / "pgmap"    / "images",
    "tcfg_pgm":  EVAL / "table1_sdxl" / "tuned_cfg_pgmap" / "pgmap" / "images",
}
PROMPTS_FILE = EVAL / "full_sdxl_best" / "mapc" / "scores.jsonl"  # has prompts in order


def load_prompts():
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f: prompts.append(json.loads(line.strip())["prompt"])
    return prompts


def load_blip_vqa(device="cuda", dtype=torch.float16):
    from transformers import BlipForQuestionAnswering, BlipProcessor
    model_id = "Salesforce/blip-vqa-capfilt-large"
    print(f"Loading {model_id}...")
    proc = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return model, proc


@torch.no_grad()
def score_image(model, proc, image: Image.Image, prompt: str, device: str) -> float:
    """Returns P('yes') for 'Is this image showing: {prompt}?'"""
    q = f"Is this image showing: {prompt}?"
    inputs = proc(images=image, text=q, return_tensors="pt").to(device, dtype=next(model.parameters()).dtype)
    # Generate answer; check if "yes" or "no"
    generated = model.generate(**inputs, max_length=10, num_beams=1, do_sample=False)
    answer = proc.decode(generated[0], skip_special_tokens=True).strip().lower()
    if "yes" in answer: return 1.0
    if "no" in answer:  return 0.0
    return 0.5  # ambiguous


@torch.no_grad()
def score_image_logits(model, proc, image: Image.Image, prompt: str, device: str,
                        yes_id: int, no_id: int) -> float:
    """Returns sigmoid-normalised yes-vs-no probability (continuous)."""
    q = f"Is this image accurately described by: {prompt}?"
    inputs = proc(images=image, text=q, return_tensors="pt").to(device, dtype=next(model.parameters()).dtype)
    # Force decode yes/no first token; return P(yes) / (P(yes)+P(no))
    decoder_start = torch.tensor([[model.config.text_config.bos_token_id or 30522]], device=device)
    out = model(**inputs, decoder_input_ids=decoder_start)
    logits = out.logits[0, -1]  # last token logits (V,)
    p_yes = torch.softmax(logits[[yes_id, no_id]].float(), dim=-1)[0].item()
    return p_yes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", default="blip_vqa_alignment.jsonl")
    ap.add_argument("--out_summary", default="blip_vqa_summary.json")
    ap.add_argument("--max_prompts", type=int, default=-1, help="cap for testing; -1 = all")
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    prompts = load_prompts()
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]
    n = len(prompts)
    print(f"Loaded {n} prompts; methods: {args.methods}")

    # Verify image dirs
    for m in args.methods:
        d = METHODS[m]
        if not d.exists():
            print(f"[WARN] missing {d}; dropping {m}")
            args.methods.remove(m)
            continue
        cnt = len(list(d.glob("*.png")))
        print(f"  {m}: {cnt} images at {d}")

    model, proc = load_blip_vqa(device=args.device)
    yes_id = proc.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = proc.tokenizer.encode("no",  add_special_tokens=False)[0]
    print(f"yes-token id: {yes_id}, no-token id: {no_id}")

    # Open output, resume if existing
    done_keys = set()
    if os.path.exists(args.out_jsonl):
        with open(args.out_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    done_keys.add((r["method"], r["i"]))
                except: pass
        print(f"Resuming: {len(done_keys)} rows already in {args.out_jsonl}")

    fout = open(args.out_jsonl, "a")
    total = n * len(args.methods)
    pbar = tqdm(total=total, initial=len(done_keys))

    summary = {m: [] for m in args.methods}

    for m in args.methods:
        d = METHODS[m]
        for i, prompt in enumerate(prompts):
            if (m, i) in done_keys:
                pbar.update(1)
                continue
            img_path = d / f"{i:05d}.png"
            if not img_path.exists():
                pbar.update(1); continue
            img = Image.open(img_path).convert("RGB")
            # Resize to 384x384 for BLIP
            img_proc = img.resize((384, 384), Image.LANCZOS)
            try:
                p_yes = score_image(model, proc, img_proc, prompt, args.device)
            except Exception as e:
                p_yes = float("nan")
            row = {"method": m, "i": i, "prompt": prompt, "blip_vqa_yes": p_yes}
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            summary[m].append(p_yes)
            pbar.update(1)
    fout.close()
    pbar.close()

    # Aggregate: re-read from disk (handles resume case where summary lists are empty)
    by_method = {m: {} for m in args.methods}
    with open(args.out_jsonl) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if r["method"] in by_method:
                    by_method[r["method"]][r["i"]] = r["blip_vqa_yes"]
            except: pass
    agg = {}
    base_scores = by_method.get("baseline", {})
    for m in args.methods:
        s = np.array([v for v in by_method[m].values() if not (v is None or np.isnan(v))])
        agg[m] = {"mean": float(s.mean()) if len(s) > 0 else None, "n": int(len(s))}
        if base_scores and m != "baseline":
            wins = 0; tot = 0
            for i, mv in by_method[m].items():
                bv = base_scores.get(i)
                if bv is None or mv is None or np.isnan(mv) or np.isnan(bv): continue
                wins += 1 if mv > bv else 0; tot += 1
            agg[m]["win_rate_vs_baseline"] = (wins / max(tot, 1)) if tot > 0 else None
            agg[m]["n_paired"] = tot

    with open(args.out_summary, "w") as f:
        json.dump(agg, f, indent=2)

    print("\n=== BLIP-VQA alignment summary ===")
    for m in args.methods:
        a = agg[m]
        mean_str = f"{a['mean']:.4f}" if a['mean'] is not None else "N/A"
        print(f"  {m:>10}: mean P(yes)={mean_str} (n={a['n']})", end="")
        if a.get("win_rate_vs_baseline") is not None:
            print(f", win-rate vs baseline = {100*a['win_rate_vs_baseline']:.2f}% (n_paired={a['n_paired']})")
        else:
            print()


if __name__ == "__main__":
    main()
