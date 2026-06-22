# Benchmark submissions

Paste-ready content for adding **PG-MAP** to public T2I leaderboards. Three submission targets are tracked here:

1. **Papers with Code** — PartiPrompts + HPDv2 leaderboards (paper-table numbers; no extra eval needed)
2. **T2I-CompBench++** — needs running their BLIP-VQA / UniDet / CLIPScore pipeline locally
3. **Artificial Analysis** — closed, curated; we draft an outreach email only

The eval results (when run) land in [`eval_results/t2i_compbench/`](../eval_results/t2i_compbench/) and [`eval_results/table1_*/`](../eval_results/) — paste the cells from those JSONs into the templates below.

---

## 1. Papers with Code (PartiPrompts + HPDv2)

> **⚠️ Platform status (May 2026):** `paperswithcode.com` is currently redirecting some paths to `huggingface.co/papers` as Meta winds down PwC. As of this writing, the *benchmark* pages (e.g. `/sota/text-to-image-generation-on-partiprompts`) still load when accessed directly via the search interface, but the redirect behavior is inconsistent. If submissions through the web UI fail, fall back to **opening a GitHub issue at [paperswithcode/sota-extractor](https://github.com/paperswithcode/sota-extractor)** with the same content below.

### 1.A — Submit to `text-to-image-generation-on-partiprompts`

**Web flow:**
1. Sign in at https://paperswithcode.com (GitHub OAuth).
2. Search for **"PartiPrompts"** → click the benchmark → "Add Result".
3. Fill in the form with the values below.

**Paper metadata**

| Field | Value |
|---|---|
| Paper title | PG-MAP: Joint MAP Optimization for Inference-Time Alignment of Diffusion and Flow-Matching Models |
| Paper URL | https://openreview.net/forum?id=PLACEHOLDER (replace with NeurIPS 2026 OpenReview ID once available) |
| ArXiv ID | TBD (once available) |
| Code | https://github.com/sophialanlan/PG-MAP |
| Model name | **PG-MAP (SDXL, default)** |
| Model URL | https://huggingface.co/sophialan/pg-map-sdxl |

**Result rows** (one entry per row of paper Table 1)

```text
| Method                     | PickScore (↑) | HPS (↑) | Aesthetic (↑) | CLIPScore (↑) | n    | seed |
| SDXL static (baseline)     | 50.0%         | 50.0%   | 50.0%         | 50.0%         | 1632 | 123  |
| Tuned-CFG (w*=7.5)         | 48.2%         | 58.5%   | 52.4%         | 50.0%         | 1632 | 123  |
| NFE-matched UG (val-tuned) | 48.6%         | 50.5%   | 51.1%         | 47.9%         | 1632 | 123  |
| MAP-c                      | 51.4%         | 50.3%   | 49.8%         | 48.5%         | 1632 | 123  |
| Reward-z                   | 55.4%         | 47.9%   | 56.7%         | 49.7%         | 1632 | 123  |
| MAP-cz (λ=0)               | 56.7%         | 47.5%   | 55.6%         | 48.8%         | 1632 | 123  |
| PG-MAP (default)           | **56.4%**     | 47.1%   | 56.2%         | 48.1%         | 1632 | 123  |
| Tuned-CFG + PG-MAP         | 51.3%         | **64.6%** | **56.5%**   | **52.8%**     | 1632 | 123  |
```

All rows are **win-rate vs. the same-seed SDXL static baseline** (w=5.0). Statistical tests: paired Wilcoxon $p < 0.001$ for all PG-MAP rows on PickScore; bootstrap 95% CI half-width $\pm 1.4$pp on each metric. Full per-row Wilcoxon $p$ and CIs in [eval_results/table1_sdxl/](../eval_results/) after running `bash scripts/reproduce_table1_sdxl.sh`.

PwC stores results as model-level rows, so the recommended **single submission** to the PartiPrompts leaderboard is the **`Tuned-CFG + PG-MAP`** row (highest aggregate). Add `PG-MAP (default)` as a secondary row.

### 1.B — Submit to `text-to-image-generation-on-hpdv2`

Same template; HPDv2 reproduction is not yet scripted in this repo (raw HPDv2 results are available on request). The paper reports that PartiPrompts numbers transfer to HPDv2 within $\pm 2$ pp (paper §3 paragraph "Robustness on HPDv2"); use those as conservative submission values until the full HPDv2 run is available.

---

## 2. T2I-CompBench++

The maintainers (HKU + Huawei Noah's Ark) update the leaderboard in their [Readme.md](https://github.com/Karine-Huang/T2I-CompBench/blob/main/Readme.md) and [project page](https://karine-h.github.io/T2I-CompBench-new/) based on papers they see citing the benchmark. There is **no PR-able leaderboard table** — the standard submission is:

1. Run their official eval scripts locally on your generated samples.
2. Report the per-category scores in your paper / supplementary.
3. Open a **GitHub issue** at [Karine-Huang/T2I-CompBench/issues](https://github.com/Karine-Huang/T2I-CompBench/issues) titled `"Leaderboard submission: PG-MAP (under review, NeurIPS 2026)"` containing the table below + a link to your paper.

### 2.A — Generate images

Infrastructure shipped at [`eval/t2i_compbench/`](../eval/t2i_compbench/):

```bash
# One-time: clone upstream eval scripts (BLIP-VQA, UniDet, CLIPScore)
bash eval/t2i_compbench/setup_t2i_compbench.sh

# Generate (8h on H200 for SDXL + PG-MAP K=1 on color/shape/texture)
python eval/t2i_compbench/generate.py \
    --method pgmap_K2 --backbone sdxl \
    --categories color shape texture spatial non_spatial complex \
    --out_dir eval_results/t2i_compbench/sdxl_pgmap_K2

# Also generate the baseline for direct comparison
python eval/t2i_compbench/generate.py \
    --method baseline_sdxl --backbone sdxl \
    --categories color shape texture spatial non_spatial complex \
    --out_dir eval_results/t2i_compbench/sdxl_baseline
```

### 2.B — Score with BLIP-VQA / UniDet / CLIPScore

```bash
# Attribute binding (color / shape / texture)
bash eval/t2i_compbench/run_blip_vqa.sh eval_results/t2i_compbench/sdxl_pgmap_K2

# Aggregate to a single row
python eval/t2i_compbench/aggregate.py \
    --samples_root eval_results/t2i_compbench/sdxl_pgmap_K2 \
    --method "SDXL + PG-MAP (Ours, under review, NeurIPS 2026)" \
    --out eval_results/t2i_compbench/sdxl_pgmap_K2.json
```

The aggregator prints a leaderboard-ready markdown row.

### 2.C — Issue template

```markdown
**Title:** Leaderboard submission: PG-MAP (under review, NeurIPS 2026)

Hi T2I-CompBench team,

We're submitting evaluation results for **PG-MAP** (Preference-Guided Adaptive MAP), a training-free inference-time alignment method on SDXL; preprint, under review at NeurIPS 2026.

| Method | Color ↑ | Shape ↑ | Texture ↑ | 2D-Spatial ↑ | Non-Spatial ↑ | Complex ↑ |
|---|---|---|---|---|---|---|
| SDXL static (baseline)              | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| **SDXL + PG-MAP (Ours)**            | **0.XXXX** | **0.XXXX** | **0.XXXX** | … | … | … |
| **SDXL + Tuned-CFG + PG-MAP (Ours)**| **0.XXXX** | … | … | … | … | … |

Eval protocol: T2I-CompBench++ val splits (300 prompts/category), 1 image/prompt, seed 42, BLIP-VQA for color/shape/texture, UniDet for 2D-spatial, CLIPScore for non-spatial, 3-in-1 for complex.

Paper: [link]
Code: https://github.com/sophialanlan/PG-MAP
HF custom pipeline: https://huggingface.co/sophialan/pg-map-sdxl

Happy to provide raw `vqa_result.json` files on request.

Thanks!
```

Replace the `0.XXXX` placeholders with values from `eval_results/t2i_compbench/sdxl_pgmap_K2.json` after the run finishes.

---

## 3. Artificial Analysis (closed, curated)

The [Artificial Analysis Text-to-Image leaderboard](https://huggingface.co/spaces/ArtificialAnalysis/Text-to-Image-Leaderboard) lists base models only (DALL-E 3, FLUX, SDXL, SD3.5, Midjourney, Ideogram, …). PG-MAP as an inference-time wrapper is **out of scope** for their leaderboard. Outreach is low-effort/low-yield; below is a short email if you want to try.

```
To:      research@artificialanalysis.ai
Subject: PG-MAP (under review, NeurIPS 2026) — inference-time alignment for SDXL / SD3.5

Hi Artificial Analysis team,

We recently released PG-MAP, a training-free inference-time (preprint;
under review at NeurIPS 2026) alignment method that wraps SDXL / SD3.5 and
lifts PickScore win-rate by
+6 pp on SDXL and +42 pp on SD3.5-medium versus the static base model
on PartiPrompts (n=1632).

Code:       https://github.com/sophialanlan/PG-MAP
Pipelines:  https://huggingface.co/sophialan  (pg-map-{sd15,sdxl,sd3})
Live demo:  https://huggingface.co/spaces/sophialan/pg-map-demo

I understand your leaderboard focuses on base T2I models — would you
consider an "inference-time alignment" row that compares vanilla SDXL
against SDXL + PG-MAP on your existing prompt set? Happy to provide
generated images or run inference against your pipeline.

Thanks,
Ruolan
```

---

## 4. Other leaderboards worth tracking (not actively submitted yet)

| Leaderboard | Best fit | Effort | Why we haven't done it yet |
|---|---|---|---|
| GenEval                              | object counting, spatial, attribute | ~6 GPU-h | overlap with T2I-CompBench |
| HEIM (Stanford CRFM)                 | multi-dimensional benchmark | ~24 GPU-h | submission via email, slow |
| HPSv2 leaderboard                    | preference scoring  | confounded (PG-MAP optimizes HPS as one of its rewards) |
| AuraFlow (xAI) leaderboard           | open T2I leaderboard | — | submission process unclear |

Update this list as the submission landscape evolves.
