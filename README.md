# PG-MAP: Joint MAP Optimization for Inference-Time Alignment of Diffusion and Flow-Matching Models

> **Preprint &middot; under review at NeurIPS 2026** &middot; **Paper:** [arXiv:2606.22958](https://arxiv.org/abs/2606.22958) &middot; Ruolan Sun, Pawel Polak &middot; Stony Brook University

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sophialanlan/PG-MAP/blob/main/notebooks/colab_pgmap_quickstart.ipynb)
[![Reproduce figures](https://img.shields.io/badge/Colab-Reproduce%20paper%20figures-success?logo=googlecolab)](https://colab.research.google.com/github/sophialanlan/PG-MAP/blob/main/notebooks/reproduce_paper_figures.ipynb)
[![HF Space](https://img.shields.io/badge/🤗%20Demo-Gradio%20Space-orange)](https://huggingface.co/spaces/sophialan/pg-map-demo)
[![HF Pipelines](https://img.shields.io/badge/🤗%20Pipelines-sd15%20%2F%20sdxl%20%2F%20sd3-yellow)](https://huggingface.co/sophialan)
[![PyPI](https://img.shields.io/badge/PyPI-pgmap--align-blue)](https://pypi.org/project/pgmap-align/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-9cf)](comfyui/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2606.22958-b31b1b.svg)](https://arxiv.org/abs/2606.22958)

**PG-MAP** (*Preference-Guided Adaptive MAP*) is a training-free inference-time framework that re-optimizes the conditioning $c$ and the latent $z_t$ at every denoising step via a **trajectory-level Gibbs-MAP / proximal energy** objective with forward-consistency coupling. The same objective $\mathcal{J}_t$ instantiates on both diffusion (SD 1.5, SDXL) and flow-matching (SD3.5-medium) backbones with transport-specific active sets.

This repository is the **public reproducibility release** that backs the paper. It contains:

- Reference implementation of PG-MAP and all variants (MAP-$c$, Reward-$z$, MAP-$cz$, UG-FM, Tuned-CFG$+$PG-MAP)
- Exact configs and seeds for every row in every paper table
- Smoke tests that finish in < 10 min on a single GPU
- One-command reproduction scripts for every main-text table
- Post-hoc analysis utilities (win-rate, Wilcoxon, bootstrap CI, BLIP-VQA audit, CRR oracle)

The code corresponds to tag **`v1.5.2`** (release tag for the current preprint).

---

## Headline numbers

| Backbone | Method | PickScore | HPS | Aesthetic | CLIP |
|---|---|---|---|---|---|
| SD 1.5  | PG-MAP (default) | **56.8%** | 52.8% | 54.0% | 50.6% |
| SD 1.5  | Tuned-CFG + PG-MAP | 53.6% | **66.0%** | **60.2%** | **56.0%** |
| SDXL    | PG-MAP (default) | **56.4%** | 47.1% | 56.2% | 48.1% |
| SDXL    | Tuned-CFG + PG-MAP | 51.3% | **64.6%** | **56.5%** | **52.8%** |
| SD3.5-m | **UG-FM** | **91.9%** | **75.7%** | 51.7% | 54.2% |

PartiPrompts $n=1632$, seed $123$, single seed per prompt, win-rate vs same-seed static baseline. Full numbers + statistical tests live in [docs/REPRODUCE.md](docs/REPRODUCE.md).

---

## Repository layout

```
PG-MAP/
├── pgmap_core.py            # Algorithm 1: inner gradient loop on (c, z_t)
├── pgmap_config.py          # Dataclasses + preset configs (sd15/sdxl)
├── pgmap_reward.py          # Frozen reward wrapper (PickScore / HPS / CLIP / Aesthetic / ImageReward)
├── pgmap_sd15.py            # SD 1.5 generation pipeline
├── pgmap_sdxl.py            # SDXL generation pipeline (dual encoders, pooled emb, time_ids)
├── pgmap_flow_core.py       # Flow-matching version of the inner step (UG-FM)
├── pgmap_flow_sd3.py        # SD3.5-medium pipeline (FlowMatchEulerDiscreteScheduler)
├── pgmap_*_variants.py      # MAP-cz Newton step, FM trust-region variants, etc.
├── pgmap_eval.py            # Main evaluation CLI (PartiPrompts, all methods, scoring)
├── validate_criteria.py     # ScoreManager (PickScore + HPS + AES + CLIP)
├── utils/                   # Frozen scorer wrappers (PickScore / HPS / AES / CLIP)
├── eval/                    # Post-hoc analysis (win-rate, CRR oracle, BLIP-VQA)
├── configs/                 # YAML config per table row (exact hyperparameters)
│   ├── sd15/
│   ├── sdxl/
│   └── sd3/
├── scripts/                 # Reproduction entry points
│   ├── reproduce_table1_sd15.sh
│   ├── reproduce_table1_sdxl.sh
│   ├── reproduce_table2_fm.sh
│   ├── reproduce_table4_crr.sh
│   └── slurm/               # SLURM batch versions
├── tests/smoke/             # < 10 min smoke tests per backbone
├── flowchef_baseline/       # FlowChef comparison (Table 2 head-to-head)
├── docs/
│   ├── REPRODUCE.md         # Detailed table-by-table reproduction
│   ├── HYPERPARAMETERS.md   # Full hyperparameter sweep grids
│   └── HUMAN_EVAL.md        # 100-rater study protocol
└── run_*_experiment.py      # Multi-method driver scripts (newB, mapcz_newton, fm_variants)
```

---

## Benchmark submissions

| Leaderboard | Status | Where to find scripts / results |
|---|---|---|
| **Papers with Code** (PartiPrompts) | submission templates ready | [docs/BENCHMARKS_SUBMISSION.md §1](docs/BENCHMARKS_SUBMISSION.md) |
| **T2I-CompBench++** (attribute binding, spatial, complex) | eval pipeline shipped; run via `eval/t2i_compbench/generate.py` | [eval/t2i_compbench/](eval/t2i_compbench/) + [docs/BENCHMARKS_SUBMISSION.md §2](docs/BENCHMARKS_SUBMISSION.md) |
| **HEIM / GenEval / HPSv2** | not actively submitted (yet) | [docs/BENCHMARKS_SUBMISSION.md §4](docs/BENCHMARKS_SUBMISSION.md) |

## ComfyUI custom nodes

A drop-in ComfyUI bundle lives at [`comfyui/`](comfyui/) — three nodes (Reward Loader → Config Builder → Sampler) for practitioner workflows. Install:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sophialanlan/PG-MAP pg-map
ln -s pg-map/comfyui pg-map-nodes
pip install pgmap-align>=1.5.0      # inside ComfyUI's Python environment
```

Restart ComfyUI; nodes appear under the **PG-MAP** category. Sample workflow at [`comfyui/workflows/pgmap_sdxl_basic.json`](comfyui/workflows/pgmap_sdxl_basic.json). Full walkthrough in [comfyui/README.md](comfyui/README.md).

## One-line drop-in (HuggingFace Hub custom pipelines)

The PG-MAP custom pipelines are published on the HuggingFace Hub. After `pip install pgmap-align`, any user can drop PG-MAP into an existing diffusers stack with a single argument change:

```python
from diffusers import DiffusionPipeline
from pgmap import sdxl_defaults, FrozenRewardModel
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    custom_pipeline="sophialan/pg-map-sdxl",   # ← only diff vs vanilla SDXL
    torch_dtype=torch.float16, variant="fp16",
).to("cuda")

image = pipe(
    "a phoenix rising from ashes",
    pg_map_config=sdxl_defaults(),
    reward_model=FrozenRewardModel("pickscore", device="cuda"),
).images[0]
```

| Backbone | HF custom pipeline | Try it |
|---|---|---|
| Stable Diffusion 1.5  | [`sophialan/pg-map-sd15`](https://huggingface.co/sophialan/pg-map-sd15) | [Colab ▶](https://colab.research.google.com/github/sophialanlan/PG-MAP/blob/main/notebooks/colab_pgmap_quickstart.ipynb) |
| SDXL                  | [`sophialan/pg-map-sdxl`](https://huggingface.co/sophialan/pg-map-sdxl) | [Space ▶](https://huggingface.co/spaces/sophialan/pg-map-demo) |
| SD 3.5-medium (UG-FM) | [`sophialan/pg-map-sd3`](https://huggingface.co/sophialan/pg-map-sd3)  | [Space ▶](https://huggingface.co/spaces/sophialan/pg-map-demo) |

Pass `pg_map_config=None` and the pipeline falls through to the vanilla parent class — these are strict supersets of the standard diffusers pipelines.

## Project page

A self-contained **project page with a paired image gallery** lives at [docs/site/](docs/site/) — `index.html` + `style.css` + 16 web-optimized JPEGs (~2.4 MB). Open it locally with `python -m http.server` from `docs/site/`, host it via GitHub Pages, or zip it for sharing / attaching to a release. See [docs/site/README.md](docs/site/README.md) for the three deployment options.

## Quick start

### 1. Environment

```bash
# Python 3.11, PyTorch 2.x, CUDA 12.x
conda create -n pgmap python=3.11 -y
conda activate pgmap
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. HPS checkpoint (optional, only needed for HPS scoring)

```bash
# Download HPS v2 (1.9 GB) — not bundled in the repo
bash scripts/download_hps.sh
# This places utils/hps/HPS_v2_compressed.pt
```

PickScore is loaded from the HuggingFace Hub on first use, and the aesthetic checkpoint (3.6 MB) is bundled at [utils/aesthetics_model/sac+logos+ava1-l14-linearMSE.pth](utils/aesthetics_model/sac+logos+ava1-l14-linearMSE.pth).

### 3. Smoke test (~5 min on one A100/H200/RTX6000)

```bash
bash tests/smoke/smoke_sd15.sh    # 4 prompts, SD 1.5, ~2 min
bash tests/smoke/smoke_sdxl.sh    # 4 prompts, SDXL,  ~5 min
bash tests/smoke/smoke_fm.sh      # 4 prompts, SD3.5, ~5 min
```

A successful smoke run produces an `out_dir/<method>/images/` folder, a `scores.jsonl`, and a `scores_summary.json` with PickScore / HPS / AES / CLIP means.

### 4. Reproduce a single paper table

```bash
# Table 1, SD 1.5 (rows: baseline + Tuned-CFG + UG + MAP-c + Reward-z + MAP-cz + PG-MAP + Tuned-CFG+PG-MAP)
bash scripts/reproduce_table1_sd15.sh

# Table 1, SDXL
bash scripts/reproduce_table1_sdxl.sh

# Table 2, SD3.5-medium (UG-FM headline + FlowChef head-to-head)
bash scripts/reproduce_table2_fm.sh

# Table 4, CRR-MAP oracle routing diagnostic
bash scripts/reproduce_table4_crr.sh
```

Each script writes to `eval_results/<table>_<backbone>/<row>/` and prints win-rate, Wilcoxon $p$-value, and bootstrap 95% CI to stdout.

### 5. Reproduce everything

```bash
bash scripts/reproduce_all.sh        # ~24 GPU-h on a single H200
```

For SLURM clusters, equivalent sbatch versions live in [scripts/slurm/](scripts/slurm/).

---

## Hardware and compute

| Backbone | DDIM/Euler steps | Resolution | Per-image wall-clock¹ | Table 1 full run² |
|---|---|---|---|---|
| SD 1.5  | 30 | 512²  | ~6 s | ~3 h |
| SDXL    | 50 | 1024² | ~30 s | ~14 h |
| SD3.5-m | 28 | 1024² | ~45 s (UG-FM, $K=4$) | ~20 h |

¹ On an NVIDIA RTX PRO 6000 Blackwell with PG-MAP at $K=2$. ² PartiPrompts $n=1632$, all methods sequential.

Peak VRAM: 16 GB (SD 1.5), 28 GB (SDXL), 50 GB (SD3.5 + reward backward). The full backprop through the SD3.5 transformer is the load-bearing axis vs FlowChef, so don't disable it on FM rows.

---

## Reproducibility checklist

- [x] Single seed per prompt; **seed = 123** master; PartiPrompts shuffled deterministically.
- [x] All hyperparameters exposed as CLI args / YAML configs &mdash; nothing hidden inside model code.
- [x] Generation is **deterministic** under fixed seed + fixed hardware (RTX PRO 6000 Blackwell, fp16).
- [x] Cross-GPU reproducibility (A100 / H100 / RTX 6000): within bootstrap CI half-width.
- [x] Statistical tests: paired Wilcoxon ($p$) + bootstrap 95% CI on win-rates, 1000 resamples.
- [x] Reward-model robustness rows reported separately so PickScore-as-optimizer is not the only signal.
- [x] BLIP-VQA text-faithfulness audit included as a no-regression control.
- [x] Human-eval study protocol + tie-rate breakdown documented in [docs/HUMAN_EVAL.md](docs/HUMAN_EVAL.md).

---

## Method overview

Per denoising step $t$:

$$\mathcal{J}_t(c, z_t) = \underbrace{-\tfrac{1}{2\beta_{t\mid s}}\|r_t(c, z_t)\|^2}_{\text{forward-consistency}} \underbrace{-\tfrac{1}{2\sigma_c^2}\|c-\mu_t\|^2 -\tfrac{1}{2\sigma_z(t)^2}\|z_t-z_t^{\text{ddim}}\|^2}_{\text{Gaussian anchors}} + \underbrace{\lambda\, Q(\hat{x}_0(z_t,c),\, y)}_{\text{preference reward}}$$

with $r_t(c,z_t) = z_t - \sqrt{a_{t\mid s}}\,\hat z_{s,\theta}(z_t, t, c)$ (one-step DDIM consistency residual). $K$ ascent steps on $(c, z_t)$ run inside the high-noise refinement window (fraction $\rho$ of denoising steps), with the reward gate active in the inner sub-window of fraction $\rho_Q$. The latent prior is schedule-adaptive: $\sigma_z(t) = \gamma\sqrt{1-\bar\alpha_t}$ on DDPM and $\sigma_z(t)=\gamma(1-t)$ on FM.

Special cases (set ablation flags in [pgmap_config.py](pgmap_config.py)):

| Setting | Method | Active set $\mathcal{A}_t$ |
|---|---|---|
| `optimize_c=True, optimize_z=False, use_reward=False` | MAP-$c$ | $\{c\}$ |
| `optimize_c=False, optimize_z=True, use_reward=True` | Reward-$z$ | $\{z_t\}$ |
| `optimize_c=True, optimize_z=True, use_reward=False` | MAP-$cz$ ($\lambda=0$) | $\{c, z_t\}$ |
| `optimize_c=True, optimize_z=True, use_reward=True`  | **PG-MAP** (default) | $\{c, z_t\}$ |
| (FM, data-side gate) `optimize_z=True only` | **UG-FM** | $\{z_t\}$ |

CFG is on a different control surface (denoiser vector field, not query point), so it is **composable** with PG-MAP rather than subsumed: the Tuned-CFG + PG-MAP row stacks the two.

See the paper §2 and [pgmap_core.py](pgmap_core.py) for the gradient derivation; the flow-matching reduction is in [pgmap_flow_core.py](pgmap_flow_core.py).

---

## Citation

```bibtex
@misc{sun2026pgmap,
  title={{PG-MAP}: Joint {MAP} Optimization for Inference-Time Alignment of Diffusion and Flow-Matching Models},
  author={Sun, Ruolan and Polak, Pawel},
  year={2026},
  eprint={2606.22958},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  note={Under review at NeurIPS 2026},
  url={https://arxiv.org/abs/2606.22958}
}
```

---

## License

Code: MIT (see [LICENSE](LICENSE)).
Pretrained checkpoints are subject to their original licenses (SD 1.5: CreativeML Open RAIL-M; SDXL: CreativeML Open RAIL++-M; SD3.5: Stability AI Community License; PickScore / HPS / aesthetic: research-use).

## Contact

Open an issue or email `ruolan.sun@stonybrook.edu`.
