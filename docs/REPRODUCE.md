# Reproduction guide

This document walks through reproducing every paper table from a clean install. The high-level commands are listed in the top-level [README.md](../README.md); this page adds row-by-row detail, sanity checks, and known gotchas.

All commands assume the conda env from [scripts/setup_env.sh](../scripts/setup_env.sh) is activated and the repo root is the current working directory.

```bash
conda activate pgmap
cd <repo-root>
```

The master seed is **`123`** and the prompt set is **PartiPrompts shuffled with that seed**; all four tables use the same `n=1632` prompt list. Quick-start smoke-test verification before any full run:

```bash
bash tests/smoke/smoke_sd15.sh    # ~2 min
bash tests/smoke/smoke_sdxl.sh    # ~5 min
bash tests/smoke/smoke_fm.sh      # ~5 min, needs SD3.5 license accepted
```

---

## Table 1: PartiPrompts on diffusion backbones

**Command:**

```bash
bash scripts/reproduce_table1_sd15.sh    # ~3 h on H200
bash scripts/reproduce_table1_sdxl.sh    # ~14 h on H200
```

**Outputs:** `eval_results/table1_{sd15,sdxl}/<method>/scores_summary.json` for each of the 8 rows. The `scores_summary.json` contains:

```json
{
  "means": {"pickscore": {"trained": 0.215, "ref": 0.207}, ...},
  "winrate": {"pickscore": 0.568, "hps": 0.528, ...},
  "n": 1632
}
```

**Row map** (column header in the paper $\Leftrightarrow$ method dir):

| Paper row | Method dir | Config |
|---|---|---|
| Baseline (reference)                | `baseline/`              | [configs/sd15/baseline.yaml](../configs/sd15/baseline.yaml)         |
| Tuned-CFG (w$^\star$ per metric)    | `tuned_cfg_only/`        | val sweep grid in `benchmark_cfg.py`                                 |
| UG (NFE-matched)                    | `ug/test/eta_<best>/`    | val sweep grid in `benchmark_ug.py`                                  |
| MAP-c                               | `mapc/`                  | [configs/sd15/mapc.yaml](../configs/sd15/mapc.yaml)                  |
| Reward-z                            | `reward_z/`              | [configs/sd15/reward_z.yaml](../configs/sd15/reward_z.yaml)          |
| MAP-cz ($\lambda{=}0$)              | `joint_cz/`              | [configs/sd15/mapcz.yaml](../configs/sd15/mapcz.yaml)                |
| **PG-MAP** (default)                | `pgmap/`                 | [configs/sd15/pgmap.yaml](../configs/sd15/pgmap.yaml)                |
| Tuned-CFG + PG-MAP                  | `tuned_cfg_pgmap/pgmap/` | [configs/sd15/tuned_cfg_pgmap.yaml](../configs/sd15/tuned_cfg_pgmap.yaml) |

**Statistical tests.** Each row directory also contains:
- `wilcoxon.json` &mdash; one-sided paired Wilcoxon $p$-values vs the baseline.
- `bootstrap.json` &mdash; 95% CIs on the per-metric win-rate, 1000 resamples.

**Sanity check.** After the script finishes, the PG-MAP row should reproduce:

| Backbone | PickScore | HPS | Aesthetic | CLIP |
|---|---|---|---|---|
| SD 1.5   | $56.8\% \pm 1.4$ | $52.8\% \pm 1.4$ | $54.0\% \pm 1.4$ | $50.6\% \pm 1.4$ |
| SDXL     | $56.4\% \pm 1.4$ | $47.1\% \pm 1.4$ | $56.2\% \pm 1.4$ | $48.1\% \pm 1.4$ |

(95% bootstrap CI half-widths; cross-GPU drift is within this margin.)

---

## Table 2: Flow matching on SD3.5-medium

**Command:**

```bash
bash scripts/reproduce_table2_fm.sh    # ~20 h on H200
```

This drives 4 rows:

| Paper row | Method dir | Config |
|---|---|---|
| Baseline (reference)         | `flow_baseline/`       | [configs/sd3/baseline.yaml](../configs/sd3/baseline.yaml)             |
| FlowChef (always-on)         | `flowchef_alwayson/`   | [configs/sd3/flowchef_alwayson.yaml](../configs/sd3/flowchef_alwayson.yaml) |
| FlowChef (gating-matched)    | `flowchef_dataside/`   | [configs/sd3/flowchef_dataside.yaml](../configs/sd3/flowchef_dataside.yaml) |
| **UG-FM** (Ours, headline)   | `ug_fm_data/`          | [configs/sd3/ug_fm.yaml](../configs/sd3/ug_fm.yaml)                  |

**Bitwise audit (recommended before headline runs).** At $\eta_z{=}0$ the UG-FM and FlowChef adapters must reproduce the static SD3.5 baseline at $0/255$ pixel deviation, since the inner ascent becomes the identity. Run:

```bash
cd flowchef_baseline
python audit_eta0.py --n 20 --seed 42
```

Both pipelines should report `max_abs_diff = 0` over 20 prompts.

**The load-bearing axis.** UG-FM and FlowChef differ in **one line**: FlowChef wraps the velocity call in `with torch.no_grad():` (gradient skipping), UG-FM does not. See [flowchef_baseline/README.md](../flowchef_baseline/README.md) for the side-by-side. The $16.9$ pp PickScore gap (gating-matched FlowChef vs UG-FM) attributes to the Jacobian factor $I - (1-t)\,\partial_z v_\theta$ that gradient skipping discards.

**Sanity check.** UG-FM row should reproduce:

```
PickScore 91.9% ± 1.0   HPS 75.7% ± 1.3   Aesthetic 51.7% ± 1.4   CLIP 54.2% ± 1.4
```

with one-sided Wilcoxon $p < 10^{-100}$ on PickScore and HPS.

---

## Table 3: Human evaluation

The human-eval study is **not** automatically reproduced — it requires running the volunteer human-eval site (the live deployment is not redistributed). Study materials, IRB exempt status documentation, and tie-rate breakdown are in [docs/HUMAN_EVAL.md](HUMAN_EVAL.md). The 62-prompt pair pool used in the paper is deterministic given the master seed; the pair pool and its builder are available on request.

If you only need to reproduce the table's PG-MAP candidate images (the LHS of each pair), they are the SDXL PG-MAP row from Table 1 — already produced by `bash scripts/reproduce_table1_sdxl.sh`. The three RHS baselines (SDXL static / Tuned-CFG / NFE-matched UG) are also already in `eval_results/table1_sdxl/`.

---

## Table 4: CRR-MAP oracle routing diagnostic

**Command:**

```bash
bash scripts/reproduce_table4_crr.sh    # ~5 min (post-hoc aggregation only)
```

**Prerequisite:** Table 1 already reproduced (so the per-prompt scores for MAP-$c$, MAP-$cz$, and Tuned-CFG$+$PG-MAP exist on disk). The script reads each method's `scores.jsonl` and computes the oracle ceiling: the win-rate achieved if a perfect selector could dispatch each prompt to its best-scoring variant.

The default aggregator is *balanced-rank* (within-prompt 4-metric rank-sum). Alternative aggregates are toggled via `--aggregate {balanced_rank,pickscore_only,clip_only,pareto_sum}` and reported in App. C.

**Outputs:**

```
eval_results/table4_crr/
├── sd15_crr.json    # SD 1.5 oracle ceiling
├── sdxl_crr.json    # SDXL oracle ceiling
└── sd3_crr.json     # SD3.5 FM oracle ceiling (UG-FM eta_z regimes)
```

---

## Post-hoc analyses

| Script | Produces |
|---|---|
| `eval/compute_blip_vqa_alignment.py`     | BLIP-VQA text-faithfulness audit (App. paragraph: PG-MAP doesn't trade alignment) |
| `eval/compute_blip_vqa_alignment_fm.py`  | Same for FM rows |
| `eval/compute_partiprompts_categories.py`| Per-category breakdown of PartiPrompts (12 prompt categories) |
| `eval/compute_crr_multiseed.py`          | Multi-seed CRR stability ($n=5$ seeds, $n=20$ prompts per seed) |
| `eval/compute_cfg_winrate.py`            | Tuned-CFG row win-rate vs static baseline |
| `eval/compute_ug_winrate.py`             | NFE-matched UG row win-rate |

All take `--results_dir <path-to-eval_results>` and write a JSON next to it.

---

## Common gotchas

1. **`HF_HOME` not set.** Without [scripts/_common.sh](../scripts/_common.sh) sourcing or an env override, diffusers re-downloads SDXL (~7 GB) on every run.
2. **HPS checkpoint missing.** Rows that score with `--reward_model hps` (or the default scorer set that includes HPS) need [scripts/download_hps.sh](../scripts/download_hps.sh) to have been run once.
3. **`utils/` import error.** The PartiPrompts loader and the four frozen scorers live under `utils/`; this is a regular Python package (has `__init__.py`). If you symlink `utils/` elsewhere, make sure `PYTHONPATH` includes the repo root.
4. **SDXL CFG scale.** The paper's `Tuned-CFG+PG-MAP` row uses $w^\star{=}7.5$ for SDXL (vs $w{=}5.0$ for the static baseline). Win-rate is still computed against the $w{=}5.0$ reference — don't change the baseline.
5. **FM peak VRAM.** SD3.5 + reward backward peaks at ~50 GB. Reduce by lowering `K_ug` from 4 to 2 (~$-3$ pp PickScore on the headline, still strong).
6. **PartiPrompts shuffling.** The shuffle is deterministic given `seed=123`. If you change the seed, the prompt subset shifts — table numbers will not match.
