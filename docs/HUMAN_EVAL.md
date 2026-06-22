# Human evaluation study (Table 3)

## Study design

Pairwise comparison of PG-MAP against three baselines on SDXL outputs.

| Field | Value |
|---|---|
| Backbone           | SDXL (50 DDIM, $w{=}5.0$ for static / $w^\star{=}7.5$ for tuned-CFG) |
| Method (LHS)       | PG-MAP, $\lambda{=}0.05$, $K{=}2$, $\eta_z{=}0.005$, PickScore reward |
| Method (RHS)       | one of: SDXL static / Tuned-CFG / NFE-matched UG |
| Prompts            | 62 PartiPrompts (covering 12 PartiPrompts categories) |
| Raters             | 100 volunteers (call for participation via lab mailing list) |
| Judgments per pair | 100 (one per rater) |
| Total judgments    | $62 \times 3 \times 100 = 18{,}600$, of which $6{,}200$ pairwise per-prompt (3 comparisons $\times$ 62 prompts $\times$ ~33 raters per condition rotation) |
| Pair order         | Randomized left/right within each rater |
| Rater task         | "Which image better matches the prompt and looks better overall?" with options: left / right / tie |
| Tie rate           | ~$22\%$ overall (lowest for UG comparison, highest for tuned-CFG) |
| IRB status         | exempt (no PII collected, volunteer, no deception) |

## Result

| Comparison | $n_{\text{decisive}}$ | PG-MAP wins | two-sided $p$ |
|---|---|---|---|
| vs. SDXL static               | 1458 | **60.2%** | $5.9 \times 10^{-15}$ |
| vs. Tuned-CFG ($w^\star{=}7.5$) | 1883 | **56.0%** | $1.8 \times 10^{-7}$  |
| vs. NFE-matched UG            | 1794 | **66.8%** | $1.5 \times 10^{-46}$ |

PG-MAP wins **every** comparison. The largest lift ($\sim 2{:}1$) is against compute-matched UG, confirming that the framework wins outside the PickScore optimizer signal.

## Reproducing the pair pool

The 62-prompt subset is a deterministic build from PartiPrompts seed 123. The builder
(`eval.build_human_eval_pairs`) is available on request; it is invoked as:

```bash
# builder available on request — not shipped in this repo
python -m eval.build_human_eval_pairs \
    --seed 123 \
    --n 62 \
    --strata "Concrete,Abstract,Symbolic,Style" \
    --output data/human_eval_pairs.json
```

(The actual rater session JSON is not redistributed because it contains IP-style timestamps; the pair pool itself is available on request, not committed here.)

## Reproducing the candidate images

The LHS images (PG-MAP) and RHS images (the three baselines) are exactly the corresponding rows of Table 1 SDXL:

- PG-MAP                       → `eval_results/table1_sdxl/pgmap/images/`
- SDXL static                  → `eval_results/table1_sdxl/baseline/images/`
- Tuned-CFG ($w^\star{=}7.5$)  → `eval_results/table1_sdxl/tuned_cfg_only/`
- NFE-matched UG ($\eta_z^\star{=}0.1$) → `eval_results/table1_sdxl/ug/test/eta_1e-01/images/`

So Table 3 is implicitly reproduced once `bash scripts/reproduce_table1_sdxl.sh` finishes — you just need to run your own rater study on the resulting pairs.

## Rater interface

The volunteer study used a minimal browser-based two-image picker. We do not redistribute the live deployment, but the static HTML template (`data/human_eval_template.html`) is available on request for adaptation; the layout is standard (two images side-by-side, three-button vote, persistent rater id in localStorage).

## Bias notes

- **Volunteer pool.** Recruited from a Stony Brook University CS mailing list. Skews technical but unfamiliar with PG-MAP / SDXL specifics. Documented in App. B.
- **Order randomization.** Left/right side randomized per rater per pair.
- **No reward-model contamination.** PickScore was used as PG-MAP's optimizer reward, but raters did *not* see PickScore values during voting — they judged from images alone.
- **No tie-breaking.** Ties are excluded from the win-rate (the denominator is $n_{\text{decisive}}$), which is the standard convention for pairwise preference studies.
