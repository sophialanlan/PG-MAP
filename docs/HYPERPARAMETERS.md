# Hyperparameter reference

The exact values used to produce every paper number. Every cell of every table is either:
- a single config in [configs/sd15/](../configs/sd15/), [configs/sdxl/](../configs/sdxl/), or [configs/sd3/](../configs/sd3/), or
- a one-line modification of one of those configs (e.g. swap reward model).

## Per-backbone defaults

| Parameter | SD 1.5 | SDXL | SD3.5-medium |
|---|---|---|---|
| Steps                            | 30           | 50           | 28 (Euler)        |
| CFG scale $w$ (static baseline)  | 7.5          | 5.0          | 7.0               |
| CFG scale $w^\star$ (tuned row)  | 12.0         | 7.5          | n/a               |
| Resolution                       | 512$^2$      | 1024$^2$     | 1024$^2$          |
| Negative prompt                  | "blurry, low quality" | "blurry, low quality" | "" |
| `dtype`                          | fp16         | fp16         | fp16              |

## PG-MAP refinement (DDPM rows)

| Parameter | SD 1.5 | SDXL | Notes |
|---|---|---|---|
| $K$ (inner steps)                | 2            | 2            | matches Algorithm 1 default |
| $\eta_c$                         | $10^{-4}$    | $10^{-3}$    | SDXL c is larger (dual encoder), so larger LR is safe |
| $\eta_z$                         | $0.005$      | $0.005$      | ~$20\times$ smaller than UG default (paired with adaptive prior) |
| $\rho$ (refinement window)       | $0.4$        | $0.5$        | fraction of denoising steps |
| $\rho_Q$ (reward sub-window)     | $0.3$        | $0.3$        | reward gate inside the refinement window |
| $\sigma_c$                       | $1.0$        | $1.0$        | conditioning prior std |
| $\gamma$                         | $1.0$        | $1.0$        | $\sigma_z(t) = \gamma\sqrt{1-\bar\alpha_t}$ on DDPM |
| $\lambda$ (reward weight)        | $0.05$       | $0.10$       | unit-norm reward gradient |
| Reward model                     | PickScore v1 | PickScore v1 | also tested HPS / ImageReward in App. |
| `grad_norm_strategy`             | `unit`       | `unit`       | unit-normalize before scaling by $\lambda$ |
| Optimizer                        | `sgd`        | `sgd`        | newB Newton variant available in `pgmap_variants.py` |

## UG-FM refinement (Flow Matching row)

| Parameter | SD3.5-medium |
|---|---|
| Active set                       | $\{z_t\}$ only |
| Gate                             | data-side (last $\sim 1/3$ of trajectory) |
| $K_{UG}$                         | $4$ |
| $\eta_z$                         | $0.1$ |
| Full backprop through $v_\theta$ | **Yes** — load-bearing axis vs FlowChef |
| $\sigma_z(t)$                    | $\gamma(1-t)$, $\gamma{=}0.5$ |
| Reward model                     | PickScore v1 |

The conditioning branch ($c$-side) drops out on FM because SD3.5's concatenated CLIP-L + CLIP-G + T5-XXL representation has ~$1.4$M optimizable parameters and a unit-normalized gradient cannot move any single direction. The noise-side window drops out because of local Euler amplification ($\delta z^{(K)} \approx \prod_j (I + \Delta t_j\,\partial_z v_\theta)\,\delta z^{(k_0)}$): a noise-side perturbation traverses $\sim 25$ factors, while a data-side perturbation has only $1$--$3$. Full diagnostic in App. C of the paper.

## NFE matching for the UG / FlowChef baselines

| Backbone | PG-MAP NFE per prompt | UG matched $K_{UG}$ | FlowChef matched $K$ |
|---|---|---|---|
| SD 1.5   | $9\!\cdot\!(2{+}1) + 3\!\cdot\!(2{+}1) + 18 = 54$ | $3$ | n/a |
| SDXL     | $15\!\cdot\!(2{+}1) + 10\!\cdot\!(2{+}1) + 25 = 100$ | $3$ | n/a |
| SD3.5-m  | UG-FM at $K{=}4$ → $\sim 116$ vel calls | n/a | $K{=}1$ (released default) |

See `benchmark_ug.py` and `flowchef_baseline/flowchef_eval.py` for the derivation.

## Statistical test settings

| Test | Sides | $n$ | Resamples / iterations |
|---|---|---|---|
| Wilcoxon signed-rank (paired)   | one-sided    | 1632 | -- |
| Bootstrap CI on win-rate        | two-sided    | 1632 | $1000$ resamples |
| Holm-Bonferroni (multi-test)    | --           | --   | applied per-table |
| Cohen's $d$ (effect size)       | paired       | 1632 | -- |
| Binomial CI (human eval)        | two-sided    | varies (Table 3) | exact |

Random seed for bootstrap = `123` (fixed for reproducibility).

## What lambda controls (intuition)

$\lambda$ in $\mathcal{J}_t$ sets the reward step size **relative to the consistency term**. Empirically:

- $\lambda{=}0$ recovers MAP-$cz$ (joint refinement without reward signal).
- $\lambda \in [0.03, 0.10]$ is the productive range; PickScore lifts plateau at $\lambda \approx 0.1$.
- $\lambda > 0.5$ pushes the image off-manifold (visible artifacts, CLIPScore drops $>2$ pp).
- Reward gradient is unit-normalized before scaling, so $\lambda$ has the same meaning across reward models (PickScore / HPS / ImageReward).

Full $\lambda$-sweep in App. ablations; the script is

```bash
python pgmap_eval.py --backbone sd15 --num_prompts 200 \
    --ablation lambda_sweep --out_dir eval_results/ablation/lambda
```

## What gamma controls (intuition)

$\gamma$ in $\sigma_z(t) = \gamma\sqrt{1-\bar\alpha_t}$ sets the trust-region radius for $z_t$. Empirically:

- $\gamma{\to}0$ hard-anchors $z_t = z_t^{\text{ddim}}$ — PickScore win-rate collapses to $10\%$ (latent never moves).
- $\gamma \in [0.5, 2.0]$ all give ~$56\%$ PickScore on SDXL — robust plateau.
- $\gamma{\to}\infty$ recovers Universal Guidance (no latent anchor); paired with $\eta_z{=}0.1$ this is the UG row.

Default $\gamma{=}1.0$ is mid-plateau and stable across backbones.
