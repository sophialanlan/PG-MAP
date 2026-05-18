# FlowChef head-to-head baseline

This directory contains the **FlowChef adapter** used for the Table 2 comparison in the paper.

## What's here

| File | Purpose |
|---|---|
| `flowchef_core.py`   | `flowchef_refine_step` — the ONE-line algorithmic difference vs UG-FM (`with torch.no_grad():` wrapping the velocity call). |
| `flowchef_sd3.py`    | SD3.5-medium pipeline wrapper that calls `flowchef_refine_step` per Euler step. |
| `flowchef_eval.py`   | CLI driver: `--num_prompts`, `--seed`, `--K`, `--eta_z`, `--gate_side {alwayson,data,noise}`, `--out_root`. |
| `audit_eta0.py`      | Bitwise audit: at $\eta_z{=}0$ the FlowChef pipeline must reproduce the static SD3.5 baseline at $0/255$ pixel deviation. |
| `aggregate_results.py` | (Legacy) Dev-time aggregator. The portable replacement is [eval/aggregate_table2.py](../eval/aggregate_table2.py). |

## How to use

Use the top-level [`scripts/reproduce_table2_fm.sh`](../scripts/reproduce_table2_fm.sh) — it drives this directory and the FM-variants pipeline together so that win-rates are computed against a shared baseline.

For a manual single-row run:

```bash
# Always-on (released FlowChef default)
python flowchef_baseline/flowchef_eval.py \
    --num_prompts 1632 --seed 123 \
    --K 1 --eta_z 1.0 --gate_side alwayson \
    --out_root eval_results/table2_fm/flowchef_alwayson

# Data-side (gating-matched against UG-FM; isolates the full-backprop axis)
python flowchef_baseline/flowchef_eval.py \
    --num_prompts 1632 --seed 123 \
    --K 1 --eta_z 1.0 --gate_side data \
    --out_root eval_results/table2_fm/flowchef_dataside
```

## The variable under test

PG-MAP's UG-FM and FlowChef differ in **one line**:

```python
# flowchef_baseline/flowchef_core.py — FlowChef variant
with torch.no_grad():
    v = predict_velocity(z, t, c)        # ← gradient skipping

# pgmap_flow_core.py — UG-FM variant
v = predict_velocity(z, t, c)             # ← full backprop through v_theta
```

That single difference accounts for the $16.9$ pp PickScore gap when gating is matched (FlowChef data-side $75.0\%$ vs UG-FM $91.9\%$, paired Wilcoxon $p < 10^{-91}$). See paper §3.2 paragraph "Head-to-head FM baseline".

## Bitwise audit (recommended before headline runs)

```bash
python flowchef_baseline/audit_eta0.py --n 20 --seed 42
```

Expected: `max_abs_diff = 0` over 20 prompts. Confirms that the comparison is apples-to-apples — any non-zero win-rate is attributable to the inner-ascent step, not to scheduler / decoder drift.

## Citation

```bibtex
@inproceedings{patel2025flowchef,
  title={{FlowChef}: Steering Rectified Flow Models via Per-Step Gradient Skipping},
  author={Patel, Aniket and ...},
  booktitle={NeurIPS},
  year={2025}
}
```
