# Smoke tests

Fast (≤ 10 min) end-to-end tests that exercise the full pipeline on 4 prompts per backbone. Use them after `pip install -r requirements.txt` to confirm the install and HF caches are working before launching a full table run.

```bash
bash tests/smoke/smoke_sd15.sh    # ~2 min, ~12 GB VRAM
bash tests/smoke/smoke_sdxl.sh    # ~5 min, ~24 GB VRAM
bash tests/smoke/smoke_fm.sh      # ~5 min, ~40 GB VRAM (needs SD3.5 license accepted)
```

Each script writes to `eval_results/smoke_<backbone>/` and runs `_check_smoke.py` to verify:

1. The expected method directories exist.
2. The images directory contains exactly 4 PNGs (one per prompt).
3. `scores_summary.json` is present and metric means are within loose sanity bounds (e.g. PickScore in $[0.10, 0.40]$, Aesthetic in $[1.0, 10.0]$).

Failure modes are explicit (`[SMOKE] FAIL — ...`). Exit code is 0 on success.

The tests do **not** validate win-rates — they only verify the pipeline runs and produces the expected on-disk artifacts. Use [scripts/reproduce_table1_sd15.sh](../../scripts/reproduce_table1_sd15.sh) etc. for actual paper-number validation.
