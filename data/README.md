# Data

This directory is reserved for **per-run inputs** (the prompt set is otherwise pulled live from HuggingFace).

## PartiPrompts

PartiPrompts is loaded at run time via `pgmap_eval.py:load_parti_prompts(seed, n)`. The exact `n=1632` ordering used by every paper table is **deterministic** given the master seed `123` — there is no static prompt JSON to redistribute.

To dump the exact prompt list to a JSON file for offline inspection:

```python
from pgmap_eval import load_parti_prompts
import json

prompts = load_parti_prompts(seed=123, n=1632)
json.dump(prompts, open("data/partiprompts_n1632_seed123.json", "w"), indent=2)
```

The first 10 prompts (for sanity-checking your install):

```
0: "A peaceful lakeside landscape"
1: "Three-quarters front view of a blue 1977 Corvette coming around a curve ..."
...
```

(actual prompts vary slightly if upstream `nateraw/parti-prompts` ever changes — pinned via the master seed + cleaning rule).

## Human-evaluation pair pool

The 62 PartiPrompts used in the Table 3 human study are a deterministic subset:

```bash
# builder available on request — not shipped in this repo
python -m eval.build_human_eval_pairs \
    --seed 123 --n 62 \
    --strata "Concrete,Abstract,Symbolic,Style" \
    --output data/human_eval_pairs.json
```

(The builder script `eval.build_human_eval_pairs` is available on request; the resulting `data/human_eval_pairs.json` is not committed here but is reproducible from the seed.)

## Custom prompts

Pass `--prompt_file path/to/my_prompts.json` to `pgmap_eval.py` to override PartiPrompts entirely. The JSON should be a flat list of strings.
