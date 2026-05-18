# Configs

YAML configs that document the **exact hyperparameters used for every row of every paper table**. The shell scripts in [`../scripts/`](../scripts/) invoke `pgmap_eval.py` with arguments matching these YAML files. The YAMLs are kept as human-readable, greppable, single-source-of-truth references.

## Schema

```yaml
table:          # which paper table this row appears in
row:            # which row within the table
backbone:       # sd15 | sdxl | sd3
method:         # baseline | mapc | reward_z | joint_cz | pgmap | ug_fm | flowchef | tuned_cfg_pgmap
model_id:       # HuggingFace model id
num_prompts:    # PartiPrompts size (1632 = full, 200 = val, 100 = smoke pool)
seed:           # master seed (123 for all paper tables)
generation:     # backbone-level params
  steps:        # DDIM / Euler steps
  guidance:     # CFG scale w
  height/width: # resolution
refinement:     # PG-MAP inner-loop params (only for refinement methods)
  rho:          # refinement-window fraction
  rho_Q:        # reward sub-window fraction
  K:            # inner gradient ascent steps per denoising step
  eta_c:        # learning rate for c
  eta_z:        # learning rate for z_t
  optimizer:    # sgd | adam
prior:
  sigma_c:      # conditioning prior std
  gamma:        # latent prior scale (sigma_z(t) = gamma * sqrt(1 - alpha_bar_t) for DDPM)
reward:         # (only for methods that use Q)
  model_name:   # pickscore | hps | clip | aesthetic | imagereward
  lambda_reward:
  grad_norm_strategy:  # unit | adaptive | raw
expected_winrate:
  pickscore: …  # paper-reported win-rate (used to sanity-check reproduction)
  hps:       …
  clip:      …
  aesthetic: …
```

## Directory layout

```
configs/
├── sd15/   # Stable Diffusion 1.5  rows of Tables 1 and 4
├── sdxl/   # SDXL                  rows of Tables 1 and 4
└── sd3/    # SD3.5-medium          rows of Table 2 (flow matching)
```

Each row file is named after the method, e.g. `configs/sdxl/pgmap.yaml`.
