# PG-MAP for ComfyUI

ComfyUI custom node bundle for **PG-MAP** (Preference-Guided Adaptive MAP; preprint, under review at NeurIPS 2026). Drop PG-MAP into any ComfyUI workflow as a three-node combo: load a reward model, build a config, then sample.

## Install

### Option A — `git clone` into ComfyUI

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sophialanlan/PG-MAP pg-map
cd pg-map
pip install -e .                 # install the pgmap-align python package
# OR (if you've already pip-installed pgmap-align elsewhere)
pip install pgmap-align>=1.5.0
```

Then **link the `comfyui/` subdirectory** so ComfyUI picks it up as a custom node:

```bash
# From inside ComfyUI/custom_nodes
ln -s pg-map/comfyui pg-map-nodes
```

Restart ComfyUI. The new nodes appear under the **PG-MAP** category in the right-click menu.

### Option B — symlink the inner comfyui/ dir only

If you already have the PG-MAP repo cloned somewhere else:

```bash
ln -s /path/to/PG-MAP/comfyui ComfyUI/custom_nodes/pg-map-nodes
pip install pgmap-align>=1.5.0
```

## Nodes

| Node | Inputs | Outputs | Notes |
|---|---|---|---|
| **PG-MAP Reward Loader** | reward model name, device | `PGMAP_REWARD` | Loads PickScore / HPS / CLIP / Aesthetic / ImageReward. |
| **PG-MAP Config Builder** | backbone choice + sliders for $\lambda$, $\eta_z$, $\eta_c$, $K$, $\rho$, $\rho_Q$, $\sigma_c$, $\gamma$ + 3 ablation booleans | `PGMAP_CONFIG` | Defaults match the paper's recommended row for the chosen backbone. |
| **PG-MAP Sampler** | backbone, prompts, seed, steps, cfg_scale, dims, optional `PGMAP_CONFIG`, optional `PGMAP_REWARD` | `IMAGE` (ComfyUI tensor) | Loads the official `sophialan/pg-map-{sd15,sdxl,sd3}` custom pipeline from HF Hub (cached per-process). |

## Sample workflow

`workflows/pgmap_sdxl_basic.json` — three-node graph: Reward Loader → Config Builder → Sampler → `SaveImage`. Drag-drop the JSON into ComfyUI to load it.

## Why a separate pipeline (vs reusing ComfyUI's `MODEL`)

PG-MAP requires a differentiable forward path through the UNet/MMDiT and the VAE in the same autograd graph. ComfyUI's `ModelPatcher` wrapper interposes its own forward dispatch and lazy-paged weight handling, which makes reaching the underlying diffusers UNet fragile across ComfyUI versions. The PG-MAP Sampler instead loads its own self-contained diffusers pipeline (cached in process memory) so behavior is bit-identical to the PyPI / HF Hub flow.

Cost: one extra UNet copy in VRAM when a vanilla ComfyUI pipeline is loaded alongside (~5 GB for SDXL fp16). If you're hitting OOM, run only the PG-MAP nodes — they're a strict superset of vanilla sampling once you uncheck the PG-MAP config inputs.

## Hardware

| Backbone | Min VRAM (PG-MAP) | Min VRAM (vanilla mode) |
|---|---|---|
| SD 1.5 | 8 GB  | 4 GB  |
| SDXL   | 24 GB | 10 GB |
| SD 3.5-medium | 32 GB | 12 GB |

PG-MAP's reward backward through the VAE is the VRAM bottleneck. The default K=2 with PickScore reward fits 24 GB on SDXL. Set `K=1` to halve the backward memory; uncheck `use_reward` to fall back to the MAP-cz path (no reward backward at all, ~1.5× speedup).

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

## License

MIT (see [LICENSE](../LICENSE)). Pretrained checkpoints remain under their original licenses.
