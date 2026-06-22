# Model zoo

Every model loaded by this codebase, where it comes from, and what it's used for.

## Generative backbones

| Model | HF id | License | Used in |
|---|---|---|---|
| Stable Diffusion 1.5    | `stable-diffusion-v1-5/stable-diffusion-v1-5`           | CreativeML Open RAIL-M       | Table 1 SD 1.5 panel |
| SDXL base 1.0           | `stabilityai/stable-diffusion-xl-base-1.0` | CreativeML Open RAIL++-M     | Table 1 SDXL panel, Table 3 human eval |
| SD 3.5-medium           | `stabilityai/stable-diffusion-3.5-medium`  | Stability Community License  | Table 2 FM panel — **license acceptance required** |

## Reward / scoring models

| Model | HF id / source | License | Used in |
|---|---|---|---|
| PickScore v1                  | `yuvalkirstain/PickScore_v1`               | MIT       | PG-MAP optimizer reward (default) + scorer |
| HPS v2                        | `xswu/HPSv2` (checkpoint)                  | research  | scorer; also tested as optimizer reward |
| LAION aesthetic predictor     | bundled `utils/aesthetics_model/sac+logos+ava1-l14-linearMSE.pth` | MIT-like  | scorer (Aesthetic column) |
| CLIP ViT-L/14                 | `openai/clip-vit-large-patch14`            | MIT       | CLIPScore scorer |
| ImageReward                   | `THUDM/ImageReward`                        | Apache-2  | robustness row (Apx) |
| BLIP-VQA                      | `Salesforce/blip-vqa-base`                 | BSD-3     | text-faithfulness audit (App.) |

## PartiPrompts

| Dataset | HF id | License | Used in |
|---|---|---|---|
| PartiPrompts | `nateraw/parti-prompts` | Apache-2 | every table |

Loaded by [pgmap_eval.py:load_parti_prompts](../pgmap_eval.py) — 1632 prompts after cleaning, shuffled deterministically with the master seed.

## On-disk sizes

| Item | Size |
|---|---|
| SD 1.5 weights (fp16)                  | ~3.4 GB |
| SDXL base weights (fp16)               | ~6.9 GB |
| SD 3.5-medium weights (fp16)           | ~5.4 GB |
| PickScore CLIP-ViT-H weights (fp16)    | ~1.7 GB |
| HPS v2 checkpoint                      | ~1.9 GB |
| LAION aesthetic predictor              | ~3.6 MB (bundled) |
| Open CLIP ViT-L (for CLIPScore)        | ~890 MB |
| BLIP-VQA                               | ~1.0 GB |

Set `HF_HOME` somewhere with at least 25 GB free before the first run.
