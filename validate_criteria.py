#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, math, random, argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import StableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler

# -------------------------
# scoring utils (PickScore / AES / HPS / CLIP)
# -------------------------
from typing import Callable

def _find_callable(mod, candidates: List[str]) -> Optional[Callable]:
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None

# -------------------------
# scoring utils (PickScore / AES / HPS / CLIP)  -- all Selectors
# -------------------------
class ScoreManager:
    """
    All scorers expose: Selector(device).score(images, prompt) -> [s_ref, s_trained]
    """
    def __init__(
        self,
        device: str = "cuda",
        enable_pickscore: bool = True,
        enable_aes: bool = True,
        enable_hps: bool = True,
        enable_clip: bool = True,
    ):
        self.device = device

        self.ps = None
        self.aes = None
        self.hps = None
        self.clip = None

        if enable_pickscore:
            from utils.pickscore_utils import Selector
            self.ps = Selector(device)

        if enable_aes:
            from utils.aes_utils import Selector
            self.aes = Selector(device)

        if enable_hps:
            from utils.hps_utils import Selector
            self.hps = Selector(device)

        if enable_clip:
            from utils.clip_utils import Selector
            self.clip = Selector(device)

    @torch.no_grad()
    def score_pair(self, ref_im: Image.Image, tr_im: Image.Image, prompt: str) -> Dict[str, Any]:
        ims = [ref_im, tr_im]

        def _sel_score(sel, name: str):
            s = sel.score(ims, prompt)
            # 期望是 [ref, trained]
            if not isinstance(s, (list, tuple, np.ndarray)) or len(s) < 2:
                raise RuntimeError(f"{name}.score() 返回不符合预期：{type(s)} {s}")
            ref_s, tr_s = float(s[0]), float(s[1])
            return {"ref": ref_s, "trained": tr_s, "delta": tr_s - ref_s}

        out: Dict[str, Any] = {}
        if self.ps is not None:
            out["pickscore"] = _sel_score(self.ps, "pickscore")
        if self.aes is not None:
            out["aes"] = _sel_score(self.aes, "aes")
        if self.hps is not None:
            out["hps"] = _sel_score(self.hps, "hps")
        if self.clip is not None:
            out["clip"] = _sel_score(self.clip, "clip")

        return out


def _aggregate(scores_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {}
    for row in scores_rows:
        ms = row.get("metrics", {})
        for k, v in ms.items():
            metrics.setdefault(k, {"ref": [], "trained": [], "delta": [], "win": 0, "n": 0})
            metrics[k]["ref"].append(v["ref"])
            metrics[k]["trained"].append(v["trained"])
            metrics[k]["delta"].append(v["delta"])
            metrics[k]["win"] += 1 if (v["delta"] > 0) else 0
            metrics[k]["n"] += 1

    summary = {}
    for k, d in metrics.items():
        ref = np.array(d["ref"], dtype=np.float64)
        tr = np.array(d["trained"], dtype=np.float64)
        de = np.array(d["delta"], dtype=np.float64)
        summary[k] = {
            "mean_ref": float(ref.mean()) if len(ref) else None,
            "mean_trained": float(tr.mean()) if len(tr) else None,
            "mean_delta": float(de.mean()) if len(de) else None,
            "win_rate": float(d["win"] / max(d["n"], 1)),
            "n": int(d["n"]),
        }
    return summary
# -------------------------
# MUST match your training stage names
# -------------------------
STAGE_LABELS = [
    "Global scene and mood",
    "Composition and symmetry",
    "Object count grounding",
    "Spatial layout of objects",
    "Object identity / category consistency",
    "Human pose and limb structure",
    "Correct number of limbs, eyes, and body parts",
    "Accurate readable text if applicable",
    "Lighting and shadow realism",
    "Material texture and surface detail",
    "Facial correctness and expression realism",
    "Sharp edges, clean boundaries, final polish",
]
NUM_STAGES = len(STAGE_LABELS)
PATCH_SYSTEM = """
You are a prompt-patch generator for a diffusion-based image model with 12 denoising stages.
Generate 12 prompt patches that progress from coarse → fine detail:

[Global scene and mood]
[Composition and symmetry]
[Object count grounding]
[Spatial layout of objects]
[Object identity / category consistency]
[Human pose and limb structure]
[Correct number of limbs, eyes, and body parts]
[Accurate readable text if applicable]
[Lighting and shadow realism]
[Material texture and surface detail]
[Facial correctness and expression realism]
[Sharp edges, clean boundaries, final polish]

Rules:
- Output exactly 12 patches in the above bracketed order.
- Each patch must be concise: 1–2 sentences, ideally 20–50 tokens.
- Focus ONLY on the current stage; do NOT repeat the base prompt; do NOT mention camera specs unless needed.
- Use concrete constraints (counts, left/right, relative positions). Avoid vague words like "beautiful" or "nice".
- If the base prompt contains no humans/text, do NOT hallucinate them.
""".strip()

def stage_id_from_t(t: torch.Tensor, num_stages: int, T: int) -> int:
    tt = int(t.item())
    sid = ((T - 1 - tt) * num_stages) // T
    return max(0, min(num_stages - 1, sid))
# -------------------------
# utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _clean_prompt(s: str) -> str:
    s = s.replace("<|endoftext|>", " ")
    s = " ".join(s.split())
    return s.strip()

def make_grid(imgs: List[Image.Image], ncols=4, pad=8) -> Image.Image:
    if len(imgs) == 0:
        return Image.new("RGB", (64, 64), (255, 255, 255))
    w, h = imgs[0].size
    n = len(imgs)
    nrows = (n + ncols - 1) // ncols
    grid = Image.new("RGB",
                     (ncols * w + (ncols - 1) * pad, nrows * h + (nrows - 1) * pad),
                     (255, 255, 255))
    for i, im in enumerate(imgs):
        r, c = divmod(i, ncols)
        grid.paste(im, (c * (w + pad), r * (h + pad)))
    return grid

def make_big_pair_grid(ref_imgs: List[Image.Image], tr_imgs: List[Image.Image], pad=8) -> Image.Image:
    assert len(ref_imgs) == len(tr_imgs)
    n = len(ref_imgs)
    if n == 0:
        return Image.new("RGB", (64, 64), (255, 255, 255))
    w, h = ref_imgs[0].size
    grid_w = 2 * w + pad
    grid_h = n * h + (n - 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    for r in range(n):
        y = r * (h + pad)
        grid.paste(ref_imgs[r], (0, y))
        grid.paste(tr_imgs[r], (w + pad, y))
    return grid

@torch.no_grad()
def generate_images(pipe, prompts: List[str], seeds: List[int],
                    bs: int, height: int, width: int,
                    guidance: float, steps: int) -> List[Image.Image]:
    assert len(prompts) == len(seeds)
    out: List[Image.Image] = []
    device = pipe.device

    for i in tqdm(range(0, len(prompts), bs), desc=f"Generating ({pipe.__class__.__name__})", leave=False):
        sub_p = prompts[i:i+bs]
        sub_s = seeds[i:i+bs]
        gens = [torch.Generator(device=device).manual_seed(int(s)) for s in sub_s]
        res = pipe(
            sub_p,
            height=height,
            width=width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gens,
        )
        out.extend([im.convert("RGB") for im in res.images])
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


# -------------------------
# criteria-generator (Qwen LoRA) -> 12 patches
# -------------------------
def parse_patches(raw_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    cur = None
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = line[1:-1].strip()
            out[cur] = ""
        else:
            if cur is not None:
                out[cur] = (out[cur] + " " + line).strip() if out[cur] else line
    # fill missing
    for k in STAGE_LABELS:
        out.setdefault(k, "")
    return out

class CriteriaGeneratorLoRA:
    """
    Uses a base LLM + a LoRA adapter to generate stage patches.
    Works with Qwen/Qwen2.5-7B-Instruct + your saved adapter dir.
    """
    def __init__(self, base_model: str, lora_dir: str, device: str = "cuda", max_new_tokens: int = 700):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, lora_dir)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def generate_patches(self, prompt: str, seed: int = 0) -> Dict[str, str]:
        # 让 greedy 解码在不同机器上尽量稳定
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        messages = [
            {"role": "system", "content": PATCH_SYSTEM},
            {"role": "user", "content": f'Base prompt: "{prompt}"'},
        ]

        # 有些 tokenizer 没有 chat_template，做个 fallback
        if hasattr(self.tok, "apply_chat_template"):
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # fallback: 直接拼接
            text = PATCH_SYSTEM + "\n\nUser: " + messages[-1]["content"] + "\nAssistant:"

        inputs = self.tok(text, return_tensors="pt").to(self.model.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,      # greedy -> deterministic (不需要 temperature/top_p)
            temperature=None,
            top_p=None,
            top_k=None,
        )

        decoded = self.tok.decode(out[0], skip_special_tokens=True)

        # 只要能 parse 出 [xxx] 结构就行
        patches = parse_patches(decoded)
        return patches


# -------------------------
# prompt modes
# -------------------------
def build_prompt_only(base_prompt: str) -> str:
    return base_prompt.strip()

def build_full_patch(base_prompt: str, patches: Dict[str, str]) -> str:
    parts = []
    for s in STAGE_LABELS:
        p = (patches.get(s, "") or "").strip()
        if p:
            parts.append(f"[{s}] {p}")
    if not parts:
        return base_prompt.strip()
    return base_prompt.strip() + ", " + " ".join(parts)


# -------------------------
# main
# -------------------------
def load_sdxl(model_dir: str, device: torch.device, torch_dtype):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        safety_checker=None,
        add_watermarker=None,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config
    )
    return pipe

def _token_len(tok, text: str) -> int:
    # add_special_tokens=True 模拟真实 encoder 输入
    return tok(text, add_special_tokens=True, truncation=False, return_tensors="pt").input_ids.shape[-1]

def truncate_concat_to_77(pipe, base_prompt: str, patch_prompt: str, max_len: int = 77):
    """
    返回 (cond_text, patch_prompt_trunc)
    目标：对 tok1 和 tok2 都满足 token_len(cond_text) <= max_len
    策略：只裁 patch 部分，base_prompt 保留不动。
    """
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2

    base_prompt = " ".join(base_prompt.strip().split())
    patch_prompt = " ".join(patch_prompt.strip().split())

    if not patch_prompt:
        return base_prompt, patch_prompt

    def ok(cond_text: str) -> bool:
        return (_token_len(tok1, cond_text) <= max_len) and (_token_len(tok2, cond_text) <= max_len)

    # 先试一次不截断
    cond = f"{base_prompt}, {patch_prompt}"
    if ok(cond):
        return cond, patch_prompt

    # 只裁 patch：按“词”从后往前砍（简单有效）
    words = patch_prompt.split(" ")
    if len(words) <= 1:
        # patch 太短也放不下，直接不要 patch
        return base_prompt, ""

    lo, hi = 0, len(words)
    best = ""

    # 二分找最大能塞下的 patch 前缀
    while lo <= hi:
        mid = (lo + hi) // 2
        cand_patch = " ".join(words[:mid]).strip()
        cand_cond = f"{base_prompt}, {cand_patch}" if cand_patch else base_prompt
        if ok(cand_cond):
            best = cand_patch
            lo = mid + 1
        else:
            hi = mid - 1

    if not best:
        return base_prompt, ""
    return f"{base_prompt}, {best}", best

@torch.no_grad()
def sdxl_encode_prompt(pipe: StableDiffusionXLPipeline, prompts: List[str], device, dtype):
    # 复用 pipeline 内部的两个 tokenizer/text_encoder，保证和 pipeline 完全一致
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    te1, te2 = pipe.text_encoder, pipe.text_encoder_2

    def _encode(tok, te):
        text_inputs = tok(
            prompts,
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        out = te(input_ids, output_hidden_states=True)
        pooled = out[0]
        hidden = out.hidden_states[-2]
        return hidden, pooled

    h1, p1 = _encode(tok1, te1)
    h2, p2 = _encode(tok2, te2)

    prompt_embeds = torch.cat([h1, h2], dim=-1).to(dtype)
    pooled_prompt_embeds = p2.to(dtype)  # 按 diffusers 习惯用第二个 encoder 的 pooled
    return prompt_embeds, pooled_prompt_embeds


def stage_id_from_sigma(sigma: float, sigmas: torch.Tensor, num_stages: int) -> int:
    # sigma 越大越 noisy -> stage 越小（更 global）
    # 用当前 sigma 在整个推理 sigmas 序列中的“相对位置”来分桶
    sigmas = sigmas.detach().cpu().float()
    # sigmas 通常是从大到小
    idx = int((sigmas - sigma).abs().argmin().item())
    frac = idx / max(len(sigmas) - 1, 1)   # 0..1  (0=最 noisy)
    sid = int(frac * num_stages)
    sid = max(0, min(num_stages - 1, sid))
    return sid

def stage_id_from_step_index(step_i: int, num_steps: int, num_stages: int) -> int:
    # 兜底：按推理步数位置映射（0=最 noisy）
    frac = step_i / max(num_steps - 1, 1)
    sid = int(frac * num_stages)
    sid = max(0, min(num_stages - 1, sid))
    return sid

@torch.no_grad()
def build_stage_condition_embeddings(
    pipe,
    base_prompt: str,
    patch_text: str,
    stage_name: str,
    device,
    dtype,
    patch_fusion: str = "embed_add",
    patch_alpha: float = 0.6,
):
    # base embeds
    base_emb, base_pool = sdxl_encode_prompt(pipe, [base_prompt], device, dtype)

    if (patch_text is None) or (patch_text.strip() == ""):
        return base_emb, base_pool

    # 给 patch 加 stage tag（和训练一致）
    patch_prompt = f"[{stage_name}] {patch_text.strip()}"
    patch_emb, patch_pool = sdxl_encode_prompt(pipe, [patch_prompt], device, dtype)

    if patch_fusion == "concat":
        # 注意：concat 会受 77 token 截断影响（你本来就是想避免它）
        cond_text, patch_prompt_trunc = truncate_concat_to_77(pipe, base_prompt, patch_prompt, max_len=77)
        #cond_text, patch_prompt_trunc = truncate_concat_to_77(pipe,patch_prompt, base_prompt, max_len=77)
        emb, pool = sdxl_encode_prompt(pipe, [cond_text], device, dtype)
        return emb, pool

    # embed_add（推荐）
    alpha = float(patch_alpha)
    emb = base_emb + alpha * patch_emb
    pool = base_pool + alpha * patch_pool
    return emb, pool

from PIL import ImageDraw, ImageFont

def _wrap_text(draw, text, font, max_w):
    # 简单按像素宽度wrap
    lines = []
    for para in text.split("\n"):
        words = para.split(" ")
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if draw.textlength(test, font=font) <= max_w:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        lines.append("")  # 段落空行
    if lines and lines[-1] == "":
        lines.pop()
    return lines

def make_compare_card(ref_img, tr_img, prompt, patches_dict, out_path,
                      pad=24, title_h=140, footer_h=520, font_size=26):
    ref_img = ref_img.convert("RGB")
    tr_img  = tr_img.convert("RGB")

    # 保证同尺寸（用 ref 的尺寸）
    W, H = ref_img.size
    tr_img = tr_img.resize((W, H), Image.BICUBIC)

    card_w = pad + W + pad + W + pad
    card_h = pad + title_h + pad + H + pad + footer_h + pad
    canvas = Image.new("RGB", (card_w, card_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 字体（集群上一般有 DejaVuSans；没有就 fallback）
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        font_small = ImageFont.truetype("DejaVuSans.ttf", int(font_size * 0.85))
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # 标题区：prompt
    x0, y0 = pad, pad
    draw.text((x0, y0), "Prompt:", fill=(0, 0, 0), font=font)
    prompt_lines = _wrap_text(draw, prompt, font, card_w - 2*pad)
    yy = y0 + int(font_size * 1.4)
    for ln in prompt_lines[:4]:  # 你也可以去掉[:4]让它全显示
        draw.text((x0, yy), ln, fill=(0, 0, 0), font=font)
        yy += int(font_size * 1.2)

    # 中间图片区：左 ref 右 trained
    img_y = pad + title_h + pad
    canvas.paste(ref_img, (pad, img_y))
    canvas.paste(tr_img,  (pad + W + pad, img_y))
    draw.text((pad, img_y - int(font_size*1.1)), "REF", fill=(0,0,0), font=font)
    draw.text((pad + W + pad, img_y - int(font_size*1.1)), "TRAINED", fill=(0,0,0), font=font)

    # 底部 patches 文本区
    foot_y = img_y + H + pad
    draw.text((pad, foot_y), "12-stage patches:", fill=(0,0,0), font=font)

    text_block = []
    for s in STAGE_LABELS:
        txt = (patches_dict.get(s, "") or "").strip()
        text_block.append(f"[{s}] {txt}")
    patch_text = "\n".join(text_block)

    lines = _wrap_text(draw, patch_text, font_small, card_w - 2*pad)
    yy = foot_y + int(font_size * 1.4)
    max_lines = int((footer_h - int(font_size*1.6)) / (font_size * 1.1))
    for ln in lines[:max_lines]:
        draw.text((pad, yy), ln, fill=(0,0,0), font=font_small)
        yy += int(font_size * 1.1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, quality=95)

@torch.no_grad()
def generate_images_stagewise_sdxl(
    pipe: StableDiffusionXLPipeline,
    base_prompts: List[str],
    patches_list: List[Dict[str, str]],
    seeds: List[int],
    height: int,
    width: int,
    steps: int,
    guidance: float,
    patch_fusion: str = "embed_add",
    patch_alpha: float = 0.6,
):
    assert len(base_prompts) == len(patches_list) == len(seeds)

    device = pipe._execution_device
    unet_dtype = pipe.unet.dtype
    out_images = []

    for i in tqdm(range(len(base_prompts)), desc="Generating stage-wise (SDXL)"):
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # 如果 scheduler 有 sigmas，用它做“噪声强度”一致映射
        sigmas = getattr(pipe.scheduler, "sigmas", None)
        if sigmas is not None:
            sigmas = sigmas.to(device)

        prompt = base_prompts[i].strip()
        patches = patches_list[i]
        g = torch.Generator(device=device).manual_seed(int(seeds[i]))

        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.unet.config.in_channels,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=g,
        )

        add_time_ids = torch.tensor(
            [height, width, 0, 0, height, width],
            device=device,
            dtype=unet_dtype,
        ).unsqueeze(0)

        for step_i, t in enumerate(timesteps):
            # === stage id: 高噪声 -> global，低噪声 -> 细节 ===
            # timesteps: tensor like [999, 965, ..., 0]
  
            T = pipe.scheduler.config.num_train_timesteps  # usually 1000

            # 和训练代码一模一样的公式
            sid = int(((T - 1 - t) * NUM_STAGES) // T)
            sid = max(0, min(NUM_STAGES - 1, sid))

            stage_name = STAGE_LABELS[sid]
            patch = (patches.get(stage_name, "") or "").strip()

            # === embeddings: 和训练一致的 patch_fusion ===
            cond_embeds, cond_pooled = build_stage_condition_embeddings(
                pipe,
                base_prompt=prompt,
                patch_text=patch,
                stage_name=stage_name,
                device=device,
                dtype=unet_dtype,
                patch_fusion=patch_fusion,
                patch_alpha=patch_alpha,
            )

            uncond_embeds, uncond_pooled = sdxl_encode_prompt(pipe, [""], device, unet_dtype)

            pe = torch.cat([uncond_embeds, cond_embeds], dim=0)
            pp = torch.cat([uncond_pooled, cond_pooled], dim=0)
            time_ids = add_time_ids.repeat(2, 1)

            latent_model_input = torch.cat([latents] * 2, dim=0)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.to(pipe.unet.dtype)

            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=pe,
                added_cond_kwargs={"time_ids": time_ids, "text_embeds": pp},
            ).sample

            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance * (noise_text - noise_uncond)

            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents / pipe.vae.config.scaling_factor
        vae = pipe.vae
        vae_orig_dtype = next(vae.parameters()).dtype
        vae = vae.to(dtype=torch.float32)


        # ✅ 强制用 float32 送进 VAE（见下一节原因）
        latents_f32 = latents.float()

        img_t = pipe.vae.decode(latents_f32).sample  # [1,3,H,W] float tensor
        vae.to(dtype=vae_orig_dtype)


        img_t = (img_t / 2 + 0.5).clamp(0, 1)


        img = img_t[0].detach().cpu().permute(1, 2, 0).numpy()
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)  # ✅ 兜底止血
        img = (img * 255).round().clip(0, 255).astype("uint8")
        out_images.append(Image.fromarray(img, mode="RGB"))

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return out_images

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained_dir", required=True, help="your trained SDXL pipeline dir (outputs/sdxl_criteria_dpo_run1)")
    ap.add_argument("--ref_dir", default="stabilityai/stable-diffusion-xl-base-1.0", help="reference SDXL")
    ap.add_argument("--base_llm", default="Qwen/Qwen2.5-7B-Instruct", help="base LLM for criteria generator")
    ap.add_argument("--lora_dir", required=True, help="criteria generator LoRA dir (e.g., criteria_lora_nb)")
    ap.add_argument("--out_dir", default="eval_parti_criteria", type=str)
    ap.add_argument("--num_prompts", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--gen_seeds", type=int, nargs="+", default=[123], help="image generation seeds; will cycle")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--patch_fusion", type=str, default="embed_add", choices=["concat", "embed_add"])
    ap.add_argument("--patch_alpha", type=float, default=0.6)
    ap.add_argument("--score", action="store_true", help="run automatic scorers (pickscore/aes/hps/clip)")
    ap.add_argument("--score_device", type=str, default="cuda", help="device for scorers (usually cuda)")
    ap.add_argument("--no_pickscore", action="store_true")
    ap.add_argument("--no_aes", action="store_true")
    ap.add_argument("--no_hps", action="store_true")
    ap.add_argument("--no_clip", action="store_true")
    
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # 1) load prompts: PartiPrompts
    from datasets import load_dataset
    ds = load_dataset("nateraw/parti-prompts", split="train")
    # column name in this dataset is usually "Prompt"
    key = "Prompt" if "Prompt" in ds.features else (list(ds.features.keys())[0])
    all_prompts = [_clean_prompt(x) for x in ds[key] if isinstance(x, str) and x.strip()]
    rng = np.random.default_rng(args.seed)
    prompts = rng.choice(all_prompts, size=min(args.num_prompts, len(all_prompts)), replace=False).tolist()

    # generation seeds per prompt
    seeds = [int(args.gen_seeds[i % len(args.gen_seeds)]) for i in range(len(prompts))]

    # 2) load SDXL ref + trained
    pipe_tr = load_sdxl(args.trained_dir, device, torch_dtype)
    pipe_rf = load_sdxl(args.ref_dir, device, torch_dtype)

    # 3) load criteria-generator LoRA
    crit_gen = CriteriaGeneratorLoRA(args.base_llm, args.lora_dir, device=str(device))

    # 4) build two modes
    prompts_prompt_only = [build_prompt_only(p) for p in prompts]

    # generate patches (deterministic seed per prompt)
    patches_list = []
    for i, p in enumerate(tqdm(prompts, desc="Generating 12-stage patches (LLM)")):
        patches = crit_gen.generate_patches(p, seed=args.seed + i)
        patches_list.append(patches)
        '''
        print("\n" + "="*80)
        print(f"[PROMPT {i}] {p}")
        for s in STAGE_LABELS:
            txt = (patches.get(s, "") or "").strip()
            print(f"[{s}] {txt}")
        print("="*80 + "\n")
        '''
    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for i, (p, sd) in enumerate(zip(prompts, seeds)):
            f.write(json.dumps({
                "i": i,
                "seed": int(sd),
                "prompt": p,
                "patches": patches_list[i],  # dict: stage->text
            }, ensure_ascii=False) + "\n")
    print("[OK] wrote", meta_path)
    def save_pair(mode_name: str,
                left_name: str,
                right_name: str,
                left_imgs: List[Image.Image],
                right_imgs: List[Image.Image],
                mode_prompts: Optional[List[str]] = None):
        assert len(left_imgs) == len(right_imgs)
        outd = os.path.join(args.out_dir, mode_name)
        os.makedirs(outd, exist_ok=True)

        # grids
        grid_n = min(16, len(left_imgs))
        sel = list(range(len(left_imgs)))
        random.Random(args.seed).shuffle(sel)
        sel = sel[:grid_n]

        sel_left = [left_imgs[i] for i in sel]
        sel_right = [right_imgs[i] for i in sel]

        make_grid(sel_left, ncols=4).save(os.path.join(outd, f"grid_{left_name}.jpg"), quality=95)
        make_grid(sel_right, ncols=4).save(os.path.join(outd, f"grid_{right_name}.jpg"), quality=95)
        make_big_pair_grid(sel_left, sel_right).save(os.path.join(outd, f"grid_{left_name}_vs_{right_name}.jpg"), quality=95)

        # per-sample
        img_dir_left = os.path.join(outd, f"imgs_{left_name}")
        img_dir_right = os.path.join(outd, f"imgs_{right_name}")
        os.makedirs(img_dir_left, exist_ok=True)
        os.makedirs(img_dir_right, exist_ok=True)

        with open(os.path.join(outd, "prompts.tsv"), "w", encoding="utf-8") as f:
            f.write("i\tseed\tbase_prompt\tmode_prompt\n")
            for i in range(len(prompts)):
                mp = mode_prompts[i] if mode_prompts is not None else prompts[i]
                f.write(f"{i}\t{seeds[i]}\t{prompts[i].replace(chr(9),' ')}\t{mp.replace(chr(9),' ')}\n")
                left_imgs[i].save(os.path.join(img_dir_left, f"{i:05d}.png"))
                right_imgs[i].save(os.path.join(img_dir_right, f"{i:05d}.png"))

        print(f"[OK] saved {mode_name} -> {outd}")

    imgs_ref_base = generate_images(
        pipe_rf, prompts, seeds,
        bs=args.bs, height=args.height, width=args.width,
        guidance=args.guidance, steps=args.steps
    )
    imgs_tr_stage = generate_images_stagewise_sdxl(
        pipe_tr, prompts, patches_list, seeds,
        height=args.height, width=args.width,
        steps=args.steps, guidance=args.guidance,
        patch_fusion=args.patch_fusion,
        patch_alpha=args.patch_alpha,
    )
    card_dir = os.path.join(args.out_dir, "compare_cards")
    ref_dir  = os.path.join(args.out_dir, "imgs_ref_base")
    tr_dir   = os.path.join(args.out_dir, "imgs_tr_stagewise")
    os.makedirs(card_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(tr_dir, exist_ok=True)

    for i in range(len(prompts)):
        # 原图也存一份
        imgs_ref_base[i].save(os.path.join(ref_dir, f"{i:05d}.png"))
        imgs_tr_stage[i].save(os.path.join(tr_dir,  f"{i:05d}.png"))

        # 卡片图（左ref右train，上prompt下patch）
        out_path = os.path.join(card_dir, f"{i:05d}.png")
        make_compare_card(
            ref_img=imgs_ref_base[i],
            tr_img=imgs_tr_stage[i],
            prompt=prompts[i],
            patches_dict=patches_list[i],
            out_path=out_path,
        )
            # -------------------------
    # scoring
    # -------------------------
    if args.score:
        scorer = ScoreManager(
            device=args.score_device,
            enable_pickscore=(not args.no_pickscore),
            enable_aes=(not args.no_aes),
            enable_hps=(not args.no_hps),
            enable_clip=(not args.no_clip),
        )

        score_path = os.path.join(args.out_dir, "scores.jsonl")
        rows = []
        with open(score_path, "w", encoding="utf-8") as f:
            for i in tqdm(range(len(prompts)), desc="Scoring"):
                metrics = scorer.score_pair(imgs_ref_base[i], imgs_tr_stage[i], prompts[i])
                row = {
                    "i": i,
                    "seed": int(seeds[i]),
                    "prompt": prompts[i],
                    "metrics": metrics,
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = _aggregate(rows)
        summary_path = os.path.join(args.out_dir, "scores_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("[OK] wrote", score_path)
        print("[OK] wrote", summary_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("[OK] wrote compare cards ->", card_dir)

'''
    # ---- 生成四个条件的图 ----
    # (1) Ref + base
    imgs_ref_base = generate_images(
        pipe_rf, prompts, seeds,
        bs=args.bs, height=args.height, width=args.width,
        guidance=args.guidance, steps=args.steps
    )

    # (2) Trained + base
    imgs_tr_base = generate_images(
        pipe_tr, prompts, seeds,
        bs=args.bs, height=args.height, width=args.width,
        guidance=args.guidance, steps=args.steps
    )

    # (3) Ref + stagewise patch
    imgs_ref_stage = generate_images_stagewise_sdxl(
        pipe_rf, prompts, patches_list, seeds,
        height=args.height, width=args.width,
        steps=args.steps, guidance=args.guidance,
        patch_fusion=args.patch_fusion,
        patch_alpha=args.patch_alpha,
    )


    # (4) Trained + stagewise patch
    imgs_tr_stage = generate_images_stagewise_sdxl(
        pipe_tr, prompts, patches_list, seeds,
        height=args.height, width=args.width,
        steps=args.steps, guidance=args.guidance,
        patch_fusion=args.patch_fusion,
        patch_alpha=args.patch_alpha,
    )
    # A: baseline vs trained（看 UNet 变强没）
    save_pair(
        mode_name="A_ref_base_vs_trained_base",
        left_name="ref_base",
        right_name="trained_base",
        left_imgs=imgs_ref_base,
        right_imgs=imgs_tr_base,
        mode_prompts=prompts
    )

    # B: patch 注入对 ref 的作用（光靠 patch 的收益）
    save_pair(
        mode_name="B_ref_base_vs_ref_stagewise",
        left_name="ref_base",
        right_name="ref_stagewise",
        left_imgs=imgs_ref_base,
        right_imgs=imgs_ref_stage,
        mode_prompts=prompts
    )

    # C: 你的完整方法 vs baseline（end-to-end）
    save_pair(
        mode_name="C_ref_base_vs_trained_stagewise",
        left_name="ref_base",
        right_name="trained_stagewise",
        left_imgs=imgs_ref_base,
        right_imgs=imgs_tr_stage,
        mode_prompts=prompts
    )

    # （可选）D: 在 stagewise 条件下看 UNet 变强没
    save_pair(
        mode_name="D_ref_stagewise_vs_trained_stagewise",
        left_name="ref_stagewise",
        right_name="trained_stagewise",
        left_imgs=imgs_ref_stage,
        right_imgs=imgs_tr_stage,
        mode_prompts=prompts
    )

    print(f"\n[DONE] All results in: {args.out_dir}\n")
'''

if __name__ == "__main__":
    main()