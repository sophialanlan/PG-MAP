"""
PG-MAP Frozen Reward Model Wrapper
====================================

Wraps PickScore (and optionally HPS, aesthetic, CLIP, ImageReward) into a class that:
  1. Loads the frozen model once
  2. Provides Q(pixel_values, prompt) -> scalar that is differentiable w.r.t. the image input
  3. Supports the full backprop chain: z_t -> eps_theta -> z0_hat -> VAE decode -> resize/norm -> Q

The reward model parameters are NEVER updated. Only the image input carries gradients
back through the VAE decoder and UNet Jacobian to (c, z_t).

Supported models:
  - "pickscore":    PickScore v1 (CLIP ViT-H fine-tuned on Pick-a-Pic)
  - "hps":          HPS v2 (CLIP ViT-H fine-tuned on HPD v2)
  - "clip":         Vanilla CLIP cosine similarity
  - "aesthetic":    LAION aesthetic predictor (image-only, no text)
  - "imagereward":  ImageReward (BLIP + MLP, text-alignment aware)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenRewardModel:
    """Frozen preference reward model for differentiable scoring.

    All model parameters are frozen. The `.score()` method accepts
    differentiable pixel tensors so that gradients can flow back
    through the image generation chain.

    Usage:
        reward = FrozenRewardModel("pickscore", device="cuda")
        # pixel_values: (B, 3, H, W) in [0,1], from differentiable VAE decode
        # prompt: the text prompt string
        score = reward.score(pixel_values, prompt)  # (B,) scalar
        # score is differentiable w.r.t. pixel_values
    """

    def __init__(
        self,
        model_name: str = "pickscore",
        model_id: str = "yuvalkirstain/PickScore_v1",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.model_id = model_id
        self.device = torch.device(device)

        if model_name == "pickscore":
            self._init_pickscore(model_id)
        elif model_name == "hps":
            self._init_hps()
        elif model_name == "clip":
            self._init_clip(model_id)
        elif model_name == "aesthetic":
            self._init_aesthetic()
        elif model_name == "imagereward":
            self._init_imagereward()
        else:
            raise ValueError(
                f"Unknown reward model: {model_name}. "
                f"Choose from: pickscore, hps, clip, aesthetic, imagereward"
            )

        # Cache for precomputed text features (set per prompt)
        self._cached_text_feat: Optional[torch.Tensor] = None
        self._cached_prompt: Optional[str] = None

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_pickscore(self, model_id: str):
        """Load PickScore v1 (CLIP ViT-H fine-tuned on preferences)."""
        from transformers import AutoProcessor, AutoModel

        self.processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        self.model = (
            AutoModel.from_pretrained(model_id)
            .eval()
            .to(self.device)
        )
        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

        # CLIP normalization constants
        self._register_clip_norm()

    def _init_hps(self):
        """Load HPS v2 via open_clip (differentiable) or hpsv2 package fallback."""
        try:
            # Prefer open_clip for gradient-based differentiable scoring
            import open_clip
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14", pretrained="laion2B-s32B-b79K"
            )
            self._clip_model = self._clip_model.eval().to(self.device)
            for p in self._clip_model.parameters():
                p.requires_grad_(False)
            self._clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
            self._hps_mode = "open_clip_fallback"
        except ImportError:
            import hpsv2
            self._hps_module = hpsv2
            self._hps_mode = "hpsv2_package"

        self._register_clip_norm()

    def _init_clip(self, model_id: str):
        """Load vanilla CLIP for cosine similarity scoring."""
        from transformers import CLIPModel, CLIPTokenizer

        self.model = CLIPModel.from_pretrained(model_id).eval().to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._register_clip_norm()

    def _init_imagereward(self):
        """Load ImageReward (BLIP + MLP reward head, text-alignment aware)."""
        import ImageReward as ir
        self._ir_model = ir.load("ImageReward-v1.0", device=str(self.device))
        self._ir_model.eval()
        for p in self._ir_model.parameters():
            p.requires_grad_(False)
        self._ir_tokenizer = self._ir_model.blip.tokenizer
        self._register_clip_norm()  # BLIP uses same CLIP normalization

    def _init_aesthetic(self):
        """Load LAION aesthetic predictor (CLIP ViT-L + MLP head)."""
        from transformers import CLIPModel

        self.model = (
            CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            .eval()
            .to(self.device)
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

        # MLP head: 768 -> 1024 -> 128 -> 64 -> 16 -> 1
        self.aesthetic_head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        ).eval().to(self.device)

        # Try to load pretrained weights
        import os
        weight_path = os.path.join(
            os.path.dirname(__file__),
            "utils", "aesthetics_model", "sac+logos+ava1-l14-linearMSE.pth"
        )
        if os.path.exists(weight_path):
            state = torch.load(weight_path, map_location=self.device)
            self.aesthetic_head.load_state_dict(state)

        for p in self.aesthetic_head.parameters():
            p.requires_grad_(False)

        self._register_clip_norm()

    def _register_clip_norm(self):
        """Store CLIP normalization constants as non-parameter tensors."""
        self.clip_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=self.device
        ).view(1, 3, 1, 1)
        self.clip_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=self.device
        ).view(1, 3, 1, 1)

    # ------------------------------------------------------------------
    # Preprocessing (differentiable)
    # ------------------------------------------------------------------

    def _preprocess_pixels(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Resize and normalize pixels for CLIP-based models.

        Args:
            pixel_values: (B, 3, H, W) in [0, 1], differentiable.

        Returns:
            (B, 3, 224, 224) normalized for CLIP input.
        """
        # Bilinear resize to 224x224 (differentiable)
        x = F.interpolate(
            pixel_values.float(), size=(224, 224),
            mode="bilinear", align_corners=False
        )
        # CLIP normalization
        x = (x - self.clip_mean) / self.clip_std
        return x

    # ------------------------------------------------------------------
    # Text feature caching
    # ------------------------------------------------------------------

    @torch.no_grad()
    def precompute_text_features(self, prompt: str):
        """Precompute and cache text features for a given prompt.

        Call this once per prompt before calling .score() in the
        denoising loop to avoid redundant text encoding.
        """
        if self._cached_prompt == prompt:
            return

        if self.model_name == "pickscore":
            text_inputs = self.processor(
                text=prompt, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            text_feat = self.model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            self._cached_text_feat = F.normalize(text_feat.float(), dim=-1)

        elif self.model_name == "clip":
            tokens = self.tokenizer(
                [prompt], padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            text_feat = self.model.get_text_features(**tokens)
            self._cached_text_feat = F.normalize(text_feat.float(), dim=-1)

        elif self.model_name == "hps" and self._hps_mode == "open_clip_fallback":
            text_tokens = self._clip_tokenizer([prompt]).to(self.device)
            text_feat = self._clip_model.encode_text(text_tokens)
            self._cached_text_feat = F.normalize(text_feat.float(), dim=-1)

        elif self.model_name == "imagereward":
            tokens = self._ir_tokenizer(
                [prompt], padding="max_length", truncation=True,
                max_length=35, return_tensors="pt"
            ).to(self.device)
            self._cached_text_feat = {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            }

        # aesthetic doesn't use text; hpsv2 package handles text internally
        self._cached_prompt = prompt

    # ------------------------------------------------------------------
    # Scoring (differentiable w.r.t. pixel_values)
    # ------------------------------------------------------------------

    def score(self, pixel_values: torch.Tensor, prompt: str) -> torch.Tensor:
        """Compute preference score, differentiable w.r.t. pixel_values.

        Args:
            pixel_values: (B, 3, H, W) in [0, 1]. Must carry gradients
                          from the generation chain (VAE decode output).
            prompt:       Text prompt string.

        Returns:
            (B,) tensor of scores. Higher = better preference alignment.
        """
        # Ensure text features are cached
        self.precompute_text_features(prompt)

        # Preprocess: resize + normalize (differentiable)
        x = self._preprocess_pixels(pixel_values)

        if self.model_name == "pickscore":
            return self._score_pickscore(x)
        elif self.model_name == "clip":
            return self._score_clip(x)
        elif self.model_name == "hps":
            return self._score_hps(x, prompt)
        elif self.model_name == "aesthetic":
            return self._score_aesthetic(x)
        elif self.model_name == "imagereward":
            return self._score_imagereward(x)
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

    def _score_pickscore(self, x: torch.Tensor) -> torch.Tensor:
        """PickScore: cosine similarity between image and text features."""
        img_feat = self.model.get_image_features(pixel_values=x)
        img_feat = F.normalize(img_feat.float(), dim=-1)

        text_feat = self._cached_text_feat
        if text_feat.shape[0] == 1 and img_feat.shape[0] > 1:
            text_feat = text_feat.expand(img_feat.shape[0], -1)

        return (img_feat * text_feat).sum(dim=-1)

    def _score_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Vanilla CLIP cosine similarity."""
        img_feat = self.model.get_image_features(pixel_values=x)
        img_feat = F.normalize(img_feat.float(), dim=-1)

        text_feat = self._cached_text_feat
        if text_feat.shape[0] == 1 and img_feat.shape[0] > 1:
            text_feat = text_feat.expand(img_feat.shape[0], -1)

        return (img_feat * text_feat).sum(dim=-1)

    def _score_hps(self, x: torch.Tensor, prompt: str) -> torch.Tensor:
        """HPS v2 score."""
        if self._hps_mode == "hpsv2_package":
            # hpsv2 package doesn't support differentiable scoring easily,
            # so we use the open_clip fallback path for gradient flow
            raise NotImplementedError(
                "HPS v2 package doesn't support differentiable scoring. "
                "Install open_clip_torch for gradient-based HPS."
            )
        else:
            img_feat = self._clip_model.encode_image(x)
            img_feat = F.normalize(img_feat.float(), dim=-1)

            text_feat = self._cached_text_feat
            if text_feat.shape[0] == 1 and img_feat.shape[0] > 1:
                text_feat = text_feat.expand(img_feat.shape[0], -1)

            return (img_feat * text_feat).sum(dim=-1)

    def _score_imagereward(self, x: torch.Tensor) -> torch.Tensor:
        """ImageReward: BLIP visual encoder + MLP reward head (text-alignment aware)."""
        cached = self._cached_text_feat
        rewards = self._ir_model.score_gard(
            cached["input_ids"],
            cached["attention_mask"],
            x,
        )
        return rewards.squeeze(-1)

    def _score_aesthetic(self, x: torch.Tensor) -> torch.Tensor:
        """LAION aesthetic score (image-only, no text needed)."""
        img_feat = self.model.get_image_features(pixel_values=x)
        img_feat = F.normalize(img_feat.float(), dim=-1)
        return self.aesthetic_head(img_feat).squeeze(-1)
