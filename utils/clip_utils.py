"""Minimal CLIP scorer used by ``pgmap_eval._load_utils_scorers``.

Mirrors the pickscore_utils.Selector API: ``Selector(device).score(images, prompt) -> list[float]``.
Uses the standard openai/clip-vit-large-patch14 model for vanilla CLIPScore.
"""
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

processor_name_or_path = "openai/clip-vit-large-patch14"
model_name_or_path = "openai/clip-vit-large-patch14"


class Selector:
    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).eval().to(device)

    def score(self, images, prompt, softmax=False):
        image_inputs = self.processor(
            images=images, padding=True, truncation=True,
            max_length=77, return_tensors="pt",
        ).to(self.device)
        text_inputs = self.processor(
            text=prompt, padding=True, truncation=True,
            max_length=77, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            scores = (text_embs @ image_embs.T)[0]

            if softmax:
                scores = self.model.logit_scale.exp() * scores
                probs = torch.softmax(scores, dim=-1)
                return probs.cpu().tolist()
            return scores.cpu().tolist()
