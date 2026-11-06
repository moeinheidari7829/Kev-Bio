"""Qwen3.5 vision-language backbone initialised from Kev, the USFM encoder and LoRA helpers."""
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
REPO = Path(__file__).resolve().parents[1]
DEFAULT_USFM = Path(os.environ.get("USFM_WEIGHTS", REPO / "weights" / "USFM_latest.pth"))

# Kev reuses Qwen special tokens as <state>, <q>, <opt>, </opt>, <decide>
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
QWEN_VISION_LORA = r".*blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])|.*merger\.linear_fc[12]"
VIT_LORA = r".*blocks\.\d+\.(attn\.(qkv|proj)|mlp\.fc[12])"     # USFM and timm DINOv2
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text):
    """Tokenize text without ever producing a special token."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def resolve_run(run):
    """A local directory, or a Hugging Face repo id with an optional @revision."""
    if os.path.isdir(run): return str(run)
    from huggingface_hub import snapshot_download
    repo, _, revision = str(run).partition("@")
    return snapshot_download(repo, revision=revision or None, allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"])


def inject_lora(module, rank, pattern, name):
    from peft import LoraConfig, inject_adapter_in_model
    inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=pattern), module, adapter_name=name)
    params = [p for n, p in module.named_parameters() if "lora_" in n]
    if not params: raise ValueError(f"no layers matched {pattern}")
    for p in params: p.requires_grad_(True)
    return params


def lora_state(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items() if "lora_" in k}


class VLMBackbone(nn.Module):
    """Qwen3.5 (vision tower + language model) with the Kev LoRA adapter on the language model."""

    def __init__(self, base, adapter, device, revision=None, weights_dtype="fp32", max_pixels=640 * 640, trainable=True):
        super().__init__()
        from peft import PeftModel
        from transformers import AutoImageProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration
        self.base, self.revision, self.max_pixels = base, revision, max_pixels
        self.image_processor = AutoImageProcessor.from_pretrained(base, revision=revision, use_fast=False)
        self.image_processor.max_pixels = max_pixels
        self.image_processor.size = {"shortest_edge": 56 * 56, "longest_edge": max_pixels}
        self.tok = AutoTokenizer.from_pretrained(base, revision=revision)
        self.vlm = Qwen3_5ForConditionalGeneration.from_pretrained(
            base, revision=revision, dtype=DTYPES[weights_dtype], attn_implementation="sdpa").model
        for p in self.vlm.parameters(): p.requires_grad_(False)
        self.peft = PeftModel.from_pretrained(self.vlm.language_model, adapter, is_trainable=trainable)
        cfg = self.vlm.config
        self.image_token_id, self.vision_start, self.vision_end = cfg.image_token_id, cfg.vision_start_token_id, cfg.vision_end_token_id
        self.merge = self.vlm.visual.spatial_merge_size
        self.device = device
        self.to(device)

    @classmethod
    def from_run(cls, run, device, weights_dtype="fp32", trainable=False, max_pixels=None):
        """Kev (e.g. jaredpalmer/kev-0.8b@v1.0) or a trained Kev-Bio run."""
        path = resolve_run(run)
        meta = torch.load(Path(path) / "head.pt", map_location="cpu", weights_only=False)
        max_pixels = max_pixels or meta.get("multimodal", {}).get("max_pixels", 640 * 640)
        return cls(meta["base"], path, device, revision=meta.get("base_revision"), weights_dtype=weights_dtype,
                   max_pixels=max_pixels, trainable=trainable), meta


class USFMEncoder(nn.Module):
    """USFM ViT-B/16 -> [B, 196, 768] patch tokens of the image resized to 224 x 224."""
    SIZE, GRID = 224, 14
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def __init__(self, weights=DEFAULT_USFM):
        super().__init__()
        from .usfm_vit import VisionTransformer
        if not Path(weights).exists():
            raise FileNotFoundError(f"USFM weights not found at {weights}; see README or set USFM_WEIGHTS")
        self.vit = VisionTransformer(img_size=self.SIZE, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                     qkv_bias=True, init_values=0.1, use_abs_pos_emb=False, use_rel_pos_bias=False,
                                     use_shared_rel_pos_bias=True, use_mean_pooling=False, num_classes=0)
        state = torch.load(weights, map_location="cpu", weights_only=False)
        state.pop("mask_token", None)
        self.vit.load_state_dict(state, strict=True)
        for p in self.vit.parameters(): p.requires_grad_(False)

    def preprocess(self, image):
        x = np.asarray(image.convert("RGB").resize((self.SIZE, self.SIZE), Image.BICUBIC), dtype=np.float32) / 255.0
        x = (x - np.array(self.MEAN, dtype=np.float32)) / np.array(self.STD, dtype=np.float32)
        return torch.from_numpy(x).permute(2, 0, 1)[None]

    def forward(self, pixels):
        v = self.vit
        x = v.patch_embed(pixels)
        x = torch.cat([v.cls_token.expand(x.shape[0], -1, -1), x], 1)
        bias = v.rel_pos_bias()
        for block in v.blocks:
            x = block(x, rel_pos_bias=bias)
        return v.norm(x)[:, 1:]


class Scaler:
    """Standardises the auxiliary head's measurement target: lengths as log(value / sqrt(W * H)), angles in degrees."""

    def __init__(self, stats):
        self.stats = stats

    def transform(self, qid, v, rec):
        if self.stats[qid].get("relative"):
            return math.log(max(v, 1e-6) / math.sqrt(rec["meta"]["width"] * rec["meta"]["height"]))
        return v

    def forward(self, qid, v, rec):
        s = self.stats[qid]
        return (self.transform(qid, v, rec) - s["mean"]) / s["std"]

    @classmethod
    def fit(cls, records, specs):
        stats = {}
        for qid, m in specs["by_id"].items():
            pairs = [(r["values"][qid], r) for r in records if qid in r["values"]]
            if not pairs: continue
            s = cls({qid: {"relative": None if m["kind"] == "angle" else "gm"}})
            t = np.array([s.transform(qid, v, r) for v, r in pairs])
            std = float(t.std()) if len(t) > 1 and t.std() > 1e-6 else 1.0
            stats[qid] = {**s.stats[qid], "mean": float(t.mean()), "std": std,
                          "train_median": float(np.median([v for v, _ in pairs]))}
        return cls(stats)
