# scripts/text_conditioning.py

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class FrozenClinicalBERT(nn.Module):
    """
    Frozen ClinicalBERT text encoder.

    Returns one embedding per text prompt using the [CLS] token.
    """

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 128,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

        # BERT-base-style models are usually 768, but read from config to be safe.
        self.hidden_size = int(self.model.config.hidden_size)

    @torch.no_grad()
    def forward(self, texts: str | Sequence[str], device: torch.device) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Keep ClinicalBERT in fp32 for stable text embeddings.
        with torch.amp.autocast("cuda", enabled=False):
            outputs = self.model(**encoded)

            # [B, hidden_size]
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state[:, 0]

        return emb.float()


class TextToControlChannels(nn.Module):
    """
    Tiny trainable projection from ClinicalBERT embedding to K scalar channels.
    These K values are expanded spatially and optionally gated by the ROI mask.
    """

    def __init__(
        self,
        text_dim: int = 768,
        out_channels: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels),
        )

        # Small init makes this less disruptive to pretrained ControlNet.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        return self.net(text_emb)


def _normalize_text_batch(text_prompts, batch_size: int) -> list[str]:
    if text_prompts is None:
        return [""] * batch_size

    if isinstance(text_prompts, str):
        return [text_prompts] * batch_size

    text_prompts = list(text_prompts)

    if len(text_prompts) != batch_size:
        raise ValueError(
            f"Expected {batch_size} text prompts, got {len(text_prompts)}."
        )

    return text_prompts


def make_roi_gate(
    labels: torch.Tensor,
    target_text_labels: Sequence[int] | None,
    out_size: Sequence[int] | None = None,
) -> torch.Tensor:
    """
    Build [B, 1, H, W, D] binary gate.
    If target_text_labels is provided, text applies only to those labels.
    Otherwise text applies to every non-background voxel.
    """

    if hasattr(labels, "as_tensor"):
        labels_t = labels.as_tensor()
    else:
        labels_t = labels

    labels_t = labels_t.long()

    if labels_t.ndim != 5 or labels_t.shape[1] != 1:
        raise ValueError(f"Expected labels [B,1,H,W,D], got {labels_t.shape}")

    if target_text_labels is None or len(target_text_labels) == 0:
        roi = labels_t > 0
    else:
        roi = torch.zeros_like(labels_t, dtype=torch.bool)
        for lab in target_text_labels:
            roi |= labels_t == int(lab)

    roi = roi.float()

    if out_size is not None and tuple(roi.shape[2:]) != tuple(out_size):
        roi = F.interpolate(roi, size=tuple(out_size), mode="nearest")

    return roi


def build_text_control_channels(
    text_prompts,
    labels: torch.Tensor,
    text_encoder: FrozenClinicalBERT,
    text_projector: TextToControlChannels,
    target_text_labels: Sequence[int] | None,
    out_channels: int,
    mode: str = "roi_gated",
    dropout_prob: float = 0.0,
    is_training: bool = False,
) -> torch.Tensor:
    """
    Return [B, K, H, W, D] text condition channels.

    mode:
      - "roi_gated": text values only inside target ROI labels.
      - "global": text values repeated everywhere.
      - "zero": all zeros.
    """

    if hasattr(labels, "as_tensor"):
        labels_t = labels.as_tensor()
    else:
        labels_t = labels

    device = labels_t.device
    dtype = torch.float32
    batch_size = labels_t.shape[0]
    spatial_size = labels_t.shape[2:]

    if mode == "zero" or text_encoder is None or text_projector is None:
        return torch.zeros(
            batch_size,
            out_channels,
            *spatial_size,
            device=device,
            dtype=dtype,
        )

    texts = _normalize_text_batch(text_prompts, batch_size)

    if is_training and dropout_prob > 0:
        kept_texts = []
        for t in texts:
            if torch.rand((), device=device).item() < dropout_prob:
                kept_texts.append("")
            else:
                kept_texts.append(t)
        texts = kept_texts

    text_emb = text_encoder(texts, device=device)
    text_vec = text_projector(text_emb).float()  # [B, K]

    if text_vec.shape[1] != out_channels:
        raise ValueError(
            f"text_projector produced {text_vec.shape[1]} channels, "
            f"but expected {out_channels}."
        )

    text_cond = text_vec[:, :, None, None, None].expand(
        batch_size,
        out_channels,
        *spatial_size,
    )

    if mode == "global":
        return text_cond

    if mode == "roi_gated":
        roi_gate = make_roi_gate(
            labels=labels_t,
            target_text_labels=target_text_labels,
            out_size=spatial_size,
        )
        return text_cond * roi_gate

    raise ValueError(f"Unknown text condition mode: {mode}")