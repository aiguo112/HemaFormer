"""HemaFormer: ConvNeXt-Tiny tokens + TransformerEncoder + multi-task heads.

Population aggregation via global self-attention over smear patch tokens.
Optional BiomedBERT grounding (train-only InfoNCE).
Ablation: use_transformer=False → global-pool backbone, skip TransformerEncoder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

log = logging.getLogger(__name__)

BIOMEDBERT_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
BERT_FALLBACK_ID = "bert-base-uncased"


@dataclass
class HemaFormerConfig:
    backbone: str = "convnext_tiny"
    n_layers: int = 3
    d_model: int = 768
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_anemia: int = 2
    num_background: int = 3
    num_features: int = 7
    num_cbc: int = 4
    proj_dim: int = 256
    cbc_head: bool = True
    grounding: bool = True
    use_transformer: bool = True
    use_background: bool = True
    use_features: bool = True
    max_text_len: int = 128
    max_tokens: int = 1024


class HemaFormer(nn.Module):
    def __init__(self, cfg: HemaFormerConfig | None = None):
        super().__init__()
        self.cfg = cfg or HemaFormerConfig()
        c = self.cfg

        self.backbone = timm.create_model(
            c.backbone,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 3, 224, 224))
        if feat.ndim != 4:
            raise RuntimeError(f"Expected NCHW features, got {tuple(feat.shape)}")
        feat_dim = int(feat.shape[1])
        if feat_dim != c.d_model:
            self.input_proj = nn.Linear(feat_dim, c.d_model)
        else:
            self.input_proj = nn.Identity()

        self.use_transformer = bool(c.use_transformer)
        self.use_grad_checkpoint = True

        if self.use_transformer:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, c.d_model))
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + c.max_tokens, c.d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=c.d_model,
                nhead=c.n_heads,
                dim_feedforward=int(c.d_model * c.mlp_ratio),
                dropout=c.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=c.n_layers)
            self.encoder_norm = nn.LayerNorm(c.d_model)
        else:
            self.cls_token = None
            self.pos_embed = None
            self.encoder = None
            self.encoder_norm = nn.LayerNorm(c.d_model)

        self.head_anemia = nn.Linear(c.d_model, c.num_anemia)
        self.head_background = (
            nn.Linear(c.d_model, c.num_background) if c.use_background else None
        )
        self.head_features = (
            nn.Linear(c.d_model, c.num_features) if c.use_features else None
        )
        self.head_cbc = nn.Linear(c.d_model, c.num_cbc) if c.cbc_head else None

        self.text_encoder = None
        self.text_tokenizer = None
        self.grounding_enabled = False
        self.img_proj = nn.Linear(c.d_model, c.proj_dim)
        self.txt_proj = nn.Linear(768, c.proj_dim)
        if c.grounding:
            self._try_load_text_encoder()

        self._init_heads()

    def _init_heads(self) -> None:
        for m in (self.head_anemia, self.head_background, self.head_features, self.head_cbc):
            if m is None:
                continue
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for m in (self.img_proj, self.txt_proj):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def _try_load_text_encoder(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer

            for model_id in (BIOMEDBERT_ID, BERT_FALLBACK_ID):
                try:
                    log.info("Loading text encoder: %s", model_id)
                    # Prefer local cache to avoid hung hub revision downloads mid-study.
                    try:
                        self.text_tokenizer = AutoTokenizer.from_pretrained(
                            model_id, local_files_only=True
                        )
                        self.text_encoder = AutoModel.from_pretrained(
                            model_id, local_files_only=True
                        )
                    except Exception:
                        self.text_tokenizer = AutoTokenizer.from_pretrained(model_id)
                        self.text_encoder = AutoModel.from_pretrained(model_id)
                    hidden = int(self.text_encoder.config.hidden_size)
                    if hidden != self.txt_proj.in_features:
                        self.txt_proj = nn.Linear(hidden, self.cfg.proj_dim)
                        nn.init.trunc_normal_(self.txt_proj.weight, std=0.02)
                        nn.init.zeros_(self.txt_proj.bias)
                    self.grounding_enabled = True
                    self.cfg.grounding = True
                    log.info("Grounding ON with %s", model_id)
                    return
                except Exception as e:  # noqa: BLE001
                    log.warning("Failed to load %s: %s", model_id, e)
            raise RuntimeError("All text encoders failed")
        except Exception as e:  # noqa: BLE001
            log.warning("Grounding disabled (weights unavailable): %s", e)
            self.text_encoder = None
            self.text_tokenizer = None
            self.grounding_enabled = False
            self.cfg.grounding = False

    def tokenize_reports(self, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        assert self.text_tokenizer is not None
        cleaned = [(t or "").strip() or "peripheral smear" for t in texts]
        enc = self.text_tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_len,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        assert self.text_encoder is not None
        toks = self.tokenize_reports(texts, device)
        out = self.text_encoder(**toks)
        last = out.last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1).float()
        return (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Return smear embedding z_img [B, D]."""
        x = images.to(memory_format=torch.channels_last)
        feat = self.backbone(x)  # [B,C,H,W]

        if not self.use_transformer:
            # Ablation: global average pool → linear proj → LayerNorm
            pooled = feat.mean(dim=(2, 3))
            z = self.input_proj(pooled)
            return self.encoder_norm(z)

        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.input_proj(tokens)
        n = tokens.shape[1]
        if n > self.cfg.max_tokens:
            raise RuntimeError(
                f"Spatial tokens {n} exceed max_tokens={self.cfg.max_tokens} ({h}x{w})"
            )
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]

        if self.training and self.use_grad_checkpoint:
            enc = tokens
            for layer in self.encoder.layers:
                enc = torch.utils.checkpoint.checkpoint(layer, enc, use_reentrant=False)
        else:
            enc = self.encoder(tokens)
        enc = self.encoder_norm(enc)
        return enc[:, 0]

    def forward(
        self,
        images: torch.Tensor,
        report_texts: list[str] | None = None,
        *,
        return_grounding: bool = False,
    ) -> dict[str, Any]:
        z_img = self.encode_image(images)
        out: dict[str, Any] = {
            "z_img": z_img,
            "logits_anemia": self.head_anemia(z_img),
            "logits_background": (
                self.head_background(z_img) if self.head_background is not None else None
            ),
            "logits_features": (
                self.head_features(z_img) if self.head_features is not None else None
            ),
            "pred_cbc": self.head_cbc(z_img) if self.head_cbc is not None else None,
        }

        if (
            return_grounding
            and self.grounding_enabled
            and report_texts is not None
            and self.training
        ):
            z_text = self.encode_text(report_texts, images.device)
            out["z_img_proj"] = F.normalize(self.img_proj(z_img), dim=-1)
            out["z_text_proj"] = F.normalize(self.txt_proj(z_text), dim=-1)
        return out

    def param_groups(self, backbone_lr: float, new_lr: float) -> list[dict]:
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        text_ids: set[int] = set()
        if self.text_encoder is not None:
            text_ids = {id(p) for p in self.text_encoder.parameters()}

        backbone_params, new_params = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in backbone_ids or id(p) in text_ids:
                backbone_params.append(p)
            else:
                new_params.append(p)
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if new_params:
            groups.append({"params": new_params, "lr": new_lr})
        return groups


def info_nce_loss(
    z_img: torch.Tensor,
    z_text: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = z_img @ z_text.t() / temperature
    labels = torch.arange(z_img.size(0), device=z_img.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    diff = (pred - target) ** 2
    m = mask.float()
    return (diff * m).sum() / m.sum().clamp(min=1.0)
