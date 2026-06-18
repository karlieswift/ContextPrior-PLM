from __future__ import annotations

import torch
import torch.nn as nn

from ..config import MotifInferenceConfig
from .backbone import ProteinMotifInferenceBackbone


class ProteinClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProteinMotifClassifier(nn.Module):
    def __init__(self, cfg: MotifInferenceConfig, num_classes: int):
        super().__init__()
        self.backbone = ProteinMotifInferenceBackbone(cfg)
        self.head = ProteinClassificationHead(cfg.hidden_dim, num_classes, cfg.classifier_dropout)

    def forward(
        self,
        x: torch.Tensor | None,
        residue_mask: torch.Tensor | None = None,
        motif_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        feats = self.backbone(x, residue_mask, motif_mask, **kwargs)
        feats['logits'] = self.head(feats['protein_feat'])
        return feats


class ProteinMotifPretrainModel(nn.Module):

    def __init__(self, cfg: MotifInferenceConfig):
        super().__init__()
        self.backbone = ProteinMotifInferenceBackbone(cfg)
        self.token_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor | None,
        residue_mask: torch.Tensor | None = None,
        motif_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        feats = self.backbone(x, residue_mask, motif_mask, **kwargs)
        feats['token_logits'] = self.token_head(feats['local_field'])
        if feats.get('motif_state') is not None:
            feats['state_proj'] = self.state_proj(feats['motif_state'])
        return feats


# Backward-compatible aliases
MotifTabModelClassifier = ProteinMotifClassifier
