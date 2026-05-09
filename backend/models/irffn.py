"""IRFFN — Inverted Residual FFN classifier head.

Per IMPLEMENTATION_PLAN §6.2:
    Linear(d → 4d) + GELU + Dropout + Linear(4d → d)
    + 잔차 (입력 d-d와)

The 4× expansion mirrors the inverted-residual pattern from MobileNetV2; here
applied as the final pre-classifier feature transform (not a stack of conv
blocks) over the (B, d=256) clip feature produced by TemporalAggregator.

`IRFFNClassifier` composes IRFFN with a final Linear(d → num_classes) so the
top-level KSLRNet just instantiates this module instead of two separate ones.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IRFFN(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = int(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"channel dim {x.shape[-1]} does not match module dim {self.dim}")
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class IRFFNClassifier(nn.Module):
    """IRFFN followed by Linear(d → num_classes) for logits output."""

    def __init__(
        self,
        dim: int = 256,
        hidden_dim: int = 1024,
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.irffn = IRFFN(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.irffn(x)
        return self.head(x)
