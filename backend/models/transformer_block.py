"""TransformerBlock — LPU → LMHSA → MLPConv with PreNorm + residuals.

Per IMPLEMENTATION_PLAN §6.2 + the original architecture descriptions
(architecture.md §3, document.md §3, transformer.md §5):

    1. LPU (Local Perception Unit)
       - Element-wise / depthwise conv + identity residual.
       - Replaces absolute positional embedding.
    2. LMHSA (Lightweight Multi-Head Self-Attention)
       - Pre-Norm. Conv-strided K/V to reduce attention cost.
       - Adds relative position bias (TODO: only when N>1).
    3. MLPConv
       - Pre-Norm. Two 1×1 Conv1d layers with GELU expansion (mlp_ratio×).
       - "Maps global attention features back to local pixel info."
       - For our N=1 landmark token this is parameter-equivalent to a Linear→Linear MLP,
         but kept as Conv1d to match the architecture description.

Input/Output: (B, N, dim).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .lmhsa import LightweightMHSA
from .lpu import LocalPerceptionUnit


class MLPConv(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, C) → (B, C, N) → conv 1×1 → (B, C, N) → (B, N, C)
        h = x.transpose(1, 2)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return h.transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        kv_stride: int = 2,
    ) -> None:
        super().__init__()
        self.lpu = LocalPerceptionUnit(dim=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LightweightMHSA(
            dim=dim,
            num_heads=num_heads,
            kv_stride=kv_stride,
            attn_dropout=dropout,
            proj_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPConv(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, dim)
        x = self.lpu(x)                          # LPU has built-in residual
        x = x + self.attn(self.norm1(x))         # PreNorm + LMHSA
        x = x + self.mlp(self.norm2(x))          # PreNorm + MLPConv
        return x
