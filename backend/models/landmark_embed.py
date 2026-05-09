"""LandmarkEmbed — Linear projection of flat landmarks → 1-token TransformerBlock.

Per IMPLEMENTATION_PLAN §6.1, the per-frame landmark stream is:

    flatten landmarks → Linear(in_dim → embed_dim)
                     → LPU → LMHSA → MLPConv (sequence length 1)
                     → embed_dim-d per-frame token

Hand stream:  in_dim = 2 hands × 21 lm × 3 coords = 126
Face stream:  in_dim =          97 lm × 3 coords = 291  (lab subset)

Because the resulting sequence is exactly one token per frame, LPU degenerates
to identity (see lpu.py) and LMHSA's KV-stride is auto-skipped (see lmhsa.py).
The block parameters still exist so the architecture matches the design and so
the same module can be reused later with longer sequences.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .transformer_block import TransformerBlock


class LandmarkEmbed(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        kv_stride: int = 2,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.proj = nn.Linear(in_dim, embed_dim)
        self.block = TransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            kv_stride=kv_stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*T, in_dim) flat landmarks → (B*T, embed_dim)."""
        if x.ndim != 2:
            raise ValueError(f"LandmarkEmbed expects (B*T, in_dim); got {tuple(x.shape)}")
        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"in_dim mismatch: got {x.shape[1]}, expected {self.in_dim}"
            )
        h = self.proj(x).unsqueeze(1)            # (B*T, 1, embed_dim)
        h = self.block(h)                        # (B*T, 1, embed_dim)
        return h.squeeze(1)                      # (B*T, embed_dim)
