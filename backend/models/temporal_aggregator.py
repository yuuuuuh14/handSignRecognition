"""TemporalAggregator — 2-layer Pre-LN Transformer over the 16-frame sequence.

Per IMPLEMENTATION_PLAN §6.1 / §6.2:
    Input  : (B, T=16, d=256)         — per-frame fused token
    Output : (B, d)                   — mean-pooled clip representation

Architecture (per block, ×2):
    x = x + MHA(LayerNorm(x))         # d=256, h=4, dropout=0.1
    x = x + FFN(LayerNorm(x))         # 256 → 512 → 256, GELU, dropout 0.1
    finally: mean over time → (B, d)

Positional embedding (T=16, d=256):
    'sinusoidal' (default, per config) — non-learned; deterministic across runs.
    'learned'                          — `nn.Parameter` initialized small.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sinusoidal_pe(num_tokens: int, dim: int) -> torch.Tensor:
    """Standard 'Attention Is All You Need' sinusoidal positional encoding."""
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal pe requires even dim, got {dim}")
    pe = torch.zeros(num_tokens, dim)
    position = torch.arange(0, num_tokens, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TemporalBlock(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalAggregator(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        num_frames: int = 16,
        pos_embed: str = "sinusoidal",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_frames = int(num_frames)

        if pos_embed == "sinusoidal":
            self.register_buffer("pos_embed", _sinusoidal_pe(num_frames, dim), persistent=False)
            self._pos_learned: nn.Parameter | None = None
        elif pos_embed == "learned":
            self._pos_learned = nn.Parameter(torch.zeros(1, num_frames, dim))
            nn.init.trunc_normal_(self._pos_learned, std=0.02)
            self.pos_embed = None  # type: ignore[assignment]
        else:
            raise ValueError(f"pos_embed must be 'sinusoidal' or 'learned'; got {pos_embed!r}")

        self.blocks = nn.ModuleList(
            TemporalBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, d); got {tuple(x.shape)}")
        B, T, d = x.shape
        if T != self.num_frames:
            raise ValueError(
                f"T={T} does not match aggregator num_frames={self.num_frames}"
            )
        if d != self.dim:
            raise ValueError(f"channel dim {d} does not match module dim {self.dim}")

        if self._pos_learned is not None:
            x = x + self._pos_learned
        else:
            x = x + self.pos_embed.unsqueeze(0)         # (1, T, d) broadcast

        for blk in self.blocks:
            x = blk(x)
        x = self.norm_out(x)
        return x.mean(dim=1)                            # mean-pool over T → (B, d)
