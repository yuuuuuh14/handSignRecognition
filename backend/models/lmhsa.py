"""LightweightMHSA — multi-head self-attention with KV stride reduction.

Per IMPLEMENTATION_PLAN §6.2:
    Q = Linear(x)                     # (B, N, d)
    K = Linear(Conv1d(x, k=2, s=2))   # (B, N/s, d)
    V = Linear(Conv1d(x, k=2, s=2))   # (B, N/s, d)
    B = nn.Parameter(h, N, N/s)       # relative position bias
    attn = softmax(QK^T / sqrt(d_k) + B) · V

The depthwise stride-s Conv1d cuts K and V sequence length by `kv_stride`,
giving Q·K^T cost O(N · N/s · d) instead of O(N² · d) — significant only
for long sequences.

Sequence-length 1 fallback (per the plan's "구현 주의" note):
    For our landmark stream each per-frame token is a single 64-d vector
    (N==1), so the stride reduction is meaningless and the relative-position
    bias degenerates. We skip the Conv1d reduction and use the standard
    self-attention path. The relative-position bias is omitted in this first
    implementation — see TODO below for when it'll be added.

Input  : (B, N, dim)
Output : (B, N, dim)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightMHSA(nn.Module):
    # TODO(image-stream LMHSA): when N>1 image-stream Transformer is added,
    # implement the relative-position bias B(h, N, N/s) here. For our current
    # landmark-stream usage N==1, so B degenerates to a per-head scalar
    # already absorbed by softmax normalization.

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        kv_stride: int = 2,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if kv_stride < 1:
            raise ValueError("kv_stride must be ≥ 1")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.kv_stride = int(kv_stride)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        # Depthwise stride conv on the channel-first form (B, C, N).
        if kv_stride > 1:
            self.kv_reduce = nn.Conv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=kv_stride,
                stride=kv_stride,
                groups=dim,
            )
        else:
            self.kv_reduce = None

        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

    # ─────────────────────────────────────────────
    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # (B, M, C) → (B, h, M, head_dim)
        B, M, _ = t.shape
        return t.view(B, M, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        # (B, h, N, head_dim) → (B, N, C)
        B, _, N, _ = t.shape
        return t.transpose(1, 2).contiguous().view(B, N, self.dim)

    # ─────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"LMHSA expects (B, N, C); got shape {tuple(x.shape)}")
        B, N, C = x.shape
        if C != self.dim:
            raise ValueError(f"channel dim {C} does not match module dim {self.dim}")

        # KV reduction — skip if degenerate or stride==1.
        if self.kv_reduce is not None and N >= self.kv_stride and N > 1:
            kv = self.kv_reduce(x.transpose(1, 2)).transpose(1, 2)  # (B, N/s, C)
        else:
            kv = x

        Q = self._split_heads(self.q_proj(x))    # (B, h, N, d_h)
        K = self._split_heads(self.k_proj(kv))   # (B, h, M, d_h)
        V = self._split_heads(self.v_proj(kv))   # (B, h, M, d_h)

        # Scaled dot-product attention. (TODO: + relative position bias for N>1)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)              # (B, h, N, d_h)
        out = self._merge_heads(out)             # (B, N, C)
        out = self.proj_drop(self.out_proj(out))
        return out
