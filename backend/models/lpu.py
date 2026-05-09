"""LocalPerceptionUnit — depthwise conv + residual.

Per IMPLEMENTATION_PLAN §6.2:
    LPU(x) = DepthwiseConv2d(x, k=3, p=1) + x   (image-stream form)

The image-stream form would operate on (B, C, H, W). Our pipeline only
exercises the **landmark stream** where each frame is a single d-dim token,
so we implement the 1D form: DepthwiseConv1d over the sequence axis with a
residual connection. This matches the plan's note:

    "본 프로젝트의 landmark stream은 frame당 단일 token이므로 LPU는
     1D conv 형태로 구현하되 sequence 길이가 1이면 identity로 동작."

Input shape: (B, N, C). For N==1 the conv path becomes a per-channel scalar
multiply (the kernel reads zero-padding for the two missing neighbours), so
we short-circuit to identity to avoid wasted compute and to keep the design
intent of "LPU is a no-op when there is no local neighbourhood".
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LocalPerceptionUnit(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("LPU kernel_size must be odd")
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,                # depthwise
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"LPU expects (B, N, C); got shape {tuple(x.shape)}")
        if x.shape[1] == 1:
            return x
        # (B, N, C) → (B, C, N) → conv → (B, N, C)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return h + x
