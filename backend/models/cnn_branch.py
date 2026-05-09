"""CNNBranch — 4-block conv tower with two spatial halvings.

Per IMPLEMENTATION_PLAN §6.2:
    Block 1: Conv(32→64,  k=3, s=1) + BN + GELU + MaxPool 2×2
    Block 2: Conv(64→64,  k=3, s=1) + BN + GELU
    Block 3: Conv(64→128, k=3, s=1) + BN + GELU + MaxPool 2×2
    Block 4: Conv(128→128,k=3, s=1) + BN + GELU
    GlobalAveragePool → 128-d

Input  : (B', 32, 32, 32)  — typically the GrainModule output for a flattened
                              batch of hand or face crops, where B' = B*T*2
                              for hands or B*T for face.
Output : (B', 128)

Captures local shape/texture (fine finger-shape differences the Transformer
landmark stream would miss). The 4-block channel schedule keeps the module
small enough to share weights across the two hand slots — see kslr_net.py
for the (Left, Right) weight-sharing wiring.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_act(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.GELU(),
    )


class CNNBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        channels: tuple[int, int, int, int] = (64, 64, 128, 128),
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        if c4 != out_dim:
            raise ValueError(
                f"channels[3] ({c4}) must equal out_dim ({out_dim}); "
                f"GAP preserves channel dim and that's the produced feature size."
            )
        self.block1 = _conv_bn_act(in_channels, c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = _conv_bn_act(c1, c2)
        self.block3 = _conv_bn_act(c2, c3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = _conv_bn_act(c3, c4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"CNNBranch expects (B, C, H, W); got shape {tuple(x.shape)}")
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.gap(x)              # (B', C, 1, 1)
        return x.flatten(1)          # (B', C)
