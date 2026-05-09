"""GrainModule — replaces ViT's fixed-patch linear projection.

Per IMPLEMENTATION_PLAN §6.2:
    Conv2d(in, 32, k=3, s=1, p=1) + BN + GELU
    Conv2d(32, 32, k=3, s=1, p=1) + BN + GELU
    Conv2d(32, 32, k=3, s=2, p=1) + BN + GELU
    LayerNorm (channel-last)

Input  : (B, in_channels, H, W)
Output : (B, 32, H/2, W/2)

The two stride-1 convs preserve fine spatial information that ViT-style
patching (16×16 linear projection) would discard; the third conv halves
spatial resolution. The trailing LayerNorm operates over the channel axis
in channel-last memory layout, matching the convention used by ConvNeXt
and the original CMT paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ChannelLastLayerNorm(nn.Module):
    """LayerNorm applied across the channel dim of an (B, C, H, W) tensor."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, H, W, C) → norm over last dim → (B, C, H, W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class GrainModule(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 32) -> None:
        super().__init__()
        c = out_channels
        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.conv3 = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c)
        self.act = nn.GELU()
        self.norm = _ChannelLastLayerNorm(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.norm(x)
        return x
