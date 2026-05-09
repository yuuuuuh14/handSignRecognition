"""Loss factory.

Per IMPLEMENTATION_PLAN §7.2 the only supported loss is
    nn.CrossEntropyLoss(label_smoothing=0.1)
exposed via build_loss(cfg) so trainer code stays config-driven.
"""
from __future__ import annotations

import torch.nn as nn


def build_loss(cfg: dict) -> nn.Module:
    loss_cfg = cfg["train"]["loss"]
    name = str(loss_cfg["name"]).lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)))
    raise ValueError(f"unknown loss '{name}' (only 'cross_entropy' is supported)")
