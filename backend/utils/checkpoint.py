"""Checkpoint save/load utilities.

A checkpoint stores enough state to resume training mid-run:
    model.state_dict()       — required
    optimizer.state_dict()   — for resuming optimizer momentum/Adam moments
    scheduler.state_dict()   — for resuming LR schedule
    scaler.state_dict()      — for resuming AMP gradient scaler
    epoch, global_step       — for log alignment
    best_metric              — to know if a freshly resumed run beats history
    metrics                  — last validation metrics (informational)
    extra                    — anything else the caller wants to round-trip
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if (scaler is not None and hasattr(scaler, "state_dict")) else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(payload["scaler"])
    return payload
