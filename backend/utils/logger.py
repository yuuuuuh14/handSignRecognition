"""Lightweight logging wrapper.

Backends:
    'tensorboard' — torch.utils.tensorboard.SummaryWriter
    'none'        — no-op (drop everything; useful for unit tests / smoke runs)

The wrapper exposes a small surface: scalar(tag, value, step) and close().
Callers should not assume a specific backend type — pass whatever the config
says and let the wrapper handle availability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class Logger:
    def __init__(self, log_dir: Path | str, backend: str = "tensorboard") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self._writer: Any | None = None

        if backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(str(self.log_dir))
            except ImportError:
                print(f"[warn] tensorboard not available; falling back to no-op logger.")
                self.backend = "none"
        elif backend == "none":
            pass
        else:
            raise ValueError(f"unknown logging backend: {backend!r}")

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, float(value), int(step))

    def scalars(self, mapping: dict[str, float], step: int) -> None:
        for tag, val in mapping.items():
            self.scalar(tag, val, step)

    def text(self, tag: str, text: str, step: int = 0) -> None:
        if self._writer is not None and hasattr(self._writer, "add_text"):
            self._writer.add_text(tag, text, int(step))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
