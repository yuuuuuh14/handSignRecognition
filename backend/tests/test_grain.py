"""Phase 5 unit test: GrainModule shape & parameter sanity."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.grain import GrainModule


def test_grain_shape_64_to_32():
    m = GrainModule(in_channels=3, out_channels=32)
    x = torch.randn(2, 3, 64, 64)
    y = m(x)
    assert y.shape == (2, 32, 32, 32), y.shape


def test_grain_shape_arbitrary_even():
    m = GrainModule(in_channels=3, out_channels=32)
    for h, w in [(48, 48), (64, 80), (96, 96)]:
        x = torch.randn(1, 3, h, w)
        y = m(x)
        assert y.shape == (1, 32, h // 2, w // 2), (h, w, y.shape)


def test_grain_param_count():
    m = GrainModule(in_channels=3, out_channels=32)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    # Plan §6.3 budgets ~28K for the (shared) image grain. 3 convs (≈19.4K) +
    # BN/LN (≈400) → ~19.5K. Plan target was a rough estimate; keep an upper
    # bound check so any future channel-doubling is flagged.
    assert n < 30_000, f"Grain params {n} exceeds 30K budget"


def test_grain_forward_no_nans():
    m = GrainModule().eval()
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        y = m(x)
    assert torch.isfinite(y).all()


def test_grain_for_face_crop_pathway():
    """Face crops are also (3, 64, 64) — exercise the same module."""
    m = GrainModule(in_channels=3, out_channels=32)
    x = torch.randn(8, 3, 64, 64)   # (B*T, 3, 64, 64) flattened batch
    y = m(x)
    assert y.shape == (8, 32, 32, 32)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[ok] {name}")
    print("\n=== test_grain.py: PASS ===")
