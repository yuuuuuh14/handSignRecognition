"""Phase 5 unit test: LightweightMHSA + LocalPerceptionUnit shape assertions.

Two regimes are exercised because the plan permits both:
  - Landmark-stream regime (N=1): KV stride must be skipped, output should
    equal a standard 1-token self-attention (LPU is identity).
  - General regime (N>1): KV stride reduces K,V length to N/s; attention
    output should still be (B, N, C).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.lmhsa import LightweightMHSA
from models.lpu import LocalPerceptionUnit


# ─────────────────────────────────────────────
# LightweightMHSA
# ─────────────────────────────────────────────
def test_lmhsa_shape_n1_landmark_stream():
    """Our actual usage: per-frame landmark token (N=1)."""
    m = LightweightMHSA(dim=64, num_heads=4, kv_stride=2)
    x = torch.randn(8, 1, 64)
    y = m(x)
    assert y.shape == (8, 1, 64), y.shape


def test_lmhsa_shape_n8_with_kv_stride():
    """Image-stream regime: stride should reduce K,V length but output stays N."""
    m = LightweightMHSA(dim=64, num_heads=4, kv_stride=2)
    x = torch.randn(2, 8, 64)
    y = m(x)
    assert y.shape == (2, 8, 64), y.shape


def test_lmhsa_shape_no_stride():
    m = LightweightMHSA(dim=64, num_heads=4, kv_stride=1)
    x = torch.randn(2, 4, 64)
    y = m(x)
    assert y.shape == (2, 4, 64)


def test_lmhsa_invalid_dim_heads():
    try:
        LightweightMHSA(dim=63, num_heads=4)
        raise AssertionError("expected ValueError for non-divisible dim")
    except ValueError:
        pass


def test_lmhsa_short_sequence_skips_stride():
    """If N < kv_stride, reduction must not be applied — fall back gracefully."""
    m = LightweightMHSA(dim=64, num_heads=4, kv_stride=4)
    x = torch.randn(1, 2, 64)   # N=2 < stride=4
    y = m(x)
    assert y.shape == (1, 2, 64)


def test_lmhsa_param_count():
    m = LightweightMHSA(dim=64, num_heads=4, kv_stride=2)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    # 4× Linear(64→64) ≈ 16640, plus depthwise Conv1d k=2 ≈ 192 → ~17K.
    assert n < 25_000, f"LMHSA params {n} exceeds 25K"


def test_lmhsa_forward_no_nans():
    m = LightweightMHSA(dim=64, num_heads=4).eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        y = m(x)
    assert torch.isfinite(y).all()


# ─────────────────────────────────────────────
# LocalPerceptionUnit
# ─────────────────────────────────────────────
def test_lpu_shape_n1_is_identity():
    m = LocalPerceptionUnit(dim=64)
    x = torch.randn(8, 1, 64)
    y = m(x)
    assert y.shape == x.shape
    assert torch.equal(y, x), "LPU(N=1) must be exact identity"


def test_lpu_shape_n8():
    m = LocalPerceptionUnit(dim=64)
    x = torch.randn(2, 8, 64)
    y = m(x)
    assert y.shape == (2, 8, 64)


def test_lpu_residual_is_added():
    m = LocalPerceptionUnit(dim=32)
    # Zero the depthwise conv weights → output should equal input (residual only).
    with torch.no_grad():
        m.conv.weight.zero_()
        m.conv.bias.zero_()
    x = torch.randn(1, 4, 32)
    y = m(x)
    assert torch.allclose(y, x, atol=1e-6), "with zero conv weights, LPU should equal input"


def test_lpu_param_count():
    m = LocalPerceptionUnit(dim=64, kernel_size=3)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    # depthwise conv k=3 over 64 channels = 64*1*3 + 64 = 256
    assert n == 256, f"unexpected LPU param count {n}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[ok] {name}")
    print("\n=== test_lmhsa.py: PASS ===")
