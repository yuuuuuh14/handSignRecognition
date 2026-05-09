"""Profile parameter count and MACs of KSLRNet against the IMPLEMENTATION_PLAN budget.

Usage:
    python scripts/profile_macs.py
    python scripts/profile_macs.py --config configs/lab_dataset.yaml --batch 1

Reports:
  - Per-submodule parameter counts (full breakdown)
  - Total trainable parameters
  - Pass/fail vs the plan target (1.4M – 1.6M)
  - MACs estimate via ptflops (input dict reconstructed from config)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from models.kslr_net import KSLRNet


PLAN_TARGET_LO = 1_400_000
PLAN_TARGET_HI = 1_600_000


def _format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:6.2f}M"
    if n >= 1_000:
        return f"{n/1_000:6.1f}K"
    return f"{n:6d}"


def _module_param_table(model: torch.nn.Module) -> int:
    print(f"  {'submodule':<28s} {'# params':>10s}  {'%':>5s}")
    print(f"  {'-'*28} {'-'*10}  {'-'*5}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        if n == 0:
            continue
        pct = n / total * 100 if total else 0.0
        print(f"  {name:<28s} {_format_count(n):>10s}  {pct:>4.1f}%")
    print(f"  {'-'*28} {'-'*10}  {'-'*5}")
    print(f"  {'TOTAL':<28s} {_format_count(total):>10s}  100.0%")
    return total


def _macs_via_ptflops(model: torch.nn.Module, cfg: dict, batch: int) -> int | None:
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        print("[skip] ptflops not installed; skipping MAC estimate.")
        return None

    T = int(cfg["clip"]["num_frames"])
    H = int(cfg["input"]["hand"]["crop_size"])
    N_face = int(cfg["input"]["face"]["num_landmarks"])
    N_hand = 21

    def make_inputs(_input_res):
        return {
            "hand_lm": torch.randn(batch, T, 2, N_hand, 3),
            "face_lm": torch.randn(batch, T, N_face, 3),
            "hand_crop": torch.randn(batch, T, 2, 3, H, H),
            "face_crop": torch.randn(batch, T, 3, H, H),
        }

    macs, params = get_model_complexity_info(
        model,
        (1,),
        input_constructor=make_inputs,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    return int(macs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/lab_dataset.yaml")
    ap.add_argument("--batch", type=int, default=1, help="batch size for MAC profiling")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model = KSLRNet(cfg).eval()

    print("=" * 50)
    print("KSLRNet -- parameter breakdown")
    print("=" * 50)
    total = _module_param_table(model)

    print()
    print(f"Plan target:    {_format_count(PLAN_TARGET_LO)} -- {_format_count(PLAN_TARGET_HI)}")
    if PLAN_TARGET_LO <= total <= PLAN_TARGET_HI:
        print(f"Status:         WITHIN BUDGET [OK]")
    elif total < PLAN_TARGET_LO:
        print(f"Status:         BELOW budget by {_format_count(PLAN_TARGET_LO - total)}")
    else:
        print(f"Status:         OVER  budget by {_format_count(total - PLAN_TARGET_HI)} "
              f"({total/PLAN_TARGET_HI:.2f}x of upper bound)")

    print()
    print("=" * 50)
    print("MAC count (ptflops)")
    print("=" * 50)
    macs = _macs_via_ptflops(model, cfg, args.batch)
    if macs is not None:
        print(f"  Total MACs (B={args.batch}): {macs:,}  ~= {macs/1e9:.2f} GMACs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
