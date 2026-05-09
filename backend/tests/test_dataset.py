"""Phase 4 verification: load 1 batch from real recorded data.

Validation criterion (IMPLEMENTATION_PLAN §10 Phase 4): "iter 1 batch 정상 출력".

Runs as both a pytest test and a standalone script:
    pytest tests/test_dataset.py
    python tests/test_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python tests/test_dataset.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import build_dataset
from data.splits import discover_clips, split_clips, summarize_clips
from data_collection.mediapipe_runner import (
    NUM_FACE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    NUM_HANDS,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "lab_dataset.yaml"


def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _pick_clips(cfg: dict):
    """Use signer-split if any train signer has data; otherwise fall back to all clips
    so this still works during early dev when only signer 1 is recorded but
    train_signers includes 1..8."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    clips = discover_clips(raw_dir)
    train, test = split_clips(clips, cfg["data"]["train_signers"], cfg["data"]["test_signers"])
    return clips, train, test


def test_one_batch():
    cfg = _load_cfg()
    all_clips, train, test = _pick_clips(cfg)

    if not all_clips:
        # Not a failure — early-dev path before any clip is recorded.
        print("[skip] no clips found under data/raw/ — skipping shape test")
        return

    print(f"[info] discovered {len(all_clips)} clips across "
          f"{len(set(c.signer_id for c in all_clips))} signer(s)")
    summary = summarize_clips(all_clips)
    for sid in sorted(summary):
        per = ", ".join(f"class {c}: {summary[sid][c]}" for c in sorted(summary[sid]))
        print(f"        signer {sid}: {per}")

    use_clips = train if train else all_clips
    print(f"[info] using {len(use_clips)} clip(s) (train split)" if train
          else f"[info] no train-signer data yet; using all {len(use_clips)} for shape verification")

    T = int(cfg["clip"]["num_frames"])
    H = int(cfg["input"]["hand"]["crop_size"])

    # Eval (no augment) — deterministic
    ds_eval = build_dataset(use_clips, cfg, train=False)
    item = ds_eval[0]
    assert item["hand_lm"].shape == (T, NUM_HANDS, NUM_HAND_LANDMARKS, 3), item["hand_lm"].shape
    assert item["face_lm"].shape == (T, NUM_FACE_LANDMARKS, 3), item["face_lm"].shape
    assert item["hand_crop"].shape == (T, NUM_HANDS, 3, H, H), item["hand_crop"].shape
    assert item["face_crop"].shape == (T, 3, H, H), item["face_crop"].shape
    assert item["hand_mask"].shape == (T, NUM_HANDS) and item["hand_mask"].dtype == torch.bool
    assert item["face_mask"].shape == (T,) and item["face_mask"].dtype == torch.bool
    assert item["label"].dtype == torch.long
    assert item["hand_crop"].dtype == torch.float32
    assert 0.0 <= item["hand_crop"].min().item() and item["hand_crop"].max().item() <= 1.0
    print("[ok]   eval __getitem__ shapes & dtypes")

    # Train (with augment) — make sure aug path doesn't crash on real data
    ds_train = build_dataset(use_clips, cfg, train=True)
    item_aug = ds_train[0]
    assert item_aug["hand_lm"].shape == (T, NUM_HANDS, NUM_HAND_LANDMARKS, 3)
    assert torch.isfinite(item_aug["hand_lm"]).all()
    assert torch.isfinite(item_aug["face_lm"]).all()
    print("[ok]   train __getitem__ (augment on)")

    # Determinism check: eval path should give bit-exact identical tensors on repeat calls
    item_a = ds_eval[0]
    item_b = ds_eval[0]
    for k in ("hand_lm", "face_lm", "hand_crop", "face_crop"):
        if not torch.equal(item_a[k], item_b[k]):
            raise AssertionError(f"eval path is non-deterministic for key '{k}'")
    print("[ok]   eval path is deterministic")

    # Sanity: normalized hand wrist (lm 0) should be ~0
    hand_lm = item["hand_lm"].numpy()       # (T, 2, 21, 3)
    hand_mask = item["hand_mask"].numpy()
    if hand_mask.any():
        wrist = hand_lm[hand_mask][:, 0, :]   # (N_detected, 3)
        max_wrist = float(np.abs(wrist).max())
        if max_wrist > 1e-5:
            raise AssertionError(f"wrist not at origin after normalization: |max|={max_wrist}")
        print(f"[ok]   hand wrist landmark = 0 after normalization (|max|={max_wrist:.2e})")

    # Sanity: normalized face nose (subset idx FACE_NOSE_TIP_IDX) should be ~0
    from data_collection.mediapipe_runner import FACE_NOSE_TIP_IDX
    face_lm = item["face_lm"].numpy()
    face_mask = item["face_mask"].numpy()
    if face_mask.any():
        nose = face_lm[face_mask][:, FACE_NOSE_TIP_IDX, :]
        max_nose = float(np.abs(nose).max())
        if max_nose > 1e-5:
            raise AssertionError(f"nose not at origin after normalization: |max|={max_nose}")
        print(f"[ok]   face nose tip = 0 after normalization (|max|={max_nose:.2e})")

    # 1 batch via DataLoader (Phase 4 primary verification)
    bs = min(int(cfg["train"]["batch_size"]), len(use_clips))
    loader = DataLoader(ds_eval, batch_size=bs, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    B = batch["label"].shape[0]
    assert batch["hand_lm"].shape == (B, T, NUM_HANDS, NUM_HAND_LANDMARKS, 3)
    assert batch["face_lm"].shape == (B, T, NUM_FACE_LANDMARKS, 3)
    assert batch["hand_crop"].shape == (B, T, NUM_HANDS, 3, H, H)
    assert batch["face_crop"].shape == (B, T, 3, H, H)
    assert batch["label"].shape == (B,)
    print(f"[ok]   DataLoader batch shapes (B={B})")

    print()
    print("=== Phase 4: PASS ===")


if __name__ == "__main__":
    test_one_batch()
