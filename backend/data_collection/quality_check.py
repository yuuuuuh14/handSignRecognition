"""Quality statistics for a captured clip.

A clip is 16 frames; each frame produces a hand_mask (2,) and face_mask scalar.
The recorder shows these stats to the user immediately after capture so they can
decide whether to keep the clip (Enter) or retry (Backspace).
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np


# Threshold below which the UI shows a red warning (per IMPLEMENTATION_PLAN §4.1).
WARN_THRESHOLD: float = 0.80


@dataclasses.dataclass
class QualityStats:
    num_frames: int
    left_hand_rate: float       # fraction of frames with Left hand detected
    right_hand_rate: float      # fraction of frames with Right hand detected
    any_hand_rate: float        # fraction with at least one hand detected
    both_hands_frames: int      # number of frames with both hands detected
    face_rate: float            # fraction of frames with face detected
    warn: bool                  # True if any of the rates fall below WARN_THRESHOLD

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def compute_quality_stats(
    hand_mask: np.ndarray | Sequence[Sequence[bool]],
    face_mask: np.ndarray | Sequence[bool],
    *,
    warn_threshold: float = WARN_THRESHOLD,
) -> QualityStats:
    """Compute per-clip detection rates.

    Args:
        hand_mask: (T, 2) boolean array — slot 0 = Left, slot 1 = Right.
        face_mask: (T,)   boolean array.
        warn_threshold: any_hand_rate or face_rate below this triggers warn=True.
    """
    hm = np.asarray(hand_mask, dtype=bool)
    fm = np.asarray(face_mask, dtype=bool)
    if hm.ndim != 2 or hm.shape[1] != 2:
        raise ValueError(f"hand_mask must be (T, 2), got {hm.shape}")
    if fm.shape != (hm.shape[0],):
        raise ValueError(f"face_mask shape {fm.shape} does not match hand_mask T={hm.shape[0]}")

    t = hm.shape[0]
    left = float(hm[:, 0].mean()) if t else 0.0
    right = float(hm[:, 1].mean()) if t else 0.0
    any_hand = float(hm.any(axis=1).mean()) if t else 0.0
    both = int(hm.all(axis=1).sum())
    face = float(fm.mean()) if t else 0.0
    warn = (any_hand < warn_threshold) or (face < warn_threshold)

    return QualityStats(
        num_frames=t,
        left_hand_rate=left,
        right_hand_rate=right,
        any_hand_rate=any_hand,
        both_hands_frames=both,
        face_rate=face,
        warn=warn,
    )
