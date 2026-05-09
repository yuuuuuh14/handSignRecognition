"""Landmark normalization (translation + scale invariance).

Per IMPLEMENTATION_PLAN §5.1:
  Hand: translate by wrist (lm 0); scale by ||wrist - middle MCP (lm 9)||.
  Face: translate by nose tip (subset idx FACE_NOSE_TIP_IDX);
        scale by mouth-corner distance.

Shapes accepted (any leading batch dims):
  hand_lm   : (..., 2, 21, 3)
  hand_mask : (..., 2)
  face_lm   : (..., N_face, 3)
  face_mask : (...,)

Slots/frames flagged as not-detected are explicitly zeroed out. The recorder
already writes zeros for missing slots, but we re-zero after normalization to
preserve that contract (subtracting wrist from a zero slot, then scaling, would
otherwise produce non-zero garbage).
"""
from __future__ import annotations

import numpy as np

from data_collection.mediapipe_runner import (
    FACE_MOUTH_LEFT_IDX,
    FACE_MOUTH_RIGHT_IDX,
    FACE_NOSE_TIP_IDX,
)

HAND_WRIST_IDX: int = 0
HAND_MIDDLE_MCP_IDX: int = 9


def _safe_scale(scale: np.ndarray, eps: float) -> np.ndarray:
    return np.where(scale < eps, np.float32(1.0), scale.astype(np.float32))


def normalize_hand_landmarks(
    hand_lm: np.ndarray,
    hand_mask: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Translate by wrist, scale by ||wrist → middle MCP||. Returns float32 array same shape."""
    lm = np.asarray(hand_lm, dtype=np.float32)
    mask = np.asarray(hand_mask, dtype=bool)
    if lm.shape[-3:] != (2, 21, 3):
        raise ValueError(f"hand_lm last dims must be (2, 21, 3); got {lm.shape}")
    if mask.shape != lm.shape[:-2]:
        raise ValueError(f"hand_mask shape {mask.shape} incompatible with hand_lm {lm.shape}")

    wrist = lm[..., HAND_WRIST_IDX:HAND_WRIST_IDX + 1, :]            # (..., 2, 1, 3)
    mid_mcp = lm[..., HAND_MIDDLE_MCP_IDX:HAND_MIDDLE_MCP_IDX + 1, :]
    out = lm - wrist
    delta_xy = (mid_mcp - wrist)[..., :2]                            # (..., 2, 1, 2)
    scale = np.linalg.norm(delta_xy, axis=-1, keepdims=True)         # (..., 2, 1, 1)
    scale = _safe_scale(scale, eps)
    out = out / scale                                                # broadcast over (21, 3)
    out = out * mask[..., None, None]                                # zero out missing slots
    return out.astype(np.float32, copy=False)


def normalize_face_landmarks(
    face_lm: np.ndarray,
    face_mask: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Translate by nose tip, scale by ||left-mouth → right-mouth||."""
    lm = np.asarray(face_lm, dtype=np.float32)
    mask = np.asarray(face_mask, dtype=bool)
    if lm.shape[-1] != 3:
        raise ValueError(f"face_lm last dim must be 3; got {lm.shape}")
    if mask.shape != lm.shape[:-2]:
        raise ValueError(f"face_mask shape {mask.shape} incompatible with face_lm {lm.shape}")

    nose = lm[..., FACE_NOSE_TIP_IDX:FACE_NOSE_TIP_IDX + 1, :]       # (..., 1, 3)
    out = lm - nose
    ml_xy = out[..., FACE_MOUTH_LEFT_IDX, :2]                        # (..., 2)
    mr_xy = out[..., FACE_MOUTH_RIGHT_IDX, :2]
    scale = np.linalg.norm(mr_xy - ml_xy, axis=-1, keepdims=True)    # (..., 1)
    scale = _safe_scale(scale, eps)[..., None]                       # (..., 1, 1)
    out = out / scale
    out = out * mask[..., None, None]
    return out.astype(np.float32, copy=False)
