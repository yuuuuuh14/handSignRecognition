"""Train-time augmentation for landmark + crop streams.

Per IMPLEMENTATION_PLAN §5.2:
  - Landmark Gaussian noise (σ=0.01, on normalized coords)
  - Random rotation ±15°    — shared by all landmarks (hand + face) within one clip
  - Random scale ×[0.9,1.1] — shared by all landmarks within one clip
  - Time jitter ±2 frames   — only meaningful when frames.npy stores more than 16
  - ColorJitter(brightness=0.2, contrast=0.2) on hand and face crops
  - Horizontal flip is forbidden (would swap left/right hand semantically)

The clip-level params (rotation, scale, color jitter) are sampled ONCE per
__getitem__ call via `sample_clip_params(...)` and then applied to both hand
and face streams so the modalities stay geometrically consistent.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np


@dataclasses.dataclass
class AugmentConfig:
    landmark_noise_sigma: float = 0.01
    rotation_deg: float = 15.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    time_jitter_frames: int = 2
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2


@dataclasses.dataclass
class ClipAugParams:
    rotation_rad: float
    scale: float
    brightness: float
    contrast: float


def sample_clip_params(cfg: AugmentConfig, rng: np.random.Generator) -> ClipAugParams:
    return ClipAugParams(
        rotation_rad=math.radians(float(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))),
        scale=float(rng.uniform(*cfg.scale_range)),
        brightness=float(rng.uniform(1.0 - cfg.color_jitter_brightness,
                                     1.0 + cfg.color_jitter_brightness)),
        contrast=float(rng.uniform(1.0 - cfg.color_jitter_contrast,
                                   1.0 + cfg.color_jitter_contrast)),
    )


def _rotation_matrix_xy(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def augment_landmarks(
    lm: np.ndarray,                # (..., D) where D ∈ {2, 3} ; xy in last 2 channels
    cfg: AugmentConfig,
    params: ClipAugParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply per-element noise + clip-level rotation + clip-level scale.

    Z (depth) channel, if present, is scaled but not rotated (rotation only acts
    in the image plane).
    """
    out = np.asarray(lm, dtype=np.float32).copy()
    if cfg.landmark_noise_sigma > 0:
        noise = rng.normal(0.0, cfg.landmark_noise_sigma, size=out.shape).astype(np.float32)
        out = out + noise
    R = _rotation_matrix_xy(params.rotation_rad)
    out[..., :2] = out[..., :2] @ R.T
    out = out * params.scale
    return out


def color_jitter_crop(crop: np.ndarray, params: ClipAugParams) -> np.ndarray:
    """Apply contrast then brightness to a uint8 image stack (..., H, W, 3).

    Returns same shape uint8. Mean is computed over (H, W, C) per crop tile so
    the mean is consistent across frames within the same crop stream.
    """
    if crop.dtype != np.uint8:
        raise ValueError(f"expected uint8 crop, got {crop.dtype}")
    f = crop.astype(np.float32)
    mean = f.mean(axis=(-3, -2, -1), keepdims=True)
    f = (f - mean) * params.contrast + mean      # contrast around the mean
    f = f * params.brightness                    # brightness multiplier
    return np.clip(f, 0.0, 255.0).astype(np.uint8)


def time_jitter_window(num_frames_stored: int, num_frames_target: int,
                       max_shift: int, rng: np.random.Generator | None) -> tuple[int, int]:
    """Pick a [start, start+target) window from a stored clip.

    - If stored == target: returns (0, target). max_shift is ignored.
    - If stored > target with rng=None (eval): center-crop.
    - If stored > target with rng provided  (train): center ± uniform[-shift, +shift]
      clamped to valid range.
    """
    if num_frames_stored < num_frames_target:
        raise ValueError(
            f"clip has {num_frames_stored} frames, less than target {num_frames_target}"
        )
    if num_frames_stored == num_frames_target:
        return 0, num_frames_target
    headroom = num_frames_stored - num_frames_target
    center = headroom // 2
    if rng is None:
        start = center
    else:
        shift = int(rng.integers(-max_shift, max_shift + 1))
        start = max(0, min(headroom, center + shift))
    return start, start + num_frames_target
