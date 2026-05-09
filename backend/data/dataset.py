"""ClipDataset: torch Dataset over recorded clips on disk.

Per IMPLEMENTATION_PLAN §5.3, each item returns:
    {
        'hand_lm':    Tensor (T, 2, 21, 3)  float32  — wrist-relative normalized
        'face_lm':    Tensor (T, N_face, 3) float32  — nose-relative normalized
        'hand_crop':  Tensor (T, 2, 3, H, W) float32 in [0,1]
        'face_crop':  Tensor (T, 3, H, W)    float32 in [0,1]
        'hand_mask':  Tensor (T, 2)          bool
        'face_mask':  Tensor (T,)            bool
        'label':      Tensor scalar          int64
    }

Crops are computed on-the-fly from the saved frames + saved (un-normalized)
landmarks via mediapipe_runner.extract_*_crops, so the recorder doesn't need
to write crop tensors to disk. Augmentations (when enabled) are sampled fresh
each call, so DataLoader workers naturally produce independent randomness.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data_collection.mediapipe_runner import (
    NUM_FACE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    NUM_HANDS,
    extract_face_crop,
    extract_hand_crops,
)

from .augment import (
    AugmentConfig,
    augment_landmarks,
    color_jitter_crop,
    sample_clip_params,
    time_jitter_window,
)
from .normalizer import normalize_face_landmarks, normalize_hand_landmarks
from .splits import ClipPath


class ClipDataset(Dataset):
    def __init__(
        self,
        clips: Sequence[ClipPath],
        num_frames: int = 16,
        crop_size: int = 64,
        hand_crop_margin: float = 0.25,
        face_crop_margin: float = 0.15,
        augment: Optional[AugmentConfig] = None,
    ) -> None:
        self.clips: list[ClipPath] = list(clips)
        self.num_frames = int(num_frames)
        self.crop_size = int(crop_size)
        self.hand_crop_margin = float(hand_crop_margin)
        self.face_crop_margin = float(face_crop_margin)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.clips)

    # ───────── disk loading ─────────

    def _load_clip(self, cp: ClipPath) -> dict[str, np.ndarray]:
        d = cp.path
        return {
            "frames": np.load(d / "frames.npy"),
            "hand_lm": np.load(d / "hand_landmarks.npy"),
            "face_lm": np.load(d / "face_landmarks.npy"),
            "hand_mask": np.load(d / "hand_mask.npy"),
            "face_mask": np.load(d / "face_mask.npy"),
        }

    # ───────── per-item pipeline ─────────

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cp = self.clips[idx]
        raw = self._load_clip(cp)

        frames = raw["frames"]
        hand_lm = raw["hand_lm"].astype(np.float32, copy=False)
        face_lm = raw["face_lm"].astype(np.float32, copy=False)
        hand_mask = raw["hand_mask"].astype(bool, copy=False)
        face_mask = raw["face_mask"].astype(bool, copy=False)

        self._validate_shapes(cp, frames, hand_lm, face_lm, hand_mask, face_mask)

        rng = np.random.default_rng() if self.augment is not None else None

        # Time jitter (no-op when stored T == target T)
        s, e = time_jitter_window(
            num_frames_stored=frames.shape[0],
            num_frames_target=self.num_frames,
            max_shift=self.augment.time_jitter_frames if self.augment else 0,
            rng=rng,
        )
        frames = frames[s:e]
        hand_lm = hand_lm[s:e]
        face_lm = face_lm[s:e]
        hand_mask = hand_mask[s:e]
        face_mask = face_mask[s:e]

        # Crops use *un-normalized* landmark coords (those are still in [0,1]
        # MediaPipe normalized image space — what extract_* expects).
        hand_crop = self._extract_hand_crops(frames, hand_lm, hand_mask)
        face_crop = self._extract_face_crops(frames, face_lm, face_mask)

        # Normalize landmarks (wrist-relative / nose-relative)
        hand_lm = normalize_hand_landmarks(hand_lm, hand_mask)
        face_lm = normalize_face_landmarks(face_lm, face_mask)

        # Augment with shared clip-level params
        if self.augment is not None and rng is not None:
            params = sample_clip_params(self.augment, rng)
            hand_lm = augment_landmarks(hand_lm, self.augment, params, rng)
            face_lm = augment_landmarks(face_lm, self.augment, params, rng)
            hand_lm = hand_lm * hand_mask[..., None, None]   # re-zero masked slots
            face_lm = face_lm * face_mask[..., None, None]
            hand_crop = color_jitter_crop(hand_crop, params)
            face_crop = color_jitter_crop(face_crop, params)

        # Crops uint8 (T,...,H,W,3) → float32 (T,...,3,H,W) ∈ [0,1]
        hand_crop_t = (
            torch.from_numpy(hand_crop).permute(0, 1, 4, 2, 3).contiguous().float() / 255.0
        )
        face_crop_t = (
            torch.from_numpy(face_crop).permute(0, 3, 1, 2).contiguous().float() / 255.0
        )

        return {
            "hand_lm": torch.from_numpy(np.ascontiguousarray(hand_lm)).float(),
            "face_lm": torch.from_numpy(np.ascontiguousarray(face_lm)).float(),
            "hand_crop": hand_crop_t,
            "face_crop": face_crop_t,
            "hand_mask": torch.from_numpy(hand_mask),
            "face_mask": torch.from_numpy(face_mask),
            "label": torch.tensor(int(cp.class_id), dtype=torch.long),
        }

    # ───────── helpers ─────────

    def _validate_shapes(
        self,
        cp: ClipPath,
        frames: np.ndarray,
        hand_lm: np.ndarray,
        face_lm: np.ndarray,
        hand_mask: np.ndarray,
        face_mask: np.ndarray,
    ) -> None:
        T = frames.shape[0]
        if T < self.num_frames:
            raise ValueError(f"{cp.path}: stored T={T} < target {self.num_frames}")
        if frames.ndim != 4 or frames.shape[3] != 3:
            raise ValueError(f"{cp.path}: frames shape {frames.shape} not (T,H,W,3)")
        if hand_lm.shape != (T, NUM_HANDS, NUM_HAND_LANDMARKS, 3):
            raise ValueError(
                f"{cp.path}: hand_lm shape {hand_lm.shape} != "
                f"({T}, {NUM_HANDS}, {NUM_HAND_LANDMARKS}, 3)"
            )
        if face_lm.shape != (T, NUM_FACE_LANDMARKS, 3):
            raise ValueError(
                f"{cp.path}: face_lm shape {face_lm.shape} != ({T}, {NUM_FACE_LANDMARKS}, 3). "
                f"Was this clip recorded with an older face subset? Re-record after "
                f"updating data_collection.mediapipe_runner."
            )
        if hand_mask.shape != (T, NUM_HANDS):
            raise ValueError(f"{cp.path}: hand_mask shape {hand_mask.shape} != ({T}, {NUM_HANDS})")
        if face_mask.shape != (T,):
            raise ValueError(f"{cp.path}: face_mask shape {face_mask.shape} != ({T},)")

    def _extract_hand_crops(
        self, frames: np.ndarray, hand_lm: np.ndarray, hand_mask: np.ndarray
    ) -> np.ndarray:
        T = frames.shape[0]
        out = np.zeros((T, NUM_HANDS, self.crop_size, self.crop_size, 3), dtype=np.uint8)
        for t in range(T):
            out[t] = extract_hand_crops(
                frames[t], hand_lm[t], hand_mask[t],
                size=self.crop_size, margin=self.hand_crop_margin,
            )
        return out

    def _extract_face_crops(
        self, frames: np.ndarray, face_lm: np.ndarray, face_mask: np.ndarray
    ) -> np.ndarray:
        T = frames.shape[0]
        out = np.zeros((T, self.crop_size, self.crop_size, 3), dtype=np.uint8)
        for t in range(T):
            out[t] = extract_face_crop(
                frames[t], face_lm[t], bool(face_mask[t]),
                size=self.crop_size, margin=self.face_crop_margin,
            )
        return out


# ──────────────────────────────────────────────────────────
# Builder helper — used by trainer/test scripts to construct from yaml config
# ──────────────────────────────────────────────────────────
def build_augment_config(cfg: dict) -> AugmentConfig:
    aug_cfg = cfg["augment"]
    return AugmentConfig(
        landmark_noise_sigma=float(aug_cfg["landmark_noise_sigma"]),
        rotation_deg=float(aug_cfg["rotation_deg"]),
        scale_range=tuple(aug_cfg["scale_range"]),
        time_jitter_frames=int(aug_cfg["time_jitter_frames"]),
        color_jitter_brightness=float(aug_cfg["color_jitter"]["brightness"]),
        color_jitter_contrast=float(aug_cfg["color_jitter"]["contrast"]),
    )


def build_dataset(
    clips: Sequence[ClipPath],
    cfg: dict,
    *,
    train: bool,
) -> ClipDataset:
    return ClipDataset(
        clips=clips,
        num_frames=int(cfg["clip"]["num_frames"]),
        crop_size=int(cfg["input"]["hand"]["crop_size"]),
        hand_crop_margin=float(cfg["input"]["hand"]["crop_margin"]),
        face_crop_margin=float(cfg["input"]["face"]["crop_margin"]),
        augment=build_augment_config(cfg) if (train and cfg["augment"]["enabled"]) else None,
    )
