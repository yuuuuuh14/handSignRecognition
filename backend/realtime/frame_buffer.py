"""Circular frame buffer for real-time inference.

Holds the most recent `capacity` frames and their MediaPipe outputs, so the
inference pipeline can take a snapshot of (T, ...) tensors at every stride
step. Once the buffer is full, push() rotates out the oldest entry.

Stored arrays per slot (matching the recorder's on-disk schema, sans
hand_world which the model does not consume):

    frame      : (H, W, 3) uint8
    hand_lm    : (2, 21, 3) float32  — MediaPipe normalized image coords
    face_lm    : (N_face, 3) float32 — MediaPipe normalized image coords
    hand_mask  : (2,) bool
    face_mask  : ()  bool
"""
from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np


class FrameSnapshot(NamedTuple):
    frames: np.ndarray       # (T, H, W, 3) uint8
    hand_lm: np.ndarray      # (T, 2, 21, 3) float32
    face_lm: np.ndarray      # (T, N_face, 3) float32
    hand_mask: np.ndarray    # (T, 2) bool
    face_mask: np.ndarray    # (T,) bool


class FrameBuffer:
    def __init__(self, capacity: int = 16) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._frames: deque[np.ndarray] = deque(maxlen=self.capacity)
        self._hand_lm: deque[np.ndarray] = deque(maxlen=self.capacity)
        self._face_lm: deque[np.ndarray] = deque(maxlen=self.capacity)
        self._hand_mask: deque[np.ndarray] = deque(maxlen=self.capacity)
        self._face_mask: deque[bool] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._frames)

    def is_full(self) -> bool:
        return len(self._frames) == self.capacity

    def clear(self) -> None:
        self._frames.clear()
        self._hand_lm.clear()
        self._face_lm.clear()
        self._hand_mask.clear()
        self._face_mask.clear()

    def push(
        self,
        frame: np.ndarray,
        hand_lm: np.ndarray,
        face_lm: np.ndarray,
        hand_mask: np.ndarray,
        face_mask: bool,
    ) -> None:
        # Defensive copies (frames stay in the buffer for crop extraction at
        # snapshot time; if the caller reuses the same array, we'd get a torn
        # view).
        self._frames.append(np.ascontiguousarray(frame))
        self._hand_lm.append(np.ascontiguousarray(hand_lm, dtype=np.float32))
        self._face_lm.append(np.ascontiguousarray(face_lm, dtype=np.float32))
        self._hand_mask.append(np.asarray(hand_mask, dtype=bool).copy())
        self._face_mask.append(bool(face_mask))

    def snapshot(self) -> FrameSnapshot:
        if not self.is_full():
            raise RuntimeError(
                f"buffer not full: {len(self._frames)}/{self.capacity} frames"
            )
        return FrameSnapshot(
            frames=np.stack(self._frames, axis=0),
            hand_lm=np.stack(self._hand_lm, axis=0),
            face_lm=np.stack(self._face_lm, axis=0),
            hand_mask=np.stack(self._hand_mask, axis=0),
            face_mask=np.array(self._face_mask, dtype=bool),
        )
