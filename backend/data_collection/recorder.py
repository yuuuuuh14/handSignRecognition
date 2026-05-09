"""Clip recorder: 16-frame capture state machine + on-disk save.

UI-agnostic. The OpenCV window and keyboard handling live in scripts/record.py;
this module owns capture state, buffer, and persistence so it can be unit-tested
without a webcam.

State machine
─────────────
    IDLE       ── user presses [0–9] ──> IDLE (class selected)
    IDLE       ── user presses Space  ──> CAPTURING (only if class selected)
    CAPTURING  ── 16th frame pushed   ──> REVIEW
    REVIEW     ── user presses Enter  ──> save → IDLE
    REVIEW     ── user presses Bksp   ──> discard → IDLE

Storage layout (per IMPLEMENTATION_PLAN §4.2)
─────────────────────────────────────────────
    data/raw/{signer_id}/{class_id}/{YYYYMMDD_HHMMSS_mmm}/
        frames.npy            (T, H, W, 3) uint8
        hand_landmarks.npy    (T, 2, 21, 3) float32
        hand_world.npy        (T, 2, 21, 3) float32
        face_landmarks.npy    (T, N_face, 3) float32
        hand_mask.npy         (T, 2) bool
        face_mask.npy         (T,)  bool
        meta.json             {signer_id, class_id, timestamp, fps, vocabulary_version}
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from .mediapipe_runner import FrameResult
from .quality_check import QualityStats, compute_quality_stats


class RecState(str, Enum):
    IDLE = "idle"
    CAPTURING = "capturing"
    REVIEW = "review"


@dataclasses.dataclass
class _Buffer:
    frames: list[np.ndarray] = dataclasses.field(default_factory=list)
    hand_landmarks: list[np.ndarray] = dataclasses.field(default_factory=list)
    hand_world: list[np.ndarray] = dataclasses.field(default_factory=list)
    hand_mask: list[np.ndarray] = dataclasses.field(default_factory=list)
    face_landmarks: list[np.ndarray] = dataclasses.field(default_factory=list)
    face_mask: list[bool] = dataclasses.field(default_factory=list)


class ClipRecorder:
    """Drive a recording session for one signer."""

    def __init__(
        self,
        signer_id: int,
        save_root: Path | str,
        num_frames: int = 16,
        fps: int = 30,
        vocabulary_version: int = 0,
        num_classes: int = 10,
    ) -> None:
        self.signer_id = int(signer_id)
        self.save_root = Path(save_root)
        self.num_frames = int(num_frames)
        self.fps = int(fps)
        self.vocabulary_version = int(vocabulary_version)
        self.num_classes = int(num_classes)

        self.state: RecState = RecState.IDLE
        self.current_class_id: Optional[int] = None
        self._buffer: Optional[_Buffer] = None
        self.last_quality: Optional[QualityStats] = None
        self.last_save_path: Optional[Path] = None
        self.saved_count: dict[int, int] = {c: 0 for c in range(num_classes)}

    # ───────── state queries ─────────

    @property
    def progress(self) -> float:
        if self.state != RecState.CAPTURING or self._buffer is None:
            return 0.0
        return min(1.0, len(self._buffer.frames) / self.num_frames)

    @property
    def captured_frame_count(self) -> int:
        return 0 if self._buffer is None else len(self._buffer.frames)

    # ───────── transitions ─────────

    def select_class(self, class_id: int) -> bool:
        if self.state != RecState.IDLE:
            return False
        if not (0 <= class_id < self.num_classes):
            return False
        self.current_class_id = class_id
        return True

    def start_capture(self) -> bool:
        if self.state != RecState.IDLE or self.current_class_id is None:
            return False
        self._buffer = _Buffer()
        self.state = RecState.CAPTURING
        return True

    def push_frame(self, frame_bgr: np.ndarray, result: FrameResult) -> None:
        """Add one frame to the active capture; auto-finalize at num_frames."""
        if self.state != RecState.CAPTURING or self._buffer is None:
            return
        self._buffer.frames.append(frame_bgr.copy())
        self._buffer.hand_landmarks.append(result.hand_landmarks.copy())
        self._buffer.hand_world.append(result.hand_world.copy())
        self._buffer.hand_mask.append(result.hand_mask.copy())
        self._buffer.face_landmarks.append(result.face_landmarks.copy())
        self._buffer.face_mask.append(bool(result.face_mask))
        if len(self._buffer.frames) >= self.num_frames:
            self._finalize_capture()

    def _finalize_capture(self) -> None:
        assert self._buffer is not None
        self.last_quality = compute_quality_stats(
            np.stack(self._buffer.hand_mask, axis=0),
            np.array(self._buffer.face_mask, dtype=bool),
        )
        self.state = RecState.REVIEW

    def commit(self) -> Optional[Path]:
        if self.state != RecState.REVIEW or self._buffer is None:
            return None
        path = self._save_clip(self._buffer)
        self.last_save_path = path
        self.saved_count[self.current_class_id] = self.saved_count.get(self.current_class_id, 0) + 1
        self._buffer = None
        self.state = RecState.IDLE
        return path

    def discard(self) -> None:
        if self.state != RecState.REVIEW:
            return
        self._buffer = None
        self.last_quality = None
        self.state = RecState.IDLE

    # ───────── persistence ─────────

    def _save_clip(self, buf: _Buffer) -> Path:
        assert self.current_class_id is not None
        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
        out_dir = self.save_root / str(self.signer_id) / str(self.current_class_id) / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "frames.npy", np.stack(buf.frames, axis=0))
        np.save(out_dir / "hand_landmarks.npy", np.stack(buf.hand_landmarks, axis=0))
        np.save(out_dir / "hand_world.npy", np.stack(buf.hand_world, axis=0))
        np.save(out_dir / "face_landmarks.npy", np.stack(buf.face_landmarks, axis=0))
        np.save(out_dir / "hand_mask.npy", np.stack(buf.hand_mask, axis=0))
        np.save(out_dir / "face_mask.npy", np.array(buf.face_mask, dtype=bool))

        meta = {
            "signer_id": self.signer_id,
            "class_id": self.current_class_id,
            "timestamp": ts,
            "fps": self.fps,
            "vocabulary_version": self.vocabulary_version,
            "num_frames": len(buf.frames),
            "quality": self.last_quality.to_dict() if self.last_quality else None,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out_dir
