"""MediaPipe Tasks wrapper for KSLR data collection and real-time inference.

per-frame outputs (FrameResult):
    hand_landmarks  (2, 21, 3) float32  — normalized image coords; (Left, Right) slot order
    hand_world      (2, 21, 3) float32  — world coords in meters (camera frame)
    hand_mask       (2,)       bool     — True where the slot was detected
    face_landmarks  (N_face, 3) float32 — subset (FACE_OVAL ∪ LIPS ∪ EYEBROWS); zeros if missing
    face_mask       bool                — True if a face was detected

Detectors run in mp.tasks.vision.RunningMode.VIDEO. Callers must feed monotonic
timestamps (ms) per video stream and reuse the same MediaPipeRunner instance for
the full clip / webcam session.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable

import numpy as np

# mediapipe is imported eagerly: this module is only useful with it installed.
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ──────────────────────────────────────────────────────────
# Face-mesh subset (FACE_OVAL ∪ LIPS ∪ LEFT_EYEBROW ∪ RIGHT_EYEBROW + nose tip)
#
# Indices come from the Tasks-API constant
# `mp.tasks.python.vision.FaceLandmarksConnections`. Each entry is a
# Connection(start=int, end=int) namedtuple-like; we union the endpoint
# indices of the four chosen contour groups, plus landmark 1 (nose tip)
# explicitly so that face normalization can use a nose-relative origin
# (per IMPLEMENTATION_PLAN §5.1).
# ──────────────────────────────────────────────────────────
# Reference landmarks used by data/normalizer.py (canonical MediaPipe
# Face Mesh indices, BEFORE subsetting):
#   1   — nose tip            (origin for face normalization)
#   61  — left mouth corner   (paired with 291 for scale)
#   291 — right mouth corner
_FACE_NOSE_TIP_RAW: int = 1
_FACE_MOUTH_LEFT_RAW: int = 61
_FACE_MOUTH_RIGHT_RAW: int = 291


def _connection_indices(connections: Iterable) -> set[int]:
    out: set[int] = set()
    for c in connections:
        out.add(int(c.start))
        out.add(int(c.end))
    return out


_FLC = mp_vision.FaceLandmarksConnections
FACE_SUBSET_INDICES: list[int] = sorted(
    _connection_indices(_FLC.FACE_LANDMARKS_FACE_OVAL)
    | _connection_indices(_FLC.FACE_LANDMARKS_LIPS)
    | _connection_indices(_FLC.FACE_LANDMARKS_LEFT_EYEBROW)
    | _connection_indices(_FLC.FACE_LANDMARKS_RIGHT_EYEBROW)
    | {_FACE_NOSE_TIP_RAW}
)
NUM_FACE_LANDMARKS: int = len(FACE_SUBSET_INDICES)
NUM_HAND_LANDMARKS: int = 21
NUM_HANDS: int = 2

# Positions of normalization-reference landmarks within the (NUM_FACE_LANDMARKS,)
# subset array. data/normalizer.py uses these instead of the raw MediaPipe
# indices, since the recorder only stores the subset on disk.
FACE_NOSE_TIP_IDX: int = FACE_SUBSET_INDICES.index(_FACE_NOSE_TIP_RAW)
FACE_MOUTH_LEFT_IDX: int = FACE_SUBSET_INDICES.index(_FACE_MOUTH_LEFT_RAW)
FACE_MOUTH_RIGHT_IDX: int = FACE_SUBSET_INDICES.index(_FACE_MOUTH_RIGHT_RAW)

# Slot convention: slot 0 = Left hand, slot 1 = Right hand (per MediaPipe handedness label).
_HANDEDNESS_TO_SLOT = {"Left": 0, "Right": 1}


@dataclasses.dataclass
class FrameResult:
    hand_landmarks: np.ndarray   # (2, 21, 3) float32
    hand_world: np.ndarray       # (2, 21, 3) float32
    hand_mask: np.ndarray        # (2,) bool
    face_landmarks: np.ndarray   # (N_face, 3) float32
    face_mask: bool


class MediaPipeRunner:
    """Stateful per-video Hand+Face landmarker. Use as context manager."""

    def __init__(
        self,
        hand_model_path: str | Path,
        face_model_path: str | Path,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_faces: int = 1,
    ) -> None:
        hand_model_path = Path(hand_model_path)
        face_model_path = Path(face_model_path)
        if not hand_model_path.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found: {hand_model_path}. "
                f"Run `python scripts/download_models.py`."
            )
        if not face_model_path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found: {face_model_path}. "
                f"Run `python scripts/download_models.py`."
            )

        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(face_model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._hand = mp_vision.HandLandmarker.create_from_options(hand_opts)
        self._face = mp_vision.FaceLandmarker.create_from_options(face_opts)

    def __enter__(self) -> "MediaPipeRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for attr in ("_hand", "_face"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.close()
                setattr(self, attr, None)

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FrameResult:
        """Run hand + face detection on a single OpenCV BGR frame.

        Args:
            frame_bgr: (H, W, 3) uint8 BGR image (cv2.VideoCapture format).
            timestamp_ms: Monotonic timestamp in milliseconds. Must strictly increase
                within a single MediaPipeRunner instance (VIDEO mode requirement).
        """
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"frame_bgr must be (H,W,3), got {frame_bgr.shape}")

        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_res = self._hand.detect_for_video(mp_image, timestamp_ms)
        face_res = self._face.detect_for_video(mp_image, timestamp_ms)

        hand_landmarks = np.zeros((NUM_HANDS, NUM_HAND_LANDMARKS, 3), dtype=np.float32)
        hand_world = np.zeros((NUM_HANDS, NUM_HAND_LANDMARKS, 3), dtype=np.float32)
        hand_mask = np.zeros((NUM_HANDS,), dtype=bool)

        for i, lms in enumerate(hand_res.hand_landmarks or []):
            cat = hand_res.handedness[i][0] if hand_res.handedness else None
            slot = _HANDEDNESS_TO_SLOT.get(cat.category_name if cat else "", -1)
            if slot < 0:
                continue
            if hand_mask[slot]:
                # Same handedness label appeared twice (rare): try to use the empty slot.
                other = 1 - slot
                if hand_mask[other]:
                    continue
                slot = other
            hand_landmarks[slot] = np.asarray(
                [(p.x, p.y, p.z) for p in lms], dtype=np.float32
            )
            world_lms = (hand_res.hand_world_landmarks or [])
            if i < len(world_lms):
                hand_world[slot] = np.asarray(
                    [(p.x, p.y, p.z) for p in world_lms[i]], dtype=np.float32
                )
            hand_mask[slot] = True

        face_landmarks = np.zeros((NUM_FACE_LANDMARKS, 3), dtype=np.float32)
        face_mask = False
        if face_res.face_landmarks:
            full = face_res.face_landmarks[0]
            full_arr = np.asarray([(p.x, p.y, p.z) for p in full], dtype=np.float32)
            face_landmarks = full_arr[FACE_SUBSET_INDICES]
            face_mask = True

        return FrameResult(
            hand_landmarks=hand_landmarks,
            hand_world=hand_world,
            hand_mask=hand_mask,
            face_landmarks=face_landmarks,
            face_mask=face_mask,
        )


# ──────────────────────────────────────────────────────────
# Crop extraction (used by data/dataset.py and realtime/webcam_pipeline.py)
# ──────────────────────────────────────────────────────────
def _square_bbox(
    landmarks_xy: np.ndarray,        # (N, 2) normalized [0,1]
    frame_hw: tuple[int, int],
    margin: float,
) -> tuple[int, int, int, int]:
    h, w = frame_hw
    xs = landmarks_xy[:, 0] * w
    ys = landmarks_xy[:, 1] * h
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * margin)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    half = side * 0.5
    return (
        max(0, int(round(cx - half))),
        max(0, int(round(cy - half))),
        min(w, int(round(cx + half))),
        min(h, int(round(cy + half))),
    )


def extract_hand_crops(
    frame_bgr: np.ndarray,
    hand_landmarks: np.ndarray,    # (2, 21, 3) normalized
    hand_mask: np.ndarray,         # (2,) bool
    size: int = 64,
    margin: float = 0.25,
) -> np.ndarray:
    """Returns (2, size, size, 3) uint8 BGR. Missing/invalid hands → zeros."""
    import cv2  # local import: cv2 is heavy and not all callers need it
    out = np.zeros((NUM_HANDS, size, size, 3), dtype=np.uint8)
    h, w = frame_bgr.shape[:2]
    for i in range(NUM_HANDS):
        if not hand_mask[i]:
            continue
        x0, y0, x1, y1 = _square_bbox(hand_landmarks[i, :, :2], (h, w), margin)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        out[i] = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return out


def extract_face_crop(
    frame_bgr: np.ndarray,
    face_landmarks: np.ndarray,    # (N_face, 3) normalized
    face_mask: bool,
    size: int = 64,
    margin: float = 0.15,
) -> np.ndarray:
    """Returns (size, size, 3) uint8 BGR. Missing face → zeros."""
    import cv2
    if not face_mask:
        return np.zeros((size, size, 3), dtype=np.uint8)
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = _square_bbox(face_landmarks[:, :2], (h, w), margin)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
