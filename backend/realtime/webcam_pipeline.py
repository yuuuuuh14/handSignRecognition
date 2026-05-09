"""WebcamPipeline — drives MediaPipe → FrameBuffer → KSLRNet inference.

Per IMPLEMENTATION_PLAN §9:
    - Buffer capacity = 16 frames
    - Inference runs every `stride` frames once the buffer is full
    - Output is smoothed by EMA or majority-vote (configurable)

Per-frame call sequence:
    pipeline.step_frame(frame_bgr, timestamp_ms)
        → MediaPipeRunner.process_frame
        → FrameBuffer.push
        → if buffer full and step % stride == 0:
              build model inputs (crops + normalized landmarks, batched)
              forward through KSLRNet
              softmax → top-1 + confidence
              run smoothing
        → returns (mp_result, prediction_or_None)

The pipeline does NOT own the cv2.VideoCapture or the display window — those
live in scripts/demo.py so the pipeline class stays unit-testable with
synthetic frames.
"""
from __future__ import annotations

import dataclasses
from collections import Counter, deque
from typing import Optional

import numpy as np
import torch

from data.normalizer import normalize_face_landmarks, normalize_hand_landmarks
from data_collection.mediapipe_runner import (
    NUM_HANDS,
    FrameResult,
    MediaPipeRunner,
    extract_face_crop,
    extract_hand_crops,
)
from realtime.frame_buffer import FrameBuffer, FrameSnapshot


@dataclasses.dataclass
class Prediction:
    label_id: int
    confidence: float
    raw_label_id: int                  # argmax of latest logits, before smoothing
    raw_confidence: float


# ─────────────────────────────────────────────
# Smoothers
# ─────────────────────────────────────────────
class _EMASmoother:
    def __init__(self, alpha: float, num_classes: int) -> None:
        self.alpha = float(alpha)
        self.num_classes = int(num_classes)
        self._probs: np.ndarray | None = None

    def update(self, probs: np.ndarray) -> tuple[int, float]:
        if self._probs is None:
            self._probs = probs.copy()
        else:
            self._probs = self.alpha * probs + (1.0 - self.alpha) * self._probs
        idx = int(self._probs.argmax())
        return idx, float(self._probs[idx])


class _MajoritySmoother:
    def __init__(self, window: int = 5, num_classes: int = 10) -> None:
        self.window = int(window)
        self.num_classes = int(num_classes)
        self._history: deque[tuple[int, float]] = deque(maxlen=self.window)

    def update(self, probs: np.ndarray) -> tuple[int, float]:
        idx = int(probs.argmax())
        conf = float(probs[idx])
        self._history.append((idx, conf))
        # Most common label, with mean confidence among that label's votes.
        counts = Counter(label for label, _ in self._history)
        winner, _ = counts.most_common(1)[0]
        confs = [c for label, c in self._history if label == winner]
        return winner, float(np.mean(confs)) if confs else conf


class _NoSmoother:
    def update(self, probs: np.ndarray) -> tuple[int, float]:
        idx = int(probs.argmax())
        return idx, float(probs[idx])


def _build_smoother(method: str, alpha: float, num_classes: int):
    method = (method or "none").lower()
    if method == "ema":
        return _EMASmoother(alpha=alpha, num_classes=num_classes)
    if method == "majority":
        return _MajoritySmoother(window=5, num_classes=num_classes)
    if method == "none":
        return _NoSmoother()
    raise ValueError(f"unknown smoothing method '{method}'")


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────
class WebcamPipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        mp_runner: MediaPipeRunner,
        cfg: dict,
        device: torch.device,
    ) -> None:
        self.model = model.eval().to(device)
        self.mp_runner = mp_runner
        self.cfg = cfg
        self.device = device

        self.num_frames = int(cfg["clip"]["num_frames"])
        self.stride = int(cfg["realtime"]["stride"])
        self.crop_size = int(cfg["input"]["hand"]["crop_size"])
        self.hand_crop_margin = float(cfg["input"]["hand"]["crop_margin"])
        self.face_crop_margin = float(cfg["input"]["face"]["crop_margin"])
        self.num_classes = int(cfg["data"]["num_classes"])

        self.buffer = FrameBuffer(capacity=self.num_frames)

        smoothing_cfg = cfg["realtime"]["smoothing"]
        self.smoother = _build_smoother(
            method=str(smoothing_cfg["method"]),
            alpha=float(smoothing_cfg.get("alpha", 0.6)),
            num_classes=self.num_classes,
        )

        self.step = 0
        self.last_prediction: Optional[Prediction] = None
        self.last_inference_ms: float = 0.0

    # ─────────────────────────────────────────
    def step_frame(
        self, frame_bgr: np.ndarray, timestamp_ms: int
    ) -> tuple[FrameResult, Optional[Prediction]]:
        """Process one webcam frame; return MediaPipe result + (smoothed) prediction.

        prediction is None until the buffer fills (~16 frames at 30 FPS ≈ 533ms);
        thereafter it updates every `stride` frames and is sticky in between.
        """
        result = self.mp_runner.process_frame(frame_bgr, timestamp_ms)
        self.buffer.push(
            frame=frame_bgr,
            hand_lm=result.hand_landmarks,
            face_lm=result.face_landmarks,
            hand_mask=result.hand_mask,
            face_mask=result.face_mask,
        )

        if self.buffer.is_full() and (self.step % self.stride == 0):
            self._run_inference(self.buffer.snapshot())
        self.step += 1
        return result, self.last_prediction

    # ─────────────────────────────────────────
    def _run_inference(self, snap: FrameSnapshot) -> None:
        inputs = self._build_model_inputs(snap)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(**inputs)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.last_inference_ms = (t1 - t0) * 1000.0

        raw_idx = int(probs.argmax())
        raw_conf = float(probs[raw_idx])
        smoothed_idx, smoothed_conf = self.smoother.update(probs)
        self.last_prediction = Prediction(
            label_id=smoothed_idx,
            confidence=smoothed_conf,
            raw_label_id=raw_idx,
            raw_confidence=raw_conf,
        )

    # ─────────────────────────────────────────
    def _build_model_inputs(self, snap: FrameSnapshot) -> dict[str, torch.Tensor]:
        T = self.num_frames
        H = self.crop_size

        hand_crop = np.zeros((T, NUM_HANDS, H, H, 3), dtype=np.uint8)
        face_crop = np.zeros((T, H, H, 3), dtype=np.uint8)
        for t in range(T):
            hand_crop[t] = extract_hand_crops(
                snap.frames[t], snap.hand_lm[t], snap.hand_mask[t],
                size=H, margin=self.hand_crop_margin,
            )
            face_crop[t] = extract_face_crop(
                snap.frames[t], snap.face_lm[t], bool(snap.face_mask[t]),
                size=H, margin=self.face_crop_margin,
            )

        hand_lm_n = normalize_hand_landmarks(snap.hand_lm, snap.hand_mask)
        face_lm_n = normalize_face_landmarks(snap.face_lm, snap.face_mask)

        # Add batch dim and move to device.
        hand_crop_t = (
            torch.from_numpy(hand_crop).permute(0, 1, 4, 2, 3).contiguous().float() / 255.0
        ).unsqueeze(0).to(self.device)
        face_crop_t = (
            torch.from_numpy(face_crop).permute(0, 3, 1, 2).contiguous().float() / 255.0
        ).unsqueeze(0).to(self.device)

        return {
            "hand_lm": torch.from_numpy(hand_lm_n).float().unsqueeze(0).to(self.device),
            "face_lm": torch.from_numpy(face_lm_n).float().unsqueeze(0).to(self.device),
            "hand_crop": hand_crop_t,
            "face_crop": face_crop_t,
            "hand_mask": torch.from_numpy(snap.hand_mask).bool().unsqueeze(0).to(self.device),
            "face_mask": torch.from_numpy(snap.face_mask).bool().unsqueeze(0).to(self.device),
        }
