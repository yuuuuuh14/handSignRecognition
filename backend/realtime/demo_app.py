"""Overlay rendering for the real-time KSLR demo.

Pure functions over a (BGR) frame + state — no I/O, no cv2 window. Lets the
demo CLI script orchestrate the camera/window loop while this module owns
visual layout.

Layout:
    [top-left]    Signer-facing status: device, FPS, infer ms
    [top-right]   Buffer fill progress (until first prediction)
    [bottom]      Predicted label + confidence bar
                  Top-3 candidates list (for debugging signs that look alike)
"""
from __future__ import annotations

import cv2
import numpy as np

from data_collection.mediapipe_runner import FrameResult
from realtime.webcam_pipeline import Prediction
from utils.text_overlay import KoreanTextRenderer


# Colors (BGR — OpenCV)
_C_LEFT_HAND = (0, 255, 0)
_C_RIGHT_HAND = (0, 200, 255)
_C_FACE = (255, 0, 255)
_C_OK = (0, 255, 0)
_C_WARN = (0, 165, 255)        # orange
_C_LOW = (0, 0, 255)           # red
_C_TEXT = (255, 255, 255)
_C_BAR_BG = (60, 60, 60)


def draw_landmarks(img: np.ndarray, result: FrameResult) -> None:
    """In-place: draw hand + face landmarks as small dots."""
    h, w = img.shape[:2]
    for i, color in enumerate((_C_LEFT_HAND, _C_RIGHT_HAND)):
        if not result.hand_mask[i]:
            continue
        for x, y, _ in result.hand_landmarks[i]:
            cv2.circle(img, (int(x * w), int(y * h)), 2, color, -1)
    if result.face_mask:
        for x, y, _ in result.face_landmarks:
            cv2.circle(img, (int(x * w), int(y * h)), 1, _C_FACE, -1)


def draw_progress_bar(
    img: np.ndarray,
    fraction: float,
    org: tuple[int, int],
    size: tuple[int, int],
    fill_color: tuple[int, int, int] = _C_OK,
) -> None:
    x, y = org
    w, h = size
    cv2.rectangle(img, (x, y), (x + w, y + h), _C_BAR_BG, -1)
    fw = int(w * max(0.0, min(1.0, fraction)))
    if fw > 0:
        cv2.rectangle(img, (x, y), (x + fw, y + h), fill_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), _C_TEXT, 1)


def _conf_color(confidence: float) -> tuple[int, int, int]:
    if confidence >= 0.7:
        return _C_OK
    if confidence >= 0.4:
        return _C_WARN
    return _C_LOW


def render_demo_frame(
    frame_bgr: np.ndarray,
    result: FrameResult,
    prediction: Prediction | None,
    *,
    vocabulary: dict[int, str],
    renderer: KoreanTextRenderer,
    fps: float,
    inference_ms: float,
    buffer_fill: int,
    buffer_capacity: int,
    top3: list[tuple[int, float]] | None = None,
) -> np.ndarray:
    """Return an overlay-decorated copy of `frame_bgr`."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    draw_landmarks(out, result)

    # ── Top-left status ─────────────────────────────────────
    renderer.text(f"FPS {fps:5.1f}", (10, 10), color=_C_TEXT, small=True)
    if inference_ms > 0:
        renderer.text(f"infer {inference_ms:5.1f} ms", (10, 32), color=_C_TEXT, small=True)

    # ── Buffer fill (top-right) until first prediction ──────
    if prediction is None:
        bar_w, bar_h = 180, 14
        bar_x, bar_y = w - bar_w - 10, 12
        frac = buffer_fill / max(1, buffer_capacity)
        draw_progress_bar(out, frac, (bar_x, bar_y), (bar_w, bar_h), fill_color=_C_WARN)
        renderer.text(
            f"Warming up... {buffer_fill}/{buffer_capacity}",
            (bar_x, bar_y + bar_h + 6),
            color=_C_WARN, small=True,
        )

    # ── Bottom: predicted label + confidence ────────────────
    if prediction is not None:
        label = vocabulary.get(prediction.label_id, f"class_{prediction.label_id}")
        conf = prediction.confidence
        c = _conf_color(conf)

        # Big label
        renderer.text(label, (20, h - 90), color=c, small=False)

        # Confidence bar
        bar_w, bar_h = w - 40, 18
        bar_x, bar_y = 20, h - 50
        draw_progress_bar(out, conf, (bar_x, bar_y), (bar_w, bar_h), fill_color=c)
        renderer.text(
            f"confidence {conf*100:5.1f}%   class id {prediction.label_id}",
            (bar_x, bar_y + bar_h + 4),
            color=_C_TEXT, small=True,
        )

        # Top-3 (debug aid)
        if top3:
            top3_str = "  ".join(
                f"{vocabulary.get(i, f'c{i}')}:{p*100:.0f}%" for i, p in top3
            )
            renderer.text(f"top-3: {top3_str}", (20, 60), color=_C_TEXT, small=True)

    # ── Footer hint ─────────────────────────────────────────
    renderer.text("[q]=quit   [r]=reset buffer", (10, h - 14), color=_C_TEXT, small=True)

    return renderer.flush(out)


def topk_from_probs(probs: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    if k <= 0 or probs.size == 0:
        return []
    k = min(k, probs.size)
    idx = np.argsort(-probs)[:k]
    return [(int(i), float(probs[i])) for i in idx]
