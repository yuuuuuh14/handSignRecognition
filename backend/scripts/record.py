"""Webcam recorder GUI for KSLR data collection.

Usage:
    python scripts/record.py --signer 1
    python scripts/record.py --signer 1 --config configs/lab_dataset.yaml --camera 0

Keyboard (per IMPLEMENTATION_PLAN §4.1):
    1 .. 9 , 0   select class id (key '1' → class 0, ..., key '0' → class 9)
    Space        start a 16-frame capture (only when a class is selected)
    Enter        save the just-captured clip
    Backspace    discard and retry
    q            quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow `python scripts/record.py ...` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from data_collection.mediapipe_runner import MediaPipeRunner, NUM_HANDS
from data_collection.recorder import ClipRecorder, RecState
from data_collection.quality_check import WARN_THRESHOLD


# Key mapping: '1'..'9' → class 0..8 ; '0' → class 9.
_KEY_TO_CLASS: dict[int, int] = {ord(str(i)): i - 1 for i in range(1, 10)}
_KEY_TO_CLASS[ord("0")] = 9

KEY_ENTER = 13
KEY_BACKSPACE = 8
KEY_SPACE = 32

WINDOW_NAME = "KSLR Recorder"

# Colors (BGR — OpenCV convention; renderer converts to RGB for PIL internally)
_COLOR_LEFT_HAND = (0, 255, 0)       # green
_COLOR_RIGHT_HAND = (0, 200, 255)    # orange
_COLOR_FACE = (255, 0, 255)          # magenta
_COLOR_OK = (0, 255, 0)
_COLOR_WARN = (0, 0, 255)            # red
_COLOR_TEXT = (255, 255, 255)        # white


# ──────────────────────────────────────────────────────────
# Korean-capable text overlay (PIL)
#
# OpenCV's putText uses Hershey vector fonts and only renders ASCII glyphs
# (Korean characters appear as '?'). We render text via PIL with a TrueType
# font that has Korean coverage (Malgun Gothic on Windows, Apple SD Gothic
# Neo on macOS, NanumGothic on Linux) and composite the result back onto the
# OpenCV BGR frame. Calls are buffered and flushed once per frame so we do
# only one BGR↔RGB↔PIL roundtrip per frame instead of per draw call.
# ──────────────────────────────────────────────────────────
_FONT_CANDIDATES: list[Path] = [
    Path("C:/Windows/Fonts/malgun.ttf"),                        # Windows Malgun Gothic
    Path("C:/Windows/Fonts/malgunbd.ttf"),
    Path("C:/Windows/Fonts/gulim.ttc"),
    Path("C:/Windows/Fonts/batang.ttc"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),         # macOS
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),    # Linux (Nanum)
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
]


def _load_korean_font(size: int) -> ImageFont.ImageFont:
    """Return the first available Korean-capable TrueType font at the given size."""
    for path in _FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    print("[warn] no Korean-capable font found; falling back to PIL default "
          "(Korean characters may render as boxes)", file=sys.stderr)
    return ImageFont.load_default()


class _OverlayRenderer:
    """Buffers text-draw calls and flushes them via a single PIL roundtrip per frame."""

    def __init__(self, default_size: int = 18, small_size: int = 14) -> None:
        self.font_default = _load_korean_font(default_size)
        self.font_small = _load_korean_font(small_size)
        self._calls: list[tuple[str, tuple[int, int], tuple[int, int, int], ImageFont.ImageFont]] = []

    def text(self, text: str, org: tuple[int, int],
             color: tuple[int, int, int] = _COLOR_TEXT, small: bool = False) -> None:
        font = self.font_small if small else self.font_default
        self._calls.append((text, org, color, font))

    def flush(self, img_bgr: np.ndarray) -> np.ndarray:
        if not self._calls:
            return img_bgr
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        for text, org, color_bgr, font in self._calls:
            fill_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            draw.text(org, text, font=font, fill=fill_rgb,
                      stroke_width=2, stroke_fill=(0, 0, 0))
        self._calls.clear()
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# Module-level renderer is initialized in main() so unit tests can import this
# module without instantiating fonts.
_RENDERER: _OverlayRenderer | None = None


def _put_text(img: np.ndarray, text: str, org: tuple[int, int],
              color: tuple[int, int, int] = _COLOR_TEXT, small: bool = False) -> None:
    """Buffer a text-draw call. Caller must invoke `_RENDERER.flush(img)` per frame."""
    if _RENDERER is None:
        # Fallback for any code path that draws before main() runs (shouldn't happen).
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)
        return
    _RENDERER.text(text, org, color, small=small)


def _draw_landmarks(img: np.ndarray, result) -> None:
    h, w = img.shape[:2]
    for i, color in enumerate((_COLOR_LEFT_HAND, _COLOR_RIGHT_HAND)):
        if not result.hand_mask[i]:
            continue
        for x, y, _ in result.hand_landmarks[i]:
            cv2.circle(img, (int(x * w), int(y * h)), 3, color, -1)
    if result.face_mask:
        for x, y, _ in result.face_landmarks:
            cv2.circle(img, (int(x * w), int(y * h)), 1, _COLOR_FACE, -1)


def _draw_progress_bar(img: np.ndarray, progress: float) -> None:
    h, w = img.shape[:2]
    bar_x0, bar_x1 = 20, w - 20
    bar_y0, bar_y1 = h - 40, h - 20
    cv2.rectangle(img, (bar_x0, bar_y0), (bar_x1, bar_y1), _COLOR_TEXT, 2)
    fill_x1 = bar_x0 + int((bar_x1 - bar_x0) * progress)
    cv2.rectangle(img, (bar_x0, bar_y0), (fill_x1, bar_y1), _COLOR_OK, -1)


def _draw_status(img: np.ndarray, recorder: ClipRecorder, vocab: dict) -> None:
    h, w = img.shape[:2]
    cls = recorder.current_class_id
    label = vocab.get(cls, f"class_{cls}") if cls is not None else "(none)"
    line1 = f"Signer {recorder.signer_id} | Class: {cls if cls is not None else '-'}  {label}"
    line2 = f"State: {recorder.state.value.upper()}"
    if cls is not None:
        line2 += f"   saved this session: {recorder.saved_count.get(cls, 0)}"
    _put_text(img, line1, (10, 25))
    _put_text(img, line2, (10, 50))

    if recorder.state == RecState.CAPTURING:
        _draw_progress_bar(img, recorder.progress)
        _put_text(img,
                  f"Capturing... {recorder.captured_frame_count}/{recorder.num_frames}",
                  (10, h - 50))
    elif recorder.state == RecState.REVIEW and recorder.last_quality:
        q = recorder.last_quality
        warn_color = _COLOR_WARN if q.warn else _COLOR_OK
        lines = [
            f"REVIEW  hand any:{q.any_hand_rate*100:5.1f}%  L:{q.left_hand_rate*100:5.1f}%"
            f"  R:{q.right_hand_rate*100:5.1f}%  both:{q.both_hands_frames}/{q.num_frames}",
            f"        face:{q.face_rate*100:5.1f}%   "
            f"{'WARN (<' + str(int(WARN_THRESHOLD*100)) + '%)' if q.warn else 'OK'}",
            "[Enter]=save   [Backspace]=discard",
        ]
        for i, t in enumerate(lines):
            _put_text(img, t, (10, h - 80 + i * 22), color=warn_color if i < 2 else _COLOR_TEXT)
    else:
        _put_text(img,
                  "[1-9,0]=class   [Space]=start capture   [q]=quit",
                  (10, h - 20), small=True)


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def _load_vocabulary(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    classes = raw.get("classes", {}) or {}
    return {int(k): str(v) for k, v in classes.items()}


def _open_camera(index: int, cfg: dict) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {index}")
    w, h = cfg["clip"]["webcam_resolution"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, cfg["clip"]["webcam_fps"])
    return cap


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--signer", type=int, required=True, help="Signer id (1..N)")
    p.add_argument("--config", default="configs/lab_dataset.yaml")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    vocab_path = Path(cfg["data"]["vocabulary_path"])
    vocab = _load_vocabulary(vocab_path)
    vocab_version = (yaml.safe_load(vocab_path.read_text(encoding="utf-8")) or {}).get("version", 0)

    cap = _open_camera(args.camera, cfg)
    runner = MediaPipeRunner(
        cfg["mediapipe"]["hand"]["model_asset_path"],
        cfg["mediapipe"]["face"]["model_asset_path"],
        num_hands=cfg["mediapipe"]["hand"]["num_hands"],
        min_hand_detection_confidence=cfg["mediapipe"]["hand"]["min_hand_detection_confidence"],
        min_hand_presence_confidence=cfg["mediapipe"]["hand"]["min_hand_presence_confidence"],
        min_tracking_confidence=cfg["mediapipe"]["hand"]["min_tracking_confidence"],
        num_faces=cfg["mediapipe"]["face"]["num_faces"],
    )
    recorder = ClipRecorder(
        signer_id=args.signer,
        save_root=Path(cfg["data"]["raw_dir"]),
        num_frames=cfg["clip"]["num_frames"],
        fps=cfg["clip"]["webcam_fps"],
        vocabulary_version=int(vocab_version),
        num_classes=cfg["data"]["num_classes"],
    )

    print(f"Recording for signer {args.signer}. Press 'q' to quit.")
    print(f"Saving under {recorder.save_root.resolve()}")

    global _RENDERER
    _RENDERER = _OverlayRenderer(default_size=18, small_size=14)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    t_start = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[error] camera read failed", file=sys.stderr)
                break

            ts_ms = int((time.monotonic() - t_start) * 1000)
            result = runner.process_frame(frame, ts_ms)
            recorder.push_frame(frame, result)

            display = frame.copy()
            _draw_landmarks(display, result)
            _draw_status(display, recorder, vocab)
            display = _RENDERER.flush(display)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 0xFF:
                continue
            if key == ord("q"):
                break

            if recorder.state == RecState.IDLE:
                if key in _KEY_TO_CLASS:
                    recorder.select_class(_KEY_TO_CLASS[key])
                elif key == KEY_SPACE:
                    if not recorder.start_capture():
                        print("[hint] select a class first (keys 1..9, 0)")
            elif recorder.state == RecState.REVIEW:
                if key == KEY_ENTER:
                    path = recorder.commit()
                    if path is not None:
                        print(f"[saved] {path}")
                elif key == KEY_BACKSPACE:
                    recorder.discard()
                    print("[discarded]")
            # CAPTURING state ignores keys (auto-advances when buffer is full)
    finally:
        runner.close()
        cap.release()
        cv2.destroyAllWindows()

    print("\nSession summary (saved clips per class):")
    for cls in sorted(recorder.saved_count):
        n = recorder.saved_count[cls]
        if n:
            print(f"  class {cls}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
