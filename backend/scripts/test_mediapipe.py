"""Phase 1 smoke test: run MediaPipeRunner on a single image and print a summary.

Usage:
    python scripts/test_mediapipe.py path/to/image.jpg
    python scripts/test_mediapipe.py path/to/image.jpg --save-overlay debug.png

검증 기준 (IMPLEMENTATION_PLAN §10 Phase 1):
    "단일 이미지에서 landmark 추출"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/test_mediapipe.py ...` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from data_collection.mediapipe_runner import (
    MediaPipeRunner,
    NUM_FACE_LANDMARKS,
    extract_face_crop,
    extract_hand_crops,
)


def _draw_overlay(img: np.ndarray, res, save_path: Path) -> None:
    """Save a debug overlay image showing detected landmarks."""
    out = img.copy()
    h, w = out.shape[:2]
    for i, color in enumerate([(0, 255, 0), (0, 200, 255)]):  # Left=green, Right=orange
        if not res.hand_mask[i]:
            continue
        for x, y, _ in res.hand_landmarks[i]:
            cv2.circle(out, (int(x * w), int(y * h)), 2, color, -1)
    if res.face_mask:
        for x, y, _ in res.face_landmarks:
            cv2.circle(out, (int(x * w), int(y * h)), 1, (255, 0, 255), -1)
    cv2.imwrite(str(save_path), out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path, help="Path to a single image (jpg/png)")
    p.add_argument("--hand-model", default="models_assets/hand_landmarker.task")
    p.add_argument("--face-model", default="models_assets/face_landmarker.task")
    p.add_argument("--save-overlay", type=Path, default=None,
                   help="Optional output path for overlay debug image")
    args = p.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"[error] cannot read image: {args.image}", file=sys.stderr)
        return 1

    print(f"image:        {args.image}  shape={img.shape}")
    print(f"face subset:  {NUM_FACE_LANDMARKS} landmarks "
          f"(FACE_OVAL ∪ LIPS ∪ EYEBROWS)")

    with MediaPipeRunner(args.hand_model, args.face_model) as runner:
        res = runner.process_frame(img, timestamp_ms=0)

    print()
    print(f"hand_mask:    {res.hand_mask.tolist()}  (Left, Right)")
    print(f"hand_lm:      shape={res.hand_landmarks.shape}  dtype={res.hand_landmarks.dtype}")
    if res.hand_mask.any():
        xs = res.hand_landmarks[res.hand_mask, :, 0]
        ys = res.hand_landmarks[res.hand_mask, :, 1]
        print(f"              x∈[{xs.min():.3f}, {xs.max():.3f}]  y∈[{ys.min():.3f}, {ys.max():.3f}]")
    print(f"face_mask:    {res.face_mask}")
    print(f"face_lm:      shape={res.face_landmarks.shape}  dtype={res.face_landmarks.dtype}")

    hand_crops = extract_hand_crops(img, res.hand_landmarks, res.hand_mask)
    face_crop = extract_face_crop(img, res.face_landmarks, res.face_mask)
    print()
    print(f"hand_crops:   shape={hand_crops.shape}  nonzero_per_hand="
          f"{[int((hand_crops[i] > 0).any()) for i in range(2)]}")
    print(f"face_crop:    shape={face_crop.shape}  nonzero={(face_crop > 0).any()}")

    if args.save_overlay is not None:
        _draw_overlay(img, res, args.save_overlay)
        print(f"\noverlay saved: {args.save_overlay}")

    print("\n[ok] Phase 1 smoke test complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
