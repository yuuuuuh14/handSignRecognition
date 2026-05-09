"""Download the MediaPipe Tasks .task model files used by the data collection pipeline.

Usage:
    python scripts/download_models.py [--out models_assets] [--force]

Model URLs are pulled from the official MediaPipe model card pages:
    https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


MODELS: dict[str, str] = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="models_assets", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Re-download if file exists")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname, url in MODELS.items():
        dst = out_dir / fname
        if dst.exists() and not args.force:
            print(f"[skip]  {dst} already exists ({dst.stat().st_size / 1024:.1f} KB)")
            continue
        print(f"[fetch] {url}")
        print(f"     -> {dst}")
        urllib.request.urlretrieve(url, dst)
        print(f"        done, {dst.stat().st_size / 1024:.1f} KB")

    print("\nAll MediaPipe models ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
