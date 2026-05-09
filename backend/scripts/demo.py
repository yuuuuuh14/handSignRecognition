"""Real-time KSLR webcam demo.

Usage:
    python scripts/demo.py --ckpt runs/kslr_lab_v0/best.pt
    python scripts/demo.py --ckpt runs/kslr_lab_v0/best.pt --camera 1 --device cpu

Keyboard:
    q         quit
    r         reset frame buffer (clears the warming-up state and prediction)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import yaml

from data_collection.mediapipe_runner import MediaPipeRunner
from models.kslr_net import KSLRNet
from realtime.demo_app import render_demo_frame, topk_from_probs
from realtime.webcam_pipeline import WebcamPipeline
from utils.checkpoint import load_checkpoint
from utils.text_overlay import KoreanTextRenderer


WINDOW_NAME = "KSLR Demo"
KEY_RESET = ord("r")
KEY_QUIT = ord("q")


def _resolve_device(prefer: str) -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_vocabulary(cfg: dict) -> dict[int, str]:
    vp = Path(cfg["data"]["vocabulary_path"])
    if not vp.exists():
        return {}
    raw = yaml.safe_load(vp.read_text(encoding="utf-8")) or {}
    return {int(k): str(v) for k, v in (raw.get("classes") or {}).items()}


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
    p.add_argument("--config", default="configs/lab_dataset.yaml")
    p.add_argument("--ckpt", required=True, help="path to KSLRNet checkpoint")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument("--no-window", action="store_true",
                   help="run pipeline headless (for benchmarking; no cv2.imshow)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device:
        cfg["device"]["prefer"] = args.device
    device = _resolve_device(str(cfg["device"]["prefer"]))
    print(f"[demo] device={device}  ckpt={args.ckpt}")

    # ── Build model + load checkpoint ──────────────────────────
    model = KSLRNet(cfg)
    ck = load_checkpoint(args.ckpt, model=model, map_location=device)
    print(f"[demo] checkpoint epoch={ck.get('epoch')}, "
          f"best_metric={ck.get('best_metric')}")
    model.to(device).eval()

    # ── MediaPipe runner ──────────────────────────────────────
    runner = MediaPipeRunner(
        cfg["mediapipe"]["hand"]["model_asset_path"],
        cfg["mediapipe"]["face"]["model_asset_path"],
        num_hands=cfg["mediapipe"]["hand"]["num_hands"],
        min_hand_detection_confidence=cfg["mediapipe"]["hand"]["min_hand_detection_confidence"],
        min_hand_presence_confidence=cfg["mediapipe"]["hand"]["min_hand_presence_confidence"],
        min_tracking_confidence=cfg["mediapipe"]["hand"]["min_tracking_confidence"],
        num_faces=cfg["mediapipe"]["face"]["num_faces"],
    )

    # ── Pipeline + UI ──────────────────────────────────────────
    pipeline = WebcamPipeline(model=model, mp_runner=runner, cfg=cfg, device=device)
    vocab = _load_vocabulary(cfg)
    renderer = KoreanTextRenderer(default_size=28, small_size=14)

    cap = _open_camera(args.camera, cfg)
    if not args.no_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    print(f"[demo] streaming from camera {args.camera}. Press 'q' to quit.")
    fps_smooth = 0.0
    last_t = time.perf_counter()
    t_start = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[error] camera read failed", file=sys.stderr)
                break

            ts_ms = int((time.monotonic() - t_start) * 1000)
            result, prediction = pipeline.step_frame(frame, ts_ms)

            now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, now - last_t)
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth > 0 else inst_fps
            last_t = now

            # top-3 derived from the smoother's current state, else None
            top3 = None
            sm = pipeline.smoother
            probs = getattr(sm, "_probs", None)
            if probs is not None and isinstance(probs, np.ndarray):
                top3 = topk_from_probs(probs, k=3)

            display = render_demo_frame(
                frame_bgr=frame,
                result=result,
                prediction=prediction,
                vocabulary=vocab,
                renderer=renderer,
                fps=fps_smooth,
                inference_ms=pipeline.last_inference_ms,
                buffer_fill=len(pipeline.buffer),
                buffer_capacity=pipeline.buffer.capacity,
                top3=top3,
            )

            if not args.no_window:
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key == KEY_QUIT:
                    break
                elif key == KEY_RESET:
                    pipeline.buffer.clear()
                    pipeline.last_prediction = None
                    pipeline.step = 0
                    print("[demo] buffer reset")
    finally:
        cap.release()
        runner.close()
        if not args.no_window:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
