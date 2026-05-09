"""Evaluate a saved KSLRNet checkpoint on the test split.

Usage:
    python scripts/evaluate.py --ckpt runs/kslr_lab_v0/best.pt
    python scripts/evaluate.py --ckpt runs/kslr_lab_v0/latest.pt --split train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import build_dataset
from data.splits import discover_clips, split_clips
from engine.evaluator import Evaluator
from models.kslr_net import KSLRNet
from utils.checkpoint import load_checkpoint


def _resolve_device(prefer: str) -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_class_labels(cfg: dict) -> list[str] | None:
    vp = Path(cfg["data"]["vocabulary_path"])
    if not vp.exists():
        return None
    raw = yaml.safe_load(vp.read_text(encoding="utf-8")) or {}
    classes = raw.get("classes", {}) or {}
    n = int(cfg["data"]["num_classes"])
    return [str(classes.get(i, f"class_{i}")) for i in range(n)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/lab_dataset.yaml")
    p.add_argument("--ckpt", required=True, help="path to checkpoint file (best.pt / latest.pt)")
    p.add_argument("--split", choices=["test", "train"], default="test",
                   help="which split to evaluate on")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument("--out-dir", default=None,
                   help="where to write eval_report.json + confusion_matrix.png "
                        "(default: dir of --ckpt)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device:
        cfg["device"]["prefer"] = args.device
    device = _resolve_device(str(cfg["device"]["prefer"]))

    clips = discover_clips(cfg["data"]["raw_dir"])
    if not clips:
        print(f"[error] no clips under {cfg['data']['raw_dir']}.", file=sys.stderr)
        return 1
    train_clips, test_clips = split_clips(
        clips, cfg["data"]["train_signers"], cfg["data"]["test_signers"]
    )
    eval_clips = test_clips if args.split == "test" else train_clips
    if not eval_clips:
        print(f"[error] {args.split} split is empty.", file=sys.stderr)
        return 1

    eval_ds = build_dataset(eval_clips, cfg, train=False)
    loader = DataLoader(
        eval_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["device"]["num_workers"]),
        pin_memory=bool(cfg["device"]["pin_memory"]) and device.type == "cuda",
    )

    model = KSLRNet(cfg)
    ck = load_checkpoint(args.ckpt, model=model, map_location=device)
    model.to(device).eval()
    print(f"[eval] loaded {args.ckpt} (epoch {ck.get('epoch')}, "
          f"best {ck.get('best_metric')}) on {len(eval_clips)} {args.split} clip(s)")

    evaluator = Evaluator(
        model=model,
        num_classes=int(cfg["data"]["num_classes"]),
        device=device,
        class_labels=_load_class_labels(cfg),
    )
    metrics = evaluator.evaluate(loader)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent
    paths = evaluator.save_report(metrics, out_dir)

    print()
    print(f"top1                {metrics['top1']*100:>6.2f}%")
    print(f"top3                {metrics['top3']*100:>6.2f}%")
    print(f"latency/clip        {metrics['avg_latency_ms_per_clip']:>6.2f} ms")
    print(f"samples             {metrics['num_samples']}")
    print()
    print(f"report:             {paths['report']}")
    if paths.get("confusion_matrix"):
        print(f"confusion matrix:   {paths['confusion_matrix']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
