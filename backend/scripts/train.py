"""Train KSLRNet on the lab dataset.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/lab_dataset.yaml
    python scripts/train.py --epochs 5 --batch-size 4    # quick smoke run
    python scripts/train.py --resume runs/kslr_lab_v0/latest.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import build_dataset
from data.splits import discover_clips, split_clips, summarize_clips
from engine.trainer import Trainer
from models.kslr_net import KSLRNet
from utils.checkpoint import load_checkpoint


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(prefer: str) -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "cuda":
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
    return torch.device("cpu")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/lab_dataset.yaml")
    p.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    p.add_argument("--epochs", type=int, default=None, help="override cfg.train.epochs")
    p.add_argument("--batch-size", type=int, default=None, help="override cfg.train.batch_size")
    p.add_argument("--num-workers", type=int, default=None, help="override cfg.device.num_workers")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None,
                   help="override cfg.device.prefer")
    p.add_argument("--name", default=None, help="override cfg.experiment.name")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["device"]["num_workers"] = int(args.num_workers)
    if args.device is not None:
        cfg["device"]["prefer"] = args.device
    if args.name is not None:
        cfg["experiment"]["name"] = args.name

    _set_seed(int(cfg["experiment"]["seed"]))
    device = _resolve_device(str(cfg["device"]["prefer"]))
    print(f"[train] device={device}  config={args.config}")

    # ── Discover and split clips ──────────────────────────────────
    clips = discover_clips(cfg["data"]["raw_dir"])
    if not clips:
        print(f"[error] no clips under {cfg['data']['raw_dir']}/. Record some first.",
              file=sys.stderr)
        return 1
    train_clips, test_clips = split_clips(
        clips, cfg["data"]["train_signers"], cfg["data"]["test_signers"]
    )
    print(f"[train] discovered {len(clips)} clips: train={len(train_clips)}, test={len(test_clips)}")
    summary = summarize_clips(train_clips)
    for sid in sorted(summary):
        per = ", ".join(f"c{c}:{summary[sid][c]}" for c in sorted(summary[sid]))
        print(f"        signer {sid}: {per}")
    if not train_clips:
        print(f"[error] train split is empty (no clips for signers {cfg['data']['train_signers']}).",
              file=sys.stderr)
        return 1

    # ── DataLoaders ──────────────────────────────────────────────
    train_ds = build_dataset(train_clips, cfg, train=True)
    test_ds = build_dataset(test_clips, cfg, train=False) if test_clips else None
    bs = int(cfg["train"]["batch_size"])
    bs = min(bs, max(1, len(train_ds)))
    nw = int(cfg["device"]["num_workers"])
    pm = bool(cfg["device"]["pin_memory"]) and device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pm,
        drop_last=False,
    )
    test_loader = (
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)
        if test_ds is not None else None
    )

    # ── Model ───────────────────────────────────────────────────
    model = KSLRNet(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model params: {n_params:,} ({n_params/1e6:.2f}M)")

    # ── Run dir + Trainer ───────────────────────────────────────
    run_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        run_dir=run_dir,
    )

    # ── Resume ──────────────────────────────────────────────────
    if args.resume:
        ck = load_checkpoint(
            args.resume,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            map_location=device,
        )
        trainer.global_step = int(ck.get("global_step", 0))
        trainer.best_metric = ck.get("best_metric")
        print(f"[train] resumed from {args.resume} (epoch {ck.get('epoch')})")

    # ── Save resolved config snapshot for reproducibility ───────
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8"
    )

    summary_out = trainer.fit()
    print(f"[train] done. best {summary_out['monitored']} = {summary_out['best_metric']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
