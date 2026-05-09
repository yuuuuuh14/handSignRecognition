"""Training engine.

Per IMPLEMENTATION_PLAN §7:
    - Optimizer  : Adam, lr=1e-3, weight_decay=1e-4
    - Schedule   : 5-epoch linear warmup → cosine annealing to min_lr=1e-5
    - Loss       : CrossEntropy(label_smoothing=0.1)
    - AMP        : on (CUDA only)
    - Grad clip  : max_norm = 1.0
    - Logging    : TensorBoard
    - Checkpoint : save best (val_top1) + latest + every-N-epoch snapshots
    - Early stop : 30-epoch plateau on val_top1 (mode=max)

If `val_loader` is None (e.g. when only signer 1 has been recorded so far),
the trainer falls back to monitoring train_top1 for best-tracking and disables
early stopping, while still saving a 'latest' checkpoint each epoch.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.checkpoint import save_checkpoint
from utils.logger import Logger

from .evaluator import _MODEL_INPUT_KEYS, _move_inputs
from .losses import build_loss


class Trainer:
    def __init__(
        self,
        cfg: dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device,
        run_dir: Path | str,
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = build_loss(cfg)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self._amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)

        log_cfg = cfg["train"]["logging"]
        self.logger = Logger(self.run_dir / "tensorboard", backend=str(log_cfg["backend"]))
        self.log_every = int(log_cfg["log_every_n_steps"])

        es = cfg["train"]["early_stopping"]
        self.es_metric = str(es["metric"])
        self.es_patience = int(es["patience_epochs"])
        self.es_mode = str(es["mode"])
        self.grad_clip = float(cfg["train"]["grad_clip"])
        self.epochs = int(cfg["train"]["epochs"])
        self.save_every = int(cfg["train"]["checkpoint"]["save_every_n_epochs"])

        self.best_metric: float | None = None
        self.epochs_since_best = 0
        self.global_step = 0

        # Decide what we monitor for best/early-stop given val data availability.
        if self.val_loader is None:
            # Fall back to a train-side proxy of the requested metric.
            fallback = self.es_metric.replace("val_", "train_") \
                if self.es_metric.startswith("val_") else f"train_{self.es_metric}"
            print(f"[trainer] no val_loader; tracking '{fallback}' instead of '{self.es_metric}'.")
            self.monitored_metric = fallback
            self.early_stop_disabled = True
        else:
            self.monitored_metric = self.es_metric
            self.early_stop_disabled = False

    # ──────────────────────────────────────────────────────────
    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = self.cfg["train"]["optimizer"]
        name = str(opt_cfg["name"]).lower()
        if name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=float(opt_cfg["lr"]),
                weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
            )
        if name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=float(opt_cfg["lr"]),
                weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
            )
        raise ValueError(f"unknown optimizer '{name}'")

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        s_cfg = self.cfg["train"]["scheduler"]
        name = str(s_cfg["name"]).lower()
        if name != "cosine_with_warmup":
            raise ValueError(f"unknown scheduler '{name}'")

        warmup = int(s_cfg["warmup_epochs"])
        total = int(self.cfg["train"]["epochs"])
        base_lr = float(self.cfg["train"]["optimizer"]["lr"])
        min_lr = float(s_cfg.get("min_lr", 0.0))
        min_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            progress = (epoch - warmup) / max(1, total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    # ──────────────────────────────────────────────────────────
    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        t_start = time.perf_counter()

        for batch in self.train_loader:
            inputs = _move_inputs(batch, self.device)
            target = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self._amp_enabled):
                logits = self.model(**inputs)
                loss = self.criterion(logits, target)

            if self._amp_enabled:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

            B = target.shape[0]
            loss_sum += loss.item() * B
            correct += (logits.argmax(dim=-1) == target).sum().item()
            total += B

            if self.log_every > 0 and self.global_step % self.log_every == 0:
                self.logger.scalar("train/loss_step", loss.item(), self.global_step)
                self.logger.scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                )
            self.global_step += 1

        dur = time.perf_counter() - t_start
        return {
            "train_loss": loss_sum / max(1, total),
            "train_top1": correct / max(1, total),
            "train_seconds": dur,
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        loss_sum = 0.0
        correct = 0
        correct_top3 = 0
        total = 0
        for batch in self.val_loader:
            inputs = _move_inputs(batch, self.device)
            target = batch["label"].to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self._amp_enabled):
                logits = self.model(**inputs)
                loss = self.criterion(logits, target)
            B = target.shape[0]
            loss_sum += loss.item() * B
            correct += (logits.argmax(dim=-1) == target).sum().item()
            top3 = logits.topk(min(3, logits.shape[-1]), dim=-1).indices
            correct_top3 += (top3 == target.unsqueeze(-1)).any(dim=-1).sum().item()
            total += B
        return {
            "val_loss": loss_sum / max(1, total),
            "val_top1": correct / max(1, total),
            "val_top3": correct_top3 / max(1, total),
        }

    # ──────────────────────────────────────────────────────────
    def _is_better(self, current: float, best: float | None) -> bool:
        if best is None:
            return True
        return current > best if self.es_mode == "max" else current < best

    def _save_periodic(self, epoch: int, metrics: dict) -> None:
        ck_cfg = self.cfg["train"]["checkpoint"]
        if ck_cfg.get("save_latest", True):
            save_checkpoint(
                self.run_dir / "latest.pt",
                model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                scaler=self.scaler, epoch=epoch, global_step=self.global_step,
                best_metric=self.best_metric, metrics=metrics, extra={"cfg": self.cfg},
            )
        if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
            save_checkpoint(
                self.run_dir / f"epoch_{epoch+1:03d}.pt",
                model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                scaler=self.scaler, epoch=epoch, global_step=self.global_step,
                best_metric=self.best_metric, metrics=metrics, extra={"cfg": self.cfg},
            )

    def _save_best(self, epoch: int, metrics: dict) -> None:
        if not self.cfg["train"]["checkpoint"].get("save_best", True):
            return
        save_checkpoint(
            self.run_dir / "best.pt",
            model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            scaler=self.scaler, epoch=epoch, global_step=self.global_step,
            best_metric=self.best_metric, metrics=metrics, extra={"cfg": self.cfg},
        )

    # ──────────────────────────────────────────────────────────
    def fit(self) -> dict:
        for epoch in range(self.epochs):
            train_m = self._train_one_epoch(epoch)
            val_m = self._validate(epoch)
            self.scheduler.step()

            metrics = {**train_m, **val_m, "lr": self.optimizer.param_groups[0]["lr"]}
            self.logger.scalars(
                {f"epoch/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                step=epoch,
            )

            current = float(metrics.get(self.monitored_metric, float("nan")))
            print(
                f"epoch {epoch:>3d} | "
                + " ".join(f"{k}:{v:.4f}" for k, v in metrics.items()
                           if isinstance(v, float) and k != "train_seconds")
                + f" | dur:{metrics['train_seconds']:.1f}s"
            )

            self._save_periodic(epoch, metrics)
            if not math.isnan(current) and self._is_better(current, self.best_metric):
                self.best_metric = current
                self.epochs_since_best = 0
                self._save_best(epoch, metrics)
            else:
                self.epochs_since_best += 1

            if not self.early_stop_disabled and self.epochs_since_best >= self.es_patience:
                print(f"[trainer] early stopping at epoch {epoch} "
                      f"({self.es_patience} epochs without {self.monitored_metric} improvement)")
                break

        self.logger.close()
        return {"best_metric": self.best_metric, "monitored": self.monitored_metric,
                "final_epoch": epoch}
