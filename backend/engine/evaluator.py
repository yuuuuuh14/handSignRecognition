"""Evaluator — top-1 / top-3 / per-class F1 / confusion matrix / latency.

Per IMPLEMENTATION_PLAN §8 the report includes:
    Top-1 accuracy (primary)
    Top-3 accuracy
    Per-class precision / recall / F1
    Confusion matrix (saved as PNG)
    Average inference time per clip (ms)

evaluate() returns a dict; save_report() writes eval_report.json + confusion_matrix.png
to a directory. Confusion-matrix PNG saving is best-effort: if matplotlib is
unavailable, the JSON report is still written.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

# Keys expected by KSLRNet.forward(). Must match data.dataset.ClipDataset output.
_MODEL_INPUT_KEYS = (
    "hand_lm",
    "face_lm",
    "hand_crop",
    "face_crop",
    "hand_mask",
    "face_mask",
)


def _move_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        k: v.to(device, non_blocking=True)
        for k, v in batch.items()
        if k in _MODEL_INPUT_KEYS
    }


def _per_class_prf(
    pred: torch.Tensor,             # (N,) int64
    target: torch.Tensor,           # (N,) int64
    num_classes: int,
) -> dict[str, list[float]]:
    pred = pred.long()
    target = target.long()
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return {"precision": precisions, "recall": recalls, "f1": f1s}


def _confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(pred.tolist(), target.tolist()):
        cm[t, p] += 1
    return cm


def _save_confusion_matrix_png(
    cm: np.ndarray, path: Path, class_labels: list[str] | None = None
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, n * 0.6), max(4, n * 0.6)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.figure.colorbar(im, ax=ax)
    ticks = list(range(n))
    labels = class_labels if class_labels is not None else [str(i) for i in ticks]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    # annotate
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        device: torch.device,
        class_labels: list[str] | None = None,
    ) -> None:
        self.model = model
        self.num_classes = int(num_classes)
        self.device = device
        self.class_labels = class_labels

    @torch.no_grad()
    def evaluate(self, loader: Iterable) -> dict:
        self.model.eval()
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_logits_topk: list[torch.Tensor] = []
        latencies_ms: list[float] = []

        is_cuda = self.device.type == "cuda"
        for batch in loader:
            inputs = _move_inputs(batch, self.device)
            target = batch["label"].to(self.device, non_blocking=True)

            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = self.model(**inputs)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            B = target.shape[0]
            latencies_ms.append((t1 - t0) * 1000.0 / B)

            all_preds.append(logits.argmax(dim=-1).cpu())
            all_targets.append(target.cpu())
            top3 = logits.topk(min(3, self.num_classes), dim=-1).indices.cpu()
            all_logits_topk.append(top3)

        pred = torch.cat(all_preds)
        target = torch.cat(all_targets)
        top3 = torch.cat(all_logits_topk)

        top1 = (pred == target).float().mean().item()
        top3_acc = (top3 == target.unsqueeze(-1)).any(dim=-1).float().mean().item()
        prf = _per_class_prf(pred, target, self.num_classes)
        cm = _confusion_matrix(pred, target, self.num_classes)

        return {
            "top1": top1,
            "top3": top3_acc,
            "per_class": prf,
            "confusion_matrix": cm.tolist(),
            "avg_latency_ms_per_clip": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
            "num_samples": int(target.shape[0]),
        }

    def save_report(self, metrics: dict, save_dir: Path | str) -> dict[str, Path]:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        report_path = save_dir / "eval_report.json"
        report_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

        cm_path = save_dir / "confusion_matrix.png"
        cm_saved = _save_confusion_matrix_png(
            np.asarray(metrics["confusion_matrix"]),
            cm_path,
            class_labels=self.class_labels,
        )
        return {
            "report": report_path,
            "confusion_matrix": cm_path if cm_saved else None,
        }
