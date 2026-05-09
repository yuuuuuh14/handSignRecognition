"""Discover saved clips on disk and split them by signer (signer-independent).

The recorder writes one clip per directory under
    data/raw/{signer_id}/{class_id}/{timestamp}/

splits.discover_clips walks that tree and returns a list of ClipPath; splits.split_clips
partitions a clip list by signer id according to the train/test signer lists in
configs/lab_dataset.yaml. Random splits are intentionally unsupported — see
IMPLEMENTATION_PLAN §2 ("signer-independent; random split 금지").
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class ClipPath:
    signer_id: int
    class_id: int
    path: Path


def discover_clips(raw_dir: Path | str) -> list[ClipPath]:
    """Walk data/raw/ and return all complete clips, sorted by (signer, class, timestamp)."""
    raw_dir = Path(raw_dir)
    out: list[ClipPath] = []
    if not raw_dir.exists():
        return out

    for signer_dir in sorted(raw_dir.iterdir()):
        if not signer_dir.is_dir():
            continue
        try:
            signer_id = int(signer_dir.name)
        except ValueError:
            continue
        for class_dir in sorted(signer_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                class_id = int(class_dir.name)
            except ValueError:
                continue
            for ts_dir in sorted(class_dir.iterdir()):
                if not ts_dir.is_dir():
                    continue
                # Skip in-progress / corrupt saves
                if not (ts_dir / "frames.npy").exists():
                    continue
                if not (ts_dir / "meta.json").exists():
                    continue
                out.append(ClipPath(signer_id=signer_id, class_id=class_id, path=ts_dir))
    return out


def split_clips(
    clips: Iterable[ClipPath],
    train_signers: Iterable[int],
    test_signers: Iterable[int],
) -> tuple[list[ClipPath], list[ClipPath]]:
    """Partition clips by signer id. Signers in neither list are dropped silently
    (lets you record signer 11 for ad-hoc testing without poisoning a split)."""
    train_set = set(int(s) for s in train_signers)
    test_set = set(int(s) for s in test_signers)
    overlap = train_set & test_set
    if overlap:
        raise ValueError(f"signer ids appear in both train and test: {sorted(overlap)}")

    train: list[ClipPath] = []
    test: list[ClipPath] = []
    for c in clips:
        if c.signer_id in train_set:
            train.append(c)
        elif c.signer_id in test_set:
            test.append(c)
    return train, test


def summarize_clips(clips: Iterable[ClipPath]) -> dict[int, dict[int, int]]:
    """Return {signer_id: {class_id: count}} for quick inspection."""
    out: dict[int, dict[int, int]] = {}
    for c in clips:
        out.setdefault(c.signer_id, {}).setdefault(c.class_id, 0)
        out[c.signer_id][c.class_id] += 1
    return out
