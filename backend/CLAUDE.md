# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repository contains **design documents** (`document.md` Korean, `architecture.md`, `transformer.md`) and a **finalized implementation spec** (`IMPLEMENTATION_PLAN.md`). No source code yet.

**When the user asks to implement, follow `IMPLEMENTATION_PLAN.md` — it supersedes the original design docs where they conflict.** The original docs describe a single-frame RGB model targeting the public KSL Benchmark (77 classes, 89%); the implementation plan adapts that architecture for the user's actual goal: a **lab-built dataset (10 classes, 10 signers) with a real-time MediaPipe-based webcam demo**.

Build order, file layout, module specs, parameter budget, and recording protocol are all in `IMPLEMENTATION_PLAN.md`. Start at Phase 0.

## Project goal

Build a Korean Sign Language Recognition (KSLR) system for the user's lab:
1. Collect a custom 10-class KSL dataset using MediaPipe Tasks (Hand + Face Landmarker).
2. Train a hybrid CNN-Transformer model (~1.5M params) on 16-frame clips with both landmark and RGB-crop input.
3. Deploy as a real-time webcam demo with ≤300ms end-to-end latency.

The original architecture (parallel CNN + convolutional Transformer with Grain / LPU / LMHSA / IRFFN modules) is preserved, extended with per-frame multi-modal fusion and a lightweight Temporal Transformer aggregator over the clip.

## Planned architecture (from `document.md`)

The pipeline is **parallel**, not sequential like CMT (CNN Meets Transformer). Pay attention to this — it's the core design choice and easy to get wrong.

```
Input → Grain Module → ┬─ CNN Branch ─────────┐
                       │                      ├─ Concat → Classification (IRFFN) → Logits
                       └─ Transformer Branch ─┘
```

Key modules and the reason each exists:

- **Grain Module** — replaces ViT's fixed-patch linear projection. Two 3×3 convs (stride 1) followed by a 3×3 conv (stride 2). Goal: 2× downsample, channels → 32, while preserving spatial info that ViT patching would lose.
- **Transformer Branch**
  - **LPU (Local Perception Unit)**: element-wise conv instead of absolute positional embedding — robust to rotation/translation.
  - **LMHSA (Lightweight Multi-Head Self-Attention)**: reduces K and V dims via conv before attention; adds relative position bias `B`. Formula: `softmax(qk^T / sqrt(d_k) + B) · v`.
  - **MLP Conv** after attention to re-localize global features.
- **CNN Branch** — 4 blocks of 3×3 convs for local shape/texture (fine finger-shape differences the Transformer would miss).
- **IRFFN (Inverted Residual FFN)** classifier — expand dim 4×, then contract, with GELU.

## Target benchmarks

The original design doc cites these public benchmarks for **reference only** — the implementation targets a self-collected lab dataset (10 classes, 10 signers) defined in `IMPLEMENTATION_PLAN.md`:

| Dataset (reference)  | Classes | Target accuracy | Params |
|----------------------|---------|-----------------|--------|
| KSL Benchmark (orig) | 77      | 89.00%          | 1.5 M  |
| Lab Dataset (orig)   | 20      | 98.30%          | 1.52 M |

Known weakness to watch for: confusion between classes with similar hand *trajectory* but different hand *shape* (doc cites labels 13 and 16) — apply the same scrutiny to the new lab dataset.

## Language note

`document.md` is in Korean. Preserve Korean text when quoting the doc; module names (Grain, LPU, LMHSA, IRFFN) and equations are language-neutral and should be used verbatim in code.
