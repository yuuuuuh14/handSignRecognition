"""KSLRNet — top-level Korean Sign Language Recognition network.

Per IMPLEMENTATION_PLAN §6.1, the model is a 4-stream parallel architecture:

  Hand crops  (B,T,2,3,64,64) → Grain → CNN-4block → GAP → 128
                              → reshape (B,T,2,128) → flatten last 2 → 256-d
  Face crop   (B,T,3,64,64)   → Grain → CNN-4block → GAP → 128-d
  Hand LM     (B,T,2,21,3)    → Linear(126→64) + Transformer block → 64-d
  Face LM     (B,T,N_face,3)  → Linear(N_face*3→64) + Transformer block → 64-d
  ───────────── per-frame concat (256+128+64+64 = 512) → Linear(512→256) ────────
  TemporalAggregator (B,T,256) → mean over T → (B,256)
  IRFFNClassifier → (B,num_classes)

Weight-sharing decisions (in this first implementation):
  - Hand crop CNN: shared across the two hand slots (Left, Right) by reshaping
    the slot axis into the batch axis before the conv pass (no separate left/right
    weights). Mirror flips at training time would otherwise be needed.
  - Hand vs Face crops: separate Grain + CNN towers (face has different texture
    distribution, scale, and crop margin).
  - Landmark embed: separate Hand/Face modules (different in_dim).

forward() returns logits (B, num_classes). The model accepts the 6 fields from
ClipDataset.__getitem__ (label is handled outside). hand_mask and face_mask are
forwarded but currently unused — the architecture relies on the recorder having
zeroed missing slots, and on the CNN/Linear layers learning to handle zero
inputs naturally. (TODO: explicit mask injection if accuracy on partial-detection
clips suffers.)
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from data_collection.mediapipe_runner import (
    NUM_FACE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    NUM_HANDS,
)

from .cnn_branch import CNNBranch
from .grain import GrainModule
from .irffn import IRFFNClassifier
from .landmark_embed import LandmarkEmbed
from .temporal_aggregator import TemporalAggregator


class KSLRNet(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.num_frames = int(cfg["clip"]["num_frames"])
        self.num_hands = NUM_HANDS
        self.num_hand_landmarks = NUM_HAND_LANDMARKS
        self.num_face_landmarks = NUM_FACE_LANDMARKS
        self.crop_size = int(cfg["input"]["hand"]["crop_size"])
        self.num_classes = int(cfg["data"]["num_classes"])

        m = cfg["model"]
        grain_out = int(m["grain"]["out_channels"])
        cnn_out = int(m["cnn_branch"]["out_dim"])
        per_frame_dim = int(m["per_frame_fusion_dim"])

        # ── Image streams ──────────────────────────────────────────
        self.hand_grain = GrainModule(in_channels=3, out_channels=grain_out)
        self.face_grain = GrainModule(in_channels=3, out_channels=grain_out)

        # CNNBranch hardcodes the 4-block schedule (32→64→64→128). The config's
        # cnn_branch.channels list is informational only.
        self.hand_cnn = CNNBranch(in_channels=grain_out, out_dim=cnn_out)
        self.face_cnn = CNNBranch(in_channels=grain_out, out_dim=cnn_out)

        # ── Landmark streams ───────────────────────────────────────
        hand_lm_proj_dim = int(m["hand_landmark_embed"]["proj_dim"])
        face_lm_proj_dim = int(m["face_landmark_embed"]["proj_dim"])
        self.hand_lm_embed = LandmarkEmbed(
            in_dim=self.num_hands * self.num_hand_landmarks * 3,
            embed_dim=hand_lm_proj_dim,
            num_heads=int(m["hand_landmark_embed"]["transformer"]["heads"]),
            mlp_ratio=float(m["hand_landmark_embed"]["transformer"]["mlp_ratio"]),
            dropout=float(m["temporal"]["dropout"]),
        )
        self.face_lm_embed = LandmarkEmbed(
            in_dim=self.num_face_landmarks * 3,
            embed_dim=face_lm_proj_dim,
            num_heads=int(m["face_landmark_embed"]["transformer"]["heads"]),
            mlp_ratio=float(m["face_landmark_embed"]["transformer"]["mlp_ratio"]),
            dropout=float(m["temporal"]["dropout"]),
        )

        # ── Per-frame fusion ───────────────────────────────────────
        fused_in = (
            self.num_hands * cnn_out      # 2 × 128 = 256 (hand crops, flattened over slots)
            + cnn_out                     # 128 (face crop)
            + hand_lm_proj_dim            # 64
            + face_lm_proj_dim            # 64
        )
        self.fuse = nn.Linear(fused_in, per_frame_dim)

        # ── Temporal aggregation ──────────────────────────────────
        self.temporal = TemporalAggregator(
            dim=per_frame_dim,
            depth=int(m["temporal"]["depth"]),
            num_heads=int(m["temporal"]["heads"]),
            mlp_ratio=float(m["temporal"]["mlp_ratio"]),
            dropout=float(m["temporal"]["dropout"]),
            num_frames=self.num_frames,
            pos_embed=str(m["temporal"]["pos_embed"]),
        )

        # ── Classifier head ───────────────────────────────────────
        self.classifier = IRFFNClassifier(
            dim=per_frame_dim,
            hidden_dim=int(m["irffn"]["hidden_dim"]),
            num_classes=self.num_classes,
            dropout=float(m["irffn"]["dropout"]),
        )

    # ──────────────────────────────────────────────────────────
    def forward(
        self,
        hand_lm: torch.Tensor,        # (B, T, 2, 21, 3)
        face_lm: torch.Tensor,        # (B, T, N_face, 3)
        hand_crop: torch.Tensor,      # (B, T, 2, 3, H, W)
        face_crop: torch.Tensor,      # (B, T, 3, H, W)
        hand_mask: torch.Tensor | None = None,   # (B, T, 2)  — currently unused
        face_mask: torch.Tensor | None = None,   # (B, T,)    — currently unused
    ) -> torch.Tensor:
        B = hand_lm.shape[0]
        T = self.num_frames
        H = self.crop_size

        # ── Hand crop stream ───────────────────────────────────────
        # (B, T, 2, 3, H, W) → (B*T*2, 3, H, W)
        hc = hand_crop.reshape(B * T * self.num_hands, 3, H, H)
        hc = self.hand_cnn(self.hand_grain(hc))                     # (B*T*2, cnn_out)
        hc = hc.reshape(B, T, self.num_hands * hc.shape[-1])        # (B, T, 2*cnn_out)

        # ── Face crop stream ───────────────────────────────────────
        fc = face_crop.reshape(B * T, 3, H, H)
        fc = self.face_cnn(self.face_grain(fc))                     # (B*T, cnn_out)
        fc = fc.reshape(B, T, fc.shape[-1])                         # (B, T, cnn_out)

        # ── Hand landmark stream ───────────────────────────────────
        hl = hand_lm.reshape(B * T, self.num_hands * self.num_hand_landmarks * 3)
        hl = self.hand_lm_embed(hl)                                 # (B*T, hand_lm_proj_dim)
        hl = hl.reshape(B, T, hl.shape[-1])

        # ── Face landmark stream ───────────────────────────────────
        fl = face_lm.reshape(B * T, self.num_face_landmarks * 3)
        fl = self.face_lm_embed(fl)
        fl = fl.reshape(B, T, fl.shape[-1])

        # ── Per-frame fusion ───────────────────────────────────────
        fused = torch.cat([hc, fc, hl, fl], dim=-1)                 # (B, T, 512)
        fused = self.fuse(fused)                                    # (B, T, per_frame_dim)

        # ── Temporal + classifier ─────────────────────────────────
        clip_feat = self.temporal(fused)                            # (B, per_frame_dim)
        return self.classifier(clip_feat)                           # (B, num_classes)
