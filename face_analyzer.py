#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║         Combined Face Identity + Expression Analyzer                    ║
║  Runs DPAIN (who) and EmoNeXt (what emotion) on every face at once      ║
║  Each face gets:  PersonName — Expression  label + confidence bars      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Usage:                                                                 ║
║      python face_analyzer.py          # interactive mode selector GUI   ║
║      python face_analyzer.py --webcam # webcam directly                 ║
║      python face_analyzer.py video.mp4 # video file directly            ║
║                                                                         ║
║  Controls (both modes):                                                 ║
║      q / ESC  — Quit                                                    ║
║      s        — Save screenshot                                         ║
║      t        — Toggle TTA (Test-Time Augmentation)                     ║
║      f        — Toggle probability bars                                 ║
║      r        — Toggle unknown-identity rejection                       ║
║      d        — Toggle summary dashboard                                ║
║      SPACE    — Play / Pause  (video mode only)                         ║
║      ← →      — Seek ±5 s     (video mode only)                         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import time
import threading
import traceback
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import mediapipe as mp

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent

# ── EmoNeXt ──────────────────────────────────────────────────────
EMONEXT_DIR        = ROOT / "emonext" / "emonext_outputs" / "saved_models"
EMONEXT_CKPT_CANDIDATES = [
    EMONEXT_DIR / "emonext_final.pth",
    EMONEXT_DIR / "best_model.pth",
]
YUNET_CANDIDATES   = [
    ROOT / "emonext" / "face_detection_yunet_2023mar.onnx",
    ROOT / "Affectnet_v2" / "face_detection_yunet_2023mar.onnx",
]

# ── DPAIN ────────────────────────────────────────────────────────
DPAIN_DIR          = ROOT / "Identity_v1" / "identity_outputs"
DPAIN_CKPT         = DPAIN_DIR / "checkpoints" / "best_model.pth"
DPAIN_CONFIG       = DPAIN_DIR / "configs" / "experiment_config.json"
DPAIN_CLASS_MAP    = DPAIN_DIR / "configs" / "class_mapping.json"
DPAIN_CENTROIDS    = DPAIN_DIR / "embeddings" / "identity_centroids.json"
MEDIAPIPE_MODEL    = ROOT / "Identity_v1" / "face_landmarker_v2_with_blendshapes.task"

# ── Output dirs ──────────────────────────────────────────────────
SCREENSHOT_DIR     = ROOT / "combined_outputs" / "screenshots"
ANNOTATED_DIR      = ROOT / "combined_outputs" / "annotated_videos"
for _d in [SCREENSHOT_DIR, ANNOTATED_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS  (loaded from saved configs)
# ═══════════════════════════════════════════════════════════════════

# EmoNeXt
EMOTION_NAMES = ["Neutral", "Happy", "Sad", "Surprise",
                 "Fear", "Disgust", "Anger", "Contempt"]
EMOTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Neutral":  (180, 180, 180),
    "Happy":    (0, 220, 100),
    "Sad":      (255, 140, 0),
    "Surprise": (0, 200, 255),
    "Fear":     (200, 100, 220),
    "Disgust":  (0, 120, 20),
    "Anger":    (0, 0, 220),
    "Contempt": (0, 180, 180),
}
EMONEXT_IMG_SIZE = 224
EMONEXT_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
EMONEXT_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# DPAIN — populated after loading configs
IDENTITY_NAMES: List[str] = []
IDX_TO_LABEL: Dict[str, str] = {}
LABEL_TO_IDX: Dict[str, int] = {}
DPAIN_IMG_SIZE  = 224
DPAIN_EMBED_DIM = 128
DPAIN_MEAN = (0.6324, 0.5407, 0.4785)
DPAIN_STD  = (0.2853, 0.2862, 0.2807)
REJECTION_THRESHOLD = 0.75

IDENTITY_COLORS = [
    (255, 100, 100), (255, 200, 50), (50, 200, 50),
    (0, 255, 128),   (0, 200, 255), (180, 105, 255),
    (128, 0, 255),   (100, 150, 200), (0, 165, 255),
    (255, 0, 200),   (80, 200, 200), (200, 100, 200),
    (0, 230, 230),   (255, 200, 200), (100, 255, 200),
]
UNKNOWN_COLOR = (50, 50, 200)

# MediaPipe eye landmark indices (same as training)
LEFT_EYE_IDX  = [33, 133, 159, 145, 160, 144, 158, 153]
RIGHT_EYE_IDX = [362, 263, 386, 374, 387, 373, 385, 380]

# ═══════════════════════════════════════════════════════════════════
# ──────────────────────  EMONEXT ARCHITECTURE  ────────────────────
# ═══════════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
    def forward(self, x):
        if self.p == 0. or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


class SqueezeExcite(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(ch // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.GELU(),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class EmoNeXtBlock(nn.Module):
    def __init__(self, ch, expand=3, se_ratio=4, drop_path=0.0):
        super().__init__()
        exp = ch * expand
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, exp, 1, bias=False), nn.BatchNorm2d(exp), nn.GELU(),
            SqueezeExcite(exp, se_ratio),
            nn.Conv2d(exp, ch, 1, bias=False), nn.BatchNorm2d(ch),
        )
        self.dp = DropPath(drop_path)
    def forward(self, x):
        return x + self.dp(self.block(x))


class MultiFrequencySpatialAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=d, dilation=d, groups=ch, bias=False),
                nn.BatchNorm2d(ch), nn.GELU(),
            ) for d in [1, 2, 3]
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.Sigmoid(),
        )
    def forward(self, x):
        gate = self.fuse(torch.cat([b(x) for b in self.branches], 1))
        return x * gate + x


class Downsample(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(ic, oc, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(oc), nn.GELU(),
        )
    def forward(self, x): return self.c(x)


class FacialRegionAwarePooling(nn.Module):
    def __init__(self, ch, nr=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(ch * (nr + 1), 256), nn.GELU(),
            nn.Linear(256, nr + 1), nn.Softmax(dim=-1),
        )
    def forward(self, x):
        B, C, H, W = x.shape
        g = F.adaptive_avg_pool2d(x, 1).flatten(1)
        h2, w2 = H // 2, W // 2
        regions = [x[:, :, :h2, :w2], x[:, :, :h2, w2:],
                   x[:, :, h2:, :w2], x[:, :, h2:, w2:]]
        rfs = [F.adaptive_avg_pool2d(r, 1).flatten(1) for r in regions]
        all_f = [g] + rfs
        gates = self.gate(torch.cat(all_f, -1))
        stacked = torch.stack(all_f, 1)
        return (stacked * gates.unsqueeze(-1)).reshape(B, -1), gates


class EmoNeXt(nn.Module):
    def __init__(self, num_classes=8, channels=None, depths=None,
                 expand_ratio=3, se_ratio=4, dropout=0.2, drop_path=0.15,
                 use_mfsa=True):
        super().__init__()
        if channels is None: channels = [64, 128, 256, 384]
        if depths   is None: depths   = [2, 3, 4, 2]
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0]//2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]//2), nn.GELU(),
            nn.Conv2d(channels[0]//2, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]), nn.GELU(),
        )
        dp_rates = list(torch.linspace(0, drop_path, sum(depths)).numpy())
        bi = 0
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(4):
            blks = []
            for _ in range(depths[i]):
                blks.append(EmoNeXtBlock(channels[i], expand_ratio, se_ratio, dp_rates[bi]))
                bi += 1
            if use_mfsa and i >= 2:
                blks.append(MultiFrequencySpatialAttention(channels[i]))
            self.stages.append(nn.Sequential(*blks))
            if i < 3:
                self.downsamples.append(Downsample(channels[i], channels[i+1]))
        self.frap = FacialRegionAwarePooling(channels[-1])
        fd = channels[-1] * 5
        self.head = nn.Sequential(
            nn.Linear(fd, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, num_classes),
        )
        self.embed_proj = nn.Linear(fd, 256)

    def forward(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        pooled, gates = self.frap(x)
        return {"logits": self.head(pooled), "embeddings": self.embed_proj(pooled)}


# ═══════════════════════════════════════════════════════════════════
# ──────────────────────  DPAIN ARCHITECTURE  ──────────────────────
# ═══════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, ic, oc, k=3, s=1, p=1, d=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, k, stride=s, padding=p, dilation=d, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(oc)
        self.act  = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))


class AdaptiveChannelGate(nn.Module):
    def __init__(self, ch, red=4):
        super().__init__()
        mid = max(ch // red, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.gate(x).unsqueeze(-1).unsqueeze(-1)


class StructuralBlock(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(ic, oc, k=3, s=stride, p=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc))
        self.gate     = AdaptiveChannelGate(oc)
        self.shortcut = nn.Identity() if (stride == 1 and ic == oc) else nn.Sequential(
            nn.Conv2d(ic, oc, 1, stride=stride, bias=False), nn.BatchNorm2d(oc))
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.gate(out)
        return self.act(out + residual)


class DetailBlock(nn.Module):
    def __init__(self, ic, oc, dilation=2):
        super().__init__()
        self.conv1 = ConvBlock(ic, oc, k=3, p=dilation, d=dilation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc))
        self.gate     = AdaptiveChannelGate(oc)
        self.shortcut = nn.Identity() if ic == oc else nn.Sequential(
            nn.Conv2d(ic, oc, 1, bias=False), nn.BatchNorm2d(oc))
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.gate(out)
        return self.act(out + residual)


class AdaptiveFusionGate(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch * 2, ch), nn.ReLU(inplace=True),
            nn.Linear(ch, 2), nn.Softmax(dim=1),
        )
    def forward(self, s, d):
        if s.shape[2:] != d.shape[2:]:
            d = F.adaptive_avg_pool2d(d, s.shape[2:])
        w = self.fc(torch.cat([
            F.adaptive_avg_pool2d(s, 1).flatten(1),
            F.adaptive_avg_pool2d(d, 1).flatten(1)], 1))
        return w[:, 0:1, None, None] * s + w[:, 1:2, None, None] * d


class MultiScalePool(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.spatial_att = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())
    def forward(self, x):
        B, C = x.shape[:2]
        att = self.spatial_att(x)
        return (F.adaptive_avg_pool2d(x, 1).view(B, C)
                + F.adaptive_max_pool2d(x, 1).view(B, C)
                + (x * att).sum([2, 3]) / (att.sum([2, 3]) + 1e-8))


class DPAINModel(nn.Module):
    def __init__(self, embed_dim=128, dropout=0.0):
        super().__init__()
        # Shared stem
        self.stem = nn.Sequential(
            ConvBlock(3, 32, k=5, s=2, p=2),
            ConvBlock(32, 48),
        )
        # Structural path
        self.struct_s1 = StructuralBlock(48, 64, stride=2)
        self.struct_s2 = StructuralBlock(64, 96, stride=2)
        self.struct_s3 = StructuralBlock(96, 128, stride=2)
        self.struct_s4 = StructuralBlock(128, 160, stride=2)
        # Detail path
        self.detail_s1    = DetailBlock(48, 64, dilation=2)
        self.detail_down1 = nn.MaxPool2d(2)
        self.detail_s2    = DetailBlock(64, 96, dilation=2)
        self.detail_down2 = nn.MaxPool2d(2)
        self.detail_s3    = DetailBlock(96, 128, dilation=2)
        self.detail_down3 = nn.MaxPool2d(2)
        self.detail_s4    = DetailBlock(128, 160, dilation=1)
        self.detail_down4 = nn.MaxPool2d(2)
        # Fusion + pooling
        self.fusion = AdaptiveFusionGate(160)
        self.pool   = MultiScalePool(160)
        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(160, 128), nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True), nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x):
        shared = self.stem(x)
        # structural
        s = self.struct_s1(shared)
        s = self.struct_s2(s)
        s = self.struct_s3(s)
        s = self.struct_s4(s)
        # detail
        d = self.detail_s1(shared)
        d = self.detail_down1(d)
        d = self.detail_s2(d)
        d = self.detail_down2(d)
        d = self.detail_s3(d)
        d = self.detail_down3(d)
        d = self.detail_s4(d)
        d = self.detail_down4(d)
        emb = self.embedding_head(self.pool(self.fusion(s, d)))
        return F.normalize(emb, p=2, dim=1)


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════

def load_emonext(device: torch.device) -> EmoNeXt:
    ckpt_path = next((p for p in EMONEXT_CKPT_CANDIDATES if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            "EmoNeXt checkpoint not found. Expected:\n" +
            "\n".join(f"  {p}" for p in EMONEXT_CKPT_CANDIDATES))
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = data.get("cfg", {})
    model = EmoNeXt(
        num_classes  = len(EMOTION_NAMES),
        channels     = cfg.get("CHANNELS",     [64, 128, 256, 384]),
        depths       = cfg.get("DEPTHS",       [2, 3, 4, 2]),
        expand_ratio = int(cfg.get("EXPAND_RATIO", 3)),
        se_ratio     = int(cfg.get("SE_RATIO",     4)),
        dropout      = float(cfg.get("DROPOUT",    0.2)),
        drop_path    = float(cfg.get("DROP_PATH",  0.15)),
        use_mfsa     = cfg.get("USE_MFSA", True),
    )
    state = data.get("model_state") or data.get("model_state_dict") or data
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    val_acc = data.get("val_acc", "?")
    acc_str = f"{val_acc*100:.1f}%" if isinstance(val_acc, float) else str(val_acc)
    print(f"  ✅ EmoNeXt loaded: {ckpt_path.name}  ({params:.2f}M params, val_acc={acc_str})")
    return model


def load_dpain(device: torch.device):
    global IDENTITY_NAMES, IDX_TO_LABEL, LABEL_TO_IDX, DPAIN_EMBED_DIM, DPAIN_IMG_SIZE

    for path, name in [(DPAIN_CONFIG, "experiment_config.json"),
                       (DPAIN_CLASS_MAP, "class_mapping.json"),
                       (DPAIN_CENTROIDS, "identity_centroids.json"),
                       (DPAIN_CKPT,      "best_model.pth")]:
        if not path.exists():
            raise FileNotFoundError(
                f"DPAIN file not found: {path}\n"
                "Run Identity_Recognition_System.ipynb first.")

    with open(DPAIN_CONFIG) as f:
        cfg = json.load(f)
    DPAIN_EMBED_DIM = cfg.get("embedding_dim", 128)
    DPAIN_IMG_SIZE  = cfg.get("image_size", 224)

    with open(DPAIN_CLASS_MAP) as f:
        mapping = json.load(f)
    IDX_TO_LABEL = mapping.get("idx_to_label", mapping.get("idx_to_class", {}))
    LABEL_TO_IDX = mapping.get("label_to_idx", mapping.get("class_to_idx", {}))
    IDENTITY_NAMES = [IDX_TO_LABEL[str(i)] for i in range(len(IDX_TO_LABEL))]

    with open(DPAIN_CENTROIDS) as f:
        centroids_dict = json.load(f)
    cent_arr = np.array([centroids_dict[n] for n in IDENTITY_NAMES], dtype=np.float32)
    norms = np.linalg.norm(cent_arr, axis=1, keepdims=True) + 1e-8
    cent_arr /= norms
    centroid_tensor = torch.tensor(cent_arr).to(device)

    model = DPAINModel(embed_dim=DPAIN_EMBED_DIM, dropout=0.0)
    ckpt  = torch.load(DPAIN_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    epoch = ckpt.get("epoch", "?")
    print(f"  ✅ DPAIN loaded: {DPAIN_CKPT.name}  ({params:.2f}M params, epoch={epoch})")
    print(f"     Identities ({len(IDENTITY_NAMES)}): {', '.join(IDENTITY_NAMES)}")
    return model, centroid_tensor


# ═══════════════════════════════════════════════════════════════════
# FACE DETECTION  (YuNet — same as video_expression_analyzer.py)
# ═══════════════════════════════════════════════════════════════════

def resolve_yunet() -> Path:
    p = next((c for c in YUNET_CANDIDATES if c.exists()), None)
    if p is None:
        raise FileNotFoundError(
            "YuNet ONNX not found. Tried:\n" +
            "\n".join(f"  {c}" for c in YUNET_CANDIDATES))
    return p.resolve()


def create_yunet(model_path: Path, w: int, h: int):
    scales = [1.0, 0.5]
    detectors = []
    for sc in scales:
        dw, dh = int(w * sc), int(h * sc)
        det = cv2.FaceDetectorYN.create(
            str(model_path), "", (dw, dh),
            score_threshold=0.45, nms_threshold=0.3, top_k=5000)
        detectors.append((det, sc))
    return detectors


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0: return 0.0
    return inter / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter, 1e-6)


def detect_faces_yunet(detectors, frame: np.ndarray) -> List[dict]:
    raw = []
    for det, sc in detectors:
        small = cv2.resize(frame, None, fx=sc, fy=sc) if abs(sc-1) > 1e-6 else frame
        det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = det.detect(small)
        if faces is None: continue
        inv = 1.0 / sc
        for f in faces:
            if float(f[14]) < 0.45: continue
            x1,y1 = int(f[0]*inv), int(f[1]*inv)
            bw,bh = int(f[2]*inv), int(f[3]*inv)
            if bw < 24 or bh < 24: continue
            le = (float(f[4]*inv), float(f[5]*inv))
            re = (float(f[6]*inv), float(f[7]*inv))
            ed = ((le[0]-re[0])**2 + (le[1]-re[1])**2)**0.5
            if ed < bw * 0.15: continue
            raw.append({"bbox":(x1,y1,x1+bw,y1+bh), "left_eye":le, "right_eye":re, "score":float(f[14])})
    raw.sort(key=lambda f: f["score"], reverse=True)
    kept = []
    for c in raw:
        if not any(_iou(c["bbox"], k["bbox"]) > 0.35 for k in kept):
            kept.append(c)
    return kept


# ═══════════════════════════════════════════════════════════════════
# FACE ALIGNMENT — YuNet eye-based (EmoNeXt pipeline)
# ═══════════════════════════════════════════════════════════════════

def align_face_yunet(frame_rgb: np.ndarray, face: dict,
                     margin=0.25, target=224) -> Optional[np.ndarray]:
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = face["bbox"]
    bw, bh = max(1, x2-x1), max(1, y2-y1)
    le = np.array(face["left_eye"])
    re = np.array(face["right_eye"])
    angle = np.degrees(np.arctan2(re[1]-le[1], re[0]-le[0]))
    center = ((le[0]+re[0])/2, (le[1]+re[1])/2)

    # Rotate around eye center within a generous ROI
    sz = max(bw, bh)
    roi_exp = sz * 2.0
    cx, cy = (x1+x2)/2, (y1+y2)/2
    rx1, ry1 = max(0, int(cx-roi_exp)), max(0, int(cy-roi_exp))
    rx2, ry2 = min(w, int(cx+roi_exp)), min(h, int(cy+roi_exp))
    roi = frame_rgb[ry1:ry2, rx1:rx2]
    if roi.size == 0: return None
    rh, rw = roi.shape[:2]

    eye_c = (center[0]-rx1, center[1]-ry1)
    M = cv2.getRotationMatrix2D(eye_c, angle, 1.0)
    rotated = cv2.warpAffine(roi, M, (rw, rh), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    corners = np.array([[float(x1-rx1), float(y1-ry1)],
                        [float(x2-rx1), float(y1-ry1)],
                        [float(x2-rx1), float(y2-ry1)],
                        [float(x1-rx1), float(y2-ry1)]])
    rc = (M @ np.hstack([corners, np.ones((4,1))]).T).T
    rx_min, ry_min = rc[:,0].min(), rc[:,1].min()
    rx_max, ry_max = rc[:,0].max(), rc[:,1].max()
    rbw, rbh = rx_max-rx_min, ry_max-ry_min

    cx1 = max(0, int(rx_min - margin*rbw))
    cy1 = max(0, int(ry_min - margin*rbh))
    cx2 = min(rw, int(rx_max + margin*rbw))
    cy2 = min(rh, int(ry_max + margin*rbh))
    side = max(cx2-cx1, cy2-cy1)
    ccx, ccy = (cx1+cx2)//2, (cy1+cy2)//2
    sx1 = max(0, ccx-side//2); sy1 = max(0, ccy-side//2)
    sx2 = min(rw, sx1+side);   sy2 = min(rh, sy1+side)
    crop = rotated[sy1:sy2, sx1:sx2]
    if crop.size == 0: crop = roi
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_LANCZOS4)


# ═══════════════════════════════════════════════════════════════════
# FACE ALIGNMENT — MediaPipe landmark-based (DPAIN pipeline)
# ═══════════════════════════════════════════════════════════════════

def _eye_centers(lms, w, h):
    le = np.array([(lms[i].x*w, lms[i].y*h) for i in LEFT_EYE_IDX]).mean(0)
    re = np.array([(lms[i].x*w, lms[i].y*h) for i in RIGHT_EYE_IDX]).mean(0)
    return le, re


def align_face_mediapipe(frame_rgb: np.ndarray, landmarks, w, h,
                          margin=0.3, target=224) -> Optional[np.ndarray]:
    le, re = _eye_centers(landmarks, w, h)
    angle = np.degrees(np.arctan2(re[1]-le[1], re[0]-le[0]))
    center = ((le[0]+re[0])/2, (le[1]+re[1])/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame_rgb, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    pts = np.array([(landmarks[i].x*w, landmarks[i].y*h) for i in range(len(landmarks))])
    rp = (M @ np.hstack([pts, np.ones((len(pts),1))]).T).T
    xmn, ymn = rp[:,0].min(), rp[:,1].min()
    xmx, ymx = rp[:,0].max(), rp[:,1].max()
    bw, bh = xmx-xmn, ymx-ymn
    x1 = max(0, int(xmn - margin*bw)); y1 = max(0, int(ymn - margin*bh))
    x2 = min(w, int(xmx + margin*bw)); y2 = min(h, int(ymx + margin*bh))
    side = max(x2-x1, y2-y1)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    x1 = max(0, cx-side//2); y1 = max(0, cy-side//2)
    x2 = min(w, x1+side);    y2 = min(h, y1+side)
    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0: return None
    resized = cv2.resize(crop, (target, target), interpolation=cv2.INTER_LANCZOS4)
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def lm_bbox(landmarks, w, h, margin=0.15):
    xs = [landmarks[i].x*w for i in range(len(landmarks))]
    ys = [landmarks[i].y*h for i in range(len(landmarks))]
    bw = max(xs)-min(xs); bh = max(ys)-min(ys)
    return (max(0,int(min(xs)-margin*bw)), max(0,int(min(ys)-margin*bh)),
            min(w,int(max(xs)+margin*bw)), min(h,int(max(ys)+margin*bh)))


# ═══════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

def preprocess_emonext(face_rgb: np.ndarray) -> torch.Tensor:
    img = face_rgb.astype(np.float32) / 255.0
    img = (img - EMONEXT_MEAN) / EMONEXT_STD
    return torch.from_numpy(img).permute(2, 0, 1)


def preprocess_dpain(face_rgb: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(face_rgb)
    t = TF.to_tensor(pil)
    return TF.normalize(t, DPAIN_MEAN, DPAIN_STD)


# ═══════════════════════════════════════════════════════════════════
# INFERENCE  (batched — one GPU pass for all faces per model)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def infer_emonext_batch(model: EmoNeXt, tensors: List[torch.Tensor],
                        device: torch.device) -> np.ndarray:
    if not tensors: return np.empty((0, len(EMOTION_NAMES)))
    batch = torch.stack(tensors).to(device)
    out = model(batch)
    return F.softmax(out["logits"], dim=1).cpu().numpy()


@torch.no_grad()
def infer_dpain_batch(model: DPAINModel, tensors: List[torch.Tensor],
                      centroids: torch.Tensor, device: torch.device) -> np.ndarray:
    if not tensors: return np.empty((0, len(IDENTITY_NAMES)))
    batch = torch.stack(tensors).to(device)
    embs = model(batch)                     # (N, embed_dim)  L2-normalised
    sims = torch.matmul(embs, centroids.T)  # (N, num_ids)  cosine similarity
    return sims.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════
# PER-FACE TRACK  (temporal EMA smoothing for both outputs)
# ═══════════════════════════════════════════════════════════════════

class FaceTrack:
    _next_id = 0
    def __init__(self, bbox):
        self.id = FaceTrack._next_id; FaceTrack._next_id += 1
        self.bbox = bbox
        self.frames_missing = 0
        # EMA buffers
        n_emo = len(EMOTION_NAMES)
        n_id  = len(IDENTITY_NAMES) if IDENTITY_NAMES else 1
        self.emo_probs  = np.ones(n_emo,  dtype=np.float32) / n_emo
        self.id_sims    = np.zeros(n_id,  dtype=np.float32)
        self.EMA        = 0.45

    def update_emotion(self, probs: np.ndarray):
        self.emo_probs = self.EMA * probs + (1 - self.EMA) * self.emo_probs

    def update_identity(self, sims: np.ndarray):
        self.id_sims = self.EMA * sims + (1 - self.EMA) * self.id_sims

    @property
    def emotion(self): return EMOTION_NAMES[int(np.argmax(self.emo_probs))]
    @property
    def emo_conf(self):  return float(np.max(self.emo_probs))
    @property
    def identity(self):
        if not IDENTITY_NAMES: return "?"
        return IDENTITY_NAMES[int(np.argmax(self.id_sims))]
    @property
    def id_conf(self):
        if not IDENTITY_NAMES: return 0.0
        return float(np.max(self.id_sims))


class Tracker:
    def __init__(self, iou_thr=0.30, max_missing=12):
        self.tracks: List[FaceTrack] = []
        self.iou_thr = iou_thr
        self.max_missing = max_missing

    def update(self, faces: List[dict]) -> List[FaceTrack]:
        for t in self.tracks: t.frames_missing += 1

        used_t, used_d = set(), set()
        for di, face in enumerate(faces):
            best_iou, best_ti = 0.0, -1
            for ti, t in enumerate(self.tracks):
                if ti in used_t: continue
                iou = _iou(face["bbox"], t.bbox)
                if iou > best_iou: best_iou, best_ti = iou, ti
            if best_iou >= self.iou_thr and best_ti >= 0:
                self.tracks[best_ti].bbox = face["bbox"]
                self.tracks[best_ti].frames_missing = 0
                used_t.add(best_ti); used_d.add(di)

        for di, face in enumerate(faces):
            if di not in used_d:
                t = FaceTrack(face["bbox"]); self.tracks.append(t)

        self.tracks = [t for t in self.tracks if t.frames_missing <= self.max_missing]
        return [t for t in self.tracks if t.frames_missing == 0]

    def clear(self): self.tracks.clear()


# ═══════════════════════════════════════════════════════════════════
# DRAWING
# ═══════════════════════════════════════════════════════════════════

def _tint(frame, x1, y1, x2, y2, color, alpha):
    x1,y1,x2,y2 = max(0,int(x1)),max(0,int(y1)),min(frame.shape[1]-1,int(x2)),min(frame.shape[0]-1,int(y2))
    if x2<=x1 or y2<=y1: return
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.addWeighted(
        np.full_like(roi, color, dtype=np.uint8), alpha, roi, 1-alpha, 0)


def draw_face(frame: np.ndarray, track: FaceTrack,
              show_bars: bool, use_rejection: bool):
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = track.bbox
    x1,y1,x2,y2 = max(0,int(x1)),max(0,int(y1)),min(fw-1,int(x2)),min(fh-1,int(y2))
    if x2-x1 < 4 or y2-y1 < 4: return

    is_unknown = use_rejection and track.id_conf < REJECTION_THRESHOLD
    emo   = track.emotion
    ename = IDENTITY_NAMES[int(np.argmax(track.id_sims))] if IDENTITY_NAMES else "?"
    id_color   = UNKNOWN_COLOR if is_unknown else (
        IDENTITY_COLORS[LABEL_TO_IDX.get(ename, 0) % len(IDENTITY_COLORS)]
        if IDENTITY_NAMES else (200, 200, 200))
    emo_color  = EMOTION_COLORS.get(emo, (200, 200, 200))

    # Box — dual-colour corner brackets
    corner = min(20, (x2-x1)//4, (y2-y1)//4)
    thickness = 2
    for sx, sy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (sx, sy), (sx+dx*corner, sy), id_color, thickness)
        cv2.line(frame, (sx, sy), (sx, sy+dy*corner), id_color, thickness)

    # Combined label:  "Name — Emotion"
    id_str  = "Unknown" if is_unknown else ename.capitalize()
    label   = f"{id_str}  |  {emo}  {track.emo_conf:.0%}"
    conf_id = track.id_conf

    font, fsc, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    (tw, th), bl = cv2.getTextSize(label, font, fsc, thick)
    lx = max(0, min(x1, fw-tw-10))
    ly = max(th+8, y1-4)
    _tint(frame, lx, ly-th-6, lx+tw+8, ly+bl, (18,18,18), 0.72)
    cv2.rectangle(frame, (lx, ly-th-6), (lx+tw+8, ly+bl), id_color, 1)
    cv2.putText(frame, label, (lx+4, ly-2), font, fsc, (245,245,245), thick, cv2.LINE_AA)

    # Identity confidence badge (bottom-left of bbox)
    badge = f"{conf_id:.0%}" if not is_unknown else "?"
    cv2.putText(frame, badge, (x1+4, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                id_color, 1, cv2.LINE_AA)

    if show_bars:
        # ── Top-3 expression bars ──────────────────────────────
        bar_w  = min(120, x2-x1)
        bar_h  = 11
        bar_x  = max(0, min(x2+6, fw-bar_w-2))
        by     = y1
        top3_e = np.argsort(track.emo_probs)[::-1][:3]
        for i, ei in enumerate(top3_e):
            byi = by + i*(bar_h+2)
            if byi + bar_h > fh-30: break
            ec  = EMOTION_COLORS.get(EMOTION_NAMES[ei], (180,180,180))
            fill = int(track.emo_probs[ei] * bar_w)
            _tint(frame, bar_x, byi, bar_x+bar_w, byi+bar_h, (25,25,25), 0.70)
            if fill > 0:
                _tint(frame, bar_x, byi, bar_x+fill, byi+bar_h, ec, 0.60)
            cv2.rectangle(frame, (bar_x, byi), (bar_x+bar_w, byi+bar_h), (80,80,80), 1)
            cv2.putText(frame, f"{EMOTION_NAMES[ei][:6]} {track.emo_probs[ei]:.0%}",
                        (bar_x+2, byi+bar_h-2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.29, (235,235,235), 1, cv2.LINE_AA)

        # ── Top-3 identity bars ────────────────────────────────
        bar_x2 = max(0, x1-bar_w-6)
        if bar_x2 < 0: bar_x2 = max(0, min(x2+6, fw-bar_w-2))
        top3_i = np.argsort(track.id_sims)[::-1][:3]
        for i, ii in enumerate(top3_i):
            byi = y1 + i*(bar_h+2)
            if byi + bar_h > fh-30: break
            ic  = IDENTITY_COLORS[ii % len(IDENTITY_COLORS)]
            sim = max(0, track.id_sims[ii])
            fill = int(sim * bar_w)
            _tint(frame, bar_x2, byi, bar_x2+bar_w, byi+bar_h, (25,25,25), 0.70)
            if fill > 0:
                _tint(frame, bar_x2, byi, bar_x2+fill, byi+bar_h, ic, 0.60)
            cv2.rectangle(frame, (bar_x2, byi), (bar_x2+bar_w, byi+bar_h), (80,80,80), 1)
            nm = IDENTITY_NAMES[ii][:8] if IDENTITY_NAMES else "?"
            cv2.putText(frame, f"{nm} {sim:.0%}",
                        (bar_x2+2, byi+bar_h-2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.29, (235,235,235), 1, cv2.LINE_AA)


def draw_dashboard(frame: np.ndarray, tracks: List[FaceTrack],
                   use_rejection: bool, mode: str, fps: float):
    h, w = frame.shape[:2]
    px, py = 10, 10
    pw = 230
    lines = []
    for t in sorted(tracks, key=lambda x: x.id):
        is_unk = use_rejection and t.id_conf < REJECTION_THRESHOLD
        name   = "Unknown" if is_unk else t.identity.capitalize()
        lines.append((name, t.emotion, t.emo_conf, t.id_conf, t.id))

    ph = 52 + len(lines) * 24 + 14
    ph = min(ph, h - py - 10)
    _tint(frame, px, py, px+pw, py+ph, (15,15,15), 0.82)
    cv2.rectangle(frame, (px, py), (px+pw, py+ph), (90,90,90), 1)

    cv2.putText(frame, f"FaceAnalyzer | {mode} | {fps:.0f} FPS",
                (px+8, py+18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
    cv2.putText(frame, f"Faces: {len(lines)}",
                (px+8, py+36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170,170,170), 1)
    cv2.line(frame, (px+6, py+44), (px+pw-6, py+44), (80,80,80), 1)

    for i, (name, emo, econf, iconf, tid) in enumerate(lines):
        ly = py + 58 + i * 24
        if ly + 14 > py + ph: break
        ic = IDENTITY_COLORS[LABEL_TO_IDX.get(name.lower(), i) % len(IDENTITY_COLORS)] \
             if name != "Unknown" else UNKNOWN_COLOR
        ec = EMOTION_COLORS.get(emo, (180, 180, 180))
        cv2.putText(frame, f"#{tid+1}", (px+6, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.33, (140,140,140), 1)
        cv2.putText(frame, name, (px+28, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, ic, 1, cv2.LINE_AA)
        cv2.putText(frame, f"→ {emo} {econf:.0%}", (px+120, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, ec, 1, cv2.LINE_AA)


def draw_hud(frame, mode, fps, use_tta, use_rejection):
    h, w = frame.shape[:2]
    hints = "q:Quit  s:Screenshot  t:TTA  f:Bars  r:Rejection  d:Dashboard"
    if mode == "video": hints += "  SPACE:Pause  ←→:Seek"
    cv2.putText(frame, hints, (10, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120,120,120), 1)


# ═══════════════════════════════════════════════════════════════════
# MODE SELECTOR  (same style as video_expression_analyzer.py)
# ═══════════════════════════════════════════════════════════════════

def mode_selector() -> Tuple[str, Optional[str]]:
    """Show a simple GUI window to pick webcam or video."""
    win = "FaceAnalyzer — Select Mode"
    W, H = 480, 280
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    selected = [None]   # mutable for closure

    def on_mouse(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN: return
        if 60 <= y <= 130:
            if 40 <= x <= W-40: selected[0] = "webcam"
        elif 150 <= y <= 220:
            if 40 <= x <= W-40: selected[0] = "video"

    cv2.setMouseCallback(win, on_mouse)

    while selected[0] is None:
        canvas[:] = (25, 25, 25)
        cv2.putText(canvas, "FaceAnalyzer", (120, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, "Identity + Expression in real-time", (60, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)

        for rect, label, color in [
            ((40, 80, W-40, 130), "Webcam  (live feed)",  (0, 160, 80)),
            ((40,150, W-40, 200), "Video File  (select)", (0, 100, 200)),
        ]:
            cv2.rectangle(canvas, (rect[0],rect[1]), (rect[2],rect[3]), color, -1)
            cv2.rectangle(canvas, (rect[0],rect[1]), (rect[2],rect[3]), (255,255,255), 1)
            cv2.putText(canvas, label,
                        (rect[0]+20, rect[1]+(rect[3]-rect[1])//2+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)

        cv2.putText(canvas, "or press  W = Webcam    V = Video    Q = Quit",
                    (28, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            cv2.destroyWindow(win); return "quit", None
        if key == ord('w'): selected[0] = "webcam"
        if key == ord('v'): selected[0] = "video"

    cv2.destroyWindow(win)
    video_path = None
    if selected[0] == "video":
        video_path = _pick_video()
        if video_path is None:
            print("  No video selected."); return "quit", None
    return selected[0], video_path


def _pick_video() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        p = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files","*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
                       ("All files","*.*")],
            initialdir=str(Path.home()))
        root.destroy()
        return p if p else None
    except Exception:
        p = input("  Enter video path: ").strip().strip('"')
        return p if Path(p).exists() else None


# ═══════════════════════════════════════════════════════════════════
# WORKER THREAD  (detection + inference, never blocks display)
# ═══════════════════════════════════════════════════════════════════

class DetectionWorker:
    def __init__(self, emo_model, dpain_model, centroids,
                 yunet_detectors, device):
        self.emo    = emo_model
        self.dpain  = dpain_model
        self.cents  = centroids
        self.yunets = yunet_detectors
        self.dev    = device
        self.tracker = Tracker()

        self._frame_lock  = threading.Lock()
        self._result_lock = threading.Lock()
        self._latest_frame  = None
        self._latest_tracks: List[FaceTrack] = []
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push_frame(self, frame: np.ndarray):
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def get_tracks(self) -> List[FaceTrack]:
        with self._result_lock:
            return list(self._latest_tracks)

    def stop(self):
        self._running = False
        self._thread.join(timeout=3.0)

    def _run(self):
        while self._running:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.005); continue
            try:
                self._process(frame)
            except Exception:
                traceback.print_exc()
                time.sleep(0.01)

    def _process(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = detect_faces_yunet(self.yunets, frame)
        active_tracks = self.tracker.update(faces)

        if not faces:
            with self._result_lock:
                self._latest_tracks = []
            return

        emo_tensors:  List[torch.Tensor] = []
        dpain_tensors: List[torch.Tensor] = []

        for face in faces:
            # EmoNeXt alignment
            aligned_emo = align_face_yunet(frame_rgb, face, margin=0.25,
                                           target=EMONEXT_IMG_SIZE)
            if aligned_emo is not None:
                emo_tensors.append(preprocess_emonext(aligned_emo))
            else:
                emo_tensors.append(torch.zeros(3, EMONEXT_IMG_SIZE, EMONEXT_IMG_SIZE))

            # DPAIN alignment (MediaPipe-style: just use the same crop + CLAHE)
            aligned_id = _align_dpain_from_yunet(frame_rgb, face,
                                                  margin=0.3, target=DPAIN_IMG_SIZE)
            if aligned_id is not None:
                dpain_tensors.append(preprocess_dpain(aligned_id))
            else:
                dpain_tensors.append(torch.zeros(3, DPAIN_IMG_SIZE, DPAIN_IMG_SIZE))

        emo_probs_arr  = infer_emonext_batch(self.emo, emo_tensors, self.dev)
        dpain_sims_arr = infer_dpain_batch(self.dpain, dpain_tensors, self.cents, self.dev)

        for track, emo_p, id_s in zip(active_tracks, emo_probs_arr, dpain_sims_arr):
            track.update_emotion(emo_p)
            track.update_identity(id_s)

        with self._result_lock:
            self._latest_tracks = list(active_tracks)


def _align_dpain_from_yunet(frame_rgb: np.ndarray, face: dict,
                             margin=0.3, target=224) -> Optional[np.ndarray]:
    """Use YuNet bbox + eye landmarks for DPAIN alignment with CLAHE (matches training)."""
    crop = align_face_yunet(frame_rgb, face, margin=margin, target=target)
    if crop is None: return None
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ═══════════════════════════════════════════════════════════════════
# MAIN LOOP LOGIC
# ═══════════════════════════════════════════════════════════════════

def run_webcam(emo_model, dpain_model, centroids, yunet_detectors, device):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam."); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"  ✅ Webcam: {int(cap.get(3))}×{int(cap.get(4))}")

    worker = DetectionWorker(emo_model, dpain_model, centroids, yunet_detectors, device)
    fps_buf = deque(maxlen=30)
    ss_cnt = 0; show_bars = True; show_dash = True; use_tta = False; use_rej = True
    win = "FaceAnalyzer — Webcam"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret: break

            worker.push_frame(frame)
            tracks = worker.get_tracks()

            for t in tracks:
                draw_face(frame, t, show_bars, use_rej)
            if show_dash:
                draw_dashboard(frame, tracks, use_rej, "Webcam", np.mean(fps_buf) if fps_buf else 0)
            draw_hud(frame, "webcam", np.mean(fps_buf) if fps_buf else 0, use_tta, use_rej)

            fps_buf.append(1.0 / max(time.time()-t0, 1e-6))
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            elif key == ord('s'):
                ss_cnt += 1
                p = SCREENSHOT_DIR / f"webcam_{ss_cnt:04d}.png"
                cv2.imwrite(str(p), frame)
                print(f"  📸 {p.name}")
            elif key == ord('t'): use_tta = not use_tta
            elif key == ord('f'): show_bars = not show_bars
            elif key == ord('r'): use_rej = not use_rej
            elif key == ord('d'): show_dash = not show_dash
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop(); cap.release(); cv2.destroyAllWindows()


def run_video(video_path: str, emo_model, dpain_model, centroids,
              yunet_detectors, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}"); sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ✅ Video: {Path(video_path).name}  {vw}×{vh}  {fps_v:.1f}fps  {total} frames")

    worker = DetectionWorker(emo_model, dpain_model, centroids, yunet_detectors, device)
    fps_buf = deque(maxlen=30)
    ss_cnt = 0; show_bars = True; show_dash = True
    use_tta = False; use_rej = True; paused = True
    frame_idx = 0; writer = None; recording = False
    win = f"FaceAnalyzer — {Path(video_path).name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(vw, 1440), min(vh+50, 860))

    ret, frame = cap.read()
    if not ret: print("  Cannot read first frame."); sys.exit(1)
    cur_frame = frame.copy()

    # person-emotion summary for video
    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    try:
        while True:
            t0 = time.time()

            if not paused:
                ret, frame = cap.read()
                if ret:
                    cur_frame = frame
                    frame_idx += 1
                else:
                    paused = True; print("\n  End of video. Paused.")

            display = cur_frame.copy()
            worker.push_frame(cur_frame)
            tracks = worker.get_tracks()

            for t in tracks:
                draw_face(display, t, show_bars, use_rej)
                # accumulate summary
                if not paused:
                    is_unk = use_rej and t.id_conf < REJECTION_THRESHOLD
                    pname  = "Unknown" if is_unk else t.identity.capitalize()
                    summary[pname][t.emotion] += 1

            if show_dash:
                draw_dashboard(display, tracks, use_rej, "Video",
                               np.mean(fps_buf) if fps_buf else 0)

            # Progress bar
            prog = frame_idx / max(total-1, 1)
            bh_pbar = 8
            bary = display.shape[0] - bh_pbar - 2
            barw = display.shape[1]
            cv2.rectangle(display, (0, bary), (barw, bary+bh_pbar), (50,50,50), -1)
            cv2.rectangle(display, (0, bary), (int(prog*barw), bary+bh_pbar), (0,140,255), -1)
            ts = f"{int(frame_idx/fps_v//60):02d}:{int(frame_idx/fps_v%60):02d} / " \
                 f"{int(total/fps_v//60):02d}:{int(total/fps_v%60):02d}"
            cv2.putText(display, ts, (6, bary-4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180,180,180), 1)
            if paused:
                cv2.putText(display, "PAUSED", (barw-80, bary-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,100), 1)
            if recording:
                cv2.circle(display, (display.shape[1]-18, 18), 8, (0,0,220), -1)

            draw_hud(display, "video", np.mean(fps_buf) if fps_buf else 0, use_tta, use_rej)

            if recording and writer is not None:
                writer.write(display)

            fps_buf.append(1.0 / max(time.time()-t0, 1e-6))
            cv2.imshow(win, display)

            wait = 30 if paused else max(1, int(1000/fps_v - (time.time()-t0)*1000))
            key  = cv2.waitKey(wait) & 0xFF
            if key in (ord('q'), 27): break
            elif key == ord(' '): paused = not paused
            elif key == ord('s'):
                ss_cnt += 1
                p = SCREENSHOT_DIR / f"video_{ss_cnt:04d}.png"
                cv2.imwrite(str(p), display); print(f"  📸 {p.name}")
            elif key == ord('t'): use_tta = not use_tta
            elif key == ord('f'): show_bars = not show_bars
            elif key == ord('r'): use_rej = not use_rej
            elif key == ord('d'): show_dash = not show_dash
            elif key == 81:   # LEFT
                tgt = max(0, frame_idx - int(5*fps_v))
                cap.set(cv2.CAP_PROP_POS_FRAMES, tgt); frame_idx = tgt
                ret, f = cap.read()
                if ret: cur_frame = f
                worker.tracker.clear()
            elif key == 83:   # RIGHT
                tgt = min(total-1, frame_idx + int(5*fps_v))
                cap.set(cv2.CAP_PROP_POS_FRAMES, tgt); frame_idx = tgt
                ret, f = cap.read()
                if ret: cur_frame = f
                worker.tracker.clear()
            elif key == ord('r') and not recording:
                pass  # r is rejection — use capital R or separate key if needed

    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        if writer: writer.release()
        cap.release(); cv2.destroyAllWindows()

        # Print who-did-what summary
        if summary:
            print("\n" + "═"*58)
            print("  Video Summary — Identity × Expression")
            print("═"*58)
            for person, emos in sorted(summary.items()):
                total_p = sum(emos.values())
                top_emo = max(emos, key=emos.get)
                print(f"  {person:<18s}  ({total_p:>5d} detections)   "
                      f"dominant: {top_emo}")
                for emo, cnt in sorted(emos.items(), key=lambda x: -x[1])[:4]:
                    pct = cnt/total_p*100
                    bar = "█" * int(pct/4)
                    print(f"    {emo:<12s} {cnt:>5d} ({pct:4.0f}%) {bar}")
            print("═"*58)


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Combined Identity + Expression Analyzer")
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--webcam", action="store_true")
    args = parser.parse_args()

    print("\n" + "═"*58)
    print("  FaceAnalyzer — Identity + Expression")
    print("═"*58)
    print(f"  Device : {DEVICE}")

    # ── Load models ──
    print("\n  Loading models...")
    emo_model               = load_emonext(DEVICE)
    dpain_model, centroids  = load_dpain(DEVICE)
    print()

    # ── Load YuNet ──
    yunet_path = resolve_yunet()
    print(f"  ✅ YuNet: {yunet_path.name}")
    print("  (Face detectors initialised per-frame for thread safety)\n")
    # We pass the path; each worker creates its own detectors inside its thread
    # to avoid thread-safety issues with cv2.FaceDetectorYN.
    # Pre-create them here just to verify the file loads:
    _test_detectors = create_yunet(yunet_path, 640, 480)
    del _test_detectors

    # Store yunet path so worker can recreate detectors
    global _YUNET_PATH
    _YUNET_PATH = yunet_path

    # ── Choose mode ──
    if args.webcam:
        mode, video_path = "webcam", None
    elif args.video:
        mode, video_path = "video", args.video
    else:
        mode, video_path = mode_selector()

    if mode == "quit":
        print("  Bye!"); sys.exit(0)

    print(f"  Mode: {mode.upper()}")
    print("═"*58 + "\n")

    # Build detectors (will be rebuilt thread-safe inside worker)
    # We use a factory approach — worker creates its own copy
    class YuNetFactory:
        def __init__(self, path, w=1280, h=720):
            self.path, self.w, self.h = path, w, h
        def build(self, w=None, h=None):
            return create_yunet(self.path, w or self.w, h or self.h)

    yunet_factory = create_yunet(yunet_path, 1280, 720)  # default size, resized per-frame

    if mode == "webcam":
        run_webcam(emo_model, dpain_model, centroids, yunet_factory, DEVICE)
    else:
        run_video(video_path, emo_model, dpain_model, centroids, yunet_factory, DEVICE)


if __name__ == "__main__":
    main()
