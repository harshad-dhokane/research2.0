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

    # Dynamic scale: reference designed for 720p
    sc   = max(0.5, min(fh / 720.0, 2.5))
    bw   = x2 - x1

    is_unknown = use_rejection and track.id_conf < REJECTION_THRESHOLD
    emo   = track.emotion
    ename = IDENTITY_NAMES[int(np.argmax(track.id_sims))] if IDENTITY_NAMES else "?"
    id_color  = UNKNOWN_COLOR if is_unknown else (
        IDENTITY_COLORS[LABEL_TO_IDX.get(ename, 0) % len(IDENTITY_COLORS)]
        if IDENTITY_NAMES else (200, 200, 200))
    emo_color = EMOTION_COLORS.get(emo, (200, 200, 200))

    # Corner bracket brackets — scaled, thicker
    corner    = min(int(36*sc), bw//4, (y2-y1)//4)
    thickness = max(2, int(3*sc))
    for sx, sy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (sx, sy), (sx+dx*corner, sy), id_color, thickness)
        cv2.line(frame, (sx, sy), (sx, sy+dy*corner), id_color, thickness)

    # ── Main label: two lines stacked ──────────────────────────
    id_str   = "Unknown" if is_unknown else ename.capitalize()
    conf_id  = track.id_conf
    font     = cv2.FONT_HERSHEY_SIMPLEX
    # Scale font to be clearly readable regardless of resolution
    fsc_name = max(0.70, min(1.20, bw / 180.0 * sc))
    fsc_emo  = max(0.58, min(1.00, bw / 200.0 * sc))
    thick    = max(2, int(2 * sc))
    pad      = max(6, int(7 * sc))

    # Line 1: identity name (large)
    line1 = id_str
    (tw1, th1), bl1 = cv2.getTextSize(line1, font, fsc_name, thick)
    # Line 2: emotion + confidence (slightly smaller)
    line2 = f"{emo}  {track.emo_conf:.0%}"
    (tw2, th2), bl2 = cv2.getTextSize(line2, font, fsc_emo, thick)

    box_w  = max(tw1, tw2) + pad * 2
    box_h  = th1 + th2 + pad * 3 + bl1
    lx     = max(0, min(x1, fw - box_w - 4))
    ly_top = max(0, y1 - box_h - 4)

    # Dark semi-transparent background
    _tint(frame, lx, ly_top, lx + box_w, ly_top + box_h, (10, 10, 10), 0.80)
    cv2.rectangle(frame, (lx, ly_top), (lx + box_w, ly_top + box_h), id_color, max(1, thickness - 1))

    # Draw name in identity colour
    cv2.putText(frame, line1,
                (lx + pad, ly_top + pad + th1),
                font, fsc_name, id_color, thick, cv2.LINE_AA)
    # Draw emotion in emotion colour, below the name
    y_emo = ly_top + pad * 2 + th1 + th2
    cv2.putText(frame, line2,
                (lx + pad, y_emo),
                font, fsc_emo, EMOTION_COLORS.get(emo, (240, 240, 240)), thick, cv2.LINE_AA)

    # Identity confidence badge (bottom-left of bbox)
    badge     = f"ID {conf_id:.0%}" if not is_unknown else "?"
    fsc_badge = max(0.45, 0.55 * sc)
    (bw2, bh2), _ = cv2.getTextSize(badge, font, fsc_badge, max(1, int(sc)))
    _tint(frame, x1, y2 - bh2 - 8, x1 + bw2 + 8, y2 + 2, (10, 10, 10), 0.70)
    cv2.putText(frame, badge, (x1 + 4, y2 - 4), font,
                fsc_badge, id_color, max(1, int(sc)), cv2.LINE_AA)

    if show_bars:
        bar_h   = max(18, int(24 * sc))
        bar_w   = min(int(200 * sc), bw)
        bar_fsc = max(0.38, 0.46 * sc)

        # ── Top-3 expression bars (right side of face) ────────
        bar_x = max(0, min(x2+6, fw-bar_w-2))
        top3_e = np.argsort(track.emo_probs)[::-1][:3]
        for i, ei in enumerate(top3_e):
            byi = y1 + i*(bar_h+3)
            if byi + bar_h > fh-30: break
            ec   = EMOTION_COLORS.get(EMOTION_NAMES[ei], (180,180,180))
            fill = int(track.emo_probs[ei] * bar_w)
            _tint(frame, bar_x, byi, bar_x+bar_w, byi+bar_h, (25,25,25), 0.70)
            if fill > 0:
                _tint(frame, bar_x, byi, bar_x+fill, byi+bar_h, ec, 0.65)
            cv2.rectangle(frame, (bar_x, byi), (bar_x+bar_w, byi+bar_h), (90,90,90), 1)
            cv2.putText(frame, f"{EMOTION_NAMES[ei][:7]} {track.emo_probs[ei]:.0%}",
                        (bar_x+3, byi+bar_h-3), font, bar_fsc, (235,235,235), 1, cv2.LINE_AA)

        # ── Top-3 identity bars (left side, fallback to right) ─
        bar_x2 = x1 - bar_w - 6
        if bar_x2 < 0: bar_x2 = max(0, min(x2+6, fw-bar_w-2))
        top3_i = np.argsort(track.id_sims)[::-1][:3]
        for i, ii in enumerate(top3_i):
            byi = y1 + i*(bar_h+3)
            if byi + bar_h > fh-30: break
            ic   = IDENTITY_COLORS[ii % len(IDENTITY_COLORS)]
            sim  = max(0.0, float(track.id_sims[ii]))
            fill = int(sim * bar_w)
            _tint(frame, bar_x2, byi, bar_x2+bar_w, byi+bar_h, (25,25,25), 0.70)
            if fill > 0:
                _tint(frame, bar_x2, byi, bar_x2+fill, byi+bar_h, ic, 0.65)
            cv2.rectangle(frame, (bar_x2, byi), (bar_x2+bar_w, byi+bar_h), (90,90,90), 1)
            nm = IDENTITY_NAMES[ii][:9] if IDENTITY_NAMES else "?"
            cv2.putText(frame, f"{nm} {sim:.0%}",
                        (bar_x2+3, byi+bar_h-3), font, bar_fsc, (235,235,235), 1, cv2.LINE_AA)


def draw_dashboard(frame_h: int, tracks: List[FaceTrack],
                   use_rejection: bool, mode: str, fps: float) -> np.ndarray:
    """Build and return a scaled side panel to np.hstack with the video frame."""
    sc   = max(0.8, min(frame_h / 720.0, 3.0))
    pw   = int(380 * sc)          # wider panel
    pad  = int(18 * sc)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs_title = 1.1  * sc          # header title
    fs_norm  = 0.85 * sc          # sub-header / face count
    fs_item  = 0.82 * sc          # name + emotion label
    fs_small = 0.65 * sc          # section headers / hints
    th2  = max(2, int(2 * sc))
    th1  = max(1, int(sc))
    ls   = int(44 * sc)           # per-face row height

    panel = np.zeros((frame_h, pw, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)

    # Vertical divider line
    cv2.line(panel, (0, 0), (0, frame_h), (70, 70, 70), 2)

    lines = []
    for t in sorted(tracks, key=lambda x: x.id):
        is_unk = use_rejection and t.id_conf < REJECTION_THRESHOLD
        name   = "Unknown" if is_unk else t.identity.capitalize()
        lines.append((t.id, name, t.emotion, t.emo_conf, t.id_conf))

    y = int(34 * sc)

    # ── Title ──────────────────────────────────────────────────
    cv2.putText(panel, "FaceAnalyzer", (pad, y), font, fs_title,
                (0, 220, 255), th2, cv2.LINE_AA)
    y += int(34 * sc)
    cv2.putText(panel, f"{mode}  |  {fps:.0f} FPS", (pad, y), font,
                fs_small, (160, 160, 160), th1, cv2.LINE_AA)
    y += int(30 * sc)
    cv2.putText(panel, f"Faces detected: {len(lines)}", (pad, y), font,
                fs_norm, (220, 220, 220), th1, cv2.LINE_AA)
    y += int(ls * 0.85)
    cv2.line(panel, (pad, y), (pw - pad, y), (80, 80, 80), max(1, int(sc)))
    y += int(20 * sc)
    cv2.putText(panel, "Identity  |  Expression", (pad, y), font,
                fs_small, (0, 200, 255), th1, cv2.LINE_AA)
    y += int(28 * sc)

    bh  = max(16, int(20 * sc))   # bar height — taller and easier to read
    bar_max = pw - pad * 2

    for tid, name, emo, econf, iconf in lines:
        if y + int(ls * 3.2) > frame_h - pad: break
        ic = IDENTITY_COLORS[LABEL_TO_IDX.get(name.lower(), tid) % len(IDENTITY_COLORS)] \
             if name != "Unknown" else UNKNOWN_COLOR
        ec = EMOTION_COLORS.get(emo, (180, 180, 180))

        # ── Face # + Name (large, coloured) ──────────────────
        cv2.putText(panel, f"#{tid+1}", (pad, y), font,
                    fs_small, (140, 140, 140), th1, cv2.LINE_AA)
        cv2.putText(panel, name, (pad + int(42 * sc), y), font,
                    fs_item * 1.10, ic, th2, cv2.LINE_AA)
        y += int(36 * sc)

        # ── Expression label (large, emotion colour) ─────────
        cv2.putText(panel, f"{emo}  {econf:.0%}",
                    (pad + int(42 * sc), y), font,
                    fs_item, EMOTION_COLORS.get(emo, (220, 220, 220)), th2, cv2.LINE_AA)
        y += int(30 * sc)

        # ── Expression confidence bar ─────────────────────────
        fill = int(max(0.0, econf) * bar_max)
        _tint(panel, pad, y, pad + bar_max, y + bh, (40, 40, 40), 0.85)
        if fill > 0:
            _tint(panel, pad, y, pad + fill, y + bh, ec, 0.70)
        cv2.rectangle(panel, (pad, y), (pad + bar_max, y + bh), (80, 80, 80), 1)
        cv2.putText(panel, f"{emo[:8]}  {econf:.0%}",
                    (pad + 4, y + bh - 4), font,
                    max(0.42, 0.50 * sc), (240, 240, 240), 1, cv2.LINE_AA)
        y += bh + int(8 * sc)

        # ── Identity similarity bar ───────────────────────────
        id_fill = int(max(0.0, iconf) * bar_max)
        _tint(panel, pad, y, pad + bar_max, y + bh, (40, 40, 40), 0.85)
        if id_fill > 0:
            _tint(panel, pad, y, pad + id_fill, y + bh, ic, 0.65)
        cv2.rectangle(panel, (pad, y), (pad + bar_max, y + bh), (80, 80, 80), 1)
        cv2.putText(panel, f"ID conf: {iconf:.0%}",
                    (pad + 4, y + bh - 4), font,
                    max(0.42, 0.50 * sc), (220, 220, 220), 1, cv2.LINE_AA)
        y += bh + int(16 * sc)

        cv2.line(panel, (pad, y), (pw - pad, y), (60, 60, 60), 1)
        y += int(14 * sc)

    # ── Controls hint at bottom ───────────────────────────────
    ctrl = ["q:Quit  s:Screenshot",
            "b:Bars  r:Rejection  d:Dashboard"]
    if mode.lower() == "video":
        ctrl.append("SPACE:Pause  ←/→:Seek 5s")
    yb = frame_h - int(len(ctrl) * 22 * sc) - pad
    for ht in ctrl:
        if yb > y:
            cv2.putText(panel, ht, (pad, yb), font,
                        max(0.38, 0.44 * sc), (95, 95, 95), 1, cv2.LINE_AA)
        yb += int(22 * sc)

    return panel


def draw_hud(frame, mode, fps, use_tta, use_rejection, show_bars):
    h, w = frame.shape[:2]
    sc  = max(0.6, min(h / 720.0, 2.0))
    fsc = max(0.45, 0.52 * sc)
    status = (f"FPS:{fps:.0f}  "
              f"Bars:{'ON' if show_bars else 'off'}  "
              f"Rej:{'ON' if use_rejection else 'off'}  "
              f"TTA:{'ON' if use_tta else 'off'}  "
              "| b:Bars  r:Rej  d:Dash  s:Shot  q:Quit")
    (sw, sh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, fsc, 1)
    _tint(frame, 0, h - sh - 14, sw + 12, h, (10, 10, 10), 0.65)
    cv2.putText(frame, status,
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, fsc, (170, 170, 170), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
# MODE SELECTOR  (same style as video_expression_analyzer.py)
# ═══════════════════════════════════════════════════════════════════

def mode_selector() -> Tuple[str, Optional[str]]:
    """Terminal-based mode selector (avoids Qt NULL-handle race on Wayland/X11)."""
    print("\n" + "─"*52)
    print("  Select mode:")
    print("    [1]  Webcam  (live feed)")
    print("    [2]  Video file")
    print("    [q]  Quit")
    print("─"*52)
    while True:
        try:
            choice = input("  Choice [1/2/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "quit", None
        if choice in ("q", "quit", "exit"):
            return "quit", None
        if choice in ("1", "w", "webcam"):
            return "webcam", None
        if choice in ("2", "v", "video"):
            video_path = _pick_video()
            if video_path is None:
                print("  No video selected.")
                return "quit", None
            return "video", video_path
        print("  Please enter 1, 2, or q.")


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
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ✅ Webcam: {fw}×{fh}")

    # Side-panel width (compute once from frame height)
    _sc = max(0.8, min(fh / 720.0, 3.0))
    panel_w = int(320 * _sc)

    worker = DetectionWorker(emo_model, dpain_model, centroids, yunet_detectors, device)
    fps_buf = deque(maxlen=30)
    ss_cnt = 0; show_bars = True; show_dash = True; use_tta = False; use_rej = True
    win = "FaceAnalyzer — Webcam"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1920, fw + panel_w), min(1080, fh))

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret: break

            worker.push_frame(frame)
            tracks = worker.get_tracks()
            fps_now = float(np.mean(fps_buf)) if fps_buf else 0.0

            for t in tracks:
                draw_face(frame, t, show_bars, use_rej)
            draw_hud(frame, "webcam", fps_now, use_tta, use_rej, show_bars)

            if show_dash:
                panel   = draw_dashboard(fh, tracks, use_rej, "Webcam", fps_now)
                display = np.hstack([frame, panel])
            else:
                display = frame

            fps_buf.append(1.0 / max(time.time()-t0, 1e-6))
            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            elif key == ord('s'):
                ss_cnt += 1
                p = SCREENSHOT_DIR / f"webcam_{ss_cnt:04d}.png"
                cv2.imwrite(str(p), display)
                print(f"  📸 {p.name}")
            elif key == ord('t'): use_tta = not use_tta
            elif key == ord('b'): show_bars = not show_bars
            elif key == ord('r'): use_rej = not use_rej
            elif key == ord('d'):
                show_dash = not show_dash
                cv2.resizeWindow(win,
                    min(1920, fw + (panel_w if show_dash else 0)), min(1080, fh))
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

    # Side-panel width
    _sc = max(0.8, min(vh / 720.0, 3.0))
    panel_w = int(320 * _sc)

    worker = DetectionWorker(emo_model, dpain_model, centroids, yunet_detectors, device)
    fps_buf = deque(maxlen=30)
    ss_cnt = 0; show_bars = True; show_dash = True
    use_tta = False; use_rej = True; paused = True
    frame_idx = 0; writer = None; recording = False
    win = f"FaceAnalyzer — {Path(video_path).name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1920, vw + panel_w), min(1080, vh))

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
            fps_now = float(np.mean(fps_buf)) if fps_buf else 0.0

            for t in tracks:
                draw_face(display, t, show_bars, use_rej)
                if not paused:
                    is_unk = use_rej and t.id_conf < REJECTION_THRESHOLD
                    pname  = "Unknown" if is_unk else t.identity.capitalize()
                    summary[pname][t.emotion] += 1

            # Progress bar (drawn on video frame before panel is added)
            sc_pbar = max(0.5, min(vh / 720.0, 2.5))
            pbar_h  = max(8, int(22 * sc_pbar))
            bary    = display.shape[0] - pbar_h - 2
            prog    = frame_idx / max(total - 1, 1)
            cv2.rectangle(display, (0, bary), (vw, bary + pbar_h), (40, 40, 40), -1)
            cv2.rectangle(display, (0, bary),
                          (int(prog * vw), bary + pbar_h), (0, 140, 255), -1)
            ts = (f"{int(frame_idx/fps_v//60):02d}:{int(frame_idx/fps_v%60):02d} / "
                  f"{int(total/fps_v//60):02d}:{int(total/fps_v%60):02d}")
            cv2.putText(display, ts, (8, bary - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.35, 0.42*sc_pbar),
                        (180, 180, 180), 1)
            if paused:
                cv2.putText(display, "PAUSED", (vw - int(90*sc_pbar), bary - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.38, 0.44*sc_pbar),
                            (200, 200, 100), 1)
            if recording:
                cv2.circle(display, (vw - 18, 18), max(6, int(8*sc_pbar)), (0, 0, 220), -1)

            draw_hud(display, "video", fps_now, use_tta, use_rej, show_bars)

            if show_dash:
                panel   = draw_dashboard(vh, tracks, use_rej, "Video", fps_now)
                out     = np.hstack([display, panel])
            else:
                out = display

            if recording and writer is not None:
                writer.write(out)

            fps_buf.append(1.0 / max(time.time() - t0, 1e-6))
            cv2.imshow(win, out)

            wait = 30 if paused else max(1, int(1000/fps_v - (time.time()-t0)*1000))
            key  = cv2.waitKey(wait) & 0xFF
            if key in (ord('q'), 27): break
            elif key == ord(' '): paused = not paused
            elif key == ord('s'):
                ss_cnt += 1
                p = SCREENSHOT_DIR / f"video_{ss_cnt:04d}.png"
                cv2.imwrite(str(p), out); print(f"  📸 {p.name}")
            elif key == ord('t'): use_tta = not use_tta
            elif key == ord('b'): show_bars = not show_bars
            elif key == ord('r'): use_rej = not use_rej
            elif key == ord('d'):
                show_dash = not show_dash
                cv2.resizeWindow(win,
                    min(1920, vw + (panel_w if show_dash else 0)), min(1080, vh))
            elif key == 81:   # LEFT arrow
                tgt = max(0, frame_idx - int(5*fps_v))
                cap.set(cv2.CAP_PROP_POS_FRAMES, tgt); frame_idx = tgt
                ret, f = cap.read()
                if ret: cur_frame = f
                worker.tracker.clear()
            elif key == 83:   # RIGHT arrow
                tgt = min(total - 1, frame_idx + int(5*fps_v))
                cap.set(cv2.CAP_PROP_POS_FRAMES, tgt); frame_idx = tgt
                ret, f = cap.read()
                if ret: cur_frame = f
                worker.tracker.clear()

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
