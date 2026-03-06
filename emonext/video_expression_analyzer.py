#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════╗
║  EmoNeXt — Video Expression Analyzer (Multi-Face, Classroom Ready)    ║
║  Detects ALL faces via YuNet, classifies emotion with EmoNeXt         ║
║  Draws bounding boxes + expression labels for every face              ║
╚═══════════════════════════════════════════════════════════════════════╝

Features:
  • Mode selection GUI: choose Webcam Live or Video File
  • File dialog to select any video from the device
  • YuNet multi-face detection (handles classrooms with many faces)
  • EmoNeXt model loaded from emonext_outputs/saved_models/
  • Per-face bounding box with emotion label + confidence bar
  • Temporal smoothing (EMA) for stable per-face predictions
  • Simple face tracking across frames via IoU matching
  • Live stats panel showing emotion distribution
  • Option to save annotated output video

Controls (Video mode):
  SPACE  : Pause / Resume
  q/ESC  : Quit
  s      : Screenshot
  r      : Start/Stop recording annotated video
  d      : Toggle dashboard panel
  ←/→    : Seek ±5 seconds
  ↑/↓    : Increase/Decrease playback speed
  1-9    : Jump to 10%-90% of video

Controls (Webcam mode):
  q/ESC  : Quit
  s      : Screenshot
  r      : Start/Stop recording
  d      : Toggle dashboard panel

Usage:
  python video_expression_analyzer.py                     # opens mode selector
  python video_expression_analyzer.py --webcam            # webcam directly
  python video_expression_analyzer.py path/to/video.mp4   # direct video path
  python video_expression_analyzer.py --device cuda video.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = SCRIPT_DIR / "saved_models"
SCREENSHOT_DIR = SCRIPT_DIR / "screenshots"
ANNOTATED_DIR = SCRIPT_DIR / "annotated_videos"

for _d in [SCREENSHOT_DIR, ANNOTATED_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"}

YUNET_MODEL_CANDIDATES = [
    PROJECT_ROOT / "Affectnet_v2" / "face_detection_yunet_2023mar.onnx",
    PROJECT_ROOT / "face_detection_yunet_2023mar.onnx",
    SCRIPT_DIR / "face_detection_yunet_2023mar.onnx",
]

CLASS_NAMES = ["Neutral", "Happy", "Sad", "Surprise",
               "Fear", "Disgust", "Anger", "Contempt"]
NUM_CLASSES = 8
IMG_SIZE = 224

# Normalization used during training (mean=0.5, std=0.5)
NORM_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
NORM_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# BGR colors for OpenCV
EMOTION_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Neutral":  (200, 200, 200),
    "Happy":    (0, 230, 118),
    "Sad":      (255, 152, 0),
    "Surprise": (255, 191, 0),
    "Fear":     (219, 112, 147),
    "Disgust":  (0, 100, 0),
    "Anger":    (0, 0, 255),
    "Contempt": (0, 180, 180),
}

EMOTION_EMOJI: Dict[str, str] = {
    "Neutral": "😐", "Happy": "😊", "Sad": "😢", "Surprise": "😲",
    "Fear": "😨", "Disgust": "🤢", "Anger": "😠", "Contempt": "😏",
}


# ═══════════════════════════════════════════════════════════════════
# EMONEXT ARCHITECTURE  (mirrors the notebook exactly)
# ═══════════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class EmoNeXtBlock(nn.Module):
    def __init__(self, channels, expand_ratio=3, se_ratio=4, drop_path=0.0):
        super().__init__()
        expanded = channels * expand_ratio
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, expanded, 1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.GELU(),
            SqueezeExcite(expanded, reduction=se_ratio),
            nn.Conv2d(expanded, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.block(x))


class MultiFrequencySpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.freq_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            ) for d in [1, 2, 3]
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        freq_outs = [branch(x) for branch in self.freq_branches]
        gate = self.fuse(torch.cat(freq_outs, dim=1))
        return x * gate + x


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class FacialRegionAwarePooling(nn.Module):
    def __init__(self, channels, num_regions=4):
        super().__init__()
        self.num_regions = num_regions
        self.gate = nn.Sequential(
            nn.Linear(channels * (num_regions + 1), 256),
            nn.GELU(),
            nn.Linear(256, num_regions + 1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        global_feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
        h_mid, w_mid = H // 2, W // 2
        regions = [
            x[:, :, :h_mid, :w_mid],
            x[:, :, :h_mid, w_mid:],
            x[:, :, h_mid:, :w_mid],
            x[:, :, h_mid:, w_mid:],
        ]
        region_feats = [F.adaptive_avg_pool2d(r, 1).flatten(1) for r in regions]
        all_feats = [global_feat] + region_feats
        all_cat = torch.cat(all_feats, dim=-1)
        gates = self.gate(all_cat)
        stacked = torch.stack(all_feats, dim=1)
        fused = (stacked * gates.unsqueeze(-1)).reshape(B, -1)
        return fused, gates


class EmoNeXt(nn.Module):
    def __init__(self, num_classes=8, channels=[64, 128, 256, 384],
                 depths=[2, 3, 4, 2], expand_ratio=3, se_ratio=4,
                 dropout=0.2, drop_path=0.15, use_mfsa=True):
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0] // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0] // 2),
            nn.GELU(),
            nn.Conv2d(channels[0] // 2, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, total_blocks)]
        block_idx = 0
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for stage_i in range(4):
            blocks = []
            for j in range(depths[stage_i]):
                blocks.append(EmoNeXtBlock(
                    channels[stage_i], expand_ratio, se_ratio,
                    drop_path=dp_rates[block_idx]
                ))
                block_idx += 1
            if use_mfsa and stage_i >= 2:
                blocks.append(MultiFrequencySpatialAttention(channels[stage_i]))
            self.stages.append(nn.Sequential(*blocks))
            if stage_i < 3:
                self.downsamples.append(Downsample(channels[stage_i], channels[stage_i + 1]))

        self.frap = FacialRegionAwarePooling(channels[-1])
        frap_dim = channels[-1] * 5
        self.head = nn.Sequential(
            nn.Linear(frap_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self.embed_proj = nn.Linear(frap_dim, 256)

    def forward_features(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        pooled, region_gates = self.frap(features)
        logits = self.head(pooled)
        embeddings = self.embed_proj(pooled)
        return {
            'logits': logits,
            'embeddings': embeddings,
            'region_gates': region_gates,
        }


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════
def load_model(device: torch.device) -> EmoNeXt:
    """Load EmoNeXt from saved checkpoint."""
    ckpt_candidates = [
        MODEL_DIR / "emonext_final.pth",
        MODEL_DIR / "best_model.pth",
    ]
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No EmoNeXt checkpoint found. Tried:\n" +
            "\n".join(f"  - {p}" for p in ckpt_candidates)
        )

    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = data.get("cfg", {})

    # Build model with saved config
    channels = cfg.get("CHANNELS", [64, 128, 256, 384])
    depths = cfg.get("DEPTHS", [2, 3, 4, 2])
    expand_ratio = int(cfg.get("EXPAND_RATIO", 3))
    se_ratio = int(cfg.get("SE_RATIO", 4))
    dropout = float(cfg.get("DROPOUT", 0.2))
    drop_path = float(cfg.get("DROP_PATH", 0.15))
    use_mfsa = cfg.get("USE_MFSA", True)

    model = EmoNeXt(
        num_classes=NUM_CLASSES,
        channels=channels,
        depths=depths,
        expand_ratio=expand_ratio,
        se_ratio=se_ratio,
        dropout=dropout,
        drop_path=drop_path,
        use_mfsa=use_mfsa,
    )

    state = data.get("model_state") or data.get("model_state_dict") or data
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device).eval()

    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✅ EmoNeXt loaded: {ckpt_path.name} ({total_p:.2f}M params)")
    if missing:
        print(f"     ⚠ Missing keys: {len(missing)}")
    if unexpected:
        print(f"     ⚠ Unexpected keys: {len(unexpected)}")
    val_acc = data.get("val_acc", "?")
    val_f1 = data.get("val_f1", "?")
    if isinstance(val_acc, (int, float)):
        print(f"     Val Acc: {val_acc*100:.1f}%, Val F1: {val_f1*100:.1f}%")

    return model


# ═══════════════════════════════════════════════════════════════════
# YUNET FACE DETECTION
# ═══════════════════════════════════════════════════════════════════
def resolve_yunet() -> Path:
    for c in YUNET_MODEL_CANDIDATES:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        "YuNet ONNX model not found. Tried:\n" +
        "\n".join(f"  - {p}" for p in YUNET_MODEL_CANDIDATES)
    )


def create_yunet_detectors(model_path: Path, frame_w: int, frame_h: int):
    """Create multi-scale YuNet detectors for robust face detection.
    Always includes 1.0 (full res) for small-face coverage."""
    scales = [1.0, 0.5]
    detectors = []
    for scale in scales:
        dw, dh = int(frame_w * scale), int(frame_h * scale)
        det = cv2.FaceDetectorYN.create(
            str(model_path), "", (dw, dh),
            score_threshold=0.5, nms_threshold=0.3, top_k=5000
        )
        detectors.append((det, scale))
    return detectors


def detect_faces(detectors, frame: np.ndarray) -> List[dict]:
    """Detect faces using multi-scale YuNet."""
    all_faces = []
    for detector, scale in detectors:
        if abs(scale - 1.0) > 1e-6:
            small = cv2.resize(frame, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_LINEAR)
        else:
            small = frame
        detector.setInputSize((small.shape[1], small.shape[0]))
        _, raw = detector.detect(small)
        if raw is None:
            continue
        inv = 1.0 / max(scale, 1e-6)
        MIN_FACE_PX = 24  # ignore tiny spurious detections
        for f in raw:
            score = float(f[14])
            if score < 0.5:
                continue
            x1 = int(f[0] * inv)
            y1 = int(f[1] * inv)
            bw = int(f[2] * inv)
            bh = int(f[3] * inv)
            # Filter: minimum size
            if bw < MIN_FACE_PX or bh < MIN_FACE_PX:
                continue
            # Filter: reasonable aspect ratio (0.5 – 2.0)
            aspect = bw / max(bh, 1)
            if aspect < 0.5 or aspect > 2.0:
                continue
            # Filter: eyes must be inside bbox and separated
            le_x, le_y = float(f[4] * inv), float(f[5] * inv)
            re_x, re_y = float(f[6] * inv), float(f[7] * inv)
            eye_dist = ((le_x - re_x) ** 2 + (le_y - re_y) ** 2) ** 0.5
            if eye_dist < bw * 0.15:  # eyes too close = not a real face
                continue
            all_faces.append({
                "bbox": (x1, y1, x1 + bw, y1 + bh),
                "left_eye": (le_x, le_y),
                "right_eye": (re_x, re_y),
                "score": score,
            })

    # NMS
    if not all_faces:
        return []
    all_faces.sort(key=lambda f: f["score"], reverse=True)
    kept = []
    for cand in all_faces:
        overlap = False
        for k in kept:
            if _bbox_iou(cand["bbox"], k["bbox"]) > 0.35:
                overlap = True
                break
        if not overlap:
            kept.append(cand)
    return kept


def _bbox_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


# ═══════════════════════════════════════════════════════════════════
# FACE PREPROCESSING (matches training pipeline)
# ═══════════════════════════════════════════════════════════════════
def crop_face(frame_rgb: np.ndarray, face: dict,
              margin: float = 0.25) -> Optional[np.ndarray]:
    """Crop and resize face to 224×224, matching training preprocessing."""
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = face["bbox"]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)

    # Add margin and make square
    size = max(bw, bh)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = size * (1.0 + margin) / 2.0

    nx1 = max(0, int(cx - half))
    ny1 = max(0, int(cy - half))
    nx2 = min(w, int(cx + half))
    ny2 = min(h, int(cy + half))

    crop = frame_rgb[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None

    face_img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return face_img


def preprocess_face(face_rgb: np.ndarray) -> torch.Tensor:
    """Normalize face image to match training: (x/255 - 0.5) / 0.5."""
    img = face_rgb.astype(np.float32) / 255.0
    img = (img - NORM_MEAN) / NORM_STD
    tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC → CHW
    return tensor


# ═══════════════════════════════════════════════════════════════════
# FACE TRACKER (simple IoU-based)
# ═══════════════════════════════════════════════════════════════════
class FaceTrack:
    """Per-face track with temporal EMA smoothing."""
    def __init__(self, track_id: int, bbox: tuple, ema_alpha: float = 0.4):
        self.track_id = track_id
        self.bbox = bbox
        self.ema_alpha = ema_alpha
        self.probs = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
        self.frames_seen = 0
        self.frames_missing = 0

    def update(self, bbox: tuple, raw_probs: np.ndarray):
        self.bbox = bbox
        self.frames_seen += 1
        self.frames_missing = 0
        alpha = min(self.ema_alpha, 1.0)
        self.probs = alpha * raw_probs + (1 - alpha) * self.probs

    @property
    def label(self) -> str:
        return CLASS_NAMES[int(np.argmax(self.probs))]

    @property
    def confidence(self) -> float:
        return float(np.max(self.probs))


class SimpleTracker:
    """IoU-based multi-face tracker."""
    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 10):
        self.tracks: List[FaceTrack] = []
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing

    def update(self, faces: List[dict], probs_list: List[np.ndarray]) -> List[FaceTrack]:
        # Mark all tracks as missing initially
        for t in self.tracks:
            t.frames_missing += 1

        # Match detections to existing tracks
        used_tracks = set()
        used_dets = set()

        for di, (face, probs) in enumerate(zip(faces, probs_list)):
            best_iou = 0.0
            best_ti = -1
            for ti, track in enumerate(self.tracks):
                if ti in used_tracks:
                    continue
                iou = _bbox_iou(face["bbox"], track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_ti = ti
            if best_iou >= self.iou_threshold and best_ti >= 0:
                self.tracks[best_ti].update(face["bbox"], probs)
                used_tracks.add(best_ti)
                used_dets.add(di)

        # Create new tracks for unmatched detections
        for di, (face, probs) in enumerate(zip(faces, probs_list)):
            if di not in used_dets:
                track = FaceTrack(self.next_id, face["bbox"])
                track.update(face["bbox"], probs)
                self.tracks.append(track)
                self.next_id += 1

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.frames_missing <= self.max_missing]

        # Return active tracks (currently detected)
        return [t for t in self.tracks if t.frames_missing == 0]


# ═══════════════════════════════════════════════════════════════════
# DRAWING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def draw_face_box(frame: np.ndarray, track: FaceTrack, bbox_scale: float = 1.0):
    """Draw a professional thin bounding box + compact translucent label."""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = track.bbox
    x1 = max(0, int(x1 * bbox_scale))
    y1 = max(0, int(y1 * bbox_scale))
    x2 = min(fw, int(x2 * bbox_scale))
    y2 = min(fh, int(y2 * bbox_scale))
    bw, bh = x2 - x1, y2 - y1
    if bw < 5 or bh < 5:
        return

    label = track.label
    conf = track.confidence
    color = EMOTION_COLORS_BGR.get(label, (200, 200, 200))

    # ── Thin border (1px) ──────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # ── Compact label text ─────────────────────────────
    text = f"{label} {conf:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.30, min(0.50, bw / 250.0))
    text_thick = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thick)
    pad = 2

    # Label position: above box if space, else inside top
    lbl_x = x1
    if y1 - th - 2 * pad > 0:
        lbl_y1 = y1 - th - 2 * pad
        lbl_y2 = y1
    else:
        lbl_y1 = y1
        lbl_y2 = y1 + th + 2 * pad

    lbl_x2 = min(fw, lbl_x + tw + 2 * pad)
    lbl_y2 = min(fh, lbl_y2)
    lbl_y1 = max(0, lbl_y1)

    # Semi-transparent dark background (alpha blend only the label region)
    roi = frame[lbl_y1:lbl_y2, lbl_x:lbl_x2]
    if roi.size > 0:
        overlay = roi.copy()
        overlay[:] = (0, 0, 0)
        cv2.addWeighted(overlay, 0.5, roi, 0.5, 0, roi)
        frame[lbl_y1:lbl_y2, lbl_x:lbl_x2] = roi

    # Colored text matching the box border
    text_y = lbl_y2 - pad - 1 if (y1 - th - 2 * pad > 0) else lbl_y1 + th + pad - 1
    cv2.putText(frame, text, (lbl_x + pad, text_y),
                font, font_scale, color, text_thick, cv2.LINE_AA)


def draw_dashboard(frame: np.ndarray, tracks: List[FaceTrack],
                   fps: float, frame_num: int, total_frames: int):
    """Draw a dynamically-scaled info panel on the right side of the frame."""
    h, w = frame.shape[:2]

    # ── Dynamic scaling based on frame height ─────────────
    # Reference: designed for 720p. Scale everything proportionally.
    scale = h / 720.0
    scale = max(0.6, min(scale, 4.0))  # clamp for sanity

    panel_w = int(260 * scale)
    pad = int(10 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Font sizes (scaled)
    fs_title  = 0.7 * scale
    fs_normal = 0.55 * scale
    fs_emo    = 0.50 * scale
    fs_small  = 0.42 * scale
    th_title  = max(1, int(2 * scale))
    th_normal = max(1, int(1 * scale))

    # Line spacings (scaled)
    ls_title  = int(35 * scale)
    ls_normal = int(25 * scale)
    ls_emo    = int(24 * scale)
    ls_ctrl   = int(20 * scale)
    ls_sep    = int(15 * scale)

    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    y = int(30 * scale)

    # ── Title ─────────────────────────────────────────────
    cv2.putText(panel, "EmoNeXt Analyzer", (pad, y), font,
                fs_title, (0, 220, 255), th_title, cv2.LINE_AA)
    y += ls_title

    # ── Stats ─────────────────────────────────────────────
    cv2.putText(panel, f"Faces: {len(tracks)}", (pad, y), font,
                fs_normal, (200, 200, 200), th_normal, cv2.LINE_AA)
    y += ls_normal
    cv2.putText(panel, f"FPS: {fps:.1f}", (pad, y), font,
                fs_normal, (200, 200, 200), th_normal, cv2.LINE_AA)
    y += ls_normal
    progress = frame_num / max(total_frames, 1)
    cv2.putText(panel, f"Progress: {progress:.0%}", (pad, y), font,
                fs_normal, (200, 200, 200), th_normal, cv2.LINE_AA)
    y += int(12 * scale)
    bar_h = max(4, int(8 * scale))
    cv2.rectangle(panel, (pad, y), (panel_w - pad, y + bar_h), (80, 80, 80), -1)
    cv2.rectangle(panel, (pad, y),
                  (pad + int((panel_w - 2 * pad) * progress), y + bar_h),
                  (0, 200, 255), -1)
    y += bar_h + int(20 * scale)

    # ── Separator ─────────────────────────────────────────
    cv2.line(panel, (pad, y), (panel_w - pad, y), (80, 80, 80), max(1, int(scale)))
    y += int(20 * scale)

    # ── Emotion Distribution ──────────────────────────────
    cv2.putText(panel, "Emotion Distribution", (pad, y), font,
                fs_normal, (0, 200, 255), th_normal, cv2.LINE_AA)
    y += ls_normal

    if tracks:
        emotion_counts = defaultdict(int)
        for t in tracks:
            emotion_counts[t.label] += 1

        label_x = pad
        bar_x = int(115 * scale)
        bar_max = panel_w - bar_x - pad

        for cls in CLASS_NAMES:
            count = emotion_counts.get(cls, 0)
            color = EMOTION_COLORS_BGR.get(cls, (200, 200, 200))
            cv2.putText(panel, cls, (label_x, y), font,
                        fs_emo, color, th_normal, cv2.LINE_AA)
            # Bar
            bar_len = int(bar_max * count / max(len(tracks), 1))
            bar_top = y - int(11 * scale)
            cv2.rectangle(panel, (bar_x, bar_top), (bar_x + bar_len, y),
                          color, -1)
            if bar_len > 0:
                cv2.putText(panel, str(count), (bar_x + bar_len + int(5 * scale), y),
                            font, fs_small, (200, 200, 200), th_normal, cv2.LINE_AA)
            y += ls_emo
    else:
        cv2.putText(panel, "No faces detected", (pad, y), font,
                    fs_emo, (120, 120, 120), th_normal, cv2.LINE_AA)
        y += ls_emo

    y += ls_sep
    cv2.line(panel, (pad, y), (panel_w - pad, y), (80, 80, 80), max(1, int(scale)))
    y += int(20 * scale)

    # ── Controls help ─────────────────────────────────────
    cv2.putText(panel, "Controls", (pad, y), font,
                fs_normal, (0, 200, 255), th_normal, cv2.LINE_AA)
    y += ls_normal
    controls = [
        ("SPACE", "Pause / Resume"),
        ("q/ESC", "Quit"),
        ("s", "Screenshot"),
        ("r", "Record"),
        ("d", "Toggle panel"),
        ("</>", "Seek +/-5s"),
    ]
    for key, desc in controls:
        cv2.putText(panel, f"  {key:6s} {desc}", (pad, y), font,
                    fs_small, (160, 160, 160), th_normal, cv2.LINE_AA)
        y += ls_ctrl

    return np.hstack([frame, panel])


# ═══════════════════════════════════════════════════════════════════
# MODE SELECTION GUI
# ═══════════════════════════════════════════════════════════════════
def select_mode() -> Optional[str]:
    """Show a GUI to let user choose: Webcam or Video File.
    Returns 'webcam', 'video', or None (cancelled)."""
    import tkinter as tk

    choice = [None]

    root = tk.Tk()
    root.title("EmoNeXt — Select Mode")
    root.configure(bg="#1e1e2e")
    root.resizable(False, False)

    try:
        root.attributes("-topmost", True)
        root.after(300, lambda: root.attributes("-topmost", False))
    except Exception:
        pass

    BG = "#1e1e2e"
    HEADER = "#11111b"
    ACCENT = "#f5c211"
    FG = "#cdd6f4"

    # Header
    hdr = tk.Frame(root, bg=HEADER, height=55)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    tk.Label(hdr, text="🎬  EmoNeXt Expression Analyzer", bg=HEADER,
             fg=ACCENT, font=("Segoe UI", 16, "bold")).pack(pady=12)

    # Subtitle
    tk.Label(root, text="Choose an input source:", bg=BG, fg="#a6adc8",
             font=("Segoe UI", 11)).pack(pady=(20, 10))

    # Button frame
    btn_frame = tk.Frame(root, bg=BG)
    btn_frame.pack(pady=10)

    def pick_webcam():
        choice[0] = "webcam"
        root.destroy()

    def pick_video():
        choice[0] = "video"
        root.destroy()

    def pick_cancel():
        choice[0] = None
        root.destroy()

    # Webcam button
    wcam_btn = tk.Button(
        btn_frame, text="📹  Webcam Live", bg="#27ae60", fg="white",
        font=("Segoe UI", 13, "bold"), relief="raised", bd=2,
        activebackground="#2ecc71", activeforeground="white",
        highlightthickness=0, cursor="hand2",
        command=pick_webcam, width=20, height=2,
    )
    wcam_btn.grid(row=0, column=0, padx=15, pady=8)

    # Video button
    vid_btn = tk.Button(
        btn_frame, text="🎬  Select Video File", bg="#2980b9", fg="white",
        font=("Segoe UI", 13, "bold"), relief="raised", bd=2,
        activebackground="#3498db", activeforeground="white",
        highlightthickness=0, cursor="hand2",
        command=pick_video, width=20, height=2,
    )
    vid_btn.grid(row=0, column=1, padx=15, pady=8)

    # Cancel
    tk.Button(
        root, text="Cancel", bg="#45475a", fg=FG,
        font=("Segoe UI", 10), relief="raised", bd=2,
        activebackground="#585b70", activeforeground=FG,
        highlightthickness=0, cursor="hand2",
        command=pick_cancel, width=12,
    ).pack(pady=(5, 20))

    # Center on screen
    root.update_idletasks()
    w = root.winfo_reqwidth()
    h = root.winfo_reqheight()
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"+{x}+{y}")

    root.mainloop()
    return choice[0]


# ═══════════════════════════════════════════════════════════════════
# VIDEO FILE BROWSER
# ═══════════════════════════════════════════════════════════════════
def select_video() -> Optional[str]:
    """Open a custom-built video browser GUI to select a video file."""
    import tkinter as tk
    from tkinter import ttk
    from datetime import datetime

    selected_path = [None]

    # ── Build the custom GUI ──────────────────────────────
    root = tk.Tk()
    root.title("🎬 EmoNeXt — Select Video")
    root.geometry("900x620")
    root.configure(bg="#1e1e2e")
    root.resizable(True, True)

    # Try to bring to front
    try:
        root.attributes("-topmost", True)
        root.after(200, lambda: root.attributes("-topmost", False))
    except Exception:
        pass

    # ── Styles ────────────────────────────────────────────
    BG       = "#1e1e2e"
    BG2      = "#282840"
    FG       = "#cdd6f4"
    ACCENT   = "#f5c211"
    BTN_BG   = "#45475a"
    BTN_HOVER= "#585b70"
    SEL_BG   = "#3a3a5c"
    HEADER   = "#11111b"

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Dark.TFrame", background=BG)
    style.configure("Dark.TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
    style.configure("Header.TLabel", background=HEADER, foreground=ACCENT,
                    font=("Segoe UI", 14, "bold"))
    style.configure("Path.TLabel", background=BG2, foreground=FG, font=("Consolas", 10))
    style.configure("Dark.TButton", background=BTN_BG, foreground=FG,
                    font=("Segoe UI", 10, "bold"), padding=(12, 6))
    style.map("Dark.TButton",
              background=[("active", BTN_HOVER), ("pressed", ACCENT)])
    style.configure("Select.TButton", background="#27ae60", foreground="white",
                    font=("Segoe UI", 11, "bold"), padding=(20, 8))
    style.map("Select.TButton",
              background=[("active", "#2ecc71"), ("pressed", "#1e8449")])

    # ── Header ────────────────────────────────────────────
    header_frame = tk.Frame(root, bg=HEADER, height=50)
    header_frame.pack(fill="x")
    header_frame.pack_propagate(False)
    tk.Label(header_frame, text="🎬  EmoNeXt Video Browser", bg=HEADER,
             fg=ACCENT, font=("Segoe UI", 15, "bold")).pack(side="left", padx=15, pady=10)
    tk.Label(header_frame, text="Select a video for expression analysis", bg=HEADER,
             fg="#a6adc8", font=("Segoe UI", 10)).pack(side="left", padx=10, pady=10)

    # ── Navigation bar ────────────────────────────────────
    nav_frame = tk.Frame(root, bg=BG2, height=40)
    nav_frame.pack(fill="x", padx=0, pady=(0, 0))
    nav_frame.pack_propagate(False)

    current_dir = [str(Path.home())]
    path_var = tk.StringVar(value=current_dir[0])

    def go_up():
        parent = str(Path(current_dir[0]).parent)
        navigate_to(parent)

    def go_home():
        navigate_to(str(Path.home()))

    btn_up = tk.Button(nav_frame, text="⬆ Up", bg=BTN_BG, fg=FG,
                       font=("Segoe UI", 9, "bold"), relief="raised", bd=2,
                       activebackground=BTN_HOVER, activeforeground=FG,
                       highlightthickness=0, command=go_up, cursor="hand2")
    btn_up.pack(side="left", padx=(10, 4), pady=6)

    btn_home = tk.Button(nav_frame, text="🏠 Home", bg=BTN_BG, fg=FG,
                         font=("Segoe UI", 9, "bold"), relief="raised", bd=2,
                         activebackground=BTN_HOVER, activeforeground=FG,
                         highlightthickness=0, command=go_home, cursor="hand2")
    btn_home.pack(side="left", padx=4, pady=6)

    path_entry = tk.Entry(nav_frame, textvariable=path_var, bg="#313244", fg=FG,
                          insertbackground=FG, font=("Consolas", 10),
                          relief="flat", bd=0)
    path_entry.pack(side="left", fill="x", expand=True, padx=10, pady=8, ipady=2)

    def on_path_enter(event=None):
        p = path_var.get().strip()
        if Path(p).is_dir():
            navigate_to(p)
        elif Path(p).is_file():
            selected_path[0] = p
            root.destroy()

    path_entry.bind("<Return>", on_path_enter)

    btn_go = tk.Button(nav_frame, text="Go ➜", bg=BTN_BG, fg=FG,
                       font=("Segoe UI", 9, "bold"), relief="raised", bd=2,
                       activebackground=BTN_HOVER, activeforeground=FG,
                       highlightthickness=0, command=on_path_enter, cursor="hand2")
    btn_go.pack(side="right", padx=10, pady=6)

    # ── File list area ────────────────────────────────────
    list_frame = tk.Frame(root, bg=BG)
    list_frame.pack(fill="both", expand=True, padx=10, pady=5)

    # Column headers
    col_header = tk.Frame(list_frame, bg="#313244", height=28)
    col_header.pack(fill="x")
    col_header.pack_propagate(False)
    tk.Label(col_header, text="  Name", bg="#313244", fg="#a6adc8",
             font=("Segoe UI", 9, "bold"), anchor="w").pack(side="left", padx=(10, 0))
    tk.Label(col_header, text="Size", bg="#313244", fg="#a6adc8",
             font=("Segoe UI", 9, "bold"), width=10).pack(side="right", padx=10)
    tk.Label(col_header, text="Type", bg="#313244", fg="#a6adc8",
             font=("Segoe UI", 9, "bold"), width=10).pack(side="right")

    # Scrollable canvas for file list
    canvas = tk.Canvas(list_frame, bg=BG, highlightthickness=0, bd=0)
    scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg=BG)

    scroll_frame.bind("<Configure>",
                      lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Mouse wheel scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120 or -event.num + 4)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-3, "units"))
    canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(3, "units"))

    # ── Info & buttons at bottom ──────────────────────────
    bottom_frame = tk.Frame(root, bg=BG2, height=60)
    bottom_frame.pack(fill="x", side="bottom")
    bottom_frame.pack_propagate(False)

    info_var = tk.StringVar(value="Navigate to a video file and click it to select")
    info_label = tk.Label(bottom_frame, textvariable=info_var, bg=BG2, fg="#a6adc8",
                          font=("Segoe UI", 9), anchor="w")
    info_label.pack(side="left", padx=15, pady=5)

    def do_cancel():
        selected_path[0] = None
        root.destroy()

    def do_select():
        root.destroy()

    btn_cancel = tk.Button(bottom_frame, text="Cancel", bg="#e74c3c", fg="white",
                           font=("Segoe UI", 10, "bold"), relief="raised", bd=2,
                           activebackground="#c0392b", activeforeground="white",
                           highlightthickness=0, command=do_cancel,
                           cursor="hand2", padx=15, pady=4)
    btn_cancel.pack(side="right", padx=10, pady=12)

    btn_select = tk.Button(bottom_frame, text="▶  Open Selected", bg="#27ae60",
                           fg="white", font=("Segoe UI", 10, "bold"), relief="raised", bd=2,
                           activebackground="#2ecc71", activeforeground="white",
                           highlightthickness=0, command=do_select,
                           cursor="hand2", padx=15, pady=4, state="disabled")
    btn_select.pack(side="right", padx=5, pady=12)

    # ── Row rendering ─────────────────────────────────────
    row_widgets = []
    selected_row = [None]

    def format_size(size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 ** 2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 ** 3:
            return f"{size_bytes / 1024**2:.1f} MB"
        else:
            return f"{size_bytes / 1024**3:.2f} GB"

    def on_row_click(path_str, is_dir, row_frame):
        if is_dir:
            navigate_to(path_str)
        else:
            # Deselect previous
            for w in row_widgets:
                w.configure(bg=BG)
                for child in w.winfo_children():
                    child.configure(bg=BG)
            # Select this
            row_frame.configure(bg=SEL_BG)
            for child in row_frame.winfo_children():
                child.configure(bg=SEL_BG)
            selected_path[0] = path_str
            selected_row[0] = row_frame
            btn_select.configure(state="normal")
            name = Path(path_str).name
            sz = format_size(Path(path_str).stat().st_size)
            info_var.set(f"✅ Selected: {name}  ({sz})")

    def on_row_double(path_str, is_dir):
        if is_dir:
            navigate_to(path_str)
        else:
            selected_path[0] = path_str
            root.destroy()

    def navigate_to(directory: str):
        d = Path(directory)
        if not d.is_dir():
            return
        current_dir[0] = str(d)
        path_var.set(str(d))
        selected_path[0] = None
        btn_select.configure(state="disabled")
        info_var.set(f"📂  {d}")

        # Clear old rows
        for w in scroll_frame.winfo_children():
            w.destroy()
        row_widgets.clear()

        try:
            entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            info_var.set("⚠ Permission denied")
            return

        # Separate dirs and video files
        dirs = [e for e in entries if e.is_dir() and not e.name.startswith('.')]
        vids = [e for e in entries
                if e.is_file() and e.suffix.lower() in VIDEO_EXTS]

        vid_count = len(vids)
        count_text = f"{vid_count} video{'s' if vid_count != 1 else ''}" if vid_count else "No videos"
        info_var.set(f"📂  {d.name}/  —  {len(dirs)} folders, {count_text}")

        row_idx = 0
        for entry in dirs + vids:
            is_dir = entry.is_dir()
            bg_color = BG

            row = tk.Frame(scroll_frame, bg=bg_color, cursor="hand2")
            row.pack(fill="x", padx=2, pady=1)
            row_widgets.append(row)

            # Icon
            icon = "📁" if is_dir else "🎬"
            fg_name = "#89b4fa" if is_dir else FG

            tk.Label(row, text=icon, bg=bg_color, font=("Segoe UI", 11),
                     width=3).pack(side="left", padx=(8, 0))
            tk.Label(row, text=entry.name, bg=bg_color, fg=fg_name,
                     font=("Segoe UI", 10), anchor="w").pack(side="left", padx=5, fill="x", expand=True)

            if not is_dir:
                try:
                    sz = format_size(entry.stat().st_size)
                except OSError:
                    sz = "?"
                ext = entry.suffix.upper().lstrip(".")
                tk.Label(row, text=ext, bg=bg_color, fg="#a6adc8",
                         font=("Segoe UI", 9), width=8).pack(side="right", padx=5)
                tk.Label(row, text=sz, bg=bg_color, fg="#a6adc8",
                         font=("Segoe UI", 9), width=10).pack(side="right")
            else:
                tk.Label(row, text="Folder", bg=bg_color, fg="#585b70",
                         font=("Segoe UI", 9), width=8).pack(side="right", padx=5)
                tk.Label(row, text="", bg=bg_color, width=10).pack(side="right")

            # Hover effect
            def _enter(e, r=row, d=is_dir):
                if r != selected_row[0]:
                    c = "#313244"
                    r.configure(bg=c)
                    for ch in r.winfo_children():
                        ch.configure(bg=c)

            def _leave(e, r=row, d=is_dir):
                if r != selected_row[0]:
                    r.configure(bg=BG)
                    for ch in r.winfo_children():
                        ch.configure(bg=BG)

            path_str = str(entry)
            row.bind("<Enter>", _enter)
            row.bind("<Leave>", _leave)
            row.bind("<Button-1>", lambda e, p=path_str, d=is_dir, r=row: on_row_click(p, d, r))
            row.bind("<Double-Button-1>", lambda e, p=path_str, d=is_dir: on_row_double(p, d))
            for child in row.winfo_children():
                child.bind("<Button-1>", lambda e, p=path_str, d=is_dir, r=row: on_row_click(p, d, r))
                child.bind("<Double-Button-1>", lambda e, p=path_str, d=is_dir: on_row_double(p, d))

            row_idx += 1

        # Scroll to top
        canvas.yview_moveto(0)

    # ── Initial navigation ────────────────────────────────
    navigate_to(current_dir[0])

    # Center window on screen
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - 900) // 2
    y = (sh - 620) // 2
    root.geometry(f"900x620+{x}+{y}")

    root.mainloop()
    return selected_path[0]


def draw_progress_bar(frame, frame_num: int, total_frames: int,
                      vid_fps: float, paused: bool = False,
                      speed: float = 1.0):
    """Draw a sleek progress bar + time overlay at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 40
    bar_y = h - bar_h

    # Semi-transparent dark background (only blend the bar region)
    roi_y = max(0, bar_y - 4)
    roi = frame[roi_y:h, 0:w]
    bar_overlay = roi.copy()
    cv2.rectangle(bar_overlay, (0, 0), (w, h - roi_y), (15, 15, 25), -1)
    cv2.addWeighted(bar_overlay, 0.82, roi, 0.18, 0, frame[roi_y:h, 0:w])

    # Thin separator line
    cv2.line(frame, (0, bar_y - 4), (w, bar_y - 4), (60, 60, 90), 1)

    # Progress track (background)
    margin_l = 90
    margin_r = 170
    track_y = bar_y + bar_h // 2
    track_h = 6
    cv2.rectangle(frame, (margin_l, track_y - track_h // 2),
                  (w - margin_r, track_y + track_h // 2),
                  (50, 50, 70), -1, cv2.LINE_AA)

    # Progress fill
    progress = frame_num / max(1, total_frames) if total_frames > 0 else 0
    bar_width = w - margin_l - margin_r
    fill_w = int(bar_width * min(1.0, progress))
    if fill_w > 0:
        cv2.rectangle(frame, (margin_l, track_y - track_h // 2),
                      (margin_l + fill_w, track_y + track_h // 2),
                      (0, 200, 255), -1, cv2.LINE_AA)

    # Seek dot
    dot_x = margin_l + fill_w
    cv2.circle(frame, (dot_x, track_y), 7, (0, 230, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (dot_x, track_y), 7, (255, 255, 255), 1, cv2.LINE_AA)

    # Time text: current / total
    current_s = frame_num / max(1.0, vid_fps)
    total_s = total_frames / max(1.0, vid_fps)
    time_cur = f"{int(current_s // 60):02d}:{int(current_s % 60):02d}"
    time_tot = f"{int(total_s // 60):02d}:{int(total_s % 60):02d}"
    time_str = f"{time_cur} / {time_tot}"
    cv2.putText(frame, time_str, (w - margin_r + 10, track_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 220), 1, cv2.LINE_AA)

    # Play / Pause indicator
    if paused:
        # Draw pause icon (two vertical bars)
        bx, by = 20, track_y - 8
        cv2.rectangle(frame, (bx, by), (bx + 5, by + 16), (255, 200, 0), -1)
        cv2.rectangle(frame, (bx + 10, by), (bx + 15, by + 16), (255, 200, 0), -1)
    else:
        # Draw play triangle
        pts = np.array([[20, track_y - 8], [20, track_y + 8], [35, track_y]], np.int32)
        cv2.fillPoly(frame, [pts], (0, 230, 118))

    # Speed indicator (if not 1x)
    if abs(speed - 1.0) > 0.01:
        spd_str = f"{speed:.1f}x"
        cv2.putText(frame, spd_str, (45, track_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 200), 1, cv2.LINE_AA)

    return frame


def run_analyzer(source, device: torch.device, is_webcam: bool = False) -> Optional[str]:
    """Main analysis loop — threaded detection for smooth playback.
    Args:
        source: video file path (str) or camera index (int, usually 0)
        device: torch device
        is_webcam: True for live webcam, False for video file
    Returns:
        'switch_webcam' / 'switch_video' to change mode, or None to exit.
    """
    switch_mode = None  # will be set if user presses 'b'
    mode_label = "📹 Webcam Live" if is_webcam else "🎬 Video File"
    print("\n" + "=" * 65)
    print(f"  {mode_label} — EmoNeXt Expression Analyzer")
    print("=" * 65)

    # ── Load model ────────────────────────────────────────
    print("\n📦 Loading model...")
    model = load_model(device)

    # ── Setup YuNet face detector ─────────────────────────
    print("\n👤 Setting up face detection...")
    yunet_path = resolve_yunet()
    print(f"  ✅ YuNet: {yunet_path.name}")

    # ── Open source ───────────────────────────────────────
    if is_webcam:
        print("\n📹 Opening webcam...")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("  ❌ Failed to open webcam. Check permissions / camera.")
            return
        # Set webcam resolution hints
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        print(f"\n🎬 Opening: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"  ❌ Failed to open video: {source}")
            return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
    duration = total_frames / vid_fps if vid_fps > 0 and not is_webcam else 0

    print(f"  Resolution : {frame_w}×{frame_h}")
    print(f"  FPS        : {vid_fps:.1f}")
    if not is_webcam:
        print(f"  Duration   : {duration:.1f}s ({total_frames} frames)")

    # ── Detection resolution (higher for small-face coverage) ──
    MAX_DET_W = 1920
    if frame_w > MAX_DET_W:
        det_scale = MAX_DET_W / frame_w
        det_w = MAX_DET_W
        det_h = int(frame_h * det_scale)
    else:
        det_scale = 1.0
        det_w = frame_w
        det_h = frame_h

    # ── Display resolution (for smooth playback) ──
    MAX_DISPLAY_W = 1280
    if frame_w > MAX_DISPLAY_W:
        disp_scale = MAX_DISPLAY_W / frame_w
        disp_w = MAX_DISPLAY_W
        disp_h = int(frame_h * disp_scale)
        print(f"  Display    : {disp_w}×{disp_h} (downscaled for playback)")
    else:
        disp_scale = 1.0
        disp_w = frame_w
        disp_h = frame_h

    # Ratio: detection coords → display coords
    det_to_disp = disp_w / det_w
    print(f"  Detection  : {det_w}×{det_h} (face detection resolution)")

    # Create detectors at DETECTION resolution (not display)
    detectors = create_yunet_detectors(yunet_path, det_w, det_h)

    # State
    show_dashboard = True
    paused = False
    recording = False
    writer = None
    speed_mult = 1.0
    fps_deque = deque(maxlen=60)

    window_name = "EmoNeXt — Webcam Live" if is_webcam else "EmoNeXt — Video Analyzer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    dash_panel_w = int(260 * max(0.6, min(disp_h / 720.0, 4.0)))
    cv2.resizeWindow(window_name, min(1920, disp_w + dash_panel_w), min(1080, disp_h))

    # ── Mouse-click seeking (video only) ──────────────────
    PBAR_H = 40
    PBAR_MARGIN_L = 90
    PBAR_MARGIN_R = 170
    seek_target = [None]
    mouse_param = {"display_shape": (disp_h, disp_w)}
    mouse_cb_set = False

    def _mouse_cb(event, x, y, flags, param):
        if is_webcam:
            return  # no seeking in webcam mode
        dh, dw = param.get("display_shape", (disp_h, disp_w))
        bar_y_start = dh - PBAR_H - 4
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE) and (flags & cv2.EVENT_FLAG_LBUTTON):
            if y >= bar_y_start and PBAR_MARGIN_L <= x <= dw - PBAR_MARGIN_R:
                bar_width = dw - PBAR_MARGIN_L - PBAR_MARGIN_R
                if bar_width > 0:
                    pct = max(0.0, min(1.0, (x - PBAR_MARGIN_L) / bar_width))
                    seek_target[0] = int(pct * total_frames)

    # ══════════════════════════════════════════════════════
    # BACKGROUND DETECTION THREAD
    # ══════════════════════════════════════════════════════
    latest_tracks: List[FaceTrack] = []
    tracks_lock = threading.Lock()
    det_frame_slot = [None]
    det_frame_lock = threading.Lock()
    det_stop = threading.Event()
    tracker = SimpleTracker(iou_threshold=0.3, max_missing=8)

    def _detection_worker():
        nonlocal latest_tracks
        while not det_stop.is_set():
            with det_frame_lock:
                snap = det_frame_slot[0]
                det_frame_slot[0] = None
            if snap is None:
                time.sleep(0.005)
                continue
            try:
                faces = detect_faces(detectors, snap)
                frame_rgb = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
                probs_list = []
                if faces:
                    batch_tensors = []
                    valid_indices = []
                    for i, face in enumerate(faces):
                        face_img = crop_face(frame_rgb, face, margin=0.25)
                        if face_img is not None:
                            tensor = preprocess_face(face_img)
                            batch_tensors.append(tensor)
                            valid_indices.append(i)
                    if batch_tensors:
                        batch = torch.stack(batch_tensors).to(device)
                        with torch.no_grad():
                            out = model(batch)
                            logits = out['logits'].float()
                            batch_probs = F.softmax(logits, dim=1).cpu().numpy()
                        prob_map = {}
                        for bi, fi in enumerate(valid_indices):
                            prob_map[fi] = batch_probs[bi]
                        for i in range(len(faces)):
                            probs_list.append(
                                prob_map.get(i, np.ones(NUM_CLASSES) / NUM_CLASSES))
                    else:
                        probs_list = [np.ones(NUM_CLASSES) / NUM_CLASSES] * len(faces)
                with tracks_lock:
                    latest_tracks = tracker.update(faces, probs_list)
            except Exception as e:
                import traceback
                print(f"  ⚠ Detection thread error: {e}")
                traceback.print_exc()

    det_thread = threading.Thread(target=_detection_worker, daemon=True)
    det_thread.start()

    if is_webcam:
        print("\n📹 Live — press 'q' or ESC to quit\n")
    else:
        print("\n▶ Playing... (press 'q' or ESC to quit)\n")

    frame_num = 0
    frame = None
    frame_interval = 1.0 / vid_fps
    start_time = time.time()

    while True:
        t0 = time.time()

        # ── Seek (video only) ─────────────────────────────
        if not is_webcam and seek_target[0] is not None:
            target = max(0, min(total_frames - 1, seek_target[0]))
            seek_target[0] = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target - 1))
                ret, frame = cap.read()
                if not ret:
                    break
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        elif not paused:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    time.sleep(0.01)
                    continue
                print("\n  ✅ Video ended.")
                break
            if not is_webcam:
                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                frame_num += 1

        if frame is None:
            continue

        # ── Webcam: flip for mirror view ──────────────────
        if is_webcam:
            frame = cv2.flip(frame, 1)

        # ── Prepare detection frame (higher res for small faces) ──
        raw_frame = frame  # reference to original before display resize
        with det_frame_lock:
            if det_scale < 1.0:
                det_frame_slot[0] = cv2.resize(raw_frame, (det_w, det_h),
                                               interpolation=cv2.INTER_LINEAR)
            else:
                det_frame_slot[0] = raw_frame.copy()

        # ── Downscale to display resolution ───────────────
        if disp_scale < 1.0:
            frame = cv2.resize(frame, (disp_w, disp_h),
                               interpolation=cv2.INTER_LINEAR)

        # ── Get latest results ────────────────────────────
        with tracks_lock:
            active_tracks = list(latest_tracks)

        # ── Draw results (scale bboxes from detection→display) ──
        display = frame
        for track in active_tracks:
            draw_face_box(display, track, bbox_scale=det_to_disp)

        # ── Subtle HUD overlay (top-left) ─────────────────
        hud_text = f"Faces: {len(active_tracks)}"
        cv2.putText(display, hud_text, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, hud_text, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        # ── Mode-switch hint ──────────────────────────────
        switch_hint = "[B] Switch to Video" if is_webcam else "[B] Switch to Webcam"
        cv2.putText(display, switch_hint, (12, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

        # ── Webcam: show elapsed time ─────────────────────
        if is_webcam:
            elapsed_total = time.time() - start_time
            mins, secs = divmod(int(elapsed_total), 60)
            cv2.putText(display, f"LIVE  {mins:02d}:{secs:02d}",
                        (disp_w - 160, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 1, cv2.LINE_AA)

        # ── Dashboard ─────────────────────────────────────
        elapsed = time.time() - t0
        fps_deque.append(1.0 / max(elapsed, 1e-6))
        avg_fps = np.mean(fps_deque)

        if show_dashboard:
            display = draw_dashboard(display, active_tracks, avg_fps,
                                     frame_num, total_frames)

        # ── Progress bar (video only) ─────────────────────
        if not is_webcam:
            display = draw_progress_bar(display, frame_num, total_frames,
                                        vid_fps, paused, speed_mult)

        mouse_param["display_shape"] = display.shape[:2]

        # ── Recording ─────────────────────────────────────
        if recording and writer is not None:
            writer.write(display)

        # ── Show ──────────────────────────────────────────
        cv2.imshow(window_name, display)

        if not mouse_cb_set:
            try:
                cv2.setMouseCallback(window_name, _mouse_cb, mouse_param)
                mouse_cb_set = True
            except cv2.error:
                pass

        # ── Timing ────────────────────────────────────────
        if is_webcam:
            key = cv2.waitKey(1) & 0xFF
        else:
            elapsed_ms = (time.time() - t0) * 1000.0
            target_ms = (frame_interval * 1000.0) / speed_mult if not paused else 30.0
            wait_ms = max(1, int(target_ms - elapsed_ms))
            key = cv2.waitKey(wait_ms) & 0xFF

        # ── Key handling ──────────────────────────────────
        if key in (ord('q'), 27):
            break
        elif key == ord('b'):
            switch_mode = "switch_video" if is_webcam else "switch_webcam"
            print(f"  🔄 Switching to {'Video' if is_webcam else 'Webcam'}...")
            break
        elif key == ord('d'):
            show_dashboard = not show_dashboard
        elif key == ord('s'):
            ss_path = SCREENSHOT_DIR / f"emonext_ss_{int(time.time())}.png"
            cv2.imwrite(str(ss_path), display)
            print(f"  📸 Screenshot: {ss_path.name}")
        elif key == ord('r'):
            if not recording:
                prefix = "webcam" if is_webcam else "video"
                rec_path = ANNOTATED_DIR / f"emonext_{prefix}_{int(time.time())}.mp4"
                h_out, w_out = display.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                rec_fps = vid_fps if not is_webcam else 30.0
                writer = cv2.VideoWriter(str(rec_path), fourcc, rec_fps, (w_out, h_out))
                recording = True
                print(f"  🔴 Recording started: {rec_path.name}")
            else:
                recording = False
                if writer:
                    writer.release()
                    writer = None
                print("  ⏹ Recording stopped")

        # Video-only controls
        if not is_webcam:
            if key == ord(' '):
                paused = not paused
                print(f"  {'⏸ Paused' if paused else '▶ Resumed'}")
            elif key == 82 or key == ord('+'):
                speed_mult = min(4.0, speed_mult + 0.25)
                print(f"  ⏩ Speed: {speed_mult:.2f}x")
            elif key == 84 or key == ord('-'):
                speed_mult = max(0.25, speed_mult - 0.25)
                print(f"  ⏪ Speed: {speed_mult:.2f}x")
            elif key == 81:
                new_pos = max(0, frame_num - int(vid_fps * 5))
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            elif key == 83:
                new_pos = min(total_frames - 1, frame_num + int(vid_fps * 5))
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            elif ord('1') <= key <= ord('9'):
                pct = (key - ord('0')) / 10.0
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * pct))
                print(f"  ⏭ Jumped to {pct:.0%}")

    # ── Cleanup ───────────────────────────────────────────
    det_stop.set()
    det_thread.join(timeout=2)
    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    if switch_mode:
        print(f"  🔄 Mode switch requested.")
    else:
        print("\n✅ Analyzer closed.")
    return switch_mode


# ═══════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="EmoNeXt Expression Analyzer — Webcam or Video"
    )
    parser.add_argument("video", nargs="?", default=None,
                        help="Path to video file (opens mode selector if omitted)")
    parser.add_argument("--webcam", action="store_true",
                        help="Launch directly in webcam mode")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device (default: auto)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"  Device: {device}")

    # ── Determine initial mode ─────────────────────────────
    if args.webcam:
        next_action = "webcam"
    elif args.video:
        if not Path(args.video).is_file():
            print(f"  ❌ File not found: {args.video}")
            sys.exit(1)
        next_action = "video"
    else:
        next_action = "select"  # show mode selector GUI

    camera_idx = args.camera
    video_path = args.video

    # ── Main loop: allows switching between webcam ↔ video ──
    while next_action is not None:
        if next_action == "webcam":
            result = run_analyzer(camera_idx, device, is_webcam=True)
        elif next_action == "video":
            if not video_path or not Path(video_path).is_file():
                video_path = select_video()
            if not video_path or not Path(video_path).is_file():
                print("  ❌ No valid video selected. Exiting.")
                break
            result = run_analyzer(video_path, device, is_webcam=False)
        elif next_action == "select":
            mode = select_mode()
            if mode == "webcam":
                next_action = "webcam"
                continue
            elif mode == "video":
                next_action = "video"
                continue
            else:
                print("  ❌ Cancelled.")
                break
        else:
            break

        # Handle mode switch requests from run_analyzer
        if result == "switch_webcam":
            next_action = "webcam"
        elif result == "switch_video":
            video_path = None  # force file picker
            next_action = "video"
        else:
            next_action = None  # quit


if __name__ == "__main__":
    main()
