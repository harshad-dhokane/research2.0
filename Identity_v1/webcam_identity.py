"""
DPAIN Webcam Identity Recognition — Live Test
===============================================
Tests the trained Dual-Path Adaptive Identity Network (DPAIN) model
on live webcam feed. Detects faces using MediaPipe, aligns them, and
identifies the person in real-time. Supports multiple simultaneous faces.

Usage:
    python webcam_identity.py

Controls:
    q / ESC  — Quit
    s        — Save screenshot
    t        — Toggle TTA on/off
    f        — Toggle probability bars
    r        — Toggle unknown rejection
"""

import sys
import os
import json
import time
from pathlib import Path
from collections import deque

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import mediapipe as mp

# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
CHECKPOINT_PATH = BASE_DIR / 'identity_outputs' / 'checkpoints' / 'best_model.pth'
CLASS_MAPPING_PATH = BASE_DIR / 'identity_outputs' / 'configs' / 'class_mapping.json'
CONFIG_PATH = BASE_DIR / 'identity_outputs' / 'configs' / 'experiment_config.json'
CENTROIDS_PATH = BASE_DIR / 'identity_outputs' / 'embeddings' / 'identity_centroids.json'
FACE_MODEL_PATH = str(BASE_DIR / 'face_landmarker_v2_with_blendshapes.task')
SCREENSHOT_DIR = BASE_DIR / 'identity_outputs' / 'screenshots'
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load experiment config
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# Load class mapping
with open(CLASS_MAPPING_PATH, 'r') as f:
    mapping = json.load(f)
    IDX_TO_LABEL = mapping.get('idx_to_label', mapping.get('idx_to_class', {}))
    LABEL_TO_IDX = mapping.get('label_to_idx', mapping.get('class_to_idx', {}))
    CLASS_NAMES = [IDX_TO_LABEL[str(i)] for i in range(len(IDX_TO_LABEL))]

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = CONFIG.get('image_size', 224)
EMBEDDING_DIM = CONFIG.get('embedding_dim', 128)

# Normalization (computed on Identity training set)
NORM_MEAN = (0.6324, 0.5407, 0.4785)
NORM_STD  = (0.2853, 0.2862, 0.2807)

# Rejection threshold — show name only if confidence > 80%
REJECTION_THRESHOLD = 0.80

# Identity colors (BGR for OpenCV) — distinct per person
IDENTITY_COLORS = [
    (255, 100, 100),   # aamrapali   — light blue
    (255, 200, 50),    # asraar      — cyan
    (50, 200, 50),     # avanti      — green
    (0, 255, 128),     # bhavesh     — spring green
    (0, 200, 255),     # dnyaneshwari— gold
    (180, 105, 255),   # gayatri     — pink
    (128, 0, 255),     # kamlesh     — purple
    (100, 150, 200),   # naseeruddin — tan
    (0, 165, 255),     # nikhil      — orange
    (255, 0, 200),     # nikhil kere — magenta
    (80, 200, 200),    # rakesh      — olive
    (200, 100, 200),   # rutuja      — violet
    (0, 230, 230),     # sakshi      — yellow
    (255, 200, 200),   # satish      — light blue-white
    (100, 255, 200),   # zafar       — mint
]

# Unknown person color
UNKNOWN_COLOR = (0, 0, 200)  # dark red

# Eye landmark indices for alignment
LEFT_EYE_INDICES = [33, 133, 159, 145, 160, 144, 158, 153]
RIGHT_EYE_INDICES = [362, 263, 386, 374, 387, 373, 385, 380]


# ──────────────────────────────────────────────
#  MODEL ARCHITECTURE (must match training)
# ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Basic conv → batchnorm → activation block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AdaptiveChannelGate(nn.Module):
    """Channel-wise attention that amplifies identity-relevant features."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class StructuralBlock(nn.Module):
    """Structural path: standard conv with residual + channel gate."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.gate = AdaptiveChannelGate(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.gate(out)
        return self.act(out + residual)


class DetailBlock(nn.Module):
    """Detail path: dilated convolutions to preserve spatial resolution."""
    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.gate = AdaptiveChannelGate(out_ch)
        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.gate(out)
        return self.act(out + residual)


class AdaptiveFusionGate(nn.Module):
    """Learns to weight structural vs detail features per-image."""
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, struct_feat, detail_feat):
        if struct_feat.shape[2:] != detail_feat.shape[2:]:
            detail_feat = F.adaptive_avg_pool2d(detail_feat, struct_feat.shape[2:])
        combined = torch.cat([
            F.adaptive_avg_pool2d(struct_feat, 1).flatten(1),
            F.adaptive_avg_pool2d(detail_feat, 1).flatten(1)
        ], dim=1)
        weights = self.fc(combined)
        w_s = weights[:, 0:1].unsqueeze(-1).unsqueeze(-1)
        w_d = weights[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        return w_s * struct_feat + w_d * detail_feat


class MultiScalePool(nn.Module):
    """Multi-scale pooling for pose-invariant aggregation."""
    def __init__(self, channels):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        att_weights = self.spatial_att(x)
        weighted = (x * att_weights).sum(dim=[2, 3]) / (att_weights.sum(dim=[2, 3]) + 1e-8)
        return avg_pool + max_pool + weighted


class DPAINModel(nn.Module):
    """
    Dual-Path Adaptive Identity Network (DPAIN)
    Input:  (B, 3, 224, 224) aligned face images
    Output: (B, embedding_dim) L2-normalized identity embeddings
    """
    def __init__(self, embedding_dim=128, dropout=0.3):
        super().__init__()

        # Shared stem
        self.stem = nn.Sequential(
            ConvBlock(3, 32, kernel_size=5, stride=2, padding=2),   # 224→112
            ConvBlock(32, 48, kernel_size=3, stride=1, padding=1),  # 112×112
        )

        # Structural path
        self.struct_s1 = StructuralBlock(48, 64, stride=2)    # 112→56
        self.struct_s2 = StructuralBlock(64, 96, stride=2)    # 56→28
        self.struct_s3 = StructuralBlock(96, 128, stride=2)   # 28→14
        self.struct_s4 = StructuralBlock(128, 160, stride=2)  # 14→7

        # Detail path
        self.detail_s1 = DetailBlock(48, 64, dilation=2)
        self.detail_down1 = nn.MaxPool2d(2)
        self.detail_s2 = DetailBlock(64, 96, dilation=2)
        self.detail_down2 = nn.MaxPool2d(2)
        self.detail_s3 = DetailBlock(96, 128, dilation=2)
        self.detail_down3 = nn.MaxPool2d(2)
        self.detail_s4 = DetailBlock(128, 160, dilation=1)
        self.detail_down4 = nn.MaxPool2d(2)

        # Adaptive fusion
        self.fusion = AdaptiveFusionGate(160)

        # Multi-scale pooling
        self.pool = MultiScalePool(160)

        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(160, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x):
        shared = self.stem(x)

        s = self.struct_s1(shared)
        s = self.struct_s2(s)
        s = self.struct_s3(s)
        s = self.struct_s4(s)

        d = self.detail_s1(shared)
        d = self.detail_down1(d)
        d = self.detail_s2(d)
        d = self.detail_down2(d)
        d = self.detail_s3(d)
        d = self.detail_down3(d)
        d = self.detail_s4(d)
        d = self.detail_down4(d)

        fused = self.fusion(s, d)
        pooled = self.pool(fused)
        emb = self.embedding_head(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


# ──────────────────────────────────────────────
#  FACE ALIGNMENT (same as training pipeline)
# ──────────────────────────────────────────────

def create_landmarker():
    """Create MediaPipe FaceLandmarker for multi-face detection."""
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        num_faces=10,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    return FaceLandmarker.create_from_options(options)


def get_eye_centers(landmarks, w, h):
    """Get eye center coordinates from landmarks."""
    left_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE_INDICES])
    right_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE_INDICES])
    return left_eye.mean(axis=0), right_eye.mean(axis=0)


def align_face(img_np, landmarks, w, h, margin=0.3, target_size=224):
    """Align face using eye landmarks: rotate → crop → resize → CLAHE."""
    left_eye, right_eye = get_eye_centers(landmarks, w, h)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    rotated = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # Compute rotated landmark positions
    all_pts = np.array([(landmarks[i].x * w, landmarks[i].y * h)
                        for i in range(len(landmarks))])
    ones = np.ones((all_pts.shape[0], 1))
    pts_h = np.hstack([all_pts, ones])
    rotated_pts = (M @ pts_h.T).T

    x_min, y_min = rotated_pts.min(axis=0)
    x_max, y_max = rotated_pts.max(axis=0)
    bw = x_max - x_min
    bh = y_max - y_min
    x_min = max(0, int(x_min - margin * bw))
    y_min = max(0, int(y_min - margin * bh))
    x_max = min(w, int(x_max + margin * bw))
    y_max = min(h, int(y_max + margin * bh))

    crop_w = x_max - x_min
    crop_h = y_max - y_min
    side = max(crop_w, crop_h)
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    cropped = rotated[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = rotated
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    # Apply CLAHE (same as training alignment pipeline)
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def get_face_bbox_from_landmarks(landmarks, w, h, margin=0.15):
    """Get bounding box from landmarks for drawing on frame."""
    xs = [landmarks[i].x * w for i in range(len(landmarks))]
    ys = [landmarks[i].y * h for i in range(len(landmarks))]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bw = x_max - x_min
    bh = y_max - y_min
    x_min = max(0, int(x_min - margin * bw))
    y_min = max(0, int(y_min - margin * bh))
    x_max = min(w, int(x_max + margin * bw))
    y_max = min(h, int(y_max + margin * bh))
    return x_min, y_min, x_max, y_max


# ──────────────────────────────────────────────
#  INFERENCE FUNCTIONS
# ──────────────────────────────────────────────

def preprocess_face(face_np, target_size=224):
    """Convert aligned face numpy array (RGB) to model input tensor."""
    face_pil = Image.fromarray(face_np)
    tensor = TF.to_tensor(face_pil)
    tensor = TF.normalize(tensor, NORM_MEAN, NORM_STD)
    return tensor.unsqueeze(0)


def predict_single(model, tensor, centroid_tensor, device):
    """Single forward pass → cosine similarity against centroids."""
    with torch.no_grad():
        embedding = model(tensor.to(device))
        # Cosine similarity against all identity centroids
        similarities = torch.matmul(embedding, centroid_tensor.T)  # (1, num_classes)
    return similarities.cpu().numpy()[0], embedding.cpu().numpy()[0]


def predict_tta(model, face_np, centroid_tensor, device, target_size=224):
    """Test-Time Augmentation: average embeddings over 4 views, then compute similarity."""
    tta_views = []

    # 1. Original
    tta_views.append(preprocess_face(face_np, target_size))

    # 2. Horizontal flip
    flipped = np.fliplr(face_np).copy()
    tta_views.append(preprocess_face(flipped, target_size))

    # 3-4. Small rotations
    for angle in [-5, 5]:
        h, w = face_np.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(face_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        tta_views.append(preprocess_face(rotated, target_size))

    # Average embeddings then compute similarity
    all_embs = []
    for view in tta_views:
        with torch.no_grad():
            emb = model(view.to(device))
            all_embs.append(emb)

    avg_emb = torch.mean(torch.cat(all_embs, dim=0), dim=0, keepdim=True)
    avg_emb = F.normalize(avg_emb, p=2, dim=1)

    similarities = torch.matmul(avg_emb, centroid_tensor.T).cpu().numpy()[0]
    return similarities


# ──────────────────────────────────────────────
#  DISPLAY HELPERS
# ──────────────────────────────────────────────

def get_color(identity_name, is_unknown=False):
    """Get color for an identity."""
    if is_unknown:
        return UNKNOWN_COLOR
    idx = LABEL_TO_IDX.get(identity_name, -1)
    if 0 <= idx < len(IDENTITY_COLORS):
        return IDENTITY_COLORS[idx]
    return (200, 200, 200)


def draw_prediction(frame, bbox, identity, confidence, all_sims, face_id=0,
                    show_bars=True, is_unknown=False):
    """Draw bounding box, identity label, and compact similarity bars for each face."""
    x1, y1, x2, y2 = bbox
    color = get_color(identity, is_unknown)
    frame_h, frame_w = frame.shape[:2]

    # Draw face bounding box
    thickness = 3 if not is_unknown else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Face ID circle
    cv2.circle(frame, (x1 + 14, y1 + 14), 14, color, -1)
    cv2.putText(frame, str(face_id + 1), (x1 + 8, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Draw identity label above box
    if is_unknown:
        label = f"Unknown ({confidence:.0%})"
    else:
        label = f"{identity.capitalize()} ({confidence:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thick = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thick)

    # Label background
    label_y1 = max(0, y1 - th - 12)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 5), font, font_scale, (0, 0, 0), thick)

    # Draw compact similarity bars below bounding box
    if show_bars and all_sims is not None:
        bar_x = x1
        bar_y_start = min(y2 + 5, frame_h - (5 * 18 + 5))
        bar_height = 14
        bar_max_width = min(160, x2 - x1, frame_w - bar_x - 5)
        if bar_max_width < 60:
            bar_max_width = 160
            bar_x = max(0, min(x1, frame_w - bar_max_width - 5))

        # Top 5 identities
        sorted_indices = np.argsort(all_sims)[::-1][:5]
        for i, idx in enumerate(sorted_indices):
            y = bar_y_start + i * (bar_height + 3)
            if y + bar_height > frame_h - 40:
                break
            name = CLASS_NAMES[idx]
            sim = max(0, all_sims[idx])  # clamp negative similarities
            bar_color = IDENTITY_COLORS[idx] if idx < len(IDENTITY_COLORS) else (200, 200, 200)

            # Bar background
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_width, y + bar_height),
                          (40, 40, 40), -1)
            # Bar fill (similarity 0→1 range)
            bar_width = int(sim * bar_max_width)
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height),
                          bar_color, -1)
            # Bar border
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_width, y + bar_height),
                          (80, 80, 80), 1)
            # Label
            cv2.putText(frame, f"{name}: {sim:.0%}", (bar_x + 2, y + bar_height - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)

    return frame


def draw_info(frame, fps, use_tta, num_faces, use_rejection):
    """Draw FPS, face count, and status info on frame."""
    h, w = frame.shape[:2]

    # Status bar at bottom
    cv2.rectangle(frame, (0, h - 35), (w, h), (30, 30, 30), -1)

    face_text = f"Faces: {num_faces}" if num_faces > 0 else "No face detected"
    rej_text = f"Reject: {'ON' if use_rejection else 'OFF'}"
    info_text = (f"FPS: {fps:.1f} | {face_text} | TTA: {'ON' if use_tta else 'OFF'} | "
                 f"{rej_text} | {DEVICE}")
    cv2.putText(frame, info_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Controls hint
    cv2.putText(frame, "q:Quit | s:Screenshot | t:TTA | f:Bars | r:Rejection",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return frame


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DPAIN Webcam Identity Recognition (Multi-Face)")
    print("=" * 60)
    print(f"  Device     : {DEVICE}")
    print(f"  Max faces  : 10")
    print(f"  Identities : {NUM_CLASSES}")
    print(f"  Names      : {', '.join(CLASS_NAMES)}")
    print(f"  Image size : {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Embed dim  : {EMBEDDING_DIM}")
    print(f"  Rejection  : {REJECTION_THRESHOLD:.3f}")
    print("=" * 60)

    # ── Load model ──
    print("\n  Loading DPAIN model...")
    model = DPAINModel(embedding_dim=EMBEDDING_DIM, dropout=0.0).to(DEVICE)

    if CHECKPOINT_PATH.exists():
        print(f"  Loading checkpoint: {CHECKPOINT_PATH.name}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('epoch', '?')
        test_loss = checkpoint.get('test_loss', '?')
        print(f"  ✓ Loaded (epoch {best_epoch}, test_loss: {test_loss})")
    else:
        print(f"  ✗ No checkpoint found at: {CHECKPOINT_PATH}")
        sys.exit(1)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── Load identity centroids ──
    print("  Loading identity centroids...")
    if CENTROIDS_PATH.exists():
        with open(CENTROIDS_PATH, 'r') as f:
            centroids_dict = json.load(f)
        # Build centroid array ordered by class index
        centroid_list = []
        for i in range(NUM_CLASSES):
            name = CLASS_NAMES[i]
            centroid_list.append(centroids_dict[name])
        centroid_array = np.array(centroid_list, dtype=np.float32)
        # Normalize centroids
        norms = np.linalg.norm(centroid_array, axis=1, keepdims=True) + 1e-8
        centroid_array = centroid_array / norms
        centroid_tensor = torch.tensor(centroid_array, dtype=torch.float32).to(DEVICE)
        print(f"  ✓ Loaded {len(centroid_list)} identity centroids")
    else:
        print(f"  ✗ Centroids not found at: {CENTROIDS_PATH}")
        sys.exit(1)

    # ── Create MediaPipe landmarker ──
    print("  Initializing MediaPipe face detector...")
    landmarker = create_landmarker()
    print("  ✓ Face detector ready")

    # ── Open webcam ──
    print("\n  Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ✗ Cannot open webcam! Check your camera connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ✓ Webcam opened: {actual_w}×{actual_h}")

    # ── Settings ──
    use_tta = False
    show_bars = True
    use_rejection = True
    fps_buffer = deque(maxlen=30)
    screenshot_count = 0
    MAX_TRACKED_FACES = 10

    # Per-face prediction smoothing (keyed by face index)
    face_sim_buffers = {i: deque(maxlen=5) for i in range(MAX_TRACKED_FACES)}

    print("\n  ✓ Starting live identity recognition... Press 'q' or ESC to quit.\n")

    try:
        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("  ✗ Failed to read frame")
                break

            # Convert BGR → RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # Detect all faces
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect(mp_image)

            num_faces = len(result.face_landmarks) if result.face_landmarks else 0

            if num_faces > 0:
                for face_idx, landmarks in enumerate(result.face_landmarks):
                    if face_idx >= MAX_TRACKED_FACES:
                        break

                    # Bounding box for display
                    bbox = get_face_bbox_from_landmarks(landmarks, w, h)

                    # Align face
                    try:
                        aligned_face = align_face(frame_rgb, landmarks, w, h,
                                                   margin=0.3, target_size=IMAGE_SIZE)

                        # Predict
                        if use_tta:
                            sims = predict_tta(model, aligned_face, centroid_tensor,
                                               DEVICE, IMAGE_SIZE)
                        else:
                            tensor = preprocess_face(aligned_face, IMAGE_SIZE)
                            sims, _ = predict_single(model, tensor, centroid_tensor, DEVICE)

                        # Smooth similarities per face
                        face_sim_buffers[face_idx].append(sims)
                        avg_sims = np.mean(list(face_sim_buffers[face_idx]), axis=0)

                        pred_class = np.argmax(avg_sims)
                        identity = CLASS_NAMES[pred_class]
                        confidence = avg_sims[pred_class]

                        # Unknown rejection
                        is_unknown = use_rejection and confidence < REJECTION_THRESHOLD

                        # Draw results
                        frame = draw_prediction(frame, bbox, identity, confidence,
                                                avg_sims, face_id=face_idx,
                                                show_bars=show_bars, is_unknown=is_unknown)

                    except Exception as e:
                        # Face alignment failed
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                      (0, 255, 255), 2)
                        cv2.putText(frame, f"Face {face_idx+1}: align failed",
                                    (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Clear buffers for faces no longer visible
                for i in range(num_faces, MAX_TRACKED_FACES):
                    face_sim_buffers[i].clear()
            else:
                # No faces — clear all
                for buf in face_sim_buffers.values():
                    buf.clear()

            # FPS
            elapsed = time.time() - t_start
            fps_buffer.append(1.0 / max(elapsed, 1e-6))
            avg_fps = np.mean(fps_buffer)

            # Draw info bar
            frame = draw_info(frame, avg_fps, use_tta, num_faces, use_rejection)

            # Show
            cv2.imshow('DPAIN Identity Recognition', frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                print("\n  Quitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = SCREENSHOT_DIR / f"identity_screenshot_{screenshot_count:04d}.png"
                cv2.imwrite(str(filename), frame)
                print(f"  📸 Screenshot saved: {filename.name}")
            elif key == ord('t'):
                use_tta = not use_tta
                for buf in face_sim_buffers.values():
                    buf.clear()
                print(f"  TTA {'enabled' if use_tta else 'disabled'}")
            elif key == ord('f'):
                show_bars = not show_bars
                print(f"  Similarity bars {'shown' if show_bars else 'hidden'}")
            elif key == ord('r'):
                use_rejection = not use_rejection
                print(f"  Unknown rejection {'enabled' if use_rejection else 'disabled'}")

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("\n  ✓ Webcam released. Goodbye!")


if __name__ == '__main__':
    main()
