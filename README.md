# Research2.0 — Facial Expression & Identity Recognition

Two deep-learning research projects trained from scratch on limited hardware (NVIDIA GTX 1650, 4 GB VRAM):

| Project | Task | Architecture | Accuracy |
|---------|------|--------------|----------|
| **EmoNeXt** | 8-class Facial Expression Recognition (FER) | EmoNeXt CNN + MFSA + FRAP | **72.85 %** |
| **Identity_v1 (DPAIN)** | Person Identity Recognition | Dual-Path CNN + Adaptive Fusion Gate | **95.38 %** |

Both models were built without pretrained weights ("from scratch") to demonstrate task-specific architectural design.

---

## Repository Structure

```
Research_Final/
├── requirements.txt              ← unified dependencies (this repo)
├── README.md                     ← you are here
│
├── emonext/                      ← EmoNeXt FER project
│   ├── AffectNet_EmoNeXt_Scratch.ipynb
│   ├── emonext_outputs/
│   │   ├── saved_models/best_model.pth
│   │   ├── plots/
│   │   └── logs/
│   └── README.md                 ← detailed research report
│
└── Identity_v1/                  ← DPAIN Identity project
    ├── Identity_Recognition_System.ipynb
    ├── webcam_identity.py
    ├── face_landmarker_v2_with_blendshapes.task
    ├── identity_outputs/
    │   ├── checkpoints/best_model.pth
    │   ├── embeddings/
    │   └── figures/
    └── README.md                 ← detailed research report
```

> **Note:** Dataset folders (`AffectNet_*`, `Affect8_*`, `Identity_Dataset/`) are excluded from this repo due to size. See the individual READMEs for dataset preparation instructions.

---

## System Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| **OS** | Ubuntu 20.04 | Ubuntu 22.04 / 24.04 |
| **Python** | 3.10 | 3.10.12 |
| **GPU** | NVIDIA (4 GB VRAM) | GTX 1650 4 GB |
| **CUDA** | 11.8 | 12.8 |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 20 GB free | — |

CPU-only training is supported but will be significantly slower.

---

## Installation

### 1 — Clone the repo

```bash
git clone https://github.com/harshad-dhokane/research2.0.git
cd research2.0
```

### 2 — Create a Python 3.10 virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3 — Install PyTorch (CUDA-specific step)

Choose **one** of the following depending on your GPU / CUDA version:

**CUDA 12.8 (recommended, GTX 1650+ with latest drivers):**
```bash
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

**CUDA 11.8:**
```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

**CPU only (no GPU):**
```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

Verify installation:
```bash
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

### 4 — Install all other dependencies

```bash
pip install -r requirements.txt
```

### 5 — Download MediaPipe face landmark model (required for Identity_v1)

```bash
wget -O Identity_v1/face_landmarker_v2_with_blendshapes.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

> The file is ~29 MB. If `wget` fails, download it manually from the [MediaPipe models page](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) and place it inside `Identity_v1/`.

---

## Running the Projects

### EmoNeXt — Facial Expression Recognition

**Training / evaluation (Jupyter):**
```bash
source .venv/bin/activate
jupyter lab
# open: emonext/AffectNet_EmoNeXt_Scratch.ipynb
# Run All Cells
```

**Dataset required:** AffectNet 8-class  
Place images in:
```
emonext/Dataset_Combined/
    Anger/  Contempt/  disgust/  fear/  happy/  neutral/  sad/  surprise/
```

**Output artifacts:**
- Best model weights: `emonext/emonext_outputs/saved_models/best_model.pth`
- Training logs: `emonext/emonext_outputs/logs/`
- Plots: `emonext/emonext_outputs/plots/`

---

### Identity_v1 — DPAIN Identity Recognition

**Training / evaluation (Jupyter):**
```bash
source .venv/bin/activate
jupyter lab
# open: Identity_v1/Identity_Recognition_System.ipynb
# Run All Cells
```

**Dataset required:** Identity Dataset (15 subjects)  
Place images in:
```
Identity_v1/Identity_Dataset/
    <person_name>/
        image1.jpg
        image2.jpg
        ...
```

**Live webcam demo:**
```bash
source .venv/bin/activate
cd Identity_v1
python webcam_identity.py
```
> Press `q` to quit the webcam window.

**Output artifacts:**
- Best model weights: `Identity_v1/identity_outputs/checkpoints/best_model.pth`
- Embeddings: `Identity_v1/identity_outputs/embeddings/`
- Evaluation figures: `Identity_v1/identity_outputs/figures/`

---

## Troubleshooting

### `torch.cuda.is_available()` returns `False`
- Ensure your NVIDIA driver is installed: `nvidia-smi`
- Verify you installed the correct PyTorch CUDA variant (Step 3 above)
- Driver version ≥ 525 is required for CUDA 12.x

### MediaPipe `FileNotFoundError`
- Confirm `face_landmarker_v2_with_blendshapes.task` exists inside `Identity_v1/`
- Re-run Step 5 of the installation

### `onnxruntime` GPU inference
- By default `onnxruntime` runs on CPU.  
- For GPU ONNX inference, replace with: `pip install onnxruntime-gpu==1.23.2`

### Jupyter kernel not found
```bash
python -m ipykernel install --user --name=research_env --display-name "Research2.0 (.venv)"
```
Then select `Research2.0 (.venv)` as the kernel in JupyterLab.

### `NaN` loss during Identity training on low-VRAM GPU
- AMP (Automatic Mixed Precision) is intentionally disabled for GTX 1650 — FP16 `torch.cdist` produces NaN.
- Training will run in FP32; expect ~30 % slower throughput but stable gradients.

---

## Results Summary

### EmoNeXt (AffectNet 8-class)

| Metric | Value |
|--------|-------|
| Test Accuracy | **72.85 %** |
| Macro F1 | **69.66 %** |
| Parameters | 10.41 M |
| Training data | 29,042 images |

### DPAIN Identity Recognition

| Metric | Value |
|--------|-------|
| Test Accuracy | **95.38 %** |
| EER | **2.058 %** |
| Genuine Similarity Gap | **0.906** |
| Parameters | 2.0 M |
| Training data | 703 images (15 subjects) |

---

## Detailed Documentation

- EmoNeXt research report → [emonext/README.md](emonext/README.md)
- DPAIN identity research report → [Identity_v1/README.md](Identity_v1/README.md)

---

## Citation

If you use this work, please cite:
```
@misc{dhokane2025research2,
  author    = {Harshad Dhokane},
  title     = {EmoNeXt and DPAIN: From-Scratch Deep Learning for FER and Identity Recognition},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/harshad-dhokane/research2.0}
}
```
