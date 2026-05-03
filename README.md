# 👁️ Eye-Know-You

### Real-Time Face Recognition, Object Detection & Navigation System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-00FFFF?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Eye-Know-You** is a real-time computer-vision pipeline that fuses **face recognition**, **YOLOv8 object detection**, and **spatial navigation assistance** — announcing recognised people, detected objects, and proximity warnings through **text-to-speech**, all from a single webcam feed.

---

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Configuration](#-configuration) · [Usage](#-usage) · [Troubleshooting](#-troubleshooting)

---

## ✨ Features

| Category | Capability | Details |
|:---|:---|:---|
| 🧑 **Face Recognition** | Real-time identification | dlib 128-d embeddings with configurable distance tolerance |
| 📦 **Object Detection** | YOLOv8 tracking | 80+ COCO classes with persistent track IDs across frames |
| 🧭 **Navigation** | Spatial awareness | Distance estimation, zone classification, tier-based proximity alerts |
| 🔊 **Audio Announcements** | Text-to-Speech | Cross-platform `pyttsx3` with cooldown de-duplication and per-tier nav warnings |
| 🛡️ **Reliability & Safety** | Core fail-safes | Camera auto-recovery, strict face enrollment validation, and SHA256 model integrity checks |
| 💾 **Smart Caching** | JSON embedding cache | Encoded faces saved as JSON (no unsafe pickle) and reloaded instantly |
| 🔄 **Hot-Reload** | Live database rebuild | Press `R` at runtime to re-scan and rebuild the face database |
| ⚙️ **Env-based Config** | `.env` file support | All parameters configurable via `.env` without touching source code |
| 🖥️ **CLI Interface** | Full argument support | Override any setting directly from the command line |

---

## 🏗️ Architecture

The system is built on clean, modular principles — each module has a single responsibility and communicates through well-defined interfaces.

### High-Level Data Flow

```
┌─────────────┐
│   Webcam    │
│  (OpenCV)   │
└──────┬──────┘
       │  BGR frame
       ▼
┌──────────────────────────────────────────────────────────────┐
│                      VisionPipeline                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Step 1: Navigation (parallel)                       │    │
│  │  YoloDetector.track_and_draw() → distance + zone     │    │
│  │  → colour-coded bounding boxes + nav TTS warnings    │    │
│  └──────────────────────────┬───────────────────────────┘    │
│                             │                                │
│  ┌──────────────────────────▼───────────────────────────┐    │
│  │  Step 2: Face Recognition (parallel)                 │    │
│  │  FaceDetector → FaceEncoder → FaceMatcher            │    │
│  │  → name labels + person TTS                          │    │
│  └──────────────────────────┬───────────────────────────┘    │
│                             │                                │
│  ┌──────────────────────────▼───────────────────────────┐    │
│  │  Step 3: Object TTS (reuses Step 1 results)          │    │
│  └──────────────────────────┬───────────────────────────┘    │
│                             │                                │
│  ┌──────────────────────────▼───────────────────────────┐    │
│  │  Step 4: HUD overlay (distance-tier legend)          │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌───────────────┐                                           │
│  │   Speaker     │──▶ "Ahmed is here."                       │
│  │   (pyttsx3)   │    "Danger! chair ahead, 65cm, stop!"    │
│  └───────────────┘                                           │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────┐
│  OpenCV Window  │
│  (Annotated)    │
└─────────────────┘
```

> **Note:** Steps 1 and 2 run concurrently in a `ThreadPoolExecutor`. Each worker receives its own frame copy, eliminating data races.

### Project Structure

```
Eye-Know-You/
├── 📄 .env                  ◀ Your local config (git-ignored)
├── 📄 .env.example          ◀ Template for new contributors
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 📂 data/
│   ├── 📂 raw/              ◀ Per-person image folders (your training data)
│   │   ├── 📂 Alice/
│   │   │   ├── photo1.jpg
│   │   │   └── photo2.png
│   │   └── 📂 Bob/
│   │       └── photo1.jpg
│   └── 📂 embeddings/       ◀ Auto-generated JSON cache
│       └── known_faces.json
│
├── 📂 models/
│   └── 📂 detection/
│       └── yolov8n.pt       ◀ YOLO weights (auto-downloaded or manual)
│
└── 📂 src/
    ├── 📄 app.py             ◀ Entry point & camera loop
    ├── 📄 config.py          ◀ Centralised configuration (reads from .env)
    │
    ├── 📂 recognition/       ◀ Face-recognition sub-system
    │   ├── face_detector.py  ◀ Locates faces (bounding boxes)
    │   ├── face_encoder.py   ◀ Converts faces → 128-d vectors
    │   └── face_matcher.py   ◀ Matches vectors against known database
    │
    ├── 📂 detection/         ◀ Object-detection sub-system
    │   └── yolo_detector.py  ◀ YOLOv8 wrapper (detect + track methods)
    │
    ├── 📂 navigation/        ◀ Spatial navigation sub-system
    │   └── nav.py            ◀ Zone classification, distance tiers, constants
    │
    ├── 📂 pipeline/          ◀ Orchestration layer
    │   └── main_pipeline.py  ◀ Ties all modules together
    │
    └── 📂 audio/             ◀ Text-to-speech sub-system
        └── speaker.py        ◀ Background TTS with cooldown + nav warnings
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- A webcam (USB or built-in)
- ~500 MB disk space for model weights and dependencies

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Eye-Know-You.git
cd Eye-Know-You
```

### 2. Create & Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> **Blocked by execution policy?** Run this first:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `dlib` compilation requires CMake and a C++ compiler. On Windows `dlib-bin` (precompiled) is used.
> On Ubuntu you may need: `sudo apt install cmake build-essential`

### 4. Configure Your Environment

```bash
cp .env.example .env
```

Open `.env` and set your values (see [Configuration](#-configuration) for details). The project runs with sensible defaults even without a `.env` file.

### 5. Prepare YOLO Weights

Place the YOLOv8n weights at:

```
models/detection/yolov8n.pt
```

If missing, Ultralytics will attempt to auto-download them on first run.

### 6. Add Your Known Faces

Create one sub-folder per person inside `data/raw/`:

```
data/raw/
├── Alice/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.png
├── Bob/
│   └── profile.jpg
└── Charlie/
    ├── img_001.jpeg
    └── img_002.webp
```

> **Tips for best accuracy:**
> - Use **3–5 clear, well-lit photos** per person
> - Include different angles (front, slight left/right)
> - Ensure only **one face** is visible per image
> - Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`

---

## ⚙️ Configuration

All parameters can be set in three places, in order of priority:

```
CLI arguments  >  environment variables / .env file  >  built-in defaults
```

### `.env` File

Copy `.env.example` to `.env` and customise:

```ini
# Camera
EKY_CAMERA_INDEX=0          # 0 = built-in, 1/2/... = USB cameras

# Face recognition
EKY_FACE_TOLERANCE=0.5      # lower = stricter matching
EKY_FACE_MODEL=hog          # "hog" (fast) or "cnn" (GPU-accelerated)

# Object detection
EKY_DETECTION_THRESHOLD=0.5 # minimum YOLO confidence (0.0–1.0)

# Navigation
EKY_FOCAL_LENGTH=820        # camera focal length in pixels

# Audio
EKY_COOLDOWN_SECONDS=3      # seconds between repeat announcements

# Paths (relative to project root)
EKY_RAW_DIR=data/raw
EKY_EMBEDDINGS_DIR=data/embeddings
EKY_DETECTION_MODEL=models/detection/yolov8n.pt

# Model Security
EKY_MODEL_HASH=f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36
```

### CLI Arguments

CLI flags override `.env` values at runtime without modifying any file:

| Argument | Type | Description |
|:---|:---:|:---|
| `--camera` | `int` | OpenCV camera device index |
| `--tolerance` | `float` | Face-match distance threshold — **lower = stricter** |
| `--cooldown` | `int` | Seconds before re-announcing the same name |
| `--detection-threshold` | `float` | Minimum YOLO confidence score |

### Tolerance Guide

| Value | Behaviour | Use Case |
|:---:|:---|:---|
| `0.3` | Very strict — few false positives, may miss some matches | High-security scenarios |
| `0.4` | Strict — good balance for small databases | Recommended for < 10 people |
| `0.5` | Default — balanced accuracy | General use |
| `0.6` | Lenient — catches more matches, may confuse similar faces | Large databases, varied lighting |

### Distance Calibration

The `EKY_FOCAL_LENGTH` value drives monocular distance estimation:

```
distance_cm = (known_object_width_cm × focal_length) / pixel_width
```

To calibrate for your camera: place a known object at a measured distance, note its pixel width, then solve for:

```
focal_length = (distance_cm × pixel_width) / known_object_width_cm
```

### Navigation Tier Thresholds

| Tier | Distance | Colour | Audio |
|:---|:---|:---|:---|
| `CRITICAL` | < 80 cm | 🔴 Red | *"Danger! chair ahead, 65cm, stop!"* |
| `NEAR` | < 150 cm | 🟠 Orange | *"Warning! chair ahead, 120cm"* |
| `MEDIUM` | < 300 cm | 🟡 Yellow | *"chair ahead"* |
| Silent | > 300 cm | ⚪ White | *(no announcement)* |

### Model Integrity & Custom YOLO Weights

To prevent executing tampered or corrupted models, the system checks the `yolov8n.pt` SHA256 hash before loading it. 
If you decide to retrain YOLO or use a custom weights file, you must update the `EKY_MODEL_HASH` in your `.env` file to match the new file's hash.

**How to obtain the SHA256 hash:**

*Using Python (Cross-platform):*
```bash
python -c "import hashlib; print(hashlib.sha256(open('models/detection/yolov8n.pt', 'rb').read()).hexdigest())"
```

*Using Linux/macOS Terminal:*
```bash
sha256sum models/detection/yolov8n.pt
# or on macOS: shasum -a 256 models/detection/yolov8n.pt
```

*Using Windows PowerShell:*
```powershell
(Get-FileHash models\detection\yolov8n.pt -Algorithm SHA256).Hash.ToLower()
```
Then paste the resulting lowercase hash into your `.env` file as `EKY_MODEL_HASH=your_new_hash_here`.

---

## 🎮 Usage

### Basic Run

```bash
python -m src.app
```

### With Custom Options

```bash
python -m src.app --camera 0 --tolerance 0.4 --cooldown 5 --detection-threshold 0.6
```

### Runtime Controls

| Key | Action |
|:---:|:---|
| <kbd>Esc</kbd> | Quit the application gracefully |
| <kbd>R</kbd> | Rebuild the face database (hot-reload after adding new images) |

---

## 🧠 How It Works — End to End

### Startup Phase

```
1. Load .env  →  parse CLI args  →  build Config
2. Load YOLO weights (yolov8n.pt)
3. Load/build face database:
   a. Try loading from  data/embeddings/known_faces.json
   b. If missing → scan  data/raw/  → detect + encode each image → save JSON
4. Start TTS background worker thread
5. Open webcam  →  enter main loop
```

### Per-Frame Processing

```
1. Read BGR frame from webcam
2. Fork into two parallel threads:
   a. Navigation thread — track_and_draw() → distance → zone → tier → TTS warn
   b. Face thread       — detect_faces() on a clean RGB copy
3. Merge results
4. Encode faces → match against database → TTS speak names
5. Queue object TTS announcements
6. Draw face boxes + name labels on the nav-annotated frame
7. Overlay HUD distance legend
8. Display composite frame in OpenCV window
9. Check for key press (Esc → quit, R → rebuild DB)
```

### Shutdown Phase

```
1. Signal TTS worker thread to stop
2. Release camera device
3. Destroy all OpenCV windows
4. Log "Cleanup complete – goodbye."
```

---

## 🛡️ Security & Design Decisions

| Concern | Solution |
|:---|:---|
| **No pickle** | Face encodings stored as JSON (lists of floats). Pickle allows arbitrary code execution on deserialization — JSON is safe. |
| **No hard-coded secrets** | All tuneable values live in `.env` which is git-ignored. `.env.example` documents every variable safely. |
| **Frame immutability** | Navigation and face threads each receive their own `frame.copy()`. The original is never mutated. |
| **Cooldown-based memory** | TTS uses timestamp-based cooldowns; nav cache entries older than 30 s are evicted automatically. |
| **Graceful degradation** | If a frame read fails, the loop continues. All exceptions in the frame loop are caught and logged. |
| **Resource cleanup** | A `finally` block guarantees the camera, TTS engine, and thread pool are released — even on crash. |
| **Supported extensions** | Only `.jpg`, `.jpeg`, `.png`, `.webp` are processed; other files are silently skipped. |
| **Cross-platform TTS** | `pyttsx3` works on Windows (SAPI5), Linux (espeak), and macOS (NSSpeechSynthesizer). |

---

## 🔧 Troubleshooting

<details>
<summary><strong>Camera not opening</strong></summary>

- Try different `--camera` values: `0`, `1`, `2`, etc. or set `EKY_CAMERA_INDEX` in `.env`
- Check if another app (Zoom, Teams) is holding the camera
- On Linux, ensure your user has access to `/dev/video*`
</details>

<details>
<summary><strong>"No face found in: photo.jpg"</strong></summary>

- Ensure the image contains exactly **one clearly visible face**
- Try a different photo with better lighting and higher resolution
- Blurry, low-res, or side-profile images may fail HOG detection
</details>

<details>
<summary><strong>dlib / CMake installation error</strong></summary>

**Windows:** Use `dlib-bin` (precompiled) — already in `requirements.txt`

**Linux:**
```bash
sudo apt install cmake build-essential python3-dev
pip install dlib
```

**macOS:**
```bash
brew install cmake
pip install dlib
```
</details>

<details>
<summary><strong>Wrong faces matched (false positives)</strong></summary>

- **Lower** the tolerance: `EKY_FACE_TOLERANCE=0.35` in `.env` or `--tolerance 0.35`
- Add **more reference photos** per person (3–5 recommended)
- Ensure reference photos are well-lit and show the face clearly
</details>

<details>
<summary><strong>TTS not working / no audio</strong></summary>

- **Windows:** Works out of the box (uses SAPI5)
- **Linux:** Install espeak: `sudo apt install espeak`
- **macOS:** Uses NSSpeechSynthesizer (built-in)
</details>

<details>
<summary><strong>YOLO model missing</strong></summary>

If `models/detection/yolov8n.pt` is missing, Ultralytics will attempt to auto-download it. To download manually:

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```
</details>

<details>
<summary><strong>Distance estimation is inaccurate</strong></summary>

- Calibrate `EKY_FOCAL_LENGTH` for your camera (see [Distance Calibration](#distance-calibration))
- Ensure the object is fully visible in frame (clipped objects give wrong pixel widths)
- The system uses a 5-frame moving average to smooth readings
</details>

---

## 📦 Core Dependencies

| Package | Version | Purpose |
|:---|:---:|:---|
| `opencv-python` | 4.13 | Camera capture, image processing, UI display |
| `face-recognition` | 1.3.0 | Face detection (HOG/CNN) and 128-d encoding |
| `dlib-bin` | 20.0.1 | Pre-compiled dlib backend for face_recognition |
| `ultralytics` | 8.4.38 | YOLOv8 inference and tracking engine |
| `torch` | 2.11.0 | PyTorch backend for YOLO |
| `torchvision` | 0.26.0 | Image transforms for the neural network pipeline |
| `pyttsx3` | 2.99 | Cross-platform text-to-speech |
| `python-dotenv` | 1.2.2 | `.env` file loading for configuration |
| `numpy` | 2.4.4 | Numerical operations and array manipulation |
| `scipy` | 1.17.1 | Scientific computing utilities |

---

<div align="center">

*Eye-Know-You — Because your computer should know who it's looking at.*

</div>