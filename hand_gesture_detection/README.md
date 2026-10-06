# Real-Time Hand, Finger Detection & Gesture Recognition AI

A high-performance, real-time computer vision system built with **Google MediaPipe Tasks API** and **OpenCV** in Python.

---

## 🌟 Key Features

1. **Multi-Hand Tracking & Detection**:
   - Detects single or multiple hands simultaneously.
   - Distinguishes **Left Hand** vs. **Right Hand** with high accuracy.
   - Tracks all **21 3D landmarks** per hand with full skeletal connectivity.

2. **Finger State & Count Analysis**:
   - Individual status detection for each finger: **Thumb, Index, Middle, Ring, Pinky**.
   - Real-time finger extension detection (Extended `1` vs. Folded `0`).
   - Live finger counter (0 to 5 extended fingers per hand).

3. **Hybrid AI & Geometric Gesture Recognition**:
   - **Deep Learning Model Gestures** (Google MediaPipe Gesture Recognizer):
     - ✊ **Closed Fist** (`Closed_Fist`)
     - ✋ **Open Palm** (`Open_Palm`)
     - ☝️ **Pointing Up** (`Pointing_Up`)
     - 👍 **Thumbs Up** (`Thumb_Up`)
     - 👎 **Thumbs Down** (`Thumb_Down`)
     - ✌️ **Victory / Peace** (`Victory`)
     - 🤟 **I Love You** (`ILoveYou`)
   - **Geometric / Heuristic Custom Gestures**:
     - 👌 **OK Sign** (Thumb & Index touching circle, remaining 3 extended)
     - 🤘 **Rock / Horns** (Index & Pinky extended, Middle & Ring folded)
     - 🤙 **Call Me** (Thumb & Pinky extended, middle fingers folded)
     - 👉 **Gun / Point** (Thumb & Index extended)
     - 🤏 **Pinch / Click** (Precision Thumb-Index pinch tracking with visual distance meter)

4. **Sci-Fi Cyber HUD Overlay**:
   - Real-time **FPS counter** and detected hand counter.
   - Dynamic corner-bracket bounding boxes.
   - Per-hand status cards with confidence progress bar, extended finger count, and 5-finger status badges (`T:1`, `I:1`, `M:0`, `R:0`, `P:0`).
   - Interactive keyboard shortcuts to toggle layers on the fly.
   - One-key snapshot capture (`ENTER`) saved directly to disk.

---

## 📂 Project Structure

```
hand_gesture_detection/
├── detector.py             # Core HandGestureDetector engine
├── visualizer.py           # Futuristic HUD & skeleton renderer
├── main.py                 # Real-time interactive webcam application
├── test_image.py           # Static image detection test runner
├── gesture_recognizer.task # MediaPipe deep learning model bundle
├── requirements.txt        # Python package dependencies
├── snapshots/              # Captured screenshots folder
└── README.md               # Documentation & usage guide
```

## 🌐 Live Web Version (GitHub Pages Ready)

This repository includes a standalone Web application ([`index.html`](index.html)) built with **MediaPipe WebAssembly & WebGL**:
- Runs 100% in the browser on desktop and mobile devices without installing Python.
- Deploys **for free** on **GitHub Pages** in 1 click!

To test locally in your browser:
Double-click `index.html` or open it with your web browser.

---

## 🚀 Quick Start

### 1. Requirements

Ensure you have Python installed. The required packages are:
- `opencv-python`
- `mediapipe`
- `numpy`

To install:
```bash
py -m pip install -r requirements.txt
```

### 2. Run with Webcam (Live Feed)

```bash
py main.py
```

#### Custom Options:
```bash
# Use a secondary webcam (e.g., USB external webcam index 1)
py main.py --camera 1

# Change resolution (e.g. 1920x1080)
py main.py --width 1920 --height 1080

# Track up to 4 hands at once
py main.py --max-hands 4

# Disable selfie mirror mode
py main.py --no-mirror
```

### 3. Run on a Static Image

```bash
# Test on sample image:
py test_image.py

# Test on your own photo:
py test_image.py --image "path/to/photo.jpg" --output "annotated_result.png"
```

---

## ⌨️ Interactive Controls (in Webcam Window)

| Key | Action |
|---|---|
| **`S`** | Toggle Hand Skeleton & Colored Joints |
| **`B`** | Toggle Bounding Box & Floating Badge |
| **`C`** | Toggle Contrast Boost (Enhances Palm in Flat/Washed Lighting) |
| **`P`** | Toggle Pinch Detector & Distance Line |
| **`H`** | Toggle Cyber HUD Overlay & Status Panels |
| **`M`** | Toggle Mirror / Selfie Mode |
| **`SPACE`** | Pause / Resume Live Feed |
| **`ENTER`** | Save Snapshot to `snapshots/` folder |
| **`Q` / `ESC`** | Exit application |
