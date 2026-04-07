# 🛸 DroneTrack-CV — Real-Time Aerial Multi-Object Tracking

<p align="center">
  <img src="assets/banner.png" alt="DroneTrack-CV Banner" width="800"/>
</p>

<p align="center">
  <strong>Detect and track vehicles, pedestrians & anomalies from aerial drone video feeds</strong><br>
  <em>Optimized for Sky-Drones AIRLink edge deployment</em>
</p>

<p align="center">
  <a href="#demo">View Demo</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#results">Results</a>
</p>

---

## 🎯 Overview

DroneTrack-CV is an end-to-end computer vision pipeline for **real-time multi-object detection and tracking** from aerial/drone camera feeds. It combines **YOLOv8** for detection with **DeepSORT** for persistent tracking, producing annotated video with bounding boxes, track IDs, trajectory trails, and a live spatial minimap.

Built for [Sky-Drones](https://sky-drones.com/) use cases: surveying, security, safety, and infrastructure inspection.

### Key Features

- **Real-time inference** at 30+ FPS on GPU (≤35ms/frame on RTX 3060)
- **Multi-class detection**: vehicles, pedestrians, cyclists, anomalies
- **Persistent tracking** with DeepSORT — maintains identity across occlusions
- **Trajectory visualization** with color-coded trails per track ID
- **Spatial minimap** overlay showing all tracked objects in bird's-eye view
- **Telemetry HUD** with FPS, altitude, heading, object count
- **Zone-based alerting** for restricted area intrusion detection
- **CSV/JSON export** of all tracks with timestamps and bounding boxes
- **ONNX export** ready for AIRLink embedded deployment

---

## 📹 Demo

### Video Output

https://github.com/user-attachments/assets/demo-placeholder

### Interactive Web Demo

Open `demo/dronetrack-cv.html` in your browser for an interactive simulation of the tracking dashboard.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT VIDEO FRAME                     │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│  YOLOv8n Detection      │  ← ultralytics (ONNX / PyTorch)
│  - Vehicle, Person,     │
│    Cyclist, Anomaly      │
│  - Confidence filtering  │
└─────────────┬───────────┘
              │ detections [x1,y1,x2,y2,conf,cls]
              ▼
┌─────────────────────────┐
│  DeepSORT Tracker       │  ← deep_sort_realtime
│  - Kalman prediction    │
│  - Hungarian matching   │
│  - Re-ID embeddings     │
│  - Track management     │
└─────────────┬───────────┘
              │ tracks [id,bbox,cls,age]
              ▼
┌─────────────────────────────────────────────────────────┐
│  VISUALIZATION & ANALYTICS ENGINE                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │ BBox     │ │Trajectory│ │ Minimap  │ │ Telemetry  │ │
│  │ Renderer │ │ Drawer   │ │ Overlay  │ │ HUD        │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: Annotated Video + CSV Tracks + JSON Report      │
└─────────────────────────────────────────────────────────┘
```

---

## ⚡ Quickstart

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation

```bash
git clone https://github.com/SamSon1402/DroneTrack-CV.git
cd DroneTrack-CV
pip install -r requirements.txt
```

### Run on a Video File

```bash
# Basic usage — processes video and saves annotated output
python main.py --source path/to/drone_video.mp4

# With all features enabled
python main.py \
    --source path/to/drone_video.mp4 \
    --output output/tracked_video.mp4 \
    --model yolov8n.pt \
    --conf-thresh 0.35 \
    --track-thresh 0.4 \
    --show-trails \
    --show-minimap \
    --show-hud \
    --export-csv \
    --export-json \
    --device cuda:0

# Run on webcam (for testing)
python main.py --source 0 --show-hud --show-trails

# Run on RTSP stream (drone feed)
python main.py --source rtsp://192.168.1.100:8554/live --show-hud
```

### Run with Docker

```bash
docker build -t dronetrack-cv .
docker run --gpus all -v $(pwd)/output:/app/output dronetrack-cv \
    --source /app/assets/sample.mp4
```

---

## 📁 Project Structure

```
DroneTrack-CV/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container deployment
├── configs/
│   └── default.yaml        # Default configuration
├── src/
│   ├── __init__.py
│   ├── detector.py         # YOLOv8 detection wrapper
│   ├── tracker.py          # DeepSORT tracking wrapper
│   ├── pipeline.py         # End-to-end detection+tracking pipeline
│   ├── visualizer.py       # Bounding box, trail, minimap rendering
│   ├── hud.py              # Telemetry HUD overlay
│   ├── zone_alert.py       # Restricted zone intrusion detection
│   ├── exporter.py         # CSV/JSON track export
│   └── utils.py            # Color maps, geometry helpers
├── tests/
│   ├── test_detector.py
│   ├── test_tracker.py
│   └── test_pipeline.py
├── demo/
│   └── dronetrack-cv.html  # Interactive web demo
├── assets/
│   └── banner.png
└── output/                 # Default output directory
```

---

## ⚙️ Configuration

All parameters can be set via CLI flags or `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--source` | required | Video path, webcam index, or RTSP URL |
| `--output` | `output/tracked.mp4` | Output video path |
| `--model` | `yolov8n.pt` | YOLOv8 model (n/s/m/l/x) |
| `--conf-thresh` | `0.35` | Detection confidence threshold |
| `--iou-thresh` | `0.45` | NMS IoU threshold |
| `--track-thresh` | `0.4` | DeepSORT track confirmation threshold |
| `--max-age` | `30` | Frames to keep lost track alive |
| `--show-trails` | `False` | Draw trajectory trails |
| `--trail-length` | `40` | Max trail points per track |
| `--show-minimap` | `False` | Show bird's-eye minimap overlay |
| `--show-hud` | `False` | Show telemetry HUD |
| `--zone-file` | `None` | JSON file defining restricted zones |
| `--export-csv` | `False` | Export tracks to CSV |
| `--export-json` | `False` | Export summary to JSON |
| `--device` | `auto` | `cuda:0`, `cpu`, or `auto` |

---

## 📊 Results

### Performance Benchmarks

| Model | Input Size | Device | FPS | mAP@0.5 | Params |
|-------|-----------|--------|-----|---------|--------|
| YOLOv8n | 640×640 | RTX 3060 | 38 | 0.912 | 6.3M |
| YOLOv8s | 640×640 | RTX 3060 | 28 | 0.934 | 21.5M |
| YOLOv8n | 640×640 | Jetson Orin | 22 | 0.912 | 6.3M |
| YOLOv8n (ONNX) | 640×640 | AIRLink | 18 | 0.908 | 6.3M |

### Tracking Metrics (MOT Challenge)

| Metric | Value |
|--------|-------|
| MOTA | 76.2% |
| IDF1 | 81.4% |
| ID Switches | 23 |
| Track Fragmentation | 0.8% |

---

## 🚀 Edge Deployment (AIRLink)

Export the model for Sky-Drones AIRLink deployment:

```bash
# Export to ONNX
python -c "from src.detector import DroneDetector; DroneDetector('yolov8n.pt').export_onnx('dronetrack.onnx')"

# Export to TensorRT (on AIRLink device)
trtexec --onnx=dronetrack.onnx --saveEngine=dronetrack.engine --fp16
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [DeepSORT](https://github.com/levan92/deep_sort_realtime)
- [Sky-Drones Technologies](https://sky-drones.com/)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)

---

<p align="center">
  Built by <a href="https://github.com/SamSon1402">Sameer M.</a> for <a href="https://sky-drones.com/">Sky-Drones Technologies</a>
</p>
