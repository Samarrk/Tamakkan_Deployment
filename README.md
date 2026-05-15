# Tamakkan — Real-Time Driver Assistance System

Tamakkan is a real-time driver assistance system designed to run on edge
hardware (NVIDIA Jetson Orin NX). A USB-connected dashcam feeds video to
the Jetson, which runs a multi-model perception pipeline and streams
alerts to a phone app over local Wi-Fi.

## System Architecture

[USB camera]
↓
[Jetson Orin NX 16GB]
├── FastAPI server (REST + WebSocket)
├── TamakkanPipeline (orchestrator)
│ ├── TamakkanTracker — YOLOv11s + ByteTrack
│ ├── LightClassifier — HSV traffic-light color
│ ├── RedLightDetector — red-light running events
│ ├── LaneDetector — UFLD lane lines
│ ├── DepthEstimator — Depth Anything V2 Small (tailgating)
│ └── TamakkanOCR — EasyOCR speed-limit signs
└── AlertEngine (priority-aware dispatcher)
↓ (WebSocket over local Wi-Fi)
[Phone app]
├── Real-time voice alerts (TTS on phone)
├── Live session view
└── Post-session report + score

## Hardware

| Component    | Spec                                         |
| ------------ | -------------------------------------------- |
| Compute      | Seeed reComputer J4012 — Jetson Orin NX 16GB |
| OS           | JetPack 5.1.3 / L4T R35.5 / Ubuntu 20.04     |
| Compute libs | CUDA 11.4, cuDNN 8.6, TensorRT 8.5.2         |
| Camera       | Arducam 1080P Low Light (USB UVC)            |

## Project Layout

src/tamakkan/ perception modules (one file per model)
server/ FastAPI + WebSocket server
scripts/ ONNX export, benchmarks, helpers
tests/ unit tests
weights/ model weights (gitignored; download manually)
docs/ diagrams + design notes

## Status

🚧 Work in progress — code is being migrated module-by-module from
the development repo to this clean deployment repo.

| Module                             | Status               |
| ---------------------------------- | -------------------- |
| TamakkanTracker (YOLO + ByteTrack) | ⏳ pending migration |
| LightClassifier                    | ⏳ pending migration |
| RedLightDetector                   | ⏳ pending migration |
| OCR handler                        | ⏳ pending migration |
| DepthEstimator                     | ⏳ pending migration |
| LaneDetector                       | ⏳ pending migration |
| Pipeline orchestrator              | ⏳ not started       |
| AlertEngine                        | ⏳ not started       |
| FastAPI server                     | ⏳ not started       |
| TensorRT optimization              | ⏳ not started       |

## Authors

Samar — Graduation project, Umm al qura University, 2026.

## Notes

Development history of earlier iterations lives in a separate repo
(link to be added).
