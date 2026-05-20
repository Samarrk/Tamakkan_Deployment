# Tamakkan — Real-Time Driver Assistance System

Tamakkan is a real-time driver assistance system designed to run on edge
hardware (NVIDIA Jetson Orin NX). A USB-connected dashcam feeds video to
the Jetson, which runs a multi-model perception pipeline and streams
alerts to a phone app over local Wi-Fi.

## System Architecture

```
[USB camera]
      ↓
[Jetson Orin NX 16GB]
  ├── FastAPI server (REST + WebSocket)        ← server/app.py
  ├── TamakkanPipeline (orchestrator)          ← src/tamakkan/pipeline.py
  │     ├── Models (perception)
  │     │     ├── TamakkanTracker — YOLOv11s + ByteTrack
  │     │     ├── DepthEstimator — Depth Anything V2 Small
  │     │     ├── LaneDetector — UFLD-v2 CULane ResNet-18
  │     │     ├── LightClassifier — HSV traffic-light color
  │     │     └── SpeedSignOCR — EasyOCR with per-track caching
  │     └── Detectors (mistakes + state)
  │           ├── RedLightDetector — red light ahead / ran
  │           ├── LaneViolationDetector — sustained lane drift
  │           ├── TailgatingDetector — close + class-aware geometry
  │           ├── NearMissDetector — close + closing fast (VRU-aware)
  │           └── SpeedLimitReader — live speed-limit state
  ├── AlertEngine (priority + cooldown)        ← src/tamakkan/alert_engine.py
  └── SessionState (events + score)            ← src/tamakkan/session_state.py
      ↓ (WebSocket over local Wi-Fi)
[Phone app — React Native / Expo]
  ├── Real-time voice alerts (TTS, English)
  ├── Live session view (HUD, live speed limit)
  └── Post-session summary (score + mistake list)
```

## Hardware

| Component    | Spec                                         |
| ------------ | -------------------------------------------- |
| Compute      | Seeed reComputer J4012 — Jetson Orin NX 16GB |
| OS           | JetPack 5.1.3 / L4T R35.5 / Ubuntu 20.04     |
| Compute libs | CUDA 11.4, cuDNN 8.6, TensorRT 8.5.2         |
| Camera       | Arducam 1080P Low Light (USB UVC)            |

## Project Layout

```
src/tamakkan/
  events.py              wire-format dataclasses (Alert, SessionEvent, …)
  session_state.py       per-session bookkeeping + score
  alert_engine.py        priority + cooldown for spoken alerts
  pipeline.py            per-frame orchestrator
  models/                one file per perception model
  detectors/             one file per mistake/state detector
server/app.py            FastAPI + WebSocket wrapper around the pipeline
scripts/                 test client, diagnostic probes, smoke tests
third_party/             vendored Depth Anything V2 and UFLD-v2
weights/                 gitignored — download separately (see below)
BACKEND_SPEC.md          authoritative wire contract for phone ↔ Jetson
```

## Status

End-to-end backend complete and validated on PC against the phone app
over local Wi-Fi using recorded Saudi road footage. Real-time alerts,
TTS, and live speed-limit updates all working. Next phase is on-Jetson
deployment + TensorRT FP16 optimization.

| Module                                   | Status         |
| ---------------------------------------- | -------------- |
| TamakkanTracker (YOLO + ByteTrack)       | ✅ done        |
| DepthEstimator (DAv2 ViT-S)              | ✅ done        |
| LaneDetector (UFLD-v2)                   | ✅ done        |
| LightClassifier (HSV)                    | ✅ done        |
| SpeedSignOCR (EasyOCR + per-track cache) | ✅ done        |
| RedLightDetector                         | ✅ done        |
| LaneViolationDetector                    | ✅ done        |
| TailgatingDetector                       | ✅ done        |
| NearMissDetector                         | ✅ done        |
| SpeedLimitReader                         | ✅ done        |
| AlertEngine                              | ✅ done        |
| SessionState + scoring                   | ✅ done        |
| TamakkanPipeline orchestrator            | ✅ done        |
| FastAPI + WebSocket server               | ✅ done        |
| PC ↔ phone integration                   | ✅ validated   |
| Jetson deployment                        | ⏳ in progress |
| TensorRT FP16 optimization               | ⏳ planned     |
| systemd auto-start                       | ⏳ planned     |

## Running on PC (development)

### Prerequisites

- Python 3.10+ with CUDA-capable PyTorch
- Weights placed in `weights/`:
  - `best.pt` (YOLOv11s, custom-trained on Saudi road classes)
  - `bytetrack_tamakkan.yaml` (ByteTrack config — already in repo)
  - `depth_anything_v2_vits.pth` (DAv2 ViT-S)
  - `culane_res18_v2.pth` (UFLD-v2 CULane ResNet-18)
- `pip install -r requirements.txt`

### Run the server against a recorded video

```bash
# from the repo root
python -m server.app --source path/to/clip.mp4
```

The server boots, loads all models, and listens on `0.0.0.0:8000`.

### Validate the wire contract without a phone

```bash
# in a second terminal
python scripts/test_server_client.py --host 127.0.0.1 --port 8000
```

This simulates the phone app: opens a WebSocket, calls `/sessions/start`,
prints every alert as it arrives, calls `/sessions/{id}/stop`, and prints
the final `SessionSummary`. Sanity-checked at the end.

### Run with a live webcam

```bash
python -m server.app --source 0
```

`--source 0` is camera index 0. Use `--source 1` etc. for other USB
cameras.

### Tuning knobs

```bash
python -m server.app --source 0 \
  --depth-every-n 5 \
  --lanes-every-n 3 \
  --ocr-frame-skip 1 \
  --max-fps 30
```

Defaults are conservative on depth (heaviest model) and aggressive on
OCR (cache makes it cheap). Tune on real Jetson FPS.

## Diagnostic scripts

| Script                            | Purpose                                  |
| --------------------------------- | ---------------------------------------- |
| `scripts/test_server_client.py`   | phone-app stand-in over WS               |
| `scripts/diagnose_speed_limit.py` | trace why a clip's speed sign isn't read |
| `scripts/probe_depth.py`          | per-frame depth scale + direction        |
| `scripts/probe_tailgating.py`     | per-frame lead-vehicle reld/box_h        |
| `scripts/test_models_video.py`    | all 5 models on one clip                 |
| `scripts/test_detectors_video.py` | all 4 detectors on one clip              |

Each model and detector also has a standalone `__main__` smoke test —
e.g. `python -m tamakkan.events`, `python -m tamakkan.session_state`.

## Frontend (phone app)

The React Native / Expo app lives in a separate repo. Its wire contract
with the backend is documented in [`BACKEND_SPEC.md`](BACKEND_SPEC.md):

- `POST /sessions/start` → `{session_id}`
- `WS  /ws/session/{id}` → streams `alert`, `speed_limit`, `status` messages
- `POST /sessions/{id}/stop` → `SessionSummary` (score + events + metadata)

## Honest limitations

Documented in detail in each detector's module docstring. Highlights:

- Monocular depth is relative, not metric. Tailgating and near-miss
  alerts use relative-only thresholds.
- No IMU, no driver-facing camera, no vehicle-speed source. Therefore
  no harsh-braking, harsh-acceleration, speeding, drowsiness, or phone-use
  detection. The system intentionally only reports what its sensors
  support.
- Lane model is the least reliable component on Saudi road markings.
  Safety-critical detectors (tailgating, near-miss) deliberately do
  not depend on it.
- Near-miss positive case is detector-tuned to suppress noise; a
  controlled live-camera test on Jetson is still required to confirm
  it fires on a genuine rapid approach.

## Authors

Samar — Graduation project, Umm Al-Qura University, 2026.

## Notes

Development history of earlier iterations lives in a separate repo
(link to be added). Architectural decisions and trade-offs are
documented inline in each module's docstring.
