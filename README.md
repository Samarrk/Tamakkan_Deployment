# Tamakkan — Real-Time Driver Assistance System

Tamakkan is a real-time driver assistance system designed to run on edge
hardware (NVIDIA Jetson Orin NX). A USB-connected dashcam feeds video to
the Jetson, which runs a multi-model perception pipeline and streams
alerts to a phone app over local Wi-Fi.

**Status:** Deployed and validated in a moving vehicle on Saudi roads.
Achieves 13.1 wall-FPS on Jetson Orin NX with TensorRT FP16 engines for
the three heavy models (YOLO, depth, lane). End-to-end demo flow —
sign-up, session start, real-time alerts, session save with summary —
successfully demonstrated.

## System Architecture

[USB camera]
↓
[Jetson Orin NX 16GB] — auto-starts on boot via systemd
├── FastAPI server (REST + WebSocket) ← server/app.py
├── TamakkanPipeline (orchestrator) ← src/tamakkan/pipeline.py
│ ├── Models (perception)
│ │ ├── TamakkanTracker — YOLOv11s + ByteTrack (TRT FP16)
│ │ ├── DepthEstimator — Depth Anything V2 ViT-S (TRT FP16)
│ │ ├── LaneDetector — UFLD-v2 CULane ResNet-18 (TRT FP16)
│ │ ├── LightClassifier — HSV traffic-light color
│ │ └── SpeedSignOCR — EasyOCR with per-track caching (off by default)
│ └── Detectors (mistakes + state)
│ ├── RedLightDetector — red light ahead / ran
│ ├── LaneViolationDetector — sustained lane drift
│ ├── TailgatingDetector — close + class-aware geometry
│ ├── NearMissDetector — close + closing fast (VRU-aware)
│ └── SpeedLimitReader — live speed-limit state
├── AlertEngine (priority + cooldown) ← src/tamakkan/alert_engine.py
└── SessionState (events + score) ← src/tamakkan/session_state.py
↓ (WebSocket over local Wi-Fi)
[Phone app — React Native / Expo]
├── Real-time voice alerts (TTS, English)
├── Live session view (HUD, live speed limit)
├── Post-session summary (score + mistake list)
└── Sync to team database (independent of Jetson)

## Hardware

| Component    | Spec                                             |
| ------------ | ------------------------------------------------ |
| Compute      | Seeed reComputer J4012 — Jetson Orin NX 16GB     |
| OS           | JetPack 5.1.3 / L4T R35.5 / Ubuntu 20.04         |
| Compute libs | CUDA 11.4, cuDNN 8.6, TensorRT 8.5.2             |
| Camera       | Arducam 1080P Low Light (USB UVC)                |
| USB Wi-Fi    | TP-Link Archer T2U Plus (RTL8821AU, DKMS driver) |
| Demo router  | 4G LTE router with SIM (self-contained internet) |
| In-car power | Jeep cabin 230V/150W AC outlet                   |

## Project Layout

src/tamakkan/
events.py wire-format dataclasses (Alert, SessionEvent, …)
session_state.py per-session bookkeeping + score
alert_engine.py priority + cooldown for spoken alerts
pipeline.py per-frame orchestrator
models/ one file per perception model
detectors/ one file per mistake/state detector
server/app.py FastAPI + WebSocket wrapper around the pipeline
scripts/ test client, diagnostic probes, ONNX export scripts
third_party/ vendored Depth Anything V2 and UFLD-v2
weights/ gitignored — download separately (see below)
deploy/ systemd unit + install/operations README
BACKEND_SPEC.md authoritative wire contract for phone ↔ Jetson

## Status

| Module                                   | Status                             |
| ---------------------------------------- | ---------------------------------- |
| TamakkanTracker (YOLO + ByteTrack)       | ✅ TRT FP16 engine                 |
| DepthEstimator (DAv2 ViT-S)              | ✅ TRT FP16 engine                 |
| LaneDetector (UFLD-v2)                   | ✅ TRT FP16 engine                 |
| LightClassifier (HSV)                    | ✅ done                            |
| SpeedSignOCR (EasyOCR + per-track cache) | ✅ done (off by default)           |
| RedLightDetector                         | ✅ done                            |
| LaneViolationDetector                    | ✅ verified in-car                 |
| TailgatingDetector                       | ✅ verified in-car                 |
| NearMissDetector                         | ⚠️ tuned, positive case unverified |
| SpeedLimitReader                         | ✅ done                            |
| AlertEngine                              | ✅ done                            |
| SessionState + scoring                   | ✅ done                            |
| TamakkanPipeline orchestrator            | ✅ done                            |
| FastAPI + WebSocket server               | ✅ done                            |
| PC ↔ phone integration                   | ✅ validated                       |
| Jetson deployment                        | ✅ done                            |
| TensorRT FP16 optimization               | ✅ done (7.3 → 13.1 FPS)           |
| CUDA thread-safe inference               | ✅ primary-context fix             |
| systemd auto-start                       | ✅ done                            |
| In-car demo                              | ✅ validated                       |
| APK build                                | ⏳ planned                         |

## Running on PC (development)

### Prerequisites

- Python 3.10+ with CUDA-capable PyTorch
- Weights placed in `weights/`:
  - `best.pt` (YOLOv11s, custom-trained on Saudi road classes)
  - `bytetrack_tamakkan.yaml` (ByteTrack config — already in repo)
  - `depth_anything_v2_vits.pth` (DAv2 ViT-S)
  - `culane_res18_v2.pth` (UFLD-v2 CULane ResNet-18)
- `pip install -r requirements.txt`

The PC path uses PyTorch weights (`.pth`). The Jetson path uses TensorRT
engines (`.engine`). Each model wrapper auto-detects which to load by
file extension.

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
the final `SessionSummary`.

### Run with a live webcam

```bash
python -m server.app --source 0
```

`--source 0` is camera index 0. Use `--source 1` etc. for other USB
cameras.

### Tuning knobs

```bash
python -m server.app --source 0 \
  --depth-every-n 8 \
  --lanes-every-n 5 \
  --ocr-frame-skip 999 \
  --max-fps 30
```

These are the production defaults baked into both the `_CONFIG` dict and
the argparse defaults. OCR is effectively disabled (`--ocr-frame-skip 999`)
because EasyOCR's CRAFT text detector was costing ~5 wall-FPS regardless
of caching, and speed-limit reading is informational rather than safety
critical. To re-enable for debugging, lower the value (try 30 for once
per second at 30fps).

## Deploying on Jetson

The Jetson runs as a systemd service that auto-starts on boot. Full
installation and operations docs are in [`deploy/README.md`](deploy/README.md).

### Quick install (one-time)

```bash
sudo cp deploy/tamakkan.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tamakkan
sudo systemctl start tamakkan
```

After boot, the pipeline loads in ~6.4 seconds and the server is
listening on `0.0.0.0:8000`.

### Live operations

```bash
sudo systemctl status tamakkan     # is it running?
sudo systemctl restart tamakkan    # manual restart
journalctl -u tamakkan -f          # follow logs
```

### Health check from another host

```bash
curl http://<jetson-ip>:8000/health
# expect: {"status":"ok","pipeline_loaded":true,"active_session_id":null}
```

### Building TensorRT engines

The repo includes ONNX export scripts for the three heavy models:

```bash
# DepthAnythingV2 ViT-S → ONNX (1×3×518×518, opset 16)
python scripts/export_depth_onnx.py

# UFLD-v2 → ONNX (1×3×320×1600, opset 16, dict→tuple wrapper)
python scripts/export_ufld_onnx.py

# Then build TRT engines with trtexec
trtexec --onnx=weights/depth_anything_v2_vits.onnx \
        --saveEngine=weights/depth_anything_v2_vits.engine --fp16

trtexec --onnx=weights/culane_res18_v2.onnx \
        --saveEngine=weights/culane_res18_v2.engine --fp16 --workspace=2048
```

YOLO is exported in one line via Ultralytics:

```python
from ultralytics import YOLO
YOLO("weights/best.pt").export(format="engine", half=True, imgsz=1280, workspace=4)
```

## Performance

| Stage                              | Wall-FPS | Δ    |
| ---------------------------------- | -------- | ---- |
| Baseline (PyTorch FP16 everywhere) | 7.3      | —    |
| + YOLO TRT engine                  | 7.3      | 0    |
| + Cadence + OCR threshold tuning   | 7.3      | 0    |
| + Depth TRT engine                 | 7.6      | +0.3 |
| + OCR disabled by default          | 12.6     | +5.0 |
| + UFLD-v2 TRT engine               | **13.1** | +0.5 |

Total: 7.3 → 13.1 wall-FPS (+79%) on real Saudi dashcam footage at 1080p.

The biggest win wasn't TensorRT — it was profiling. The three TRT
conversions combined gave +0.8 FPS. Disabling EasyOCR (informational only)
gave +5.0 FPS. The lesson: profile before optimizing.

## Diagnostic scripts

| Script                            | Purpose                                  |
| --------------------------------- | ---------------------------------------- |
| `scripts/test_server_client.py`   | phone-app stand-in over WS               |
| `scripts/diagnose_speed_limit.py` | trace why a clip's speed sign isn't read |
| `scripts/probe_depth.py`          | per-frame depth scale + direction        |
| `scripts/probe_tailgating.py`     | per-frame lead-vehicle reld/box_h        |
| `scripts/test_models_video.py`    | all 5 models on one clip                 |
| `scripts/test_detectors_video.py` | all 4 detectors on one clip              |
| `scripts/export_depth_onnx.py`    | DAv2 ViT-S → ONNX for TRT build          |
| `scripts/export_ufld_onnx.py`     | UFLD-v2 → ONNX for TRT build             |

Each model and detector also has a standalone `__main__` smoke test —
e.g. `python -m tamakkan.events`, `python -m tamakkan.session_state`,
`python -m tamakkan.models.depth_model /path/to/image.jpg weights/...engine`.

## Frontend (phone app)

The React Native / Expo app lives in a separate repo. Its wire contract
with the backend is documented in [`BACKEND_SPEC.md`](BACKEND_SPEC.md):

- `POST /sessions/start` → `{session_id}`
- `WS  /ws/session/{id}` → streams `alert`, `speed_limit`, `status` messages
- `POST /sessions/{id}/stop` → `SessionSummary` (score + events + metadata)

The app has a Settings screen where users can configure the Jetson's IP
address, allowing the system to work on different networks (home Wi-Fi,
4G router in the car, etc.) without code changes.

## In-car demo setup

For mobile demos away from any existing Wi-Fi, the system uses a
self-contained network:
[Jeep cabin 230V outlet]
├── Jetson power brick (~30W)
└── 4G LTE router with SIM (~5W)
[4G Router (Teddybear)]
├── Jetson connects via USB Wi-Fi (wlan0 = 192.168.8.199)
└── Phone connects via Wi-Fi (also gets internet for DB sync)

Both Jetson and phone connect to the router. The router provides
internet to the phone via cellular for database sync, and a local LAN
for the phone-to-Jetson WebSocket. Self-contained — works anywhere with
cell coverage.

The Jetson auto-connects to the router on boot via `nmcli` autoconnect
priority. Total cold-start time (power-on to ready): ~30-40 seconds.

## Critical bug fix — CUDA thread safety

The TRT inference backends use the **CUDA primary context** with explicit
push/pop per `infer()` call. This is required for thread safety under
FastAPI/uvicorn, which dispatches the frame loop to a worker thread
distinct from the one that initialized CUDA. The original `pycuda.autoinit`
pattern silently failed with `invalid resource handle` errors under the
server — bench tests passed, production failed. The primary-context
pattern allows TRT engines to be called from any thread.

This bug surfaced only during phone integration testing. Single-threaded
CLI smoke tests had been clean for days before that. Captured in commit
`988efc0`.

## Honest limitations

Documented in detail in each detector's module docstring. Highlights:

- **Monocular depth is relative, not metric.** Tailgating and near-miss
  alerts use relative-only thresholds.
- **No IMU, no driver-facing camera, no vehicle-speed source.** Therefore
  no harsh-braking, harsh-acceleration, speeding, drowsiness, or phone-use
  detection. The system intentionally only reports what its sensors
  support.
- **Lane model is the least reliable component on Saudi road markings.**
  Safety-critical detectors (tailgating, near-miss) deliberately do not
  depend on it.
- **Traffic light detection didn't fire during the in-car test** despite
  bench validation. Likely a model-domain issue (lighting, distance,
  training data). Worth investigating with more in-car footage.
- **Near-miss positive case is detector-tuned to suppress noise**; a
  controlled live-camera test on Jetson is still required to confirm it
  fires on a genuine rapid approach.
- **EasyOCR speed-limit reading is disabled by default** for performance.
  Re-enable path documented in §4 of [`BACKEND_SPEC.md`](BACKEND_SPEC.md).

## Commit history

The production engineering work is captured in seven well-documented
commits:

| Hash      | Description                                    |
| --------- | ---------------------------------------------- |
| `0aba083` | YOLO TensorRT engine wiring                    |
| `ce11642` | Cadence + OCR threshold tuning                 |
| `682ca47` | Depth Anything V2 TensorRT engine wiring       |
| `5b6524c` | OCR disabled by default for real-time FPS      |
| `7a936aa` | UFLD-v2 TensorRT engine wiring                 |
| `f5858a1` | systemd auto-start + argparse defaults fix     |
| `988efc0` | CUDA primary context fix for FastAPI threading |

Each commit message documents the bug, the fix, the verification, and
the trade-offs. They are intended to read as a self-contained log of
the engineering work.

## Authors

Samar — Graduation project, 2026.

## Notes

Development history of earlier iterations lives in a separate repo.
Architectural decisions and trade-offs are documented inline in each
module's docstring.
