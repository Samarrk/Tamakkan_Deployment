"""
src/tamakkan/pipeline.py

The Tamakkan perception pipeline. One per session.

Role
----
Glue. Takes BGR frames in, returns per-frame results out. Owns:

  - the 5 models (tracker, depth, lanes, light, OCR via reader)
  - the 4 detectors
  - the alert engine
  - the session state
  - the FRAME CADENCE POLICY (which heavy models run every Nth frame,
    with last-result reuse on the in-between frames)

Frame cadence — the central performance decision
------------------------------------------------
On Jetson Orin NX, naively running every model every frame produces
roughly 5-7 FPS even with PyTorch FP16. That's too slow for safety
alerts (a car at 60 km/h covers 3.3 m in 200 ms). Mitigation:

  - tracker: EVERY frame. ByteTrack track continuity requires it; skip
    and IDs are lost. Non-negotiable.
  - depth: every Nth frame (default 5). Heaviest model after YOLO. A
    car ~5 m ahead does not teleport, and depth is consumed relatively
    over a rolling window in near_miss / used as a relative magnitude
    in tailgating — neither needs single-frame freshness.
  - lanes: every Nth frame (default 3). Lane lines move slowly; the
    UFLD-v2 wrapper already smooths over 5 internally.
  - light classifier: every frame; it's HSV inside red_light_detector,
    effectively free, no skip control here.
  - OCR / speed-limit reader: own frame_skip + track-id cache, set on
    the reader. Configured via ocr_frame_skip.

Detectors and the alert engine always run every frame. They are cheap;
the cost is in the models. On a skipped depth/lanes frame, the pipeline
passes the LAST cached depth map / lane list to the detectors that
need them. Detectors do not know or care whether they got a fresh
result or a reused one — same shape, same semantics.

Defaults are CONSERVATIVE on depth (5) because the user identified it
as the likely Jetson bottleneck. These are tuning knobs — measure real
FPS on Jetson, then re-tune.

What this file does NOT do
--------------------------
- I/O. No camera open, no WebSocket send, no file write. The pipeline
  takes a frame in and returns a result out. The FastAPI server in
  server/app.py wraps it with the network and lifecycle. Keeping I/O
  out means the same pipeline runs in test scripts, in the server, and
  later under any other harness (a phone push-style protocol, batch
  evaluation, whatever) without rewriting.

- Session bookkeeping math (score, summary). That's SessionState's job.
  The pipeline records events into state; the server calls
  state.to_summary() at the end.

Threading
---------
Same posture as the rest: one pipeline instance per session, single-
threaded per session. The server may run multiple sessions
concurrently in separate threads/processes, each with its own pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# Make vendored third-party packages importable.
# (depth_anything_v2, ufld_v2 live under <repo>/third_party/.)
import sys as _sys
from pathlib import Path as _Path
_repo = _Path(__file__).resolve().parents[2]
_tp = _repo / "third_party"
if _tp.is_dir() and str(_tp) not in _sys.path:
    _sys.path.insert(0, str(_tp))

from tamakkan.events import Alert, SessionEvent, SpeedLimitChange
from tamakkan.alert_engine import AlertEngine
from tamakkan.session_state import SessionState

from tamakkan.models.tracker import TamakkanTracker, Track
from tamakkan.models.depth_model import DepthEstimator
from tamakkan.models.lane_model import LaneDetector, Lane
from tamakkan.models.ocr_model import SpeedSignOCR

from tamakkan.detectors.red_light_detector import RedLightDetector
from tamakkan.detectors.lane_violation_detector import LaneViolationDetector
from tamakkan.detectors.tailgating_detector import TailgatingDetector
from tamakkan.detectors.near_miss_detector import NearMissDetector
from tamakkan.detectors.speed_limit_reader import SpeedLimitReader


# ── Cadence defaults (conservative on depth — see module docstring) ───────────
DEFAULT_DEPTH_EVERY_N = 8    # depth inference every Nth frame
DEFAULT_LANES_EVERY_N = 5    # lane inference every Nth frame
DEFAULT_OCR_FRAME_SKIP = 999   # speed-limit reader skips frames internally


# ── Per-frame result ──────────────────────────────────────────────────────────
@dataclass
class PipelineFrameResult:
    """
    What process_frame() returns for one frame.

    Designed for the FastAPI server to consume directly:
      - If alert is not None, send it as the kind=alert WebSocket message.
      - If speed_limit_change is not None, send it as kind=speed_limit.
      - frame_idx / frame_seconds are diagnostic.
      - tracks / lanes / depth_map are exposed for OPTIONAL test-harness
        visualization. The server doesn't read them.
    """
    frame_idx:          int
    frame_seconds:      float
    alert:              Optional[Alert]              = None
    speed_limit_change: Optional[SpeedLimitChange]   = None
    session_events:     List[SessionEvent]           = field(default_factory=list)

    # Diagnostic / for visualization only — not part of the wire contract.
    tracks:    List[Track] = field(default_factory=list)
    lanes:     List[Lane]  = field(default_factory=list)
    depth_map: Optional[np.ndarray] = None
    # True if depth was freshly computed this frame; False if reused.
    depth_was_fresh: bool = False
    lanes_was_fresh: bool = False


# ── Pipeline ──────────────────────────────────────────────────────────────────
class TamakkanPipeline:
    """
    Construct once per session. Call process_frame(frame) per frame.
    Call end_session() once to seal state; to_summary() any time.

    Args:
        yolo_weights: path to YOLOv11s best.pt
        bytetrack_config: path to bytetrack_tamakkan.yaml
        depth_weights: path to depth_anything_v2_vits.pth
        lane_weights: path to culane_res18_v2.pth
        device: 'cuda:0' / 'cpu' / None (auto). Same value used by all
            three GPU models. None falls back to CPU silently — for
            the Jetson you want 'cuda:0' explicitly so the pipeline
            crashes loud if CUDA isn't visible.
        fps: input stream FPS, passed to detectors that have frame-
            based cooldowns. If None, falls back to detector defaults
            (30). For a USB camera that doesn't expose a real FPS,
            measure it yourself first and pass it.
        session: pre-constructed SessionState. Pass one from the
            server (which manages session ids/lifecycle); construct
            internally for standalone test scripts.
        depth_every_n / lanes_every_n: cadence knobs (see docstring).
        ocr_frame_skip: passed to SpeedLimitReader.
        min_alert_gap_seconds: passed to AlertEngine.
    """

    def __init__(
        self,
        yolo_weights:          str,
        bytetrack_config:      str,
        depth_weights:         str,
        lane_weights:          str,
        device:                Optional[str]   = None,
        fps:                   Optional[float] = None,
        session:               Optional[SessionState] = None,
        depth_every_n:         int   = DEFAULT_DEPTH_EVERY_N,
        lanes_every_n:         int   = DEFAULT_LANES_EVERY_N,
        ocr_frame_skip:        int   = DEFAULT_OCR_FRAME_SKIP,
        min_alert_gap_seconds: float = 4.0,
    ):
        # ── Models ────────────────────────────────────────────────────────────
        # Construct all five up front. On Jetson the cold-start cost
        # (model load + CUDA warmup) is several seconds; we eat it once
        # at session start, never again.
        self.tracker = TamakkanTracker(
            weights         = yolo_weights,
            tracker_config  = bytetrack_config,
            device          = device,
        )
        self.depth = DepthEstimator(
            weights_path = depth_weights,
            variant      = "vits",
            device       = device,
        )
        self.lane_detector = LaneDetector(
            weights_path = lane_weights,
            device       = device,
        )
        self.ocr = SpeedSignOCR(device=device)

        # ── Detectors ─────────────────────────────────────────────────────────
        self.red_light_det      = RedLightDetector(fps=fps or 30.0)
        self.lane_violation_det = LaneViolationDetector(fps=fps or 30.0)
        self.tailgating_det     = TailgatingDetector(fps=fps or 30.0)
        self.near_miss_det      = NearMissDetector(fps=fps or 30.0)
        self.speed_reader       = SpeedLimitReader(
            ocr        = self.ocr,
            frame_skip = ocr_frame_skip,
        )

        # ── State + alerts ────────────────────────────────────────────────────
        self.session = session if session is not None else SessionState()
        self.alert_engine = AlertEngine(min_gap_seconds=min_alert_gap_seconds)

        # ── Cadence state ─────────────────────────────────────────────────────
        self.depth_every_n = max(1, depth_every_n)
        self.lanes_every_n = max(1, lanes_every_n)

        self.frame_idx: int = 0
        self._last_depth: Optional[np.ndarray] = None
        self._last_lanes: List[Lane] = []

    # ── Public API ────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> PipelineFrameResult:
        """
        Run the full pipeline on one BGR frame. Returns a
        PipelineFrameResult describing what (if anything) the phone
        should be told.

        Side effects: events recorded into self.session; FPS ticked.
        """
        if frame is None or frame.size == 0:
            raise ValueError("process_frame received empty frame")

        t_start = time.time()
        self.frame_idx += 1

        # ── 1. Tracker (always) ───────────────────────────────────────────────
        tracks = self.tracker.update(frame)

        # ── 2. Depth (every Nth frame, otherwise reuse last) ──────────────────
        depth_fresh = False
        if (self.frame_idx % self.depth_every_n == 0) or self._last_depth is None:
            # Also force a fresh inference if we have nothing cached yet,
            # so the very first frame produces usable depth instead of None.
            self._last_depth = self.depth.predict(frame)
            depth_fresh = True
        depth_map = self._last_depth

        # ── 3. Lanes (every Nth frame, otherwise reuse last) ──────────────────
        lanes_fresh = False
        if (self.frame_idx % self.lanes_every_n == 0) or not self._last_lanes:
            # On the very first frames we don't have cached lanes yet —
            # also run if cache is empty so detectors see something.
            self._last_lanes = self.lane_detector.update(frame)
            lanes_fresh = True
        lanes = self._last_lanes

        # ── 4. Detectors (always — they're cheap) ────────────────────────────
        # red_light returns a LIST (it can fire AHEAD + RAN same frame).
        # The other three return Optional[event]. We flatten the list,
        # plus the three optionals (filtered None), into one events list
        # for the alert engine.
        red_light_events: List = self.red_light_det.update(tracks, frame)
        lane_event   = self.lane_violation_det.update(lanes, frame)
        tail_event   = self.tailgating_det.update(tracks, depth_map, frame)
        near_event   = self.near_miss_det.update(tracks, depth_map, frame)

        detector_events = []
        detector_events.extend(red_light_events)
        for ev in (lane_event, tail_event, near_event):
            if ev is not None:
                detector_events.append(ev)

        # ── 5. Speed-limit reader (separate channel, not a mistake) ──────────
        speed_change = self.speed_reader.update(tracks, frame)
        if speed_change is not None:
            # Update SessionState; dedup is handled there too (no-op if
            # the value hasn't actually changed since last set).
            self.session.set_speed_limit(speed_change.limit_kmh)

        # ── 6. Alert engine: pick at most one alert, canonicalize all ────────
        session_time_s = self.session.session_time_s()
        engine_out = self.alert_engine.process(
            detector_events,
            session_time_s = session_time_s,
        )

        # Record every canonicalized session event (engine never
        # suppresses recording, only spoken alerts).
        for sev in engine_out.session_events:
            self.session.record_event(sev)

        # ── 7. Per-frame bookkeeping ──────────────────────────────────────────
        frame_seconds = time.time() - t_start
        self.session.tick_fps(frame_seconds)

        return PipelineFrameResult(
            frame_idx          = self.frame_idx,
            frame_seconds      = frame_seconds,
            alert              = engine_out.alert,
            speed_limit_change = speed_change,
            session_events     = engine_out.session_events,
            tracks             = tracks,
            lanes              = lanes,
            depth_map          = depth_map,
            depth_was_fresh    = depth_fresh,
            lanes_was_fresh    = lanes_fresh,
        )

    def end_session(self) -> None:
        """
        Mark the session as ended in SessionState. Idempotent.
        After this, to_summary() returns a frozen summary.
        """
        self.session.end()

    def to_summary(self):
        """Convenience accessor — same as self.session.to_summary()."""
        return self.session.to_summary()

    def reset(self) -> None:
        """
        Clear cross-frame state so this pipeline can be reused for a
        fresh session. The session itself is NOT replaced here — the
        server is responsible for handing the pipeline a new
        SessionState when it starts a new session.
        """
        self.frame_idx = 0
        self._last_depth = None
        self._last_lanes = []
        self.tracker.reset()
        self.lane_detector.reset()
        self.ocr.reset()
        self.red_light_det.reset()
        self.lane_violation_det.reset()
        self.tailgating_det.reset()
        self.near_miss_det.reset()
        self.speed_reader.reset()
        self.alert_engine.reset()


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Loads the pipeline against real weights and runs it over one video.
    This is the FIRST integration test that exercises every model and
    detector together. Run on PC first; on Jetson later for FPS
    measurement.

    Usage:
      python -m tamakkan.pipeline <video_path>

    Assumes weights live at <repo>/weights/{best.pt, bytetrack_tamakkan.yaml,
    depth_anything_v2_vits.pth, culane_res18_v2.pth}. Adjust if your
    layout differs.
    """
    import sys
    import cv2

    if len(sys.argv) < 2:
        print("usage: python -m tamakkan.pipeline <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    # Resolve repo root assuming this file is at src/tamakkan/pipeline.py
    repo = Path(__file__).resolve().parents[2]
    weights_dir = repo / "weights"

    yolo_w   = str(weights_dir / "best.engine") if (weights_dir / "best.engine").exists() else str(weights_dir / "best.pt")
    bt_cfg   = str(weights_dir / "bytetrack_tamakkan.yaml")
    depth_w  = str(weights_dir / "depth_anything_v2_vits.engine") if (weights_dir / "depth_anything_v2_vits.engine").exists() else str(weights_dir / "depth_anything_v2_vits.pth")
    lane_w   = str(weights_dir / "culane_res18_v2.engine") if (weights_dir / "culane_res18_v2.engine").exists() else str(weights_dir / "culane_res18_v2.pth")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"could not open video: {video_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"video: {video_path}")
    print(f"fps:   {fps:.1f}  frames: {total}")
    print(f"cadence: depth every {DEFAULT_DEPTH_EVERY_N}, "
          f"lanes every {DEFAULT_LANES_EVERY_N}, "
          f"ocr skip {DEFAULT_OCR_FRAME_SKIP}")

    pipeline = TamakkanPipeline(
        yolo_weights     = yolo_w,
        bytetrack_config = bt_cfg,
        depth_weights    = depth_w,
        lane_weights     = lane_w,
        fps              = fps,
    )

    alerts_spoken = 0
    events_recorded = 0
    speed_changes = 0
    n = 0

    t_wall_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        result = pipeline.process_frame(frame)

        events_recorded += len(result.session_events)
        if result.alert is not None:
            alerts_spoken += 1
            sub = ("/" + result.alert.subtype.value) if result.alert.subtype else ""
            print(f"  frame {n:>5}  ALERT  "
                  f"{result.alert.event_type.value}{sub}  "
                  f"sev={result.alert.severity.value}  "
                  f"vru={result.alert.is_vru}  "
                  f"\"{result.alert.message_en}\"")
        if result.speed_limit_change is not None:
            speed_changes += 1
            print(f"  frame {n:>5}  SPEED  "
                  f"limit -> {result.speed_limit_change.limit_kmh} km/h")

    cap.release()
    wall = time.time() - t_wall_start

    pipeline.end_session()
    summary = pipeline.to_summary()

    print()
    print(f"--- run complete ---")
    print(f"  frames processed:       {n}")
    print(f"  wall seconds:           {wall:.1f}")
    print(f"  effective pipeline FPS: {n / wall:.1f}")
    print(f"  alerts spoken:          {alerts_spoken}")
    print(f"  events recorded:        {events_recorded}")
    print(f"  speed-limit changes:    {speed_changes}")
    print()
    print("--- session summary ---")
    import json
    print(json.dumps(summary.to_dict(), indent=2))