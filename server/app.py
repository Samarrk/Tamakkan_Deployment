"""
server/app.py

FastAPI + WebSocket wrapper around TamakkanPipeline.

What this file does
-------------------
- Loads the pipeline ONCE at boot. Models stay warm; new sessions reset
  detector/tracker state but keep the loaded weights on GPU. Cold-start
  is ~10s on PC, longer on Jetson — we eat it once at server start, not
  per session.
- Owns the camera or video source. POST /sessions/start opens it; POST
  /sessions/{id}/stop closes it. Demo uses --source 0 for webcam or
  --source path/to/clip.mp4 for testing against a recorded drive.
- Runs the per-frame processing loop in a background asyncio task. The
  loop pushes Alerts, SpeedLimitChanges, and StatusMessages onto a queue
  that the WebSocket consumes.
- Enforces one active session at a time. Second /sessions/start while
  one is active returns 409 Conflict. Matches the physical reality:
  one car, one driver, one drive.

What this file does NOT do
--------------------------
- Auth. Demo uses no login; team DB handles users separately.
- Persistence. Jetson stays stateless across sessions (BACKEND_SPEC.md
  §1). All history is forwarded to the team DB by the phone after the
  drive ends.
- Phone-side anything. The server is REST + WebSocket; it doesn't know
  if the consumer is a phone, a test script, or curl.

CLI
---
  python -m server.app --source 0                  # webcam
  python -m server.app --source path/to/clip.mp4   # video file
  python -m server.app --source rtsp://...         # IP camera (untested)
  python -m server.app --host 0.0.0.0 --port 8000  # network config

Defaults bind to 0.0.0.0:8000 so phone-on-same-Wi-Fi (or Jetson hotspot)
can reach the server without firewall fiddling.

Threading model
---------------
- main thread: uvicorn event loop
- frame loop: an asyncio.Task running pipeline.process_frame in an
  executor (because OpenCV + torch are blocking).
- one WebSocket per session: pulls from session.alert_queue.

SessionState mutation happens only inside the frame loop's executor
call, and only one frame loop exists at a time (single-session
constraint), so no cross-thread races.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Make src/ and third_party/ importable ─────────────────────────────────────
# Server lives at <repo>/server/app.py; package code is at <repo>/src/tamakkan,
# vendored deps are at <repo>/third_party/.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[1]
for _p in (_REPO / "src", _REPO / "third_party"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tamakkan.pipeline import TamakkanPipeline                        # noqa: E402
from tamakkan.session_state import SessionState                       # noqa: E402
from tamakkan.events import (                                         # noqa: E402
    Alert,
    SpeedLimitChange,
    StatusMessage,
    SessionSummary,
)


# ── Runtime config (populated by argparse in __main__) ────────────────────────
# We stash these on the module so the lifespan can read them without a
# global mutable config object. They're set once at process start.
_CONFIG: Dict[str, Any] = {
    "source":           "0",        # camera index as str OR path
    "host":             "0.0.0.0",
    "port":             8000,
    "yolo_weights":     None,       # resolved from --weights-dir
    "bytetrack_config": None,
    "depth_weights":    None,
    "lane_weights":     None,
    "device":           None,       # None = auto
    "depth_every_n":    8,
    "lanes_every_n":    5,
    "ocr_frame_skip":   999,
    "max_fps":          None,       # if set, sleep between frames to cap
}


def _resolve_source(src_str: str) -> Any:
    """
    Camera index ('0', '1', ...) → int; everything else → str path/URL.
    OpenCV's VideoCapture takes either.
    """
    try:
        return int(src_str)
    except ValueError:
        return src_str


# ── Active session container ──────────────────────────────────────────────────
class ActiveSession:
    """
    Holds everything tied to one in-flight drive: the SessionState, the
    asyncio frame loop task, the VideoCapture, and a queue of pending
    WebSocket messages.

    One instance lives at module scope (_active_session) for the duration
    of the drive; cleared on stop. We don't keep finished sessions in
    memory beyond their summary — the team DB owns history.
    """

    def __init__(self, session_id: str, state: SessionState):
        self.session_id: str = session_id
        self.state: SessionState = state
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_task: Optional[asyncio.Task] = None
        # Bounded queue so a slow/disconnected WebSocket can't cause
        # unbounded memory growth. New messages are dropped when full;
        # losing an alert is far better than OOM'ing the Jetson.
        self.alert_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=64)
        self.ended: bool = False
        self.summary: Optional[SessionSummary] = None    # set on stop
        # Diagnostic
        self.frames_processed: int = 0
        self.frames_dropped: int = 0


# Module-scope state. One pipeline, at most one active session.
_pipeline: Optional[TamakkanPipeline] = None
_active_session: Optional[ActiveSession] = None
_finished_summaries: Dict[str, SessionSummary] = {}    # last few for /summary


# ── Lifespan: load the pipeline once at server boot ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan. Loads pipeline at startup, releases at shutdown.

    We load models even though no session is active yet — this is the
    "warm hardware before the demo starts" pattern. First request after
    boot is then ~instant instead of paying the cold-start penalty.
    """
    global _pipeline

    cfg = _CONFIG
    print("=" * 60)
    print("Tamakkan server starting")
    print(f"  source:           {cfg['source']}")
    print(f"  weights:")
    print(f"    yolo            {cfg['yolo_weights']}")
    print(f"    bytetrack       {cfg['bytetrack_config']}")
    print(f"    depth           {cfg['depth_weights']}")
    print(f"    lanes           {cfg['lane_weights']}")
    print(f"  device:           {cfg['device'] or 'auto'}")
    print(f"  cadence:")
    print(f"    depth every     {cfg['depth_every_n']}")
    print(f"    lanes every     {cfg['lanes_every_n']}")
    print(f"    ocr skip        {cfg['ocr_frame_skip']}")
    print("=" * 60)

    t0 = time.time()
    _pipeline = TamakkanPipeline(
        yolo_weights     = cfg["yolo_weights"],
        bytetrack_config = cfg["bytetrack_config"],
        depth_weights    = cfg["depth_weights"],
        lane_weights     = cfg["lane_weights"],
        device           = cfg["device"],
        depth_every_n    = cfg["depth_every_n"],
        lanes_every_n    = cfg["lanes_every_n"],
        ocr_frame_skip   = cfg["ocr_frame_skip"],
    )
    cold = time.time() - t0
    print(f"pipeline loaded in {cold:.1f}s — ready to accept sessions")

    yield

    # Shutdown: stop any active session cleanly.
    if _active_session is not None and not _active_session.ended:
        print("shutdown: ending active session")
        await _shutdown_active_session()


app = FastAPI(title="Tamakkan Backend", lifespan=lifespan)

# CORS open for the demo: phone may hit us from any origin on the LAN /
# Jetson hotspot. Tighten if/when the app is published.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────
class StartSessionRequest(BaseModel):
    device_id: Optional[str] = None    # accepted for spec parity; not used


class StartSessionResponse(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    active_session_id: Optional[str]


# ── REST endpoints ────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status            = "ok",
        pipeline_loaded   = _pipeline is not None,
        active_session_id = _active_session.session_id if _active_session else None,
    )


@app.post("/sessions/start", response_model=StartSessionResponse)
async def start_session(_req: StartSessionRequest) -> StartSessionResponse:
    """
    Open the camera/video source, start the per-frame loop, return a
    session id. The phone then opens a WebSocket to
    /ws/session/{session_id} to receive alerts.

    One-session-at-a-time. Concurrent /start calls get 409.
    """
    global _active_session

    if _pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not loaded yet")

    if _active_session is not None and not _active_session.ended:
        raise HTTPException(
            status_code=409,
            detail=f"a session is already active: {_active_session.session_id}",
        )

    # Build a fresh SessionState and reset the pipeline so detectors,
    # cooldowns, and the alert engine start clean. Models stay loaded.
    state = SessionState()
    _pipeline.session = state
    _pipeline.reset()

    # Open the source. Camera index or file path — same call.
    src = _resolve_source(_CONFIG["source"])
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise HTTPException(
            status_code=500,
            detail=f"could not open source: {_CONFIG['source']}",
        )

    sess = ActiveSession(session_id=state.session_id, state=state)
    sess.cap = cap

    # Kick off the frame loop. It will push messages to sess.alert_queue
    # and call pipeline.process_frame until the source ends or the
    # session is stopped.
    sess.frame_task = asyncio.create_task(_frame_loop(sess))

    _active_session = sess
    print(f"session started: {sess.session_id}  source={_CONFIG['source']}")
    return StartSessionResponse(session_id=sess.session_id)


@app.post("/sessions/{session_id}/stop")
async def stop_session(session_id: str):
    """
    Stop the active session and return its SessionSummary.

    Idempotent-ish: after a session ends, calling stop again returns
    the cached summary. Calling stop on an unknown id 404s.
    """
    # First: was this the active session? Stop it cleanly.
    global _active_session

    if _active_session is not None and _active_session.session_id == session_id:
        summary = await _shutdown_active_session()
        return summary.to_dict()

    # Otherwise: maybe it's a finished one we still have cached.
    if session_id in _finished_summaries:
        return _finished_summaries[session_id].to_dict()

    raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")


@app.get("/sessions/{session_id}/summary")
async def get_summary(session_id: str):
    """
    Idempotent summary read. Works whether the session is active (returns
    summary-so-far) or finished.
    """
    if _active_session is not None and _active_session.session_id == session_id:
        if _active_session.ended and _active_session.summary is not None:
            return _active_session.summary.to_dict()
        return _active_session.state.to_summary().to_dict()

    if session_id in _finished_summaries:
        return _finished_summaries[session_id].to_dict()

    raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/session/{session_id}")
async def session_ws(ws: WebSocket, session_id: str):
    """
    Subscribe to the live alert stream for one session. The server
    pushes messages as they arrive in the session's queue:
      - kind=status, state=active   (sent immediately on connect)
      - kind=alert                  (when detectors fire and pass the engine)
      - kind=speed_limit            (when OCR confirms a new limit)
      - kind=status, state=ended    (sent when the session stops)

    Disconnect does NOT end the session — pipeline keeps running and
    events keep recording into SessionState. Reconnect picks up alerts
    from that point on (history while disconnected is lost; not buffered).
    """
    await ws.accept()

    sess = _active_session
    if sess is None or sess.session_id != session_id:
        # Closing with a code helps clients distinguish "wrong id" from
        # a transient network drop. 4404 isn't a real WS code, just a
        # convenient mnemonic.
        await ws.send_json({
            "kind": "status",
            "state": "ended",
            "timestamp": time.time(),
            "reason": "unknown_session",
        })
        await ws.close(code=4404)
        return

    # Active-status handshake on connect.
    await ws.send_json(StatusMessage(state="active", timestamp=time.time()).to_dict())

    try:
        while True:
            # Wait on either a new queued message or the session ending.
            # If the session ends, we drain remaining queued messages
            # before sending the final ended status, so the phone gets
            # the last alert(s) before the connection closes.
            try:
                msg = await asyncio.wait_for(sess.alert_queue.get(), timeout=1.0)
                await ws.send_json(msg)
            except asyncio.TimeoutError:
                if sess.ended:
                    # Drain any leftover, then bye.
                    while not sess.alert_queue.empty():
                        await ws.send_json(sess.alert_queue.get_nowait())
                    await ws.send_json(
                        StatusMessage(state="ended", timestamp=time.time()).to_dict()
                    )
                    await ws.close()
                    return
                # else loop, give the queue another chance
                continue
    except WebSocketDisconnect:
        # Phone went away. Pipeline keeps running; reconnect any time.
        print(f"ws disconnected: {session_id} (session continues)")


# ── Frame loop ────────────────────────────────────────────────────────────────
async def _frame_loop(sess: ActiveSession) -> None:
    """
    The per-session worker. Reads frames from sess.cap, runs them through
    the pipeline, and queues messages for the WebSocket.

    Runs until:
      - the source ends (video file finishes), OR
      - sess.ended is set (caller invoked /sessions/{id}/stop), OR
      - an uncaught exception (logged, session ends).

    pipeline.process_frame is blocking (heavy CUDA work), so we run it
    in the default thread executor to keep the asyncio loop responsive
    to the WebSocket. The session's queue lives in the event loop, so
    we marshal frame results back to it via call_soon_threadsafe.
    """
    assert _pipeline is not None
    assert sess.cap is not None
    loop = asyncio.get_event_loop()

    max_fps = _CONFIG.get("max_fps")
    frame_min_dt = (1.0 / max_fps) if max_fps else 0.0

    try:
        while not sess.ended:
            t_loop = time.time()

            # cv2.VideoCapture.read is blocking. For a webcam it returns
            # the next frame; for a file it returns False at EOF.
            ok, frame = await loop.run_in_executor(None, sess.cap.read)
            if not ok or frame is None:
                # End of stream (video file finished or camera dropped).
                # Stop the session naturally.
                print(f"session {sess.session_id}: source ended")
                break

            # Process the frame in the executor — this is where the heavy
            # CUDA work happens.
            try:
                result = await loop.run_in_executor(
                    None, _pipeline.process_frame, frame
                )
            except Exception as e:
                print(f"session {sess.session_id}: pipeline error: {e!r}")
                break

            sess.frames_processed += 1

            # Push any messages the pipeline produced. Bounded queue: if
            # the consumer (WebSocket) is gone or slow, we drop oldest.
            if result.alert is not None:
                _push_or_drop(sess, result.alert.to_dict())
            if result.speed_limit_change is not None:
                _push_or_drop(sess, result.speed_limit_change.to_dict())

            # Optional FPS cap so a webcam doesn't melt the Jetson.
            if frame_min_dt > 0:
                slack = frame_min_dt - (time.time() - t_loop)
                if slack > 0:
                    await asyncio.sleep(slack)

    finally:
        # Cleanup is centralised in _shutdown_active_session.
        if not sess.ended:
            # We exited the loop on our own (source ended). Trigger the
            # normal shutdown path so summary/cleanup runs.
            await _shutdown_active_session(triggered_internally=True)


def _push_or_drop(sess: ActiveSession, msg: Dict[str, Any]) -> None:
    """
    Non-blocking enqueue. If the queue is full (WebSocket not reading
    fast enough or disconnected), drop the OLDEST message and push the
    new one — alerts are time-sensitive, the newest is most relevant.
    """
    try:
        sess.alert_queue.put_nowait(msg)
    except asyncio.QueueFull:
        try:
            _ = sess.alert_queue.get_nowait()
            sess.alert_queue.put_nowait(msg)
            sess.frames_dropped += 1
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            sess.frames_dropped += 1


async def _shutdown_active_session(triggered_internally: bool = False) -> SessionSummary:
    """
    Stop the frame loop, close the camera, seal the session, cache the
    summary. Returns the SessionSummary.

    Safe to call multiple times — second call returns the cached summary.
    """
    global _active_session

    sess = _active_session
    if sess is None:
        raise HTTPException(status_code=400, detail="no active session")

    if sess.ended and sess.summary is not None:
        return sess.summary

    sess.ended = True

    # Cancel the frame loop unless we were called from inside it.
    if sess.frame_task is not None and not triggered_internally:
        sess.frame_task.cancel()
        try:
            await sess.frame_task
        except (asyncio.CancelledError, Exception):
            pass

    # Release the camera.
    if sess.cap is not None:
        try:
            sess.cap.release()
        except Exception:
            pass
        sess.cap = None

    # Seal the session and build the summary.
    sess.state.end()
    summary = sess.state.to_summary()
    sess.summary = summary

    # Cache for late /summary calls. Cap the cache so a long-running
    # server doesn't accumulate forever — for the demo this never trips,
    # but it's good hygiene.
    _finished_summaries[sess.session_id] = summary
    if len(_finished_summaries) > 16:
        # Drop the oldest by insertion order.
        oldest = next(iter(_finished_summaries))
        _finished_summaries.pop(oldest, None)

    print(
        f"session ended: {sess.session_id}  "
        f"frames={sess.frames_processed}  dropped_alerts={sess.frames_dropped}  "
        f"score={summary.score:.2f}  label={summary.score_label.value}  "
        f"events={len(summary.events)}"
    )

    # Clear the active slot so a new session can start.
    _active_session = None
    return summary


# ── CLI entry ─────────────────────────────────────────────────────────────────
def _default_weights_dir() -> Path:
    return _REPO / "weights"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tamakkan-server",
        description="Tamakkan backend (FastAPI + WebSocket).",
    )
    p.add_argument(
        "--source",
        default="0",
        help="Camera index (0, 1, ...) or video file path. Default: 0.",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host. Default: 0.0.0.0")
    p.add_argument("--port", type=int, default=8000, help="Bind port. Default: 8000")

    p.add_argument(
        "--weights-dir",
        default=str(_default_weights_dir()),
        help=f"Directory with model weights. Default: {_default_weights_dir()}",
    )
    p.add_argument("--device", default=None, help="cuda:0 / cpu / None (auto)")
    p.add_argument("--depth-every-n", type=int, default=8)
    p.add_argument("--lanes-every-n", type=int, default=5)
    p.add_argument("--ocr-frame-skip", type=int, default=999)
    p.add_argument(
        "--max-fps",
        type=float,
        default=None,
        help="Cap pipeline FPS by sleeping between frames. None = uncapped.",
    )
    return p


def main():
    args = _build_argparser().parse_args()

    weights_dir = Path(args.weights_dir).resolve()
    if not weights_dir.is_dir():
        print(f"ERROR: weights dir not found: {weights_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve weight files and stash everything in _CONFIG for the
    # lifespan to read at startup.
    _CONFIG.update({
        "source":           args.source,
        "host":              args.host,
        "port":              args.port,
        "yolo_weights":      str(weights_dir / "best.engine") if (weights_dir / "best.engine").exists() else str(weights_dir / "best.pt"),
        "bytetrack_config":  str(weights_dir / "bytetrack_tamakkan.yaml"),
        "depth_weights":     str(weights_dir / "depth_anything_v2_vits.engine") if (weights_dir / "depth_anything_v2_vits.engine").exists() else str(weights_dir / "depth_anything_v2_vits.pth"),
        "lane_weights":      str(weights_dir / "culane_res18_v2.engine") if (weights_dir / "culane_res18_v2.engine").exists() else str(weights_dir / "culane_res18_v2.pth"),
        "device":            args.device,
        "depth_every_n":     args.depth_every_n,
        "lanes_every_n":     args.lanes_every_n,
        "ocr_frame_skip":    args.ocr_frame_skip,
        "max_fps":           args.max_fps,
    })

    # Verify weights exist before bothering uvicorn.
    for label, path in [
        ("yolo",      _CONFIG["yolo_weights"]),
        ("bytetrack", _CONFIG["bytetrack_config"]),
        ("depth",     _CONFIG["depth_weights"]),
        ("lanes",     _CONFIG["lane_weights"]),
    ]:
        if not Path(path).is_file():
            print(f"ERROR: missing {label} weights: {path}", file=sys.stderr)
            sys.exit(1)

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()