"""
scripts/test_server_client.py

Phone-app simulator. Validates the entire Tamakkan backend wire
contract end-to-end WITHOUT needing the real phone app to exist.

What it does
------------
1. Hits POST /sessions/start on the server → gets a session_id.
2. Opens the WebSocket ws://host:port/ws/session/{session_id}.
3. Listens for incoming alert / speed_limit / status messages and
   prints each as it arrives — exactly what the phone app does, minus
   the TTS and the pretty UI.
4. After --duration seconds (or Ctrl+C), POSTs /sessions/{id}/stop.
5. Pretty-prints the returned SessionSummary.

Usage
-----
  # In another terminal, start the server:
  #   python -m server.app --source path/to/clip.mp4

  python scripts/test_server_client.py --host 127.0.0.1 --port 8000 --duration 60

If --duration is omitted, the script runs until the source ends (server
sends status=ended) or Ctrl+C.

Why this exists
---------------
Catches contract bugs (wrong field name, wrong endpoint URL, wrong
WebSocket message shape) before the phone app integrates. If this
script works, the phone app will work — same JSON, same endpoints.

Dependencies
------------
  pip install httpx websockets

httpx is used over requests because we already pull async-friendly libs
for the server side, and websockets is the standard pure-asyncio client.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from typing import Any, Dict, Optional

import httpx
import websockets


# ── pretty printing ───────────────────────────────────────────────────────────
def _fmt_alert(msg: Dict[str, Any]) -> str:
    et  = msg.get("event_type", "?")
    sub = msg.get("subtype")
    sev = msg.get("severity", "?")
    vru = msg.get("is_vru", False)
    txt = msg.get("message_en", "")
    sub_part = f"/{sub}" if sub else ""
    vru_part = " [VRU]" if vru else ""
    return f"ALERT  {et}{sub_part}  sev={sev}{vru_part}  \"{txt}\""


def _fmt_speed(msg: Dict[str, Any]) -> str:
    lim = msg.get("limit_kmh")
    return f"SPEED  limit -> {lim} km/h"


def _fmt_status(msg: Dict[str, Any]) -> str:
    return f"STATUS {msg.get('state', '?')}"


def _pretty_print_msg(msg: Dict[str, Any]) -> None:
    kind = msg.get("kind", "?")
    ts = msg.get("session_time_s")
    ts_str = f"t={ts:7.2f}s " if isinstance(ts, (int, float)) else "         "
    if kind == "alert":
        line = _fmt_alert(msg)
    elif kind == "speed_limit":
        line = _fmt_speed(msg)
    elif kind == "status":
        line = _fmt_status(msg)
    else:
        line = f"?      {json.dumps(msg)}"
    print(f"  {ts_str} {line}")


# ── main ──────────────────────────────────────────────────────────────────────
async def run(host: str, port: int, duration: Optional[float]) -> int:
    base_http = f"http://{host}:{port}"
    base_ws   = f"ws://{host}:{port}"

    # 1) Health check
    print(f"--- GET {base_http}/health ---")
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{base_http}/health")
        except httpx.ConnectError:
            print(f"ERROR: could not connect to {base_http}. Is the server running?")
            return 1
        print(f"  {r.status_code}  {r.json()}")
        if r.status_code != 200:
            return 2

        # 2) Start session
        print(f"\n--- POST {base_http}/sessions/start ---")
        r = await client.post(
            f"{base_http}/sessions/start",
            json={"device_id": "test_client_v1"},
        )
        if r.status_code != 200:
            print(f"  ERROR {r.status_code}: {r.text}")
            return 3
        session_id = r.json()["session_id"]
        print(f"  session_id: {session_id}")

    # 3) Open WebSocket and stream
    ws_url = f"{base_ws}/ws/session/{session_id}"
    print(f"\n--- WS {ws_url} ---")

    stop_event = asyncio.Event()
    received_ended_status = False

    def _request_stop(*_args):
        stop_event.set()

    # Catch Ctrl+C so we still issue the /stop call.
    loop = asyncio.get_event_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                loop.add_signal_handler(sig, _request_stop)
            except NotImplementedError:
                # Windows: signal handlers in asyncio aren't supported on
                # ProactorEventLoop. Ctrl+C will still raise KeyboardInterrupt
                # in the main task; just no graceful path.
                pass

    async def consume() -> None:
        nonlocal received_ended_status
        try:
            async with websockets.connect(ws_url) as ws:
                while not stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    msg = json.loads(raw)
                    _pretty_print_msg(msg)
                    if msg.get("kind") == "status" and msg.get("state") == "ended":
                        received_ended_status = True
                        stop_event.set()
                        break
        except websockets.exceptions.ConnectionClosed:
            print("  (websocket closed by server)")
        except Exception as e:
            print(f"  websocket error: {e!r}")

    # Optional duration timer
    async def timeout() -> None:
        if duration is not None:
            await asyncio.sleep(duration)
            print(f"\n  (--duration {duration}s elapsed; stopping)")
            stop_event.set()

    consumer = asyncio.create_task(consume())
    timer    = asyncio.create_task(timeout())

    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        stop_event.set()

    consumer.cancel()
    timer.cancel()
    for t in (consumer, timer):
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass

    # 4) Stop session (unless the server already ended on its own — the
    # /stop call still works in that case, returning the cached summary).
    print(f"\n--- POST {base_http}/sessions/{session_id}/stop ---")
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.post(f"{base_http}/sessions/{session_id}/stop")
        except httpx.ConnectError:
            print(f"  ERROR: could not reach server for stop call")
            return 4

    if r.status_code != 200:
        print(f"  ERROR {r.status_code}: {r.text}")
        return 5

    summary = r.json()

    print("\n--- SessionSummary ---")
    print(json.dumps(summary, indent=2))

    # 5) Sanity checks on what we got
    print("\n--- sanity checks ---")
    errs = []
    counts = summary.get("event_counts", {})
    events = summary.get("events", [])
    counts_total = sum(counts.values())
    if counts_total != len(events):
        errs.append(
            f"event_counts total ({counts_total}) != events[] length ({len(events)})"
        )

    score = summary.get("score")
    if not (isinstance(score, (int, float)) and 0.0 <= score <= 5.0):
        errs.append(f"score out of range: {score!r}")

    label = summary.get("score_label")
    if label not in ("EXCELLENT", "GOOD", "IMPROVING", "NEEDS WORK"):
        errs.append(f"unexpected score_label: {label!r}")

    if errs:
        print("  FAILURES:")
        for e in errs:
            print(f"    - {e}")
        return 6
    print("  all OK ✓")
    return 0


def main():
    p = argparse.ArgumentParser(
        prog="test_server_client",
        description="Phone-app stand-in. Validates the Tamakkan wire contract.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Auto-stop after this many seconds. Omit to run until "
             "source ends or Ctrl+C.",
    )
    args = p.parse_args()

    try:
        rc = asyncio.run(run(args.host, args.port, args.duration))
    except KeyboardInterrupt:
        print("\n(interrupted)")
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()