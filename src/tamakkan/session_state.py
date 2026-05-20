"""
src/tamakkan/session_state.py

Per-session state container for the Tamakkan backend.

Role
----
One SessionState instance lives for the duration of one drive. It owns
everything that has to persist across frames within a session and
disappear at the end of it:

  - session id + start time
  - current speed limit (live)
  - the sorted list of every recorded SessionEvent so far
  - the set of speed limits seen this session (for summary metadata)
  - a rolling FPS tracker for the pipeline (for summary metadata)
  - the score-from-events function

Everything else stays out. The session does NOT hold model handles,
detector handles, the FastAPI app, or anything about other sessions —
those belong to pipeline.py and server/app.py respectively.

Why this is a thin file
-----------------------
The Jetson is stateless across sessions (BACKEND_SPEC.md §1). There is
no cross-session aggregation here — running totals, weekly stats, and
trip history all belong to the team DB. This module's only job is to
collect what happened in ONE drive and produce a SessionSummary at the
end of it, in the exact wire format from events.py.

Threading note
--------------
SessionState is NOT thread-safe by itself. In the FastAPI server one
session = one pipeline + one WebSocket; only the pipeline mutates state,
the server reads to_summary() once at session-end. If we later run the
pipeline in a worker thread, add a lock around record_event() and the
speed-limit setter — they are the only mutation points.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, List, Optional, Set

from tamakkan.events import (
    SessionEvent,
    SessionSummary,
    EventType,
    RedLightSubtype,
    score_to_label,
)


# ── Scoring weights (BACKEND_SPEC.md §6.1) ────────────────────────────────────
# Isolated in one place so the DB team can override / tweak without
# touching the rest of the file.
SCORE_START      = 5.0
SCORE_MIN        = 0.0
SCORE_MAX        = 5.0

PENALTY_NEAR_MISS         = 1.0
PENALTY_RED_LIGHT_RAN     = 1.0
PENALTY_TAILGATING        = 0.5
PENALTY_RED_LIGHT_AHEAD   = 0.0    # ahead = warning, no score impact (spec §6.1)
PENALTY_LANE_DEPARTURE    = 0.4


def compute_score(events: List[SessionEvent]) -> float:
    """
    Per-session score on a 0-5 scale, computed from the event list.

    Start at 5.0, subtract a penalty for each event, clamp to [0, 5].
    Weights live as module constants above so they're trivial to tune
    without re-reading this function.

    Pure function: takes a list, returns a number. No state. No I/O.
    DB team can call this themselves on the events[] in SessionSummary
    if they want the same number, or replace it with their own.
    """
    score = SCORE_START
    for ev in events:
        if ev.event_type == EventType.NEAR_MISS:
            score -= PENALTY_NEAR_MISS
        elif ev.event_type == EventType.RED_LIGHT:
            if ev.subtype == RedLightSubtype.RAN:
                score -= PENALTY_RED_LIGHT_RAN
            else:
                score -= PENALTY_RED_LIGHT_AHEAD
        elif ev.event_type == EventType.TAILGATING:
            score -= PENALTY_TAILGATING
        elif ev.event_type == EventType.LANE_DEPARTURE:
            score -= PENALTY_LANE_DEPARTURE
        # any future EventType silently scores zero — failing open is the
        # right behaviour for an unrecognised event in a safety system.
    return max(SCORE_MIN, min(SCORE_MAX, score))


# ── FPS tracker ───────────────────────────────────────────────────────────────
# Rolling-window average of per-frame processing time. Used only for the
# summary metadata so the team can see how the Jetson is actually doing
# in the field. Sized for ~30 seconds at 15 fps.

_FPS_WINDOW_FRAMES = 450


@dataclass
class _FPSTracker:
    """
    Rolling average frame-processing FPS. Internal helper — not exported.

    Pipeline calls .tick() after each processed frame with the
    measured per-frame seconds. .average_fps() reports the windowed
    average, or None if no frames have been ticked yet.
    """
    window: int = _FPS_WINDOW_FRAMES
    _frame_seconds: Deque[float] = field(
        default_factory=lambda: deque(maxlen=_FPS_WINDOW_FRAMES)
    )

    def tick(self, frame_seconds: float):
        if frame_seconds > 0:
            self._frame_seconds.append(frame_seconds)

    def average_fps(self) -> Optional[float]:
        if not self._frame_seconds:
            return None
        avg_s = sum(self._frame_seconds) / len(self._frame_seconds)
        return (1.0 / avg_s) if avg_s > 0 else None


# ── SessionState ──────────────────────────────────────────────────────────────
class SessionState:
    """
    Bookkeeping for one drive. Construct at session-start, mutate via
    record_event() / set_speed_limit() / tick_fps() during the drive,
    call to_summary() at session-end.

    Public attributes are deliberately readable (no getters/setters) —
    server/app.py and pipeline.py both need to inspect the current state
    for the WebSocket / per-frame logic, and Python idioms favour direct
    access over Java-style encapsulation here. Only mutate through the
    designated methods so derived state (events sort order, speed-limit
    history, fps window) stays consistent.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        started_at: Optional[float] = None,
        fps_window_frames: int = _FPS_WINDOW_FRAMES,
    ):
        # Unique-ish id with a readable prefix. Server may override with
        # its own scheme; pipeline-only / smoke-test usage gets a default.
        self.session_id: str = session_id or f"s_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        # Wall-clock start. Used for session_time_s on every event and
        # for the ISO timestamps in the final summary.
        self.started_at_epoch: float = (
            started_at if started_at is not None else time.time()
        )
        self.ended_at_epoch:   Optional[float] = None

        # Live state
        self.current_speed_limit: Optional[int] = None
        self.speed_limits_seen:   Set[int] = set()

        # Event log — kept in insertion order; pipeline always records
        # events as they happen in wall-clock order, so no resort needed.
        self.events: List[SessionEvent] = []

        # Performance tracker
        self._fps = _FPSTracker(window=fps_window_frames)

    # ── Time helpers ──────────────────────────────────────────────────────────
    def session_time_s(self, timestamp: Optional[float] = None) -> float:
        """
        Seconds since session start at the given wall-clock timestamp
        (defaults to now). Used to stamp every Alert + SessionEvent.
        Never negative.
        """
        t = timestamp if timestamp is not None else time.time()
        return max(0.0, t - self.started_at_epoch)

    # ── Mutators ──────────────────────────────────────────────────────────────
    def record_event(self, event: SessionEvent) -> None:
        """Append one canonical SessionEvent to the session log."""
        self.events.append(event)

    def set_speed_limit(self, limit_kmh: Optional[int]) -> bool:
        """
        Update the live speed limit.

        Returns True if the value actually changed (caller should push a
        SpeedLimitChange over the WebSocket), False if it was a no-op
        (same value as current; no point spamming the phone).

        Records every distinct seen value into speed_limits_seen for the
        final summary metadata.
        """
        if limit_kmh is not None:
            self.speed_limits_seen.add(int(limit_kmh))

        if limit_kmh == self.current_speed_limit:
            return False

        self.current_speed_limit = limit_kmh
        return True

    def tick_fps(self, frame_seconds: float) -> None:
        """Record one processed frame's wall-clock seconds for FPS averaging."""
        self._fps.tick(frame_seconds)

    def end(self, ended_at: Optional[float] = None) -> None:
        """
        Mark the session as ended. Idempotent — calling end() twice
        keeps the FIRST end time (session-end is a one-time event).
        """
        if self.ended_at_epoch is None:
            self.ended_at_epoch = ended_at if ended_at is not None else time.time()

    # ── Read helpers ──────────────────────────────────────────────────────────
    @property
    def is_ended(self) -> bool:
        return self.ended_at_epoch is not None

    @property
    def duration_seconds(self) -> int:
        """Whole-seconds session duration. If not yet ended, duration so far."""
        end = self.ended_at_epoch if self.ended_at_epoch is not None else time.time()
        return max(0, int(round(end - self.started_at_epoch)))

    def average_fps(self) -> Optional[float]:
        return self._fps.average_fps()

    # ── Summary ───────────────────────────────────────────────────────────────
    def to_summary(self) -> SessionSummary:
        """
        Build the final SessionSummary from current state.

        Safe to call before end() (returns "summary so far" with the
        current wall-clock as the end time). After end(), repeated calls
        return the same fixed summary.

        Score is computed here from self.events using compute_score —
        single source of truth, can't drift from the event log.
        """
        end_epoch = (
            self.ended_at_epoch if self.ended_at_epoch is not None else time.time()
        )

        score = compute_score(self.events)
        label = score_to_label(score)

        # speed_limits_seen sorted so the summary is deterministic.
        metadata = {
            "speed_limits_seen": sorted(self.speed_limits_seen),
        }
        avg_fps = self.average_fps()
        if avg_fps is not None:
            metadata["model_fps_avg"] = round(avg_fps, 1)

        return SessionSummary(
            session_id       = self.session_id,
            started_at       = _to_iso(self.started_at_epoch),
            ended_at         = _to_iso(end_epoch),
            duration_seconds = max(0, int(round(end_epoch - self.started_at_epoch))),
            score            = score,
            score_label      = label,
            events           = list(self.events),    # defensive copy
            metadata         = metadata,
        )


def _to_iso(epoch: float) -> str:
    """
    UTC ISO-8601 with 'Z' suffix to match the spec example exactly
    (e.g. "2026-05-19T17:00:00Z"). Truncates to whole seconds — the
    summary doesn't need sub-second precision and the app's date
    parsers behave better without microseconds.
    """
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    from tamakkan.events import (
        EventType,
        Severity,
        RedLightSubtype,
        SessionEvent,
    )

    print("--- empty session (clean drive) ---")
    s = SessionState(session_id="s_test_clean")
    # simulate a short drive
    time.sleep(0.01)
    s.tick_fps(1 / 14.0)        # ~14 fps
    s.tick_fps(1 / 15.0)
    s.set_speed_limit(80)
    s.set_speed_limit(80)       # no-op
    changed = s.set_speed_limit(100)
    print(f"  100 was a change: {changed}")
    s.end()
    summary = s.to_summary()
    print(json.dumps(summary.to_dict(), indent=2))
    print(f"  score: {summary.score}  label: {summary.score_label.value}")

    print("\n--- session with 5 events (mixed) ---")
    s2 = SessionState(session_id="s_test_mixed")
    t0 = s2.started_at_epoch

    def _ev(et, sub, sev, vru, dt_s):
        ts = t0 + dt_s
        return SessionEvent(et, sub, sev, vru, dt_s, ts)

    s2.record_event(_ev(EventType.LANE_DEPARTURE, None, Severity.MEDIUM, False, 12.0))
    s2.record_event(_ev(EventType.TAILGATING,     None, Severity.HIGH,   False, 60.0))
    s2.record_event(_ev(EventType.RED_LIGHT,      RedLightSubtype.AHEAD, Severity.HIGH, False, 90.0))
    s2.record_event(_ev(EventType.RED_LIGHT,      RedLightSubtype.RAN,   Severity.CRITICAL, False, 130.0))
    s2.record_event(_ev(EventType.NEAR_MISS,      None, Severity.CRITICAL, True, 200.0))
    s2.set_speed_limit(80)
    s2.set_speed_limit(100)
    s2.set_speed_limit(60)
    for _ in range(20):
        s2.tick_fps(1 / 14.3)
    time.sleep(0.01)
    s2.end()
    summary = s2.to_summary()

    # Expected score: 5.0 - 0.4 (lane) - 0.5 (tail) - 0 (ahead) - 1.0 (ran) - 1.0 (vru) = 2.1
    print(f"  events:              {len(summary.events)}")
    print(f"  event_counts:        {summary.to_dict()['event_counts']}")
    print(f"  score:               {summary.score}     (expect 2.1)")
    print(f"  score_label:         {summary.score_label.value}     (expect NEEDS WORK)")
    print(f"  speed_limits_seen:   {summary.metadata['speed_limits_seen']}")
    print(f"  model_fps_avg:       {summary.metadata.get('model_fps_avg')}")

    print("\n--- floor at zero (catastrophic drive) ---")
    s3 = SessionState(session_id="s_test_floor")
    for _ in range(10):
        s3.record_event(_ev(EventType.NEAR_MISS, None, Severity.CRITICAL, True, 1.0))
    s3.end()
    print(f"  score: {compute_score(s3.events)}  (expect 0.0)")

    print("\n--- session_time_s ---")
    s4 = SessionState(session_id="s_test_time", started_at=1000.0)
    print(f"  at t=1042.5  -> {s4.session_time_s(1042.5)}  (expect 42.5)")
    print(f"  at t=999.0   -> {s4.session_time_s(999.0)}   (expect 0.0, clamped)")

    print("\n--- to_summary() before end() ---")
    s5 = SessionState(session_id="s_test_inprogress")
    s5.record_event(_ev(EventType.TAILGATING, None, Severity.HIGH, False, 5.0))
    summary = s5.to_summary()
    print(f"  in-progress duration_seconds: {summary.duration_seconds} (>= 0)")
    print(f"  in-progress score: {summary.score}  (expect 4.5)")