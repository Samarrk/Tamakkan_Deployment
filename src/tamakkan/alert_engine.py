"""
src/tamakkan/alert_engine.py

Per-frame alert prioritizer and cross-detector cooldown.

Role
----
Each frame, up to 4 detectors can fire (red_light, lane_violation,
tailgating, near_miss). The phone must receive AT MOST ONE alert per
frame — BACKEND_SPEC.md §3.1 says "never a burst." Spoken TTS alerts
overlap badly and exhaust driver attention, so the engine sits between
the detectors and the WebSocket and enforces one-at-a-time delivery.

What this file decides
----------------------
1. Priority. When multiple detectors fire the same frame, pick the
   single most-important one to speak. Priority is by severity, with
   VRU near-miss outranking other criticals, and red-light-ran
   outranking other criticals after that.

2. Cross-detector cooldown. Even with per-detector cooldowns inside
   each detector, the engine adds a floor: no two SPOKEN alerts within
   `min_gap_seconds`. Prevents two unrelated detectors firing back-to-
   back from clobbering each other on the phone's TTS.

3. Critical override. A new event with strictly higher severity than
   the last-spoken event BYPASSES the cooldown. If you just got a
   lane-departure (medium) warning and 1.2 seconds later a pedestrian
   steps in front of you, the VRU alert (critical) must speak NOW.
   Same-severity events still respect the cooldown.

What this file does NOT decide
------------------------------
- Whether an event happened. Detectors decide that. The engine never
  suppresses the SessionEvent — only the spoken Alert. The post-session
  score is computed from events[], so a driver who tailgates in a way
  that trips the engine's spoken cooldown still loses score for it.
  (Detectors already have encounter-level debounce inside themselves;
  sustained situations emit one event per encounter, not per frame, so
  this isn't double-suppressed in practice.)
- WebSocket I/O. The engine returns objects; the FastAPI server sends
  them. Keeps the engine testable without a network.

Inputs / outputs
----------------
process(detector_events, session_time_s) -> EngineOutput
where:
  detector_events    list of whatever the 4 detectors returned this frame.
                     None entries are OK. Up to 4 items.
  session_time_s     seconds since session start, supplied by the pipeline.
  EngineOutput       .alert       Optional[Alert]   — speak this if not None
                     .session_events List[SessionEvent] — always record these

Threading
---------
Same posture as SessionState: one-pipeline-one-engine, not thread-safe.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

from tamakkan.events import (
    Alert,
    SessionEvent,
    Severity,
    canonicalize,
)


# ── Cooldown defaults (tunable on real footage) ───────────────────────────────
# 4.0 seconds floor between SPOKEN alerts. Roughly the time the phone's
# TTS takes to read a typical safety phrase plus a beat of silence.
# Set on the conservative side because TTS overlap is jarring; tune
# downward only after on-road testing if drivers complain about missed
# warnings.
DEFAULT_MIN_GAP_SECONDS = 4.0


# ── Priority ──────────────────────────────────────────────────────────────────
# When more than one event fires the same frame, the one with the LOWEST
# priority number wins (standard "lower=more urgent" convention).
#
# Tiers:
#   0  VRU near-miss               — life at imminent risk
#   1  red_light ran               — already happened, critical
#   2  vehicle near-miss           — vehicle close + closing fast
#   3  red_light ahead             — about to happen, can still react
#   4  tailgating                  — ongoing risky behaviour
#   5  lane_departure              — distraction / drift
#
# This ordering is opinionated and based on the same principle the spec
# uses for severity: imminent + irreversible > imminent + avoidable >
# ongoing > drift. If the team wants to change it, change only the
# numbers below — the dispatch logic doesn't care about the absolute
# values, only the ordering.

_PRI_VRU_NEAR_MISS      = 0
_PRI_RED_LIGHT_RAN      = 1
_PRI_VEH_NEAR_MISS      = 2
_PRI_RED_LIGHT_AHEAD    = 3
_PRI_TAILGATING         = 4
_PRI_LANE_DEPARTURE     = 5

_SEVERITY_RANK = {
    Severity.CRITICAL: 3,
    Severity.HIGH:     2,
    Severity.MEDIUM:   1,
}


def _priority_for(internal_type: str, is_vru: bool) -> int:
    """
    Map (internal detector event type, is_vru flag) -> priority number.

    Falls back to a low priority (large number) for unknown types so a
    future detector doesn't silently steal the channel from existing
    ones until it's been ranked here intentionally.
    """
    if internal_type == "NEAR_MISS":
        return _PRI_VRU_NEAR_MISS if is_vru else _PRI_VEH_NEAR_MISS
    if internal_type == "RED_LIGHT_RAN":
        return _PRI_RED_LIGHT_RAN
    if internal_type == "RED_LIGHT_AHEAD":
        return _PRI_RED_LIGHT_AHEAD
    if internal_type == "TAILGATING":
        return _PRI_TAILGATING
    if internal_type == "LANE_DEPARTURE":
        return _PRI_LANE_DEPARTURE
    return 999  # unknown — never wins against a known event


# ── Output type ───────────────────────────────────────────────────────────────
@dataclass
class EngineOutput:
    """
    What process() returns for one frame.

    alert            Optional spoken alert. None = nothing to speak this frame.
    session_events   Every event that fired this frame, canonicalized.
                     Pipeline records ALL of these to SessionState — the
                     spoken-alert cooldown does not suppress recording.
    """
    alert:          Optional[Alert] = None
    session_events: List[SessionEvent] = field(default_factory=list)


# ── Engine ────────────────────────────────────────────────────────────────────
class AlertEngine:
    """
    Stateful. One AlertEngine per session. Construct once, call
    process(detector_events, session_time_s) every frame.
    """

    def __init__(self, min_gap_seconds: float = DEFAULT_MIN_GAP_SECONDS):
        self.min_gap_seconds: float = min_gap_seconds

        # Wall-clock timestamp of the LAST alert actually emitted, plus
        # its severity, so the critical-override rule can fire when a
        # strictly more severe event arrives during a stale cooldown.
        self._last_alert_ts: Optional[float] = None
        self._last_alert_severity: Optional[Severity] = None

    def reset(self) -> None:
        """Clear cooldown state. Call between unrelated sessions / clips."""
        self._last_alert_ts = None
        self._last_alert_severity = None

    # ── Public API ────────────────────────────────────────────────────────────
    def process(
        self,
        detector_events: Iterable[Any],
        session_time_s: float,
        now: Optional[float] = None,
    ) -> EngineOutput:
        """
        Decide which (if any) detector event to speak this frame, and
        canonicalize every event for recording.

        Args:
            detector_events: iterable of whatever the 4 detectors
                returned. None entries are tolerated and skipped.
            session_time_s: seconds since session start (from
                SessionState.session_time_s()).
            now: wall-clock seconds to use for cooldown math. Defaults
                to time.time(). Test injection only — real callers
                should pass None.

        Returns:
            EngineOutput with at most one alert and the full list of
            canonicalized session events.
        """
        if now is None:
            now = time.time()

        # 1. Canonicalize every detector event. Skip Nones; log unknown
        #    types loudly via ValueError so a misbehaving detector
        #    surfaces immediately instead of dropping silently.
        canonical: List[tuple[Alert, SessionEvent, str, bool]] = []
        for ev in detector_events:
            if ev is None:
                continue
            alert, sess_ev = canonicalize(ev, session_time_s=session_time_s)
            internal_type = (
                ev.type.value if hasattr(ev.type, "value") else str(ev.type)
            )
            is_vru = bool(getattr(ev, "is_vru", False))
            canonical.append((alert, sess_ev, internal_type, is_vru))

        if not canonical:
            return EngineOutput(alert=None, session_events=[])

        all_session_events = [c[1] for c in canonical]

        # 2. Pick the highest-priority candidate this frame.
        canonical.sort(key=lambda c: _priority_for(c[2], c[3]))
        winner_alert, _, _winner_type, _winner_vru = canonical[0]

        # 3. Apply cross-detector cooldown, with critical-override.
        if not self._cooldown_allows(winner_alert.severity, now):
            return EngineOutput(alert=None, session_events=all_session_events)

        # 4. Emit. Update cooldown state.
        self._last_alert_ts = now
        self._last_alert_severity = winner_alert.severity
        return EngineOutput(alert=winner_alert, session_events=all_session_events)

    # ── Internals ─────────────────────────────────────────────────────────────
    def _cooldown_allows(self, candidate_severity: Severity, now: float) -> bool:
        """
        Cooldown rule:

          - If no prior alert was emitted yet, always allow.
          - If the gap since the last alert is >= min_gap_seconds, allow.
          - Otherwise, allow ONLY if the candidate's severity strictly
            outranks the last-emitted severity (critical-override).

        Note: "strictly outranks" means the new severity must be higher
        than the prior one — a critical can preempt a high or medium,
        a high can preempt a medium, but a critical cannot preempt
        another critical mid-cooldown (the prior critical is already
        being spoken; doubling up makes it worse).
        """
        if self._last_alert_ts is None:
            return True

        if (now - self._last_alert_ts) >= self.min_gap_seconds:
            return True

        # Cooldown still active. Critical-override?
        if self._last_alert_severity is None:
            return True   # defensive: state mismatch, allow rather than block

        return _SEVERITY_RANK[candidate_severity] > _SEVERITY_RANK[
            self._last_alert_severity
        ]


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Exercises the engine on fake detector events. Uses SimpleNamespace
    so we don't depend on importing the real detector dataclasses here.
    """
    from types import SimpleNamespace

    def fake(internal_type: str, *, is_vru: bool = False, message_en: str = "msg",
             timestamp: float = 0.0):
        ns = SimpleNamespace()
        ns.type = SimpleNamespace(value=internal_type)
        ns.is_vru = is_vru
        ns.message_en = message_en
        ns.timestamp = timestamp
        return ns

    print("--- single event, no prior cooldown ---")
    engine = AlertEngine(min_gap_seconds=4.0)
    out = engine.process(
        [fake("TAILGATING", message_en="Too close")],
        session_time_s=10.0,
        now=100.0,
    )
    assert out.alert is not None and out.alert.event_type.value == "tailgating"
    assert len(out.session_events) == 1
    print("  spoken:", out.alert.event_type.value, out.alert.severity.value)

    print("\n--- two events same frame; priority wins ---")
    engine = AlertEngine(min_gap_seconds=4.0)
    out = engine.process(
        [
            fake("TAILGATING",        message_en="Too close"),
            fake("NEAR_MISS",         message_en="Pedestrian!", is_vru=True),
            fake("LANE_DEPARTURE",    message_en="Drifting"),
        ],
        session_time_s=10.0,
        now=100.0,
    )
    assert out.alert.event_type.value == "near_miss"
    assert out.alert.is_vru is True
    assert len(out.session_events) == 3   # all 3 still recorded
    print("  spoken:", out.alert.event_type.value, "(VRU)")
    print("  recorded:", [e.event_type.value for e in out.session_events])

    print("\n--- cooldown blocks same-severity follow-up ---")
    engine = AlertEngine(min_gap_seconds=4.0)
    out1 = engine.process(
        [fake("TAILGATING", message_en="Too close")],
        session_time_s=10.0, now=100.0,
    )
    out2 = engine.process(
        [fake("TAILGATING", message_en="Still too close")],
        session_time_s=11.5, now=101.5,   # only 1.5s later
    )
    print(f"  first  alert: {out1.alert.event_type.value}")
    print(f"  second alert: {out2.alert}  (expect None — within cooldown)")
    print(f"  second session_events len: {len(out2.session_events)}  (expect 1 — still recorded)")
    assert out1.alert is not None
    assert out2.alert is None
    assert len(out2.session_events) == 1

    print("\n--- critical OVERRIDES cooldown ---")
    engine = AlertEngine(min_gap_seconds=4.0)
    engine.process(
        [fake("LANE_DEPARTURE", message_en="Drifting")],
        session_time_s=10.0, now=100.0,
    )
    out = engine.process(
        [fake("NEAR_MISS", is_vru=True, message_en="Pedestrian!")],
        session_time_s=11.2, now=101.2,   # within cooldown but critical
    )
    print(f"  critical mid-cooldown spoken? {out.alert is not None}  (expect True)")
    assert out.alert is not None
    assert out.alert.severity.value == "critical"

    print("\n--- critical does NOT override another critical mid-cooldown ---")
    engine = AlertEngine(min_gap_seconds=4.0)
    engine.process(
        [fake("NEAR_MISS", is_vru=True, message_en="Pedestrian!")],
        session_time_s=10.0, now=100.0,
    )
    out = engine.process(
        [fake("NEAR_MISS", is_vru=True, message_en="Another pedestrian!")],
        session_time_s=11.0, now=101.0,
    )
    print(f"  second critical mid-cooldown spoken? {out.alert is not None}  (expect False)")
    assert out.alert is None

    print("\n--- cooldown expires after min_gap_seconds ---")
    engine = AlertEngine(min_gap_seconds=4.0)
    engine.process(
        [fake("TAILGATING", message_en="Too close")],
        session_time_s=10.0, now=100.0,
    )
    out = engine.process(
        [fake("TAILGATING", message_en="Still close")],
        session_time_s=14.5, now=104.5,
    )
    print(f"  4.5s later spoken? {out.alert is not None}  (expect True)")
    assert out.alert is not None

    print("\n--- empty input ---")
    out = AlertEngine().process([], session_time_s=10.0, now=100.0)
    print(f"  alert: {out.alert}  events: {len(out.session_events)}  (both empty)")
    assert out.alert is None and out.session_events == []

    print("\n--- None entries tolerated ---")
    out = AlertEngine().process([None, None], session_time_s=10.0, now=100.0)
    print(f"  alert: {out.alert}  events: {len(out.session_events)}  (both empty)")
    assert out.alert is None and out.session_events == []

    print("\n--- red light ran beats tailgating same frame ---")
    engine = AlertEngine()
    out = engine.process(
        [
            fake("TAILGATING"),
            fake("RED_LIGHT_RAN"),
        ],
        session_time_s=10.0, now=100.0,
    )
    print(f"  spoken: {out.alert.event_type.value} subtype={out.alert.subtype.value}")
    assert out.alert.event_type.value == "red_light"
    assert out.alert.subtype.value == "ran"

    print("\nall asserts passed.")