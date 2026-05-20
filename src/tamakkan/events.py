"""
src/tamakkan/events.py
 
Canonical event types and wire-format dataclasses for the Tamakkan backend.
 
Why this file exists
--------------------
Each detector has its OWN internal EventType enum (RED_LIGHT_AHEAD,
LANE_DEPARTURE, TAILGATING, NEAR_MISS, ...). Those internal enums are
useful inside the detector but they are NOT what the phone app sees on
the wire. The wire format is the smaller, app-facing taxonomy defined in
BACKEND_SPEC.md §2 — `lane_departure`, `tailgating`, `red_light`,
`near_miss` — with `subtype` carrying the red-light variant.
 
This file is the single source of truth for:
  - the canonical EventType used on the wire
  - the per-event payloads pushed over the WebSocket (Alert)
  - the per-event records stored in the session summary (SessionEvent)
  - the SessionSummary itself
  - SpeedLimitChange (not a mistake — see §4 of the spec)
  - the translator that turns ANY detector's internal event into the
    canonical Alert + SessionEvent pair
 
Anything that crosses the pipeline/server boundary uses these types.
Detector files keep their own internal dataclasses; only this module
knows how to flatten them to wire format.
 
App language
------------
The phone app is ENGLISH ONLY. Alerts carry message_en. The detector
files currently still produce message_ar fields too — those are simply
ignored here. (Cleanup: remove message_ar from the four detector files
in a later pass; not required for correctness.)
"""
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
 
 
# ── Canonical taxonomy (BACKEND_SPEC.md §2) ───────────────────────────────────
class EventType(str, Enum):
    """
    Wire-format event types. These are the EXACT strings the phone app
    sees in `event_type`. Detector-internal enums (RED_LIGHT_AHEAD,
    LANE_DEPARTURE, ...) are separate and never appear on the wire.
    """
    LANE_DEPARTURE = "lane_departure"
    TAILGATING     = "tailgating"
    RED_LIGHT      = "red_light"
    NEAR_MISS      = "near_miss"
 
 
class Severity(str, Enum):
    """
    Wire-format severity. Matches BACKEND_SPEC.md §2.1 exactly:
        lane_departure   -> medium
        tailgating       -> high
        red_light ahead  -> high
        red_light ran    -> critical
        near_miss veh    -> high
        near_miss vru    -> critical
    """
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"
 
 
class RedLightSubtype(str, Enum):
    """Carried in Alert.subtype / SessionEvent.subtype when event_type == red_light."""
    AHEAD = "ahead"
    RAN   = "ran"
 
 
# ── Wire-format payloads (BACKEND_SPEC.md §3, §6) ─────────────────────────────
@dataclass
class Alert:
    """
    The real-time alert pushed over the WebSocket (spec §3.1).
 
    One alert per frame at most — AlertEngine handles prioritization and
    cooldown so the phone never receives a burst.
 
    Fields exactly match the JSON the spec shows:
        kind             always "alert" (set by to_dict)
        event_type       canonical EventType value
        subtype          "ahead" | "ran" for red_light, else None
        severity         Severity value
        is_vru           True only for near_miss involving a person/VRU
        message_en       phone speaks this
        timestamp        wall-clock seconds
        session_time_s   seconds since session start
    """
    event_type:     EventType
    subtype:        Optional[RedLightSubtype]
    severity:       Severity
    is_vru:         bool
    message_en:     str
    timestamp:      float
    session_time_s: float
 
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind":           "alert",
            "event_type":     self.event_type.value,
            "subtype":        self.subtype.value if self.subtype is not None else None,
            "severity":       self.severity.value,
            "is_vru":         self.is_vru,
            "message_en":     self.message_en,
            "timestamp":      self.timestamp,
            "session_time_s": self.session_time_s,
        }
 
 
@dataclass
class SessionEvent:
    """
    A single event as it appears inside SessionSummary.events[] (spec §6).
 
    Smaller than Alert — the summary doesn't need to re-store the spoken
    message, just the structured fact that the event happened.
    """
    event_type:     EventType
    subtype:        Optional[RedLightSubtype]
    severity:       Severity
    is_vru:         bool
    session_time_s: float
    timestamp:      float
 
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type":     self.event_type.value,
            "subtype":        self.subtype.value if self.subtype is not None else None,
            "severity":       self.severity.value,
            "is_vru":         self.is_vru,
            "session_time_s": self.session_time_s,
            "timestamp":      self.timestamp,
        }
 
 
@dataclass
class SpeedLimitChange:
    """
    Emitted by speed_limit_reader when the live speed limit changes
    (BACKEND_SPEC.md §3.2). NOT a mistake event — never goes into
    SessionSummary.events. Pushed over the WebSocket as kind=speed_limit.
 
    limit_kmh may be None if the previous "no limit known" state should
    explicitly be cleared (not currently emitted, reserved).
    """
    limit_kmh: Optional[int]
    timestamp: float
 
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind":      "speed_limit",
            "limit_kmh": self.limit_kmh,
            "timestamp": self.timestamp,
        }
 
 
@dataclass
class StatusMessage:
    """
    Lifecycle status (BACKEND_SPEC.md §3.3). Sent by server/app.py at
    WebSocket open and at session end.
    """
    state:     str    # "active" | "ended"
    timestamp: float
 
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind":      "status",
            "state":     self.state,
            "timestamp": self.timestamp,
        }
 
 
# ── Score labels (BACKEND_SPEC.md §6 — match app's getScoreLabel) ─────────────
class ScoreLabel(str, Enum):
    EXCELLENT  = "EXCELLENT"
    GOOD       = "GOOD"
    IMPROVING  = "IMPROVING"
    NEEDS_WORK = "NEEDS WORK"   # space matches the user-facing label in the app
 
 
def score_to_label(score: float, max_score: float = 5.0) -> ScoreLabel:
    """
    Thresholds (spec §6): on a 0-5 scale,
        >= 90% (4.5)  -> EXCELLENT
        >= 75% (3.75) -> GOOD
        >= 60% (3.0)  -> IMPROVING
        else          -> NEEDS WORK
    """
    if max_score <= 0:
        return ScoreLabel.NEEDS_WORK
    pct = score / max_score
    if pct >= 0.90:
        return ScoreLabel.EXCELLENT
    if pct >= 0.75:
        return ScoreLabel.GOOD
    if pct >= 0.60:
        return ScoreLabel.IMPROVING
    return ScoreLabel.NEEDS_WORK
 
 
@dataclass
class SessionSummary:
    """
    The complete post-drive payload (BACKEND_SPEC.md §6).
 
    Returned by POST /sessions/{id}/stop and GET /sessions/{id}/summary.
    Also the object the app forwards to the team DB.
 
    `events` is the canonical SessionEvent list; the summary serializer
    builds `event_counts` from it so the two are always consistent.
    """
    session_id:       str
    started_at:       str        # ISO-8601 UTC
    ended_at:         str        # ISO-8601 UTC
    duration_seconds: int
    score:            float
    score_label:      ScoreLabel
    events:           List[SessionEvent]
    metadata:         Dict[str, Any] = field(default_factory=dict)
 
    def to_dict(self) -> Dict[str, Any]:
        # Build event_counts from events — single source of truth.
        counts: Dict[str, int] = {et.value: 0 for et in EventType}
        for e in self.events:
            counts[e.event_type.value] += 1
        return {
            "session_id":       self.session_id,
            "started_at":       self.started_at,
            "ended_at":         self.ended_at,
            "duration_seconds": self.duration_seconds,
            "score":            round(self.score, 2),
            "score_label":      self.score_label.value,
            "event_counts":     counts,
            "events":           [e.to_dict() for e in self.events],
            "metadata":         self.metadata,
        }
 
 
# ── Detector-event → canonical translator ─────────────────────────────────────
# This is the ONE place that knows the shape of every detector's
# internal event. Adding a new detector means adding one branch here;
# nothing else in the system needs to change.
#
# We accept the detector events DUCK-TYPED (by their `type` enum value)
# rather than by isinstance, so this file does NOT import from the
# detector modules. That keeps the dependency arrow one-way:
# detectors -> events  (never events -> detectors).
 
def canonicalize(
    detector_event: Any,
    session_time_s: float,
) -> tuple[Alert, SessionEvent]:
    """
    Convert any detector's internal event into the (Alert, SessionEvent)
    pair the rest of the backend uses.
 
    Args:
        detector_event: a LaneDepartureEvent / TailgatingEvent /
            RedLightEvent / NearMissEvent. Each carries a `.type` enum
            whose .value is one of:
                "LANE_DEPARTURE" | "TAILGATING"
                | "RED_LIGHT_AHEAD" | "RED_LIGHT_RAN"
                | "NEAR_MISS"
        session_time_s: seconds since session start, computed by the
            pipeline / session_state (the detector doesn't know this).
 
    Returns:
        (Alert, SessionEvent) — the WebSocket payload and the summary
        record for the same underlying detection.
 
    Raises:
        ValueError if the event's .type isn't a known internal type.
        Bugs (wrong dataclass, missing field) raise AttributeError —
        better to fail loud than silently emit malformed wire data.
    """
    internal = detector_event.type.value if hasattr(detector_event.type, "value") \
               else str(detector_event.type)
 
    ts = detector_event.timestamp
 
    if internal == "LANE_DEPARTURE":
        return (
            Alert(
                event_type     = EventType.LANE_DEPARTURE,
                subtype        = None,
                severity       = Severity.MEDIUM,
                is_vru         = False,
                message_en     = detector_event.message_en,
                timestamp      = ts,
                session_time_s = session_time_s,
            ),
            SessionEvent(
                event_type     = EventType.LANE_DEPARTURE,
                subtype        = None,
                severity       = Severity.MEDIUM,
                is_vru         = False,
                session_time_s = session_time_s,
                timestamp      = ts,
            ),
        )
 
    if internal == "TAILGATING":
        return (
            Alert(
                event_type     = EventType.TAILGATING,
                subtype        = None,
                severity       = Severity.HIGH,
                is_vru         = False,
                message_en     = detector_event.message_en,
                timestamp      = ts,
                session_time_s = session_time_s,
            ),
            SessionEvent(
                event_type     = EventType.TAILGATING,
                subtype        = None,
                severity       = Severity.HIGH,
                is_vru         = False,
                session_time_s = session_time_s,
                timestamp      = ts,
            ),
        )
 
    if internal == "RED_LIGHT_AHEAD":
        # The red-light detector's event dataclass does NOT carry
        # message_en — synthesize it here. This keeps the detector
        # minimal (it only decides "is this red ahead / was it ran")
        # and centralizes user-facing strings in one file.
        return (
            Alert(
                event_type     = EventType.RED_LIGHT,
                subtype        = RedLightSubtype.AHEAD,
                severity       = Severity.HIGH,
                is_vru         = False,
                message_en     = "Red light ahead, prepare to stop",
                timestamp      = ts,
                session_time_s = session_time_s,
            ),
            SessionEvent(
                event_type     = EventType.RED_LIGHT,
                subtype        = RedLightSubtype.AHEAD,
                severity       = Severity.HIGH,
                is_vru         = False,
                session_time_s = session_time_s,
                timestamp      = ts,
            ),
        )
 
    if internal == "RED_LIGHT_RAN":
        return (
            Alert(
                event_type     = EventType.RED_LIGHT,
                subtype        = RedLightSubtype.RAN,
                severity       = Severity.CRITICAL,
                is_vru         = False,
                message_en     = "You ran a red light",
                timestamp      = ts,
                session_time_s = session_time_s,
            ),
            SessionEvent(
                event_type     = EventType.RED_LIGHT,
                subtype        = RedLightSubtype.RAN,
                severity       = Severity.CRITICAL,
                is_vru         = False,
                session_time_s = session_time_s,
                timestamp      = ts,
            ),
        )
 
    if internal == "NEAR_MISS":
        # NearMissEvent carries is_vru and an internal severity string.
        sev = Severity.CRITICAL if detector_event.is_vru else Severity.HIGH
        return (
            Alert(
                event_type     = EventType.NEAR_MISS,
                subtype        = None,
                severity       = sev,
                is_vru         = bool(detector_event.is_vru),
                message_en     = detector_event.message_en,
                timestamp      = ts,
                session_time_s = session_time_s,
            ),
            SessionEvent(
                event_type     = EventType.NEAR_MISS,
                subtype        = None,
                severity       = sev,
                is_vru         = bool(detector_event.is_vru),
                session_time_s = session_time_s,
                timestamp      = ts,
            ),
        )
 
    raise ValueError(f"unknown detector event type: {internal!r}")
 
 
# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from types import SimpleNamespace
    import json
 
    print("--- Alert.to_dict ---")
    a = Alert(
        event_type=EventType.TAILGATING,
        subtype=None,
        severity=Severity.HIGH,
        is_vru=False,
        message_en="Following too closely",
        timestamp=1716200000.123,
        session_time_s=42.5,
    )
    print(json.dumps(a.to_dict(), indent=2))
 
    print("\n--- SpeedLimitChange.to_dict ---")
    print(SpeedLimitChange(limit_kmh=80, timestamp=1716200000.5).to_dict())
 
    print("\n--- score_to_label ---")
    for s in (5.0, 4.7, 4.0, 3.2, 2.0):
        print(f"  score={s} -> {score_to_label(s).value}")
 
    print("\n--- SessionSummary.to_dict ---")
    e1 = SessionEvent(EventType.LANE_DEPARTURE, None, Severity.MEDIUM,
                      False, 120.4, 1716200120.4)
    e2 = SessionEvent(EventType.RED_LIGHT, RedLightSubtype.RAN,
                      Severity.CRITICAL, False, 410.0, 1716200410.0)
    summary = SessionSummary(
        session_id="s_1716200000",
        started_at="2026-05-19T17:00:00Z",
        ended_at="2026-05-19T17:32:10Z",
        duration_seconds=1930,
        score=4.2,
        score_label=score_to_label(4.2),
        events=[e1, e2],
        metadata={"speed_limits_seen": [80, 100], "model_fps_avg": 14.6},
    )
    print(json.dumps(summary.to_dict(), indent=2))
 
    print("\n--- canonicalize() on a fake LANE_DEPARTURE ---")
    fake = SimpleNamespace(
        type=SimpleNamespace(value="LANE_DEPARTURE"),
        message_en="Lane departure warning: drifting right",
        timestamp=1716200120.4,
    )
    alert, sess_ev = canonicalize(fake, session_time_s=120.4)
    print("alert:        ", alert.to_dict())
    print("session_event:", sess_ev.to_dict())