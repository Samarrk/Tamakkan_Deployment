"""
src/tamakkan/detectors/red_light_detector.py
 
Detects red-light situations from traffic_light tracks + the HSV
LightClassifier.
 
Two event types
---------------
    RED_LIGHT_AHEAD  a red light is visible and stable ahead
    RED_LIGHT_RAN    a confirmed-red light passed low in the frame
                     (the car physically drove past/under it on red)
 
The violation gate (kept from the user's V2 design)
---------------------------------------------------
When a tracked light disappears we decide what it means by WHERE it was
last seen:
  - vanished high in the frame  -> went out of view at distance, the car
    never reached it. NOT a violation.
  - reached low in the frame    -> the car drove up to/past it. If it was
    a confirmed red, that's RED_LIGHT_RAN.
 
The AHEAD spam fix: scene-level cooldown
----------------------------------------
A Saudi signalised stop has MULTIPLE signal heads (typically 3+), and YOLO
also detects the signal heads of OTHER lanes / cross traffic at the same
intersection. ByteTrack additionally re-IDs these small, fast-moving
distant lights constantly. The net effect: one approach to one red
intersection produces a BURST of distinct red detections at different
screen positions and track ids.
 
From the driver's point of view that is ONE event: "the intersection
ahead is red." So AHEAD uses a SCENE-LEVEL cooldown: the first confirmed
red fires once, then ALL further AHEAD events are suppressed for
`ahead_cooldown_seconds`. Per-track spatial dedupe cannot solve this
(the lights are at genuinely different positions and ids); a scene
debounce can.
 
Honest tradeoff of the 50s default: if the car clears one red
intersection and reaches a DIFFERENT red intersection less than the
cooldown later (possible on dense city arterials), the second gets no
AHEAD warning. Mitigation in practice: the driver can see that second
red themselves (it is directly ahead). The actually-observed problem is
spam at a single stop, so the trade favours a long cooldown. This is a
deliberate, documented limitation, not a hidden assumption — tune the
value on real road footage.
 
RAN keeps spatial dedupe
------------------------
Violations are rare and genuinely per-physical-light, so RED_LIGHT_RAN
keeps the IoU/center spatial dedupe (a re-IDed light could double-fire a
violation; a scene cooldown would be wrong for a safety-critical event).
 
Timing
------
All internal logic is FRAME-BASED. The detector is told the stream `fps`
so a cooldown expressed in SECONDS converts to the right number of frames
whether the pipeline runs at 30 fps (PC) or ~12-15 fps (Jetson). The
event `timestamp` stays wall-clock for the AlertEngine / report.
"""
 
from __future__ import annotations
 
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List
 
import numpy as np
 
from tamakkan.models.light_classifier import LightClassifier
from tamakkan.models.tracker import Track
 
 
# ── Tunables (validate/adjust on real footage) ────────────────────────────────
CONFIRM_FRAMES         = 3      # consecutive red frames before AHEAD fires
DISAPPEAR_GRACE        = 5      # frames missing before a track is "really gone"
VIOLATION_Y_FRACTION   = 0.55   # a red light must reach below this fraction of
                                # frame height before vanishing to count as RAN
MIN_LIGHT_PX           = 15     # ignore traffic-light crops smaller than this
 
AHEAD_COOLDOWN_SECONDS = 50.0   # scene-level: after one AHEAD, suppress all
                                # further AHEAD for this long. Long because a
                                # Saudi stop has many heads + cross-traffic
                                # lights. See module docstring for the trade.
 
# Spatial-dedupe parameters — RAN only.
DEDUPE_FRAME_WINDOW    = 60     # suppress a near-duplicate RAN within this many
                                # frames
DEDUPE_IOU_THRESH      = 0.30
DEDUPE_CENTER_PX       = 50
NEG_EVIDENCE_DECAY     = 2      # consecutive_red decay on a non-red frame
                                # before the light has been warned
 
DEFAULT_FPS            = 30.0
 
 
class EventType(str, Enum):
    RED_LIGHT_AHEAD = "RED_LIGHT_AHEAD"
    RED_LIGHT_RAN   = "RED_LIGHT_RAN"
 
 
@dataclass
class RedLightEvent:
    type:       EventType
    track_id:   int
    timestamp:  float          # wall-clock, for AlertEngine / report
    confidence: float
    bbox:       tuple
    frame_idx:  int
 
 
@dataclass
class _TrackState:
    consecutive_red:    int   = 0
    confirmed:          bool  = False
    warned:             bool  = False   # AHEAD considered for this id already
    last_confidence:    float = 0.0
    max_red_confidence: float = 0.0
    last_bbox:          tuple = field(default_factory=tuple)
    last_seen_frame:    int   = 0
    max_y_seen:         float = 0.0
    violation_fired:    bool  = False
 
 
@dataclass
class _FiredFingerprint:
    bbox:      tuple
    frame_idx: int
 
 
def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
 
 
def _center_dist(a: tuple, b: tuple) -> float:
    acx, acy = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
    bcx, bcy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
    return float(np.hypot(acx - bcx, acy - bcy))
 
 
class RedLightDetector:
    """
    Detects red-light events. Stateful — create once, call update() per
    frame with the full track list and the frame.
 
    Args:
        fps: stream frame rate. Used only to convert second-based
             cooldowns to frame counts so durations are correct at any
             pipeline speed. The pipeline should pass the measured fps.
    """
 
    def __init__(
        self,
        fps:                    float = DEFAULT_FPS,
        confirm_frames:         int   = CONFIRM_FRAMES,
        disappear_grace:        int   = DISAPPEAR_GRACE,
        violation_y_fraction:   float = VIOLATION_Y_FRACTION,
        min_light_px:           int   = MIN_LIGHT_PX,
        ahead_cooldown_seconds: float = AHEAD_COOLDOWN_SECONDS,
        dedupe_frame_window:    int   = DEDUPE_FRAME_WINDOW,
        dedupe_iou_thresh:      float = DEDUPE_IOU_THRESH,
        dedupe_center_px:       float = DEDUPE_CENTER_PX,
    ):
        self.fps = fps if fps and fps > 0 else DEFAULT_FPS
        self.confirm_frames       = confirm_frames
        self.disappear_grace      = disappear_grace
        self.violation_y_fraction = violation_y_fraction
        self.min_light_px         = min_light_px
        self.ahead_cooldown_frames = int(ahead_cooldown_seconds * self.fps)
        self.dedupe_frame_window  = dedupe_frame_window
        self.dedupe_iou_thresh    = dedupe_iou_thresh
        self.dedupe_center_px     = dedupe_center_px
 
        self.classifier = LightClassifier()
        self.frame_idx  = 0
        self._states: dict[int, _TrackState] = {}
 
        # Scene-level AHEAD debounce: the frame index of the last AHEAD
        # fired (any light). -inf-equivalent so the first one always fires.
        self._last_ahead_frame: int = -10**9
 
        # RAN keeps spatial dedupe.
        self._recent_ran: List[_FiredFingerprint] = []
 
    # ── Public API ────────────────────────────────────────────────────────────
    def update(self, tracks: List[Track], frame: np.ndarray) -> List[RedLightEvent]:
        self.frame_idx += 1
        events: List[RedLightEvent] = []
        frame_h = frame.shape[0]
        violation_y = frame_h * self.violation_y_fraction
 
        self._prune_fingerprints()
        seen_ids: set[int] = set()
 
        for track in tracks:
            if not track.is_traffic_light:
                continue
 
            x1, y1, x2, y2 = track.bbox_int
            if (x2 - x1) < self.min_light_px or (y2 - y1) < self.min_light_px:
                continue
 
            seen_ids.add(track.track_id)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
 
            result = self.classifier.classify(crop)
            state  = self._states.setdefault(track.track_id, _TrackState())
 
            state.last_bbox       = track.bbox
            state.last_confidence = result.confidence
            state.last_seen_frame = self.frame_idx
 
            y_bottom = track.bbox[3]
            if y_bottom > state.max_y_seen:
                state.max_y_seen = y_bottom
 
            if result.color == "red":
                state.consecutive_red += 1
                if result.confidence > state.max_red_confidence:
                    state.max_red_confidence = result.confidence
            else:
                if not state.warned:
                    state.consecutive_red = max(
                        0, state.consecutive_red - NEG_EVIDENCE_DECAY
                    )
 
            if state.consecutive_red >= self.confirm_frames:
                state.confirmed = True
 
                if not state.warned:
                    state.warned = True
                    # Scene-level cooldown: one AHEAD per intersection
                    # encounter, not per signal head / per track id.
                    since = self.frame_idx - self._last_ahead_frame
                    if since >= self.ahead_cooldown_frames:
                        self._last_ahead_frame = self.frame_idx
                        events.append(RedLightEvent(
                            type       = EventType.RED_LIGHT_AHEAD,
                            track_id   = track.track_id,
                            timestamp  = time.time(),
                            confidence = max(state.max_red_confidence,
                                             state.last_confidence),
                            bbox       = track.bbox,
                            frame_idx  = self.frame_idx,
                        ))
 
        # ── Disappearance → violation check (RAN) ─────────────────────────────
        to_delete = []
        for tid, state in self._states.items():
            if tid in seen_ids:
                continue
            if (self.frame_idx - state.last_seen_frame) < self.disappear_grace:
                continue
 
            if (state.confirmed
                    and not state.violation_fired
                    and state.max_y_seen >= violation_y):
                state.violation_fired = True
                if not self._is_recent_duplicate(
                    state.last_bbox, self._recent_ran
                ):
                    events.append(RedLightEvent(
                        type       = EventType.RED_LIGHT_RAN,
                        track_id   = tid,
                        timestamp  = time.time(),
                        confidence = state.max_red_confidence,
                        bbox       = state.last_bbox,
                        frame_idx  = self.frame_idx,
                    ))
                    self._recent_ran.append(_FiredFingerprint(
                        bbox=state.last_bbox, frame_idx=self.frame_idx
                    ))
 
            to_delete.append(tid)
 
        for tid in to_delete:
            self._states.pop(tid, None)
 
        return events
 
    def reset(self):
        self._states.clear()
        self._recent_ran.clear()
        self._last_ahead_frame = -10**9
        self.frame_idx = 0
 
    # ── Internals ─────────────────────────────────────────────────────────────
    def _is_recent_duplicate(
        self, bbox: tuple, recent: List[_FiredFingerprint]
    ) -> bool:
        for fp in recent:
            if (self.frame_idx - fp.frame_idx) > self.dedupe_frame_window:
                continue
            if (_iou(bbox, fp.bbox) >= self.dedupe_iou_thresh
                    or _center_dist(bbox, fp.bbox) <= self.dedupe_center_px):
                return True
        return False
 
    def _prune_fingerprints(self):
        cutoff = self.frame_idx - self.dedupe_frame_window
        self._recent_ran = [
            fp for fp in self._recent_ran if fp.frame_idx >= cutoff
        ]
 
 
# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import cv2
    from pathlib import Path
 
    # Make tamakkan.* / third_party.* importable when run standalone.
    _repo = Path(__file__).resolve().parents[3]
    for _p in (_repo / "src", _repo / "third_party"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
 
    if len(sys.argv) < 4:
        print("usage: python red_light_detector.py "
              "<weights/best.pt> <weights/bytetrack.yaml> <video.mp4>")
        sys.exit(1)
 
    from tamakkan.models.tracker import TamakkanTracker
 
    cap = cv2.VideoCapture(sys.argv[3])
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
 
    tracker = TamakkanTracker(weights=sys.argv[1], tracker_config=sys.argv[2])
    detector = RedLightDetector(fps=fps)
    print(f"stream fps={fps:.1f}  AHEAD cooldown="
          f"{detector.ahead_cooldown_frames} frames "
          f"(~{detector.ahead_cooldown_frames / fps:.0f}s)")
 
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        tracks = tracker.update(frame)
        for ev in detector.update(tracks, frame):
            print(f"frame {n:>5}  {ev.type.value:16s}  "
                  f"id={ev.track_id:>4}  conf={ev.confidence:.2f}  "
                  f"bbox={tuple(int(v) for v in ev.bbox)}")
    cap.release()
    print(f"done, {n} frames")