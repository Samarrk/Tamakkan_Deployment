"""
src/tamakkan/detectors/lane_violation_detector.py

Lane-departure detector. Consumes the UFLD-v2 LaneDetector output
(List[Lane], in ORIGINAL frame coordinates) and fires LANE_DEPARTURE
when the car sustains an off-centre drift that does NOT self-correct.

PATCH (v2): COOLDOWN_SECONDS 3.0 -> 6.0 only. The harness run showed the
detector fires CORRECTLY (right side, real drift) but re-fires the same
departure too often. That is a cooldown problem, not a sensitivity
problem, so confirm-frames is deliberately NOT changed (raising it would
start missing real departures to fix a frequency issue). Evidence-first:
if 6s still re-fires the same departure, raise further; if it now fires
on marginal non-departures, only then touch confirm-frames.

What this detector CAN do
-------------------------
- Detect sustained lane drift (the distracted/drowsy "creeping out of
  lane" pattern), report side (left/right) and magnitude.

What this detector CANNOT do (stated limitations — keep in the thesis)
----------------------------------------------------------------------
- It cannot truly tell an INTENTIONAL lane change from dangerous drift;
  both move the car across a line. Mitigated with settle-detection
  (a change that completes and re-centres is suppressed), but a slow,
  sloppy intentional change can still fire. A real fix needs turn-signal
  input, which we do not have.
- Settle-detection also suppresses a "drifted then driver yanked back"
  event (accepted tradeoff: the driver already corrected).
- No lane markings -> no lanes -> SILENT non-detection.
- Pixel-proportional, not metric.
- Inherits 100% of the lane model's errors.

Severity: single fixed value (NOT magnitude-scaled — a large offset
usually means a confident deliberate change; scaling up would make
intentional changes the loudest false alarms).

Coordinates: UFLD-v2 returns frame coords; all spatial thresholds are
FRACTIONS of frame width resolved at runtime (resolution-independent).

Timing: frame-based, fps passed in. Event timestamp wall-clock.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from tamakkan.models.lane_model import Lane


# ── Tunables (fractions of frame width unless noted) ──────────────────────────
EGO_MAX_DIST_FRAC      = 0.475
EGO_EXPECTED_DIST_FRAC = 0.225
DEPART_THRESH_FRAC     = 0.10
MIN_EGO_SEPARATION_FRAC = 0.08

FRAMES_TO_CONFIRM      = 15      # NOT raised — see PATCH note
COOLDOWN_SECONDS       = 6.0     # was 3.0 — the actual fix this round
SETTLE_FRAMES          = 12
OFFSET_HISTORY_MAX     = 120

DEFAULT_FPS            = 30.0
SEVERITY               = "medium"


class EventType(str, Enum):
    LANE_DEPARTURE = "LANE_DEPARTURE"


@dataclass
class LaneDepartureEvent:
    type:       EventType
    side:       str
    severity:   str
    offset_px:  float
    message_en: str
    timestamp:  float
    frame_idx:  int


class LaneViolationDetector:
    """Stateful. Create once, call update(lanes, frame) per frame."""

    def __init__(
        self,
        fps:                  float = DEFAULT_FPS,
        frames_to_confirm:    int   = FRAMES_TO_CONFIRM,
        cooldown_seconds:     float = COOLDOWN_SECONDS,
        settle_frames:        int   = SETTLE_FRAMES,
        ego_max_dist_frac:    float = EGO_MAX_DIST_FRAC,
        ego_expected_frac:    float = EGO_EXPECTED_DIST_FRAC,
        depart_thresh_frac:   float = DEPART_THRESH_FRAC,
        min_ego_sep_frac:     float = MIN_EGO_SEPARATION_FRAC,
    ):
        self.fps = fps if fps and fps > 0 else DEFAULT_FPS
        self.frames_to_confirm = frames_to_confirm
        self.cooldown_frames   = int(cooldown_seconds * self.fps)
        self.settle_frames     = settle_frames
        self.ego_max_dist_frac = ego_max_dist_frac
        self.ego_expected_frac = ego_expected_frac
        self.depart_thresh_frac = depart_thresh_frac
        self.min_ego_sep_frac  = min_ego_sep_frac

        self.frame_idx = 0
        self._consecutive_off = 0
        self._offset_history: deque = deque(maxlen=OFFSET_HISTORY_MAX)
        self._last_fired_frame: Optional[int] = None
        self._was_off = False
        self._settle_counter = 0

    def update(self, lanes: List[Lane], frame) -> Optional[LaneDepartureEvent]:
        self.frame_idx += 1
        frame_w = frame.shape[1]
        depart_thresh = self.depart_thresh_frac * frame_w

        ego_left, ego_right = self._pick_ego_lanes(lanes, frame_w)
        offset = self._calculate_offset(ego_left, ego_right, frame_w)

        if offset is None:
            return None

        off = abs(offset) > depart_thresh

        if self._was_off and not off:
            self._settle_counter += 1
            if self._settle_counter >= self.settle_frames:
                self._reset_drift_state()
            return None
        if off:
            self._settle_counter = 0

        if off:
            self._was_off = True
            self._consecutive_off += 1
            self._offset_history.append(offset)
        else:
            self._reset_drift_state()
            return None

        if self._consecutive_off < self.frames_to_confirm:
            return None

        if self._last_fired_frame is not None:
            if (self.frame_idx - self._last_fired_frame) < self.cooldown_frames:
                return None

        self._last_fired_frame = self.frame_idx
        avg_offset = sum(self._offset_history) / len(self._offset_history)
        self._reset_drift_state()
        side = "right" if avg_offset > 0 else "left"
        return self._build_event(avg_offset, side)

    def reset(self):
        self.frame_idx = 0
        self._last_fired_frame = None
        self._reset_drift_state()

    # ── Internals ─────────────────────────────────────────────────────────────
    def _reset_drift_state(self):
        self._consecutive_off = 0
        self._offset_history.clear()
        self._was_off = False
        self._settle_counter = 0

    def _pick_ego_lanes(
        self, lanes: List[Lane], frame_w: float
    ) -> Tuple[Optional[Lane], Optional[Lane]]:
        car_center = frame_w / 2.0
        max_distance = self.ego_max_dist_frac * frame_w

        left = [l for l in lanes
                if l.x_at_bottom < car_center
                and (car_center - l.x_at_bottom) <= max_distance]
        right = [l for l in lanes
                 if l.x_at_bottom >= car_center
                 and (l.x_at_bottom - car_center) <= max_distance]

        ego_left = max(
            left, key=lambda l: (l.x_at_bottom, l.confidence)
        ) if left else None
        ego_right = min(
            right, key=lambda l: (l.x_at_bottom, -l.confidence)
        ) if right else None

        if ego_left is not None and ego_right is not None:
            sep = ego_right.x_at_bottom - ego_left.x_at_bottom
            if sep < self.min_ego_sep_frac * frame_w:
                if ego_left.confidence >= ego_right.confidence:
                    ego_right = None
                else:
                    ego_left = None

        return ego_left, ego_right

    def _calculate_offset(
        self, ego_left: Optional[Lane], ego_right: Optional[Lane],
        frame_w: float
    ) -> Optional[float]:
        car_center = frame_w / 2.0
        expected = self.ego_expected_frac * frame_w

        if ego_left is not None and ego_right is not None:
            lane_center = (ego_left.x_at_bottom + ego_right.x_at_bottom) / 2.0
            return car_center - lane_center
        if ego_left is not None:
            actual = car_center - ego_left.x_at_bottom
            return expected - actual
        if ego_right is not None:
            actual = ego_right.x_at_bottom - car_center
            return actual - expected
        return None

    def _build_event(self, offset_px: float, side: str) -> LaneDepartureEvent:
        if side == "right":
            en = "Lane departure warning: vehicle drifting to the right"
        else:
            en = "Lane departure warning: vehicle drifting to the left"
        return LaneDepartureEvent(
            type       = EventType.LANE_DEPARTURE,
            side       = side,
            severity   = SEVERITY,
            offset_px  = round(offset_px, 1),
            message_en = en,
            timestamp  = time.time(),
            frame_idx  = self.frame_idx,
        )


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import cv2
    from pathlib import Path

    _repo = Path(__file__).resolve().parents[3]
    for _p in (_repo / "src", _repo / "third_party"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

    if len(sys.argv) < 3:
        print("usage: python lane_violation_detector.py "
              "<weights/culane_res18_v2.pth> <video.mp4>")
        sys.exit(1)

    from tamakkan.models.lane_model import LaneDetector

    cap = cv2.VideoCapture(sys.argv[2])
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    lane = LaneDetector(weights_path=sys.argv[1])
    det = LaneViolationDetector(fps=fps)
    print(f"fps={fps:.1f}  confirm={det.frames_to_confirm}  "
          f"cooldown={det.cooldown_frames}f  settle={det.settle_frames}f")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        lanes = lane.update(frame)
        ev = det.update(lanes, frame)
        if ev:
            print(f"frame {n:>5}  {ev.type.value}  side={ev.side}  "
                  f"offset={ev.offset_px:+.0f}px")
    cap.release()
    print(f"done, {n} frames")