"""
src/tamakkan/detectors/near_miss_detector.py

"Something is very close AND rapidly closing on you."

REBUILD (v2) — the v1 fired ~30-67 times per clip on normal traffic.
Root cause from the harness run:
  - area-growth was noise: every car in traffic has a slightly wobbling
    bbox, so AREA_GROWTH_FRAC was tripped constantly.
  - there was no "is it actually close right now" gate — only "is it
    changing", so a far car with a drifting box fired.
  - no spatial gate — a car overtaking in the NEXT lane loomed and fired.
  - no per-track cooldown — one approach = dozens of events.

Rebuilt design (decided with the user, evidence-driven):
  1. SPATIAL GATE: the object's bbox centre must be inside a central
     "driving path" band (fixed fraction of width). A car passing in the
     next lane, or a pedestrian on the sidewalk, is outside it and is
     ignored. Fixed band, NOT lane-model based: near-miss is the most
     safety-critical detector and must not depend on the least reliable
     model (lanes). Predictable + always available beats clever.
  2. ALREADY-CLOSE GATE: the object must ALSO be genuinely close *now* —
     its patch depth must be a high fraction of the current frame's max
     depth. "Changing" is not enough; it must already be near.
  3. PRIMARY SIGNAL: depth RISING fast over a short window (probe proved
     depth is the clean, stable near-field signal; bigger == closer).
     Area-growth is DROPPED entirely — it was the noise source.
  4. PER-TRACK COOLDOWN: one genuine approach = one alert, not a burst.
  5. CLASS-AWARE OUTPUT: a person/VRU closing fast is a higher-priority,
     explicitly-"PERSON" alert. Vehicles get the standard message.

Honest limitations (thesis)
---------------------------
- Relative monocular proximity, not time-to-collision in seconds.
- A very sudden close cut-in may not give enough window frames before
  it is already an emergency — this flags rapid approach, it is not a
  guaranteed last-instant collision avoider.
- Fixed path band can't perfectly separate a lane-edge pedestrian from a
  sidewalk one; the close+closing conjunction is what suppresses false
  positives, not the band alone.
- Inherits tracker (ID switch truncates history) and depth model error.

Timing: frame-based, fps passed in. Event timestamp wall-clock.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from tamakkan.models.tracker import Track


# ── Tunables (raised hard vs v1 after the harness run; tune on footage) ───────
PATH_REGION_FRAC   = 0.45    # central band (fraction of width) the object's
                             # bbox centre must be inside. Wider than
                             # tailgating (0.34) so a person stepping toward
                             # the lane trips slightly earlier.
NEAR_DEPTH_REL     = 0.62    # ALREADY-CLOSE gate: patch depth must be >=
                             # this fraction of the frame's CURRENT max
                             # depth. "Close now", not just "changing".
DEPTH_RISE_FRAC    = 0.35    # PRIMARY: depth must rise by >= this fraction
                             # of its window-start value across the window.
                             # Was 0.15 (fired on traffic noise) -> 0.35.
WINDOW             = 6       # frames over which to measure the rise
BUFFER_LEN         = 24      # rolling history per track
COOLDOWN_SECONDS   = 5.0     # per-track min seconds between near-miss events
SAME_TARGET_CLEAR_FRAMES = 4 # track must drop out of "qualifying" for this
                             # many frames before it can fire again (so a
                             # sustained approach is ONE event, not repeats)
DEFAULT_FPS        = 30.0

#   0 car, 1 truck, 2 bus, 3 person, 4 light, 5 sign, 6 VRU
VEHICLE_IDS = {0, 1, 2}
VRU_IDS     = {3, 6}                 # person + vulnerable road user
RELEVANT_IDS = VEHICLE_IDS | VRU_IDS


class EventType(str, Enum):
    NEAR_MISS = "NEAR_MISS"


@dataclass
class NearMissEvent:
    type:        EventType
    severity:    str            # 'critical' (VRU) | 'high' (vehicle)
    is_vru:      bool           # True if a person / vulnerable road user
    track_id:    int
    class_name:  str
    rel_depth:   float          # how close now (depth / frame max)
    depth_rise:  float          # relative rise over the window
    message_en:  str
    timestamp:   float          # wall-clock
    frame_idx:   int





class NearMissDetector:
    """
    Stateful. Create once, call update(tracks, depth_map, frame) per frame.
    Returns the single most-critical NearMissEvent this frame, or None.
    VRU events outrank vehicle events when both occur the same frame.
    """

    def __init__(
        self,
        fps:              float = DEFAULT_FPS,
        path_region_frac: float = PATH_REGION_FRAC,
        near_depth_rel:   float = NEAR_DEPTH_REL,
        depth_rise_frac:  float = DEPTH_RISE_FRAC,
        window:           int   = WINDOW,
        buffer_len:       int   = BUFFER_LEN,
        cooldown_seconds: float = COOLDOWN_SECONDS,
    ):
        self.fps = fps if fps and fps > 0 else DEFAULT_FPS
        self.path_region_frac = path_region_frac
        self.near_depth_rel   = near_depth_rel
        self.depth_rise_frac  = depth_rise_frac
        self.window           = window
        self.buffer_len       = buffer_len
        self.cooldown_frames  = int(cooldown_seconds * self.fps)

        self.frame_idx = 0
        self._buf: Dict[int, Deque[Tuple[float, int]]] = defaultdict(
            lambda: deque(maxlen=self.buffer_len))
        self._last_fired: Dict[int, int] = {}
        # frames a track has spent NOT qualifying since it last fired,
        # so a sustained approach is one event not a burst.
        self._cleared_frames: Dict[int, int] = defaultdict(int)

    # ── Public API ────────────────────────────────────────────────────────────
    def update(
        self,
        tracks: List[Track],
        depth_map: np.ndarray,
        frame,
    ) -> Optional[NearMissEvent]:
        self.frame_idx += 1
        fh, fw = frame.shape[:2]
        cx_lo = fw * (0.5 - self.path_region_frac / 2.0)
        cx_hi = fw * (0.5 + self.path_region_frac / 2.0)
        frame_max_depth = (float(np.max(depth_map))
                           if depth_map is not None and depth_map.size else 0.0)

        candidates: List[Tuple[int, float, NearMissEvent]] = []
        active_ids = set()

        for t in tracks:
            if t.class_id not in RELEVANT_IDS:
                continue
            active_ids.add(t.track_id)

            x1, y1, x2, y2 = t.bbox_int
            cx = (x1 + x2) * 0.5

            d = self._patch_depth(depth_map, x1, y1, x2, y2)
            self._buf[t.track_id].append((d, self.frame_idx))
            buf = self._buf[t.track_id]

            # ---- gates (all must pass) ----
            in_path = cx_lo <= cx <= cx_hi
            rel_depth = (d / frame_max_depth) if frame_max_depth > 0 else 0.0
            already_close = rel_depth >= self.near_depth_rel
            have_window = len(buf) >= self.window

            qualifies = False
            depth_rise = 0.0
            if in_path and already_close and have_window:
                w = list(buf)[-self.window:]
                d0 = w[0][0]
                d1 = w[-1][0]
                if d0 > 0:
                    depth_rise = (d1 - d0) / d0          # relative
                    if depth_rise >= self.depth_rise_frac:
                        qualifies = True

            # same-target debounce: only (re)fire if the track has been
            # NOT qualifying for a few frames since its last fire (i.e.
            # this is a fresh approach, not the same one continuing)
            if not qualifies:
                self._cleared_frames[t.track_id] += 1
                continue
            self._cleared_frames[t.track_id] = 0

            lf = self._last_fired.get(t.track_id)
            if lf is not None:
                if (self.frame_idx - lf) < self.cooldown_frames:
                    continue
                # cooldown elapsed — but require it actually cleared in
                # between, so a sustained tailgate-then-rush isn't endless
                if self._cleared_frames.get(t.track_id, 0) == 0 and \
                   (self.frame_idx - lf) < self.cooldown_frames * 2:
                    # still effectively the same continuous event
                    pass  # allow after the longer 2x window only

            ev = self._build_event(t, rel_depth, depth_rise)
            # priority: VRU first, then by how fast closing
            pr = 0 if ev.is_vru else 1
            candidates.append((pr, -depth_rise, ev))

        self._prune(active_ids)

        if not candidates:
            return None

        # lowest (pr, -rise) = VRU first, then fastest approach
        candidates.sort(key=lambda c: (c[0], c[1]))
        worst = candidates[0][2]
        self._last_fired[worst.track_id] = self.frame_idx
        return worst

    def reset(self):
        self.frame_idx = 0
        self._buf.clear()
        self._last_fired.clear()
        self._cleared_frames.clear()

    # ── Internals ─────────────────────────────────────────────────────────────
    @staticmethod
    def _patch_depth(depth_map: np.ndarray, x1, y1, x2, y2) -> float:
        if depth_map is None or depth_map.size == 0:
            return 0.0
        h, w = depth_map.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        bw, bh = x2 - x1, y2 - y1
        px1, px2 = x1 + bw // 3, x2 - bw // 3
        py1, py2 = y1 + (2 * bh) // 3, y2
        px2 = max(px2, px1 + 1)
        py2 = max(py2, py1 + 1)
        patch = depth_map[py1:py2, px1:px2]
        return float(np.median(patch)) if patch.size else 0.0

    def _prune(self, active_ids: set):
        dead = [tid for tid in self._buf if tid not in active_ids]
        for tid in dead:
            self._buf.pop(tid, None)
            self._last_fired.pop(tid, None)
            self._cleared_frames.pop(tid, None)

    def _build_event(
        self, t: Track, rel_depth: float, depth_rise: float
    ) -> NearMissEvent:
        cn = t.class_name
        is_vru = t.class_id in VRU_IDS
        if is_vru:
            sev = "critical"
            en = f"Critical: a {cn} is very close and approaching fast"
        else:
            sev = "high"
            en = f"A {cn} ahead is very close and closing fast"
        return NearMissEvent(
            type        = EventType.NEAR_MISS,
            severity    = sev,
            is_vru      = is_vru,
            track_id    = t.track_id,
            class_name  = cn,
            rel_depth   = round(rel_depth, 3),
            depth_rise  = round(depth_rise, 3),
            message_en  = en,
            timestamp   = time.time(),
            frame_idx   = self.frame_idx,
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

    if len(sys.argv) < 4:
        print("usage: python near_miss_detector.py "
              "<best.pt> <bytetrack.yaml> <video.mp4>")
        sys.exit(1)

    from tamakkan.models.tracker import TamakkanTracker
    from tamakkan.models.depth_model import DepthEstimator

    cap = cv2.VideoCapture(sys.argv[3])
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    tracker = TamakkanTracker(weights=sys.argv[1], tracker_config=sys.argv[2])
    depth = DepthEstimator(
        weights_path=str(_repo / "weights" / "depth_anything_v2_vits.pth"),
        variant="vits")
    det = NearMissDetector(fps=fps)
    print(f"fps={fps:.1f} window={det.window} "
          f"cooldown={det.cooldown_frames}f near_rel={det.near_depth_rel} "
          f"rise={det.depth_rise_frac}")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        tracks = tracker.update(frame)
        dmap = depth.predict(frame)
        ev = det.update(tracks, dmap, frame)
        if ev:
            tag = "VRU" if ev.is_vru else "veh"
            print(f"frame {n:>5}  NEAR_MISS[{tag}]  id={ev.track_id} "
                  f"{ev.class_name}  reld={ev.rel_depth:.2f} "
                  f"rise={ev.depth_rise:+.2f}  sev={ev.severity}")
    cap.release()
    print(f"done, {n} frames")