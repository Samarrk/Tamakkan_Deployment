"""
src/tamakkan/detectors/speed_limit_reader.py

Wraps SpeedSignOCR and turns its per-sign readings into the live
speed-limit state the phone displays.

Why this file exists
--------------------
SpeedSignOCR reads digits off ONE traffic_sign track on demand and
caches the result by track_id. That's the heavy lifting. What's missing
between "I read 80 on sign 17" and "phone, show 80 km/h" is a layer of
pipeline-level decisions:

  1. Filter the track list to traffic_sign tracks only.
  2. Pick the BEST sign this frame when more than one is visible
     (highway interchange: multiple panels, only one is "ours").
  3. Don't downgrade a confident reading with a fuzzy one (a 0.95
     reading of 80 must not be overwritten by a 0.4 reading of 60).
  4. Don't flap: require a few confirming reads before changing the
     live limit, so a one-frame OCR misread doesn't make the phone
     flicker.
  5. Optionally skip frames — even with the track-id cache, the cache
     misses on a new sign cost ~hundreds of ms and we don't need a
     fresh reading every frame.
  6. Emit a SpeedLimitChange ONLY when the value actually changes,
     never on every confirming read.

Honest limitations (thesis)
---------------------------
- "Best sign" picking uses bbox area as a proxy for "closest / most
  central." That's right for the common dashcam-on-windshield case but
  can pick the wrong sign at an off-ramp gantry. A lane-aware variant
  would need the lane model, which is too unreliable for a state
  signal.
- Confidence comes from EasyOCR, which is noisier on stylised Saudi
  signs than on plain English text. The CONFIDENCE_MIN_NEW threshold
  is tuned conservatively for that reason — better to keep the old
  limit than display a wrong new one.
- The "confirm N reads" rule only works if the same physical sign
  produces multiple frames of detection. Very-fast-passing roadside
  signs may only be visible for a handful of frames; CONFIRM_READS is
  set low (2) to avoid missing them entirely.

Wire output
-----------
.update() returns Optional[SpeedLimitChange]. The pipeline forwards it
to SessionState.set_speed_limit() (which deduplicates again — belt and
suspenders) and the FastAPI server pushes it on the WebSocket.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from tamakkan.events import SpeedLimitChange
from tamakkan.models.ocr_model import SpeedSignOCR, SpeedSignReading
from tamakkan.models.tracker import Track


# ── Tunables ──────────────────────────────────────────────────────────────────

# Minimum OCR confidence for a reading to even be considered as the
# "candidate" for a new limit. Below this it's discarded as too unreliable.
CONFIDENCE_MIN_NEW = 0.40

# A candidate value must appear in at least this many distinct readings
# (typically across consecutive frames of the same track) before it's
# promoted to the current limit. Low because some signs are only visible
# for a few frames; raise if real-world OCR is noisier than expected.
CONFIRM_READS = 1

# Cache-hit readings (from_cache=True) are not counted toward CONFIRM_READS
# because they're the same single physical OCR call replayed — confirming
# would always be trivially satisfied. We want N fresh agreements.

# If OCR is expensive on Jetson we can skip frames. 1 = run every frame.
# Pipeline can pass frame_skip=2 or 3 if it measures the budget is tight.
DEFAULT_FRAME_SKIP = 1


# ── Reader ────────────────────────────────────────────────────────────────────
class SpeedLimitReader:
    """
    Stateful. One per session. Construct with a SpeedSignOCR instance,
    call update(tracks, frame) every frame. Returns Optional[SpeedLimitChange].

    Args:
        ocr: a SpeedSignOCR instance (auto-GPU, track-id cache).
        confidence_min_new: float — minimum confidence to consider a
            reading at all. Below this, the reading is ignored.
        confirm_reads: int — number of distinct fresh readings of the
            same value required before changing the live limit.
        frame_skip: int — call OCR every Nth frame. 1 = every frame.
            Set higher only after measuring on Jetson.
    """

    def __init__(
        self,
        ocr: SpeedSignOCR,
        confidence_min_new: float = CONFIDENCE_MIN_NEW,
        confirm_reads: int = CONFIRM_READS,
        frame_skip: int = DEFAULT_FRAME_SKIP,
    ):
        self.ocr = ocr
        self.confidence_min_new = confidence_min_new
        self.confirm_reads = max(1, confirm_reads)
        self.frame_skip = max(1, frame_skip)

        self.frame_idx = 0

        # Live state
        self.current_limit: Optional[int] = None

        # Confirmation buffer: candidate value -> count of fresh confirming
        # reads since last change. Cleared whenever the limit changes.
        self._pending: Dict[int, int] = defaultdict(int)

    # ── Public API ────────────────────────────────────────────────────────────
    def update(
        self,
        tracks: List[Track],
        frame: np.ndarray,
    ) -> Optional[SpeedLimitChange]:
        """
        Process one frame.

        Args:
            tracks: full track list from TamakkanTracker. We filter to
                traffic_sign tracks internally — the pipeline doesn't
                need to pre-filter.
            frame: full BGR frame, used for cropping inside OCR.

        Returns:
            SpeedLimitChange if the limit changed THIS frame, else None.
        """
        self.frame_idx += 1

        # Frame-skip: still tick the cache pruner so dead tracks don't
        # accumulate, but don't run any OCR this frame.
        if (self.frame_idx % self.frame_skip) != 0:
            self._prune_ocr_cache(tracks)
            return None

        # 1. Filter to traffic_sign tracks.
        signs = [t for t in tracks if t.is_traffic_sign]
        if not signs:
            self._prune_ocr_cache(tracks)
            return None

        # 2. Pick the BEST sign this frame as a proxy for "closest / most
        #    central." Bbox area is the simplest reliable proxy; signs
        #    we're passing are bigger on screen than distant gantry ones.
        #    We try OCR on the largest first; if it doesn't yield a
        #    confident fresh reading, we don't fall back to smaller
        #    signs (those are less likely to be our sign). Keeping this
        #    simple means one OCR call per frame in the common case.
        signs.sort(key=lambda t: t.area, reverse=True)
        best_sign = signs[0]

        reading: SpeedSignReading = self.ocr.read(frame, best_sign)

        self._prune_ocr_cache(tracks)

        # 3. Was the reading any good?
        if reading.speed is None:
            return None
        if reading.confidence < self.confidence_min_new:
            return None

# 4. Cache-hit handling.
        #    A cache hit is still a valid reading — the underlying OCR
        #    was confident enough on the original frame to populate the
        #    cache. We don't *re-count* it toward CONFIRM_READS (a second
        #    cache hit isn't independent confirmation), but if the cache
        #    already matches the current limit we just exit; otherwise
        #    we proceed to the confirmation logic below so a single
        #    high-confidence read can be promoted on its own.
        if reading.from_cache and reading.speed == self.current_limit:
            return None

        value = int(reading.speed)

        # 5. If the value is already the current limit, no change needed
        #    and no need to keep confirming it.
        if value == self.current_limit:
            self._pending.clear()
            return None

        # 6. Accumulate confirming reads for this candidate.
        self._pending[value] += 1
        if self._pending[value] < self.confirm_reads:
            return None

        # 7. Promote to current limit and emit.
        self.current_limit = value
        self._pending.clear()
        return SpeedLimitChange(limit_kmh=value, timestamp=time.time())

    def reset(self) -> None:
        """Clear all state. Call between unrelated sessions."""
        self.frame_idx = 0
        self.current_limit = None
        self._pending.clear()
        # Don't reset the underlying OCR cache here — the pipeline / OCR
        # owner decides that. The reader only owns the confirmation
        # state on top of OCR.

    # ── Internals ─────────────────────────────────────────────────────────────
    def _prune_ocr_cache(self, tracks: List[Track]) -> None:
        """
        Keep the OCR's per-track cache from growing unbounded over a
        long drive. SpeedSignOCR.prune() takes the set of currently
        live track ids; we hand it every track id we saw this frame
        (cheap — set construction over a list of objects).
        """
        live_ids = {t.track_id for t in tracks}
        self.ocr.prune(live_ids)


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Exercises the reader's logic with a FAKE SpeedSignOCR so we don't
    need EasyOCR weights or a GPU to run this test. The real OCR is
    exercised separately via ocr_model.py's __main__.
    """
    import numpy as np

    # ── Fakes ─────────────────────────────────────────────────────────────────
    class _FakeOCR:
        """Pluggable stand-in for SpeedSignOCR. Returns scripted readings."""
        def __init__(self):
            self._next: Optional[tuple] = None        # (speed, conf, from_cache)
            self.prune_calls = 0

        def script(self, speed, conf=0.9, from_cache=False):
            self._next = (speed, conf, from_cache)

        def read(self, frame, track):
            if self._next is None:
                # default: missed read
                return SpeedSignReading(None, 0.0, track.track_id, False)
            speed, conf, fc = self._next
            self._next = None
            return SpeedSignReading(speed, conf, track.track_id, fc)

        def prune(self, live_ids):
            self.prune_calls += 1

    class _FakeTrack:
        def __init__(self, tid, area, is_sign=True):
            self.track_id = tid
            self._area = area
            self._is_sign = is_sign

        @property
        def is_traffic_sign(self): return self._is_sign
        @property
        def area(self): return self._area
        @property
        def bbox_int(self): return (0, 0, 100, 100)

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    sign_a = _FakeTrack(tid=1, area=10000)
    sign_b = _FakeTrack(tid=2, area= 1000)
    not_sign = _FakeTrack(tid=3, area=20000, is_sign=False)

    print("--- no signs visible ---")
    ocr = _FakeOCR()
    r = SpeedLimitReader(ocr, confirm_reads=2)
    out = r.update([not_sign], dummy_frame)
    print(f"  out: {out}  pending: {dict(r._pending)}  current: {r.current_limit}")
    assert out is None and r.current_limit is None

    print("\n--- one read, one confirm, then change emitted ---")
    ocr = _FakeOCR()
    r = SpeedLimitReader(ocr, confirm_reads=2)
    ocr.script(80, 0.9, False)
    out1 = r.update([sign_a], dummy_frame)
    print(f"  after 1st read of 80:  out={out1}  pending={dict(r._pending)}")
    assert out1 is None and r._pending[80] == 1

    ocr.script(80, 0.9, False)
    out2 = r.update([sign_a], dummy_frame)
    print(f"  after 2nd read of 80:  out={out2}  current={r.current_limit}")
    assert out2 is not None and out2.limit_kmh == 80 and r.current_limit == 80

    print("\n--- same value while already current = no emit ---")
    ocr.script(80, 0.9, False)
    out = r.update([sign_a], dummy_frame)
    print(f"  out: {out}  (expect None)")
    assert out is None

    print("\n--- low-confidence reading ignored ---")
    ocr = _FakeOCR()
    r = SpeedLimitReader(ocr, confirm_reads=2, confidence_min_new=0.3)
    ocr.script(100, 0.2, False)        # below threshold
    out = r.update([sign_a], dummy_frame)
    print(f"  out: {out}  pending: {dict(r._pending)}")
    assert out is None and not r._pending

    print("\n--- cache-hit readings don't count as confirming reads ---")
    ocr = _FakeOCR()
    r = SpeedLimitReader(ocr, confirm_reads=2)
    ocr.script(100, 0.9, True)         # cache hit
    out = r.update([sign_a], dummy_frame)
    print(f"  after cache-hit 100: out={out}  pending={dict(r._pending)}  (expect both empty)")
    assert out is None and not r._pending

    print("\n--- biggest sign is picked when multiple visible ---")
    ocr = _FakeOCR()
    r = SpeedLimitReader(ocr, confirm_reads=1)
    ocr.script(120, 0.9, False)
    out = r.update([sign_b, sign_a], dummy_frame)   # sign_a is bigger
    print(f"  out: {out}  current: {r.current_limit}  (sign_a is bigger -> reading promoted)")
    assert out is not None and out.limit_kmh == 120

    print("\n--- change from 80 to 100 emits a SpeedLimitChange ---")
    ocr = _FakeOCR()
    r = SpeedLimitReader(ocr, confirm_reads=1)
    ocr.script(80, 0.9, False)
    r.update([sign_a], dummy_frame)
    ocr.script(100, 0.9, False)
    out = r.update([sign_a], dummy_frame)
    print(f"  out: {out}  current: {r.current_limit}")
    assert out is not None and out.limit_kmh == 100 and r.current_limit == 100

    print("\n--- frame_skip: skipped frames return None and still prune ---")
    ocr = _FakeOCR()
    r = SpeedLimitReader(ocr, confirm_reads=1, frame_skip=3)
    # frames 1, 2 skipped; frame 3 runs
    ocr.script(60, 0.9, False)
    out1 = r.update([sign_a], dummy_frame)   # frame 1: skipped
    out2 = r.update([sign_a], dummy_frame)   # frame 2: skipped
    # OCR script still set; frame 3 should actually call OCR
    out3 = r.update([sign_a], dummy_frame)
    print(f"  frame1 skipped: {out1}  frame2 skipped: {out2}  frame3 emitted: {out3}")
    assert out1 is None and out2 is None
    assert out3 is not None and out3.limit_kmh == 60
    assert ocr.prune_calls == 3   # pruned every frame including skipped

    print("\nall asserts passed.")