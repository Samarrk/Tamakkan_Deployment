"""
src/tamakkan/models/ocr_model.py
 
Speed-limit-sign reader for the Tamakkan pipeline (EasyOCR backend).
 
Role
----
TamakkanTracker already detects traffic_sign (class 5) and assigns each
physical sign a persistent track_id. This module is HANDED those signs
(frame + Track) and reads the number off them. It never runs its own
detector.
 
Why caching is keyed by track_id (the critical design point)
------------------------------------------------------------
The camera is on a moving car: a sign's pixel position changes every
frame as you approach it, so a position-based cache key almost never
hits and EasyOCR (the single most expensive op in the whole pipeline,
~100s of ms) ends up running repeatedly on the same physical sign.
 
ByteTrack gives each physical sign a stable track_id that survives the
sign moving across the frame. Caching by track_id means we OCR a given
physical sign essentially once, then reuse that result while the track
lives. This is what makes OCR affordable on the Jetson alongside the
other five models.
 
Design principles (same conventions as the other model wrappers)
----------------------------------------------------------------
- Single responsibility: read a number off a handed crop. No detection.
- Per-track caching + size gating so EasyOCR runs as rarely as possible.
- Dataclass return (SpeedSignReading), type hints, silent __init__,
  auto GPU detection, standalone smoke test.
"""
 
from __future__ import annotations
 
import re
import time
from dataclasses import dataclass
from typing import List, Optional
 
import cv2
import numpy as np
 
try:
    import torch
    _HAS_TORCH = True
except ImportError:  # torch should exist (other models need it) but stay safe
    _HAS_TORCH = False
 
import easyocr
 
 
# Plausible Saudi speed limits (km/h). Anything outside this set is treated
# as a misread and rejected. Confirmed range 20-120 with the user.
VALID_SPEEDS = {20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}
 
# Don't bother running EasyOCR on crops smaller than this (px). Tiny far
# signs are guaranteed-fail and OCR is the most expensive op in the system.
MIN_BOX_HEIGHT = 50
 
# How long a successful reading stays valid for a given track_id.
# A physical speed sign never changes its number, so once we've read it
# we can reuse that result for the life of the track. We still expire it
# so a recycled track_id (ByteTrack can reuse ids) can't go stale forever.
CACHE_TTL_SECONDS = 5.0
 
# EasyOCR: restrict the recognizer to digits only. Smaller search space =
# faster and more accurate than reading arbitrary text then stripping.
_OCR_ALLOWLIST = "0123456789"
 
 
@dataclass
class SpeedSignReading:
    """
    Result of attempting to read one traffic_sign track.
 
    speed     : the detected limit as an int (e.g. 80), or None if no
                confident valid reading is available.
    confidence: EasyOCR confidence in [0, 1] for the chosen reading
                (0.0 when speed is None).
    track_id  : which sign track this reading belongs to.
    from_cache: True if returned from the per-track cache (no OCR ran
                this call), False if a fresh OCR produced it.
    """
    speed: Optional[int]
    confidence: float
    track_id: int
    from_cache: bool
 
 
class SpeedSignOCR:
    """
    EasyOCR-based speed sign reader.
 
    Usage in the pipeline:
        ocr = SpeedSignOCR()                       # auto GPU
        for t in tracks:
            if t.is_traffic_sign:
                reading = ocr.read(frame, t)
                if reading.speed is not None:
                    ...                            # feed speed detector
    """
 
    def __init__(
        self,
        device: str | None = None,
        min_box_height: int = MIN_BOX_HEIGHT,
        cache_ttl_seconds: float = CACHE_TTL_SECONDS,
    ):
        # Auto-detect GPU. EasyOCR on CPU is seconds-per-call on Jetson —
        # never default to that silently.
        if device is None:
            use_gpu = _HAS_TORCH and torch.cuda.is_available()
        else:
            use_gpu = device != "cpu"
 
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)
        self.use_gpu = use_gpu
        self.min_box_height = min_box_height
        self.cache_ttl = cache_ttl_seconds
 
        # track_id -> {"speed": int, "confidence": float, "t": timestamp}
        self._cache: dict[int, dict] = {}
 
    # ── Public API ────────────────────────────────────────────────────────────
    def read(self, frame: np.ndarray, track) -> SpeedSignReading:
        """
        Read the speed off one traffic_sign track.
 
        Args:
            frame: full BGR frame (H, W, 3).
            track: a Track from TamakkanTracker. Must expose .track_id and
                   .bbox_int (x1, y1, x2, y2). Should be a traffic_sign;
                   the caller is responsible for that filtering.
 
        Returns:
            SpeedSignReading. speed is None when there is no confident
            valid reading for this track yet.
        """
        tid = int(track.track_id)
        now = time.time()
 
        # 1. Fresh cache hit → no OCR.
        cached = self._cache.get(tid)
        if cached is not None and (now - cached["t"]) <= self.cache_ttl:
            return SpeedSignReading(
                speed=cached["speed"],
                confidence=cached["confidence"],
                track_id=tid,
                from_cache=True,
            )
 
        # 2. Crop + sanity/size gates before touching EasyOCR.
        if frame is None or frame.size == 0:
            return self._none(tid)
 
        x1, y1, x2, y2 = track.bbox_int
        h_f, w_f = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f, x2), min(h_f, y2)
        if x2 <= x1 or y2 <= y1:
            return self._none(tid)
 
        if (y2 - y1) < self.min_box_height:
            # Too small to read reliably; don't waste an OCR call.
            return self._none(tid)
 
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return self._none(tid)
 
        # 3. OCR with multi-variant preprocessing (stop at first valid).
        speed, conf = self._ocr_crop(crop)
 
        if speed is not None:
            self._cache[tid] = {"speed": speed, "confidence": conf, "t": now}
            return SpeedSignReading(speed, conf, tid, from_cache=False)
 
        # No valid reading this attempt. If we have an unexpired prior
        # reading for this track, keep returning it rather than dropping
        # to None on a single bad frame.
        if cached is not None and (now - cached["t"]) <= self.cache_ttl:
            return SpeedSignReading(
                speed=cached["speed"],
                confidence=cached["confidence"],
                track_id=tid,
                from_cache=True,
            )
 
        return self._none(tid)
 
    def reset(self):
        """Clear the per-track cache. Call between unrelated clips/sessions."""
        self._cache.clear()
 
    def prune(self, live_track_ids: set[int]):
        """
        Optional housekeeping the pipeline can call periodically: drop
        cache entries for tracks that no longer exist, so the dict can't
        grow unbounded over a long drive.
        """
        dead = [tid for tid in self._cache if tid not in live_track_ids]
        for tid in dead:
            del self._cache[tid]
 
    # ── Internals ─────────────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(crop: np.ndarray) -> List[np.ndarray]:
        """
        Return several processed versions of the crop, cheapest-likely-to-
        work first: Otsu threshold, inverted Otsu (light-on-dark signs),
        plain grayscale. Small crops get a bigger upscale.
        """
        scale = 3.0 if crop.shape[0] < 80 else 2.0
        big = cv2.resize(crop, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [th, cv2.bitwise_not(th), gray]
 
    def _ocr_crop(self, crop: np.ndarray) -> tuple[Optional[int], float]:
        """
        Try each preprocessing variant. Return (speed_int, confidence)
        for the first variant that yields a whitelisted speed, else
        (None, 0.0).
        """
        for img in self._preprocess(crop):
            # detail=1 → list of (bbox, text, confidence)
            results = self.reader.readtext(
                img, detail=1, allowlist=_OCR_ALLOWLIST
            )
            if not results:
                continue
 
            # Concatenate detected digit chunks, keep best confidence.
            digits = "".join(
                re.sub(r"[^0-9]", "", txt) for (_, txt, _) in results
            )
            if not digits:
                continue
 
            try:
                value = int(digits)
            except ValueError:
                continue
 
            if value in VALID_SPEEDS:
                conf = max((c for (_, _, c) in results), default=0.0)
                return value, float(conf)
 
        return None, 0.0
 
    @staticmethod
    def _none(track_id: int) -> SpeedSignReading:
        return SpeedSignReading(
            speed=None, confidence=0.0, track_id=track_id, from_cache=False
        )
 
 
# ── Visualization helper (drawing only; used by test scripts) ─────────────────
def draw_speed(
    frame: np.ndarray,
    reading: SpeedSignReading,
    track,
) -> np.ndarray:
    """
    Draw a sign box + speed label for one reading. Mutates and returns
    `frame` (caller passes a copy if it wants the original preserved).
    Ported from the user's draw_alert(), trimmed to per-sign drawing.
    """
    if reading.speed is None:
        return frame
 
    x1, y1, x2, y2 = track.bbox_int
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 80), 3)
    label = f"{reading.speed} km/h"
    cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 200, 80), 2, cv2.LINE_AA)
    return frame
 
 
# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
 
    if len(sys.argv) < 2:
        print("usage: python ocr_model.py <sign_crop_image>")
        print("  (pass a tight crop of a speed-limit sign)")
        sys.exit(1)
 
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"could not read image: {sys.argv[1]}")
        sys.exit(1)
 
    # Minimal stand-in for a Track so we can exercise read() directly.
    class _FakeTrack:
        track_id = 1
        @property
        def bbox_int(self):
            h, w = img.shape[:2]
            return (0, 0, w, h)
 
    ocr = SpeedSignOCR()
    print(f"EasyOCR initialized, gpu={ocr.use_gpu}")
 
    import time as _t
    t0 = _t.time()
    r1 = ocr.read(img, _FakeTrack())
    dt1 = (_t.time() - t0) * 1000
 
    t0 = _t.time()
    r2 = ocr.read(img, _FakeTrack())     # should be a cache hit
    dt2 = (_t.time() - t0) * 1000
 
    print(f"first  read: speed={r1.speed} conf={r1.confidence:.2f} "
          f"cache={r1.from_cache}  ({dt1:.0f} ms)")
    print(f"second read: speed={r2.speed} conf={r2.confidence:.2f} "
          f"cache={r2.from_cache}  ({dt2:.0f} ms)  <- cache should make this ~0ms")