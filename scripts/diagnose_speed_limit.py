"""
scripts/diagnose_speed_limit.py

Why isn't the live pipeline producing a confirmed speed-limit reading
when the clip clearly has a sign?

This script runs the same components the pipeline uses, but with verbose
per-frame logging at every gate, so we can see WHERE the chain breaks:

  1. YOLO + ByteTrack — does it detect traffic_sign at all? How big?
  2. SpeedSignOCR — when called, what does it return per-track?
  3. SpeedLimitReader — what's its decision logic seeing each frame?

Usage:
    python scripts/diagnose_speed_limit.py <video_path>
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Make src/ and third_party/ importable.
_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO / "third_party"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tamakkan.models.tracker import TamakkanTracker
from tamakkan.models.ocr_model import SpeedSignOCR
from tamakkan.detectors.speed_limit_reader import SpeedLimitReader


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/diagnose_speed_limit.py <video_path>")
        sys.exit(1)

    video = sys.argv[1]
    weights_dir = _REPO / "weights"

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"could not open {video}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {video}")
    print(f"fps={fps:.1f}  frames={total}")
    print()

    tracker = TamakkanTracker(
        weights=str(weights_dir / "best.pt"),
        tracker_config=str(weights_dir / "bytetrack_tamakkan.yaml"),
    )
    ocr = SpeedSignOCR()
    reader = SpeedLimitReader(ocr, frame_skip=1)   # NO skip — see everything

    # Stats trackers
    sign_frames    = 0          # frames where at least one traffic_sign was seen
    sign_seen_ids  = set()      # track_ids assigned to traffic_signs
    too_small      = 0          # signs rejected by MIN_BOX_HEIGHT
    ocr_called     = 0          # actual OCR invocations (fresh, not cache hits)
    ocr_returned_speed = 0      # OCR returned a non-None speed
    ocr_low_conf   = 0          # speed read but below CONFIDENCE_MIN_NEW
    cache_hits     = 0          # SpeedSignReading came from cache
    confirmed      = 0          # reader actually emitted a SpeedLimitChange

    # Track-by-track read history
    per_track_reads: dict = {}   # tid -> list of (frame_idx, speed, conf, cache)

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        tracks = tracker.update(frame)

        signs = [t for t in tracks if t.is_traffic_sign]
        if signs:
            sign_frames += 1
            for t in signs:
                sign_seen_ids.add(t.track_id)

        # Mimic the reader's per-frame work for each sign so we can see
        # the per-sign details (the reader itself only calls OCR on the
        # biggest, so we'd miss smaller signs in production-by-design).
        for t in signs:
            x1, y1, x2, y2 = t.bbox_int
            h_px = y2 - y1
            if h_px < ocr.min_box_height:
                too_small += 1
                continue

            reading = ocr.read(frame, t)
            if reading.from_cache:
                cache_hits += 1
            else:
                ocr_called += 1
                if reading.speed is not None:
                    ocr_returned_speed += 1
                    if reading.confidence < reader.confidence_min_new:
                        ocr_low_conf += 1

            per_track_reads.setdefault(t.track_id, []).append(
                (n, h_px, reading.speed, reading.confidence, reading.from_cache)
            )

        # Now actually run the reader (which only OCRs the biggest sign).
        change = reader.update(tracks, frame)
        if change is not None:
            confirmed += 1
            print(f"frame {n:>5}  CONFIRMED  limit -> {change.limit_kmh} km/h")

    cap.release()

    # ── Report ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  total frames processed:        {n}")
    print(f"  frames with at least 1 sign:   {sign_frames}")
    print(f"  unique sign track_ids seen:    {len(sign_seen_ids)}")
    print()
    print(f"  signs too small (< {ocr.min_box_height}px):       {too_small}")
    print(f"  fresh OCR calls:                {ocr_called}")
    print(f"    of those, returned a speed:   {ocr_returned_speed}")
    print(f"    of those, low confidence:     {ocr_low_conf}")
    print(f"  cache hits (no OCR):            {cache_hits}")
    print()
    print(f"  SpeedLimitChange emitted:       {confirmed}")
    print()
    print("Per-track read history:")
    for tid, rows in per_track_reads.items():
        print(f"  track_id={tid}  ({len(rows)} entries):")
        for row in rows[:12]:    # show first 12 entries per track
            frame_n, h_px, speed, conf, fc = row
            cache_tag = " [CACHE]" if fc else ""
            speed_str = f"speed={speed}" if speed is not None else "speed=None"
            print(f"    f{frame_n:>5}  h={h_px:>3}px  {speed_str:<12s}  "
                  f"conf={conf:.2f}{cache_tag}")
        if len(rows) > 12:
            print(f"    ... ({len(rows) - 12} more entries)")
    print()
    print("Diagnosis hints:")
    if sign_frames == 0:
        print("  → YOLO never detected a traffic_sign. Either the model")
        print("    doesn't generalize to this clip, or the sign isn't")
        print("    visible. Check by running tracker standalone with viz.")
    elif too_small > 0 and ocr_called == 0:
        print("  → Signs detected but all rejected as too small. The bbox")
        print("    never reached MIN_BOX_HEIGHT=50px. Solutions: lower the")
        print("    threshold, or re-crop input frames bigger.")
    elif ocr_returned_speed == 0:
        print("  → OCR ran but never returned a valid speed value.")
        print("    Could be EasyOCR not recognising digits, or the value")
        print("    didn't match VALID_SPEEDS={20..120}.")
    elif ocr_low_conf == ocr_returned_speed:
        print("  → OCR read values but all below confidence threshold")
        print(f"    ({reader.confidence_min_new}). Lower CONFIDENCE_MIN_NEW.")
    elif confirmed == 0:
        print("  → OCR worked but CONFIRM_READS=2 wasn't met. Either the")
        print("    track ID flipped, or only one fresh non-cache read per id.")
    else:
        print("  → SpeedLimitChange was emitted. Pipeline integration is")
        print("    fine; the original failure was elsewhere.")


if __name__ == "__main__":
    main()