"""
scripts/test_models_video.py
 
Run all 5 finalized Tamakkan models over one or more videos and produce
separate annotated output videos per model:
 
    <name>_tracker.mp4   YOLO+ByteTrack boxes, track IDs, class, confidence
    <name>_lane.mp4      UFLD-v2 lane lines
    <name>_light.mp4     traffic-light boxes + red/green/unknown labels
    <name>_depth.mp4     Depth Anything V2 heatmap
    <name>_ocr.mp4       speed-sign boxes + read speed (km/h)
 
light and ocr both consume the tracker's Track objects (lights/signs come
from the tracker, not their own detector), so skipping the tracker also
skips light and ocr.
 
This is a MODEL sanity test, not the pipeline. No alerts / tailgating /
lane-departure / speed-violation logic here — those are the detectors,
reviewed separately. The point is to SEE whether each model perceives the
road correctly.
 
Usage (run from repo root, inside the venv with torch+ultralytics+easyocr):
    python scripts/test_models_video.py                    # ALL mp4s in scripts/
    python scripts/test_models_video.py scripts/clip.mp4   # just that one
    python scripts/test_models_video.py --skip depth       # all mp4s, no depth
    python scripts/test_models_video.py clip.mp4 --skip depth lane
 
Notes
-----
- With no path, EVERY .mp4 in scripts/ is processed (our own _tracker/_lane/
  _light/_depth/_ocr outputs are skipped automatically).
- Models are loaded ONCE and reused across all videos (load is the slow part).
- The OCR per-track cache and the lane smoother are reset between videos so
  state from one clip can't leak into another.
- PC numpy is 2.x vs Jetson 1.24 — a clean run here is strong evidence but
  not a 100% guarantee of identical Jetson behaviour.
"""
 
from __future__ import annotations
 
import argparse
import sys
import time
from pathlib import Path
 
import numpy as np
 
# ──────────────────────────────────────────────────────────────────────────────
# Path setup — make tamakkan.* and third_party.* importable from anywhere.
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
WEIGHTS_DIR = REPO_ROOT / "weights"
SCRIPTS_DIR = Path(__file__).resolve().parent
 
for p in (REPO_ROOT, SRC_DIR, THIRD_PARTY_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
 
import cv2  # noqa: E402
 
# Tags we append to our own outputs — never treat these as inputs.
_OUTPUT_TAGS = ("_tracker", "_lane", "_light", "_depth", "_ocr")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Video discovery
# ──────────────────────────────────────────────────────────────────────────────
def discover_videos(cli_arg: str | None) -> list[Path]:
    """
    If a path is given: return just that file (if it exists).
    Otherwise: return EVERY .mp4 in scripts/ that isn't one of our own
    generated output videos.
    """
    if cli_arg:
        p = Path(cli_arg)
        if p.exists():
            return [p]
        print(f"ERROR: video not found: {cli_arg}")
        return []
 
    return sorted(
        v for v in SCRIPTS_DIR.glob("*.mp4")
        if not any(tag in v.name for tag in _OUTPUT_TAGS)
    )
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────
def color_for_class(class_name: str) -> tuple[int, int, int]:
    palette = {
        "car":                  (80, 200, 80),
        "truck":                (80, 160, 255),
        "bus":                  (60, 120, 255),
        "person":               (255, 120, 80),
        "vulnerable_road_user": (255, 80, 200),
        "traffic_light":        (60, 60, 240),
        "traffic_sign":         (240, 200, 60),
    }
    return palette.get(class_name, (200, 200, 200))
 
 
def put_label(img, text, x, y, color, scale=0.5):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)
 
 
class VideoWriterLazy:
    """Opens the writer on the first frame so we match the exact size."""
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = fps if fps and fps > 0 else 30.0
        self.writer = None
 
    def write(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc,
                                          self.fps, (w, h))
        self.writer.write(frame)
 
    def release(self):
        if self.writer is not None:
            self.writer.release()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Per-video processing
# ──────────────────────────────────────────────────────────────────────────────
def process_one_video(video_path: Path, models: dict, flags: dict,
                       lane_viz, draw_speed):
    do_tracker = flags["tracker"]
    do_lane = flags["lane"]
    do_light = flags["light"]
    do_depth = flags["depth"]
    do_ocr = flags["ocr"]
 
    tracker = models.get("tracker")
    lane = models.get("lane")
    light = models.get("light")
    depth = models.get("depth")
    ocr = models.get("ocr")
 
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path.name}, skipping")
        return
 
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n--- {video_path.name}  ({total} frames @ {fps:.1f} fps) ---")
 
    stem = video_path.with_suffix("")
    w_tracker = VideoWriterLazy(Path(f"{stem}_tracker.mp4"), fps) if do_tracker else None
    w_lane = VideoWriterLazy(Path(f"{stem}_lane.mp4"), fps) if do_lane else None
    w_light = VideoWriterLazy(Path(f"{stem}_light.mp4"), fps) if do_light else None
    w_depth = VideoWriterLazy(Path(f"{stem}_depth.mp4"), fps) if do_depth else None
    w_ocr = VideoWriterLazy(Path(f"{stem}_ocr.mp4"), fps) if do_ocr else None
 
    # Reset stateful models so nothing leaks between clips.
    if lane is not None:
        lane.reset()
    if ocr is not None:
        ocr.reset()
    if tracker is not None:
        tracker.reset()
 
    t_tracker = t_lane = t_light = t_depth = t_ocr = 0.0
    n = 0
    start = time.time()
 
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
 
        tracks = []
        if tracker is not None:
            t0 = time.time()
            tracks = tracker.update(frame)
            t_tracker += time.time() - t0
 
        # ---- tracker video ----
        if w_tracker is not None:
            canvas = frame.copy()
            for tr in tracks:
                x1, y1, x2, y2 = tr.bbox_int
                c = color_for_class(tr.class_name)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), c, 2)
                put_label(canvas,
                          f"#{tr.track_id} {tr.class_name} {tr.confidence:.2f}",
                          x1, max(y1 - 6, 14), c, 0.5)
            put_label(canvas, f"frame {n}/{total}  tracks {len(tracks)}",
                      10, 24, (255, 255, 255), 0.6)
            w_tracker.write(canvas)
 
        # ---- lane video ----
        if w_lane is not None:
            t0 = time.time()
            lanes = lane.update(frame)
            t_lane += time.time() - t0
            lane_canvas = lane_viz(frame, lanes, show_roi=True)
            put_label(lane_canvas, f"frame {n}/{total}  lanes {len(lanes)}",
                      10, 24, (255, 255, 255), 0.6)
            w_lane.write(lane_canvas)
 
        # ---- light video ----
        if w_light is not None:
            t0 = time.time()
            light_canvas = frame.copy()
            n_lights = 0
            for tr in tracks:
                if not tr.is_traffic_light:
                    continue
                n_lights += 1
                x1, y1, x2, y2 = tr.bbox_int
                crop = frame[y1:y2, x1:x2]
                res = light.classify(crop)
                col = {"red": (0, 0, 255),
                       "green": (0, 200, 0),
                       "unknown": (160, 160, 160)}.get(res.color, (160, 160, 160))
                cv2.rectangle(light_canvas, (x1, y1), (x2, y2), col, 2)
                put_label(light_canvas,
                          f"{res.color} {res.confidence:.2f}",
                          x1, max(y1 - 6, 14), col, 0.5)
            t_light += time.time() - t0
            put_label(light_canvas, f"frame {n}/{total}  lights {n_lights}",
                      10, 24, (255, 255, 255), 0.6)
            w_light.write(light_canvas)
 
        # ---- depth video ----
        if w_depth is not None:
            t0 = time.time()
            dmap = depth.predict(frame)
            t_depth += time.time() - t0
            heat = depth.colorize(dmap)
            put_label(heat, f"frame {n}/{total}", 10, 24,
                      (255, 255, 255), 0.6)
            w_depth.write(heat)
 
        # ---- ocr video ----
        if w_ocr is not None:
            t0 = time.time()
            ocr_canvas = frame.copy()
            n_signs = 0
            last_speed = None
            for tr in tracks:
                if not tr.is_traffic_sign:
                    continue
                n_signs += 1
                reading = ocr.read(frame, tr)
                x1, y1, x2, y2 = tr.bbox_int
                if reading.speed is not None:
                    draw_speed(ocr_canvas, reading, tr)
                    tag = "cache" if reading.from_cache else "ocr"
                    put_label(ocr_canvas,
                              f"#{tr.track_id} {reading.speed} ({tag})",
                              x1, max(y1 - 28, 14), (0, 200, 80), 0.5)
                    last_speed = reading.speed
                else:
                    cv2.rectangle(ocr_canvas, (x1, y1), (x2, y2),
                                  (120, 120, 120), 2)
                    put_label(ocr_canvas, f"#{tr.track_id} sign ?",
                              x1, max(y1 - 6, 14), (120, 120, 120), 0.5)
            t_ocr += time.time() - t0
            banner = f"frame {n}/{total}  signs {n_signs}"
            if last_speed is not None:
                banner += f"  last read: {last_speed} km/h"
            put_label(ocr_canvas, banner, 10, 24, (255, 255, 255), 0.6)
            w_ocr.write(ocr_canvas)
 
        # housekeeping: keep the OCR cache from growing unbounded
        if ocr is not None and n % 60 == 0:
            ocr.prune({int(t.track_id) for t in tracks})
 
        if n % 25 == 0 or n == total:
            elapsed = time.time() - start
            pfps = n / elapsed if elapsed > 0 else 0
            eta = (total - n) / pfps if pfps > 0 else 0
            print(f"  frame {n:>5}/{total}  proc {pfps:4.1f} FPS  "
                  f"eta {int(eta // 60)}m{int(eta % 60):02d}s")
 
    cap.release()
    for wtr in (w_tracker, w_lane, w_light, w_depth, w_ocr):
        if wtr is not None:
            wtr.release()
 
    dur = time.time() - start
    print(f"  done: {n} frames in {dur:.1f}s ({n/dur:.1f} FPS)")
    if n:
        if do_tracker: print(f"    tracker {t_tracker/n*1000:6.1f} ms/frame")
        if do_lane:    print(f"    lane    {t_lane/n*1000:6.1f} ms/frame")
        if do_light:   print(f"    light   {t_light/n*1000:6.1f} ms/frame")
        if do_depth:   print(f"    depth   {t_depth/n*1000:6.1f} ms/frame")
        if do_ocr:     print(f"    ocr     {t_ocr/n*1000:6.1f} ms/frame")
    outs = []
    for wtr, label in ((w_tracker, "tracker"), (w_lane, "lane"),
                        (w_light, "light"), (w_depth, "depth"),
                        (w_ocr, "ocr")):
        if wtr is not None:
            outs.append(f"{label}->{wtr.path.name}")
    print("    outputs: " + "  ".join(outs))
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None,
                    help="input video path; if omitted, ALL mp4s in scripts/")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["tracker", "lane", "light", "depth", "ocr"],
                    help="models to skip, e.g. --skip depth ocr")
    args = ap.parse_args()
 
    videos = discover_videos(args.video)
    if not videos:
        print("No input videos found. Put .mp4 files in scripts/ or pass a path.")
        return
 
    skip = set(args.skip)
    do_tracker = "tracker" not in skip
    do_lane = "lane" not in skip
    do_light = "light" not in skip and do_tracker   # needs tracker boxes
    do_depth = "depth" not in skip
    do_ocr = "ocr" not in skip and do_tracker        # needs tracker boxes
 
    if not do_tracker:
        if "light" not in skip:
            print("NOTE: light needs the tracker; auto-skipped.")
        if "ocr" not in skip:
            print("NOTE: ocr needs the tracker; auto-skipped.")
 
    print("=" * 66)
    print("  Tamakkan full-video model test")
    print("=" * 66)
    print(f"videos to process: {len(videos)}")
    for v in videos:
        print(f"  - {v.name}")
    print(f"models: "
          f"{'tracker ' if do_tracker else ''}"
          f"{'lane ' if do_lane else ''}"
          f"{'light ' if do_light else ''}"
          f"{'depth ' if do_depth else ''}"
          f"{'ocr' if do_ocr else ''}")
 
    # ── Load models ONCE, reuse across all videos ─────────────────────────────
    models: dict = {}
    lane_viz = None
    draw_speed = None
 
    if do_tracker or do_light or do_ocr:
        from tamakkan.models.tracker import TamakkanTracker
        models["tracker"] = TamakkanTracker(
            weights=str(WEIGHTS_DIR / "best.pt"),
            tracker_config=str(WEIGHTS_DIR / "bytetrack_tamakkan.yaml"),
        )
        print(f"tracker loaded  device={models['tracker'].device} "
              f"half={models['tracker'].half}")
 
    if do_light:
        from tamakkan.models.light_classifier import LightClassifier
        models["light"] = LightClassifier()
        print("light classifier loaded")
 
    if do_lane:
        from tamakkan.models.lane_model import LaneDetector, visualize as _lviz
        models["lane"] = LaneDetector(
            weights_path=str(WEIGHTS_DIR / "culane_res18_v2.pth"))
        lane_viz = _lviz
        print(f"lane loaded  device={models['lane'].device} "
              f"dtype={models['lane']._dtype}")
 
    if do_depth:
        from tamakkan.models.depth_model import DepthEstimator
        models["depth"] = DepthEstimator(
            weights_path=str(WEIGHTS_DIR / "depth_anything_v2_vits.pth"),
            variant="vits")
        print(f"depth loaded  device={models['depth'].device} "
              f"dtype={models['depth']._dtype}")
 
    if do_ocr:
        from tamakkan.models.ocr_model import SpeedSignOCR, draw_speed as _dspeed
        models["ocr"] = SpeedSignOCR()
        draw_speed = _dspeed
        print(f"ocr loaded  gpu={models['ocr'].use_gpu}")
 
    flags = {"tracker": do_tracker, "lane": do_lane, "light": do_light,
             "depth": do_depth, "ocr": do_ocr}
 
    grand_start = time.time()
    for v in videos:
        process_one_video(v, models, flags, lane_viz, draw_speed)
 
    print("\n" + "=" * 66)
    print(f"ALL DONE — {len(videos)} video(s) in "
          f"{time.time() - grand_start:.1f}s total")
    print("=" * 66)
 
 
if __name__ == "__main__":
    main()