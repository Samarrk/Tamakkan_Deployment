"""
scripts/test_detectors_video.py

Visual test harness for ALL FOUR Tamakkan detectors (events), as opposed
to test_models_video.py which visualises raw MODEL output.

Wired detectors
---------------
  red_light_detector       RED_LIGHT_AHEAD / RED_LIGHT_RAN   (tracks)
  lane_violation_detector  LANE_DEPARTURE                    (lanes)
  tailgating_detector      TAILGATING                        (tracks+depth)
  near_miss_detector       NEAR_MISS                         (tracks+depth)

Because two detectors consume the depth map, the harness runs
tracker + lane model + depth model every frame. Depth is the slow model
(~17ms on RTX 5080) so this is slower than the model harness — that's
expected and fine for a correctness test (the real pipeline will own
frame-skip/cadence later).

What it draws
-------------
- UFLD-v2 lane lines (what the lane detector reasons about)
- traffic-light boxes coloured by the HSV classifier verdict
- event banners that appear the moment an event fires and PERSIST ~2s
  (instantaneous events would be an unwatchable 1-frame flash). Up to
  four banner slots stack so multiple simultaneous events are all visible.
- a small always-on HUD with per-event-type counts
- every event also printed to console (frame numbers + visual together)

Input modes
-----------
    python scripts/test_detectors_video.py <video.mp4>   # one file
    python scripts/test_detectors_video.py <folder>      # every video in it
    python scripts/test_detectors_video.py               # all videos in scripts/

Output: <name>_detect.mp4 next to each input video.

What to look for
----------------
- RED_LIGHT_AHEAD: fires on genuine red ahead, ~1 per encounter (50s
  scene cooldown), not spammy.
- LANE_DEPARTURE: fires on real drift; stays QUIET on clean intentional
  lane changes (settle-detection working); single-lane reported side
  correct.
- TAILGATING: fires when genuinely close behind a lead in the central
  path; not on adjacent-lane cars; class-aware (truck vs car vs bike).
- NEAR_MISS: fires when something is really closing fast; not on normal
  traffic flow. Tune the fraction constants from what you see.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
WEIGHTS_DIR = REPO_ROOT / "weights"
SCRIPTS_DIR = Path(__file__).resolve().parent

for p in (REPO_ROOT, SRC_DIR, THIRD_PARTY_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cv2  # noqa: E402

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
_OUTPUT_TAGS = ("_detect", "_redlight", "_lanedep", "_tailgate", "_speed",
                "_tracker", "_lane", "_light", "_depth", "_ocr")

# Banner styling per event (BGR, label).
_EVENT_STYLE = {
    "RED_LIGHT_AHEAD": ((0, 0, 220),    "RED LIGHT AHEAD"),
    "RED_LIGHT_RAN":   ((0, 0, 255),    "!! RED LIGHT VIOLATION !!"),
    "LANE_DEPARTURE":  ((200, 90, 0),   "LANE DEPARTURE"),
    "TAILGATING":      ((0, 140, 230),  "TOO CLOSE - TAILGATING"),
    "NEAR_MISS":       ((0, 0, 255),    "!! NEAR MISS - OBJECT CLOSING !!"),
}
# Which vertical slot each event type draws in (so they stack, not overlap)
_EVENT_SLOT = {
    "RED_LIGHT_AHEAD": 0, "RED_LIGHT_RAN": 0,
    "LANE_DEPARTURE":  1,
    "TAILGATING":      2,
    "NEAR_MISS":       3,
}


# ──────────────────────────────────────────────────────────────────────────────
# Input discovery
# ──────────────────────────────────────────────────────────────────────────────
def discover_videos(cli_arg: str | None) -> list[Path]:
    if cli_arg:
        p = Path(cli_arg)
        if p.is_dir():
            return sorted(
                v for v in p.iterdir()
                if v.suffix.lower() in _VIDEO_EXTS
                and not any(t in v.name for t in _OUTPUT_TAGS)
            )
        if p.is_file():
            return [p]
        print(f"ERROR: path not found: {cli_arg}")
        return []
    return sorted(
        v for v in SCRIPTS_DIR.iterdir()
        if v.suffix.lower() in _VIDEO_EXTS
        and not any(t in v.name for t in _OUTPUT_TAGS)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────────────
def put_label(img, text, x, y, color, scale=0.6, thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def draw_banner(frame, color, text, slot: int):
    """slot 0..3, each a 52px strip stacked from the top."""
    w = frame.shape[1]
    y0 = slot * 52
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, y0 + 48), color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    put_label(frame, text, 16, y0 + 33, (255, 255, 255), 0.8, 2)


def hud_y(img) -> int:
    return img.shape[0] - 16


class VideoWriterLazy:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = fps if fps and fps > 0 else 30.0
        self.writer = None

    def write(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps, (w, h))
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()


# ──────────────────────────────────────────────────────────────────────────────
# Per-video processing
# ──────────────────────────────────────────────────────────────────────────────
def process_one_video(video_path: Path, models, classes):
    tracker    = models["tracker"]
    lane_model = models["lane"]
    depth      = models["depth"]
    lane_viz   = models["lane_viz"]
    LightClassifier       = classes["LightClassifier"]
    RedLightDetector      = classes["RedLightDetector"]
    LaneViolationDetector = classes["LaneViolationDetector"]
    TailgatingDetector    = classes["TailgatingDetector"]
    NearMissDetector      = classes["NearMissDetector"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path.name}, skipping")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"\n--- {video_path.name}  ({total} frames @ {fps:.1f} fps) ---")

    # ── PER-DETECTOR WIRING ───────────────────────────────────────────────────
    red_det  = RedLightDetector(fps=fps)
    lane_det = LaneViolationDetector(fps=fps)
    tail_det = TailgatingDetector(fps=fps)
    near_det = NearMissDetector(fps=fps)
    for d in (red_det, lane_det, tail_det, near_det):
        d.reset()
    tracker.reset()
    lane_model.reset()
    viz_classifier = LightClassifier()
    print(f"  red AHEAD cooldown {red_det.ahead_cooldown_frames}f  |  "
          f"lane confirm {lane_det.frames_to_confirm}f settle "
          f"{lane_det.settle_frames}f  |  tail confirm "
          f"{tail_det.confirm_frames}f  |  near window {near_det.window}f")
    # ──────────────────────────────────────────────────────────────────────────

    out_path = video_path.with_name(f"{video_path.stem}_detect.mp4")
    writer = VideoWriterLazy(out_path, fps)
    banner_hold = int(2.0 * fps)

    # slot -> (until_frame, style)
    active = {0: (-1, None), 1: (-1, None), 2: (-1, None), 3: (-1, None)}

    n = 0
    counts = {k: 0 for k in
              ("RED_LIGHT_AHEAD", "RED_LIGHT_RAN", "LANE_DEPARTURE",
               "TAILGATING", "NEAR_MISS")}
    last_event_frame = None
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1

        tracks = tracker.update(frame)
        lanes = lane_model.update(frame)
        dmap = depth.predict(frame)

        # ── PER-DETECTOR CALLS ────────────────────────────────────────────────
        fired = []
        fired.extend(red_det.update(tracks, frame))
        le = lane_det.update(lanes, frame)
        if le is not None:
            fired.append(le)
        te = tail_det.update(tracks, dmap, frame, lanes)
        if te is not None:
            fired.append(te)
        ne = near_det.update(tracks, dmap, frame)
        if ne is not None:
            fired.append(ne)
        # ──────────────────────────────────────────────────────────────────────

        canvas = lane_viz(frame, lanes, show_roi=True)

        for tr in tracks:
            if not tr.is_traffic_light:
                continue
            x1, y1, x2, y2 = tr.bbox_int
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            res = viz_classifier.classify(crop)
            col = {"red": (0, 0, 255), "green": (0, 200, 0),
                   "unknown": (150, 150, 150)}.get(res.color, (150, 150, 150))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)

        for ev in fired:
            etype = ev.type.value
            counts[etype] = counts.get(etype, 0) + 1
            last_event_frame = n
            style = _EVENT_STYLE.get(etype, ((0, 0, 220), etype))
            slot = _EVENT_SLOT.get(etype, 0)
            active[slot] = (n + banner_hold, style)
            extra = ""
            if etype == "LANE_DEPARTURE":
                extra = f"  side={ev.side} off={ev.offset_px:+.0f}"
            elif etype == "TAILGATING":
                extra = (f"  {ev.class_name} boxh={ev.box_height_frac:.2f} "
                         f"reld={ev.rel_depth:.2f}")
            elif etype == "NEAR_MISS":
                extra = (f"  {ev.class_name} drise={ev.depth_rise:+.2f} "
                         f"agrow={ev.area_growth:+.2f}")
            print(f"  frame {n:>5}  {etype:18s}{extra}")

        for slot, (until, style) in active.items():
            if n <= until and style is not None:
                draw_banner(canvas, style[0], style[1], slot)

        since = "-" if last_event_frame is None else str(n - last_event_frame)
        put_label(canvas,
                  f"f{n}/{total} RLA{counts['RED_LIGHT_AHEAD']} "
                  f"RLR{counts['RED_LIGHT_RAN']} "
                  f"LD{counts['LANE_DEPARTURE']} "
                  f"TG{counts['TAILGATING']} "
                  f"NM{counts['NEAR_MISS']}  since {since}",
                  12, hud_y(canvas), (255, 255, 255), 0.55)

        writer.write(canvas)

        if n % 50 == 0 or n == total:
            el = time.time() - start
            pf = n / el if el > 0 else 0
            eta = (total - n) / pf if pf > 0 else 0
            print(f"  frame {n:>5}/{total}  proc {pf:4.1f} FPS  "
                  f"eta {int(eta//60)}m{int(eta%60):02d}s")

    cap.release()
    writer.release()
    dur = time.time() - start
    print(f"  done: {n} frames in {dur:.1f}s ({n/dur:.1f} FPS)")
    print(f"  events: AHEAD={counts['RED_LIGHT_AHEAD']} "
          f"RAN={counts['RED_LIGHT_RAN']} "
          f"LANE={counts['LANE_DEPARTURE']} "
          f"TAIL={counts['TAILGATING']} "
          f"NEAR={counts['NEAR_MISS']}")
    print(f"  output -> {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None
    videos = discover_videos(cli_arg)
    if not videos:
        print("No input videos found. Pass a file, a folder, or put "
              "videos in scripts/.")
        return

    print("=" * 66)
    print("  Tamakkan detector test — ALL 4 DETECTORS")
    print("=" * 66)
    print(f"videos to process: {len(videos)}")
    for v in videos:
        print(f"  - {v.name}")

    from tamakkan.models.tracker import TamakkanTracker
    from tamakkan.models.light_classifier import LightClassifier
    from tamakkan.models.lane_model import LaneDetector, visualize as lane_viz
    from tamakkan.models.depth_model import DepthEstimator
    from tamakkan.detectors.red_light_detector import RedLightDetector
    from tamakkan.detectors.lane_violation_detector import LaneViolationDetector
    from tamakkan.detectors.tailgating_detector import TailgatingDetector
    from tamakkan.detectors.near_miss_detector import NearMissDetector

    tracker = TamakkanTracker(
        weights=str(WEIGHTS_DIR / "best.pt"),
        tracker_config=str(WEIGHTS_DIR / "bytetrack_tamakkan.yaml"),
    )
    print(f"tracker loaded  device={tracker.device}")
    lane_model = LaneDetector(
        weights_path=str(WEIGHTS_DIR / "culane_res18_v2.pth"))
    print(f"lane loaded     device={lane_model.device}")
    depth = DepthEstimator(
        weights_path=str(WEIGHTS_DIR / "depth_anything_v2_vits.pth"),
        variant="vits")
    print(f"depth loaded    device={depth.device}")

    models = {"tracker": tracker, "lane": lane_model, "depth": depth,
              "lane_viz": lane_viz}
    classes = {
        "LightClassifier": LightClassifier,
        "RedLightDetector": RedLightDetector,
        "LaneViolationDetector": LaneViolationDetector,
        "TailgatingDetector": TailgatingDetector,
        "NearMissDetector": NearMissDetector,
    }

    grand = time.time()
    for v in videos:
        process_one_video(v, models, classes)

    print("\n" + "=" * 66)
    print(f"ALL DONE — {len(videos)} video(s) in {time.time() - grand:.1f}s")
    print("=" * 66)


if __name__ == "__main__":
    main()