"""
scripts/probe_tailgating.py
 
Measure the EXACT quantities the tailgating detector keys on, so its
thresholds are set from ground truth instead of guessed.
 
For every frame it finds the lead vehicle the SAME way
tailgating_detector does (vehicle class {0,1,2}, bbox centre inside the
central path band, closest by patch depth) and prints:
 
    box_h   = lead bbox height / frame height
    reld    = lead patch depth / frame max depth   <-- the key number
 
Use it on a clip where YOU KNOW a real tailgate happens. Watch the
console (and/or the written video) and read off the `reld` value during
the moment the car ahead is genuinely too close. That value (minus a
small margin) is what DEPTH_REL_FRAC should be.
 
It also writes <name>_tgprobe.mp4 with the lead boxed and the live
box_h / reld printed on the frame, so you can pause exactly when the
car is close and read the number.
 
    python scripts/probe_tailgating.py <video.mp4>
    python scripts/probe_tailgating.py <video.mp4> --every 5
"""
 
from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
 
import numpy as np
 
REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT / "src", REPO_ROOT / "third_party"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
 
import cv2  # noqa: E402
 
WEIGHTS = REPO_ROOT / "weights"
 
# must match tailgating_detector
PATH_REGION_FRAC = 0.34
VEHICLE_IDS = {0, 1, 2}
 
 
def patch_depth(depth_map, x1, y1, x2, y2) -> float:
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
 
 
def put(img, text, x, y, color, s=0.6, t=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s,
                (0, 0, 0), t + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s,
                color, t, cv2.LINE_AA)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=10,
                    help="print to console every N frames (default 10)")
    args = ap.parse_args()
 
    from tamakkan.models.tracker import TamakkanTracker
    from tamakkan.models.depth_model import DepthEstimator
 
    tracker = TamakkanTracker(
        weights=str(WEIGHTS / "best.pt"),
        tracker_config=str(WEIGHTS / "bytetrack_tamakkan.yaml"))
    depth = DepthEstimator(
        weights_path=str(WEIGHTS / "depth_anything_v2_vits.pth"),
        variant="vits")
 
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = Path(args.video).with_name(Path(args.video).stem + "_tgprobe.mp4")
    writer = None
 
    print(f"{'frame':>6} {'lead_cls':>9} {'box_h':>7} {'reld':>7}  "
          f"(reld = what DEPTH_REL_FRAC is compared against)")
    print("-" * 60)
 
    reld_samples = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        fh, fw = frame.shape[:2]
        cx_lo = fw * (0.5 - PATH_REGION_FRAC / 2.0)
        cx_hi = fw * (0.5 + PATH_REGION_FRAC / 2.0)
 
        tracks = tracker.update(frame)
        dmap = depth.predict(frame)
        fmax = float(np.max(dmap)) if dmap.size else 0.0
 
        ahead = []
        for t in tracks:
            if t.class_id not in VEHICLE_IDS:
                continue
            x1, y1, x2, y2 = t.bbox_int
            cx = (x1 + x2) * 0.5
            if cx_lo <= cx <= cx_hi:
                ahead.append(t)
 
        canvas = frame.copy()
        # draw the path band
        cv2.line(canvas, (int(cx_lo), 0), (int(cx_lo), fh), (90, 90, 90), 1)
        cv2.line(canvas, (int(cx_hi), 0), (int(cx_hi), fh), (90, 90, 90), 1)
 
        box_h = reld = 0.0
        lead_cls = "-"
        if ahead and fmax > 0:
            lead = max(ahead,
                       key=lambda t: patch_depth(dmap, *t.bbox_int))
            x1, y1, x2, y2 = lead.bbox_int
            box_h = (y2 - y1) / float(fh)
            reld = patch_depth(dmap, x1, y1, x2, y2) / fmax
            lead_cls = lead.class_name
            reld_samples.append(reld)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 3)
            put(canvas, f"LEAD {lead_cls}", x1, max(y1 - 10, 20),
                (0, 165, 255), 0.6, 2)
 
        put(canvas, f"frame {n}  box_h={box_h:.2f}  reld={reld:.2f}",
            14, 34, (255, 255, 255), 0.7, 2)
        put(canvas, "pause when the car ahead is genuinely too close, "
                    "read reld", 14, 64, (0, 220, 255), 0.55, 2)
 
        if writer is None:
            writer = cv2.VideoWriter(
                str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
        writer.write(canvas)
 
        if n % args.every == 0:
            print(f"{n:>6} {lead_cls:>9} {box_h:>7.2f} {reld:>7.3f}")
 
    cap.release()
    if writer:
        writer.release()
 
    print("\n" + "=" * 60)
    if reld_samples:
        a = np.array(reld_samples)
        print(f"reld over frames with a lead vehicle (n={len(a)}):")
        print(f"  min={a.min():.3f}  p25={np.percentile(a,25):.3f}  "
              f"median={np.median(a):.3f}  p75={np.percentile(a,75):.3f}  "
              f"max={a.max():.3f}")
        print("Look at the WRITTEN video, pause during the real tailgate,")
        print("read reld there. Set DEPTH_REL_FRAC a little BELOW that.")
    else:
        print("no lead vehicle ever found in the path band — try a clip "
              "where you're clearly following a car")
    print(f"video -> {out.name}")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()