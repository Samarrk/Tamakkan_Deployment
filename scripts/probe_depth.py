"""
scripts/probe_depth.py

Diagnostic: what does depth_model.py ACTUALLY output on real footage?

We need to know, empirically (not by assumption):
  1. Direction — is a NEAR car's depth value larger or smaller than a
     FAR car's? (Depth Anything V2 raw output can be either depending on
     config; we stripped normalization so it's raw.)
  2. Scale / range — roughly what numbers come out, and how stable are
     they frame to frame for the same object.

Method: run tracker + depth on a clip. Each frame, take the vehicle with
the LOWEST bbox bottom (closest, "near") and the one with the HIGHEST
bbox top among smaller boxes (a "far" candidate). Print a robust depth
statistic (median over a patch inside each bbox) for both, plus global
frame min/max. Watch whether near > far or near < far, and the spread.

This is a READ-ONLY diagnostic — it builds nothing, just prints.

    python scripts/probe_depth.py <video.mp4>          # one clip
    python scripts/probe_depth.py <video.mp4> --frames 200
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


def patch_median(depth: np.ndarray, x1, y1, x2, y2) -> float:
    """Median depth over the lower-central third of the bbox (where the
    car body is, avoids sky/background through glass)."""
    h, w = depth.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return float("nan")
    bw, bh = x2 - x1, y2 - y1
    px1 = x1 + bw // 3
    px2 = x2 - bw // 3
    py1 = y1 + (2 * bh) // 3      # lower third
    py2 = y2
    px2 = max(px2, px1 + 1)
    py2 = max(py2, py1 + 1)
    patch = depth[py1:py2, px1:px2]
    if patch.size == 0:
        return float("nan")
    return float(np.median(patch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=150,
                    help="how many frames to probe (default 150)")
    args = ap.parse_args()

    from tamakkan.models.tracker import TamakkanTracker
    from tamakkan.models.depth_model import DepthEstimator

    tracker = TamakkanTracker(
        weights=str(WEIGHTS / "best.pt"),
        tracker_config=str(WEIGHTS / "bytetrack_tamakkan.yaml"),
    )
    depth = DepthEstimator(
        weights_path=str(WEIGHTS / "depth_anything_v2_vits.pth"),
        variant="vits")
    print(f"depth device={depth.device} dtype={depth._dtype}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}")
        return

    print(f"\n{'frм':>5} {'glob_min':>9} {'glob_max':>9} "
          f"{'NEAR(low box)':>14} {'FAR(high box)':>14}  note")
    print("-" * 72)

    n = 0
    near_vals, far_vals = [], []
    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        tracks = tracker.update(frame)
        dmap = depth.predict(frame)

        gmin, gmax = float(np.min(dmap)), float(np.max(dmap))

        vehicles = [t for t in tracks if t.is_vehicle]
        near_d = far_d = float("nan")
        note = ""
        if len(vehicles) >= 1:
            # NEAR = vehicle whose bbox bottom is lowest in frame
            near = max(vehicles, key=lambda t: t.bbox_int[3])
            nx1, ny1, nx2, ny2 = near.bbox_int
            near_d = patch_median(dmap, nx1, ny1, nx2, ny2)
            if not np.isnan(near_d):
                near_vals.append(near_d)

            # FAR = smallest-area vehicle (proxy for farthest)
            far = min(vehicles, key=lambda t: t.area)
            if far.track_id != near.track_id:
                fx1, fy1, fx2, fy2 = far.bbox_int
                far_d = patch_median(dmap, fx1, fy1, fx2, fy2)
                if not np.isnan(far_d):
                    far_vals.append(far_d)
            else:
                note = "(only 1 vehicle)"

        if n % 5 == 0 or len(vehicles) >= 2:
            print(f"{n:>5} {gmin:>9.2f} {gmax:>9.2f} "
                  f"{near_d:>14.3f} {far_d:>14.3f}  {note}")

    cap.release()

    print("\n" + "=" * 72)
    if near_vals and far_vals:
        nm, fm = np.median(near_vals), np.median(far_vals)
        print(f"median NEAR-vehicle depth : {nm:.3f}  "
              f"(n={len(near_vals)})")
        print(f"median FAR-vehicle  depth : {fm:.3f}  "
              f"(n={len(far_vals)})")
        if nm > fm:
            print("=> NEAR > FAR : bigger depth value == CLOSER object")
        else:
            print("=> NEAR < FAR : smaller depth value == CLOSER object")
        print(f"global value range seen   : ~[{min(near_vals+far_vals):.2f},"
              f" {max(near_vals+far_vals):.2f}]")
    else:
        print("not enough vehicle samples — try a clip with more traffic "
              "or more --frames")
    print("=" * 72)
    print("This tells us depth's DIRECTION and SCALE so the proximity "
          "detectors use it correctly as a relative cross-check.")


if __name__ == "__main__":
    main()