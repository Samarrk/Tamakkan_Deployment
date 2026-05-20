"""
scripts/test_models.py

Standalone sanity test for the 4 finalized Tamakkan model wrappers:
    - light_classifier.py   (no weights, no GPU needed)
    - tracker.py            (needs weights/best.pt)
    - lane_model.py         (needs weights/culane_18.pth + vendored UFLD)
    - depth_model.py        (needs weights/depth_anything_v2_vits.pth + vendored DAv2)

Run from the repo root:
    python scripts/test_models.py
    python scripts/test_models.py path/to/your/frame.jpg

What it does
------------
1. Fixes sys.path so 'tamakkan.*' and 'third_party.*' imports resolve.
2. Imports every module first — a structural smoke test. If a module
   can't even import, that's reported before any model is loaded.
3. Runs each model on a test image, isolated in its own try/except, so
   one failure doesn't hide the others. Prints a final PASS/FAIL summary.

This is a PC sanity check. A green run here means imports + logic are
sound. It does NOT guarantee Jetson behaviour (numpy 2.x here vs 1.24 on
Jetson, torch 2.11 vs 2.1) — speed + final validation happen on Jetson.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Path setup — make 'tamakkan.*' and 'third_party.*' importable regardless of
# where this script is run from.
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"

for p in (REPO_ROOT, SRC_DIR, THIRD_PARTY_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

WEIGHTS_DIR = REPO_ROOT / "weights"

# Track results across all tests for the final summary.
RESULTS: dict[str, str] = {}


def banner(title: str):
    print("\n" + "=" * 66)
    print(f"  {title}")
    print("=" * 66)


def find_test_image(cli_arg: str | None) -> np.ndarray | None:
    """
    Resolve a test image. Priority:
      1. CLI argument path, if given
      2. Any .jpg/.png in repo root or weights/ or a 'samples' folder
      3. Extract first frame from any .mp4 found under the repo
      4. Fall back to a synthetic 720x1280 gradient (lets pipelines run,
         though real models won't find real objects in it)
    """
    import cv2

    # 1. explicit path
    if cli_arg:
        img = cv2.imread(cli_arg)
        if img is not None:
            print(f"Test image: {cli_arg}  shape={img.shape}")
            return img
        print(f"WARNING: could not read image at {cli_arg}, falling back...")

    # 2. look for an image in a few likely spots
    search_dirs = [REPO_ROOT, WEIGHTS_DIR, REPO_ROOT / "samples", REPO_ROOT / "docs"]
    for d in search_dirs:
        if not d.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            hits = list(d.glob(ext))
            if hits:
                img = cv2.imread(str(hits[0]))
                if img is not None:
                    print(f"Test image (auto-found): {hits[0]}  shape={img.shape}")
                    return img

    # 3. extract a frame from any video in the repo
    for vid in REPO_ROOT.rglob("*.mp4"):
        cap = cv2.VideoCapture(str(vid))
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            print(f"Test image (frame from {vid.name}): shape={frame.shape}")
            return frame

    # 4. synthetic fallback
    print("WARNING: no real image found — using a synthetic gradient frame.")
    print("         Models will run but won't detect real objects/lanes.")
    h, w = 720, 1280
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return np.stack([grad, grad, grad], axis=2)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — import smoke test
# ──────────────────────────────────────────────────────────────────────────────
def test_imports() -> bool:
    banner("PHASE 1 — Imports")
    all_ok = True

    modules = [
        ("light_classifier", "tamakkan.models.light_classifier",
         ["LightClassifier", "LightClassification"]),
        ("tracker", "tamakkan.models.tracker",
         ["TamakkanTracker", "Track"]),
        ("lane_model", "tamakkan.models.lane_model",
         ["LaneDetector", "Lane", "visualize"]),
        ("depth_model", "tamakkan.models.depth_model",
         ["DepthEstimator"]),
    ]

    for short, dotted, names in modules:
        try:
            mod = __import__(dotted, fromlist=names)
            for n in names:
                if not hasattr(mod, n):
                    raise ImportError(f"{dotted} has no '{n}'")
            print(f"  [OK]   {short:18s}  ({', '.join(names)})")
            RESULTS[f"import:{short}"] = "PASS"
        except Exception as e:
            print(f"  [FAIL] {short:18s}  {type(e).__name__}: {e}")
            traceback.print_exc()
            RESULTS[f"import:{short}"] = "FAIL"
            all_ok = False

    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — per-model functional tests
# ──────────────────────────────────────────────────────────────────────────────
def test_light_classifier():
    banner("PHASE 2a — LightClassifier (no weights / no GPU)")
    try:
        from tamakkan.models.light_classifier import LightClassifier

        clf = LightClassifier()
        cases = {
            "solid red":   np.full((40, 40, 3), (0, 0, 255), np.uint8),
            "solid green": np.full((40, 40, 3), (0, 255, 0), np.uint8),
            "dark/off":    np.full((40, 40, 3), 30, np.uint8),
            "gray junk":   np.full((40, 40, 3), 128, np.uint8),
        }
        for label, img in cases.items():
            r = clf.classify(img)
            print(f"  {label:12s} -> color={r.color:8s} conf={r.confidence:.3f}")
        RESULTS["run:light_classifier"] = "PASS"
        print("  [OK] LightClassifier produced output")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        RESULTS["run:light_classifier"] = "FAIL"


def test_tracker(img):
    banner("PHASE 2b — TamakkanTracker (needs weights/best.pt)")
    try:
        from tamakkan.models.tracker import TamakkanTracker

        w = WEIGHTS_DIR / "best.pt"
        cfg = WEIGHTS_DIR / "bytetrack_tamakkan.yaml"
        if not w.exists():
            print(f"  [SKIP] missing {w}")
            RESULTS["run:tracker"] = "SKIP"
            return

        t0 = time.time()
        tracker = TamakkanTracker(weights=str(w), tracker_config=str(cfg))
        print(f"  init: {time.time()-t0:.2f}s  device={tracker.device}  half={tracker.half}")

        _ = tracker.update(img)                       # warmup
        t0 = time.time()
        tracks = tracker.update(img)
        dt = time.time() - t0
        print(f"  inference: {dt*1000:.1f} ms   detected {len(tracks)} tracks")
        for tr in tracks[:8]:
            print(f"    id={tr.track_id:>3} {tr.class_name:22s} "
                  f"conf={tr.confidence:.2f} bbox={tr.bbox_int} "
                  f"vehicle={tr.is_vehicle}")
        RESULTS["run:tracker"] = "PASS"
        print("  [OK] tracker produced output")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        RESULTS["run:tracker"] = "FAIL"


def test_lane_model(img):
    banner("PHASE 2c — LaneDetector (needs weights/culane_18.pth + UFLD)")
    try:
        from tamakkan.models.lane_model import LaneDetector, visualize
        import cv2

        w = WEIGHTS_DIR / "culane_res18_v2.pth"
        if not w.exists():
            print(f"  [SKIP] missing {w}")
            RESULTS["run:lane_model"] = "SKIP"
            return

        t0 = time.time()
        det = LaneDetector(weights_path=str(w))
        print(f"  init: {time.time()-t0:.2f}s  device={det.device}  dtype={det._dtype}")

        _ = det.update(img)                           # warmup
        t0 = time.time()
        lanes = det.update(img)
        dt = time.time() - t0
        print(f"  inference: {dt*1000:.1f} ms   detected {len(lanes)} lanes")
        for i, ln in enumerate(lanes):
            print(f"    lane {i}: side={ln.side:5s} "
                  f"x_at_bottom={ln.x_at_bottom:7.1f} "
                  f"conf={ln.confidence:.3f} pts={len(ln.points_xy)}")

        out = REPO_ROOT / "lane_test_output.jpg"
        cv2.imwrite(str(out), visualize(img, lanes))
        print(f"  saved visualization -> {out.name}")
        RESULTS["run:lane_model"] = "PASS"
        print("  [OK] lane model produced output")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        RESULTS["run:lane_model"] = "FAIL"


def test_depth_model(img):
    banner("PHASE 2d — DepthEstimator (needs DAv2 weights + vendored repo)")
    try:
        from tamakkan.models.depth_model import DepthEstimator
        import cv2

        w = WEIGHTS_DIR / "depth_anything_v2_vits.pth"
        if not w.exists():
            print(f"  [SKIP] missing {w}")
            RESULTS["run:depth_model"] = "SKIP"
            return

        t0 = time.time()
        est = DepthEstimator(weights_path=str(w), variant="vits")
        print(f"  init: {time.time()-t0:.2f}s  device={est.device}  dtype={est._dtype}")

        _ = est.predict(img)                          # warmup
        t0 = time.time()
        depth = est.predict(img)
        dt = time.time() - t0
        print(f"  inference: {dt*1000:.1f} ms   depth shape={depth.shape}")
        print(f"  depth range: [{depth.min():.3f}, {depth.max():.3f}]  "
              f"mean={depth.mean():.3f}")

        out = REPO_ROOT / "depth_test_output.jpg"
        cv2.imwrite(str(out), DepthEstimator.colorize(depth))
        print(f"  saved colorized depth -> {out.name}")
        RESULTS["run:depth_model"] = "PASS"
        print("  [OK] depth model produced output")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        RESULTS["run:depth_model"] = "FAIL"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    banner("Tamakkan model sanity test")
    print(f"repo root : {REPO_ROOT}")
    print(f"weights   : {WEIGHTS_DIR}")
    try:
        import torch
        print(f"torch     : {torch.__version__}  CUDA={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu       : {torch.cuda.get_device_name(0)}")
    except Exception:
        print("torch     : NOT IMPORTABLE")

    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None

    imports_ok = test_imports()
    if not imports_ok:
        print("\nSome imports failed. Fix those before running model tests.")
        print_summary()
        return

    img = find_test_image(cli_arg)

    test_light_classifier()
    test_tracker(img)
    test_lane_model(img)
    test_depth_model(img)

    print_summary()


def print_summary():
    banner("SUMMARY")
    width = max(len(k) for k in RESULTS) if RESULTS else 20
    for k, v in RESULTS.items():
        mark = {"PASS": "[OK]  ", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}.get(v, "[?]   ")
        print(f"  {mark} {k:<{width}}  {v}")
    n_pass = sum(1 for v in RESULTS.values() if v == "PASS")
    n_fail = sum(1 for v in RESULTS.values() if v == "FAIL")
    n_skip = sum(1 for v in RESULTS.values() if v == "SKIP")
    print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print("=" * 66)


if __name__ == "__main__":
    main()
