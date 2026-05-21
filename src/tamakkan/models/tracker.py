"""
src/tamakkan/models/tracker.py

TamakkanTracker — detection + tracking wrapper for the Tamakkan pipeline.

Encapsulates YOLOv11s + ByteTrack as a single component. All downstream
modules (depth, OCR, red-light, lane, AlertEngine) consume the List[Track]
that .update() returns — they never import ultralytics directly. This means
we can swap trackers (BoT-SORT, OC-SORT, ...) without touching detectors.

Usage:
    tracker = TamakkanTracker(
        weights="weights/best.pt",          # or "weights/best.engine" on Jetson
        tracker_config="weights/bytetrack_tamakkan.yaml",
    )
    for frame in video_stream:
        tracks = tracker.update(frame)
        vehicles = [t for t in tracks if t.is_vehicle]
"""

from __future__ import annotations

# ── TensorRT shim ────────────────────────────────────────────────────────────
# JetPack 5.1.3 ships TensorRT 8.5 bindings that reference np.bool / np.long,
# both removed in numpy >= 1.24. Patch them BEFORE any ultralytics import so
# loading a .engine file doesn't crash. Harmless on PC (np already has bool).
import numpy as _np
if not hasattr(_np, "bool"):
    _np.bool = bool
if not hasattr(_np, "long"):
    _np.long = int

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Track:
    """
    Single tracked object at one frame.

    Attributes are intentionally minimal — depth, velocity, alert state,
    and any other derived data are computed by downstream modules, not here.
    The tracker's only job is stable IDs + boxes per frame.
    """
    track_id: int                                    # persistent across frames
    class_id: int                                    # 0-6, see TamakkanTracker.CLASS_NAMES
    class_name: str                                  # human-readable class name
    confidence: float                                # detection confidence [0, 1]
    bbox: tuple                                      # (x1, y1, x2, y2), sub-pixel

    # ── Geometric helpers ───────────────────────────────────────────────
    @property
    def bbox_int(self):
        """Bbox rounded to ints, clamped non-negative."""
        x1, y1, x2, y2 = self.bbox
        return (max(0, int(x1)), max(0, int(y1)),
                max(0, int(x2)), max(0, int(y2)))

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    # ── Semantic helpers ────────────────────────────────────────────────
    @property
    def is_vehicle(self) -> bool:
        return self.class_id in TamakkanTracker.VEHICLE_CLASSES

    @property
    def is_vru(self) -> bool:
        return self.class_id in TamakkanTracker.VRU_CLASSES

    @property
    def is_traffic_light(self) -> bool:
        return self.class_id in TamakkanTracker.LIGHT_CLASSES

    @property
    def is_traffic_sign(self) -> bool:
        return self.class_id in TamakkanTracker.SIGN_CLASSES


class TamakkanTracker:
    """
    YOLOv11s + ByteTrack wrapper.

    Accepts either a PyTorch .pt weights file OR a TensorRT .engine file.
    The engine path is auto-detected from the file extension. On Jetson,
    use .engine for ~2.5x speedup over .pt.
    """

    CLASS_NAMES = {
        0: "car",
        1: "truck",
        2: "bus",
        3: "person",
        4: "traffic_light",
        5: "traffic_sign",
        6: "vulnerable_road_user",
    }

    VEHICLE_CLASSES = {0, 1, 2}
    VRU_CLASSES     = {3, 6}
    LIGHT_CLASSES   = {4}
    SIGN_CLASSES    = {5}

    def __init__(
        self,
        weights: str,
        tracker_config: str = "bytetrack_tamakkan.yaml",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 1280,
        device=None,
        half: bool = True,
    ):
        if not Path(weights).exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        if not Path(tracker_config).exists():
            raise FileNotFoundError(f"Tracker config not found: {tracker_config}")

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if device == "cpu" and half:
            half = False

        # Detect TensorRT engine by extension. .engine needs task='detect'
        # passed explicitly (engine files don't carry the task tag the
        # .pt files do). Also, engines are already FP16-compiled, so
        # the `half` flag is a no-op for them.
        is_engine = Path(weights).suffix.lower() == ".engine"

        if is_engine:
            self.model = YOLO(weights, task="detect")
        else:
            self.model = YOLO(weights)

        self.weights_path = weights
        self.is_engine = is_engine
        self.tracker_config = tracker_config
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.half = half

    def update(self, frame: np.ndarray) -> List[Track]:
        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
            stream=False,
        )

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return []

        boxes       = result.boxes.xyxy.cpu().numpy()
        track_ids   = result.boxes.id.cpu().numpy().astype(int)
        class_ids   = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        return [
            Track(
                track_id   = int(track_ids[i]),
                class_id   = int(class_ids[i]),
                class_name = self.CLASS_NAMES.get(int(class_ids[i]),
                                                  f"unknown_{class_ids[i]}"),
                confidence = float(confidences[i]),
                bbox       = tuple(boxes[i].tolist()),
            )
            for i in range(len(track_ids))
        ]

    def reset(self):
        """
        Clear all tracker state. Call between unrelated video clips or at
        the start of a new driving session. Costs ~1 second.
        """
        if self.is_engine:
            self.model = YOLO(self.weights_path, task="detect")
        else:
            self.model = YOLO(self.weights_path)


# ── Standalone smoke test ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import time
    import cv2

    if len(sys.argv) < 2:
        print("usage: python tracker.py <image_path> [weights_path]")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    weights = sys.argv[2] if len(sys.argv) >= 3 else "weights/best.pt"

    tracker = TamakkanTracker(
        weights=weights,
        tracker_config="weights/bytetrack_tamakkan.yaml",
    )
    print(f"Tracker initialized on {tracker.device}, "
          f"engine={tracker.is_engine}, half={tracker.half}")

    _ = tracker.update(img)

    t0 = time.time()
    for _ in range(10):
        tracks = tracker.update(img)
    dt = (time.time() - t0) / 10
    print(f"Latency: {dt*1000:.1f} ms/frame  ({1/dt:.1f} FPS)")

    print(f"\nDetected {len(tracks)} tracks:")
    for t in tracks:
        flag = "[V]" if t.is_vehicle else "[P]" if t.is_vru else "[L]" if t.is_traffic_light else "[S]" if t.is_traffic_sign else "[?]"
        print(f"  {flag} id={t.track_id:>3}  {t.class_name:25s}  "
              f"conf={t.confidence:.3f}  bbox={t.bbox_int}")
