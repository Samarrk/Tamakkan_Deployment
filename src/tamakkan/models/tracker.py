"""
src/tamakkan/models/tracker.py

TamakkanTracker — detection + tracking wrapper for the Tamakkan pipeline.

Encapsulates YOLOv11s + ByteTrack as a single component. All downstream
modules (depth, OCR, red-light, lane, AlertEngine) consume the List[Track]
that .update() returns — they never import ultralytics directly. This means
we can swap trackers (BoT-SORT, OC-SORT, ...) without touching detectors.

Usage:
    tracker = TamakkanTracker(
        weights="weights/best.pt",
        tracker_config="weights/bytetrack_tamakkan.yaml",
    )
    for frame in video_stream:
        tracks = tracker.update(frame)
        vehicles = [t for t in tracks if t.is_vehicle]
"""

from __future__ import annotations

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
    bbox: tuple[float, float, float, float]          # (x1, y1, x2, y2), sub-pixel

    # ── Geometric helpers ───────────────────────────────────────────────
    @property
    def bbox_int(self) -> tuple[int, int, int, int]:
        """Bbox rounded to ints, clamped non-negative.

        Use this when slicing arrays (depth maps, image crops) — direct
        float indexing emits NumPy deprecation warnings and may break in
        future versions. Negative bbox edges (which can happen with some
        trackers at frame edges) are clamped to 0.
        """
        x1, y1, x2, y2 = self.bbox
        return (max(0, int(x1)), max(0, int(y1)),
                max(0, int(x2)), max(0, int(y2)))

    @property
    def center(self) -> tuple[float, float]:
        """(cx, cy) — useful for distance / lane-position checks."""
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
    # These read class membership in plain English. Downstream code uses
    # `if track.is_vehicle:` instead of repeating class-id sets everywhere.
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

    Maintains internal track state across .update() calls — DO NOT create
    a new instance per frame, you'll lose all track IDs.
    Create once at session start, call update() per frame.
    """

    # Class ID → name. Keep in sync with data.yaml used during training.
    CLASS_NAMES = {
        0: "car",
        1: "truck",
        2: "bus",
        3: "person",
        4: "traffic_light",
        5: "traffic_sign",
        6: "vulnerable_road_user",
    }

    # ── Semantic class groupings ────────────────────────────────────────
    # Single source of truth so detector files don't each hardcode their own.
    VEHICLE_CLASSES = {0, 1, 2}   # car, truck, bus
    VRU_CLASSES     = {3, 6}      # person, vulnerable_road_user
    LIGHT_CLASSES   = {4}         # traffic_light
    SIGN_CLASSES    = {5}         # traffic_sign

    def __init__(
        self,
        weights: str,
        tracker_config: str = "bytetrack_tamakkan.yaml",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 1280,
        device: str | None = None,
        half: bool = True,
    ):
        """
        Args:
            weights: path to YOLOv11s best.pt
            tracker_config: path to ByteTrack yaml
            conf: detection confidence threshold (0.25 = F1-optimal for our model)
            iou: NMS IoU threshold (Ultralytics default)
            imgsz: inference resolution. 1280 = training resolution, highest
                   accuracy. Drop to 960 or 640 if pipeline FPS budget is tight.
            device: 'cuda:0' / 'cpu' / None for auto-detect.
            half: FP16 inference. Faster on GPU, free accuracy. Auto-disabled
                  if device='cpu' (FP16 on CPU is slower than FP32).
        """
        if not Path(weights).exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        if not Path(tracker_config).exists():
            raise FileNotFoundError(f"Tracker config not found: {tracker_config}")

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # FP16 on CPU is actually slower than FP32; auto-correct.
        if device == "cpu" and half:
            half = False

        self.model = YOLO(weights)
        self.weights_path = weights
        self.tracker_config = tracker_config
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.half = half

    def update(self, frame: np.ndarray) -> List[Track]:
        """
        Process one frame, return active tracks.

        Args:
            frame: BGR numpy array (H, W, 3) — standard OpenCV format.

        Returns:
            All currently-active tracks for this frame. Empty list if
            nothing detected.
        """
        # persist=True is the key flag — Ultralytics needs it to maintain
        # tracker state across calls. Without it tracking resets every frame.
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

        # Ultralytics returns one Results per input; we always pass one frame.
        result = results[0]

        # No detections → empty list (don't return None; consumers iterate).
        if result.boxes is None or result.boxes.id is None:
            return []

        boxes       = result.boxes.xyxy.cpu().numpy()              # (N, 4)
        track_ids   = result.boxes.id.cpu().numpy().astype(int)    # (N,)
        class_ids   = result.boxes.cls.cpu().numpy().astype(int)   # (N,)
        confidences = result.boxes.conf.cpu().numpy()              # (N,)

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
        the start of a new driving session.

        Costs ~1 second (disk read + GPU upload). Never call per-frame.
        """
        # Ultralytics doesn't expose a clean tracker-state reset, so we
        # reload the model. We cached the path so we don't need ckpt_path.
        self.model = YOLO(self.weights_path)


# ── Standalone smoke test ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import time
    import cv2

    if len(sys.argv) < 2:
        print("usage: python tracker.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    tracker = TamakkanTracker(
        weights="weights/best.pt",
        tracker_config="weights/bytetrack_tamakkan.yaml",
    )
    print(f"Tracker initialized on {tracker.device}, half={tracker.half}")

    # Warmup
    _ = tracker.update(img)

    # Time 10 frames
    t0 = time.time()
    for _ in range(10):
        tracks = tracker.update(img)
    dt = (time.time() - t0) / 10
    print(f"Latency: {dt*1000:.1f} ms/frame  ({1/dt:.1f} FPS)")

    print(f"\nDetected {len(tracks)} tracks:")
    for t in tracks:
        flag = "🚗" if t.is_vehicle else "🚶" if t.is_vru else "🚦" if t.is_traffic_light else "🪧" if t.is_traffic_sign else "?"
        print(f"  {flag} id={t.track_id:>3}  {t.class_name:25s}  "
              f"conf={t.confidence:.3f}  bbox={t.bbox_int}")