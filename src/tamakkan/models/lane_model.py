"""
src/tamakkan/models/lane_model.py
 
UFLD-v2 (Ultra-Fast Lane Detection v2, CULane ResNet-18) wrapper for the
Tamakkan pipeline.
 
Why v2 (history)
----------------
v1 (CULane ResNet-18, row-anchor only) was tested on Saudi dashcam footage
and failed badly — lane lines flailed across the road regardless of code
quality. That is a model/domain-shift problem, not a wrapper bug. v2 uses a
hybrid row+column anchor scheme that generalizes much better to road
geometry outside the Chinese CULane distribution, so the lane model was
switched from v1 to v2. The public API below is byte-for-byte identical to
the v1 wrapper, so no downstream code (event_engine / detectors / pipeline)
changes.
 
Design principles (unchanged from v1 wrapper)
---------------------------------------------
- Single responsibility: BGR frame in -> List[Lane] in ORIGINAL frame
  coordinates out. Visualization is a separate function.
- Stateless except the smoothing buffer.
- FP16 on CUDA, auto device detection, silent construction.
 
v2-specific gotchas handled here
--------------------------------
1. Preprocessing is resize-THEN-crop, controlled by crop_ratio (0.6).
   The frame is resized to (train_height/crop_ratio, train_width) =
   (533, 1600), then the BOTTOM train_height (320) rows are kept. Getting
   this wrong is the #1 cause of garbage UFLD-v2 output.
2. The model returns a dict (loc_row/loc_col/exist_row/exist_col), not a
   tensor. Decode is the hybrid scheme ported from the official demo.py
   pred2coords(): ego lanes (1,2) from row anchors, outer lanes (0,3)
   from column anchors, with local-window softmax expectation for
   sub-cell precision.
3. utils/common.py in the vendored tree is a minimal stub (the original
   imports NVIDIA DALI). We never train here, so that's fine.
 
Vendored code lives in third_party/ufld_v2/. third_party/ must be on
sys.path (the test scripts and pipeline entrypoints handle that).
"""
 
from __future__ import annotations
 
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple
 
import cv2
import numpy as np
import torch
 
from ufld_v2.model.model_culane import parsingNet
 
 
# ── CULane ResNet-18 config (hardcoded from configs/culane_res18.py) ───────────
# These are BAKED INTO the pretrained weights — not tunable.
CFG_BACKBONE      = "18"
CFG_GRIDING_NUM   = 200
CFG_NUM_LANES     = 4
CFG_NUM_ROW       = 72      # number of row anchors
CFG_NUM_COL       = 81      # number of column anchors
CFG_NUM_CELL_ROW  = 200     # grid resolution along a row
CFG_NUM_CELL_COL  = 100     # grid resolution along a column
CFG_TRAIN_WIDTH   = 1600
CFG_TRAIN_HEIGHT  = 320
CFG_FC_NORM       = True
CFG_CROP_RATIO    = 0.6
 
# Anchor positions, from utils/common.py CULane branch:
#   row_anchor = np.linspace(0.42, 1, num_row)   (normalized y)
#   col_anchor = np.linspace(0,    1, num_col)   (normalized x)
ROW_ANCHOR = np.linspace(0.42, 1.0, CFG_NUM_ROW)
COL_ANCHOR = np.linspace(0.0, 1.0, CFG_NUM_COL)
 
# Which of the 4 lane slots are decoded from row vs column anchors.
# From demo.py: row lanes are the two ego-lane lines, col lanes the outer.
ROW_LANE_IDX = [1, 2]
COL_LANE_IDX = [0, 3]
 
# Local window (in grid cells) for the softmax-expectation refinement.
LOCAL_WIDTH = 1
 
# UFLD-v2's internal reference resolution for decoded coordinates. demo.py
# uses the CULane native size (1640 x 590). We decode into this space then
# rescale to the caller's actual frame size.
REF_W = 1640
REF_H = 590
 
# ── ROI for the height-span sanity filter (fractions of frame height) ─────────
ROI_TOP_FRAC    = 0.45
ROI_BOTTOM_FRAC = 0.92
 
# ── Filtering / fitting defaults ──────────────────────────────────────────────
DEFAULT_MIN_POINTS           = 6
DEFAULT_MIN_LANE_HEIGHT_FRAC = 0.30
DEFAULT_SMOOTHING_WINDOW     = 5
 
# ── ImageNet normalization (same as v1 / depth) ───────────────────────────────
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
 
# ── Visualization colors (BGR) ────────────────────────────────────────────────
LANE_COLORS = [
    ( 80,  80, 255),   # lane 0 (outer left)  red-ish
    ( 80, 255,  80),   # lane 1 (ego left)    green
    (255, 150,  80),   # lane 2 (ego right)   blue-orange
    ( 80, 220, 255),   # lane 3 (outer right) yellow
]
 
 
@dataclass
class Lane:
    """
    One detected lane line in ORIGINAL frame coordinates.
 
    Identical structure to the v1 wrapper's Lane so downstream code
    (event_engine / detectors) is unaffected by the v1->v2 swap.
 
    poly_coeffs: np.polyfit coefficients for x = f(y) in frame coords.
    x_at_bottom: x at the bottom of the ROI, in original frame coords.
    side       : 'left' or 'right' relative to frame center.
    confidence : mean existence confidence of the anchors, in [0, 1].
    points_xy  : raw decoded (x, y) points in frame coords (for drawing).
    """
    poly_coeffs: np.ndarray
    x_at_bottom: float
    side: str
    confidence: float
    points_xy: List[Tuple[int, int]] = field(default_factory=list)
 
 
class LaneSmoother:
    """
    Temporal smoother — averages polynomial coefficients over a sliding
    window per left-to-right slot. Same logic as the v1 wrapper.
    """
 
    def __init__(self, window: int = DEFAULT_SMOOTHING_WINDOW):
        self.window = window
        self.history: List[deque] = [deque(maxlen=window) for _ in range(4)]
 
    def update(self, raw_lanes: List[Lane]) -> List[Lane]:
        sorted_lanes = sorted(raw_lanes, key=lambda ln: ln.x_at_bottom)
        smoothed: List[Lane] = []
        for slot, lane in enumerate(sorted_lanes):
            if slot >= 4:
                break
            self.history[slot].append(lane.poly_coeffs.copy())
            avg = np.mean(self.history[slot], axis=0)
            smoothed.append(Lane(
                poly_coeffs=avg,
                x_at_bottom=lane.x_at_bottom,
                side=lane.side,
                confidence=lane.confidence,
                points_xy=lane.points_xy,
            ))
        for slot in range(len(sorted_lanes), 4):
            self.history[slot].clear()
        return smoothed
 
    def reset(self):
        for h in self.history:
            h.clear()
 
 
class LaneDetector:
    """
    UFLD-v2 lane detector. Public API identical to the v1 wrapper.
 
        det = LaneDetector("weights/culane_res18_v2.pth")
        lanes = det.update(bgr_frame)          # List[Lane], frame coords
    """
 
    def __init__(
        self,
        weights_path: str,
        device: str | None = None,
        min_points: int = DEFAULT_MIN_POINTS,
        min_lane_height_frac: float = DEFAULT_MIN_LANE_HEIGHT_FRAC,
        smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
 
        # Build the v2 architecture with the hardcoded CULane res18 config.
        self.net = parsingNet(
            pretrained=False,
            backbone=CFG_BACKBONE,
            num_grid_row=CFG_NUM_CELL_ROW,
            num_cls_row=CFG_NUM_ROW,
            num_grid_col=CFG_NUM_CELL_COL,
            num_cls_col=CFG_NUM_COL,
            num_lane_on_row=CFG_NUM_LANES,
            num_lane_on_col=CFG_NUM_LANES,
            use_aux=False,
            input_height=CFG_TRAIN_HEIGHT,
            input_width=CFG_TRAIN_WIDTH,
            fc_norm=CFG_FC_NORM,
        )
 
        ckpt = torch.load(weights_path, map_location=self.device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        # Multi-GPU training prefixes keys with 'module.'; strip it.
        clean = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        self.net.load_state_dict(clean, strict=False)
        self.net = self.net.to(self.device)
        self.net.eval()
 
        if self.device.type == "cuda":
            self.net = self.net.half()
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32
 
        self._mean = _IMAGENET_MEAN.to(self.device, dtype=self._dtype)
        self._std  = _IMAGENET_STD.to(self.device, dtype=self._dtype)
 
        self.min_points = min_points
        self.min_lane_height_frac = min_lane_height_frac
        self.smoother = LaneSmoother(window=smoothing_window)
 
    # ── Public API ────────────────────────────────────────────────────────────
    def update(self, bgr_frame: np.ndarray) -> List[Lane]:
        if bgr_frame is None or bgr_frame.size == 0:
            return []
 
        h_orig, w_orig = bgr_frame.shape[:2]
        tensor = self._preprocess(bgr_frame)
 
        with torch.no_grad():
            pred = self.net(tensor)
 
        raw = self._decode(pred, w_orig, h_orig)
        return self.smoother.update(raw)
 
    def reset(self):
        self.smoother.reset()
 
    # ── Internals ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _preprocess(self, bgr_frame: np.ndarray) -> torch.Tensor:
        """
        UFLD-v2 preprocessing: resize to (train_height/crop_ratio,
        train_width) then keep the BOTTOM train_height rows.
 
        resize target: (1600 wide, 533 tall);  crop: bottom 320 rows.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
 
        resize_h = int(CFG_TRAIN_HEIGHT / CFG_CROP_RATIO)   # 320 / 0.6 = 533
        resized = cv2.resize(rgb, (CFG_TRAIN_WIDTH, resize_h))
 
        # keep the bottom train_height rows (the road), drop the top sky band
        cropped = resized[resize_h - CFG_TRAIN_HEIGHT:resize_h, :, :]
 
        tensor = (
            torch.from_numpy(cropped)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=self._dtype, non_blocking=True)
            .div_(255.0)
        )
        return (tensor - self._mean) / self._std
 
    def _decode(self, pred: dict, w_orig: int, h_orig: int) -> List[Lane]:
        """
        Hybrid row/col decode ported from the official demo.py pred2coords().
 
        Produces lane points in REF_W x REF_H space, rescales them to the
        caller's frame size, then fits x = f(y) like the v1 wrapper so the
        Lane dataclass and downstream code are unchanged.
        """
        loc_row = pred["loc_row"]
        loc_col = pred["loc_col"]
        exist_row = pred["exist_row"]
        exist_col = pred["exist_col"]
 
        _, num_grid_row, num_cls_row, num_lane_row = loc_row.shape
        _, num_grid_col, num_cls_col, num_lane_col = loc_col.shape
 
        max_idx_row = loc_row.argmax(1).cpu()      # (1, num_cls_row, lanes)
        valid_row   = exist_row.argmax(1).cpu()    # (1, num_cls_row, lanes)
        max_idx_col = loc_col.argmax(1).cpu()
        valid_col   = exist_col.argmax(1).cpu()
 
        loc_row_cpu = loc_row.float().cpu()
        loc_col_cpu = loc_col.float().cpu()
 
        sx = w_orig / REF_W
        sy = h_orig / REF_H
 
        raw_lanes: List[Lane] = []
 
        # ---- row-anchored lanes (ego left/right, mostly vertical) ----
        for i in ROW_LANE_IDX:
            if valid_row[0, :, i].sum() <= num_cls_row / 2:
                continue
            xs, ys, confs = [], [], []
            for k in range(valid_row.shape[1]):
                if not valid_row[0, k, i]:
                    continue
                lo = max(0, max_idx_row[0, k, i].item() - LOCAL_WIDTH)
                hi = min(num_grid_row - 1,
                         max_idx_row[0, k, i].item() + LOCAL_WIDTH) + 1
                idx = torch.arange(lo, hi)
                prob = loc_row_cpu[0, idx, k, i].softmax(0)
                loc = (prob * idx.float()).sum() + 0.5
                x_ref = loc / (num_grid_row - 1) * REF_W
                y_ref = ROW_ANCHOR[k] * REF_H
                xs.append(float(x_ref) * sx)
                ys.append(float(y_ref) * sy)
                confs.append(1.0)
            lane = self._fit_lane(xs, ys, confs, w_orig, h_orig)
            if lane is not None:
                raw_lanes.append(lane)
 
        # ---- column-anchored lanes (outer left/right, flatter) ----
        for i in COL_LANE_IDX:
            if valid_col[0, :, i].sum() <= num_cls_col / 4:
                continue
            xs, ys, confs = [], [], []
            for k in range(valid_col.shape[1]):
                if not valid_col[0, k, i]:
                    continue
                lo = max(0, max_idx_col[0, k, i].item() - LOCAL_WIDTH)
                hi = min(num_grid_col - 1,
                         max_idx_col[0, k, i].item() + LOCAL_WIDTH) + 1
                idx = torch.arange(lo, hi)
                prob = loc_col_cpu[0, idx, k, i].softmax(0)
                loc = (prob * idx.float()).sum() + 0.5
                y_ref = loc / (num_grid_col - 1) * REF_H
                x_ref = COL_ANCHOR[k] * REF_W
                xs.append(float(x_ref) * sx)
                ys.append(float(y_ref) * sy)
                confs.append(1.0)
            lane = self._fit_lane(xs, ys, confs, w_orig, h_orig)
            if lane is not None:
                raw_lanes.append(lane)
 
        return raw_lanes
 
    def _fit_lane(self, xs, ys, confs, w_orig, h_orig):
        """Fit x = f(y), build a Lane in frame coords. Returns None if the
        points are too few or span too short a vertical extent."""
        if len(xs) <= self.min_points:
            return None
 
        roi_top    = h_orig * ROI_TOP_FRAC
        roi_bottom = h_orig * ROI_BOTTOM_FRAC
 
        span = max(ys) - min(ys)
        if span < (roi_bottom - roi_top) * self.min_lane_height_frac:
            return None
 
        try:
            xs_a = np.asarray(xs, dtype=np.float64)
            ys_a = np.asarray(ys, dtype=np.float64)
            coeffs = np.polyfit(ys_a, xs_a, deg=2)
            x_at_bottom = float(np.polyval(coeffs, roi_bottom))
            confidence = float(np.clip(np.mean(confs), 0.0, 1.0))
            return Lane(
                poly_coeffs=coeffs,
                x_at_bottom=x_at_bottom,
                side="left" if x_at_bottom < w_orig / 2 else "right",
                confidence=confidence,
                points_xy=list(zip(
                    [int(round(x)) for x in xs],
                    [int(round(y)) for y in ys],
                )),
            )
        except (np.linalg.LinAlgError, ValueError):
            return None
 
 
# ── Visualization (drawing only; not part of detection logic) ─────────────────
def visualize(
    bgr_frame: np.ndarray,
    lanes: List[Lane],
    show_roi: bool = True,
    curve_margin: int = 25,
) -> np.ndarray:
    """Draw lanes on a copy of the frame, in original frame coordinates."""
    canvas = bgr_frame.copy()
    h, w = canvas.shape[:2]
 
    roi_top    = int(h * ROI_TOP_FRAC)
    roi_bottom = int(h * ROI_BOTTOM_FRAC)
 
    if show_roi:
        cv2.line(canvas, (0, roi_top),    (w, roi_top),    (50, 50, 50), 1)
        cv2.line(canvas, (0, roi_bottom), (w, roi_bottom), (50, 50, 50), 1)
 
    for i, lane in enumerate(lanes):
        color = LANE_COLORS[i % len(LANE_COLORS)]
        if not lane.points_xy or len(lane.points_xy) < 2:
            continue
 
        xs_a = np.array([p[0] for p in lane.points_xy], dtype=np.float32)
        ys_a = np.array([p[1] for p in lane.points_xy], dtype=np.float32)
 
        y_lo = max(roi_top + 10, int(np.min(ys_a)))
        y_hi = min(roi_bottom, int(np.max(ys_a)))
        if y_hi - y_lo < 2:
            continue
 
        y_line = np.arange(y_lo, y_hi, 1).astype(np.float32)
        x_line = np.polyval(lane.poly_coeffs, y_line)
 
        x_min = float(np.min(xs_a)) - curve_margin
        x_max = float(np.max(xs_a)) + curve_margin
        keep = (x_line >= x_min) & (x_line <= x_max)
        x_line = np.clip(x_line[keep], 0, w - 1).astype(np.int32)
        y_line = y_line[keep].astype(np.int32)
        if len(y_line) < 2:
            continue
 
        pts = np.stack([x_line, y_line], axis=1).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=4)
 
        # mark the raw decoded points so we can see model vs fitted curve
        for (px, py) in lane.points_xy:
            cv2.circle(canvas, (int(px), int(py)), 3, color, -1)
 
    return canvas
 
 
# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import time
 
    if len(sys.argv) < 3:
        print("usage: python lane_model.py <weights.pth> <image_path>")
        sys.exit(1)
 
    img = cv2.imread(sys.argv[2])
    if img is None:
        print(f"could not read image: {sys.argv[2]}")
        sys.exit(1)
 
    det = LaneDetector(weights_path=sys.argv[1])
    print(f"initialized on {det.device}, dtype={det._dtype}")
    print(f"input shape {img.shape} (lanes returned in this coord space)")
 
    _ = det.update(img)  # warmup
    t0 = time.time()
    for _ in range(10):
        lanes = det.update(img)
    dt = (time.time() - t0) / 10
    print(f"latency {dt*1000:.1f} ms/frame ({1/dt:.1f} FPS)")
 
    print(f"\ndetected {len(lanes)} lanes:")
    for i, ln in enumerate(lanes):
        print(f"  lane {i}: side={ln.side:5s} "
              f"x_at_bottom={ln.x_at_bottom:8.1f} "
              f"conf={ln.confidence:.3f} pts={len(ln.points_xy)}")
 
    cv2.imwrite("lane_v2_test.jpg", visualize(img, lanes))
    print("\nsaved lane_v2_test.jpg")