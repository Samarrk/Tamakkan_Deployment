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
switched from v1 to v2.

Backend auto-detection: pass weights_path=...pth for PyTorch (PC), or
weights_path=...engine for TensorRT (Jetson). Detected by extension.

Design principles
-----------------
- Single responsibility: BGR frame in -> List[Lane] in ORIGINAL frame
  coordinates out. Visualization is a separate function.
- Stateless except the smoothing buffer.
- FP16 on CUDA, auto device detection, silent construction.

v2-specific gotchas handled here
--------------------------------
1. Preprocessing is resize-THEN-crop, controlled by crop_ratio (0.6).
   The frame is resized to (train_height/crop_ratio, train_width) =
   (533, 1600), then the BOTTOM train_height (320) rows are kept.
2. The model returns a dict (loc_row/loc_col/exist_row/exist_col).
   The TRT engine returns a tuple in the SAME order; we re-pack into
   the same dict shape so the decoder is unchanged.
3. utils/common.py in the vendored tree is a minimal stub (the original
   imports NVIDIA DALI). We never train here, so that's fine.
"""

from __future__ import annotations

# ── TensorRT shim (same as tracker.py / depth_model.py) ──────────────────────
import numpy as _np
if not hasattr(_np, "bool"):
    _np.bool = bool
if not hasattr(_np, "long"):
    _np.long = int

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
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

ROW_ANCHOR = np.linspace(0.42, 1.0, CFG_NUM_ROW)
COL_ANCHOR = np.linspace(0.0, 1.0, CFG_NUM_COL)

ROW_LANE_IDX = [1, 2]
COL_LANE_IDX = [0, 3]

REF_W = 1640
REF_H = 590

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ── Lane dataclass + smoother (unchanged) ────────────────────────────────────
@dataclass
class Lane:
    points:     List[Tuple[float, float]]
    polynomial: Tuple[float, float, float]
    side:       str
    confidence: float
    x_at_bottom: float

    @property
    def y_top(self)    -> float: return self.points[0][1]
    @property
    def y_bottom(self) -> float: return self.points[-1][1]


class LaneSmoother:
    def __init__(self, window: int = 5):
        self.window = window
        self.buffer: deque = deque(maxlen=window)

    def reset(self): self.buffer.clear()

    def update(self, lanes: List[Lane]) -> List[Lane]:
        self.buffer.append(lanes)
        if len(self.buffer) < 2:
            return lanes

        # Group by side, average polynomials + x_at_bottom for stable signal.
        by_side: dict = {}
        for past in self.buffer:
            for ln in past:
                by_side.setdefault(ln.side, []).append(ln)

        smoothed: List[Lane] = []
        for side, group in by_side.items():
            if len(group) < self.window // 2 + 1:
                continue
            polys = np.array([g.polynomial for g in group])
            xs    = np.array([g.x_at_bottom for g in group])
            avg_poly = tuple(polys.mean(axis=0))
            avg_x    = float(xs.mean())
            ref = group[-1]
            smoothed.append(Lane(
                points      = ref.points,
                polynomial  = avg_poly,
                side        = side,
                confidence  = ref.confidence,
                x_at_bottom = avg_x,
            ))
        smoothed.sort(key=lambda l: l.x_at_bottom)
        return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# TensorRT backend (Jetson)
# ─────────────────────────────────────────────────────────────────────────────
class _TRTLaneBackend:
    """
    TensorRT inference wrapper for UFLD-v2 engine at fixed 1x3x320x1600.

    Loads engine once, allocates GPU buffers once. Returns the four
    output tensors as a dict matching parsingNet's forward() output.
    """

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        cuda.init()
        # Use the primary CUDA context — can be pushed onto any thread,
        # unlike pycuda.autoinit which binds to the import thread.
        # Required for FastAPI/uvicorn worker threads.
        self._trt = trt
        self._cuda = cuda
        device = cuda.Device(0)
        self._cuda_ctx = device.retain_primary_context()
        # Push so the rest of __init__ (buffer allocation) runs in context.
        self._cuda_ctx.push()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Bind in the canonical order. Output names were set in export_ufld_onnx.py.
        self._bind_input    = self._idx("input",    fallback=0)
        self._bind_loc_row  = self._idx("loc_row",  fallback=1)
        self._bind_loc_col  = self._idx("loc_col",  fallback=2)
        self._bind_exist_r  = self._idx("exist_row",fallback=3)
        self._bind_exist_c  = self._idx("exist_col",fallback=4)

        self.input_shape    = tuple(self.engine.get_binding_shape(self._bind_input))
        self.loc_row_shape  = tuple(self.engine.get_binding_shape(self._bind_loc_row))
        self.loc_col_shape  = tuple(self.engine.get_binding_shape(self._bind_loc_col))
        self.exist_r_shape  = tuple(self.engine.get_binding_shape(self._bind_exist_r))
        self.exist_c_shape  = tuple(self.engine.get_binding_shape(self._bind_exist_c))

        # Pre-allocate pinned host + device buffers.
        self.h_input    = cuda.pagelocked_empty(int(np.prod(self.input_shape)),   dtype=np.float32)
        self.h_loc_row  = cuda.pagelocked_empty(int(np.prod(self.loc_row_shape)), dtype=np.float32)
        self.h_loc_col  = cuda.pagelocked_empty(int(np.prod(self.loc_col_shape)), dtype=np.float32)
        self.h_exist_r  = cuda.pagelocked_empty(int(np.prod(self.exist_r_shape)), dtype=np.float32)
        self.h_exist_c  = cuda.pagelocked_empty(int(np.prod(self.exist_c_shape)), dtype=np.float32)

        self.d_input    = cuda.mem_alloc(self.h_input.nbytes)
        self.d_loc_row  = cuda.mem_alloc(self.h_loc_row.nbytes)
        self.d_loc_col  = cuda.mem_alloc(self.h_loc_col.nbytes)
        self.d_exist_r  = cuda.mem_alloc(self.h_exist_r.nbytes)
        self.d_exist_c  = cuda.mem_alloc(self.h_exist_c.nbytes)

        # Bindings must be ordered by binding index, not insertion order.
        bindings_list = [None] * self.engine.num_bindings
        bindings_list[self._bind_input]   = int(self.d_input)
        bindings_list[self._bind_loc_row] = int(self.d_loc_row)
        bindings_list[self._bind_loc_col] = int(self.d_loc_col)
        bindings_list[self._bind_exist_r] = int(self.d_exist_r)
        bindings_list[self._bind_exist_c] = int(self.d_exist_c)
        self.bindings = bindings_list

        self.stream = cuda.Stream()
        # Pop the init-time context push; infer() will push/pop per call.
        self._cuda_ctx.pop()

    def _idx(self, name: str, fallback: int) -> int:
        i = self.engine.get_binding_index(name)
        return i if i >= 0 else fallback

    def infer(self, chw_float32: np.ndarray) -> dict:
        """
        Run one inference. May be called from any thread; CUDA context
        is pushed/popped per call. Returns dict matching parsingNet
        forward() shape so _decode() doesn't need to know the backend.
        """
        self._cuda_ctx.push()
        try:
            return self._infer_inner(chw_float32)
        finally:
            self._cuda_ctx.pop()

    def _infer_inner(self, chw_float32: np.ndarray) -> dict:
        np.copyto(self.h_input, chw_float32.ravel())
        self._cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        # Run
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )

        # Pull all four outputs back
        self._cuda.memcpy_dtoh_async(self.h_loc_row, self.d_loc_row, self.stream)
        self._cuda.memcpy_dtoh_async(self.h_loc_col, self.d_loc_col, self.stream)
        self._cuda.memcpy_dtoh_async(self.h_exist_r, self.d_exist_r, self.stream)
        self._cuda.memcpy_dtoh_async(self.h_exist_c, self.d_exist_c, self.stream)
        self.stream.synchronize()

        # Reshape and wrap as torch tensors so _decode() works unchanged
        # (it calls .argmax(1) / .cpu() / .softmax — torch ops).
        return {
            "loc_row":   torch.from_numpy(self.h_loc_row.reshape(self.loc_row_shape).copy()),
            "loc_col":   torch.from_numpy(self.h_loc_col.reshape(self.loc_col_shape).copy()),
            "exist_row": torch.from_numpy(self.h_exist_r.reshape(self.exist_r_shape).copy()),
            "exist_col": torch.from_numpy(self.h_exist_c.reshape(self.exist_c_shape).copy()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────
class LaneDetector:
    """
    UFLD-v2 lane detector. Backend chosen by weights_path extension:
        .pth     → PyTorch (PC, dev)
        .engine  → TensorRT (Jetson, production)
    """

    def __init__(
        self,
        weights_path: str,
        device=None,
        min_points: int = 8,
        min_lane_height_frac: float = 0.20,
        smoothing_window: int = 5,
    ):
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if isinstance(device, str) else device

        self.weights_path = weights_path
        self.is_engine = Path(weights_path).suffix.lower() == ".engine"

        if self.is_engine:
            # TRT backend: no torch model, just the engine.
            self.trt_backend = _TRTLaneBackend(weights_path)
            self.net = None
            self._dtype = torch.float32
        else:
            # PyTorch backend: build, load, half-precision on CUDA.
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
            self.trt_backend = None

        # Preprocessing constants on the right device for the PyTorch path.
        self._mean = _IMAGENET_MEAN.to(self.device, dtype=self._dtype)
        self._std  = _IMAGENET_STD.to(self.device,  dtype=self._dtype)

        self.min_points = min_points
        self.min_lane_height_frac = min_lane_height_frac
        self.smoother = LaneSmoother(window=smoothing_window)

    # ── Public API ────────────────────────────────────────────────────────────
    def update(self, bgr_frame: np.ndarray) -> List[Lane]:
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        h_orig, w_orig = bgr_frame.shape[:2]

        if self.is_engine:
            chw = self._preprocess_numpy(bgr_frame)   # (1,3,320,1600) float32
            pred = self.trt_backend.infer(chw)
        else:
            tensor = self._preprocess_torch(bgr_frame)
            with torch.no_grad():
                pred = self.net(tensor)

        raw = self._decode(pred, w_orig, h_orig)
        return self.smoother.update(raw)

    def reset(self):
        self.smoother.reset()

    # ── Preprocessing ────────────────────────────────────────────────────────
    @torch.no_grad()
    def _preprocess_torch(self, bgr_frame: np.ndarray) -> torch.Tensor:
        """PyTorch path: returns a (1,3,320,1600) tensor on self.device."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resize_h = int(CFG_TRAIN_HEIGHT / CFG_CROP_RATIO)
        resized = cv2.resize(rgb, (CFG_TRAIN_WIDTH, resize_h))
        cropped = resized[resize_h - CFG_TRAIN_HEIGHT:resize_h, :, :]
        tensor = (
            torch.from_numpy(cropped)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=self._dtype, non_blocking=True)
            .div_(255.0)
        )
        return (tensor - self._mean) / self._std

    def _preprocess_numpy(self, bgr_frame: np.ndarray) -> np.ndarray:
        """TRT path: returns a (1,3,320,1600) float32 numpy array."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resize_h = int(CFG_TRAIN_HEIGHT / CFG_CROP_RATIO)
        resized = cv2.resize(rgb, (CFG_TRAIN_WIDTH, resize_h))
        cropped = resized[resize_h - CFG_TRAIN_HEIGHT:resize_h, :, :]
        chw = cropped.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        return (chw - mean) / std

    # ── Decoder (unchanged from the original) ────────────────────────────────
    def _decode(self, pred: dict, w_orig: int, h_orig: int) -> List[Lane]:
        loc_row = pred["loc_row"]
        loc_col = pred["loc_col"]
        exist_row = pred["exist_row"]
        exist_col = pred["exist_col"]

        _, num_grid_row, num_cls_row, num_lane_row = loc_row.shape
        _, num_grid_col, num_cls_col, num_lane_col = loc_col.shape

        max_idx_row = loc_row.argmax(1).cpu()
        valid_row   = exist_row.argmax(1).cpu()
        max_idx_col = loc_col.argmax(1).cpu()
        valid_col   = exist_col.argmax(1).cpu()

        loc_row_cpu = loc_row.float().cpu()
        loc_col_cpu = loc_col.float().cpu()

        sx = w_orig / REF_W
        sy = h_orig / REF_H

        lanes: List[Lane] = []

        # Row anchors → ego lanes (idx 1, 2)
        for i in ROW_LANE_IDX:
            if valid_row[0, :, i].sum() <= num_cls_row / 2:
                continue
            pts = []
            for k in range(num_cls_row):
                if valid_row[0, k, i] == 0:
                    continue
                idx = int(max_idx_row[0, k, i])
                lo = max(0, idx - 4)
                hi = min(num_grid_row - 1, idx + 4)
                prob = loc_row_cpu[0, lo:hi+1, k, i].softmax(0)
                pos = torch.arange(lo, hi+1, dtype=torch.float32)
                loc = (prob * pos).sum().item()
                x_ref = loc / (num_grid_row - 1) * REF_W
                y_ref = ROW_ANCHOR[k] * REF_H
                pts.append((x_ref * sx, y_ref * sy))
            if len(pts) < self.min_points:
                continue
            lanes.append(self._build_lane(pts, h_orig, "ego_left" if i == 1 else "ego_right"))

        # Col anchors → outer lanes (idx 0, 3)
        for i in COL_LANE_IDX:
            if valid_col[0, :, i].sum() <= num_cls_col / 4:
                continue
            pts = []
            for k in range(num_cls_col):
                if valid_col[0, k, i] == 0:
                    continue
                idx = int(max_idx_col[0, k, i])
                lo = max(0, idx - 4)
                hi = min(num_grid_col - 1, idx + 4)
                prob = loc_col_cpu[0, lo:hi+1, k, i].softmax(0)
                pos = torch.arange(lo, hi+1, dtype=torch.float32)
                loc = (prob * pos).sum().item()
                x_ref = COL_ANCHOR[k] * REF_W
                y_ref = loc / (num_grid_col - 1) * REF_H
                pts.append((x_ref * sx, y_ref * sy))
            if len(pts) < self.min_points:
                continue
            lanes.append(self._build_lane(pts, h_orig, "outer_left" if i == 0 else "outer_right"))

        # Reject lanes that don't span enough vertical extent.
        min_h_px = self.min_lane_height_frac * h_orig
        kept = []
        for ln in lanes:
            if ln.y_bottom - ln.y_top < min_h_px:
                continue
            kept.append(ln)
        kept.sort(key=lambda l: l.x_at_bottom)
        return kept

    def _build_lane(self, pts: List[Tuple[float, float]], h_orig: int, side: str) -> Lane:
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        try:
            poly = np.polyfit(ys, xs, 2)
        except Exception:
            poly = (0.0, 0.0, float(np.mean(xs)))
        x_at_bottom = float(np.polyval(poly, h_orig - 1))
        return Lane(
            points      = pts,
            polynomial  = tuple(poly),
            side        = side,
            confidence  = 1.0,
            x_at_bottom = x_at_bottom,
        )


# ── Standalone smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("usage: python lane_model.py <image_path> [weights_path]")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    weights = sys.argv[2] if len(sys.argv) >= 3 else "weights/culane_res18_v2.pth"
    print(f"Loading {weights} ...")
    det = LaneDetector(weights_path=weights)
    print(f"  device={det.device}  is_engine={det.is_engine}")

    _ = det.update(img)  # warmup

    N = 10
    t0 = time.time()
    for _ in range(N):
        lanes = det.update(img)
    dt = (time.time() - t0) / N
    print(f"Latency: {dt*1000:.1f} ms/frame ({1/dt:.1f} FPS)")
    print(f"Lanes detected: {len(lanes)}")
    for ln in lanes:
        print(f"  {ln.side:12s}  conf={ln.confidence:.2f}  x_at_bottom={ln.x_at_bottom:.1f}  pts={len(ln.points)}")
