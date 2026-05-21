"""
src/tamakkan/models/depth_model.py

Wraps the official Depth Anything V2 model for the Tamakkan pipeline.

Design principles
-----------------
- Single responsibility: this class does inference only.
  No frame-skip logic (pipeline owns that), no caching (pipeline owns that),
  no event detection (TailgatingDetector owns that).
- Stateless after construction: predict(frame) -> depth. No hidden state
  that callers need to reason about.
- Raw output: depth values are in the model's native scale, NOT normalized
  per frame. This is critical for temporal consistency in downstream
  detectors — see the predict() docstring.
- Backend auto-detection: pass weights_path=...pth for PyTorch (PC),
  or weights_path=...engine for TensorRT (Jetson). Detected by extension.
"""

from __future__ import annotations

# ── TensorRT shim (same as tracker.py) ───────────────────────────────────────
# JetPack 5.1.3 ships TensorRT 8.5 bindings that reference np.bool / np.long,
# both removed in numpy >= 1.24. Patch them BEFORE any tensorrt import.
import numpy as _np
if not hasattr(_np, "bool"):
    _np.bool = bool
if not hasattr(_np, "long"):
    _np.long = int

from pathlib import Path

import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2


# Fixed model input size. 518 == 37 * 14 (DINOv2 patch size).
INPUT_SIZE = 518

# ImageNet preprocessing — Depth Anything V2 was trained with these stats.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# TensorRT backend (Jetson)
# ─────────────────────────────────────────────────────────────────────────────
class _TRTDepthBackend:
    """
    TensorRT inference wrapper for DAv2 ViT-S engine, fixed 1x3x518x518.

    Loads engine once, allocates GPU buffers once, runs inference per
    call. Uses pycuda for buffer management — same path the rest of the
    Jetson ecosystem uses.

    Not thread-safe. One per pipeline.
    """

    def __init__(self, engine_path: str):
        # Lazy imports — only Jetson needs these.
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  -- initializes CUDA context

        self._trt = trt
        self._cuda = cuda

        # Load engine.
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Probe binding shapes. We expect input "input" (1,3,518,518)
        # and output "depth" (1,518,518).
        self.input_idx  = self.engine.get_binding_index("input")
        self.output_idx = self.engine.get_binding_index("depth")
        if self.input_idx < 0 or self.output_idx < 0:
            # Fallback: positional bindings if names differ.
            self.input_idx  = 0
            self.output_idx = 1

        self.input_shape  = tuple(self.engine.get_binding_shape(self.input_idx))
        self.output_shape = tuple(self.engine.get_binding_shape(self.output_idx))

        # Pre-allocate page-locked host buffers + device buffers.
        # Volumes count elements; we use float32 throughout.
        input_vol  = int(np.prod(self.input_shape))
        output_vol = int(np.prod(self.output_shape))

        self.h_input  = cuda.pagelocked_empty(input_vol,  dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(output_vol, dtype=np.float32)
        self.d_input  = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()

    def infer(self, chw_float32: np.ndarray) -> np.ndarray:
        """
        Run one inference.

        Args:
            chw_float32: preprocessed input, shape (1, 3, 518, 518), float32,
                         already normalized.

        Returns:
            float32 depth map of shape (518, 518) — raw model output.
        """
        # Copy input → pinned host buffer → device.
        np.copyto(self.h_input, chw_float32.ravel())
        self._cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        # Run.
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )

        # Device → pinned host.
        self._cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output.reshape(self.output_shape).squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch backend (PC, fallback)
# ─────────────────────────────────────────────────────────────────────────────
class _PyTorchDepthBackend:
    """Original PyTorch path. Used when weights_path is a .pth file."""

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64,
                 "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128,
                 "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256,
                 "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(self, weights_path: str, variant: str, device: str):
        self.device = torch.device(device)
        self.model = DepthAnythingV2(**self.MODEL_CONFIGS[variant]).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if self.device.type == "cuda":
            self.model = self.model.half()
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        self._mean = torch.from_numpy(_IMAGENET_MEAN).to(self.device, dtype=self._dtype)
        self._std  = torch.from_numpy(_IMAGENET_STD).to(self.device,  dtype=self._dtype)

    @torch.no_grad()
    def infer(self, chw_float32: np.ndarray) -> np.ndarray:
        # chw_float32 is preprocessed (BGR→RGB→resized→/255→normalized→CHW)
        # but in float32 numpy. We need it on GPU as float16.
        tensor = torch.from_numpy(chw_float32).to(self.device, dtype=self._dtype)
        depth = self.model(tensor)
        return depth.squeeze().float().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────
class DepthEstimator:
    """
    Monocular depth estimator wrapping Depth Anything V2.

    Backend chosen by weights_path extension:
        .pth     → PyTorch (PC, dev)
        .engine  → TensorRT (Jetson, production)
    """

    MODEL_CONFIGS = _PyTorchDepthBackend.MODEL_CONFIGS

    def __init__(
        self,
        weights_path: str,
        variant: str = "vits",
        device=None,
        input_size: int = INPUT_SIZE,
    ):
        if variant not in self.MODEL_CONFIGS:
            raise ValueError(
                f"variant must be one of {list(self.MODEL_CONFIGS)}, got {variant!r}"
            )
        if input_size % 14 != 0:
            raise ValueError(
                f"input_size must be a multiple of 14, got {input_size}"
            )

        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device     = torch.device(device) if isinstance(device, str) else device
        self.input_size = input_size
        self.variant    = variant
        self.weights_path = weights_path
        self.is_engine  = Path(weights_path).suffix.lower() == ".engine"

        if self.is_engine:
            self.backend = _TRTDepthBackend(weights_path)
            self._dtype  = torch.float32  # legacy attribute for smoke test
        else:
            self.backend = _PyTorchDepthBackend(weights_path, variant, str(self.device))
            self._dtype  = self.backend._dtype

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Run one depth inference pass.

        Args:
            frame: BGR uint8 array of shape (H, W, 3).

        Returns:
            float32 depth map of shape (H, W), matching the input frame size.
            Values in the model's RAW scale. Higher = closer to camera.
        """
        if frame is None or frame.size == 0:
            raise ValueError("predict() received empty frame")

        h_orig, w_orig = frame.shape[:2]

        # BGR → RGB → resize → CHW float32 → normalize.
        # Done in numpy/CPU because the input is small (518*518*3 floats =
        # ~3 MB) and the bottleneck is the model forward, not preprocessing.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size))

        chw = resized.astype(np.float32).transpose(2, 0, 1)[None, ...]  # (1,3,H,W)
        chw /= 255.0
        chw = (chw - _IMAGENET_MEAN) / _IMAGENET_STD

        depth = self.backend.infer(chw)  # (H_model, W_model) float32

        # Resize back to original frame dimensions.
        depth = cv2.resize(depth, (w_orig, h_orig))
        return depth.astype(np.float32)

    @staticmethod
    def colorize(depth: np.ndarray) -> np.ndarray:
        d = depth - depth.min()
        d = d / (d.max() + 1e-8) * 255.0
        return cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_INFERNO)


# ── Standalone smoke test ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("usage: python depth_model.py <image_path> [weights_path]")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    weights = sys.argv[2] if len(sys.argv) >= 3 else "weights/depth_anything_v2_vits.pth"
    print(f"Loading {weights} ...")
    est = DepthEstimator(weights_path=weights, variant="vits")
    print(f"  device={est.device}  is_engine={est.is_engine}")

    # Warmup
    _ = est.predict(img)

    N = 10
    t0 = time.time()
    for _ in range(N):
        depth = est.predict(img)
    dt = (time.time() - t0) / N
    print(f"Latency: {dt*1000:.1f} ms/frame ({1/dt:.1f} FPS)")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]  shape: {depth.shape}")

    out = "depth_test.jpg"
    cv2.imwrite(out, DepthEstimator.colorize(depth))
    print(f"Saved colorized depth to {out}")
