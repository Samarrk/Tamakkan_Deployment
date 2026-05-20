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

Setup (one-time)
----------------
1. Clone the official repo and make the package importable:
       git clone https://github.com/DepthAnything/Depth-Anything-V2 third_party/
       # ensure 'depth_anything_v2' is on PYTHONPATH

2. Download ViT-S weights (~99MB):
       from huggingface_hub import hf_hub_download
       hf_hub_download(
           "depth-anything/Depth-Anything-V2-Small",
           "depth_anything_v2_vits.pth",
           local_dir="weights/",
       )

Usage
-----
    estimator = DepthEstimator(weights_path="weights/depth_anything_v2_vits.pth")
    depth_map = estimator.predict(frame_bgr)
    # for visualization only — never feed colorized depth into detectors:
    heatmap = DepthEstimator.colorize(depth_map)
"""

from __future__ import annotations

import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2


# Fixed model input size. 518 == 37 * 14 (DINOv2 patch size).
# This is what Depth Anything V2 was trained at; going higher costs speed
# and leaves the training distribution without meaningful gains.
INPUT_SIZE = 518

# ImageNet preprocessing — Depth Anything V2 was trained with these stats.
# Skipping this step is a silent correctness bug: the model still produces
# plausible-looking depth, but values are systematically biased.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DepthEstimator:
    """
    Monocular depth estimator wrapping Depth Anything V2.

    The returned depth map uses the model's raw output scale. Higher value
    means closer to the camera, but values are NOT in meters. Detectors
    that consume this output should calibrate against this raw scale; they
    must NOT re-normalize per frame because that breaks frame-to-frame
    consistency (the same object would shift in value as the scene changes).
    """

    # All three official ViT variants. Tamakkan uses ViT-S in production
    # for the latency/quality trade-off — ViT-B's quality gains don't help
    # bbox-mean tailgating decisions enough to justify the 3× slowdown.
    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64,
                 "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128,
                 "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256,
                 "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(
        self,
        weights_path: str,
        variant: str = "vits",
        device: str | None = None,
        input_size: int = INPUT_SIZE,
    ):
        """
        Args:
            weights_path: path to .pth file matching the chosen variant.
            variant: one of "vits" (default, production), "vitb", "vitl".
            device: "cuda", "cpu", or None for auto-detect.
            input_size: square model input dimension. Must be a multiple
                of 14 (DINOv2 patch size). 518 is the training resolution.
        """
        if variant not in self.MODEL_CONFIGS:
            raise ValueError(
                f"variant must be one of {list(self.MODEL_CONFIGS)}, got {variant!r}"
            )
        if input_size % 14 != 0:
            raise ValueError(
                f"input_size must be a multiple of 14 (DINOv2 patch size), got {input_size}"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.input_size = input_size
        self.variant = variant

        # Build and load the model.
        self.model = DepthAnythingV2(**self.MODEL_CONFIGS[variant]).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Use half precision on CUDA — Depth Anything V2 runs fine in FP16
        # and roughly halves latency on Orin NX.
        if self.device.type == "cuda":
            self.model = self.model.half()
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        # Pre-move normalization constants to device once, not every frame.
        self._mean = _IMAGENET_MEAN.to(self.device, dtype=self._dtype)
        self._std  = _IMAGENET_STD.to(self.device, dtype=self._dtype)

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Run one depth inference pass.

        Args:
            frame: BGR uint8 array of shape (H, W, 3). Standard OpenCV frame.

        Returns:
            float32 depth map of shape (H, W), matching the input frame size.
            Values are in the model's RAW scale (NOT [0, 255], NOT meters).
            Higher = closer to camera.

            We deliberately do NOT per-frame normalize. Doing so would make
            a stationary object's depth value drift as the rest of the scene
            changes, which destroys any chance of a stable threshold.
        """
        if frame is None or frame.size == 0:
            raise ValueError("predict() received empty frame")

        h_orig, w_orig = frame.shape[:2]

        # BGR → RGB → resize to model input → tensor → normalize.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size))

        tensor = (
            torch.from_numpy(resized)
            .permute(2, 0, 1)            # HWC → CHW
            .unsqueeze(0)                # add batch dim
            .to(self.device, dtype=self._dtype, non_blocking=True)
            .div_(255.0)
        )
        tensor = (tensor - self._mean) / self._std

        depth = self.model(tensor)       # raw model output

        # Squeeze batch + channel dims, back to CPU/numpy for OpenCV.
        depth = depth.squeeze().float().cpu().numpy()

        # Resize back to original frame dimensions so bbox indexing matches.
        depth = cv2.resize(depth, (w_orig, h_orig))

        return depth.astype(np.float32)

    @staticmethod
    def colorize(depth: np.ndarray) -> np.ndarray:
        """
        Convert raw depth map to a BGR uint8 heatmap for visualization
        or streaming to the phone app.

        This is COSMETIC ONLY. Never feed this output into detectors —
        the min-max stretch loses the absolute scale that tailgating logic
        relies on.
        """
        d = depth - depth.min()
        d = d / (d.max() + 1e-8) * 255.0
        return cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_INFERNO)


# Standalone smoke test — runs only when you execute this file directly.
# In the full pipeline, this file is just imported.
if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    if len(sys.argv) < 2:
        print("usage: python depth_model.py <image_or_video_path>")
        sys.exit(1)

    weights = "weights/depth_anything_v2_vits.pth"
    if not Path(weights).exists():
        print(f"Missing weights: {weights}")
        print("Download with:")
        print('  huggingface-cli download depth-anything/Depth-Anything-V2-Small '
              'depth_anything_v2_vits.pth --local-dir weights/')
        sys.exit(1)

    estimator = DepthEstimator(weights_path=weights, variant="vits")
    print(f"Loaded ViT-S on {estimator.device}, dtype={estimator._dtype}")

    src = sys.argv[1]
    # Try as image first, fall back to video.
    img = cv2.imread(src)
    if img is not None:
        # Warmup pass for fair timing.
        _ = estimator.predict(img)

        t0 = time.time()
        N = 10
        for _ in range(N):
            depth = estimator.predict(img)
        dt = (time.time() - t0) / N
        print(f"Single image: {dt*1000:.1f} ms/frame ({1/dt:.1f} FPS)")
        print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")

        cv2.imwrite("depth_test.jpg", DepthEstimator.colorize(depth))
        print("Saved colorized depth to depth_test.jpg")
    else:
        print(f"Could not load as image: {src}")
        sys.exit(1)