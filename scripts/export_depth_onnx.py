"""
scripts/export_depth_onnx.py

Export Depth Anything V2 ViT-S to ONNX at fixed 1x3x518x518 input.

Why fixed shape?
- DAv2's interpolate_pos_encoding has runtime branching that ONNX
  trace can't capture cleanly across variable input sizes.
- At exactly 518x518 (training resolution), the function early-returns
  the unmodified pos_embed and there's no interpolation to export.
- Our pipeline always feeds 518x518 anyway (see depth_model.py INPUT_SIZE).
- Fixed shape also lets TensorRT pick faster kernels (no shape switching).

Output:
    weights/depth_anything_v2_vits.onnx

Run on Jetson; no GPU strictly needed for export (CPU works) but using
CUDA here lets us also do a sanity-check forward pass.
"""
import sys
from pathlib import Path

# Make vendored DAv2 importable.
_repo = Path(__file__).resolve().parents[1]
_tp = _repo / "third_party"
sys.path.insert(0, str(_tp))
sys.path.insert(0, str(_repo / "src"))

import torch
import torch.nn as nn
from depth_anything_v2.dpt import DepthAnythingV2


VITS_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}

WEIGHTS_PT  = _repo / "weights" / "depth_anything_v2_vits.pth"
WEIGHTS_ONX = _repo / "weights" / "depth_anything_v2_vits.onnx"
INPUT_SIZE  = 518


def main():
    if not WEIGHTS_PT.exists():
        print(f"ERROR: weights not found: {WEIGHTS_PT}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DAv2 ViT-S on {device} ...")

    model = DepthAnythingV2(**VITS_CONFIG).to(device).eval()
    state = torch.load(str(WEIGHTS_PT), map_location=device)
    model.load_state_dict(state)
    print("  weights loaded.")

    # Force the model into FP32 for export — we'll let TensorRT do the
    # FP16 conversion. Exporting an FP16 model to ONNX is finickier than
    # exporting FP32 and letting trtexec quantize.
    model = model.float()

    # Dummy input: 1x3x518x518 (NCHW, normalized to roughly ImageNet range)
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device).float()

    print("Running a forward pass first as a sanity check...")
    with torch.no_grad():
        out = model(dummy)
    print(f"  output shape: {tuple(out.shape)} dtype: {out.dtype}")

    print(f"\nExporting to ONNX: {WEIGHTS_ONX}")
    print("  (this may print a wall of TraceWarnings — most are benign)")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(WEIGHTS_ONX),
            input_names  = ["input"],
            output_names = ["depth"],
            opset_version = 16,            # broad TensorRT 8.5 support
            do_constant_folding = True,
            # No dynamic_axes — we want a fixed shape engine for max speed
            export_params = True,
        )

    print(f"\nDone. ONNX file size: {WEIGHTS_ONX.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
