"""
scripts/export_ufld_onnx.py

Export UFLD-v2 (CULane ResNet-18) to ONNX at fixed 1x3x320x1600 input.

UFLD-v2's forward() returns a dict (loc_row, loc_col, exist_row, exist_col)
which doesn't export cleanly. We wrap parsingNet in a thin module whose
forward returns a tuple in the SAME order, then unpack on the inference
side back into a dict so lane_model.py's decoder is unchanged.

Output: weights/culane_res18_v2.onnx
"""
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo / "third_party"))
sys.path.insert(0, str(_repo / "src"))

import torch
import torch.nn as nn
from ufld_v2.model.model_culane import parsingNet


# Must match lane_model.py CFG_* values exactly.
CFG_BACKBONE      = "18"
CFG_NUM_LANES     = 4
CFG_NUM_ROW       = 72
CFG_NUM_COL       = 81
CFG_NUM_CELL_ROW  = 200
CFG_NUM_CELL_COL  = 100
CFG_TRAIN_WIDTH   = 1600
CFG_TRAIN_HEIGHT  = 320
CFG_FC_NORM       = True

WEIGHTS_PT  = _repo / "weights" / "culane_res18_v2.pth"
WEIGHTS_ONX = _repo / "weights" / "culane_res18_v2.onnx"


class UFLDTupleWrapper(nn.Module):
    """
    Wraps parsingNet so forward() returns a tuple instead of a dict.
    ONNX exports tuples cleanly; dicts are unsupported / fragile.
    """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out["loc_row"], out["loc_col"], out["exist_row"], out["exist_col"]


def main():
    if not WEIGHTS_PT.exists():
        print(f"ERROR: weights not found: {WEIGHTS_PT}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building parsingNet on {device} ...")

    net = parsingNet(
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
    ).to(device).eval()

    ckpt = torch.load(str(WEIGHTS_PT), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    clean = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    net.load_state_dict(clean, strict=False)
    print("  weights loaded.")

    # FP32 for export; trtexec will quantize to FP16.
    net = net.float()

    wrapper = UFLDTupleWrapper(net).to(device).eval()

    dummy = torch.randn(1, 3, CFG_TRAIN_HEIGHT, CFG_TRAIN_WIDTH, device=device).float()

    print("Sanity forward pass ...")
    with torch.no_grad():
        loc_row, loc_col, exist_row, exist_col = wrapper(dummy)
    print(f"  loc_row:   {tuple(loc_row.shape)}")
    print(f"  loc_col:   {tuple(loc_col.shape)}")
    print(f"  exist_row: {tuple(exist_row.shape)}")
    print(f"  exist_col: {tuple(exist_col.shape)}")

    print(f"\nExporting to ONNX: {WEIGHTS_ONX}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(WEIGHTS_ONX),
            input_names  = ["input"],
            output_names = ["loc_row", "loc_col", "exist_row", "exist_col"],
            opset_version = 16,            # avoid TRT 8.5 LayerNorm-as-op issue
            do_constant_folding = True,
            export_params = True,
        )
    print(f"\nDone. ONNX size: {WEIGHTS_ONX.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
