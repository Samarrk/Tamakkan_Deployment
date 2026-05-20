"""
third_party/ufld_v2/utils/common.py
 
MINIMAL STUB — not the original UFLD-v2 utils/common.py.
 
The original file's first line is `from data.dali_data import TrainCollect`,
which requires NVIDIA DALI plus the full UFLD-v2 training/config machinery.
We only need inference, and the only symbol the model code imports from
here is `initialize_weights`.
 
`initialize_weights` only matters when training a model from scratch — it
sets initial Conv/Linear/BN values. Since Tamakkan always loads the
pretrained CULane checkpoint, those initial values are immediately
overwritten by `load_state_dict`. So a no-op is functionally correct here
and avoids dragging in DALI / distributed / config dependencies.
 
If you ever need to TRAIN UFLD-v2 (not just run inference), use the
original repo instead of this vendored stub.
"""
 
 
def initialize_weights(*models):
    """No-op. Pretrained weights are loaded right after model construction,
    so any initialization here would be overwritten anyway."""
    return
 