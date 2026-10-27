# Source from bash after selecting a Python environment. Sets paths and runtime defaults only.
DXMZ_ACCURACY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DATA_ROOT="${DATA_ROOT:-$DXMZ_ACCURACY_ROOT/datasets}"
export MODEL_ROOT="${MODEL_ROOT:-$DXMZ_ACCURACY_ROOT/models}"
export DXNN_DEVICES="${DXNN_DEVICES:-0}"
export NVIDIA_TF32_OVERRIDE=0
unset DXMZ_ACCURACY_ROOT
