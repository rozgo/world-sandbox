#!/usr/bin/env bash
# Prepare the AMB3R-SLAM runtime on a Linux machine with an NVIDIA GPU: the Python environment,
# the native parts (DBoW2 loop retrieval, the ALIKED patch op) and the model weights (~6.5 GB).
#
#   scripts/setup_slam.sh
#   CHECKPOINTS_FROM=/path/to/checkpoints scripts/setup_slam.sh   # link weights already on disk
#
# Needs uv, a CUDA toolkit matching torch's CUDA build (13.0 in uv.lock), a C++ compiler and
# OpenCV's development files (Ubuntu: sudo apt install libopencv-dev). The build writes only
# inside third_party/amb3r-slam, as untracked files; the submodule's sources are not changed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
A=$ROOT/third_party/amb3r-slam
git -C "$ROOT" submodule update --init third_party/amb3r-slam
uv sync --project "$ROOT" --locked --extra slam
source "$ROOT/.venv/bin/activate"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
bash "$A/thirdparty/build.sh"
if [ -n "${CHECKPOINTS_FROM:-}" ]; then
    for f in DA3NESTED-GIANT-LARGE-1.1 DA3-SMALL ORBvoc.txt; do
        ln -sfn "$CHECKPOINTS_FROM/$f" "$A/checkpoints/$f"
    done
fi
(cd "$A" && bash checkpoints/download.sh)   # fetches what is still missing, ALIKED weights included
python -c "import torch, dpretrieval; print('ready:', torch.__version__, torch.cuda.get_device_name(0))"
