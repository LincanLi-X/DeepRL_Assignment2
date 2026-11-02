#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_training.sh GPU-UUID [--resume]" >&2
    exit 2
fi

question2_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
question2_gpu_uuid="$1"
shift

# The lock prevents concurrent runs of this project's training command. GPU
# access itself is direct because this server has no discovered job scheduler.
export CUDA_VISIBLE_DEVICES="$question2_gpu_uuid"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
exec flock -n "$question2_dir/.training_gpu.lock" python "$question2_dir/train.py" "$@"
