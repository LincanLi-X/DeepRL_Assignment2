#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash run_position_test.sh GPU-UUID" >&2
    exit 2
fi

question2_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="$1"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
# Share the project lock with training, so these commands cannot overlap.
exec flock -n "$question2_dir/.training_gpu.lock" python "$question2_dir/evaluate_position.py"
