#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_gpt.sh GPU-UUID [--resume]" >&2
    exit 2
fi
bonus3_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="$1"
shift
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
exec flock -n "$bonus3_dir/.gpu.lock" python "$bonus3_dir/train_gpt.py" "$@"
