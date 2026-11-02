#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: bash run_bigram_evaluation.sh GPU-UUID" >&2
    exit 2
fi
bonus3_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="$1"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec flock -n "$bonus3_dir/.gpu.lock" python "$bonus3_dir/evaluate_bigram.py"
