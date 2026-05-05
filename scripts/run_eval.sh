#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${1:-${ROOT_DIR}/results/evaluation/VGGSound-Sparse/qualitative/demo_counterflow}"
FILTER_CSV="${2:-${ROOT_DIR}/datasets/VGGSound-Sparse/vggsound_sparse_clean_fixed_offsets.csv}"
GPU="${GPU:-0}"

python "${ROOT_DIR}/evaluation/eval_vggsound_sparse_metrics.py" \
  --output_dir "${EXP_DIR}" \
  --filter_csv "${FILTER_CSV}" \
  --gpu "${GPU}"

python "${ROOT_DIR}/evaluation/eval_flam_metric.py" \
  --exp_dir "${EXP_DIR}" \
  --filter_csv "${FILTER_CSV}" \
  --gpu_id "${GPU}"
