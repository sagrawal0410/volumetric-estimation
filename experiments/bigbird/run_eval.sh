#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}"

SCAN_DIR="${SCAN_DIR:?Set SCAN_DIR to a prepared scan directory}"
SUMMARY="${SUMMARY:-reports/eval_summary.json}"

python -m volume_benchmark.run_eval "$SCAN_DIR" \
  --methods convex_hull tsdf voxel_carving \
  --summary "$SUMMARY"

echo "Method outputs under: $SCAN_DIR/outputs/"
