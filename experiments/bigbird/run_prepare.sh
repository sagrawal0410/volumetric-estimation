#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}"

OBJECT_ROOT="${OBJECT_ROOT:?Set OBJECT_ROOT to a BigBIRD object folder}"
OUT_DIR="${OUT_DIR:-prepared/bigbird_object}"
CONFIG="${CONFIG:-}"

ARGS=(bigbird --object_root "$OBJECT_ROOT" --out_dir "$OUT_DIR" --num_views 5 --validate)
if [[ -n "$CONFIG" ]]; then
  ARGS+=(--config "$CONFIG")
fi

python -m volume_benchmark.prepare_dataset "${ARGS[@]}"
echo "Prepared scan: $OUT_DIR"
