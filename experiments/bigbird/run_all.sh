#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$DIR/run_prepare.sh"
export SCAN_DIR="${OUT_DIR:-prepared/bigbird_object}"
bash "$DIR/run_eval.sh"
