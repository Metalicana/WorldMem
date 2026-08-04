#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${WORLDMEM_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/memory_diagnostics/paired_confirm}"
METRICS_DIR="${METRICS_DIR:-$OUTPUT_ROOT/fvd_prefix}"

if [ -z "${RUNS:-}" ]; then
  RUNS="$({
    find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'worldmem_diag_*' \
      -exec basename {} \;
  } | sort | paste -sd, -)"
fi
if [ -z "$RUNS" ]; then
  echo "No worldmem_diag_* run directories found under $OUTPUT_ROOT" >&2
  exit 2
fi

WORLDMEM_REPO_ROOT="$REPO_ROOT" \
WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
METRICS_DIR="$METRICS_DIR" \
RUNS="$RUNS" \
bash "$SCRIPT_DIR/evaluate_worldmem_fvd.sh"
