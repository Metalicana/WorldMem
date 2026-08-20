#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${WORLDMEM_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/memory_quality_60s}"
METRICS_DIR="${METRICS_DIR:-$OUTPUT_ROOT/metrics/retrieved_memory_quality}"
RUNS="${RUNS:-}"
LATE_START_SEC="${LATE_START_SEC:-45}"
REPLAY_COUNT="${REPLAY_COUNT:-4}"

cd "$REPO_ROOT"
cmd=(
  python utils/analyze_worldmem_retrieved_memory_quality.py
  --output_root "$OUTPUT_ROOT"
  --metrics_dir "$METRICS_DIR"
  --late_start_sec "$LATE_START_SEC"
  --replay_count "$REPLAY_COUNT"
)
if [ -n "$RUNS" ]; then
  cmd+=(--runs "$RUNS")
fi
"${cmd[@]}"

