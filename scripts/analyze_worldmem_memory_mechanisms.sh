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

OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/memory_diagnostics}"
METRICS_DIR="${METRICS_DIR:-$OUTPUT_ROOT/mechanism_analysis}"
DATA_DIR="${WORLDMEM_DATA_DIR:-data/minecraft}"
DATASET_SEED="${DATASET_SEED:-42}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-600}"
INITIAL_SKIP_FRAMES="${INITIAL_SKIP_FRAMES:-100}"
FPS="${FPS:-10}"
FUTURE_SECONDS="${FUTURE_SECONDS:-60}"
PREFIX_SECONDS="${PREFIX_SECONDS:-10,20,30,60}"
METRIC_DEVICE="${METRIC_DEVICE:-cuda}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-32}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"

cd "$REPO_ROOT"
cmd=(
  python utils/analyze_worldmem_memory_mechanisms.py
  --output_root "$OUTPUT_ROOT"
  --data_dir "$DATA_DIR"
  --metrics_dir "$METRICS_DIR"
  --dataset_seed "$DATASET_SEED"
  --context_frames "$CONTEXT_FRAMES"
  --initial_skip_frames "$INITIAL_SKIP_FRAMES"
  --fps "$FPS"
  --future_seconds "$FUTURE_SECONDS"
  --prefix_seconds "$PREFIX_SECONDS"
  --metric_device "$METRIC_DEVICE"
  --metric_batch_size "$METRIC_BATCH_SIZE"
  --bootstrap_samples "$BOOTSTRAP_SAMPLES"
)

if [ -n "${RUNS:-}" ]; then
  cmd+=(--runs "$RUNS")
fi
if [ -n "${LIMIT:-}" ]; then
  cmd+=(--limit "$LIMIT")
fi
if [ "${SKIP_LPIPS:-0}" = "1" ]; then
  cmd+=(--skip_lpips)
fi
if [ "${RECOMPUTE_FRAME_ERRORS:-0}" = "1" ]; then
  cmd+=(--recompute_frame_errors)
fi

echo "WorldMem memory mechanism analysis"
echo "Output root: $OUTPUT_ROOT"
echo "Metrics dir: $METRICS_DIR"
echo "Data dir: $DATA_DIR"
echo "Future seconds: $FUTURE_SECONDS"
echo "Runs: ${RUNS:-auto-discover worldmem_diag_*}"
echo "Video limit: ${LIMIT:-all completed videos}"
echo "Metric device: $METRIC_DEVICE"

"${cmd[@]}"
