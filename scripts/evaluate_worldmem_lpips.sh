#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORLDMEM_REPO_ROOT="${WORLDMEM_REPO_ROOT:-$DEFAULT_REPO_ROOT}"

if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/memory_policy}"
DATA_DIR="${WORLDMEM_DATA_DIR:-data/minecraft}"
METRICS_DIR="${METRICS_DIR:-$OUTPUT_ROOT/metrics/lpips_prefix}"
RUNS="${RUNS:-worldmem_unbounded_60s_n30,worldmem_fifo_b32_60s_n30,worldmem_fifo_b64_60s_n30,worldmem_rarity_irreplaceability_b32_60s_n30,worldmem_rarity_irreplaceability_b64_60s_n30,worldmem_slam_covisibility_b32_60s_n30,worldmem_slam_covisibility_b64_60s_n30}"
EVAL_DURATIONS="${EVAL_DURATIONS:-10,20,30,60}"
FPS="${FPS:-10}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-600}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
METRIC_DEVICE="${METRIC_DEVICE:-cuda}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-16}"
LPIPS_IMAGE_SIZE="${LPIPS_IMAGE_SIZE:-}"

cd "$WORLDMEM_REPO_ROOT"

cmd=(
  python utils/evaluate_worldmem_lpips_prefix_curves.py
  --output_root "$OUTPUT_ROOT"
  --data_dir "$DATA_DIR"
  --metrics_dir "$METRICS_DIR"
  --runs "$RUNS"
  --eval_durations "$EVAL_DURATIONS"
  --fps "$FPS"
  --context_frames "$CONTEXT_FRAMES"
  --seed "$SEED"
  --metric_device "$METRIC_DEVICE"
  --metric_batch_size "$METRIC_BATCH_SIZE"
)

if [ -n "$LIMIT" ]; then
  cmd+=(--limit "$LIMIT")
fi
if [ -n "$LPIPS_IMAGE_SIZE" ]; then
  cmd+=(--lpips_image_size "$LPIPS_IMAGE_SIZE")
fi

echo "WorldMem LPIPS prefix evaluation"
echo "Output root: $OUTPUT_ROOT"
echo "Data dir: $DATA_DIR"
echo "Runs: $RUNS"
echo "Eval durations: $EVAL_DURATIONS"
echo "Metrics dir: $METRICS_DIR"
echo "LPIPS image size: ${LPIPS_IMAGE_SIZE:-match-prediction-video}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
