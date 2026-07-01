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
METRICS_DIR="${METRICS_DIR:-$OUTPUT_ROOT/metrics/fvd_prefix}"
RUNS="${RUNS:-worldmem_unbounded_60s_n30,worldmem_fifo_b32_60s_n30,worldmem_fifo_b64_60s_n30}"
EVAL_DURATIONS="${EVAL_DURATIONS:-10,20,30,60}"
FPS="${FPS:-10}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-600}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
METRIC_DEVICE="${METRIC_DEVICE:-cuda}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-8}"
FVD_CLIP_LENGTH="${FVD_CLIP_LENGTH:-16}"
FVD_CLIPS_PER_VIDEO="${FVD_CLIPS_PER_VIDEO:-4}"
FVD_FRAME_STRIDE="${FVD_FRAME_STRIDE:-4}"
FVD_IMAGE_SIZE="${FVD_IMAGE_SIZE:-224}"
FVD_CACHE_DIR="${FVD_CACHE_DIR:-$STORAGE_ROOT/hf_cache/fvd}"
FVD_DETECTOR_PATH="${FVD_DETECTOR_PATH:-}"
FVD_ALLOW_DOWNLOAD="${FVD_ALLOW_DOWNLOAD:-1}"

cd "$WORLDMEM_REPO_ROOT"

cmd=(
  python utils/evaluate_worldmem_fvd_prefix_curves.py
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
  --fvd_clip_length "$FVD_CLIP_LENGTH"
  --fvd_clips_per_video "$FVD_CLIPS_PER_VIDEO"
  --fvd_frame_stride "$FVD_FRAME_STRIDE"
  --fvd_image_size "$FVD_IMAGE_SIZE"
  --fvd_cache_dir "$FVD_CACHE_DIR"
)

if [ -n "$LIMIT" ]; then
  cmd+=(--limit "$LIMIT")
fi
if [ -n "$FVD_DETECTOR_PATH" ]; then
  cmd+=(--fvd_detector_path "$FVD_DETECTOR_PATH")
fi
if [ "$FVD_ALLOW_DOWNLOAD" = "0" ]; then
  cmd+=(--no_fvd_download)
fi

echo "WorldMem FVD prefix evaluation"
echo "Output root: $OUTPUT_ROOT"
echo "Data dir: $DATA_DIR"
echo "Runs: $RUNS"
echo "Eval durations: $EVAL_DURATIONS"
echo "Metrics dir: $METRICS_DIR"
echo "FVD cache dir: $FVD_CACHE_DIR"

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
