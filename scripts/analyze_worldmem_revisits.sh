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

DATA_DIR="${WORLDMEM_DATA_DIR:-data/minecraft}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_ROOT/outputs/memory_policy/metrics/revisit_candidates}"
NUM_VIDEOS="${NUM_VIDEOS:-30}"
FUTURE_SECONDS="${FUTURE_SECONDS:-60}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-600}"
FPS="${FPS:-10}"
SEED="${SEED:-42}"
POS_THRESHOLD="${POS_THRESHOLD:-1.0}"
YAW_THRESHOLD="${YAW_THRESHOLD:-20.0}"
MIN_SEPARATION="${MIN_SEPARATION:-60}"
MAX_PAIRS_PER_VIDEO="${MAX_PAIRS_PER_VIDEO:-20}"
MIN_FUTURE_GAP="${MIN_FUTURE_GAP:-15}"
ROWS="${ROWS:-}"

cd "$WORLDMEM_REPO_ROOT"
export PYTHONPATH="$WORLDMEM_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cmd=(
  python utils/analyze_worldmem_revisits.py
  --data_dir "$DATA_DIR"
  --output_dir "$OUTPUT_DIR"
  --num_videos "$NUM_VIDEOS"
  --future_seconds "$FUTURE_SECONDS"
  --context_frames "$CONTEXT_FRAMES"
  --fps "$FPS"
  --seed "$SEED"
  --pos_threshold "$POS_THRESHOLD"
  --yaw_threshold "$YAW_THRESHOLD"
  --min_separation "$MIN_SEPARATION"
  --max_pairs_per_video "$MAX_PAIRS_PER_VIDEO"
  --min_future_gap "$MIN_FUTURE_GAP"
)

if [ -n "$ROWS" ]; then
  cmd+=(--rows "$ROWS")
fi

echo "WorldMem revisit candidate analysis"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Videos: $NUM_VIDEOS"
echo "Future seconds: $FUTURE_SECONDS"
echo "Context frames: $CONTEXT_FRAMES"
echo "Position threshold: $POS_THRESHOLD"
echo "Yaw threshold: $YAW_THRESHOLD"

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
