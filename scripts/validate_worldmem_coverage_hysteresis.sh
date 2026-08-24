#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORLDMEM_REPO_ROOT="${WORLDMEM_REPO_ROOT:-$DEFAULT_REPO_ROOT}"
if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

RUN_NAME="${RUN_NAME:-worldmem_unbounded_60s_n30}"
RUN_DIR="${RUN_DIR:-$STORAGE_ROOT/outputs/memory_policy/$RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_ROOT/outputs/memory_policy/metrics/coverage_hysteresis_unbounded_60s}"
DATA_DIR="${WORLDMEM_DATA_DIR:-data/minecraft}"

cd "$WORLDMEM_REPO_ROOT"
export PYTHONPATH="$WORLDMEM_REPO_ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"

echo "WorldMem coverage-hysteresis validation"
echo "Run dir: $RUN_DIR"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "GPU: $GPU"

python utils/validate_worldmem_coverage_hysteresis.py \
  --run_dir "$RUN_DIR" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --limit "${LIMIT:-30}" \
  --dataset_seed "${DATASET_SEED:-42}" \
  --context_frames "${CONTEXT_FRAMES:-600}" \
  --future_seconds "${FUTURE_SECONDS:-60}" \
  --fps "${FPS:-10}" \
  --initial_skip_frames "${INITIAL_SKIP_FRAMES:-100}" \
  --coverage_thresholds "${COVERAGE_THRESHOLDS:-0.80,0.85,0.90,0.95}" \
  --min_chunk_separation "${MIN_CHUNK_SEPARATION:-2}" \
  --radius "${RADIUS:-30}" \
  --fov_half_h "${FOV_HALF_H:-52.5}" \
  --fov_half_v "${FOV_HALF_V:-37.5}" \
  --device "${DEVICE:-cuda}" \
  --metric_batch_size "${METRIC_BATCH_SIZE:-64}" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES:-10000}" \
  --bootstrap_seed "${BOOTSTRAP_SEED:-17}" \
  --fallback_chunk_size "${FALLBACK_CHUNK_SIZE:-1}"
