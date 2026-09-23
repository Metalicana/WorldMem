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
OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/memory_policy}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/minecraft}"
METRICS_DIR="${METRICS_DIR:-$OUTPUT_ROOT/metrics/rfid_60s_n15}"
RUNS="${RUNS:-worldmem_unbounded_60s_n30,worldmem_fifo_b32_60s_n30,worldmem_mce_b32_60s_n15,worldmem_kcenter_coreset_b32_60s_n15,worldmem_rarity_irreplaceability_b32_60s_n30,worldmem_slam_covisibility_b32_60s_n30}"
LIMIT="${LIMIT:-15}"
TOTAL_FRAMES="${TOTAL_FRAMES:-5000}"
BATCH_SIZE="${BATCH_SIZE:-32}"

mkdir -p "$METRICS_DIR"

echo "WorldMem reconstruction FID (rFID)"
echo "Output root: $OUTPUT_ROOT"
echo "Data dir: $DATA_DIR"
echo "Runs: $RUNS"
echo "Videos: $LIMIT"
echo "Shared sampled frames: $TOTAL_FRAMES"
echo "GPU: $GPU"
echo "Metrics dir: $METRICS_DIR"

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python "$REPO_ROOT/utils/evaluate_worldmem_rfid.py" \
  --output-root "$OUTPUT_ROOT" \
  --data-dir "$DATA_DIR" \
  --metrics-dir "$METRICS_DIR" \
  --runs "$RUNS" \
  --limit "$LIMIT" \
  --duration-frames 600 \
  --total-frames "$TOTAL_FRAMES" \
  --context-frames 600 \
  --initial-skip-frames 100 \
  --dataset-seed 42 \
  --batch-size "$BATCH_SIZE" \
  --device cuda
