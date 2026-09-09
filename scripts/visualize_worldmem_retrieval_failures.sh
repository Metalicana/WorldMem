#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${WORLDMEM_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

# This analysis decodes MP4s and computes PSNR/SSIM on CPU. Hiding CUDA makes
# accidental GPU initialization immediately visible if a future dependency
# introduces it.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

QUALITY_ROOT="${QUALITY_ROOT:-$STORAGE_ROOT/outputs/memory_quality_60s}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/minecraft}"
REFERENCE_RUN="${REFERENCE_RUN:-worldmem_memquality_unbounded_60s_n15_seed101}"
POLICY_RUN="${POLICY_RUN:-worldmem_memquality_slam_covisibility_b16_60s_n15_seed101}"
POLICY_DISPLAY="${POLICY_DISPLAY:-Geometric Coverage B16}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_ROOT/outputs/memory_quality_60s/metrics/retrieval_failure_gallery_geocov_b16}"
MODES="${MODES:-actual_history,common_source}"
LIMIT="${LIMIT:-15}"
LATE_START_SEC="${LATE_START_SEC:-45}"
MAX_EXAMPLES="${MAX_EXAMPLES:-5}"
PER_BATCH="${PER_BATCH:-1}"
MIN_TARGET_GAP="${MIN_TARGET_GAP:-80}"

echo "WorldMem CPU-only retrieval-failure gallery"
echo "Quality root: $QUALITY_ROOT"
echo "Data dir: $DATA_DIR"
echo "Reference: $REFERENCE_RUN"
echo "Policy: $POLICY_RUN"
echo "Modes: $MODES"
echo "Window: ${LATE_START_SEC}-60s"
echo "Output dir: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<empty>}"

cd "$REPO_ROOT"
python utils/visualize_worldmem_retrieval_failures.py \
  --quality_root "$QUALITY_ROOT" \
  --data_dir "$DATA_DIR" \
  --reference_run "$REFERENCE_RUN" \
  --policy_run "$POLICY_RUN" \
  --policy_display "$POLICY_DISPLAY" \
  --output_dir "$OUTPUT_DIR" \
  --modes "$MODES" \
  --limit "$LIMIT" \
  --late_start_sec "$LATE_START_SEC" \
  --max_examples "$MAX_EXAMPLES" \
  --per_batch "$PER_BATCH" \
  --min_target_gap "$MIN_TARGET_GAP"
