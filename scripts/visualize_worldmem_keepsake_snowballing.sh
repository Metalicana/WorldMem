#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${WORLDMEM_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

QUALITY_ROOT="${QUALITY_ROOT:-$STORAGE_ROOT/outputs/memory_quality_60s}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/minecraft}"
UNBOUNDED_RUN="${UNBOUNDED_RUN:-worldmem_memquality_unbounded_60s_n15_seed101}"
FIFO_RUN="${FIFO_RUN:-worldmem_memquality_fifo_b128_60s_n15_seed101}"
KEEPSAKE_RUN="${KEEPSAKE_RUN:-worldmem_memquality_slam_covisibility_b16_60s_n15_seed101}"
CASES_CSV="${CASES_CSV:-$QUALITY_ROOT/metrics/retrieval_failure_gallery_geocov_b16/tables/actual_history_selected_cases.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$QUALITY_ROOT/metrics/keepsake_snowballing_4col}"
MAX_EXAMPLES="${MAX_EXAMPLES:-5}"

echo "WorldMem CPU-only KEEPSAKE snowballing figure"
echo "Ground truth | Unbounded | FIFO B128 | KEEPSAKE B16"
echo "Cases: $CASES_CSV"
echo "Output: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<empty>}"

cd "$REPO_ROOT"
if [ ! -f "$CASES_CSV" ]; then
  echo "Selected cases are missing; building the Unbounded-vs-KEEPSAKE candidate table first."
  QUALITY_ROOT="$QUALITY_ROOT" \
  DATA_DIR="$DATA_DIR" \
  REFERENCE_RUN="$UNBOUNDED_RUN" \
  POLICY_RUN="$KEEPSAKE_RUN" \
  POLICY_DISPLAY="KEEPSAKE B16" \
  OUTPUT_DIR="$(dirname "$(dirname "$CASES_CSV")")" \
  MODES=actual_history \
  bash scripts/visualize_worldmem_retrieval_failures.sh
fi

python utils/visualize_worldmem_keepsake_snowballing.py \
  --quality-root "$QUALITY_ROOT" \
  --data-dir "$DATA_DIR" \
  --unbounded-run "$UNBOUNDED_RUN" \
  --fifo-run "$FIFO_RUN" \
  --keepsake-run "$KEEPSAKE_RUN" \
  --cases-csv "$CASES_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --max-examples "$MAX_EXAMPLES"
