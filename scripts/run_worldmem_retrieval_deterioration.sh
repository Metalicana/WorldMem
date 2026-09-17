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
QUALITY_ROOT="${QUALITY_ROOT:-$STORAGE_ROOT/outputs/memory_quality_60s}"
RUN_NAME="${RUN_NAME:-worldmem_memquality_unbounded_60s_n15_seed101}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/minecraft}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$QUALITY_ROOT/metrics/retrieval_deterioration_unbounded_60s}"
TABLE_DIR="$OUTPUT_ROOT/tables"
FIGURE_DIR="$OUTPUT_ROOT/figure"
CACHE_DIR="${CACHE_DIR:-$OUTPUT_ROOT/feature_cache}"
LIMIT="${LIMIT:-15}"
DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-64}"
MODEL_NAME="${MODEL_NAME:-facebook/dinov2-base}"
MODEL_REVISION="${MODEL_REVISION:-main}"

mkdir -p "$TABLE_DIR" "$FIGURE_DIR" "$CACHE_DIR"
export HF_HOME="${HF_HOME:-$STORAGE_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$STORAGE_ROOT/tmp/matplotlib}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$MPLCONFIGDIR"
cd "$REPO_ROOT"

echo "WorldMem retrieval-deterioration diagnostic"
echo "Run: $RUN_NAME"
echo "Trajectories: $LIMIT"
echo "DINO device: cuda:0 (physical GPU selected by CUDA_VISIBLE_DEVICES=$GPU)"
echo "Output: $OUTPUT_ROOT"

CUDA_VISIBLE_DEVICES="$GPU" python utils/export_worldmem_retrieval_deterioration.py \
  --quality-root "$QUALITY_ROOT" \
  --run-name "$RUN_NAME" \
  --data-dir "$DATA_DIR" \
  --output-dir "$TABLE_DIR" \
  --cache-dir "$CACHE_DIR" \
  --limit "$LIMIT" \
  --duration-sec 60 \
  --fps 10 \
  --context-frames 600 \
  --initial-skip-frames 100 \
  --dataset-seed 42 \
  --model-name "$MODEL_NAME" \
  --model-revision "$MODEL_REVISION" \
  --device cuda:0 \
  --batch-size "$DINO_BATCH_SIZE"

# Aggregation and plotting are deliberately CPU-only.
CUDA_VISIBLE_DEVICES="" python utils/plot_worldmem_retrieval_deterioration.py \
  --input "$TABLE_DIR/query_decomposition.csv" \
  --output "$FIGURE_DIR" \
  --run "$RUN_NAME" \
  --duration 60 \
  --expected-videos "$LIMIT" \
  --fps 10 \
  --bins 8 \
  --bootstrap 10000 \
  --seed 0 \
  --overwrite

echo "Done."
echo "Figure: $FIGURE_DIR/retrieval_deterioration.png"
echo "Changes: $FIGURE_DIR/changes.csv"
