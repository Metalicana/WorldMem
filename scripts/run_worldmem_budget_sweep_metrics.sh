#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${WORLDMEM_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/memory_policy}"
LPIPS_DIR="${LPIPS_DIR:-$OUTPUT_ROOT/metrics/lpips_budget_sweep_60s_n15}"
FVD_DIR="${FVD_DIR:-$OUTPUT_ROOT/metrics/fvd_budget_sweep_60s_n15}"
FIGURE_DIR="${FIGURE_DIR:-$OUTPUT_ROOT/metrics/budget_sweep_figures_60s_n15}"
RUN_LPIPS="${RUN_LPIPS:-1}"
RUN_FVD="${RUN_FVD:-1}"
RUN_PLOT="${RUN_PLOT:-1}"
LIMIT="${LIMIT:-15}"
EVAL_DURATIONS="${EVAL_DURATIONS:-10,20,30,60}"

RUNS="${RUNS:-worldmem_unbounded_60s_n30,\
worldmem_fifo_b16_60s_n30,worldmem_fifo_b32_60s_n30,worldmem_fifo_b64_60s_n30,worldmem_fifo_b128_60s_n30,\
worldmem_rarity_irreplaceability_b16_60s_n30,worldmem_rarity_irreplaceability_b32_60s_n30,worldmem_rarity_irreplaceability_b64_60s_n30,worldmem_rarity_irreplaceability_b128_60s_n30,\
worldmem_slam_covisibility_b16_60s_n30,worldmem_slam_covisibility_b32_60s_n30,worldmem_slam_covisibility_b64_60s_n30,worldmem_slam_covisibility_b128_60s_n30,\
worldmem_kcenter_coreset_b16_60s_n15,worldmem_kcenter_coreset_b32_60s_n15,worldmem_kcenter_coreset_b64_60s_n15,worldmem_kcenter_coreset_b128_60s_n15,\
worldmem_mce_b16_60s_n15,worldmem_mce_b32_60s_n15,worldmem_mce_b64_60s_n15,worldmem_mce_b128_60s_n15}"

cd "$REPO_ROOT"

echo "WorldMem complete memory-budget sweep"
echo "Runs: $RUNS"
echo "Matched videos: $LIMIT"
echo "Durations: $EVAL_DURATIONS"
echo "LPIPS dir: $LPIPS_DIR"
echo "FVD dir: $FVD_DIR"
echo "Figure dir: $FIGURE_DIR"

if [ "$RUN_LPIPS" = "1" ]; then
  WORLDMEM_REPO_ROOT="$REPO_ROOT" \
  WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  METRICS_DIR="$LPIPS_DIR" \
  RUNS="$RUNS" \
  LIMIT="$LIMIT" \
  EVAL_DURATIONS="$EVAL_DURATIONS" \
  bash scripts/evaluate_worldmem_lpips.sh
fi

if [ "$RUN_FVD" = "1" ]; then
  WORLDMEM_REPO_ROOT="$REPO_ROOT" \
  WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  METRICS_DIR="$FVD_DIR" \
  RUNS="$RUNS" \
  LIMIT="$LIMIT" \
  EVAL_DURATIONS="$EVAL_DURATIONS" \
  bash scripts/evaluate_worldmem_fvd.sh
fi

if [ "$RUN_PLOT" = "1" ]; then
  python utils/plot_worldmem_budget_sweep.py \
    --lpips-summary "$LPIPS_DIR/summary.csv" \
    --fvd-summary "$FVD_DIR/summary.csv" \
    --output-dir "$FIGURE_DIR" \
    --duration 60 \
    --limit "$LIMIT"
fi
