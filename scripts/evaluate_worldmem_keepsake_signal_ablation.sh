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

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/memory_policy}"
METRICS_ROOT="${METRICS_ROOT:-$OUTPUT_ROOT/metrics/keepsake_signal_ablation_60s_n15}"
LIMIT="${LIMIT:-15}"
CONTROL_RUN="${CONTROL_RUN:-worldmem_slam_covisibility_b32_60s_n30}"
RUNS="${RUNS:-$CONTROL_RUN,worldmem_keepsake_pose_only_b32_60s_n15,worldmem_keepsake_appearance_only_b32_60s_n15,worldmem_keepsake_recompute_b32_60s_n15}"
EVALUATORS="${EVALUATORS:-lpips,fvd,rfid}"

mkdir -p "$METRICS_ROOT"
IFS=',' read -r -a EVALUATOR_ARRAY <<< "$EVALUATORS"

echo "WorldMem KEEPSAKE signal/priority ablation evaluation"
echo "Runs: $RUNS"
echo "Matched videos: $LIMIT"
echo "Evaluators: $EVALUATORS"
echo "Metrics root: $METRICS_ROOT"

for evaluator in "${EVALUATOR_ARRAY[@]}"; do
  evaluator="${evaluator//[[:space:]]/}"
  case "$evaluator" in
    lpips)
      GPU="$GPU" \
      WORLDMEM_REPO_ROOT="$WORLDMEM_REPO_ROOT" \
      WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      RUNS="$RUNS" \
      LIMIT="$LIMIT" \
      EVAL_DURATIONS=60 \
      METRICS_DIR="$METRICS_ROOT/lpips" \
      bash "$SCRIPT_DIR/evaluate_worldmem_lpips.sh"
      ;;
    fvd)
      GPU="$GPU" \
      WORLDMEM_REPO_ROOT="$WORLDMEM_REPO_ROOT" \
      WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      RUNS="$RUNS" \
      LIMIT="$LIMIT" \
      EVAL_DURATIONS=60 \
      METRICS_DIR="$METRICS_ROOT/fvd" \
      bash "$SCRIPT_DIR/evaluate_worldmem_fvd.sh"
      ;;
    rfid)
      GPU="$GPU" \
      WORLDMEM_REPO_ROOT="$WORLDMEM_REPO_ROOT" \
      WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      RUNS="$RUNS" \
      LIMIT="$LIMIT" \
      METRICS_DIR="$METRICS_ROOT/rfid" \
      bash "$SCRIPT_DIR/evaluate_worldmem_rfid.sh"
      ;;
    "")
      ;;
    *)
      echo "Unknown evaluator '$evaluator'. Expected lpips, fvd, or rfid." >&2
      exit 2
      ;;
  esac
done

echo "Completed KEEPSAKE signal/priority ablation evaluation."
