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

TRACE_PATHS="${TRACE_PATHS:-$STORAGE_ROOT/outputs/memory_policy/worldmem_causal_gate_shadow_b32_60s_n15/access_traces/*.jsonl}"
CALIBRATION_DIR="${CALIBRATION_DIR:-$STORAGE_ROOT/outputs/memory_policy/metrics/causal_gate_b32_60s}"

cd "$WORLDMEM_REPO_ROOT"
mkdir -p "$CALIBRATION_DIR"

echo "WorldMem causal-consistency gate calibration"
echo "Trace paths: $TRACE_PATHS"
echo "Calibration dir: $CALIBRATION_DIR"

python utils/calibrate_worldmem_causal_gate.py \
  --trace_paths "$TRACE_PATHS" \
  --output_dir "$CALIBRATION_DIR" \
  --pose_bins "${POSE_BINS:-4}" \
  --bad_quantile "${BAD_QUANTILE:-0.20}" \
  --test_fraction "${TEST_FRACTION:-0.3333333333}" \
  --split_seed "${SPLIT_SEED:-17}" \
  --max_train_clean_false_reject "${MAX_TRAIN_CLEAN_FALSE_REJECT:-0.10}" \
  --min_heldout_auc "${MIN_HELDOUT_AUC:-0.70}" \
  --min_bad_precision "${MIN_BAD_PRECISION:-0.50}" \
  --min_bad_recall "${MIN_BAD_RECALL:-0.20}" \
  --max_heldout_clean_false_reject "${MAX_HELDOUT_CLEAN_FALSE_REJECT:-0.15}"
