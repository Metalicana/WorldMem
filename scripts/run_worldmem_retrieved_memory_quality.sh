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
OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/memory_quality_60s}"
POLICY_SPECS="${POLICY_SPECS:-unbounded:,fifo:128,rarity_irreplaceability:32,slam_covisibility:16}"
NUM_VIDEOS="${NUM_VIDEOS:-15}"
FUTURE_SECONDS="${FUTURE_SECONDS:-60}"
GLOBAL_SEED="${GLOBAL_SEED:-101}"
DATASET_SEED="${DATASET_SEED:-42}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

if [ "$FUTURE_SECONDS" != "60" ]; then
  echo "This experiment is defined for complete 60-second WorldMem videos." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$STORAGE_ROOT/logs"
IFS=',' read -r -a SPECS <<< "$POLICY_SPECS"

for spec in "${SPECS[@]}"; do
  if [[ "$spec" == *:* ]]; then
    policy="${spec%%:*}"
    budget="${spec#*:}"
  else
    policy="$spec"
    budget=""
  fi
  policy="${policy//[[:space:]]/}"
  budget="${budget//[[:space:]]/}"
  [ -n "$policy" ] || continue
  budget_tag=""
  if [ -n "$budget" ]; then
    budget_tag="_b${budget}"
  fi
  run_name="worldmem_memquality_${policy}${budget_tag}_60s_n${NUM_VIDEOS}_seed${GLOBAL_SEED}"
  echo "============================================================"
  echo "Retrieved-memory quality: $run_name"
  echo "Policy: $policy  Budget: ${budget:-none}  GPU: $GPU"
  echo "============================================================"

  if ! GPU="$GPU" \
    WORLDMEM_REPO_ROOT="$REPO_ROOT" \
    WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
    MEMORY_POLICY="$policy" \
    MEMORY_BUDGET="$budget" \
    MEMORY_REFERENCE_SOURCE=predicted \
    GLOBAL_SEED="$GLOBAL_SEED" \
    GENERATION_SEED="$GLOBAL_SEED" \
    MEMORY_POLICY_SEED="$GLOBAL_SEED" \
    DATASET_SEED="$DATASET_SEED" \
    FUTURE_SECONDS=60 \
    NUM_VIDEOS="$NUM_VIDEOS" \
    LIMIT_BATCH="$NUM_VIDEOS" \
    RUN_NAME="$run_name" \
    OUTPUT_DIR="$OUTPUT_ROOT/$run_name" \
    TRACE_PATH="$OUTPUT_ROOT/$run_name/access_traces/$run_name.jsonl" \
    TRACE_RETRIEVED_MEMORY_QUALITY=true \
    TRACE_CANDIDATE_DIAGNOSTICS=false \
    TRACE_BANK_STATE=false \
    SAVE_GT_VIDEO=false \
    COMPUTE_EVAL_METRICS=false \
    PROFILE_TIMING=true \
    WANDB_MODE=disabled \
    bash "$SCRIPT_DIR/run_worldmem_memory_policy_smoke.sh"
  then
    if [ "$CONTINUE_ON_ERROR" != "1" ]; then
      exit 1
    fi
  fi
done

echo "Completed retrieved-memory quality runs."
echo "Output root: $OUTPUT_ROOT"
