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
NUM_VIDEOS="${NUM_VIDEOS:-15}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/retrieval_latency_60s_n${NUM_VIDEOS}}"
GLOBAL_SEED="${GLOBAL_SEED:-101}"
DATASET_SEED="${DATASET_SEED:-42}"
POLICY_SPECS="${POLICY_SPECS:-unbounded: slam_covisibility:32}"
read -r -a specs <<< "$POLICY_SPECS"

mkdir -p "$OUTPUT_ROOT" "$STORAGE_ROOT/logs"

for spec in "${specs[@]}"; do
  policy="${spec%%:*}"
  budget="${spec#*:}"
  budget_tag=""
  [ -z "$budget" ] || budget_tag="_b${budget}"
  run_name="worldmem_retrieval_latency_${policy}${budget_tag}_60s_n${NUM_VIDEOS}_seed${GLOBAL_SEED}"
  echo "============================================================"
  echo "Retrieval latency: $run_name"
  echo "GPU: $GPU  policy: $policy  budget: ${budget:-none}"
  echo "============================================================"
  GPU="$GPU" \
  WORLDMEM_REPO_ROOT="$REPO_ROOT" \
  WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
  MEMORY_POLICY="$policy" \
  MEMORY_BUDGET="$budget" \
  MEMORY_BANK_DEVICE=cpu \
  MEMORY_REFERENCE_SOURCE=predicted \
  MEMORY_FEATURE_BACKEND=latent \
  GLOBAL_SEED="$GLOBAL_SEED" \
  GENERATION_SEED="$GLOBAL_SEED" \
  MEMORY_POLICY_SEED="$GLOBAL_SEED" \
  DATASET_SEED="$DATASET_SEED" \
  FUTURE_SECONDS=60 \
  LIMIT_BATCH="$NUM_VIDEOS" \
  RUN_NAME="$run_name" \
  OUTPUT_DIR="$OUTPUT_ROOT/$run_name" \
  TRACE_PATH="$OUTPUT_ROOT/$run_name/access_traces/$run_name.jsonl" \
  PROFILE_RETRIEVAL_QUERIES=true \
  PROFILE_TIMING=false \
  PROFILE_CUDA_MEMORY=false \
  TRACE_CANDIDATE_DIAGNOSTICS=false \
  TRACE_BANK_STATE=false \
  TRACE_RETRIEVED_MEMORY_QUALITY=false \
  COMPUTE_EVAL_METRICS=false \
  STREAM_EVAL_METRICS=false \
  SAVE_GT_VIDEO=false \
  WANDB_MODE=disabled \
  bash "$SCRIPT_DIR/run_worldmem_memory_policy_smoke.sh"
done

if [ "${SUMMARIZE_ALL_POLICIES:-false}" = "true" ]; then
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python -m utils.summarize_worldmem_retrieval_latency_roster \
    --output-root "$OUTPUT_ROOT" \
    --expected-videos "$NUM_VIDEOS" \
    --seed "$GLOBAL_SEED" \
    --policy-specs "$POLICY_SPECS"
  exit 0
fi

unbounded_name="worldmem_retrieval_latency_unbounded_60s_n${NUM_VIDEOS}_seed${GLOBAL_SEED}"
keepsake_name="worldmem_retrieval_latency_slam_covisibility_b32_60s_n${NUM_VIDEOS}_seed${GLOBAL_SEED}"
CUDA_VISIBLE_DEVICES="" python "$REPO_ROOT/utils/summarize_worldmem_retrieval_latency.py" \
  --unbounded-trace "$OUTPUT_ROOT/$unbounded_name/access_traces/$unbounded_name.jsonl" \
  --keepsake-trace "$OUTPUT_ROOT/$keepsake_name/access_traces/$keepsake_name.jsonl" \
  --output-dir "$OUTPUT_ROOT/summary" \
  --expected-videos "$NUM_VIDEOS"

echo "Done: $OUTPUT_ROOT/summary/worldmem_retrieval_latency.csv"
