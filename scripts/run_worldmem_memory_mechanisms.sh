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

FUTURE_SECONDS="${FUTURE_SECONDS:-60}"
NUM_VIDEOS="${NUM_VIDEOS:-5}"
SEEDS_CSV="${SEEDS:-101}"
DATASET_SEED="${DATASET_SEED:-42}"
CANDIDATE_CAPS_CSV="${CANDIDATE_CAPS:-16,32,64,128,256,512}"
RANDOM_BANK_BUDGETS_CSV="${RANDOM_BANK_BUDGETS:-32}"
SMART_POLICIES_CSV="${SMART_POLICIES:-rarity_irreplaceability,slam_covisibility}"
SMART_BUDGETS_CSV="${SMART_BUDGETS:-32}"
REFERENCE_SOURCES_CSV="${REFERENCE_SOURCES:-predicted}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_CANDIDATE_CAPS="${RUN_CANDIDATE_CAPS:-1}"
RUN_RANDOM_BANK="${RUN_RANDOM_BANK:-1}"
RUN_SMART_POLICIES="${RUN_SMART_POLICIES:-1}"
RUN_DINO_ABLATION="${RUN_DINO_ABLATION:-0}"
DINO_BACKENDS_CSV="${DINO_BACKENDS:-latent,dino,dino_rgb}"
DINO_BUDGET="${DINO_BUDGET:-32}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/memory_diagnostics}"
DATA_DIR="${WORLDMEM_DATA_DIR:-data/minecraft}"

TRACE_CANDIDATE_TOP_K="${TRACE_CANDIDATE_TOP_K:-16}"
TRACE_CANDIDATE_SAMPLE_SIZE="${TRACE_CANDIDATE_SAMPLE_SIZE:-16}"
TRACE_BANK_MAX_FRAMES="${TRACE_BANK_MAX_FRAMES:-256}"

IFS=',' read -r -a SEEDS_ARRAY <<< "$SEEDS_CSV"
IFS=',' read -r -a CANDIDATE_CAPS_ARRAY <<< "$CANDIDATE_CAPS_CSV"
IFS=',' read -r -a RANDOM_BANK_BUDGETS_ARRAY <<< "$RANDOM_BANK_BUDGETS_CSV"
IFS=',' read -r -a SMART_POLICIES_ARRAY <<< "$SMART_POLICIES_CSV"
IFS=',' read -r -a SMART_BUDGETS_ARRAY <<< "$SMART_BUDGETS_CSV"
IFS=',' read -r -a REFERENCE_SOURCES_ARRAY <<< "$REFERENCE_SOURCES_CSV"
IFS=',' read -r -a DINO_BACKENDS_ARRAY <<< "$DINO_BACKENDS_CSV"

mkdir -p "$OUTPUT_ROOT"

run_one() {
  local policy="$1"
  local budget="$2"
  local seed="$3"
  local reference_source="$4"
  local feature_backend="$5"
  local retrieval_candidate_cap="${6:-}"
  local budget_tag=""
  local source_tag=""
  local feature_tag=""
  local candidate_tag=""
  local run_name

  if [ -n "$budget" ]; then
    budget_tag="_b${budget}"
  fi
  if [ "$reference_source" != "predicted" ]; then
    source_tag="_${reference_source}"
  fi
  if [ "$feature_backend" != "latent" ]; then
    feature_tag="_${feature_backend}"
  fi
  if [ -n "$retrieval_candidate_cap" ]; then
    candidate_tag="_candidatecap${retrieval_candidate_cap}"
  fi
  run_name="worldmem_diag_${policy}${budget_tag}${candidate_tag}${feature_tag}${source_tag}_${FUTURE_SECONDS}s_n${NUM_VIDEOS}_seed${seed}"

  echo "============================================================"
  echo "Diagnostic run: $run_name"
  echo "Policy: $policy  Budget: ${budget:-none}  Candidate cap: ${retrieval_candidate_cap:-none}"
  echo "Reference source: $reference_source  Features: $feature_backend"
  echo "Seed: $seed  Videos: $NUM_VIDEOS  Future: ${FUTURE_SECONDS}s"
  echo "============================================================"

  if ! WORLDMEM_REPO_ROOT="$REPO_ROOT" \
    WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
    WORLDMEM_DATA_DIR="$DATA_DIR" \
    GPU="$GPU" \
    MEMORY_POLICY="$policy" \
    MEMORY_BUDGET="$budget" \
    MEMORY_REFERENCE_SOURCE="$reference_source" \
    MEMORY_FEATURE_BACKEND="$feature_backend" \
    DATASET_SEED="$DATASET_SEED" \
    GLOBAL_SEED="$seed" \
    GENERATION_SEED="$seed" \
    MEMORY_POLICY_SEED="$seed" \
    RETRIEVAL_CANDIDATE_CAP="$retrieval_candidate_cap" \
    FUTURE_SECONDS="$FUTURE_SECONDS" \
    NUM_VIDEOS="$NUM_VIDEOS" \
    RUN_NAME="$run_name" \
    OUTPUT_DIR="$OUTPUT_ROOT/$run_name" \
    TRACE_PATH="$OUTPUT_ROOT/$run_name/access_traces/$run_name.jsonl" \
    TRACE_CANDIDATE_DIAGNOSTICS=true \
    TRACE_CANDIDATE_TOP_K="$TRACE_CANDIDATE_TOP_K" \
    TRACE_CANDIDATE_SAMPLE_SIZE="$TRACE_CANDIDATE_SAMPLE_SIZE" \
    TRACE_BANK_STATE=true \
    TRACE_BANK_MAX_FRAMES="$TRACE_BANK_MAX_FRAMES" \
    SAVE_LOCAL_PER_BATCH=true \
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
}

for seed in "${SEEDS_ARRAY[@]}"; do
  seed="${seed//[[:space:]]/}"
  [ -n "$seed" ] || continue
  for reference_source in "${REFERENCE_SOURCES_ARRAY[@]}"; do
    reference_source="${reference_source//[[:space:]]/}"
    [ -n "$reference_source" ] || continue

    if [ "$RUN_BASELINE" = "1" ] || [ "$RUN_BASELINE" = "true" ]; then
      run_one unbounded "" "$seed" "$reference_source" latent ""
    fi

    if [ "$RUN_CANDIDATE_CAPS" = "1" ] || [ "$RUN_CANDIDATE_CAPS" = "true" ]; then
      for cap in "${CANDIDATE_CAPS_ARRAY[@]}"; do
        cap="${cap//[[:space:]]/}"
        [ -n "$cap" ] || continue
        run_one unbounded "" "$seed" "$reference_source" latent "$cap"
      done
    fi

    if [ "$RUN_RANDOM_BANK" = "1" ] || [ "$RUN_RANDOM_BANK" = "true" ]; then
      for budget in "${RANDOM_BANK_BUDGETS_ARRAY[@]}"; do
        budget="${budget//[[:space:]]/}"
        [ -n "$budget" ] || continue
        run_one random_cap "$budget" "$seed" "$reference_source" latent ""
      done
    fi

    if [ "$RUN_SMART_POLICIES" = "1" ] || [ "$RUN_SMART_POLICIES" = "true" ]; then
      for policy in "${SMART_POLICIES_ARRAY[@]}"; do
        policy="${policy//[[:space:]]/}"
        [ -n "$policy" ] || continue
        for budget in "${SMART_BUDGETS_ARRAY[@]}"; do
          budget="${budget//[[:space:]]/}"
          [ -n "$budget" ] || continue
          run_one "$policy" "$budget" "$seed" "$reference_source" latent ""
        done
      done
    fi

    if [ "$RUN_DINO_ABLATION" = "1" ] || [ "$RUN_DINO_ABLATION" = "true" ]; then
      for backend in "${DINO_BACKENDS_ARRAY[@]}"; do
        backend="${backend//[[:space:]]/}"
        [ -n "$backend" ] || continue
        run_one rarity_irreplaceability "$DINO_BUDGET" "$seed" "$reference_source" "$backend" ""
      done
    fi
  done
done

echo "Completed WorldMem memory mechanism grid."
echo "Output root: $OUTPUT_ROOT"
