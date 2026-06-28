#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

POLICIES_CSV="${POLICIES:-unbounded,fifo,rarity_irreplaceability,slam_covisibility}"
BUDGETS_CSV="${BUDGETS:-32,64}"
DURATIONS_CSV="${DURATIONS:-10,20,30,60}"
NUM_VIDEOS="${NUM_VIDEOS:-30}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-600}"
FPS="${FPS:-10}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

IFS=',' read -r -a POLICIES_ARRAY <<< "$POLICIES_CSV"
IFS=',' read -r -a BUDGETS_ARRAY <<< "$BUDGETS_CSV"
IFS=',' read -r -a DURATIONS_ARRAY <<< "$DURATIONS_CSV"

run_one() {
  local policy="$1"
  local budget="$2"
  local duration="$3"
  local run_name

  if [ -n "$budget" ]; then
    run_name="worldmem_${policy}_b${budget}_${duration}s_n${NUM_VIDEOS}"
  else
    run_name="worldmem_${policy}_${duration}s_n${NUM_VIDEOS}"
  fi

  echo "============================================================"
  echo "Run: $run_name"
  echo "Policy: $policy"
  echo "Budget: ${budget:-none}"
  echo "Future seconds: $duration"
  echo "Videos: $NUM_VIDEOS"
  echo "============================================================"

  if [ -n "$budget" ]; then
    MEMORY_POLICY="$policy" \
    MEMORY_BUDGET="$budget" \
    FUTURE_SECONDS="$duration" \
    NUM_VIDEOS="$NUM_VIDEOS" \
    CONTEXT_FRAMES="$CONTEXT_FRAMES" \
    FPS="$FPS" \
    RUN_NAME="$run_name" \
    bash "$SCRIPT_DIR/run_worldmem_memory_policy_smoke.sh"
  else
    MEMORY_POLICY="$policy" \
    MEMORY_BUDGET="" \
    FUTURE_SECONDS="$duration" \
    NUM_VIDEOS="$NUM_VIDEOS" \
    CONTEXT_FRAMES="$CONTEXT_FRAMES" \
    FPS="$FPS" \
    RUN_NAME="$run_name" \
    bash "$SCRIPT_DIR/run_worldmem_memory_policy_smoke.sh"
  fi
}

for duration in "${DURATIONS_ARRAY[@]}"; do
  duration="${duration//[[:space:]]/}"
  [ -n "$duration" ] || continue

  for policy in "${POLICIES_ARRAY[@]}"; do
    policy="${policy//[[:space:]]/}"
    [ -n "$policy" ] || continue

    if [ "$policy" = "unbounded" ]; then
      if ! run_one "$policy" "" "$duration" && [ "$CONTINUE_ON_ERROR" != "1" ]; then
        exit 1
      fi
      continue
    fi

    for budget in "${BUDGETS_ARRAY[@]}"; do
      budget="${budget//[[:space:]]/}"
      [ -n "$budget" ] || continue
      if ! run_one "$policy" "$budget" "$duration" && [ "$CONTINUE_ON_ERROR" != "1" ]; then
        exit 1
      fi
    done
  done
done
