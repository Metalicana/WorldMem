#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

POLICIES_CSV="${POLICIES:-unbounded,fifo,rarity_irreplaceability,slam_covisibility}"
BUDGETS_CSV="${BUDGETS:-16,32,64,128}"
DURATIONS_CSV="${DURATIONS:-180}"
TOTAL_VIDEOS="${TOTAL_VIDEOS:-${NUM_VIDEOS:-15}}"
START_VIDEO="${START_VIDEO:-1}"
END_VIDEO="${END_VIDEO:-$TOTAL_VIDEOS}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-600}"
FPS="${FPS:-10}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
CHECK_HORIZON="${CHECK_HORIZON:-1}"
DATA_DIR="${WORLDMEM_DATA_DIR:-data/minecraft}"

IFS=',' read -r -a POLICIES_ARRAY <<< "$POLICIES_CSV"
IFS=',' read -r -a BUDGETS_ARRAY <<< "$BUDGETS_CSV"
IFS=',' read -r -a DURATIONS_ARRAY <<< "$DURATIONS_CSV"

if [ "$CHECK_HORIZON" = "1" ] || [ "$CHECK_HORIZON" = "true" ]; then
  max_duration=0
  for duration in "${DURATIONS_ARRAY[@]}"; do
    duration="${duration//[[:space:]]/}"
    [ -n "$duration" ] || continue
    if [ "$duration" -gt "$max_duration" ]; then
      max_duration="$duration"
    fi
  done

  echo "Checking WorldMem horizon availability before launching round-robin grid."
  python utils/check_worldmem_horizon_availability.py \
    --data_dir "$DATA_DIR" \
    --future_seconds "$max_duration" \
    --context_frames "$CONTEXT_FRAMES" \
    --fps "$FPS" \
    --num_videos "$TOTAL_VIDEOS" \
    --strict
fi

run_one() {
  local policy="$1"
  local budget="$2"
  local duration="$3"
  local target_count="$4"
  local run_name

  if [ -n "$budget" ]; then
    run_name="worldmem_${policy}_b${budget}_${duration}s_n${TOTAL_VIDEOS}"
  else
    run_name="worldmem_${policy}_${duration}s_n${TOTAL_VIDEOS}"
  fi

  echo "============================================================"
  echo "Round-robin target: $target_count / $TOTAL_VIDEOS"
  echo "Run: $run_name"
  echo "Policy: $policy"
  echo "Budget: ${budget:-none}"
  echo "Future seconds: $duration"
  echo "============================================================"

  if [ -n "$budget" ]; then
    MEMORY_POLICY="$policy" \
    MEMORY_BUDGET="$budget" \
    FUTURE_SECONDS="$duration" \
    NUM_VIDEOS="$target_count" \
    CONTEXT_FRAMES="$CONTEXT_FRAMES" \
    FPS="$FPS" \
    RUN_NAME="$run_name" \
    bash "$SCRIPT_DIR/run_worldmem_memory_policy_smoke.sh"
  else
    MEMORY_POLICY="$policy" \
    MEMORY_BUDGET="" \
    FUTURE_SECONDS="$duration" \
    NUM_VIDEOS="$target_count" \
    CONTEXT_FRAMES="$CONTEXT_FRAMES" \
    FPS="$FPS" \
    RUN_NAME="$run_name" \
    bash "$SCRIPT_DIR/run_worldmem_memory_policy_smoke.sh"
  fi
}

for target_count in $(seq "$START_VIDEO" "$END_VIDEO"); do
  echo "================ ROUND-ROBIN SWEEP: $target_count / $TOTAL_VIDEOS ================"

  for duration in "${DURATIONS_ARRAY[@]}"; do
    duration="${duration//[[:space:]]/}"
    [ -n "$duration" ] || continue

    for policy in "${POLICIES_ARRAY[@]}"; do
      policy="${policy//[[:space:]]/}"
      [ -n "$policy" ] || continue

      if [ "$policy" = "unbounded" ]; then
        if ! run_one "$policy" "" "$duration" "$target_count" && [ "$CONTINUE_ON_ERROR" != "1" ]; then
          exit 1
        fi
        continue
      fi

      for budget in "${BUDGETS_ARRAY[@]}"; do
        budget="${budget//[[:space:]]/}"
        [ -n "$budget" ] || continue
        if ! run_one "$policy" "$budget" "$duration" "$target_count" && [ "$CONTINUE_ON_ERROR" != "1" ]; then
          exit 1
        fi
      done
    done
  done

  echo "================ DONE ROUND-ROBIN SWEEP: $target_count / $TOTAL_VIDEOS ================"
done
