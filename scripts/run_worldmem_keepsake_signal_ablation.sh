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
TOTAL_VIDEOS="${TOTAL_VIDEOS:-${NUM_VIDEOS:-15}}"
START_VIDEO="${START_VIDEO:-1}"
END_VIDEO="${END_VIDEO:-$TOTAL_VIDEOS}"
FUTURE_SECONDS="${FUTURE_SECONDS:-60}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-600}"
FPS="${FPS:-10}"
MEMORY_BUDGET="${MEMORY_BUDGET:-32}"
DATA_DIR="${WORLDMEM_DATA_DIR:-data/minecraft}"
DATASET_SEED="${DATASET_SEED:-42}"
GLOBAL_SEED="${GLOBAL_SEED:-42}"
GENERATION_SEED="${GENERATION_SEED:-$GLOBAL_SEED}"
MEMORY_POLICY_SEED="${MEMORY_POLICY_SEED:-$GLOBAL_SEED}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
CHECK_HORIZON="${CHECK_HORIZON:-1}"
ABLATIONS_CSV="${ABLATIONS:-pose_only,appearance_only,recompute}"
CONTROL_RUN="${CONTROL_RUN:-worldmem_slam_covisibility_b32_60s_n30}"

IFS=',' read -r -a ABLATIONS_ARRAY <<< "$ABLATIONS_CSV"

if [ "$CHECK_HORIZON" = "1" ] || [ "$CHECK_HORIZON" = "true" ]; then
  cd "$WORLDMEM_REPO_ROOT"
  python utils/check_worldmem_horizon_availability.py \
    --data_dir "$DATA_DIR" \
    --future_seconds "$FUTURE_SECONDS" \
    --context_frames "$CONTEXT_FRAMES" \
    --fps "$FPS" \
    --num_videos "$TOTAL_VIDEOS" \
    --strict
fi

control_dir="$STORAGE_ROOT/outputs/memory_policy/$CONTROL_RUN/videos/test_vis/pred"
control_count=0
if [ -d "$control_dir" ]; then
  control_count="$(find "$control_dir" -maxdepth 1 -type f -name '*.mp4' -size +4k | wc -l | tr -d ' ')"
fi
echo "Existing full-signal frozen control: $CONTROL_RUN ($control_count videos found)"
echo "Ablations: $ABLATIONS_CSV"
echo "Round-robin targets: $START_VIDEO through $END_VIDEO of $TOTAL_VIDEOS"

run_variant() {
  local variant="$1"
  local target_count="$2"
  local geometry_weight
  local visual_weight
  local priority_update
  local run_name

  case "$variant" in
    pose_only)
      geometry_weight="1.0"
      visual_weight="0.0"
      priority_update="frozen"
      run_name="worldmem_keepsake_pose_only_b${MEMORY_BUDGET}_${FUTURE_SECONDS}s_n${TOTAL_VIDEOS}"
      ;;
    appearance_only)
      geometry_weight="0.0"
      visual_weight="1.0"
      priority_update="frozen"
      run_name="worldmem_keepsake_appearance_only_b${MEMORY_BUDGET}_${FUTURE_SECONDS}s_n${TOTAL_VIDEOS}"
      ;;
    recompute)
      geometry_weight="0.65"
      visual_weight="0.35"
      priority_update="recompute"
      run_name="worldmem_keepsake_recompute_b${MEMORY_BUDGET}_${FUTURE_SECONDS}s_n${TOTAL_VIDEOS}"
      ;;
    full_frozen)
      geometry_weight="0.65"
      visual_weight="0.35"
      priority_update="frozen"
      run_name="worldmem_keepsake_full_frozen_b${MEMORY_BUDGET}_${FUTURE_SECONDS}s_n${TOTAL_VIDEOS}"
      ;;
    *)
      echo "Unknown ablation '$variant'. Expected pose_only, appearance_only, recompute, or full_frozen." >&2
      return 2
      ;;
  esac

  echo "============================================================"
  echo "KEEPSAKE ablation: $variant"
  echo "Round-robin target: $target_count / $TOTAL_VIDEOS"
  echo "Run: $run_name"
  echo "Signals: geometry=$geometry_weight visual=$visual_weight"
  echo "Priority update: $priority_update"
  echo "============================================================"

  GPU="$GPU" \
  WORLDMEM_REPO_ROOT="$WORLDMEM_REPO_ROOT" \
  WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
  WORLDMEM_DATA_DIR="$DATA_DIR" \
  MEMORY_POLICY=slam_covisibility \
  MEMORY_BUDGET="$MEMORY_BUDGET" \
  MEMORY_FEATURE_BACKEND=latent \
  SLAM_GEOMETRY_WEIGHT="$geometry_weight" \
  SLAM_VISUAL_WEIGHT="$visual_weight" \
  MEMORY_PRIORITY_UPDATE="$priority_update" \
  FUTURE_SECONDS="$FUTURE_SECONDS" \
  CONTEXT_FRAMES="$CONTEXT_FRAMES" \
  FPS="$FPS" \
  NUM_VIDEOS="$target_count" \
  DATASET_SEED="$DATASET_SEED" \
  GLOBAL_SEED="$GLOBAL_SEED" \
  GENERATION_SEED="$GENERATION_SEED" \
  MEMORY_POLICY_SEED="$MEMORY_POLICY_SEED" \
  RUN_NAME="$run_name" \
  bash "$SCRIPT_DIR/run_worldmem_memory_policy_smoke.sh"
}

for target_count in $(seq "$START_VIDEO" "$END_VIDEO"); do
  echo "================ ABLATION SWEEP: $target_count / $TOTAL_VIDEOS ================"
  for variant in "${ABLATIONS_ARRAY[@]}"; do
    variant="${variant//[[:space:]]/}"
    [ -n "$variant" ] || continue
    if ! run_variant "$variant" "$target_count" && [ "$CONTINUE_ON_ERROR" != "1" ]; then
      exit 1
    fi
  done
done

echo "Completed requested KEEPSAKE ablation sweeps."
