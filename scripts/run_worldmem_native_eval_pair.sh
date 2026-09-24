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

PAPER_PROTOCOL="${PAPER_PROTOCOL:-0}"
if [ "$PAPER_PROTOCOL" = "1" ]; then
  NUM_VIDEOS="${NUM_VIDEOS:-300}"
  RFID_MAX_VIDEOS="${RFID_MAX_VIDEOS:-50}"
else
  NUM_VIDEOS="${NUM_VIDEOS:-10}"
  RFID_MAX_VIDEOS="${RFID_MAX_VIDEOS:-}"
fi
KEEPSAKE_BUDGET="${KEEPSAKE_BUDGET:-32}"
DATASET_SEED="${DATASET_SEED:-42}"
GLOBAL_SEED="${GLOBAL_SEED:-42}"
RUN_UNBOUNDED="${RUN_UNBOUNDED:-1}"
RUN_KEEPSAKE="${RUN_KEEPSAKE:-1}"
RUN_RFID="${RUN_RFID:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/memory_policy}"
METRICS_DIR="${METRICS_DIR:-$OUTPUT_ROOT/metrics/native_worldmem_10s_n${NUM_VIDEOS}}"

UNBOUNDED_RUN="${UNBOUNDED_RUN:-worldmem_native_unbounded_10s_n${NUM_VIDEOS}_seed${GLOBAL_SEED}}"
KEEPSAKE_RUN="${KEEPSAKE_RUN:-worldmem_native_keepsake_b${KEEPSAKE_BUDGET}_10s_n${NUM_VIDEOS}_seed${GLOBAL_SEED}}"

case "$NUM_VIDEOS" in
  ''|*[!0-9]*) echo "NUM_VIDEOS must be a positive integer" >&2; exit 2 ;;
esac
case "$PAPER_PROTOCOL" in
  0|1) ;;
  *) echo "PAPER_PROTOCOL must be 0 or 1" >&2; exit 2 ;;
esac
case "$KEEPSAKE_BUDGET" in
  ''|*[!0-9]*) echo "KEEPSAKE_BUDGET must be a positive integer" >&2; exit 2 ;;
esac
if [ -z "$RFID_MAX_VIDEOS" ]; then
  if [ "$NUM_VIDEOS" -lt 50 ]; then
    RFID_MAX_VIDEOS="$NUM_VIDEOS"
  else
    RFID_MAX_VIDEOS=50
  fi
fi
case "$RFID_MAX_VIDEOS" in
  ''|*[!0-9]*) echo "RFID_MAX_VIDEOS must be a positive integer" >&2; exit 2 ;;
esac
if [ "$NUM_VIDEOS" -lt 1 ] || [ "$KEEPSAKE_BUDGET" -lt 1 ]; then
  echo "NUM_VIDEOS and KEEPSAKE_BUDGET must be positive" >&2
  exit 2
fi
if [ "$RFID_MAX_VIDEOS" -gt "$NUM_VIDEOS" ]; then
  echo "RFID_MAX_VIDEOS cannot exceed NUM_VIDEOS" >&2
  exit 2
fi
if [ "$PAPER_PROTOCOL" = "1" ] && { [ "$NUM_VIDEOS" -ne 300 ] || [ "$RFID_MAX_VIDEOS" -ne 50 ]; }; then
  echo "PAPER_PROTOCOL=1 requires NUM_VIDEOS=300 and RFID_MAX_VIDEOS=50" >&2
  exit 2
fi

mkdir -p "$METRICS_DIR"

echo "WorldMem native beyond-context evaluation"
echo "Protocol: 600-frame initial bank + 100 generated frames"
echo "Generator context: 8 frames"
echo "Retrieved memories: 8"
echo "Sampling steps: 20"
echo "Dataset/global seed: $DATASET_SEED/$GLOBAL_SEED"
echo "Videos: $NUM_VIDEOS"
echo "Paper protocol: $PAPER_PROTOCOL"
echo "rFID cohort: $RFID_MAX_VIDEOS videos / $((RFID_MAX_VIDEOS * 100)) frames"
echo "GPU: $GPU"
echo "Unbounded run: $UNBOUNDED_RUN"
echo "KEEPSAKE run: $KEEPSAKE_RUN (slam_covisibility, B=$KEEPSAKE_BUDGET)"
echo "Output root: $OUTPUT_ROOT"

run_policy() {
  local policy="$1"
  local budget="$2"
  local run_name="$3"

  echo
  echo "============================================================"
  echo "Native evaluation: $run_name"
  echo "Policy: $policy"
  echo "Budget: ${budget:-unbounded}"
  echo "============================================================"

  GPU="$GPU" \
  WORLDMEM_REPO_ROOT="$REPO_ROOT" \
  WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
  OUTPUT_DIR="$OUTPUT_ROOT/$run_name" \
  RUN_NAME="$run_name" \
  MEMORY_POLICY="$policy" \
  MEMORY_BUDGET="$budget" \
  MEMORY_BANK_DEVICE=cpu \
  MEMORY_REFERENCE_SOURCE=predicted \
  MEMORY_FEATURE_BACKEND=latent \
  FUTURE_SECONDS=10 \
  FPS=10 \
  CONTEXT_FRAMES=600 \
  N_FRAMES_VALID=700 \
  SAMPLING_TIMESTEPS=20 \
  LIMIT_BATCH="$NUM_VIDEOS" \
  DATASET_SEED="$DATASET_SEED" \
  GLOBAL_SEED="$GLOBAL_SEED" \
  GENERATION_SEED="$GLOBAL_SEED" \
  MEMORY_POLICY_SEED="$GLOBAL_SEED" \
  LOG_VIDEO=true \
  SAVE_LOCAL_PER_BATCH=true \
  SAVE_GT_VIDEO=true \
  COMPUTE_EVAL_METRICS=true \
  STREAM_EVAL_METRICS=true \
  RESUME_REQUIRE_METRICS=true \
  TEST_NUM_WORKERS=0 \
  WANDB_MODE=disabled \
  DRY_RUN="${DRY_RUN:-0}" \
  bash "$REPO_ROOT/scripts/run_worldmem_memory_policy_smoke.sh"
}

if [ "$RUN_UNBOUNDED" = "1" ]; then
  run_policy unbounded "" "$UNBOUNDED_RUN"
fi

if [ "$RUN_KEEPSAKE" = "1" ]; then
  run_policy slam_covisibility "$KEEPSAKE_BUDGET" "$KEEPSAKE_RUN"
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  exit 0
fi

if [ "$RUN_RFID" = "1" ]; then
  GT_DIR="$OUTPUT_ROOT/$UNBOUNDED_RUN/videos/test_vis/gt"
  if [ ! -d "$GT_DIR" ]; then
    echo "Native rFID requires the matched reconstructed-GT directory: $GT_DIR" >&2
    echo "Run the unbounded branch first or set RUN_RFID=0." >&2
    exit 2
  fi

  for run_name in "$UNBOUNDED_RUN" "$KEEPSAKE_RUN"; do
    if { [ "$run_name" = "$UNBOUNDED_RUN" ] && [ "$RUN_UNBOUNDED" != "1" ]; } || \
       { [ "$run_name" = "$KEEPSAKE_RUN" ] && [ "$RUN_KEEPSAKE" != "1" ]; }; then
      if [ ! -d "$OUTPUT_ROOT/$run_name/videos/test_vis/pred" ]; then
        continue
      fi
    fi

    echo
    echo "Native reconstruction FID: $run_name"
    python "$REPO_ROOT/calculate_fid.py" \
      --videos_dir "$OUTPUT_ROOT/$run_name/videos/test_vis" \
      --pred_dir "$OUTPUT_ROOT/$run_name/videos/test_vis/pred" \
      --gt_dir "$GT_DIR" \
      --batch_size "${RFID_BATCH_SIZE:-32}" \
      --device cuda \
      --max_frames_per_video 100 \
      --max_videos "$RFID_MAX_VIDEOS"
  done
fi

python "$REPO_ROOT/utils/summarize_worldmem_native_eval.py" \
  --output-root "$OUTPUT_ROOT" \
  --runs "$UNBOUNDED_RUN,$KEEPSAKE_RUN" \
  --labels "WorldMem,WorldMem + KEEPSAKE" \
  --limit "$NUM_VIDEOS" \
  --rfid-videos "$RFID_MAX_VIDEOS" \
  --output-dir "$METRICS_DIR"
