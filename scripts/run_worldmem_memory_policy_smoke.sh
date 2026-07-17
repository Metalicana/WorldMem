#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORLDMEM_REPO_ROOT="${WORLDMEM_REPO_ROOT:-$DEFAULT_REPO_ROOT}"
if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

MEMORY_POLICY="${MEMORY_POLICY:-unbounded}"
MEMORY_BUDGET="${MEMORY_BUDGET:-}"
LIMIT_BATCH="${LIMIT_BATCH:-${NUM_VIDEOS:-1}}"
REQUESTED_LIMIT_BATCH="$LIMIT_BATCH"
FPS="${FPS:-10}"
FUTURE_SECONDS="${FUTURE_SECONDS:-}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-600}"
if [ -n "$FUTURE_SECONDS" ]; then
  N_FRAMES_VALID="${N_FRAMES_VALID:-$((CONTEXT_FRAMES + FUTURE_SECONDS * FPS))}"
else
  N_FRAMES_VALID="${N_FRAMES_VALID:-700}"
fi
SAMPLING_TIMESTEPS="${SAMPLING_TIMESTEPS:-20}"
DECODE_CHUNK_SIZE="${DECODE_CHUNK_SIZE:-64}"
DATA_DIR="${WORLDMEM_DATA_DIR:-data/minecraft}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-${FUTURE_SECONDS:+_${FUTURE_SECONDS}s}_n${LIMIT_BATCH}}"
RUN_NAME="${RUN_NAME:-worldmem_${MEMORY_POLICY}${MEMORY_BUDGET:+_b${MEMORY_BUDGET}}${RUN_NAME_SUFFIX}}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_ROOT/outputs/memory_policy/$RUN_NAME}"
TRACE_PATH="${TRACE_PATH:-$OUTPUT_DIR/access_traces/$RUN_NAME.jsonl}"

if { [ "$MEMORY_POLICY" = "fifo" ] || [ "$MEMORY_POLICY" = "rarity_irreplaceability" ] || [ "$MEMORY_POLICY" = "slam_covisibility" ]; } && [ -z "$MEMORY_BUDGET" ]; then
  echo "MEMORY_BUDGET is required when MEMORY_POLICY=$MEMORY_POLICY" >&2
  exit 2
fi

cd "$WORLDMEM_REPO_ROOT"
if [ ! -f main.py ]; then
  echo "Could not find main.py in WORLDMEM_REPO_ROOT=$WORLDMEM_REPO_ROOT" >&2
  echo "Set WORLDMEM_REPO_ROOT to the WorldMem repository path, not the storage path." >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR" "$(dirname "$TRACE_PATH")" "$STORAGE_ROOT/tmp"
export TMPDIR="${TMPDIR:-$STORAGE_ROOT/tmp}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-0}"
SAVE_LOCAL_PER_BATCH="${SAVE_LOCAL_PER_BATCH:-true}"
LOG_VIDEO="${LOG_VIDEO:-true}"
SAVE_GT_VIDEO="${SAVE_GT_VIDEO:-false}"
COMPUTE_EVAL_METRICS="${COMPUTE_EVAL_METRICS:-false}"
STREAM_EVAL_METRICS="${STREAM_EVAL_METRICS:-false}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-0}"
RESUME_PARTIAL="${RESUME_PARTIAL:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESUME_REQUIRE_METRICS="${RESUME_REQUIRE_METRICS:-false}"
PROFILE_CUDA_MEMORY="${PROFILE_CUDA_MEMORY:-false}"
PROFILE_TIMING="${PROFILE_TIMING:-$PROFILE_CUDA_MEMORY}"

count_completed_batches() {
  local pred_dir="$1"
  if [ ! -d "$pred_dir" ]; then
    echo 0
    return
  fi

  find "$pred_dir" -maxdepth 1 -type f -name 'video_batch*.mp4' -size +4k -exec basename {} \; \
    | sed -nE 's/^video_batch([0-9]+)_.*_rank[0-9]+\.mp4$/\1/p' \
    | sort -n \
    | uniq \
    | awk '
        BEGIN { expected = 0 }
        {
          idx = $1 + 0
          if (idx == expected) {
            expected += 1
          } else if (idx > expected) {
            exit
          }
        }
        END { print expected }
      '
}

count_completed_metric_batches() {
  local trace_path="$1"
  if [ ! -f "$trace_path" ]; then
    echo 0
    return
  fi

  sed -nE '/"event": "batch_metrics"/ s/.*"global_batch_idx": ([0-9]+).*/\1/p' "$trace_path" \
    | sort -n \
    | uniq \
    | awk '
        BEGIN { expected = 0 }
        {
          idx = $1 + 0
          if (idx == expected) {
            expected += 1
          } else if (idx > expected) {
            exit
          }
        }
        END { print expected }
      '
}

COMPLETED_VIDEO_BATCHES="$(count_completed_batches "$OUTPUT_DIR/videos/test_vis/pred")"
COMPLETED_BATCHES="$COMPLETED_VIDEO_BATCHES"
COMPLETED_METRIC_BATCHES="not-required"
if [ "$RESUME_REQUIRE_METRICS" = "true" ] || [ "$RESUME_REQUIRE_METRICS" = "1" ]; then
  COMPLETED_METRIC_BATCHES="$(count_completed_metric_batches "$TRACE_PATH")"
  if [ "$COMPLETED_METRIC_BATCHES" -lt "$COMPLETED_BATCHES" ]; then
    COMPLETED_BATCHES="$COMPLETED_METRIC_BATCHES"
  fi
fi
DATASET_START_INDEX="${DATASET_START_INDEX:-0}"
OUTPUT_BATCH_OFFSET="${OUTPUT_BATCH_OFFSET:-0}"
RESET_TRACE=0

if { [ "$SKIP_COMPLETED" = "1" ] || [ "$RESUME_PARTIAL" = "1" ]; } && [ "$COMPLETED_BATCHES" -ge "$REQUESTED_LIMIT_BATCH" ]; then
  echo "Run already has $COMPLETED_BATCHES completed batch videos; requested $REQUESTED_LIMIT_BATCH. Skipping $RUN_NAME."
  exit 0
fi

if [ "$RESUME_PARTIAL" = "1" ] && [ "$COMPLETED_BATCHES" -gt 0 ]; then
  DATASET_START_INDEX="$COMPLETED_BATCHES"
  OUTPUT_BATCH_OFFSET="$COMPLETED_BATCHES"
  LIMIT_BATCH=$((REQUESTED_LIMIT_BATCH - COMPLETED_BATCHES))
  echo "Resuming $RUN_NAME from completed batch count $COMPLETED_BATCHES; remaining batches: $LIMIT_BATCH."
else
  RESET_TRACE=1
fi

cmd=(
  python -m main +name="$RUN_NAME"
  experiment.tasks=[test]
  dataset.validation_multiplier=1
  +dataset.seed=42
  +diffusion_model_path=zeqixiao/worldmem_checkpoints/diffusion_only.ckpt
  +vae_path=zeqixiao/worldmem_checkpoints/vae_only.ckpt
  +customized_load=true
  +seperate_load=true
  dataset.n_frames=8
  dataset.save_dir="$DATA_DIR"
  +dataset.eval_start_index="$DATASET_START_INDEX"
  +dataset.n_frames_valid="$N_FRAMES_VALID"
  algorithm.diffusion.sampling_timesteps="$SAMPLING_TIMESTEPS"
  +algorithm.decode_chunk_size="$DECODE_CHUNK_SIZE"
  +algorithm.memory_condition_length=8
  +algorithm.lpips_batch_size=16
  +algorithm.log_video="$LOG_VIDEO"
  +algorithm.save_local=true
  +algorithm.save_local_per_batch="$SAVE_LOCAL_PER_BATCH"
  +algorithm.save_gt_video="$SAVE_GT_VIDEO"
  +algorithm.compute_eval_metrics="$COMPUTE_EVAL_METRICS"
  +algorithm.stream_eval_metrics="$STREAM_EVAL_METRICS"
  +algorithm.profile_cuda_memory="$PROFILE_CUDA_MEMORY"
  +algorithm.profile_timing="$PROFILE_TIMING"
  +algorithm.output_batch_offset="$OUTPUT_BATCH_OFFSET"
  +dataset.customized_validation=true
  +algorithm.n_tokens=8
  algorithm.context_frames="$CONTEXT_FRAMES"
  experiment.test.batch_size=1
  experiment.test.limit_batch="$LIMIT_BATCH"
  experiment.test.data.num_workers="$TEST_NUM_WORKERS"
  wandb.mode="$WANDB_MODE"
  wandb.entity=local
  +algorithm.memory_policy="$MEMORY_POLICY"
  +algorithm.access_trace_path="$TRACE_PATH"
  +output_dir="$OUTPUT_DIR"
)

if [ -n "$MEMORY_BUDGET" ]; then
  cmd+=(+algorithm.memory_budget="$MEMORY_BUDGET")
fi

echo "WorldMem repo root: $WORLDMEM_REPO_ROOT"
echo "Storage root: $STORAGE_ROOT"
echo "Data dir: $DATA_DIR"
echo "Memory policy: $MEMORY_POLICY"
echo "Memory budget: ${MEMORY_BUDGET:-none}"
echo "Future seconds: ${FUTURE_SECONDS:-derived-from-N_FRAMES_VALID}"
echo "Context frames: $CONTEXT_FRAMES"
echo "N frames valid: $N_FRAMES_VALID"
echo "Requested batch/videos: $REQUESTED_LIMIT_BATCH"
echo "Completed batch videos: $COMPLETED_VIDEO_BATCHES"
if [ "$RESUME_REQUIRE_METRICS" = "true" ] || [ "$RESUME_REQUIRE_METRICS" = "1" ]; then
  echo "Completed batch metrics: $COMPLETED_METRIC_BATCHES"
fi
echo "Resume-complete batch count: $COMPLETED_BATCHES"
echo "Remaining limit batch/videos: $LIMIT_BATCH"
echo "Dataset start index: $DATASET_START_INDEX"
echo "Output batch offset: $OUTPUT_BATCH_OFFSET"
echo "Log video: $LOG_VIDEO"
echo "Save local per batch: $SAVE_LOCAL_PER_BATCH"
echo "Save GT video: $SAVE_GT_VIDEO"
echo "Compute eval metrics: $COMPUTE_EVAL_METRICS"
echo "Stream eval metrics: $STREAM_EVAL_METRICS"
echo "Profile CUDA memory: $PROFILE_CUDA_MEMORY"
echo "Profile timing: $PROFILE_TIMING"
echo "Test num workers: $TEST_NUM_WORKERS"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "Decode chunk size: $DECODE_CHUNK_SIZE"
echo "Resume requires metrics: $RESUME_REQUIRE_METRICS"
echo "Output dir: $OUTPUT_DIR"
echo "Trace path: $TRACE_PATH"

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if [ "$RESET_TRACE" = "1" ]; then
  : > "$TRACE_PATH"
fi

"${cmd[@]}"
