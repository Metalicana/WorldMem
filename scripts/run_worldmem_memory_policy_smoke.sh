#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

WORLDMEM_ROOT="${WORLDMEM_ROOT:-$(pwd)}"
if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

MEMORY_POLICY="${MEMORY_POLICY:-unbounded}"
MEMORY_BUDGET="${MEMORY_BUDGET:-}"
LIMIT_BATCH="${LIMIT_BATCH:-1}"
N_FRAMES_VALID="${N_FRAMES_VALID:-700}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-600}"
SAMPLING_TIMESTEPS="${SAMPLING_TIMESTEPS:-20}"
DATA_DIR="${WORLDMEM_DATA_DIR:-data/minecraft}"
RUN_NAME="${RUN_NAME:-worldmem_${MEMORY_POLICY}${MEMORY_BUDGET:+_b${MEMORY_BUDGET}}_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_ROOT/outputs/memory_policy/$RUN_NAME}"
TRACE_PATH="${TRACE_PATH:-$OUTPUT_DIR/access_traces/$RUN_NAME.jsonl}"

if { [ "$MEMORY_POLICY" = "fifo" ] || [ "$MEMORY_POLICY" = "rarity_irreplaceability" ] || [ "$MEMORY_POLICY" = "slam_covisibility" ]; } && [ -z "$MEMORY_BUDGET" ]; then
  echo "MEMORY_BUDGET is required when MEMORY_POLICY=$MEMORY_POLICY" >&2
  exit 2
fi

cd "$WORLDMEM_ROOT"
mkdir -p "$OUTPUT_DIR" "$(dirname "$TRACE_PATH")" "$STORAGE_ROOT/tmp"
export TMPDIR="${TMPDIR:-$STORAGE_ROOT/tmp}"
export WANDB_MODE="${WANDB_MODE:-offline}"

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
  +dataset.n_frames_valid="$N_FRAMES_VALID"
  algorithm.diffusion.sampling_timesteps="$SAMPLING_TIMESTEPS"
  +algorithm.memory_condition_length=8
  +algorithm.lpips_batch_size=16
  +algorithm.log_video=true
  +algorithm.save_local=true
  +dataset.customized_validation=true
  +algorithm.n_tokens=8
  algorithm.context_frames="$CONTEXT_FRAMES"
  experiment.test.batch_size=1
  experiment.test.limit_batch="$LIMIT_BATCH"
  wandb.mode=offline
  wandb.entity=local
  +algorithm.memory_policy="$MEMORY_POLICY"
  +algorithm.access_trace_path="$TRACE_PATH"
  +output_dir="$OUTPUT_DIR"
)

if [ -n "$MEMORY_BUDGET" ]; then
  cmd+=(+algorithm.memory_budget="$MEMORY_BUDGET")
fi

echo "WorldMem root: $WORLDMEM_ROOT"
echo "Storage root: $STORAGE_ROOT"
echo "Data dir: $DATA_DIR"
echo "Memory policy: $MEMORY_POLICY"
echo "Memory budget: ${MEMORY_BUDGET:-none}"
echo "Output dir: $OUTPUT_DIR"
echo "Trace path: $TRACE_PATH"

"${cmd[@]}"
