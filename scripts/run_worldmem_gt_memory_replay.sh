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
QUALITY_ROOT="${QUALITY_ROOT:-$STORAGE_ROOT/outputs/memory_quality_60s}"
MANIFEST="${MANIFEST:-$QUALITY_ROOT/metrics/retrieved_memory_quality/gt_replay_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/gt_memory_replay_60s}"
DATASET_SEED="${DATASET_SEED:-42}"
COMPUTE_DINO="${COMPUTE_DINO:-false}"

if [ ! -f "$MANIFEST" ]; then
  echo "Replay manifest not found: $MANIFEST" >&2
  echo "Run scripts/analyze_worldmem_retrieved_memory_quality.sh first." >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT" "$STORAGE_ROOT/logs"

while IFS=$'\t' read -r rank batch_idx target_frame generation_seed expected_indices; do
  [ -n "$batch_idx" ] || continue
  base_seed=$((generation_seed - batch_idx))
  run_name="worldmem_gt_replay_unbounded_batch${batch_idx}_target${target_frame}"
  echo "============================================================"
  echo "GT memory replay rank=$rank batch=$batch_idx target=$target_frame"
  echo "Expected selected frames: $expected_indices"
  echo "GPU: $GPU  Generation base seed: $base_seed"
  echo "============================================================"

  GPU="$GPU" \
  WORLDMEM_REPO_ROOT="$REPO_ROOT" \
  WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
  MEMORY_POLICY=unbounded \
  MEMORY_REFERENCE_SOURCE=predicted \
  GLOBAL_SEED="$base_seed" \
  GENERATION_SEED="$base_seed" \
  MEMORY_POLICY_SEED="$base_seed" \
  DATASET_SEED="$DATASET_SEED" \
  DATASET_START_INDEX="$batch_idx" \
  OUTPUT_BATCH_OFFSET="$batch_idx" \
  FUTURE_SECONDS=60 \
  NUM_VIDEOS=1 \
  LIMIT_BATCH=1 \
  RUN_NAME="$run_name" \
  OUTPUT_DIR="$OUTPUT_ROOT/$run_name" \
  TRACE_PATH="$OUTPUT_ROOT/$run_name/access_traces/$run_name.jsonl" \
  TRACE_RETRIEVED_MEMORY_QUALITY=false \
  GT_MEMORY_REPLAY_TARGET_FRAME="$target_frame" \
  GT_MEMORY_REPLAY_EXPECTED_INDICES="$expected_indices" \
  GT_MEMORY_REPLAY_COMPUTE_DINO="$COMPUTE_DINO" \
  SAVE_GT_VIDEO=false \
  COMPUTE_EVAL_METRICS=false \
  PROFILE_TIMING=true \
  WANDB_MODE=disabled \
  bash "$SCRIPT_DIR/run_worldmem_memory_policy_smoke.sh"
done < <(
  python - "$MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    rows = json.load(handle)
for row in rows:
    indices = ",".join(str(value) for value in row["selected_memory_frames"])
    print(
        row["rank"],
        row["global_batch_idx"],
        row["target_frame"],
        row["generation_seed"],
        indices,
        sep="\t",
    )
PY
)

python "$REPO_ROOT/utils/summarize_worldmem_gt_memory_replay.py" \
  --output_root "$OUTPUT_ROOT" \
  --output_dir "$OUTPUT_ROOT/metrics"

