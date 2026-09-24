#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${WORLDMEM_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-${REPO_ROOT}}"
SUITE_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/memory_policy}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/minecraft}"

UNBOUNDED_RUN="${UNBOUNDED_RUN:-worldmem_unbounded_60s_n30}"
FIFO_RUN="${FIFO_RUN:-worldmem_fifo_b32_60s_n30}"
KEEPSAKE_RUN="${KEEPSAKE_RUN:-worldmem_slam_covisibility_b32_60s_n30}"
LIMIT="${LIMIT:-15}"
DATASET_SEED="${DATASET_SEED:-42}"
WO_UPDOWN="${WO_UPDOWN:-0}"
OUTPUT_DIR="${REVISIT_OUTPUT_DIR:-${SUITE_ROOT}/metrics/revisit_qualitative_60s_n15}"

TRAJECTORY_FPS="${TRAJECTORY_FPS:-10}"
EXPECTED_OUTPUT_FRAMES="${EXPECTED_OUTPUT_FRAMES:-600}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-600}"
SOURCE_START_OFFSET="${SOURCE_START_OFFSET:-100}"
MIN_SEPARATION_SEC="${MIN_SEPARATION_SEC:-5}"
ENDPOINT_POSITION_THRESHOLD="${ENDPOINT_POSITION_THRESHOLD:-0.75}"
ENDPOINT_ROTATION_THRESHOLD_DEG="${ENDPOINT_ROTATION_THRESHOLD_DEG:-15}"
DEPARTURE_POSITION_THRESHOLD="${DEPARTURE_POSITION_THRESHOLD:-2.0}"
DEPARTURE_ROTATION_THRESHOLD_DEG="${DEPARTURE_ROTATION_THRESHOLD_DEG:-45}"
MIN_AWAY_SEC="${MIN_AWAY_SEC:-1}"
MAX_CANDIDATES_PER_TRAJECTORY="${MAX_CANDIDATES_PER_TRAJECTORY:-12}"
REVISIT_SUPPRESSION_SEC="${REVISIT_SUPPRESSION_SEC:-1}"
EXPORT_PREVIEWS="${EXPORT_PREVIEWS:-1}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"

mkdir -p "$OUTPUT_DIR"

echo "WorldMem CPU-only revisit-candidate export"
echo "Unbounded: ${SUITE_ROOT}/${UNBOUNDED_RUN}"
echo "FIFO B32: ${SUITE_ROOT}/${FIFO_RUN}"
echo "KEEPSAKE B32: ${SUITE_ROOT}/${KEEPSAKE_RUN}"
echo "Data dir: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Trajectory clock: ${TRAJECTORY_FPS} fps"
echo "Expected saved-video frames: ${EXPECTED_OUTPUT_FRAMES}"
echo "Endpoint threshold: ${ENDPOINT_POSITION_THRESHOLD} blocks, ${ENDPOINT_ROTATION_THRESHOLD_DEG} degrees"
echo "Departure threshold: ${DEPARTURE_POSITION_THRESHOLD} blocks OR ${DEPARTURE_ROTATION_THRESHOLD_DEG} degrees"
echo "Minimum gap/away: ${MIN_SEPARATION_SEC}s / ${MIN_AWAY_SEC}s"

args=(
  --output-root "$OUTPUT_DIR"
  --data-dir "$DATA_DIR"
  --unbounded-run-dir "${SUITE_ROOT}/${UNBOUNDED_RUN}"
  --fifo-run-dir "${SUITE_ROOT}/${FIFO_RUN}"
  --keepsake-run-dir "${SUITE_ROOT}/${KEEPSAKE_RUN}"
  --limit "$LIMIT"
  --dataset-seed "$DATASET_SEED"
  --trajectory-fps "$TRAJECTORY_FPS"
  --expected-output-frames "$EXPECTED_OUTPUT_FRAMES"
  --context-frames "$CONTEXT_FRAMES"
  --source-start-offset "$SOURCE_START_OFFSET"
  --min-separation-sec "$MIN_SEPARATION_SEC"
  --endpoint-position-threshold "$ENDPOINT_POSITION_THRESHOLD"
  --endpoint-rotation-threshold-deg "$ENDPOINT_ROTATION_THRESHOLD_DEG"
  --departure-position-threshold "$DEPARTURE_POSITION_THRESHOLD"
  --departure-rotation-threshold-deg "$DEPARTURE_ROTATION_THRESHOLD_DEG"
  --min-away-sec "$MIN_AWAY_SEC"
  --max-candidates-per-trajectory "$MAX_CANDIDATES_PER_TRAJECTORY"
  --revisit-suppression-sec "$REVISIT_SUPPRESSION_SEC"
)

if [[ "$EXPORT_PREVIEWS" == "0" ]]; then
  args+=(--no-previews)
fi
if [[ "$ALLOW_INCOMPLETE" == "1" ]]; then
  args+=(--allow-incomplete)
fi
if [[ "$WO_UPDOWN" == "1" ]]; then
  args+=(--wo-updown)
fi

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="" python utils/export_worldmem_revisit_candidates.py "${args[@]}"
