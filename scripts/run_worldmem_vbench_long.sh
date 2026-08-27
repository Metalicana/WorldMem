#!/usr/bin/env bash
# VBench-Long evaluation over WorldMem memory-policy sweep runs. Ported from
# MemCam's slurm/newton_vbench_long_context_memory_60s.sbatch, same
# adaptation as run_worldmem_vbench.sh. Each run is staged into a clean input
# directory containing exactly the requested parsed batch IDs.
#
# IMPORTANT (carried over from MemCam's own script, which traced this
# directly out of vbench2_beta_long/__init__.py before trusting it): calling
# evaluate() auto-splits each video into 2s clips under
# <videos_path>/split_clip/ as part of preprocessing for long_custom_input.
# The staging directory absorbs that derived output, leaving generated videos
# untouched. Pass FORCE=1 to clear stale staged split clips and recompute.
#
# Same environment requirement as run_worldmem_vbench.sh -- activate whatever
# VBench-capable conda env this machine already has before running.
#
# Usage:
#   conda activate vbench
#   RUNS="worldmem_mce_b32_60s_n15" bash scripts/run_worldmem_vbench_long.sh   # smoke test one cell first
#   RUNS="<full list>" bash scripts/run_worldmem_vbench_long.sh               # then the rest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORLDMEM_REPO_ROOT="${WORLDMEM_REPO_ROOT:-$DEFAULT_REPO_ROOT}"

if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

VBENCH_ROOT="${VBENCH_ROOT:-$HOME/VBench}"
RESULTS_ROOT="${RESULTS_ROOT:-$STORAGE_ROOT/outputs/memory_policy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_ROOT/metrics/vbench_long_results}"
DIMENSIONS="${DIMENSIONS:-subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality}"
FORCE="${FORCE:-0}"
LIMIT="${LIMIT:-15}"
STAGING_ROOT="${STAGING_ROOT:-$OUTPUT_ROOT/_input_staging}"

# Defaults to a single smoke-test cell, matching MemCam's own script's
# posture ("sanity check the auto-split + dimension scoring actually works
# before committing the full grid") -- deliberately not the full 21-run list
# by default here, unlike run_worldmem_vbench.sh's default.
RUNS="${RUNS:-worldmem_mce_b32_60s_n15}"

mkdir -p "$OUTPUT_ROOT" "$STAGING_ROOT"
cd "$VBENCH_ROOT"

echo "WorldMem VBench-Long batch eval"
echo "VBench root: $VBENCH_ROOT"
echo "Results root: $RESULTS_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Dimensions: $DIMENSIONS"
echo "Matched batch limit: $LIMIT"
echo "Runs: $RUNS"
echo "Started: $(date)"

for run in $RUNS; do
  out_dir="$OUTPUT_ROOT/$run"
  video_dir="$RESULTS_ROOT/$run/videos/test_vis/pred"
  if [ ! -d "$video_dir" ]; then
    echo "[error] $run -- no directory at $video_dir" >&2
    exit 1
  fi

  stage_dir="$STAGING_ROOT/$run"
  stage_metadata="$stage_dir/input_selection.json"
  stage_cmd=(
    python "$WORLDMEM_REPO_ROOT/utils/stage_worldmem_vbench_input.py"
    --source-dir "$video_dir"
    --stage-dir "$stage_dir"
    --metadata-path "$stage_metadata"
    --repo-root "$WORLDMEM_REPO_ROOT"
    --run-name "$run"
    --limit "$LIMIT"
    --mode long_custom_input
    --dimensions "$DIMENSIONS"
  )
  if [ "$FORCE" = "1" ]; then
    stage_cmd+=(--reset-derived)
  fi
  "${stage_cmd[@]}"

  if compgen -G "$out_dir"/*_eval_results.json > /dev/null 2>&1 && [ "$FORCE" != "1" ]; then
    if [ ! -f "$out_dir/input_selection.json" ] || ! cmp -s "$stage_metadata" "$out_dir/input_selection.json"; then
      echo "[error] $run has stale or unmatched VBench-Long output. Inspect it or rerun explicitly with FORCE=1." >&2
      exit 1
    fi
    echo "[skip] $run already has matched first-$LIMIT eval results in $out_dir"
    continue
  fi

  echo "[run] $run  ($LIMIT matched videos)  $(date)"
  python vbench2_beta_long/eval_long.py \
    --videos_path "$stage_dir" \
    --dimension $DIMENSIONS \
    --mode long_custom_input \
    --dev_flag \
    --output_path "$out_dir"
  mkdir -p "$out_dir"
  cp "$stage_metadata" "$out_dir/input_selection.json"
  echo "[done] $run  $(date)"
done

echo "Finished: $(date)"
echo "Results under: $OUTPUT_ROOT"
echo "View with: python utils/aggregate_vbench_results.py --root $OUTPUT_ROOT"
