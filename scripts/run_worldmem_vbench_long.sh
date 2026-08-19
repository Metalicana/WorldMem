#!/usr/bin/env bash
# VBench-Long evaluation over WorldMem memory-policy sweep runs. Ported from
# MemCam's slurm/newton_vbench_long_context_memory_60s.sbatch, same
# adaptation as run_worldmem_vbench.sh: point straight at
# <run_dir>/videos/test_vis/pred/ instead of a flat MemCam-style directory.
#
# IMPORTANT (carried over from MemCam's own script, which traced this
# directly out of vbench2_beta_long/__init__.py before trusting it): calling
# evaluate() auto-splits each video into 2s clips under
# <videos_path>/split_clip/ as part of preprocessing for long_custom_input
# mode -- no manual scene-splitting needed, but it DOES write a split_clip/
# subfolder of re-encoded clips inside the pred/ directory as a side effect.
# That's extra disk, not just eval output. It's a subdirectory, not new
# top-level video_batch*.mp4 files, so it will not confuse
# count_completed_batches()'s maxdepth-1 glob in run_worldmem_memory_policy_smoke.sh
# or scripts/audit_worldmem_memory_policy_runs.sh -- but it does mean re-running
# this against the same run again will find (and by default skip past) a
# stale split_clip/ from a prior attempt; pass FORCE=1 to redo from scratch.
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

# Defaults to a single smoke-test cell, matching MemCam's own script's
# posture ("sanity check the auto-split + dimension scoring actually works
# before committing the full grid") -- deliberately not the full 21-run list
# by default here, unlike run_worldmem_vbench.sh's default.
RUNS="${RUNS:-worldmem_mce_b32_60s_n15}"

mkdir -p "$OUTPUT_ROOT"
cd "$VBENCH_ROOT"

echo "WorldMem VBench-Long batch eval"
echo "VBench root: $VBENCH_ROOT"
echo "Results root: $RESULTS_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Dimensions: $DIMENSIONS"
echo "Runs: $RUNS"
echo "Started: $(date)"

for run in $RUNS; do
  out_dir="$OUTPUT_ROOT/$run"
  if [ "$FORCE" != "1" ] && compgen -G "$out_dir"/*_eval_results.json > /dev/null 2>&1; then
    echo "[skip] $run already has eval results in $out_dir"
    continue
  fi

  video_dir="$RESULTS_ROOT/$run/videos/test_vis/pred"
  if [ ! -d "$video_dir" ]; then
    echo "[skip] $run -- no directory at $video_dir"
    continue
  fi

  video_count="$(find "$video_dir" -maxdepth 1 -name 'video_batch*.mp4' | wc -l)"
  echo "[run] $run  ($video_count videos)  $(date)"
  python vbench2_beta_long/eval_long.py \
    --videos_path "$video_dir" \
    --dimension $DIMENSIONS \
    --mode long_custom_input \
    --dev_flag \
    --output_path "$out_dir"
  echo "[done] $run  $(date)"
done

echo "Finished: $(date)"
echo "Results under: $OUTPUT_ROOT"
echo "View with: python utils/aggregate_vbench_results.py --root $OUTPUT_ROOT"
