#!/usr/bin/env bash
# VBench (standard, short-clip) evaluation over WorldMem memory-policy sweep
# runs. Ported from MemCam's slurm/newton_vbench_context_memory_60s.sbatch --
# same VBench invocation and dimension set, adapted to WorldMem's actual video
# layout: MemCam's runs are a flat directory of *_60s_custom.mp4 files, but
# WorldMem's smoke script saves predictions under
# <run_dir>/videos/test_vis/pred/video_batch*_0_rank0.mp4 (see log_video() in
# utils/logging_utils.py). VBench's custom_input mode just lists whatever
# video files are in --videos_path (vbench/__init__.py:build_full_info_json),
# so no renaming step is needed -- point it straight at the pred/ folder.
#
# Dimensions are the 6 MemCam settled on: none of them require a text prompt
# to score against (unlike e.g. overall_consistency or category dimensions),
# which matters here since neither MemCam's nor WorldMem's saved filenames
# encode a meaningful prompt -- these dimensions ignore that entirely.
#
# Needs a VBench-capable environment (CLIP/RAFT/DOVER-style dependencies
# beyond what generation itself needs) -- MemCam's sbatch activates a
# dedicated "vbench" conda env before running. If that env already exists on
# this machine (it should, if MemCam's VBench runs have worked here before),
# activate it before calling this script; this script does not manage conda
# itself, matching every other script in this directory.
#
# Usage:
#   conda activate vbench   # or whatever this machine's VBench env is named
#   bash scripts/run_worldmem_vbench.sh
#
# Cost note: 6 quality-model dimensions x every video in every run adds up
# fast -- the default RUNS below is the full 21-cell sweep (315 videos at
# n=15). Consider narrowing RUNS to a handful of cells first to gauge
# per-video wall-clock before committing to the whole grid, the same way the
# MCE generation sweep was profiled before being queued in full.

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
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_ROOT/metrics/vbench_results}"
DIMENSIONS="${DIMENSIONS:-subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality}"
FORCE="${FORCE:-0}"

RUNS="${RUNS:-worldmem_unbounded_60s_n30 \
worldmem_fifo_b16_60s_n30 worldmem_fifo_b32_60s_n30 worldmem_fifo_b64_60s_n30 worldmem_fifo_b128_60s_n30 \
worldmem_rarity_irreplaceability_b16_60s_n30 worldmem_rarity_irreplaceability_b32_60s_n30 worldmem_rarity_irreplaceability_b64_60s_n30 worldmem_rarity_irreplaceability_b128_60s_n30 \
worldmem_slam_covisibility_b16_60s_n30 worldmem_slam_covisibility_b32_60s_n30 worldmem_slam_covisibility_b64_60s_n30 worldmem_slam_covisibility_b128_60s_n30 \
worldmem_kcenter_coreset_b16_60s_n15 worldmem_kcenter_coreset_b32_60s_n15 worldmem_kcenter_coreset_b64_60s_n15 worldmem_kcenter_coreset_b128_60s_n15 \
worldmem_mce_b16_60s_n15 worldmem_mce_b32_60s_n15 worldmem_mce_b64_60s_n15 worldmem_mce_b128_60s_n15}"

mkdir -p "$OUTPUT_ROOT"
cd "$VBENCH_ROOT"

echo "WorldMem VBench batch eval"
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
  python evaluate.py \
    --dimension $DIMENSIONS \
    --videos_path "$video_dir" \
    --mode custom_input \
    --output_path "$out_dir"
  echo "[done] $run  $(date)"
done

echo "Finished: $(date)"
echo "Results under: $OUTPUT_ROOT"
echo "View with: python utils/aggregate_vbench_results.py --root $OUTPUT_ROOT"
