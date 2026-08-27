#!/usr/bin/env bash
# VBench (standard, short-clip) evaluation over WorldMem memory-policy sweep
# runs. Ported from MemCam's slurm/newton_vbench_context_memory_60s.sbatch --
# same VBench invocation and dimension set, adapted to WorldMem's actual video
# layout: MemCam's runs are a flat directory of *_60s_custom.mp4 files, but
# WorldMem's smoke script saves predictions under
# <run_dir>/videos/test_vis/pred/video_batch*_0_rank0.mp4. This wrapper stages
# exactly the requested parsed batch IDs before invoking VBench.
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
# The default is the frozen six-policy B32 roster, always matched on batch IDs
# 0..14. Override RUNS only for an explicitly controlled ablation.

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
LIMIT="${LIMIT:-15}"
STAGING_ROOT="${STAGING_ROOT:-$OUTPUT_ROOT/_input_staging}"

RUNS="${RUNS:-worldmem_unbounded_60s_n30 \
worldmem_fifo_b32_60s_n30 \
worldmem_rarity_irreplaceability_b32_60s_n30 \
worldmem_slam_covisibility_b32_60s_n30 \
worldmem_kcenter_coreset_b32_60s_n15 \
worldmem_mce_b32_60s_n15}"

mkdir -p "$OUTPUT_ROOT" "$STAGING_ROOT"
cd "$VBENCH_ROOT"

echo "WorldMem VBench batch eval"
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
    --mode custom_input
    --dimensions "$DIMENSIONS"
  )
  if [ "$FORCE" = "1" ]; then
    stage_cmd+=(--reset-derived)
  fi
  "${stage_cmd[@]}"

  if compgen -G "$out_dir"/*_eval_results.json > /dev/null 2>&1 && [ "$FORCE" != "1" ]; then
    if [ ! -f "$out_dir/input_selection.json" ] || ! cmp -s "$stage_metadata" "$out_dir/input_selection.json"; then
      echo "[error] $run has stale or unmatched VBench output. Inspect it or rerun explicitly with FORCE=1." >&2
      exit 1
    fi
    echo "[skip] $run already has matched first-$LIMIT eval results in $out_dir"
    continue
  fi

  echo "[run] $run  ($LIMIT matched videos)  $(date)"
  python evaluate.py \
    --dimension $DIMENSIONS \
    --videos_path "$stage_dir" \
    --mode custom_input \
    --output_path "$out_dir"
  mkdir -p "$out_dir"
  cp "$stage_metadata" "$out_dir/input_selection.json"
  echo "[done] $run  $(date)"
done

echo "Finished: $(date)"
echo "Results under: $OUTPUT_ROOT"
echo "View with: python utils/aggregate_vbench_results.py --root $OUTPUT_ROOT"
