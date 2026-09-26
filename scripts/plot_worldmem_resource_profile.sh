#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDMEM_REPO_ROOT="${WORLDMEM_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PROFILE_ROOT="${PROFILE_ROOT:?Set PROFILE_ROOT to the completed profiler directory}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROFILE_ROOT/resource_report}"
BOUNDED_POLICY="${BOUNDED_POLICY:-slam_covisibility}"
BOUNDED_LABEL="${BOUNDED_LABEL:-KEEPSAKE}"
MEMORY_BUDGET="${MEMORY_BUDGET:-32}"
FPS="${FPS:-10}"
FUTURE_SECONDS="${FUTURE_SECONDS:-60}"

cd "$WORLDMEM_REPO_ROOT"

echo "WorldMem measured resource figures"
echo "Profile root: $PROFILE_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Bounded method: $BOUNDED_LABEL ($BOUNDED_POLICY, B=$MEMORY_BUDGET)"

python utils/plot_worldmem_resource_profile.py \
  --profile-root "$PROFILE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --bounded-policy "$BOUNDED_POLICY" \
  --bounded-label "$BOUNDED_LABEL" \
  --budget "$MEMORY_BUDGET" \
  --fps "$FPS" \
  --expected-duration-sec "$FUTURE_SECONDS"
