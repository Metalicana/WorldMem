#!/usr/bin/env bash
# Audit how many videos have actually completed per (policy, budget, duration)
# run directory under the memory-policy sweep output tree -- the same
# audit-style table MemCam produces for its context-memory sweep.
#
# Reuses the exact "contiguous batch count" logic run_worldmem_memory_policy_smoke.sh
# already uses for its own --resume support (count_completed_batches): counts
# video_batch{NNNNN}_*_rank*.mp4 files >4KB under videos/test_vis/pred, and
# reports the longest unbroken run starting at 0 (so an in-flight job with a
# gap from a crashed/skipped batch shows the honest resumable count, not just
# a raw file tally).
#
# Usage:
#   bash scripts/audit_worldmem_memory_policy_runs.sh [ROOT]
#   AUDIT_FILTER='_n30$|_n15$' bash scripts/audit_worldmem_memory_policy_runs.sh
#
# ROOT defaults to the same STORAGE_ROOT/outputs/memory_policy the smoke
# script writes to (CECSL /data/ab575577/worldmem, else $HOME/worldmem_results,
# both overridable with WORLDMEM_STORAGE_ROOT).
#
# AUDIT_FILTER, if set, is an extended regex (grep -E) applied to each run
# directory's basename -- only matching runs are listed. Use it to cut through
# one-off smoke/debug directories (e.g. *_smoke, stray *_n1.._n14 probes) that
# sit alongside the real sweep cells in the same output tree.

set -euo pipefail

if [ -d /data/ab575577 ]; then
  DEFAULT_STORAGE_ROOT="/data/ab575577/worldmem"
else
  DEFAULT_STORAGE_ROOT="$HOME/worldmem_results"
fi
STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$DEFAULT_STORAGE_ROOT}"
ROOT="${1:-$STORAGE_ROOT/outputs/memory_policy}"

if [ ! -d "$ROOT" ]; then
  echo "No such directory: $ROOT" >&2
  exit 1
fi

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
        { idx = $1 + 0
          if (idx == expected) { expected += 1 }
          else if (idx > expected) { exit } }
        END { print expected }
      '
}

# Parse "worldmem_<policy>[_b<budget>][_<duration>s]_n<n>" from the right so
# policy names that themselves contain underscores (rarity_irreplaceability,
# slam_covisibility, kcenter_coreset) parse correctly.
parse_run_name() {
  local name="$1" rest requested="" duration="" budget="" policy=""
  rest="${name#worldmem_}"
  if [[ "$rest" =~ _n([0-9]+)$ ]]; then
    requested="${BASH_REMATCH[1]}"
    rest="${rest%_n${requested}}"
  fi
  if [[ "$rest" =~ _([0-9]+)s$ ]]; then
    duration="${BASH_REMATCH[1]}"
    rest="${rest%_${duration}s}"
  fi
  if [[ "$rest" =~ _b([0-9]+)$ ]]; then
    budget="${BASH_REMATCH[1]}"
    rest="${rest%_b${budget}}"
  fi
  policy="$rest"
  # Pipe-delimited, not tab: bash `read` collapses consecutive IFS-whitespace
  # delimiters (tab counts as whitespace even when set explicitly), which
  # would silently swallow empty fields -- e.g. the empty "budget" field for
  # unbounded runs -- and shift every field after it by one.
  printf '%s|%s|%s|%s\n' "$policy" "$budget" "$duration" "$requested"
}

header=$(printf '%-42s %-24s %8s %9s %6s/%-6s %6s' \
  "run" "policy" "budget" "duration" "done" "total" "pct")
echo "Root: $ROOT"
echo "$header"
printf '%s\n' "${header//?/-}"

total_requested=0
total_done=0
for run_dir in "$ROOT"/*/; do
  [ -d "$run_dir" ] || continue
  run_name="$(basename "$run_dir")"
  if [ -n "${AUDIT_FILTER:-}" ] && ! echo "$run_name" | grep -qE "$AUDIT_FILTER"; then
    continue
  fi
  IFS='|' read -r policy budget duration requested <<< "$(parse_run_name "$run_name")"
  done_count="$(count_completed_batches "$run_dir/videos/test_vis/pred")"
  if [ -n "$requested" ]; then
    pct=$(awk -v d="$done_count" -v t="$requested" 'BEGIN { printf "%5.1f%%", (t>0 ? 100.0*d/t : 0) }')
    total_requested=$((total_requested + requested))
  else
    requested="?"
    pct="    ?"
  fi
  total_done=$((total_done + done_count))
  printf '%-42s %-24s %8s %9s %6s/%-6s %6s\n' \
    "$run_name" "${policy:-?}" "${budget:-none}" "${duration:-?}s" \
    "$done_count" "$requested" "$pct"
done

printf '%s\n' "${header//?/-}"
echo "Total completed videos across all runs: $total_done"
if [ "$total_requested" -gt 0 ]; then
  echo "Total requested (where run-name encodes n): $total_requested"
fi
