#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${WORLDMEM_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

SOURCE_ROOT="${SOURCE_ROOT:-$STORAGE_ROOT/outputs/memory_quality_60s}"
SOURCE_RUN="${SOURCE_RUN:-worldmem_memquality_unbounded_60s_n15_seed101}"
CACHE_DIR="${CACHE_DIR:-$SOURCE_ROOT/metrics/retrieval_deterioration_unbounded_60s/feature_cache}"
OUTPUT="${OUTPUT:-$STORAGE_ROOT/outputs/retention_selection_60s_n15/$(date +%F_%H%M%S)}"

args=(
  --suite-root "${SUITE_ROOT:-$STORAGE_ROOT/outputs/memory_policy}"
  --source-root "$SOURCE_ROOT"
  --source-run "$SOURCE_RUN"
  --cache-dir "$CACHE_DIR"
  --output "$OUTPUT"
  --expected-videos "${NUM_VIDEOS:-15}"
  --query-stride "${QUERY_STRIDE:-1}"
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-5000}"
  --bootstrap-seed "${BOOTSTRAP_SEED:-17}"
)

if [ -n "${UNBOUNDED_TRACE_ROOT:-}" ]; then
  args+=(--unbounded-trace-root "$UNBOUNDED_TRACE_ROOT")
fi
if [ "${ALLOW_INCOMPLETE_COHORT:-0}" = "1" ]; then
  args+=(--allow-incomplete-cohort)
fi

echo "WorldMem common-source retention/selection analysis"
echo "Suite root: ${SUITE_ROOT:-$STORAGE_ROOT/outputs/memory_policy}"
echo "Common source: $SOURCE_ROOT/$SOURCE_RUN"
echo "Feature cache: $CACHE_DIR"
echo "Output: $OUTPUT"
echo "CUDA_VISIBLE_DEVICES: <empty> (cached-feature analysis is CPU only)"

CUDA_VISIBLE_DEVICES="" python "$REPO_ROOT/utils/analyze_worldmem_retention_selection.py" "${args[@]}"
