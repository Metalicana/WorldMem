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
CACHE_DIR="${CACHE_DIR:-$SOURCE_ROOT/metrics/retrieval_deterioration_unbounded_60s/feature_cache}"
OUTPUT="${OUTPUT:-$STORAGE_ROOT/outputs/retention_selection_audits/$(date +%F_%H%M%S)}"
CUDA_VISIBLE_DEVICES="" python "$REPO_ROOT/utils/audit_worldmem_retention_selection.py" \
  --suite-root "${SUITE_ROOT:-$STORAGE_ROOT/outputs/memory_policy}" \
  --source-root "$SOURCE_ROOT" \
  --source-run "${SOURCE_RUN:-worldmem_memquality_unbounded_60s_n15_seed101}" \
  --cache-dir "$CACHE_DIR" \
  --expected-videos "${NUM_VIDEOS:-15}" \
  --output "$OUTPUT"
