#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${WORLDMEM_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

RI_RUN="worldmem_rarity_irreplaceability_b32_60s_n30"
KEEPSAKE_RUN="worldmem_slam_covisibility_b32_60s_n30"
CURRENT_ROOT="${CURRENT_ROOT:-$STORAGE_ROOT/outputs/memory_policy/metrics/vbench_budget_sweep_60s_n15}"
N30_ROOT="${N30_ROOT:-$STORAGE_ROOT/outputs/memory_policy/metrics/vbench_ri_keepsake_60s_n30}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$STORAGE_ROOT/outputs/memory_policy/metrics/vbench6_paired}"

echo "Stage 1/3: paired analysis of the existing N=15 results"
if [ -f "$ANALYSIS_ROOT/n15/summary.json" ]; then
  echo "[skip] completed N=15 paired analysis"
else
  python "$REPO_ROOT/utils/analyze_worldmem_vbench6_pair.py" \
    --root "$CURRENT_ROOT" \
    --left-run "$RI_RUN" \
    --right-run "$KEEPSAKE_RUN" \
    --limit 15 \
    --output "$ANALYSIS_ROOT/n15"
fi

echo "Stage 2/3: evaluate the existing 30 videos for RI and KEEPSAKE"
RUNS="$RI_RUN $KEEPSAKE_RUN" \
LIMIT=30 \
OUTPUT_ROOT="$N30_ROOT" \
WORLDMEM_REPO_ROOT="$REPO_ROOT" \
WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
bash "$REPO_ROOT/scripts/run_worldmem_vbench.sh"

echo "Stage 3/3: paired analysis of N=30"
if [ -f "$ANALYSIS_ROOT/n30/summary.json" ]; then
  echo "[skip] completed N=30 paired analysis"
else
  python "$REPO_ROOT/utils/analyze_worldmem_vbench6_pair.py" \
    --root "$N30_ROOT" \
    --left-run "$RI_RUN" \
    --right-run "$KEEPSAKE_RUN" \
    --limit 30 \
    --output "$ANALYSIS_ROOT/n30"
fi

echo "N=15: $ANALYSIS_ROOT/n15/summary.json"
echo "N=30: $ANALYSIS_ROOT/n30/summary.json"
