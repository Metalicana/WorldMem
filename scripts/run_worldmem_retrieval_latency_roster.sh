#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export POLICY_SPECS="unbounded: fifo:32 mce:32 kcenter_coreset:32 rarity_irreplaceability:32 slam_covisibility:32"
export SUMMARIZE_ALL_POLICIES=true
export NUM_VIDEOS="${NUM_VIDEOS:-15}"
if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi
export OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/retrieval_latency_roster_60s_n${NUM_VIDEOS}}"
bash "$SCRIPT_DIR/run_worldmem_retrieval_latency.sh"
