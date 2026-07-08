#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_VIDEOS="${NUM_VIDEOS:-15}"
DURATIONS="${DURATIONS:-180}"
POLICIES="${POLICIES:-unbounded,fifo,rarity_irreplaceability,slam_covisibility}"
BUDGETS="${BUDGETS:-32,64}"

export NUM_VIDEOS DURATIONS POLICIES BUDGETS

echo "WorldMem 180s tight memory-policy grid"
echo "Videos per run: $NUM_VIDEOS"
echo "Durations: $DURATIONS"
echo "Policies: $POLICIES"
echo "Budgets: $BUDGETS"

bash "$SCRIPT_DIR/run_worldmem_memory_policy_grid.sh"
