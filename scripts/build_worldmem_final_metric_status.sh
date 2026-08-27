#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${WORLDMEM_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/memory_policy}"
METRICS_ROOT="$OUTPUT_ROOT/metrics"
STATUS_DIR="${STATUS_DIR:-$METRICS_ROOT/final_status}"
LPIPS_SUMMARY="${LPIPS_SUMMARY:-$METRICS_ROOT/lpips_prefix_60s_n15/summary.csv}"
FVD_SUMMARY="${FVD_SUMMARY:-$METRICS_ROOT/fvd_prefix_60s_n15/summary.csv}"
VBENCH_ROOT="${VBENCH_RESULTS_ROOT:-$METRICS_ROOT/vbench_results}"
VBENCH_LONG_ROOT="${VBENCH_LONG_RESULTS_ROOT:-$METRICS_ROOT/vbench_long_results}"
CUT3R_SUMMARY="${CUT3R_SUMMARY:-$METRICS_ROOT/cut3r_camera_metrics/cut3r_camera_summary.csv}"
CUT3R_VALIDITY="${CUT3R_VALIDITY:-$METRICS_ROOT/cut3r_camera_metrics_gt_sanity/validity.json}"

cd "$REPO_ROOT"
python utils/build_worldmem_final_metric_status.py \
  --output-root "$OUTPUT_ROOT" \
  --repo-root "$REPO_ROOT" \
  --metrics-dir "$STATUS_DIR" \
  --lpips-summary "$LPIPS_SUMMARY" \
  --fvd-summary "$FVD_SUMMARY" \
  --vbench-root "$VBENCH_ROOT" \
  --vbench-long-root "$VBENCH_LONG_ROOT" \
  --cut3r-summary "$CUT3R_SUMMARY" \
  --cut3r-validity "$CUT3R_VALIDITY" \
  --limit "${LIMIT:-15}"
