#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORLDMEM_REPO_ROOT="${WORLDMEM_REPO_ROOT:-$DEFAULT_REPO_ROOT}"

if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/memory_policy}"
DATA_DIR="${WORLDMEM_DATA_DIR:-data/minecraft}"
RUNS="${RUNS:-worldmem_unbounded_60s_n30,worldmem_fifo_b32_60s_n30,worldmem_fifo_b64_60s_n30,worldmem_rarity_irreplaceability_b32_60s_n30,worldmem_rarity_irreplaceability_b64_60s_n30,worldmem_slam_covisibility_b32_60s_n30,worldmem_slam_covisibility_b64_60s_n30}"
CUT3R_ROOT="${CUT3R_ROOT:-$HOME/MemCam/CUT3R}"
CUT3R_MODEL="${CUT3R_MODEL:-$CUT3R_ROOT/src/cut3r_512_dpt_4_64.pth}"
CUT3R_SIZE="${CUT3R_SIZE:-512}"
CUT3R_DEVICE="${CUT3R_DEVICE:-cuda}"
DURATION_SEC="${DURATION_SEC:-60}"
FPS="${FPS:-10}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-600}"
N_FRAMES_VALID="${N_FRAMES_VALID:-$((CONTEXT_FRAMES + DURATION_SEC * FPS))}"
SEED="${SEED:-42}"
FRAME_STRIDE="${FRAME_STRIDE:-30}"
MAX_FRAMES="${MAX_FRAMES:-120}"
ROWS="${ROWS:-}"
LIMIT="${LIMIT:-}"
RECON_DIR="${RECON_DIR:-$OUTPUT_ROOT/metrics/cut3r_pose_recon}"
METRICS_DIR="${METRICS_DIR:-$OUTPUT_ROOT/metrics/cut3r_camera_metrics}"

if TORCH_LIB_DIR="$(python - <<'PY' 2>/dev/null
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent / "lib")
PY
)"; then
  if [ -d "$TORCH_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"
  fi
fi

cd "$WORLDMEM_REPO_ROOT"

run_args=(
  python utils/run_cut3r_worldmem.py
  --output_root "$OUTPUT_ROOT"
  --data_dir "$DATA_DIR"
  --runs "$RUNS"
  --output_dir "$RECON_DIR"
  --cut3r_root "$CUT3R_ROOT"
  --model_path "$CUT3R_MODEL"
  --size "$CUT3R_SIZE"
  --device "$CUT3R_DEVICE"
  --duration_sec "$DURATION_SEC"
  --fps "$FPS"
  --context_frames "$CONTEXT_FRAMES"
  --n_frames_valid "$N_FRAMES_VALID"
  --seed "$SEED"
  --frame_stride "$FRAME_STRIDE"
  --max_frames "$MAX_FRAMES"
)

if [ -n "$ROWS" ]; then
  run_args+=(--rows "$ROWS")
fi
if [ -n "$LIMIT" ]; then
  run_args+=(--limit "$LIMIT")
fi
if [ "${FORCE:-0}" = "1" ]; then
  run_args+=(--force)
fi

eval_args=(
  python utils/evaluate_cut3r_worldmem_camera_metrics.py
  --cut3r_dir "$RECON_DIR"
  --output_dir "$METRICS_DIR"
  --runs "$RUNS"
  --duration_sec "$DURATION_SEC"
)
if [ -n "$ROWS" ]; then
  eval_args+=(--rows "$ROWS")
fi

echo "WorldMem CUT3R camera metrics"
echo "Output root: $OUTPUT_ROOT"
echo "Data dir: $DATA_DIR"
echo "Runs: $RUNS"
echo "CUT3R root: $CUT3R_ROOT"
echo "CUT3R model: $CUT3R_MODEL"
echo "Torch lib dir: ${TORCH_LIB_DIR:-unknown}"
echo "Duration seconds: $DURATION_SEC"
echo "Context frames: $CONTEXT_FRAMES"
echo "N frames valid: $N_FRAMES_VALID"
echo "Recon dir: $RECON_DIR"
echo "Metrics dir: $METRICS_DIR"
echo "Rows: ${ROWS:-all}"
echo "Limit: ${LIMIT:-none}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'Run command:'
  printf ' %q' "${run_args[@]}"
  printf '\n'
  printf 'Eval command:'
  printf ' %q' "${eval_args[@]}"
  printf '\n'
  exit 0
fi

"${run_args[@]}"
"${eval_args[@]}"
