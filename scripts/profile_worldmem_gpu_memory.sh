#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORLDMEM_REPO_ROOT="${WORLDMEM_REPO_ROOT:-$DEFAULT_REPO_ROOT}"
if [ -d /data/ab575577 ]; then
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-/data/ab575577/worldmem}"
else
  STORAGE_ROOT="${WORLDMEM_STORAGE_ROOT:-$HOME/worldmem_results}"
fi

FUTURE_SECONDS="${FUTURE_SECONDS:-60}"
NUM_VIDEOS="${NUM_VIDEOS:-1}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-600}"
FPS="${FPS:-10}"
SAMPLING_TIMESTEPS="${SAMPLING_TIMESTEPS:-20}"
DECODE_CHUNK_SIZE="${DECODE_CHUNK_SIZE:-32}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1}"
MINE_POLICY="${MINE_POLICY:-rarity_irreplaceability}"
MINE_BUDGETS="${MINE_BUDGETS:-32}"
SAVE_LOCAL_PER_BATCH="${SAVE_LOCAL_PER_BATCH:-false}"
LOG_VIDEO="${LOG_VIDEO:-false}"
PROFILE_ROOT="${PROFILE_ROOT:-$STORAGE_ROOT/outputs/memory_policy/gpu_memory_profiles/$(date +%F_%H%M%S)}"
SUMMARY_CSV="$PROFILE_ROOT/summary.csv"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. This profiler must run on a CUDA machine." >&2
  exit 2
fi

mkdir -p "$PROFILE_ROOT/logs" "$PROFILE_ROOT/runs" "$PROFILE_ROOT/access_traces" "$PROFILE_ROOT/nvidia_smi"

echo "WorldMem GPU memory profile"
echo "GPU: $GPU"
echo "Repo root: $WORLDMEM_REPO_ROOT"
echo "Storage root: $STORAGE_ROOT"
echo "Profile root: $PROFILE_ROOT"
echo "Future seconds: $FUTURE_SECONDS"
echo "Videos per run: $NUM_VIDEOS"
echo "Mine policy: $MINE_POLICY"
echo "Mine budgets: $MINE_BUDGETS"
echo

cat > "$SUMMARY_CSV" <<'CSV'
run_name,policy,budget,future_seconds,num_videos,status,wall_seconds,baseline_nvidia_smi_used_mib,peak_nvidia_smi_used_mib,net_peak_nvidia_smi_used_mib,peak_nvidia_smi_util_percent,peak_torch_allocated_mib,peak_torch_reserved_mib,output_dir,trace_path,nvidia_smi_log,run_log
CSV

SAMPLER_PID=""
cleanup_sampler() {
  if [ -n "$SAMPLER_PID" ]; then
    kill "$SAMPLER_PID" >/dev/null 2>&1 || true
    wait "$SAMPLER_PID" >/dev/null 2>&1 || true
    SAMPLER_PID=""
  fi
}
trap cleanup_sampler EXIT INT TERM

start_sampler() {
  local gpu_log="$1"
  cat > "$gpu_log" <<'CSV'
sample_time_iso,memory_used_mib,memory_free_mib,memory_total_mib,utilization_gpu_percent
CSV
  (
    while true; do
      values="$(nvidia-smi --id="$GPU" --query-gpu=memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits | head -n 1 || true)"
      if [ -n "$values" ]; then
        printf '%s,%s\n' "$(date -Is)" "$values" >> "$gpu_log"
      fi
      sleep "$SAMPLE_INTERVAL"
    done
  ) &
  SAMPLER_PID="$!"
}

summarize_nvidia_smi_log() {
  local gpu_log="$1"
  python - "$gpu_log" <<'PY'
import csv
import math
import sys

path = sys.argv[1]
used = []
utils = []
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        try:
            used.append(float(row["memory_used_mib"].strip()))
            utils.append(float(row["utilization_gpu_percent"].strip()))
        except (KeyError, TypeError, ValueError):
            pass

if not used:
    print("nan,nan,nan,nan")
else:
    baseline = used[0]
    peak = max(used)
    net = peak - baseline
    util = max(utils) if utils else math.nan
    print(f"{baseline:.3f},{peak:.3f},{net:.3f},{util:.3f}")
PY
}

summarize_trace_log() {
  local trace_path="$1"
  python - "$trace_path" <<'PY'
import json
import math
import sys

path = sys.argv[1]
peak_allocated = math.nan
peak_reserved = math.nan
try:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not str(record.get("event", "")).startswith("cuda_memory_"):
                continue
            allocated = record.get("max_memory_allocated_mib")
            reserved = record.get("max_memory_reserved_mib")
            if allocated is not None:
                peak_allocated = max(peak_allocated, float(allocated)) if not math.isnan(peak_allocated) else float(allocated)
            if reserved is not None:
                peak_reserved = max(peak_reserved, float(reserved)) if not math.isnan(peak_reserved) else float(reserved)
except FileNotFoundError:
    pass

def fmt(value):
    return "nan" if math.isnan(value) else f"{value:.3f}"

print(f"{fmt(peak_allocated)},{fmt(peak_reserved)}")
PY
}

run_profile() {
  local policy="$1"
  local budget="$2"
  local run_name

  if [ -n "$budget" ]; then
    run_name="worldmem_gpu_profile_${policy}_b${budget}_${FUTURE_SECONDS}s_n${NUM_VIDEOS}"
  else
    run_name="worldmem_gpu_profile_${policy}_${FUTURE_SECONDS}s_n${NUM_VIDEOS}"
  fi

  local output_dir="$PROFILE_ROOT/runs/$run_name"
  local trace_path="$PROFILE_ROOT/access_traces/$run_name.jsonl"
  local gpu_log="$PROFILE_ROOT/nvidia_smi/$run_name.csv"
  local run_log="$PROFILE_ROOT/logs/$run_name.log"

  echo "============================================================"
  echo "Profiling: $run_name"
  echo "Policy: $policy"
  echo "Budget: ${budget:-none}"
  echo "GPU log: $gpu_log"
  echo "Trace: $trace_path"
  echo "============================================================"

  cleanup_sampler
  start_sampler "$gpu_log"

  local start_epoch
  start_epoch="$(date +%s)"
  set +e
  if [ -n "$budget" ]; then
    GPU="$GPU" \
    WORLDMEM_REPO_ROOT="$WORLDMEM_REPO_ROOT" \
    WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
    MEMORY_POLICY="$policy" \
    MEMORY_BUDGET="$budget" \
    FUTURE_SECONDS="$FUTURE_SECONDS" \
    NUM_VIDEOS="$NUM_VIDEOS" \
    CONTEXT_FRAMES="$CONTEXT_FRAMES" \
    FPS="$FPS" \
    SAMPLING_TIMESTEPS="$SAMPLING_TIMESTEPS" \
    DECODE_CHUNK_SIZE="$DECODE_CHUNK_SIZE" \
    RUN_NAME="$run_name" \
    OUTPUT_DIR="$output_dir" \
    TRACE_PATH="$trace_path" \
    PROFILE_CUDA_MEMORY=true \
    LOG_VIDEO="$LOG_VIDEO" \
    SAVE_LOCAL_PER_BATCH="$SAVE_LOCAL_PER_BATCH" \
    SAVE_GT_VIDEO=false \
    COMPUTE_EVAL_METRICS=false \
    STREAM_EVAL_METRICS=false \
    RESUME_PARTIAL=0 \
    SKIP_COMPLETED=0 \
    WANDB_MODE=disabled \
    bash "$SCRIPT_DIR/run_worldmem_memory_policy_smoke.sh" 2>&1 | tee "$run_log"
  else
    GPU="$GPU" \
    WORLDMEM_REPO_ROOT="$WORLDMEM_REPO_ROOT" \
    WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
    MEMORY_POLICY="$policy" \
    MEMORY_BUDGET="" \
    FUTURE_SECONDS="$FUTURE_SECONDS" \
    NUM_VIDEOS="$NUM_VIDEOS" \
    CONTEXT_FRAMES="$CONTEXT_FRAMES" \
    FPS="$FPS" \
    SAMPLING_TIMESTEPS="$SAMPLING_TIMESTEPS" \
    DECODE_CHUNK_SIZE="$DECODE_CHUNK_SIZE" \
    RUN_NAME="$run_name" \
    OUTPUT_DIR="$output_dir" \
    TRACE_PATH="$trace_path" \
    PROFILE_CUDA_MEMORY=true \
    LOG_VIDEO="$LOG_VIDEO" \
    SAVE_LOCAL_PER_BATCH="$SAVE_LOCAL_PER_BATCH" \
    SAVE_GT_VIDEO=false \
    COMPUTE_EVAL_METRICS=false \
    STREAM_EVAL_METRICS=false \
    RESUME_PARTIAL=0 \
    SKIP_COMPLETED=0 \
    WANDB_MODE=disabled \
    bash "$SCRIPT_DIR/run_worldmem_memory_policy_smoke.sh" 2>&1 | tee "$run_log"
  fi
  local status="${PIPESTATUS[0]}"
  set -e
  local end_epoch
  local wall_seconds
  end_epoch="$(date +%s)"
  wall_seconds=$((end_epoch - start_epoch))

  cleanup_sampler

  local nvidia_summary
  local torch_summary
  nvidia_summary="$(summarize_nvidia_smi_log "$gpu_log")"
  torch_summary="$(summarize_trace_log "$trace_path")"
  IFS=',' read -r baseline_used peak_used net_peak_used peak_util <<< "$nvidia_summary"
  IFS=',' read -r peak_torch_allocated peak_torch_reserved <<< "$torch_summary"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$run_name" \
    "$policy" \
    "${budget:-}" \
    "$FUTURE_SECONDS" \
    "$NUM_VIDEOS" \
    "$status" \
    "$wall_seconds" \
    "$baseline_used" \
    "$peak_used" \
    "$net_peak_used" \
    "$peak_util" \
    "$peak_torch_allocated" \
    "$peak_torch_reserved" \
    "$output_dir" \
    "$trace_path" \
    "$gpu_log" \
    "$run_log" >> "$SUMMARY_CSV"

  echo
  echo "Current summary:"
  column -s, -t "$SUMMARY_CSV" || cat "$SUMMARY_CSV"
  echo

  return "$status"
}

run_profile "unbounded" ""

IFS=',' read -r -a BUDGET_ARRAY <<< "$MINE_BUDGETS"
for budget in "${BUDGET_ARRAY[@]}"; do
  budget="${budget//[[:space:]]/}"
  [ -n "$budget" ] || continue
  run_profile "$MINE_POLICY" "$budget"
done

echo "Wrote summary: $SUMMARY_CSV"
