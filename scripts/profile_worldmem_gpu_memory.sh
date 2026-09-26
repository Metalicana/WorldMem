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
HOST_SAMPLE_INTERVAL="${HOST_SAMPLE_INTERVAL:-$SAMPLE_INTERVAL}"
MINE_POLICY="${MINE_POLICY:-rarity_irreplaceability}"
MINE_POLICIES="${MINE_POLICIES:-$MINE_POLICY}"
MINE_BUDGETS="${MINE_BUDGETS:-32}"
MEMORY_BANK_DEVICES="${MEMORY_BANK_DEVICES:-cpu}"
INCLUDE_UNBOUNDED="${INCLUDE_UNBOUNDED:-1}"
SAVE_LOCAL_PER_BATCH="${SAVE_LOCAL_PER_BATCH:-false}"
LOG_VIDEO="${LOG_VIDEO:-false}"
PROFILE_ROOT="${PROFILE_ROOT:-$STORAGE_ROOT/outputs/memory_policy/gpu_memory_profiles/$(date +%F_%H%M%S)}"
SUMMARY_CSV="$PROFILE_ROOT/summary.csv"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. This profiler must run on a CUDA machine." >&2
  exit 2
fi

mkdir -p \
  "$PROFILE_ROOT/logs" \
  "$PROFILE_ROOT/runs" \
  "$PROFILE_ROOT/access_traces" \
  "$PROFILE_ROOT/nvidia_smi" \
  "$PROFILE_ROOT/host_memory"

echo "WorldMem GPU memory profile"
echo "GPU: $GPU"
echo "Repo root: $WORLDMEM_REPO_ROOT"
echo "Storage root: $STORAGE_ROOT"
echo "Profile root: $PROFILE_ROOT"
echo "Future seconds: $FUTURE_SECONDS"
echo "Videos per run: $NUM_VIDEOS"
echo "Mine policies: $MINE_POLICIES"
echo "Mine budgets: $MINE_BUDGETS"
echo "Include unbounded: $INCLUDE_UNBOUNDED"
echo "Memory bank devices: $MEMORY_BANK_DEVICES"
echo

python - \
  "$PROFILE_ROOT/environment.json" \
  "$WORLDMEM_REPO_ROOT" \
  "$GPU" \
  "$FUTURE_SECONDS" \
  "$NUM_VIDEOS" \
  "$CONTEXT_FRAMES" \
  "$FPS" \
  "$SAMPLING_TIMESTEPS" \
  "$SAMPLE_INTERVAL" \
  "$HOST_SAMPLE_INTERVAL" <<'PY'
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

(
    output_path,
    repo_root,
    physical_gpu,
    future_seconds,
    num_videos,
    context_frames,
    fps,
    sampling_timesteps,
    gpu_sample_interval,
    host_sample_interval,
) = sys.argv[1:]

def command(*args):
    result = subprocess.run(
        args,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None

cuda_available = torch.cuda.is_available()
manifest = {
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "torch_version": str(torch.__version__),
    "torch_cuda_version": str(torch.version.cuda),
    "cudnn_version": torch.backends.cudnn.version(),
    "cuda_available": cuda_available,
    "visible_cuda_device_name": (
        torch.cuda.get_device_name(0) if cuda_available else None
    ),
    "visible_cuda_capability": (
        list(torch.cuda.get_device_capability(0)) if cuda_available else None
    ),
    "physical_gpu_index": int(physical_gpu),
    "nvidia_smi_gpu": command(
        "nvidia-smi",
        f"--id={physical_gpu}",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ),
    "git_commit": command("git", "rev-parse", "HEAD"),
    "git_status_porcelain": command("git", "status", "--short"),
    "profile_settings": {
        "future_seconds": int(future_seconds),
        "num_videos": int(num_videos),
        "context_frames": int(context_frames),
        "trajectory_fps": float(fps),
        "sampling_timesteps": int(sampling_timesteps),
        "gpu_poll_interval_seconds": float(gpu_sample_interval),
        "host_poll_interval_seconds": float(host_sample_interval),
    },
}
Path(output_path).write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)
PY

cat > "$SUMMARY_CSV" <<'CSV'
run_name,policy,budget,memory_bank_device,future_seconds,num_videos,status,wall_seconds,total_seconds,retrieval_seconds,retrieval_query_count,retrieval_ms_mean,retrieval_ms_median,retrieval_ms_p95,sampling_seconds,memory_update_seconds,memory_update_count,memory_update_ms_mean,initial_memory_update_seconds,generation_memory_update_seconds,descriptor_extraction_seconds,descriptor_calls,descriptor_frames_requested,descriptor_ms_per_call,memory_update_excluding_descriptor_seconds,decode_seconds,peak_archive_mib,peak_archive_frames,final_archive_mib,final_archive_frames,peak_history_mib,final_history_mib,peak_bank_mib,peak_bank_frames,baseline_process_tree_rss_mib,peak_process_tree_rss_mib,baseline_system_memory_used_mib,peak_system_memory_used_mib,net_peak_system_memory_used_mib,baseline_nvidia_smi_used_mib,peak_nvidia_smi_used_mib,net_peak_nvidia_smi_used_mib,peak_nvidia_smi_util_percent,peak_torch_allocated_mib,peak_torch_reserved_mib,output_dir,trace_path,nvidia_smi_log,host_memory_log,run_log
CSV

SAMPLER_PID=""
HOST_SAMPLER_PID=""
cleanup_sampler() {
  if [ -n "$SAMPLER_PID" ]; then
    kill "$SAMPLER_PID" >/dev/null 2>&1 || true
    wait "$SAMPLER_PID" >/dev/null 2>&1 || true
    SAMPLER_PID=""
  fi
  if [ -n "$HOST_SAMPLER_PID" ]; then
    kill "$HOST_SAMPLER_PID" >/dev/null 2>&1 || true
    wait "$HOST_SAMPLER_PID" >/dev/null 2>&1 || true
    HOST_SAMPLER_PID=""
  fi
}

start_host_sampler() {
  local root_pid="$1"
  local host_log="$2"
  python "$WORLDMEM_REPO_ROOT/utils/sample_process_tree_memory.py" \
    --root-pid "$root_pid" \
    --output "$host_log" \
    --interval "$HOST_SAMPLE_INTERVAL" &
  HOST_SAMPLER_PID="$!"
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

summarize_timing_log() {
  local trace_path="$1"
  python - "$trace_path" <<'PY'
import json
import math
import sys

path = sys.argv[1]
keys = [
    "total_seconds",
    "retrieval_seconds",
    "sampling_seconds",
    "memory_update_seconds",
    "chunks",
    "initial_memory_update_seconds",
    "generation_memory_update_seconds",
    "descriptor_extraction_seconds",
    "descriptor_calls",
    "descriptor_frames_requested",
    "memory_update_excluding_descriptor_seconds",
    "decode_seconds",
]
values = {key: math.nan for key in keys}
try:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") != "runtime_breakdown":
                continue
            for key in keys:
                if record.get(key) is not None:
                    values[key] = float(record[key])
except FileNotFoundError:
    pass

def fmt(value):
    return "nan" if math.isnan(value) else f"{value:.3f}"

print(",".join(fmt(values[key]) for key in keys))
PY
}

summarize_bank_log() {
  local trace_path="$1"
  python - "$trace_path" <<'PY'
import json
import math
import sys

path = sys.argv[1]
peak_mib = math.nan
peak_frames = 0
peak_archive_mib = math.nan
peak_archive_frames = 0
final_archive_mib = math.nan
final_archive_frames = 0
peak_history_mib = math.nan
final_history_mib = math.nan
try:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") != "gpu_memory_bank_sync":
                if record.get("event") != "memory_archive_state":
                    continue
                archive_mib = float(record["archive_latent_payload_mib"])
                archive_frames = int(record["archive_frames"])
                history_mib = float(record["history_latent_payload_mib"])
                peak_archive_mib = max(peak_archive_mib, archive_mib) if not math.isnan(peak_archive_mib) else archive_mib
                peak_archive_frames = max(peak_archive_frames, archive_frames)
                final_archive_mib = archive_mib
                final_archive_frames = archive_frames
                peak_history_mib = max(peak_history_mib, history_mib) if not math.isnan(peak_history_mib) else history_mib
                final_history_mib = history_mib
                continue
            mib = record.get("estimated_bank_mib")
            frames = record.get("stored_memory_size")
            if mib is not None:
                peak_mib = max(peak_mib, float(mib)) if not math.isnan(peak_mib) else float(mib)
            if frames is not None:
                peak_frames = max(peak_frames, int(frames))
except FileNotFoundError:
    pass

def fmt(value):
    return "nan" if math.isnan(value) else f"{value:.6f}"

print(",".join([
    fmt(peak_archive_mib),
    str(peak_archive_frames),
    fmt(final_archive_mib),
    str(final_archive_frames),
    fmt(peak_history_mib),
    fmt(final_history_mib),
    fmt(peak_mib),
    str(peak_frames),
]))
PY
}

summarize_retrieval_queries() {
  local trace_path="$1"
  python - "$trace_path" <<'PY'
import json
import math
import statistics
import sys

values = []
try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "retrieval_query_profile":
                values.append(float(record["query_milliseconds"]))
except FileNotFoundError:
    pass

if not values:
    print("0,nan,nan,nan")
else:
    ordered = sorted(values)
    rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
    print(
        f"{len(values)},{statistics.fmean(values):.6f},"
        f"{statistics.median(values):.6f},{ordered[rank]:.6f}"
    )
PY
}

summarize_host_log() {
  local host_log="$1"
  python - "$host_log" <<'PY'
import csv
import math
import sys

rss = []
system_used = []
try:
    with open(sys.argv[1], newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rss.append(float(row["process_tree_rss_mib"]))
            system_used.append(float(row["system_memory_used_mib"]))
except FileNotFoundError:
    pass

if not rss:
    print("nan,nan,nan,nan,nan")
else:
    baseline_rss = rss[0]
    baseline_system = system_used[0]
    peak_system = max(system_used)
    print(
        f"{baseline_rss:.3f},{max(rss):.3f},{baseline_system:.3f},"
        f"{peak_system:.3f},{peak_system - baseline_system:.3f}"
    )
PY
}

run_profile() {
  local policy="$1"
  local budget="$2"
  local bank_device="$3"
  local run_name

  if [ -n "$budget" ]; then
    run_name="worldmem_gpu_profile_${bank_device}_bank_${policy}_b${budget}_${FUTURE_SECONDS}s_n${NUM_VIDEOS}"
  else
    run_name="worldmem_gpu_profile_${bank_device}_bank_${policy}_${FUTURE_SECONDS}s_n${NUM_VIDEOS}"
  fi

  local output_dir="$PROFILE_ROOT/runs/$run_name"
  local trace_path="$PROFILE_ROOT/access_traces/$run_name.jsonl"
  local gpu_log="$PROFILE_ROOT/nvidia_smi/$run_name.csv"
  local host_log="$PROFILE_ROOT/host_memory/$run_name.csv"
  local run_log="$PROFILE_ROOT/logs/$run_name.log"

  echo "============================================================"
  echo "Profiling: $run_name"
  echo "Policy: $policy"
  echo "Budget: ${budget:-none}"
  echo "Memory bank device: $bank_device"
  echo "GPU log: $gpu_log"
  echo "Host-memory log: $host_log"
  echo "Trace: $trace_path"
  echo "============================================================"

  cleanup_sampler
  start_sampler "$gpu_log"

  local start_epoch
  start_epoch="$(date +%s)"
  (
    GPU="$GPU" \
    WORLDMEM_REPO_ROOT="$WORLDMEM_REPO_ROOT" \
    WORLDMEM_STORAGE_ROOT="$STORAGE_ROOT" \
    MEMORY_POLICY="$policy" \
    MEMORY_BUDGET="$budget" \
    MEMORY_BANK_DEVICE="$bank_device" \
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
    PROFILE_TIMING=true \
    PROFILE_RETRIEVAL_QUERIES=true \
    LOG_VIDEO="$LOG_VIDEO" \
    SAVE_LOCAL_PER_BATCH="$SAVE_LOCAL_PER_BATCH" \
    SAVE_GT_VIDEO=false \
    COMPUTE_EVAL_METRICS=false \
    STREAM_EVAL_METRICS=false \
    RESUME_PARTIAL=0 \
    SKIP_COMPLETED=0 \
    WANDB_MODE=disabled \
    bash "$SCRIPT_DIR/run_worldmem_memory_policy_smoke.sh"
  ) > >(tee "$run_log") 2>&1 &
  local run_pid="$!"
  start_host_sampler "$run_pid" "$host_log"

  set +e
  wait "$run_pid"
  local status="$?"
  set -e
  if [ -n "$HOST_SAMPLER_PID" ]; then
    wait "$HOST_SAMPLER_PID" >/dev/null 2>&1 || true
    HOST_SAMPLER_PID=""
  fi
  local end_epoch
  local wall_seconds
  end_epoch="$(date +%s)"
  wall_seconds=$((end_epoch - start_epoch))

  cleanup_sampler

  local nvidia_summary
  local torch_summary
  local timing_summary
  local bank_summary
  local retrieval_query_summary
  local host_summary
  nvidia_summary="$(summarize_nvidia_smi_log "$gpu_log")"
  torch_summary="$(summarize_trace_log "$trace_path")"
  timing_summary="$(summarize_timing_log "$trace_path")"
  bank_summary="$(summarize_bank_log "$trace_path")"
  retrieval_query_summary="$(summarize_retrieval_queries "$trace_path")"
  host_summary="$(summarize_host_log "$host_log")"
  IFS=',' read -r baseline_used peak_used net_peak_used peak_util <<< "$nvidia_summary"
  IFS=',' read -r peak_torch_allocated peak_torch_reserved <<< "$torch_summary"
  IFS=',' read -r \
    total_seconds \
    retrieval_seconds \
    sampling_seconds \
    memory_update_seconds \
    generation_chunks \
    initial_memory_update_seconds \
    generation_memory_update_seconds \
    descriptor_extraction_seconds \
    descriptor_calls \
    descriptor_frames_requested \
    memory_update_excluding_descriptor_seconds \
    decode_seconds <<< "$timing_summary"
  local memory_update_count
  local memory_update_ms_mean
  local descriptor_ms_per_call
  memory_update_count="$(python - "$generation_chunks" <<'PY'
import sys
import math
value = float(sys.argv[1])
print(0 if not math.isfinite(value) else int(value) + 1)
PY
)"
  memory_update_ms_mean="$(python - "$memory_update_seconds" "$memory_update_count" <<'PY'
import sys
import math
seconds = float(sys.argv[1])
count = int(sys.argv[2])
print(
    "nan"
    if not math.isfinite(seconds) or count <= 0
    else f"{1000.0 * seconds / count:.6f}"
)
PY
)"
  descriptor_ms_per_call="$(python - "$descriptor_extraction_seconds" "$descriptor_calls" <<'PY'
import sys
import math
seconds = float(sys.argv[1])
calls_value = float(sys.argv[2])
calls = int(calls_value) if math.isfinite(calls_value) else 0
print(
    "nan"
    if not math.isfinite(seconds) or calls == 0
    else f"{1000.0 * seconds / calls:.6f}"
)
PY
)"
  IFS=',' read -r \
    peak_archive_mib \
    peak_archive_frames \
    final_archive_mib \
    final_archive_frames \
    peak_history_mib \
    final_history_mib \
    peak_bank_mib \
    peak_bank_frames <<< "$bank_summary"
  IFS=',' read -r \
    retrieval_query_count \
    retrieval_ms_mean \
    retrieval_ms_median \
    retrieval_ms_p95 <<< "$retrieval_query_summary"
  IFS=',' read -r \
    baseline_process_tree_rss_mib \
    peak_process_tree_rss_mib \
    baseline_system_memory_used_mib \
    peak_system_memory_used_mib \
    net_peak_system_memory_used_mib <<< "$host_summary"

  local -a summary_row=(
    "$run_name"
    "$policy"
    "${budget:-}"
    "$bank_device"
    "$FUTURE_SECONDS"
    "$NUM_VIDEOS"
    "$status"
    "$wall_seconds"
    "$total_seconds"
    "$retrieval_seconds"
    "$retrieval_query_count"
    "$retrieval_ms_mean"
    "$retrieval_ms_median"
    "$retrieval_ms_p95"
    "$sampling_seconds"
    "$memory_update_seconds"
    "$memory_update_count"
    "$memory_update_ms_mean"
    "$initial_memory_update_seconds"
    "$generation_memory_update_seconds"
    "$descriptor_extraction_seconds"
    "$descriptor_calls"
    "$descriptor_frames_requested"
    "$descriptor_ms_per_call"
    "$memory_update_excluding_descriptor_seconds"
    "$decode_seconds"
    "$peak_archive_mib"
    "$peak_archive_frames"
    "$final_archive_mib"
    "$final_archive_frames"
    "$peak_history_mib"
    "$final_history_mib"
    "$peak_bank_mib"
    "$peak_bank_frames"
    "$baseline_process_tree_rss_mib"
    "$peak_process_tree_rss_mib"
    "$baseline_system_memory_used_mib"
    "$peak_system_memory_used_mib"
    "$net_peak_system_memory_used_mib"
    "$baseline_used"
    "$peak_used"
    "$net_peak_used"
    "$peak_util"
    "$peak_torch_allocated"
    "$peak_torch_reserved"
    "$output_dir"
    "$trace_path"
    "$gpu_log"
    "$host_log"
    "$run_log"
  )
  python - "$SUMMARY_CSV" "${summary_row[@]}" <<'PY'
import csv
import sys

path = sys.argv[1]
row = sys.argv[2:]
with open(path, newline="", encoding="utf-8") as handle:
    header = next(csv.reader(handle))
if len(row) != len(header):
    raise RuntimeError(
        f"Profile summary row has {len(row)} fields; expected {len(header)}"
    )
with open(path, "a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow(row)
PY

  echo
  echo "Current summary:"
  python - "$SUMMARY_CSV" <<'PY'
import pandas as pd
import sys

path = sys.argv[1]
df = pd.read_csv(path)
columns = [
    "run_name",
    "policy",
    "budget",
    "memory_bank_device",
    "status",
    "wall_seconds",
    "retrieval_ms_mean",
    "descriptor_extraction_seconds",
    "memory_update_seconds",
    "peak_archive_mib",
    "peak_archive_frames",
    "peak_bank_mib",
    "peak_bank_frames",
    "peak_process_tree_rss_mib",
    "peak_nvidia_smi_used_mib",
    "peak_torch_allocated_mib",
    "peak_torch_reserved_mib",
]
print(df[columns].to_string(index=False))
PY
  echo

  return "$status"
}

IFS=',' read -r -a BANK_DEVICE_ARRAY <<< "$MEMORY_BANK_DEVICES"
IFS=',' read -r -a MINE_POLICY_ARRAY <<< "$MINE_POLICIES"
IFS=',' read -r -a BUDGET_ARRAY <<< "$MINE_BUDGETS"

for bank_device in "${BANK_DEVICE_ARRAY[@]}"; do
  bank_device="${bank_device//[[:space:]]/}"
  [ -n "$bank_device" ] || continue

  if [ "$INCLUDE_UNBOUNDED" = "1" ] || [ "$INCLUDE_UNBOUNDED" = "true" ]; then
    run_profile "unbounded" "" "$bank_device"
  fi

  for mine_policy in "${MINE_POLICY_ARRAY[@]}"; do
    mine_policy="${mine_policy//[[:space:]]/}"
    [ -n "$mine_policy" ] || continue

    for budget in "${BUDGET_ARRAY[@]}"; do
      budget="${budget//[[:space:]]/}"
      [ -n "$budget" ] || continue
      run_profile "$mine_policy" "$budget" "$bank_device"
    done
  done
done

echo "Wrote summary: $SUMMARY_CSV"
