"""Summarize synchronized WorldMem retrieval-query latency by rollout window."""

import argparse
from collections import defaultdict
import csv
import glob
import json
from pathlib import Path

import numpy as np


def bootstrap_mean_ci(values, seed=0, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty sample")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def read_profile(path_pattern, expected_policy, expected_budget, expected_videos, frames):
    paths = [Path(path) for path in sorted(glob.glob(str(path_pattern)))]
    if not paths:
        raise FileNotFoundError(f"No trace matched: {path_pattern}")
    by_trajectory = {}
    for path in paths:
        active_batch = None
        active_rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                event = row.get("event")
                if event == "memory_run_start":
                    active_batch = int(row["global_batch_idx"])
                    active_rows = []
                    continue
                if event == "memory_run_end":
                    batch = int(row["global_batch_idx"])
                    if active_batch != batch:
                        raise ValueError(f"Mismatched run boundary in {path}:{line_number}")
                    if batch in by_trajectory:
                        raise ValueError(f"Duplicate completed trajectory {batch}")
                    by_trajectory[batch] = active_rows
                    active_batch = None
                    active_rows = []
                    continue
                if event != "retrieval_query_profile":
                    continue
                if active_batch is None:
                    raise ValueError(f"Profile event outside run boundary in {path}:{line_number}")
                if row.get("memory_policy") != expected_policy:
                    raise ValueError(f"Unexpected policy in {path}:{line_number}")
                actual_budget = row.get("memory_budget")
                if expected_budget is None:
                    if actual_budget not in {None, ""}:
                        raise ValueError("Unbounded trace unexpectedly has a budget")
                elif int(actual_budget) != expected_budget:
                    raise ValueError(f"Unexpected budget in {path}:{line_number}")
                if not row.get("cuda_synchronized"):
                    raise ValueError("Latency trace is not CUDA synchronized")
                if row.get("timing_scope") != "generate_condition_indices":
                    raise ValueError("Unexpected timing scope")
                batch = int(row["global_batch_idx"])
                if batch != active_batch:
                    raise ValueError(f"Profile event has wrong batch in {path}:{line_number}")
                active_rows.append(row)

    if len(by_trajectory) != expected_videos:
        raise ValueError(
            f"Expected {expected_videos} trajectories for {expected_policy}; "
            f"found {len(by_trajectory)}"
        )
    expected = list(range(frames))
    for batch, rows in by_trajectory.items():
        rows.sort(key=lambda row: int(row["rollout_frame"]))
        observed = [int(row["rollout_frame"]) for row in rows]
        if observed != expected:
            raise ValueError(f"Incomplete or duplicate query profile for trajectory {batch}")
        if any(int(row["retrieved_memory_count"]) != 8 for row in rows):
            raise ValueError(f"Retrieved-memory count changed in trajectory {batch}")
    return by_trajectory


def summarize(label, policy, budget, trajectories, fps, windows, seed):
    output = []
    for window_index, (start_sec, end_sec) in enumerate(windows):
        start = round(start_sec * fps)
        end = round(end_sec * fps)
        trajectory_latency = []
        trajectory_candidates = []
        for rows in trajectories.values():
            selected = rows[start:end]
            trajectory_latency.append(
                float(np.mean([float(row["query_milliseconds"]) for row in selected]))
            )
            trajectory_candidates.append(
                float(np.mean([int(row["candidate_count"]) for row in selected]))
            )
        ci_low, ci_high = bootstrap_mean_ci(
            trajectory_latency, seed=seed + window_index
        )
        output.append(
            {
                "system": "WorldMem",
                "policy_label": label,
                "memory_policy": policy,
                "memory_budget": "" if budget is None else budget,
                "generated_window": f"{start_sec}-{end_sec}s",
                "trajectories": len(trajectories),
                "queries_per_trajectory": end - start,
                "candidate_count_trajectory_mean": float(
                    np.mean(trajectory_candidates)
                ),
                "query_time_ms_trajectory_mean": float(np.mean(trajectory_latency)),
                "query_time_ms_ci95_low": ci_low,
                "query_time_ms_ci95_high": ci_high,
                "retrieved_memories_per_query": 8,
            }
        )
    return output


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unbounded-trace", required=True)
    parser.add_argument("--keepsake-trace", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--duration-sec", type=int, default=60)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--keepsake-policy", default="slam_covisibility")
    parser.add_argument("--keepsake-budget", type=int, default=32)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    frames = round(args.duration_sec * args.fps)
    windows = [(0, 15), (45, 60)]
    if args.duration_sec != 60 or frames != 600:
        parser.error("The publication latency table is defined for 60s at 10 FPS")
    unbounded = read_profile(
        args.unbounded_trace, "unbounded", None, args.expected_videos, frames
    )
    keepsake = read_profile(
        args.keepsake_trace,
        args.keepsake_policy,
        args.keepsake_budget,
        args.expected_videos,
        frames,
    )
    rows = summarize(
        "Unbounded", "unbounded", None, unbounded, args.fps, windows, args.bootstrap_seed
    )
    rows += summarize(
        "Ours (B=32)",
        args.keepsake_policy,
        args.keepsake_budget,
        keepsake,
        args.fps,
        windows,
        args.bootstrap_seed + 100,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "worldmem_retrieval_latency.csv"
    write_csv(output, rows)
    for row in rows:
        print(
            f"{row['policy_label']:<14} {row['generated_window']:<7} "
            f"candidates={row['candidate_count_trajectory_mean']:.1f} "
            f"query_ms={row['query_time_ms_trajectory_mean']:.3f} "
            f"CI=[{row['query_time_ms_ci95_low']:.3f}, "
            f"{row['query_time_ms_ci95_high']:.3f}]"
        )
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
