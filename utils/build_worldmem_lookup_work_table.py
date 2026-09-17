"""Build a paper table for WorldMem's growing unbounded retrieval workload."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def load_queries(path, run_name, duration_sec, expected_videos, fps):
    by_trajectory = defaultdict(list)
    required = {
        "run_name",
        "row",
        "duration_sec",
        "rollout_frame",
        "eligible_candidate_count",
        "logged_candidate_count",
        "candidate_count_mismatch",
        "selected_set_size",
    }
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for item in reader:
            if item["run_name"] != run_name or int(item["duration_sec"]) != duration_sec:
                continue
            if int(item["candidate_count_mismatch"]) != 0:
                raise ValueError(
                    f"Candidate-count mismatch in trajectory {item['row']}, "
                    f"frame {item['rollout_frame']}"
                )
            eligible = int(item["eligible_candidate_count"])
            logged = int(item["logged_candidate_count"])
            if eligible != logged:
                raise ValueError("Eligible and logged candidate counts disagree")
            by_trajectory[int(item["row"])].append(
                {
                    "rollout_frame": int(item["rollout_frame"]),
                    "candidate_count": eligible,
                    "selected_set_size": int(item["selected_set_size"]),
                }
            )

    if len(by_trajectory) != expected_videos:
        raise ValueError(
            f"Expected {expected_videos} trajectories; found {len(by_trajectory)}"
        )
    expected_frames = list(range(round(duration_sec * fps)))
    selected_sizes = set()
    for trajectory, rows in by_trajectory.items():
        rows.sort(key=lambda row: row["rollout_frame"])
        frames = [row["rollout_frame"] for row in rows]
        if frames != expected_frames:
            raise ValueError(f"Incomplete rollout-frame coverage for trajectory {trajectory}")
        selected_sizes.update(row["selected_set_size"] for row in rows)
    if len(selected_sizes) != 1:
        raise ValueError(f"Selected memory-set size is not fixed: {sorted(selected_sizes)}")
    return by_trajectory, selected_sizes.pop()


def aggregate_windows(by_trajectory, selected_size, duration_sec, fps, window_sec, fov_samples):
    if duration_sec % window_sec:
        raise ValueError("duration-sec must be divisible by window-sec")
    rows = []
    for start_sec in range(0, duration_sec, window_sec):
        end_sec = start_sec + window_sec
        start_frame = round(start_sec * fps)
        end_frame = round(end_sec * fps)
        trajectory_means = []
        trajectory_totals = []
        all_counts = []
        for queries in by_trajectory.values():
            counts = [
                row["candidate_count"]
                for row in queries
                if start_frame <= row["rollout_frame"] < end_frame
            ]
            if len(counts) != end_frame - start_frame:
                raise ValueError(f"Incomplete query window {start_sec}-{end_sec}s")
            trajectory_means.append(float(np.mean(counts)))
            trajectory_totals.append(int(np.sum(counts)))
            all_counts.extend(counts)
        mean_candidates = float(np.mean(trajectory_means))
        rows.append(
            {
                "window": f"{start_sec}-{end_sec}s",
                "start_sec": start_sec,
                "end_sec": end_sec,
                "queries_per_trajectory": end_frame - start_frame,
                "mean_eligible_candidates_per_query": mean_candidates,
                "min_eligible_candidates": min(all_counts),
                "max_eligible_candidates": max(all_counts),
                "retrieved_memories_per_query": selected_size,
                "candidate_to_retrieved_ratio": mean_candidates / selected_size,
                "mean_candidate_fov_point_tests_per_query": mean_candidates * fov_samples,
                "mean_candidate_frame_evaluations_per_trajectory": float(
                    np.mean(trajectory_totals)
                ),
            }
        )
    baseline = rows[0]["mean_eligible_candidates_per_query"]
    for row in rows:
        row["lookup_growth_vs_first_window"] = (
            row["mean_eligible_candidates_per_query"] / baseline
        )
    return rows


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows):
    lines = [
        "| Generated window | Eligible memories / query | Retrieved memories | Candidate / retrieved | FOV point tests / query | Growth vs. first window |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {window} | {mean_eligible_candidates_per_query:.1f} "
            "({min_eligible_candidates}-{max_eligible_candidates}) | "
            "{retrieved_memories_per_query} | {candidate_to_retrieved_ratio:.1f}x | "
            "{tests:.3f}M | {lookup_growth_vs_first_window:.2f}x |".format(
                tests=row["mean_candidate_fov_point_tests_per_query"] / 1e6,
                **row,
            )
        )
    return "\n".join(lines) + "\n"


def latex(rows):
    body = []
    for row in rows:
        body.append(
            "{window} & {mean_eligible_candidates_per_query:.1f} "
            "({min_eligible_candidates}--{max_eligible_candidates}) & "
            "{retrieved_memories_per_query} & {candidate_to_retrieved_ratio:.1f}$\\times$ & "
            "{tests:.3f}M & {lookup_growth_vs_first_window:.2f}$\\times$ \\\\".format(
                tests=row["mean_candidate_fov_point_tests_per_query"] / 1e6,
                **row,
            )
        )
    return """\\begin{table}[t]
\\centering
\\caption{Unbounded WorldMem retrieval workload over a 60-second rollout. Eligible-memory counts are measured from actual logged reads; the generator always receives eight memories. Point-test counts follow the released retriever's 10,000-sample Monte Carlo FOV computation.}
\\label{tab:worldmem_lookup_work}
\\small
\\begin{tabular}{lrrrrr}
\\toprule
Window & Eligible/query & Retrieved & Cand./ret. & FOV tests/query & Growth \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-name", default="worldmem_memquality_unbounded_60s_n15_seed101"
    )
    parser.add_argument("--duration-sec", type=int, default=60)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--window-sec", type=int, default=15)
    parser.add_argument("--fov-samples", type=int, default=10000)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Missing query CSV: {args.input}")
    if args.fps <= 0 or args.fov_samples <= 0 or args.window_sec <= 0:
        parser.error("fps, fov-samples, and window-sec must be positive")
    trajectories, selected_size = load_queries(
        args.input,
        args.run_name,
        args.duration_sec,
        args.expected_videos,
        args.fps,
    )
    rows = aggregate_windows(
        trajectories,
        selected_size,
        args.duration_sec,
        args.fps,
        args.window_sec,
        args.fov_samples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "lookup_work.csv", rows)
    (args.output_dir / "lookup_work.md").write_text(markdown(rows), encoding="utf-8")
    (args.output_dir / "lookup_work.tex").write_text(latex(rows), encoding="utf-8")
    provenance = {
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "trajectories": len(trajectories),
        "queries": sum(len(rows) for rows in trajectories.values()),
        "selected_memories_per_query": selected_size,
        "interpretation": "Candidate-FOV point tests are an exact algorithmic workload proxy from logged candidate counts times the configured Monte Carlo sample count, not measured wall time or FLOPs.",
    }
    (args.output_dir / "lookup_work_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(markdown(rows), end="")
    print(f"Wrote: {args.output_dir / 'lookup_work.tex'}")


if __name__ == "__main__":
    main()
