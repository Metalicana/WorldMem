import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


QUALITY_METRICS = ("psnr", "ssim", "lpips")


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] malformed trace row {path}:{line_number}: {exc}")
    return rows


def write_csv(path, rows):
    rows = list(rows)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(values):
    return np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )


def mean(values):
    values = finite(values)
    return float(values.mean()) if values.size else None


def percentile(values, q):
    values = finite(values)
    return float(np.percentile(values, q)) if values.size else None


def worst_decile_mean(values, higher_is_better):
    values = np.sort(finite(values))
    if not values.size:
        return None
    count = max(1, int(np.ceil(0.1 * len(values))))
    selected = values[:count] if higher_is_better else values[-count:]
    return float(selected.mean())


def bootstrap_mean_interval(values, samples=4000, seed=0):
    values = finite(values)
    if not values.size:
        return None, None, None
    if values.size == 1:
        value = float(values[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    sampled = values[
        rng.integers(0, len(values), size=(int(samples), len(values)))
    ].mean(1)
    return (
        float(values.mean()),
        float(np.percentile(sampled, 2.5)),
        float(np.percentile(sampled, 97.5)),
    )


def correlation(left, right, rank=False):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) < 3 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    if rank:
        left = rank_values(left)
        right = rank_values(right)
    return float(np.corrcoef(left, right)[0, 1])


def rank_values(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    if np.any(counts > 1):
        for group in np.flatnonzero(counts > 1):
            mask = inverse == group
            ranks[mask] = ranks[mask].mean()
    return ranks


def discover_runs(output_root, runs):
    root = Path(output_root)
    if runs:
        names = [value.strip() for value in runs.split(",") if value.strip()]
        return [(name, root / name) for name in names]
    return [
        (path.name, path)
        for path in sorted(root.glob("worldmem_memquality_*"))
        if path.is_dir()
    ]


def trace_path(run_dir):
    paths = sorted((Path(run_dir) / "access_traces").glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No access trace under {run_dir}")
    return paths[0]


def run_label(row):
    policy = row.get("memory_policy", "unknown")
    budget = row.get("memory_budget")
    if budget not in {None, "", "None"}:
        return f"{policy} b{int(budget)}"
    return policy


def collect_rows(args, runs):
    retrieved = []
    following = []
    metadata = {}
    for run_name, run_dir in runs:
        rows = read_jsonl(trace_path(run_dir))
        starts = [row for row in rows if row.get("event") == "memory_run_start"]
        for row in starts:
            batch = int(row.get("global_batch_idx", row.get("batch_idx", 0)))
            metadata[(run_name, batch)] = {"run_name": run_name, **row}
        for row in rows:
            event = row.get("event")
            if event not in {
                "retrieved_memory_quality",
                "following_chunk_frame_quality",
            }:
                continue
            item = {"run_name": run_name, **row}
            item["global_batch_idx"] = int(
                item.get("global_batch_idx", item.get("batch_idx", 0))
            )
            item["horizon_sec"] = (
                int(item["target_frame"]) - args.context_frames
            ) / args.fps
            item["late_window"] = bool(item["horizon_sec"] >= args.late_start_sec)
            item["policy_label"] = run_label(item)
            if event == "retrieved_memory_quality":
                retrieved.append(item)
            else:
                following.append(item)
    return retrieved, following, metadata


def group_by(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return groups


def build_chunk_rows(retrieved, following):
    keys = ("run_name", "global_batch_idx", "target_frame")
    retrieval_groups = group_by(retrieved, keys)
    following_groups = group_by(following, keys)
    chunks = []
    for key, items in retrieval_groups.items():
        base = items[0]
        generated = [row for row in items if not row["source_is_initial_context"]]
        output = following_groups.get(key, [])
        row = {
            "run_name": key[0],
            "global_batch_idx": int(key[1]),
            "target_frame": int(key[2]),
            "target_horizon": int(base["target_horizon"]),
            "horizon_sec": float(base["horizon_sec"]),
            "late_window": bool(base["late_window"]),
            "memory_policy": base.get("memory_policy"),
            "memory_budget": base.get("memory_budget"),
            "policy_label": base["policy_label"],
            "retrieved_items": len(items),
            "retrieved_generated_items": len(generated),
            "retrieved_generated_fraction": len(generated) / max(len(items), 1),
            "retrieved_positive_overlap_fraction": mean(
                [float((item.get("selected_overlap") or 0) > 0) for item in items]
            ),
            "selected_memory_frames": [
                int(item["selected_memory_frame"])
                for item in sorted(items, key=lambda value: value["context_slot"])
            ],
            "generation_seed": base.get("generation_seed"),
        }
        for metric in QUALITY_METRICS:
            source_key = f"decoded_memory_{metric}"
            row[f"retrieved_{metric}_mean"] = mean(
                [item.get(source_key) for item in items]
            )
            row[f"retrieved_{metric}_median"] = percentile(
                [item.get(source_key) for item in items], 50
            )
            row[f"retrieved_generated_{metric}_mean"] = mean(
                [item.get(source_key) for item in generated]
            )
            row[f"following_chunk_{metric}_mean"] = mean(
                [item.get(f"following_chunk_{metric}") for item in output]
            )
        chunks.append(row)
    return chunks


def summarize_runs(retrieved, following):
    summaries = []
    run_groups = group_by(retrieved, ("run_name",))
    following_groups = group_by(following, ("run_name",))
    for (run_name,), items in run_groups.items():
        late = [row for row in items if row["late_window"]]
        generated = [row for row in items if not row["source_is_initial_context"]]
        late_generated = [
            row for row in late if not row["source_is_initial_context"]
        ]
        output = following_groups.get((run_name,), [])
        late_output = [row for row in output if row["late_window"]]
        row = {
            "run_name": run_name,
            "policy_label": items[0]["policy_label"],
            "memory_policy": items[0].get("memory_policy"),
            "memory_budget": items[0].get("memory_budget"),
            "trajectories": len({item["global_batch_idx"] for item in items}),
            "retrieval_exposures": len(items),
            "late_retrieval_exposures": len(late),
            "generated_reference_fraction": mean(
                [float(not item["source_is_initial_context"]) for item in items]
            ),
            "late_generated_reference_fraction": mean(
                [float(not item["source_is_initial_context"]) for item in late]
            ),
        }
        for metric in QUALITY_METRICS:
            key = f"decoded_memory_{metric}"
            higher_is_better = metric in {"psnr", "ssim"}
            row[f"retrieved_{metric}_mean"] = mean([item.get(key) for item in items])
            row[f"retrieved_{metric}_worst_decile_mean"] = worst_decile_mean(
                [item.get(key) for item in items], higher_is_better
            )
            row[f"retrieved_generated_{metric}_mean"] = mean(
                [item.get(key) for item in generated]
            )
            row[f"late_retrieved_{metric}_mean"] = mean(
                [item.get(key) for item in late]
            )
            row[f"late_retrieved_{metric}_worst_decile_mean"] = worst_decile_mean(
                [item.get(key) for item in late], higher_is_better
            )
            row[f"late_retrieved_generated_{metric}_mean"] = mean(
                [item.get(key) for item in late_generated]
            )
            output_key = f"following_chunk_{metric}"
            row[f"following_chunk_{metric}_mean"] = mean(
                [item.get(output_key) for item in output]
            )
            row[f"late_following_chunk_{metric}_mean"] = mean(
                [item.get(output_key) for item in late_output]
            )
        summaries.append(row)
    return summaries


def trajectory_rows(retrieved, following):
    rows = []
    retrieval_groups = group_by(retrieved, ("run_name", "global_batch_idx"))
    following_groups = group_by(following, ("run_name", "global_batch_idx"))
    for key, items in retrieval_groups.items():
        late = [item for item in items if item["late_window"]]
        output = following_groups.get(key, [])
        late_output = [item for item in output if item["late_window"]]
        row = {
            "run_name": key[0],
            "global_batch_idx": int(key[1]),
            "policy_label": items[0]["policy_label"],
        }
        for metric in QUALITY_METRICS:
            for window, retrieval_rows, output_rows in (
                ("all", items, output),
                ("late", late, late_output),
            ):
                row[f"{window}_retrieved_{metric}"] = mean(
                    [item.get(f"decoded_memory_{metric}") for item in retrieval_rows]
                )
                row[f"{window}_following_{metric}"] = mean(
                    [item.get(f"following_chunk_{metric}") for item in output_rows]
                )
        rows.append(row)
    return rows


def paired_differences(trajectory, baseline_policy, bootstrap_samples):
    by_run = group_by(trajectory, ("run_name",))
    baseline_runs = [
        run_name
        for (run_name,), rows in by_run.items()
        if rows and rows[0]["policy_label"] == baseline_policy
    ]
    if len(baseline_runs) != 1:
        raise ValueError(
            f"Expected one baseline labeled {baseline_policy!r}, found {baseline_runs}"
        )
    baseline_name = baseline_runs[0]
    baseline = {
        row["global_batch_idx"]: row for row in by_run[(baseline_name,)]
    }
    results = []
    for (run_name,), rows in by_run.items():
        if run_name == baseline_name:
            continue
        matched = [
            (row, baseline[row["global_batch_idx"]])
            for row in rows
            if row["global_batch_idx"] in baseline
        ]
        for window in ("all", "late"):
            for family in ("retrieved", "following"):
                for metric in QUALITY_METRICS:
                    key = f"{window}_{family}_{metric}"
                    differences = [
                        bounded.get(key) - unbounded.get(key)
                        for bounded, unbounded in matched
                        if bounded.get(key) is not None and unbounded.get(key) is not None
                    ]
                    estimate, low, high = bootstrap_mean_interval(
                        differences,
                        samples=bootstrap_samples,
                        seed=17,
                    )
                    results.append(
                        {
                            "run_name": run_name,
                            "policy_label": rows[0]["policy_label"],
                            "baseline_run": baseline_name,
                            "window": "45-60s" if window == "late" else "all",
                            "metric_family": family,
                            "metric": metric,
                            "bounded_minus_unbounded": estimate,
                            "ci95_low": low,
                            "ci95_high": high,
                            "matched_trajectories": len(differences),
                        }
                    )
    return results


def correlation_rows(chunks):
    rows = []
    for (run_name,), items in group_by(chunks, ("run_name",)).items():
        for window, selected in (
            ("all", items),
            ("45-60s", [row for row in items if row["late_window"]]),
        ):
            for metric in QUALITY_METRICS:
                left = [row.get(f"retrieved_{metric}_mean") for row in selected]
                right = [row.get(f"following_chunk_{metric}_mean") for row in selected]
                rows.append(
                    {
                        "run_name": run_name,
                        "policy_label": items[0]["policy_label"],
                        "window": window,
                        "metric": metric,
                        "pearson": correlation(left, right),
                        "spearman": correlation(left, right, rank=True),
                        "chunks": len(selected),
                    }
                )
    return rows


def replay_manifest(chunks, replay_policy, count):
    eligible = [
        row
        for row in chunks
        if row["policy_label"] == replay_policy
        and row["late_window"]
        and row["retrieved_generated_items"] > 0
        and row["retrieved_generated_lpips_mean"] is not None
    ]
    best_by_trajectory = {}
    for row in eligible:
        batch = row["global_batch_idx"]
        previous = best_by_trajectory.get(batch)
        if (
            previous is None
            or row["retrieved_generated_lpips_mean"]
            > previous["retrieved_generated_lpips_mean"]
        ):
            best_by_trajectory[batch] = row
    selected = sorted(
        best_by_trajectory.values(),
        key=lambda row: row["retrieved_generated_lpips_mean"],
        reverse=True,
    )[: int(count)]
    return [
        {
            "rank": rank,
            "run_name": row["run_name"],
            "policy_label": row["policy_label"],
            "global_batch_idx": row["global_batch_idx"],
            "target_frame": row["target_frame"],
            "target_horizon": row["target_horizon"],
            "horizon_sec": row["horizon_sec"],
            "generation_seed": row["generation_seed"],
            "selected_memory_frames": row["selected_memory_frames"],
            "retrieved_generated_lpips_mean": row[
                "retrieved_generated_lpips_mean"
            ],
            "retrieved_generated_psnr_mean": row[
                "retrieved_generated_psnr_mean"
            ],
            "retrieved_generated_ssim_mean": row[
                "retrieved_generated_ssim_mean"
            ],
        }
        for rank, row in enumerate(selected, start=1)
    ]


def make_plots(metrics_dir, chunks):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; skipping figures")
        return

    figure_dir = Path(metrics_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    grouped = group_by(chunks, ("policy_label",))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = (
        ("retrieved_psnr_mean", "Retrieved-memory PSNR", True),
        ("retrieved_ssim_mean", "Retrieved-memory SSIM", True),
        ("retrieved_lpips_mean", "Retrieved-memory LPIPS", False),
        ("following_chunk_lpips_mean", "Following-chunk LPIPS", False),
    )
    for axis, (key, title, _higher_is_better) in zip(axes.flat, panels):
        for (label,), rows in grouped.items():
            by_time = group_by(rows, ("horizon_sec",))
            times = sorted(value[0] for value in by_time)
            values = [mean([row.get(key) for row in by_time[(time,)]]) for time in times]
            axis.plot(times, values, label=label, linewidth=1.8)
        axis.axvspan(45, 60, color="0.9", zorder=0)
        axis.set_title(title)
        axis.set_xlabel("Generated horizon (seconds)")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "retrieved_and_following_quality_over_time.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    for (label,), rows in grouped.items():
        axis.scatter(
            [row["retrieved_lpips_mean"] for row in rows],
            [row["following_chunk_lpips_mean"] for row in rows],
            s=14,
            alpha=0.45,
            label=label,
        )
    axis.set_xlabel("Mean LPIPS of retrieved decoded memories")
    axis.set_ylabel("LPIPS of following generated chunk")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "retrieved_corruption_vs_following_chunk.png", dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate exact VAE-decoded WorldMem retrieval-quality traces."
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--runs", default="")
    parser.add_argument("--metrics_dir", type=Path, required=True)
    parser.add_argument("--context_frames", type=int, default=600)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--late_start_sec", type=float, default=45)
    parser.add_argument("--baseline_policy", default="unbounded")
    parser.add_argument("--replay_policy", default="unbounded")
    parser.add_argument("--replay_count", type=int, default=4)
    parser.add_argument("--bootstrap_samples", type=int, default=4000)
    return parser.parse_args()


def main():
    args = parse_args()
    runs = discover_runs(args.output_root, args.runs)
    if not runs:
        raise FileNotFoundError(f"No memory-quality runs under {args.output_root}")
    for run_name, run_dir in runs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing run directory: {run_dir}")
        print(f"[load] {run_name}")

    retrieved, following, _metadata = collect_rows(args, runs)
    if not retrieved or not following:
        raise RuntimeError(
            "No exact quality rows found. Runs must set "
            "TRACE_RETRIEVED_MEMORY_QUALITY=true."
        )
    chunks = build_chunk_rows(retrieved, following)
    summary = summarize_runs(retrieved, following)
    trajectory = trajectory_rows(retrieved, following)
    paired = paired_differences(
        trajectory,
        baseline_policy=args.baseline_policy,
        bootstrap_samples=args.bootstrap_samples,
    )
    correlations = correlation_rows(chunks)
    manifest = replay_manifest(chunks, args.replay_policy, args.replay_count)

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.metrics_dir / "retrieved_memory_items.csv", retrieved)
    write_csv(args.metrics_dir / "following_chunk_frames.csv", following)
    write_csv(args.metrics_dir / "chunk_quality.csv", chunks)
    write_csv(args.metrics_dir / "run_summary.csv", summary)
    write_csv(args.metrics_dir / "trajectory_summary.csv", trajectory)
    write_csv(args.metrics_dir / "paired_bounded_minus_unbounded.csv", paired)
    write_csv(args.metrics_dir / "retrieval_following_correlations.csv", correlations)
    with (args.metrics_dir / "gt_replay_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    make_plots(args.metrics_dir, chunks)

    print("\nRetrieved-memory quality summary")
    for row in summary:
        print(
            f"{row['policy_label']:<32} "
            f"late LPIPS={row['late_retrieved_lpips_mean']:.4f} "
            f"PSNR={row['late_retrieved_psnr_mean']:.2f} "
            f"SSIM={row['late_retrieved_ssim_mean']:.4f} "
            f"next LPIPS={row['late_following_chunk_lpips_mean']:.4f}"
        )
    print(f"Wrote: {args.metrics_dir}")
    print(f"Replay manifest: {args.metrics_dir / 'gt_replay_manifest.json'}")


if __name__ == "__main__":
    main()
