"""Compute common-source retention and selection gaps for WorldMem.

WorldMem retrieves eight distinct latent frames per generated frame.  This
analysis evaluates the corresponding frame identities with DINO features from
one fixed unbounded rollout and exact-index ground truth.  It does not score
policy-specific pixels and it does not modify generation.
"""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.audit_worldmem_retention_selection import (  # noqa: E402
    CONFIGS,
    audit_cache,
    choose_attempt,
    read_attempts,
    reconstruct_attempt,
    videos_by_batch,
)
from utils.export_worldmem_retrieval_deterioration import sha256_file  # noqa: E402


METRICS = (
    "full_oracle_distance",
    "bank_oracle_distance",
    "selected_distance",
    "retention_gap",
    "selection_gap",
    "total_gap",
)

DISPLAY = {
    "unbounded": "Unbounded",
    "fifo": "FIFO",
    "rarity_irreplaceability": "RI",
    "slam_covisibility": "KEEPSAKE",
    "kcenter_coreset": "K-center",
    "mce": "MCE",
}

COLORS = {
    "Unbounded": "#4A4A4A",
    "FIFO": "#D55E00",
    "RI": "#0072B2",
    "KEEPSAKE": "#2E8B57",
    "K-center": "#7A6FAF",
    "MCE": "#3195A5",
}


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer columns for empty table: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def best_k(ids, distances, k):
    """Return the deterministic best-k mean and IDs."""
    ids = np.asarray(ids, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)
    if ids.ndim != 1 or distances.shape != ids.shape:
        raise ValueError("IDs and distances must be matching vectors")
    if len(ids) < k:
        raise ValueError(f"Need at least {k} candidates; found {len(ids)}")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Candidate IDs must be distinct")
    if not np.isfinite(distances).all():
        raise ValueError("Candidate distances contain nonfinite values")
    order = np.lexsort((ids, distances))[:k]
    return float(distances[order].mean()), [int(value) for value in ids[order]]


def decompose_query(
    distance_vector,
    full_ids,
    bank_ids,
    selected_ids,
    k=8,
    tolerance=1e-6,
    full_oracle=None,
):
    """Compute the exact best-k telescoping decomposition for one query."""
    distance_vector = np.asarray(distance_vector, dtype=np.float64)
    full_ids = np.asarray(full_ids, dtype=np.int64)
    bank_ids = np.asarray(bank_ids, dtype=np.int64)
    selected_ids = np.asarray(selected_ids, dtype=np.int64)
    for name, ids in (("history", full_ids), ("bank", bank_ids), ("selection", selected_ids)):
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise ValueError(f"{name} IDs must be a distinct vector")
        if len(ids) and (ids.min() < 0 or ids.max() >= len(distance_vector)):
            raise ValueError(f"{name} IDs fall outside the distance vector")
    if len(selected_ids) != k:
        raise ValueError(f"Expected exactly {k} selected IDs; found {len(selected_ids)}")
    if not np.isin(bank_ids, full_ids).all():
        raise ValueError("Retained bank is not a subset of complete eligible history")
    if not np.isin(selected_ids, bank_ids).all():
        raise ValueError("Selected IDs are not a subset of the retained bank")

    if full_oracle is None:
        a, full_oracle_ids = best_k(full_ids, distance_vector[full_ids], k)
    else:
        a, full_oracle_ids = full_oracle
        full_oracle_ids = [int(value) for value in full_oracle_ids]
        if len(full_oracle_ids) != k or not np.isin(full_oracle_ids, full_ids).all():
            raise ValueError("Precomputed full oracle is incompatible with eligible history")
    if np.array_equal(bank_ids, full_ids):
        b, bank_oracle_ids = a, list(full_oracle_ids)
    else:
        b, bank_oracle_ids = best_k(bank_ids, distance_vector[bank_ids], k)
    c = float(distance_vector[selected_ids].mean())
    retention = b - a
    selection = c - b
    total = c - a
    if retention < -tolerance or selection < -tolerance:
        raise ValueError(
            f"Oracle ordering failed: a={a:.9f}, b={b:.9f}, c={c:.9f}"
        )
    if abs((retention + selection) - total) > tolerance:
        raise ValueError("Retention and selection gaps do not telescope")
    return {
        "full_oracle_distance": a,
        "bank_oracle_distance": b,
        "selected_distance": c,
        "retention_gap": max(0.0, retention) if retention > -tolerance else retention,
        "selection_gap": max(0.0, selection) if selection > -tolerance else selection,
        "total_gap": total,
        "full_oracle_ids": full_oracle_ids,
        "bank_oracle_ids": bank_oracle_ids,
    }


def mean_record(rows, subset):
    selected = rows if subset == "all" else [row for row in rows if row["generated_time_sec"] >= 45.0]
    if not selected:
        raise ValueError(f"No queries available for subset {subset}")
    result = {"subset": subset, "queries": len(selected)}
    for metric in METRICS:
        result[metric] = float(np.mean([row[metric] for row in selected]))
    result["selected_initial_context_count_mean"] = float(
        np.mean([row["selected_initial_context_count"] for row in selected])
    )
    result["eligible_count_mean"] = float(np.mean([row["eligible_count"] for row in selected]))
    result["retained_count_mean"] = float(np.mean([row["retained_count"] for row in selected]))
    return result


def bootstrap_interval(values, indices):
    values = np.asarray(values, dtype=np.float64)
    draws = values[indices].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_trajectories(trajectory_rows, bootstrap_samples, bootstrap_seed):
    grouped = defaultdict(list)
    for row in trajectory_rows:
        grouped[(row["run_name"], row["policy"], row["budget"], row["subset"])].append(row)
    sizes = {len(rows) for rows in grouped.values()}
    if len(sizes) != 1:
        raise ValueError(f"Configurations do not share one trajectory count: {sorted(sizes)}")
    n = sizes.pop()
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, n, size=(bootstrap_samples, n))
    summary = []
    for (run, policy, budget, subset), rows in grouped.items():
        rows.sort(key=lambda row: int(row["trajectory_id"]))
        record = {
            "run_name": run,
            "policy": policy,
            "policy_label": DISPLAY[policy],
            "budget": budget,
            "subset": subset,
            "trajectories": n,
            "queries": sum(int(row["queries"]) for row in rows),
            "k": 8,
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            low, high = bootstrap_interval(values, indices)
            record[metric] = float(np.mean(values))
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        for metric in ("selected_initial_context_count_mean", "eligible_count_mean", "retained_count_mean"):
            record[metric] = float(np.mean([float(row[metric]) for row in rows]))
        summary.append(record)
    return summary, indices


def paired_differences(trajectory_rows, bootstrap_indices):
    by_key = {
        (row["run_name"], row["subset"], int(row["trajectory_id"])): row
        for row in trajectory_rows
    }
    configs = {}
    for row in trajectory_rows:
        configs[(row["run_name"], row["subset"])] = row
    result = []
    for subset in ("all", "late"):
        baselines = [row for (run, period), row in configs.items() if period == subset and row["policy"] == "unbounded"]
        ours = [row for (run, period), row in configs.items() if period == subset and row["policy"] == "slam_covisibility" and row["budget"] == 32]
        references = [("Unbounded", baselines[0])] if baselines else []
        if ours:
            references.append(("KEEPSAKE B32", ours[0]))
        targets = [row for (run, period), row in configs.items() if period == subset]
        for reference_label, reference in references:
            for target in targets:
                if target["run_name"] == reference["run_name"]:
                    continue
                trajectory_ids = sorted(
                    trajectory_id for run, period, trajectory_id in by_key
                    if run == target["run_name"] and period == subset
                )
                for metric in ("retention_gap", "selection_gap", "total_gap"):
                    diffs = np.asarray([
                        float(by_key[(target["run_name"], subset, trajectory_id)][metric])
                        - float(by_key[(reference["run_name"], subset, trajectory_id)][metric])
                        for trajectory_id in trajectory_ids
                    ])
                    low, high = bootstrap_interval(diffs, bootstrap_indices)
                    result.append({
                        "subset": subset,
                        "reference": reference_label,
                        "reference_run": reference["run_name"],
                        "comparison_run": target["run_name"],
                        "comparison_policy": DISPLAY[target["policy"]],
                        "comparison_budget": target["budget"],
                        "metric": metric,
                        "paired_difference": float(diffs.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "trajectories": len(diffs),
                    })
    return result


def plot_tradeoff(summary, output_dir, subset):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = [row for row in summary if row["subset"] == subset]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)
    families = ["FIFO", "RI", "K-center", "MCE", "KEEPSAKE"]

    def draw(ax, labels, annotate_all=False):
        for label in labels:
            group = sorted(
                (row for row in rows if row["policy_label"] == label),
                key=lambda row: int(row["budget"]),
            )
            if not group:
                continue
            color = COLORS[label]
            xs = [row["retention_gap"] for row in group]
            ys = [row["selection_gap"] for row in group]
            ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.9, zorder=2)
            for row, x, y in zip(group, xs, ys):
                ax.errorbar(
                    x,
                    y,
                    xerr=[
                        [max(0.0, x - row["retention_gap_ci_low"])],
                        [max(0.0, row["retention_gap_ci_high"] - x)],
                    ],
                    yerr=[
                        [max(0.0, y - row["selection_gap_ci_low"])],
                        [max(0.0, row["selection_gap_ci_high"] - y)],
                    ],
                    fmt="none",
                    ecolor=color,
                    alpha=0.18,
                    linewidth=0.7,
                    zorder=1,
                )
                ax.scatter(x, y, s=82, color=color, edgecolor="white", linewidth=1.0, zorder=3)
                if annotate_all:
                    ax.annotate(
                        f"B{int(row['budget'])}", (x, y), xytext=(6, 5),
                        textcoords="offset points", fontsize=8, color=color, weight="bold"
                    )

    full, zoom = axes
    draw(full, families, annotate_all=False)
    unbounded = next((row for row in rows if row["policy"] == "unbounded"), None)
    if unbounded:
        full.scatter(
            unbounded["retention_gap"], unbounded["selection_gap"], s=115,
            color=COLORS["Unbounded"], edgecolor="white", linewidth=1.1, zorder=4,
        )
        full.annotate(
            "Unbounded", (unbounded["retention_gap"], unbounded["selection_gap"]),
            xytext=(7, 4), textcoords="offset points", fontsize=9, weight="bold",
            color=COLORS["Unbounded"],
        )
    full.set_title("(a) Full policy landscape", loc="left", weight="bold")

    draw(zoom, ["RI", "KEEPSAKE"], annotate_all=True)
    zoom.set_title("(b) RI versus KEEPSAKE", loc="left", weight="bold")
    zoom.text(
        0.98, 0.97,
        "WorldMem retrieves eight frames per query\nLines connect B16, B32, B64, and B128",
        transform=zoom.transAxes, ha="right", va="top", fontsize=8.5, color="#555555",
    )

    legend = [
        Line2D([], [], color=COLORS[label], marker="o", markersize=5, linewidth=1.7, label=label)
        for label in ["FIFO", "RI", "K-center", "MCE", "KEEPSAKE"]
        if any(row["policy_label"] == label for row in rows)
    ]
    if unbounded:
        legend.append(Line2D([], [], color=COLORS["Unbounded"], marker="o", linestyle="none", label="Unbounded"))
    full.legend(handles=legend, frameon=False, fontsize=8, ncol=2, loc="best")
    zoom.legend(handles=[item for item in legend if item.get_label() in {"RI", "KEEPSAKE"}], frameon=False, loc="best")
    for ax in axes:
        ax.set_xlabel("Retention gap (lower is better)")
        ax.grid(color="#DDE1E5", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    full.set_ylabel("Selection gap (lower is better)")
    suffix = "all" if subset == "all" else "late_45_60s"
    output = output_dir / f"worldmem_retention_selection_tradeoff_{suffix}"
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=260, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_b32_decomposition(summary, output_dir, subset):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["Unbounded", "FIFO", "MCE", "K-center", "RI", "KEEPSAKE"]
    rows = [
        row for row in summary
        if row["subset"] == subset and (row["policy"] == "unbounded" or row["budget"] == 32)
    ]
    rows.sort(key=lambda row: order.index(row["policy_label"]))
    if not rows:
        return
    y = np.arange(len(rows))
    retention = np.asarray([row["retention_gap"] for row in rows])
    selection = np.asarray([row["selection_gap"] for row in rows])
    total = retention + selection
    low = np.asarray([row["total_gap_ci_low"] for row in rows])
    high = np.asarray([row["total_gap_ci_high"] for row in rows])

    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    ax.barh(y, retention, color="#D89A45", label="Lost through eviction")
    ax.barh(y, selection, left=retention, color="#4C8F91", label="Retained but not selected")
    ax.errorbar(
        total, y, xerr=np.vstack([np.maximum(0.0, total - low), np.maximum(0.0, high - total)]), fmt="o",
        color="#222222", markersize=3.5, linewidth=0.8, capsize=2,
        label="Total gap, trajectory-bootstrap 95% CI",
    )
    ax.set_yticks(y, [row["policy_label"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("DINO distance gap (lower is better)")
    title = "WorldMem retention and selection at B32"
    if subset == "late":
        title += ": 45-60 seconds"
    ax.set_title(title, loc="left", weight="bold")
    ax.grid(axis="x", color="#E2E5E8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    suffix = "all" if subset == "all" else "late_45_60s"
    output = output_dir / f"worldmem_retention_selection_b32_{suffix}"
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=260, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def markdown_report(summary, cohort, output):
    lines = [
        "# WorldMem Common-Source Retention and Selection Gaps",
        "",
        "WorldMem uses the mean distance of eight distinct selected memories. "
        "Retention is `bank oracle - full-history oracle`; selection is "
        "`actual selection - bank oracle`. Lower is better.",
        "",
        f"Matched trajectories: `{','.join(map(str, cohort))}` (`n={len(cohort)}`).",
        "",
        "| subset | policy | budget | retention | selection | total | trajectories |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        summary,
        key=lambda item: (
            0 if item["subset"] == "all" else 1,
            list(DISPLAY.values()).index(item["policy_label"]),
            -1 if item["budget"] == "" else int(item["budget"]),
        ),
    ):
        budget = "--" if row["budget"] == "" else row["budget"]
        lines.append(
            f"| {row['subset']} | {row['policy_label']} | {budget} | "
            f"{row['retention_gap']:.4f} | {row['selection_gap']:.4f} | "
            f"{row['total_gap']:.4f} | {row['trajectories']} |"
        )
    lines.extend([
        "",
        "The common-source RGB frames are saved-MP4 proxies for WorldMem's stored "
        "latents. These gaps diagnose target-appearance retention and selection; "
        "they are not causal estimates of downstream generation quality.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-run", default="worldmem_memquality_unbounded_60s_n15_seed101")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unbounded-trace-root", type=Path)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--query-stride", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--allow-incomplete-cohort", action="store_true")
    args = parser.parse_args()
    if args.query_stride < 1:
        parser.error("--query-stride must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory to preserve previous evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    tables = args.output / "tables"
    figures.mkdir()
    tables.mkdir()

    batches = list(range(args.expected_videos))
    source_dir = args.source_root / args.source_run
    source_videos = videos_by_batch(source_dir)
    cache_records = {}
    cache_errors = {}
    for batch in batches:
        try:
            if len(source_videos[batch]) != 1:
                raise ValueError("Expected one unambiguous common-source video")
            cache_records[batch] = audit_cache(
                args.cache_dir / f"batch_{batch:05d}.npz",
                source_videos[batch][0],
                args.source_run,
                batch,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            cache_errors[batch] = str(exc)

    reconstructed = {}
    coverage = []
    trace_hashes = {}
    for run, policy, budget in CONFIGS:
        directory = (
            args.unbounded_trace_root
            if policy == "unbounded" and args.unbounded_trace_root is not None
            else args.suite_root / run
        )
        traces = sorted((directory / "access_traces").glob("*.jsonl"))
        parse_error = ""
        try:
            attempts = read_attempts(traces)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            attempts = {}
            parse_error = str(exc)
        for path in traces:
            trace_hashes[str(path)] = sha256_file(path)
        for batch in batches:
            row = {
                "run_name": run,
                "policy": policy,
                "budget": "" if budget is None else budget,
                "trajectory_id": batch,
                "trace_root": str(directory),
                "trace_status": "invalid",
                "cache_status": "verified" if batch in cache_records else "invalid",
                "queries": 0,
                "completion_status": "unverified",
                "generation_seed": "",
                "error": parse_error or cache_errors.get(batch, ""),
            }
            try:
                if parse_error:
                    raise ValueError(parse_error)
                if batch not in cache_records:
                    raise ValueError(cache_errors.get(batch, "Common-source cache unavailable"))
                attempt, completion = choose_attempt(attempts.get(batch, []))
                snapshots, selected, _ = reconstruct_attempt(attempt, policy, budget)
                reconstructed[(run, batch)] = (snapshots, selected, attempt)
                row.update(
                    trace_status="valid",
                    queries=len(selected),
                    completion_status=completion,
                    generation_seed=attempt["metadata"].get("generation_seed", ""),
                    error="",
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row["error"] = str(exc)
            coverage.append(row)
    write_csv(tables / "coverage.csv", coverage)

    valid_by_run = {
        run: {
            int(row["trajectory_id"])
            for row in coverage
            if row["run_name"] == run and row["trace_status"] == "valid" and row["cache_status"] == "verified"
        }
        for run, _, _ in CONFIGS
    }
    cohort = sorted(set(batches).intersection(*(valid_by_run[run] for run, _, _ in CONFIGS)))
    if cohort != batches and not args.allow_incomplete_cohort:
        missing = {
            run: sorted(set(batches) - valid)
            for run, valid in valid_by_run.items()
            if valid != set(batches)
        }
        raise RuntimeError(
            "Strict matched cohort is incomplete. See tables/coverage.csv. "
            f"Missing trajectories by run: {json.dumps(missing, sort_keys=True)}"
        )
    if not cohort:
        raise RuntimeError("No trajectory is valid across every requested configuration")
    print(
        "WARNING: legacy traces do not establish post-retry dataset paths or historical "
        "checkpoint revisions; provenance.json records this limitation.",
        file=sys.stderr,
        flush=True,
    )

    distances = {}
    full_oracles = {}
    for batch in cohort:
        with np.load(args.cache_dir / f"batch_{batch:05d}.npz", allow_pickle=False) as cached:
            generated = np.asarray(cached["generated"], dtype=np.float32)
            ground_truth = np.asarray(cached["ground_truth"], dtype=np.float32)
        dots = generated @ ground_truth[600:1200].T
        distances[batch] = 1.0 - np.clip(dots, -1.0, 1.0)
        for target in range(600, 1200, args.query_stride):
            full_ids = np.arange(target, dtype=np.int64)
            full_oracles[(batch, target)] = best_k(
                full_ids, distances[batch][:target, target - 600], 8
            )

    query_fields = [
        "system", "run_name", "policy", "budget", "trajectory_id", "query_id",
        "target_generated_frame", "target_dataset_local_frame", "generated_time_sec",
        "k", "eligible_count", "retained_count", *METRICS,
        "selected_initial_context_count", "source_id",
    ]
    trajectory_rows = []
    query_path = tables / "query_gaps.csv"
    sidecar_path = tables / "query_identities.jsonl"
    with query_path.open("w", newline="", encoding="utf-8") as query_handle, sidecar_path.open("w", encoding="utf-8") as sidecar:
        writer = csv.DictWriter(query_handle, fieldnames=query_fields)
        writer.writeheader()
        for run, policy, budget in CONFIGS:
            print(f"[gaps] {run}", flush=True)
            for batch in cohort:
                snapshots, selected, attempt = reconstructed[(run, batch)]
                query_rows = []
                for target in range(600, 1200, args.query_stride):
                    bank_ids = snapshots[target]
                    selected_ids = selected[target]
                    result = decompose_query(
                        distances[batch][:, target - 600],
                        np.arange(target, dtype=np.int64),
                        bank_ids,
                        selected_ids,
                        k=8,
                        full_oracle=full_oracles[(batch, target)],
                    )
                    row = {
                        "system": "WorldMem",
                        "run_name": run,
                        "policy": policy,
                        "budget": "" if budget is None else budget,
                        "trajectory_id": batch,
                        "query_id": target,
                        "target_generated_frame": target - 600,
                        "target_dataset_local_frame": target,
                        "generated_time_sec": (target - 600) / 10.0,
                        "k": 8,
                        "eligible_count": target,
                        "retained_count": len(bank_ids),
                        **{metric: result[metric] for metric in METRICS},
                        "selected_initial_context_count": sum(value < 600 for value in selected_ids),
                        "source_id": args.source_run,
                    }
                    writer.writerow(row)
                    query_rows.append(row)
                    sidecar.write(json.dumps({
                        "run_name": run,
                        "trajectory_id": batch,
                        "query_id": target,
                        "selected_ids": selected_ids,
                        "full_oracle_ids": result["full_oracle_ids"],
                        "bank_oracle_ids": result["bank_oracle_ids"],
                    }) + "\n")
                for subset in ("all", "late"):
                    aggregate = mean_record(query_rows, subset)
                    trajectory_rows.append({
                        "run_name": run,
                        "policy": policy,
                        "policy_label": DISPLAY[policy],
                        "budget": "" if budget is None else budget,
                        "trajectory_id": batch,
                        **aggregate,
                    })
    write_csv(tables / "trajectory_gaps.csv", trajectory_rows)
    summary, bootstrap_indices = summarize_trajectories(
        trajectory_rows, args.bootstrap_samples, args.bootstrap_seed
    )
    write_csv(tables / "summary.csv", summary)
    write_csv(tables / "paired_differences.csv", paired_differences(trajectory_rows, bootstrap_indices))
    markdown_report(summary, cohort, args.output / "report.md")

    for subset in ("all", "late"):
        plot_tradeoff(summary, figures, subset)
        plot_b32_decomposition(summary, figures, subset)

    provenance = {
        "analysis": "WorldMem common-source best-eight retention/selection decomposition",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "cohort": cohort,
        "common_source_run": args.source_run,
        "common_source_cache_records": cache_records,
        "trace_hashes": trace_hashes,
        "analysis_sha256": sha256_file(Path(__file__)),
        "definitions": {
            "distance": "1 - cosine(DINOv2-Base common-source frame, exact-index GT target)",
            "full_oracle": "mean of eight smallest distances in complete eligible history",
            "bank_oracle": "mean of eight smallest distances in retained bank",
            "selected": "mean distance of the eight IDs actually supplied to WorldMem",
            "retention_gap": "bank_oracle - full_oracle",
            "selection_gap": "selected - bank_oracle",
        },
        "aggregation": "query means within trajectory, then equal trajectory weighting",
        "bootstrap": {
            "unit": "trajectory",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "interval": "percentile 2.5/97.5",
        },
        "limitations": [
            "Common-source generated RGB comes from saved MP4 and is a proxy for the stored latent frame.",
            "Ground truth is used only for offline diagnosis.",
            "Best-eight independent appearance distance does not model multi-view complementarity.",
            "Policy banks and selections come from each policy's closed-loop trace, not an offline replay on one generated stream.",
            "Dataset source paths after loader retries and historical checkpoint revisions are not logged in legacy traces.",
        ],
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Matched trajectories: {len(cohort)}/{args.expected_videos}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
