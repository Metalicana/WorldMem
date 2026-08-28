#!/usr/bin/env python3
"""Build paper-style WorldMem LPIPS/FVD memory-budget sweep figures."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/worldmem_budget_sweep_mpl")


BUDGETS = (16, 32, 64, 128)
POLICY_ORDER = (
    "FIFO",
    "Latent-RI",
    "Geometric Coverage",
    "K-center",
    "MCE",
)
POLICY_PATTERNS = (
    ("FIFO", re.compile(r"^worldmem_fifo_b(?P<budget>\d+)_")),
    (
        "Latent-RI",
        re.compile(r"^worldmem_rarity_irreplaceability_b(?P<budget>\d+)_"),
    ),
    (
        "Geometric Coverage",
        re.compile(r"^worldmem_slam_covisibility_b(?P<budget>\d+)_"),
    ),
    (
        "K-center",
        re.compile(r"^worldmem_kcenter_coreset_b(?P<budget>\d+)_"),
    ),
    ("MCE", re.compile(r"^worldmem_mce_b(?P<budget>\d+)_")),
)
POLICY_STYLE = {
    "FIFO": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
    "Latent-RI": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "Geometric Coverage": {
        "color": "#5B8C1A",
        "marker": "^",
        "linestyle": "-",
    },
    "K-center": {"color": "#E69F00", "marker": "D", "linestyle": "-."},
    "MCE": {"color": "#6F5AA8", "marker": "P", "linestyle": ":"},
}
UNBOUNDED_STYLE = {
    "color": "#555555",
    "linestyle": (0, (3, 3)),
    "linewidth": 1.8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lpips-summary", type=Path, required=True)
    parser.add_argument("--fvd-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Plot available cells instead of requiring the complete sweep.",
    )
    return parser.parse_args()


def _to_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def classify_run(run_name: str) -> tuple[str, int | None] | None:
    if run_name.startswith("worldmem_unbounded_"):
        return "Unbounded", None
    for policy, pattern in POLICY_PATTERNS:
        match = pattern.match(run_name)
        if match:
            return policy, int(match.group("budget"))
    return None


def load_metric_summary(
    path: Path,
    metric: str,
    duration: int,
    limit: int,
) -> tuple[dict[tuple[str, int | None], float], dict[tuple[str, int | None], str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metric summary not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    values: dict[tuple[str, int | None], float] = {}
    sources: dict[tuple[str, int | None], str] = {}
    for row in rows:
        if _to_int(row.get("duration_sec")) != duration:
            continue
        classified = classify_run(str(row.get("run_name", "")))
        if classified is None:
            continue
        policy, budget = classified
        if budget is not None and budget not in BUDGETS:
            continue

        videos = _to_int(row.get("videos"))
        completed = _to_int(row.get("completed_videos"))
        failed = _to_int(row.get("failed_videos"))
        if videos != limit or completed != limit or failed not in (None, 0):
            continue
        value = _to_float(row.get(metric))
        if value is None:
            continue

        key = (policy, budget)
        if key in values:
            raise RuntimeError(
                f"Duplicate matched {metric} rows for {policy} budget={budget}"
            )
        values[key] = value
        sources[key] = str(row["run_name"])
    return values, sources


def missing_cells(values: dict[tuple[str, int | None], float]) -> list[str]:
    expected = [("Unbounded", None)] + [
        (policy, budget) for policy in POLICY_ORDER for budget in BUDGETS
    ]
    return [
        policy if budget is None else f"{policy} B{budget}"
        for policy, budget in expected
        if (policy, budget) not in values
    ]


def write_table(
    path: Path,
    lpips: dict[tuple[str, int | None], float],
    fvd: dict[tuple[str, int | None], float],
    lpips_sources: dict[tuple[str, int | None], str],
    fvd_sources: dict[tuple[str, int | None], str],
    duration: int,
    limit: int,
) -> None:
    fieldnames = (
        "policy",
        "budget",
        "duration_sec",
        "videos_matched",
        "lpips",
        "fvd",
        "lpips_run_name",
        "fvd_run_name",
    )
    rows = []
    keys = [("Unbounded", None)] + [
        (policy, budget) for policy in POLICY_ORDER for budget in BUDGETS
    ]
    for key in keys:
        if key not in lpips and key not in fvd:
            continue
        policy, budget = key
        rows.append(
            {
                "policy": policy,
                "budget": "" if budget is None else budget,
                "duration_sec": duration,
                "videos_matched": limit,
                "lpips": lpips.get(key, "missing"),
                "fvd": fvd.get(key, "missing"),
                "lpips_run_name": lpips_sources.get(key, "missing"),
                "fvd_run_name": fvd_sources.get(key, "missing"),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_panel(ax, values, metric: str, title: str, ylabel: str) -> None:
    for policy in POLICY_ORDER:
        points = [
            (budget, values[(policy, budget)])
            for budget in BUDGETS
            if (policy, budget) in values
        ]
        if not points:
            continue
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            label=policy,
            linewidth=2.2,
            markersize=6.5,
            **POLICY_STYLE[policy],
        )

    unbounded = values.get(("Unbounded", None))
    if unbounded is not None:
        precision = 3 if metric == "lpips" else 1
        ax.axhline(
            unbounded,
            label=f"Unbounded ({unbounded:.{precision}f})",
            **UNBOUNDED_STYLE,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS, [str(budget) for budget in BUDGETS])
    ax.set_xlabel("Memory budget (frames)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#DADCE0", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote: {path}")
    print(f"Wrote: {path.with_suffix('.pdf')}")


def main() -> None:
    args = parse_args()
    lpips, lpips_sources = load_metric_summary(
        args.lpips_summary, "lpips", args.duration, args.limit
    )
    fvd, fvd_sources = load_metric_summary(
        args.fvd_summary, "fvd", args.duration, args.limit
    )

    problems = []
    for metric, values in (("LPIPS", lpips), ("FVD", fvd)):
        missing = missing_cells(values)
        if missing:
            problems.append(f"{metric}: {', '.join(missing)}")
    if problems and not args.allow_missing:
        raise RuntimeError(
            "Incomplete matched budget sweep. Missing cells:\n  "
            + "\n  ".join(problems)
        )
    for problem in problems:
        print(f"Warning: {problem}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"worldmem_budget_sweep_{args.duration}s_n{args.limit}"
    table_path = args.output_dir / f"{stem}.csv"
    write_table(
        table_path,
        lpips,
        fvd,
        lpips_sources,
        fvd_sources,
        args.duration,
        args.limit,
    )
    print(f"Wrote: {table_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)
    plot_panel(
        axes[0],
        lpips,
        "lpips",
        f"Perceptual error at {args.duration} seconds",
        "LPIPS (lower is better)",
    )
    plot_panel(
        axes[1],
        fvd,
        "fvd",
        f"Video distribution error at {args.duration} seconds",
        "FVD (lower is better)",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        f"WorldMem memory-budget sweep, first {args.limit} matched videos",
        fontweight="bold",
    )
    save_figure(fig, args.output_dir / f"{stem}.png")
    plt.close(fig)

    for metric, values, title, ylabel in (
        (
            "lpips",
            lpips,
            f"LPIPS vs. memory budget ({args.duration}s)",
            "LPIPS (lower is better)",
        ),
        (
            "fvd",
            fvd,
            f"FVD vs. memory budget ({args.duration}s)",
            "FVD (lower is better)",
        ),
    ):
        fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
        plot_panel(ax, values, metric, title, ylabel)
        ax.legend(loc="best", frameon=False)
        path = args.output_dir / (
            f"worldmem_{metric}_vs_budget_{args.duration}s_n{args.limit}.png"
        )
        save_figure(fig, path)
        plt.close(fig)


if __name__ == "__main__":
    main()
