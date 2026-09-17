#!/usr/bin/env python3
"""Plot the complete WorldMem LPIPS, FVD, and VBench budget sweep."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/worldmem_complete_sweep_mpl")


BUDGETS = (16, 32, 64, 128)
POLICIES = ("FIFO", "Latent-RI", "Geometric Coverage", "K-center", "MCE")
STYLES = {
    "FIFO": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
    "Latent-RI": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "Geometric Coverage": {
        "color": "#2A9D55",
        "marker": "^",
        "linestyle": "-",
    },
    "K-center": {"color": "#E69F00", "marker": "D", "linestyle": "-."},
    "MCE": {"color": "#8E63A9", "marker": "P", "linestyle": ":"},
}
UNBOUNDED_STYLE = {
    "color": "#454A50",
    "linestyle": (0, (4, 3)),
    "linewidth": 2.0,
}
METRICS = (
    ("lpips", "LPIPS", "lower is better", 3),
    ("fvd", "FVD", "lower is better", 0),
    ("vbench_subject", "Subject consistency", "higher is better", 3),
    ("vbench_background", "Background consistency", "higher is better", 3),
    ("vbench_motion", "Motion smoothness", "higher is better", 3),
    ("vbench_dynamic", "Dynamic degree", "higher is better", 3),
    ("vbench_aesthetic", "Aesthetic quality", "higher is better", 3),
    ("vbench_imaging", "Imaging quality", "higher is better", 3),
)
RADAR_SERIES = (
    ("Unbounded", None, "Unbounded"),
    ("Latent-RI", 16, "Latent-RI B16"),
    ("Geometric Coverage", 16, "Geometric Coverage B16"),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root
        / "assets"
        / "results"
        / "worldmem_budget_sweep_60s_n15.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "assets" / "plots",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Combined metric table not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    rows: list[dict[str, object]] = []
    for source in source_rows:
        row: dict[str, object] = dict(source)
        row["budget"] = int(source["budget"]) if source["budget"] else None
        row["videos_matched"] = int(source["videos_matched"])
        for key, _, _, _ in METRICS:
            value = float(source[key])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {key} for {source['run_name']}")
            row[key] = value
        rows.append(row)
    validate_grid(rows)
    return rows


def validate_grid(rows: list[dict[str, object]]) -> None:
    keys = {(row["policy"], row["budget"]) for row in rows}
    expected = {("Unbounded", None)} | {
        (policy, budget) for policy in POLICIES for budget in BUDGETS
    }
    if keys != expected:
        missing = sorted(expected - keys, key=str)
        extra = sorted(keys - expected, key=str)
        raise ValueError(f"Incomplete metric grid; missing={missing}, extra={extra}")
    if len(rows) != len(expected):
        raise ValueError("Combined metric table contains duplicate policy-budget rows")
    bad_counts = [row["run_name"] for row in rows if row["videos_matched"] != 15]
    if bad_counts:
        raise ValueError(f"Rows without exactly 15 matched videos: {bad_counts}")


def metric_values(rows: list[dict[str, object]], metric: str):
    return {
        (str(row["policy"]), row["budget"]): float(row[metric])
        for row in rows
    }


def draw_panel(ax, rows, metric, title, direction, _precision, panel_label=None):
    values = metric_values(rows, metric)
    for policy in POLICIES:
        ax.plot(
            BUDGETS,
            [values[(policy, budget)] for budget in BUDGETS],
            label=policy,
            linewidth=2.1,
            markersize=6.0,
            markeredgewidth=0.8,
            markeredgecolor="white",
            **STYLES[policy],
        )

    baseline = values[("Unbounded", None)]
    ax.axhline(
        baseline,
        label="Unbounded",
        **UNBOUNDED_STYLE,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS, [str(value) for value in BUDGETS])
    ax.set_xlabel("Memory budget (frames)")
    ax.set_ylabel(f"{title} ({direction})")
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", pad=7)
    ax.grid(axis="y", color="#D9DDE2", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.10)
    if panel_label:
        ax.text(
            -0.15,
            1.04,
            panel_label,
            transform=ax.transAxes,
            fontsize=11.5,
            fontweight="bold",
            va="bottom",
        )


def add_legend(fig, axes, columns=3, y=0.015):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=columns,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.5,
    )


def save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote: {path}")


def plot_metric_group(rows, specs, shape, figsize, title, stem, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(*shape, figsize=figsize)
    axes_list = list(axes.flat)
    for index, (ax, spec) in enumerate(zip(axes_list, specs)):
        draw_panel(ax, rows, *spec, panel_label=chr(ord("a") + index))
    for ax in axes_list[len(specs) :]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.105,
        "WorldMem 60-second rollouts; first 15 matched videos per policy-budget cell.",
        ha="center",
        fontsize=9.5,
        color="#4D535A",
    )
    add_legend(fig, axes, columns=3, y=0.015)
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.91,
        bottom=0.23,
        hspace=0.43,
        wspace=0.30,
    )
    save(fig, output_dir, stem)
    plt.close(fig)


def plot_vbench_radar(rows, output_dir):
    import matplotlib.pyplot as plt

    specs = METRICS[2:]
    labels = [title for _, title, _, _ in specs]
    angles = [2 * math.pi * index / len(labels) for index in range(len(labels))]
    closed_angles = angles + angles[:1]
    values_by_metric = {
        metric: metric_values(rows, metric) for metric, _, _, _ in specs
    }
    baseline = {
        metric: values[("Unbounded", None)]
        for metric, values in values_by_metric.items()
    }

    fig, ax = plt.subplots(figsize=(8.7, 7.4), subplot_kw={"polar": True})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles, labels)
    ax.tick_params(axis="x", pad=13, labelsize=10.5)
    ax.set_ylim(90, 133)
    ax.set_yticks((90, 100, 110, 120, 130))
    ax.set_yticklabels(("90", "100", "110", "120", "130"), color="#60666D")
    ax.set_rlabel_position(22)
    ax.grid(color="#D9DDE2", linewidth=0.8)
    ax.spines["polar"].set_color("#B9BEC5")

    for policy, budget, display in RADAR_SERIES:
        relative = [
            100
            * values_by_metric[metric][(policy, budget)]
            / baseline[metric]
            for metric, _, _, _ in specs
        ]
        closed = relative + relative[:1]
        if policy == "Unbounded":
            ax.plot(
                closed_angles,
                closed,
                label=display,
                color=UNBOUNDED_STYLE["color"],
                linestyle=UNBOUNDED_STYLE["linestyle"],
                linewidth=2.1,
                zorder=2,
            )
            continue
        style = STYLES[policy]
        ax.plot(
            closed_angles,
            closed,
            label=display,
            color=style["color"],
            linewidth=2.5,
            marker=style["marker"],
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.fill(closed_angles, closed, color=style["color"], alpha=0.075)

    ax.set_title(
        "VBench profile of the strongest bounded configurations",
        fontsize=15,
        fontweight="bold",
        pad=25,
    )
    fig.text(
        0.5,
        0.065,
        "Relative score: Unbounded = 100 on every dimension; radial axis begins at 90.",
        ha="center",
        fontsize=9.5,
        color="#4D535A",
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.23, 1.12),
        frameon=False,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.09, right=0.87, top=0.88, bottom=0.13)
    save(fig, output_dir, "worldmem_vbench_radar_highlights_60s_n15")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 9.0,
        }
    )

    plot_metric_group(
        rows,
        METRICS[:2],
        (1, 2),
        (12.4, 4.8),
        "Long-horizon generation quality across memory budgets",
        "worldmem_lpips_fvd_budget_sweep_60s_n15",
        args.output_dir,
    )
    plot_metric_group(
        rows,
        METRICS[2:],
        (2, 3),
        (14.4, 8.2),
        "VBench quality across memory budgets",
        "worldmem_vbench_budget_sweep_60s_n15",
        args.output_dir,
    )
    plot_metric_group(
        rows,
        METRICS,
        (2, 4),
        (17.2, 8.2),
        "Complete WorldMem memory-budget evaluation",
        "worldmem_complete_metric_budget_sweep_60s_n15",
        args.output_dir,
    )
    plot_vbench_radar(rows, args.output_dir)


if __name__ == "__main__":
    main()
