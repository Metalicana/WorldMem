#!/usr/bin/env python3
"""Render five alternative layouts from the recorded WorldMem metric grid."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worldmem_metric_variations_mpl")

from utils.plot_worldmem_complete_metric_sweep import (
    BUDGETS, METRICS, POLICIES, STYLES, load_rows,
)

NAMES = {"Geometric Coverage": "KEEPSAKE", "Latent-RI": "Latent-RI"}
COLORS = {p: s["color"] for p, s in STYLES.items()}
BASE = "#535B63"
ROOT = Path(__file__).resolve().parents[1]


def label(policy):
    return NAMES.get(policy, policy)


def relative_improvement(value, baseline, lower):
    return 100 * (baseline - value if lower else value - baseline) / baseline


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E8EB", linewidth=0.7)
    ax.set_axisbelow(True)


def header(fig, title, subtitle):
    fig.suptitle(title, x=0.055, y=0.98, ha="left", fontsize=18, weight="bold")
    fig.text(0.055, 0.89, subtitle, fontsize=10, color=BASE)


def legend(fig):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[p], marker=STYLES[p]["marker"],
                      lw=2.2, label=label(p)) for p in POLICIES]
    handles.append(Line2D([], [], color=BASE, ls="--", label="Unbounded"))
    fig.legend(handles=handles, loc="lower center", ncol=6,
               frameon=False, bbox_to_anchor=(0.5, 0.015), fontsize=9)


def sweep(ax, lookup, spec):
    key, title, direction, _ = spec
    for p in POLICIES:
        style = STYLES[p]
        ax.plot(range(4), [lookup[p, b][key] for b in BUDGETS],
                color=COLORS[p], marker=style["marker"],
                ls=style["linestyle"], lw=2.6 if p == "Geometric Coverage" else 1.7,
                markersize=6, markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(lookup["Unbounded", None][key], color=BASE, ls="--", lw=1.6)
    ax.set_xticks(range(4), BUDGETS)
    ax.set_xlabel("Memory budget (frames)")
    ax.set_title(title, loc="left", weight="bold", fontsize=11)
    ax.set_ylabel(direction, fontsize=9, color=BASE)
    ax.margins(y=0.16, x=0.08)
    clean(ax)


def make_figures(rows):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

    lookup = {(r["policy"], r["budget"]): r for r in rows}
    subtitle = "WorldMem | 60-second rollouts | 15 videos per configuration | recorded aggregates"
    figures = []

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.9))
    header(fig, "01 / Quality versus memory budget", subtitle)
    for ax, spec in zip(axes, (METRICS[0], METRICS[1], METRICS[3])):
        sweep(ax, lookup, spec)
    legend(fig)
    fig.subplots_adjust(left=0.065, right=0.98, top=0.79, bottom=0.23, wspace=0.30)
    figures.append(("01_budget_sweep", fig))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.1))
    header(fig, "02 / The complete metric dashboard", subtitle)
    for ax, spec in zip(axes.flat, METRICS):
        sweep(ax, lookup, spec)
    legend(fig)
    fig.subplots_adjust(left=0.065, right=0.98, top=0.82, bottom=0.14,
                        hspace=0.52, wspace=0.38)
    figures.append(("02_full_dashboard", fig))

    bounded = [lookup[p, b] for p in POLICIES for b in BUDGETS]
    data = np.array([[relative_improvement(r[s[0]], lookup["Unbounded", None][s[0]],
                                          s[2] == "lower is better")
                      for s in METRICS] for r in bounded])
    limit = max(abs(data.min()), abs(data.max()))
    fig, ax = plt.subplots(figsize=(13.8, 10.7))
    header(fig, "03 / Every configuration versus unbounded",
           "Signed relative difference (%) | green = favorable metric direction | white = unbounded")
    im = ax.imshow(data, cmap="BrBG", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                   aspect="auto")
    ax.set_xticks(range(8), ["LPIPS", "FVD", "Subject", "Background", "Motion",
                             "Dynamic", "Aesthetic", "Imaging"])
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=9)
    ax.set_yticks(range(20), [f"{label(r['policy'])}  /  B{r['budget']}" for r in bounded])
    for i in range(20):
        for j in range(8):
            ax.text(j, i, f"{data[i, j]:+.1f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(data[i, j]) > 0.65 * limit else "#20282C")
    for y in (3.5, 7.5, 11.5, 15.5):
        ax.axhline(y, color="white", lw=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
    cb.set_label("Relative difference from unbounded (%)")
    fig.text(0.055, 0.025, "15 videos/cell. Metric direction is not a composite quality score; dynamic degree measures activity.",
             fontsize=9, color=BASE)
    fig.subplots_adjust(left=0.24, right=0.94, top=0.83, bottom=0.075)
    figures.append(("03_relative_heatmap", fig))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    header(fig, "04 / A controlled comparison at B32",
           "All bounded methods use 32 frames | dots = absolute scores | dashed line = unbounded | 15 videos/cell")
    for ax, spec in zip(axes.flat, METRICS):
        key, title, direction, precision = spec
        baseline = lookup["Unbounded", None][key]
        ax.axvline(baseline, color=BASE, ls="--", lw=1.4)
        values = [lookup[p, 32][key] for p in POLICIES]
        lo, hi = min(values + [baseline]), max(values + [baseline])
        span = hi - lo or 0.01
        for i, (p, value) in enumerate(zip(POLICIES, values)):
            ax.plot([baseline, value], [i, i], color=COLORS[p], alpha=0.35, lw=2)
            ax.scatter(value, i, color=COLORS[p], marker=STYLES[p]["marker"], s=60, zorder=3)
            ax.text(hi + span * 0.12, i, f"{value:.{precision}f}", va="center", fontsize=9,
                    color=COLORS[p])
        ax.set_yticks(range(5), [label(p) for p in POLICIES])
        ax.set_ylim(4.65, -0.65)
        ax.set_xlim(lo - span * 0.16, hi + span * 0.50)
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.set_xlabel(direction, fontsize=9)
        clean(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#E5E8EB", lw=0.7)
        ax.tick_params(axis="y", length=0, labelsize=9)
    fig.subplots_adjust(left=0.10, right=0.975, top=0.81, bottom=0.09,
                        hspace=0.48, wspace=0.72)
    figures.append(("04_fixed_b32_dumbbells", fig))

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8))
    header(fig, "05 / Metric trade-offs across the entire budget sweep",
           "Color = policy | point area = budget | numbers = retained frames | no aggregate score")
    for ax, xkey, ykey, xtitle, ytitle in (
        (axes[0], "lpips", "fvd", "LPIPS (lower is better)", "FVD (lower is better)"),
        (axes[1], "vbench_aesthetic", "vbench_imaging", "Aesthetic quality (higher is better)",
         "Imaging quality (higher is better)"),
    ):
        for p in POLICIES:
            group = [lookup[p, b] for b in BUDGETS]
            for r in group:
                size = 45 + r["budget"] * 0.9
                ax.scatter(r[xkey], r[ykey], s=size, color=COLORS[p],
                           marker=STYLES[p]["marker"], edgecolor="white", linewidth=0.7, zorder=3)
                offset = {16: (-14, -12), 32: (5, 5), 64: (5, -12), 128: (-14, 7)}[r["budget"]]
                ax.annotate(str(r["budget"]), (r[xkey], r[ykey]), xytext=offset,
                            textcoords="offset points", fontsize=7.5, color=COLORS[p])
        base = lookup["Unbounded", None]
        ax.scatter(base[xkey], base[ykey], marker="X", s=110, color=BASE, zorder=4)
        ax.annotate("Unbounded", (base[xkey], base[ykey]), xytext=(7, -15),
                    textcoords="offset points", fontsize=9, color=BASE)
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.margins(x=0.18, y=0.16)
        clean(ax)
    handles = [Line2D([], [], color=COLORS[p], marker=STYLES[p]["marker"],
                      ls="none", label=label(p)) for p in POLICIES]
    handles.append(Line2D([], [], color=BASE, marker="X", ls="none", label="Unbounded"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.21, wspace=0.30)
    figures.append(("05_tradeoff_scatter", fig))
    return figures


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "assets/results/worldmem_budget_sweep_60s_n15.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/plots/metric_variations")
    args = parser.parse_args()
    rows = load_rows(args.input_csv)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": "#30383E", "axes.edgecolor": "#BBC2C8",
                         "xtick.color": BASE, "ytick.color": BASE,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem, fig in make_figures(rows):
        for suffix in ("png", "pdf"):
            path = args.output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=180, facecolor="white")
            print(f"Wrote: {path}")
        paths.append(args.output_dir / f"{stem}.png")
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(18, 16), facecolor="#EDF0F2")
    for ax in axes.flat:
        ax.axis("off")
    for ax, path in zip(axes.flat, paths):
        ax.imshow(plt.imread(path))
    axes.flat[-1].text(0.08, 0.84, "WorldMem / five figure treatments", fontsize=22, weight="bold", va="top")
    axes.flat[-1].text(0.08, 0.65, "01  Main-paper budget sweep\n02  Complete metric dashboard\n03  Relative-performance heatmap\n04  Fixed-B32 comparison\n05  Metric trade-offs", fontsize=16, linespacing=1.8, va="top")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.03, hspace=0.05)
    path = args.output_dir / "00_comparison_sheet.png"
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
