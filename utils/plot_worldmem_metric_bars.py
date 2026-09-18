#!/usr/bin/env python3
"""Create cleaner bar-based WorldMem metric figures from recorded aggregates."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worldmem_metric_bars_mpl")

from utils.plot_worldmem_complete_metric_sweep import BUDGETS, load_rows
from utils.plot_worldmem_metric_variations import relative_improvement

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("Unbounded", "FIFO", "MCE", "K-center", "Latent-RI", "Geometric Coverage")
PALETTE = {"Unbounded": "#414B55", "FIFO": "#CD806C", "MCE": "#A39AB4",
           "K-center": "#D4AB5C", "Latent-RI": "#5D94BD", "Geometric Coverage": "#208B68"}
NAMES = {"Geometric Coverage": "KEEPSAKE", "Latent-RI": "RI"}
VBENCH = (("vbench_subject", "Subject consistency"),
          ("vbench_background", "Background consistency"),
          ("vbench_aesthetic", "Aesthetic quality"),
          ("vbench_imaging", "Imaging quality"),
          ("vbench_motion", "Motion smoothness"),
          ("vbench_dynamic", "Dynamic degree"))


def name(policy):
    return NAMES.get(policy, policy)


def chrome(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#DCE1E5")
    ax.tick_params(axis="both", length=0, pad=8)
    ax.grid(axis="x", color="#EBEEF1", lw=0.7)
    ax.set_axisbelow(True)


def heading(fig, title, subtitle):
    fig.text(0.065, 0.965, title, fontsize=21, weight="bold", va="top")
    fig.text(0.065, 0.885, subtitle, fontsize=10.5, color="#606B75", va="top")


def fixed_rows(lookup):
    return [lookup[p, None if p == "Unbounded" else 32] for p in ORDER]


def horizontal(ax, group, key, title, precision, percent=False):
    import numpy as np
    vals = np.array([r[key] for r in group]) * (100 if percent else 1)
    ax.barh(range(6), vals, height=0.57, color=[PALETTE[r["policy"]] for r in group])
    ax.set_yticks(range(6), [name(r["policy"]) for r in group])
    ax.set_ylim(5.65, -0.65)
    ax.set_xlim(0, max(vals) * 1.23)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.025, i, f"{v:.{precision}f}", va="center", fontsize=11,
                weight="bold" if group[i]["policy"] == "Geometric Coverage" else "normal")
    ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=15)
    ax.set_xlabel("VBench score (%)" if percent else "Score", color="#606B75", fontsize=9)
    chrome(ax)


def build(rows):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    lookup = {(r["policy"], r["budget"]): r for r in rows}
    group = fixed_rows(lookup)
    figures = []

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.6))
    heading(fig, "Selective memory, stronger generation",
            "WorldMem / 60 seconds / 15 videos per configuration / all bounded methods at B32")
    horizontal(axes[0], group, "lpips", "LPIPS / lower is better", 3)
    horizontal(axes[1], group, "fvd", "FVD / lower is better", 0)
    ax = axes[2]
    dims = VBENCH[:3]
    for j, p in enumerate(ORDER[1:]):
        vals = [relative_improvement(lookup[p, 32][k], lookup["Unbounded", None][k], False)
                for k, _ in dims]
        ax.barh(np.arange(3) + (j - 2) * 0.135, vals, height=0.115, color=PALETTE[p])
    ax.set_yticks(range(3), ["Subject", "Background", "Aesthetic"])
    ax.set_ylim(2.6, -0.6)
    ax.axvline(0, color="#414B55", lw=1)
    ax.set_title("VBench / higher is better", loc="left", fontsize=13, weight="bold", pad=15)
    ax.set_xlabel("Relative difference from unbounded (%)", fontsize=9, color="#606B75")
    chrome(ax)
    fig.legend(handles=[Patch(color=PALETTE[p], label=name(p)) for p in ORDER[1:]],
               ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.035), frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.75, bottom=0.23, wspace=0.48)
    figures.append(("01_editorial_b32", fig))

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 6.5))
    heading(fig, "The memory budget is not the quality ceiling",
            "Complete bounded sweep / 15 videos per cell / dashed reference = unbounded")
    policies = ORDER[1:]
    for ax, key, title, scale in (
        (axes[0], "lpips", "LPIPS / lower is better", 1),
        (axes[1], "fvd", "FVD / lower is better", 1),
        (axes[2], "vbench_background", "Background / higher is better", 100),
    ):
        for j, b in enumerate(BUDGETS):
            values = [lookup[p, b][key] * scale for p in policies]
            ax.bar(np.arange(5) + (j - 1.5) * 0.19, values, width=0.17,
                   color=[PALETTE[p] for p in policies], alpha=(0.40, 0.60, 0.80, 1)[j],
                   edgecolor="white", linewidth=0.4)
        ax.axhline(lookup["Unbounded", None][key] * scale, color=PALETTE["Unbounded"], ls="--", lw=1.5)
        ax.set_xticks(range(5), [name(p) for p in policies], rotation=0, fontsize=10)
        ax.set_ylim(0, max(r[key] * scale for r in rows) * 1.14)
        ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=14)
        ax.set_ylabel("VBench score (%)" if scale == 100 else "Score", fontsize=9, color="#606B75")
        chrome(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="#EBEEF1", lw=0.7)
    fig.legend(handles=[Patch(facecolor="#414B55", alpha=a, label=f"B{b}")
                        for b, a in zip(BUDGETS, (0.40, 0.60, 0.80, 1))],
               loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=4, frameon=False)
    fig.text(0.5, 0.035, "Within each policy: 16, 32, 64, 128 frames from left to right. All bar axes start at zero.",
             ha="center", fontsize=9.5, color="#606B75")
    fig.subplots_adjust(left=0.065, right=0.98, top=0.75, bottom=0.25, wspace=0.28)
    figures.append(("02_grouped_budget_bars", fig))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    heading(fig, "Six views of generation quality",
            "Standard VBench / fixed B32 comparison / 15 videos per configuration / absolute scores on a 0-100 scale")
    for ax, (key, title) in zip(axes.flat, VBENCH):
        horizontal(ax, group, key, title, 1, percent=True)
        ax.set_xlim(0, 116)
        ax.set_xticks((0, 25, 50, 75, 100))
    fig.text(0.065, 0.025, "Dynamic degree measures activity, not fidelity. Recorded aggregates; no uncertainty estimates inferred.",
             fontsize=9.5, color="#606B75")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.78, bottom=0.10, wspace=0.42, hspace=0.46)
    figures.append(("03_vbench_horizontal_bars", fig))
    return figures


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "assets/results/worldmem_budget_sweep_60s_n15.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/plots/metric_bars")
    args = parser.parse_args()
    rows = load_rows(args.input_csv)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "#202A33", "axes.labelcolor": "#606B75",
                         "xtick.color": "#606B75", "ytick.color": "#202A33", "pdf.fonttype": 42})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem, fig in build(rows):
        for suffix in ("png", "pdf"):
            path = args.output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=180, facecolor="white")
            print(f"Wrote: {path}")
        paths.append(args.output_dir / f"{stem}.png")
        plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(16, 22), facecolor="#EEF1F3")
    for ax, path in zip(axes, paths):
        ax.imshow(plt.imread(path))
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.04)
    fig.savefig(args.output_dir / "00_bar_comparison.png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
