#!/usr/bin/env python3
"""Plot WorldMem memory-policy prefix metrics from CECSL summary tables."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worldmem_mplconfig")

import matplotlib.pyplot as plt


DURATIONS = [10, 20, 30, 60]

METRICS = {
    "lpips": {
        "ylabel": "LPIPS (lower is better)",
        "title": "WorldMem 60s Prefix LPIPS, First 15 Videos",
        "output_stem": "worldmem_lpips_prefix_60s_n15",
        "values": {
            "Unbounded": [0.505903, 0.568854, 0.601760, 0.652269],
            "FIFO b16": [0.518241, 0.563879, 0.607389, 0.717445],
            "FIFO b32": [0.523885, 0.576564, 0.612585, 0.688773],
            "FIFO b64": [0.507510, 0.559912, 0.602527, 0.687605],
            "FIFO b128": [0.501106, 0.559536, 0.595753, 0.647241],
            "RI b16": [0.500104, 0.534284, 0.560150, 0.565720],
            "RI b32": [0.492754, 0.527723, 0.550394, 0.545953],
            "RI b64": [0.498620, 0.535419, 0.555312, 0.548573],
            "RI b128": [0.497253, 0.536124, 0.558436, 0.566730],
            "SLAM b16": [0.495760, 0.514547, 0.535107, 0.524506],
            "SLAM b32": [0.496085, 0.517781, 0.543519, 0.533678],
            "SLAM b64": [0.495189, 0.527610, 0.548124, 0.545439],
            "SLAM b128": [0.500675, 0.543008, 0.571432, 0.577360],
        },
    },
    "fvd": {
        "ylabel": "FVD (lower is better)",
        "title": "WorldMem 60s Prefix FVD, First 15 Videos",
        "output_stem": "worldmem_fvd_prefix_60s_n15",
        "values": {
            "Unbounded": [1294.640098, 1376.429092, 1699.577615, 3077.599804],
            "FIFO b16": [1327.136552, 1644.809272, 2087.115469, 4205.032292],
            "FIFO b32": [1432.370689, 2022.648811, 2373.869870, 3554.909072],
            "FIFO b64": [1290.118592, 1480.887390, 1958.380615, 3821.736563],
            "FIFO b128": [1278.412947, 1399.130980, 1665.320119, 2604.960196],
            "RI b16": [1159.996193, 1138.900076, 1201.590948, 1238.743749],
            "RI b32": [1137.079588, 1085.517834, 1089.521562, 1160.427844],
            "RI b64": [1140.950734, 1119.389292, 1117.093006, 1165.354452],
            "RI b128": [1238.575079, 1221.719983, 1174.987174, 1250.560853],
            "SLAM b16": [1175.592628, 1086.813921, 1110.516295, 1041.756572],
            "SLAM b32": [1179.953901, 1064.023359, 1048.372693, 1116.924792],
            "SLAM b64": [1155.893563, 1125.879012, 1123.141370, 1128.461550],
            "SLAM b128": [1254.551435, 1290.372697, 1372.669186, 1601.813935],
        },
    },
}

STYLE_BY_POLICY = {
    "Unbounded": {"color": "#111111", "linestyle": "-", "linewidth": 3.0, "marker": "o"},
    "FIFO": {"color": "#d55e00", "linestyle": ":", "linewidth": 1.7, "marker": "s"},
    "RI": {"color": "#0072b2", "linestyle": "-.", "linewidth": 1.9, "marker": "^"},
    "SLAM": {"color": "#009e73", "linestyle": "-", "linewidth": 2.1, "marker": "D"},
}

ALPHA_BY_BUDGET = {
    "b16": 1.0,
    "b32": 0.82,
    "b64": 0.66,
    "b128": 0.50,
}


def policy_for(label: str) -> str:
    if label == "Unbounded":
        return label
    return label.split()[0]


def budget_for(label: str) -> str | None:
    parts = label.split()
    return parts[1] if len(parts) > 1 else None


def plot_metric(metric_name: str, spec: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=False)

    for label, values in spec["values"].items():
        policy = policy_for(label)
        budget = budget_for(label)
        style = STYLE_BY_POLICY[policy].copy()
        style["alpha"] = ALPHA_BY_BUDGET.get(budget, 1.0)
        ax.plot(DURATIONS, values, label=label, markersize=5.5, **style)

    ax.set_title(spec["title"], fontsize=15, pad=12)
    ax.set_xlabel("Generated future duration (seconds)", fontsize=12)
    ax.set_ylabel(spec["ylabel"], fontsize=12)
    ax.set_xticks(DURATIONS)
    ax.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)

    if metric_name == "lpips":
        ax.set_ylim(0.48, 0.74)
    else:
        ax.set_ylim(900, 4400)

    ax.text(
        0.0,
        -0.18,
        "CECSL WorldMem memory-policy grid: 60s rollouts, first 15 videos per run. Lower is better.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        va="top",
    )

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Memory policy",
        borderaxespad=0.0,
    )
    legend._legend_box.align = "left"

    fig.subplots_adjust(right=0.76, bottom=0.18)

    for suffix in ("png", "pdf"):
        path = output_dir / f"{spec['output_stem']}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(path)

    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "assets" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, spec in METRICS.items():
        plot_metric(metric_name, spec, output_dir)


if __name__ == "__main__":
    main()
