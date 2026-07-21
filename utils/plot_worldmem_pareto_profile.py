#!/usr/bin/env python3
"""Plot WorldMem latency and memory-bank Pareto profiles.

By default this uses the CECSL 60s RI/unbounded anchor points recorded in the
runbook. Pass one or more --summary-csv files from
scripts/profile_worldmem_gpu_memory.sh to plot RI, SLAM, or any newer policy
without editing this script.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worldmem_mplconfig")

import matplotlib.pyplot as plt


DEFAULT_PROFILE_60S = [
    {
        "run_name": "worldmem_gpu_profile_cpu_bank_unbounded_60s_n1",
        "policy": "unbounded",
        "budget": "",
        "memory_bank_device": "cpu",
        "future_seconds": 60,
        "wall_seconds": 748,
        "retrieval_seconds": 44.010,
        "sampling_seconds": 680.023,
        "peak_bank_mib": 0.0,
        "peak_bank_frames": 0,
        "peak_nvidia_smi_used_mib": 10921.0,
        "peak_torch_allocated_mib": 9640.697,
    },
    {
        "run_name": "worldmem_gpu_profile_cpu_bank_rarity_irreplaceability_b32_60s_n1",
        "policy": "rarity_irreplaceability",
        "budget": 32,
        "memory_bank_device": "cpu",
        "future_seconds": 60,
        "wall_seconds": 716,
        "retrieval_seconds": 3.050,
        "sampling_seconds": 688.344,
        "peak_bank_mib": 0.0,
        "peak_bank_frames": 0,
        "peak_nvidia_smi_used_mib": 10915.0,
        "peak_torch_allocated_mib": 9640.697,
    },
    {
        "run_name": "worldmem_gpu_profile_gpu_bank_unbounded_60s_n1",
        "policy": "unbounded",
        "budget": "",
        "memory_bank_device": "gpu",
        "future_seconds": 60,
        "wall_seconds": 746,
        "retrieval_seconds": 44.440,
        "sampling_seconds": 678.447,
        "peak_bank_mib": 31.641,
        "peak_bank_frames": 1200,
        "peak_nvidia_smi_used_mib": 10953.0,
        "peak_torch_allocated_mib": 9673.181,
    },
    {
        "run_name": "worldmem_gpu_profile_gpu_bank_rarity_irreplaceability_b32_60s_n1",
        "policy": "rarity_irreplaceability",
        "budget": 32,
        "memory_bank_device": "gpu",
        "future_seconds": 60,
        "wall_seconds": 711,
        "retrieval_seconds": 2.705,
        "sampling_seconds": 685.963,
        "peak_bank_mib": 0.826,
        "peak_bank_frames": 32,
        "peak_nvidia_smi_used_mib": 10917.0,
        "peak_torch_allocated_mib": 9642.367,
    },
]

POLICY_LABELS = {
    "unbounded": "Unbounded",
    "fifo": "FIFO",
    "rarity_irreplaceability": "RI",
    "slam_covisibility": "SLAM",
    "kcenter_coreset": "K-center",
}

POLICY_COLORS = {
    "unbounded": "#111111",
    "fifo": "#d55e00",
    "rarity_irreplaceability": "#0072b2",
    "slam_covisibility": "#6b8e23",
    "kcenter_coreset": "#e69f00",
}

POLICY_ORDER = {
    "unbounded": 0,
    "rarity_irreplaceability": 1,
    "slam_covisibility": 2,
    "fifo": 3,
    "kcenter_coreset": 4,
}

BANK_MARKERS = {"cpu": "o", "gpu": "s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        type=Path,
        action="append",
        default=[],
        help="One or more profile summary.csv files from scripts/profile_worldmem_gpu_memory.sh.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--duration-sec",
        type=int,
        default=60,
        help="Keep rows for this future_seconds value and use it in titles/output filenames.",
    )
    parser.add_argument(
        "--output-stem",
        default="",
        help="Optional prefix for output filenames, useful when plotting a new profile.",
    )
    return parser.parse_args()


def parse_float(value, default=math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_int(value, default=0) -> int:
    parsed = parse_float(value)
    if math.isnan(parsed):
        return default
    return int(parsed)


def normalize_row(row: dict[str, object]) -> dict[str, object] | None:
    status = parse_int(row.get("status"), default=0)
    if status != 0:
        return None

    policy = str(row.get("policy") or row.get("memory_policy") or "").strip()
    bank = str(row.get("memory_bank_device") or row.get("bank") or "cpu").strip().lower()
    budget = row.get("budget") or row.get("memory_budget") or ""

    item = {
        "run_name": str(row.get("run_name") or ""),
        "policy": policy,
        "budget": "" if budget in (None, "", "nan") else int(float(budget)),
        "memory_bank_device": bank,
        "future_seconds": parse_int(row.get("future_seconds"), default=0),
        "wall_seconds": parse_float(row.get("wall_seconds")),
        "retrieval_seconds": parse_float(row.get("retrieval_seconds")),
        "sampling_seconds": parse_float(row.get("sampling_seconds")),
        "peak_bank_mib": parse_float(row.get("peak_bank_mib"), default=0.0),
        "peak_bank_frames": parse_int(row.get("peak_bank_frames"), default=0),
        "peak_nvidia_smi_used_mib": parse_float(row.get("peak_nvidia_smi_used_mib")),
        "peak_torch_allocated_mib": parse_float(row.get("peak_torch_allocated_mib")),
    }
    if math.isnan(item["wall_seconds"]) or math.isnan(item["peak_nvidia_smi_used_mib"]):
        return None
    return item


def load_summary_csvs(paths: list[Path], duration_sec: int) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                item = normalize_row(raw)
                if item is not None:
                    row_duration = int(item.get("future_seconds") or 0)
                    if row_duration and row_duration != duration_sec:
                        continue
                    rows.append(item)

    deduped = {}
    for row in rows:
        key = (
            row.get("future_seconds"),
            row["memory_bank_device"],
            row["policy"],
            row["budget"],
        )
        deduped[key] = row
    return list(deduped.values())


def policy_label(item: dict[str, object]) -> str:
    policy = str(item["policy"])
    label = POLICY_LABELS.get(policy, policy.replace("_", " ").title())
    budget = item.get("budget")
    if budget not in ("", None):
        label = f"{label} b{budget}"
    return label


def item_label(item: dict[str, object]) -> str:
    bank = str(item["memory_bank_device"]).upper()
    return f"{bank} bank\n{policy_label(item)}"


def color_for(item: dict[str, object]) -> str:
    return POLICY_COLORS.get(str(item["policy"]), "#7f7f7f")


def marker_for(item: dict[str, object]) -> str:
    return BANK_MARKERS.get(str(item["memory_bank_device"]), "D")


def finite_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = parse_float(row.get(key))
        if not math.isnan(value):
            values.append(value)
    return values


def padded_limits(values: list[float], floor=None) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-9:
        pad = max(abs(hi) * 0.01, 1.0)
    else:
        pad = (hi - lo) * 0.14
    lo -= pad
    hi += pad
    if floor is not None:
        lo = min(max(lo, floor), hi - 1e-6)
    return lo, hi


def annotate(ax, x, y, text):
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    dx = 7 if x <= (xlo + xhi) / 2 else -7
    dy = 7 if y <= (ylo + yhi) / 2 else -7
    ha = "left" if dx > 0 else "right"
    va = "bottom" if dy > 0 else "top"
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.6,
        ha=ha,
        va=va,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.78},
    )


def output_path(output_dir: Path, stem: str, base: str, suffix: str) -> Path:
    name = f"{stem}_{base}.{suffix}" if stem else f"{base}.{suffix}"
    return output_dir / name


def plot_pareto(rows: list[dict[str, object]], output_dir: Path, stem: str, duration_sec: int) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    points = []
    for item in rows:
        x = float(item["wall_seconds"])
        y = float(item["peak_nvidia_smi_used_mib"])
        points.append((x, y, item_label(item)))
        ax.scatter(
            x,
            y,
            s=125,
            marker=marker_for(item),
            color=color_for(item),
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

    ax.set_title(f"WorldMem {duration_sec}s Latency vs Peak GPU Memory", fontsize=15, pad=12)
    ax.set_xlabel("Wall time per generated video (seconds)", fontsize=12)
    ax.set_ylabel("Peak GPU memory, nvidia-smi (MiB)", fontsize=12)
    ax.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.set_xlim(*padded_limits(finite_values(rows, "wall_seconds")))
    ax.set_ylim(*padded_limits(finite_values(rows, "peak_nvidia_smi_used_mib")))
    for x, y, label in points:
        annotate(ax, x, y, label)

    ax.text(
        0.0,
        -0.18,
        "CECSL one-video profile. Marker shape is bank placement: circles are CPU-bank, squares are GPU-bank.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        va="top",
    )

    fig.subplots_adjust(bottom=0.22)
    base = f"worldmem_pareto_walltime_gpu_memory_{duration_sec}s"
    for suffix in ("png", "pdf"):
        path = output_path(output_dir, stem, base, suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(path)
    plt.close(fig)


def plot_retrieval_bank(rows: list[dict[str, object]], output_dir: Path, stem: str, duration_sec: int) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    y_values = []
    points = []
    for item in rows:
        bank_frames = int(item["peak_bank_frames"])
        x = bank_frames if bank_frames > 0 else 1
        y = float(item["retrieval_seconds"])
        y_values.append(y)
        points.append((x, y, item_label(item)))
        ax.scatter(
            x,
            y,
            s=125,
            marker=marker_for(item),
            color=color_for(item),
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

    ax.set_title(f"WorldMem {duration_sec}s Retrieval Cost vs Resident Bank", fontsize=15, pad=12)
    ax.set_xlabel("Resident GPU-bank frames (log scale; CPU-bank shown at 1)", fontsize=12)
    ax.set_ylabel("Retrieval time per generated video (seconds)", fontsize=12)
    ax.set_xscale("log")
    max_frames = max([int(item["peak_bank_frames"]) for item in rows] + [1])
    ax.set_xlim(0.7, max(1800, max_frames * 1.45))
    ax.set_ylim(0, max(y_values) * 1.16 if y_values else 1)
    for x, y, label in points:
        annotate(ax, x, y, label)
    ax.grid(True, which="both", color="#d7d7d7", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)

    ax.text(
        0.0,
        -0.18,
        "Unbounded retrieval searches a growing candidate bank; bounded policies cap retained candidates and retrieval work.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        va="top",
    )

    fig.subplots_adjust(bottom=0.22)
    base = f"worldmem_retrieval_vs_bank_frames_{duration_sec}s"
    for suffix in ("png", "pdf"):
        path = output_path(output_dir, stem, base, suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(path)
    plt.close(fig)


def plot_combined(rows: list[dict[str, object]], output_dir: Path, stem: str, duration_sec: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7))
    pareto_ax, retrieval_ax = axes

    pareto_points = []
    retrieval_points = []
    for item in rows:
        color = color_for(item)
        marker = marker_for(item)
        x = float(item["wall_seconds"])
        y = float(item["peak_nvidia_smi_used_mib"])
        pareto_points.append((x, y, item_label(item)))
        pareto_ax.scatter(
            x,
            y,
            s=112,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

        bank_frames = int(item["peak_bank_frames"])
        retrieval_x = bank_frames if bank_frames > 0 else 1
        retrieval_y = float(item["retrieval_seconds"])
        retrieval_points.append((retrieval_x, retrieval_y, item_label(item)))
        retrieval_ax.scatter(
            retrieval_x,
            retrieval_y,
            s=112,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

    pareto_ax.set_title("Latency vs Peak GPU Memory", fontsize=13, pad=10)
    pareto_ax.set_xlabel("Wall time (seconds)", fontsize=11)
    pareto_ax.set_ylabel("Peak GPU memory (MiB)", fontsize=11)
    pareto_ax.set_xlim(*padded_limits(finite_values(rows, "wall_seconds")))
    pareto_ax.set_ylim(*padded_limits(finite_values(rows, "peak_nvidia_smi_used_mib")))
    for x, y, label in pareto_points:
        annotate(pareto_ax, x, y, label)
    pareto_ax.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.75)
    pareto_ax.set_axisbelow(True)

    retrieval_ax.set_title("Retrieval Cost vs Resident Bank", fontsize=13, pad=10)
    retrieval_ax.set_xlabel("Resident GPU-bank frames (log scale; CPU=1)", fontsize=11)
    retrieval_ax.set_ylabel("Retrieval time (seconds)", fontsize=11)
    retrieval_ax.set_xscale("log")
    max_frames = max([int(item["peak_bank_frames"]) for item in rows] + [1])
    retrieval_ax.set_xlim(0.7, max(1800, max_frames * 1.45))
    retrieval_ax.set_ylim(0, max(finite_values(rows, "retrieval_seconds")) * 1.16)
    for x, y, label in retrieval_points:
        annotate(retrieval_ax, x, y, label)
    retrieval_ax.grid(True, which="both", color="#d7d7d7", linewidth=0.8, alpha=0.75)
    retrieval_ax.set_axisbelow(True)

    handles = []
    labels = []
    for policy, color in POLICY_COLORS.items():
        if any(item["policy"] == policy for item in rows):
            handles.append(
                plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="black")
            )
            labels.append(POLICY_LABELS.get(policy, policy))
    handles.extend(
        [
            plt.Line2D([0], [0], marker="o", color="black", linestyle="none", label="CPU bank"),
            plt.Line2D([0], [0], marker="s", color="black", linestyle="none", label="GPU bank"),
        ]
    )
    labels.extend(["CPU bank", "GPU bank"])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=min(len(labels), 5),
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(f"WorldMem {duration_sec}s Memory-Bank Systems Profile", fontsize=16, y=1.03)
    fig.text(
        0.5,
        0.008,
        "Profile summary rows from CECSL. Total GPU peak is model-dominated; retrieval time exposes the candidate-bank scaling effect.",
        fontsize=9,
        color="#555555",
        ha="center",
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    base = f"worldmem_memory_bank_pareto_combined_{duration_sec}s"
    for suffix in ("png", "pdf"):
        path = output_path(output_dir, stem, base, suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or (repo_root / "assets" / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_csv:
        rows = load_summary_csvs(args.summary_csv, args.duration_sec)
    else:
        if args.duration_sec != 60:
            raise RuntimeError("The built-in fallback profile is only available for --duration-sec 60.")
        rows = DEFAULT_PROFILE_60S
    if not rows:
        raise RuntimeError("No successful profile rows found to plot.")

    rows = sorted(
        rows,
        key=lambda item: (
            str(item["memory_bank_device"]) != "cpu",
            POLICY_ORDER.get(str(item["policy"]), 99),
            int(item["budget"] or 0),
        ),
    )
    plot_pareto(rows, output_dir, args.output_stem, args.duration_sec)
    plot_retrieval_bank(rows, output_dir, args.output_stem, args.duration_sec)
    plot_combined(rows, output_dir, args.output_stem, args.duration_sec)


if __name__ == "__main__":
    main()
