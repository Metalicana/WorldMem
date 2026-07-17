#!/usr/bin/env python3
"""Plot the WorldMem CPU/GPU memory-bank Pareto profile."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worldmem_mplconfig")

import matplotlib.pyplot as plt


PROFILE_60S = [
    {
        "label": "CPU bank\nUnbounded",
        "policy": "Unbounded",
        "bank": "CPU",
        "wall_seconds": 748,
        "retrieval_seconds": 44.010,
        "sampling_seconds": 680.023,
        "peak_bank_mib": 0.0,
        "peak_bank_frames": 0,
        "peak_gpu_mib": 10921.0,
        "peak_torch_allocated_mib": 9640.697,
    },
    {
        "label": "CPU bank\nRI b32",
        "policy": "RI b32",
        "bank": "CPU",
        "wall_seconds": 716,
        "retrieval_seconds": 3.050,
        "sampling_seconds": 688.344,
        "peak_bank_mib": 0.0,
        "peak_bank_frames": 0,
        "peak_gpu_mib": 10915.0,
        "peak_torch_allocated_mib": 9640.697,
    },
    {
        "label": "GPU bank\nUnbounded",
        "policy": "Unbounded",
        "bank": "GPU",
        "wall_seconds": 746,
        "retrieval_seconds": 44.440,
        "sampling_seconds": 678.447,
        "peak_bank_mib": 31.641,
        "peak_bank_frames": 1200,
        "peak_gpu_mib": 10953.0,
        "peak_torch_allocated_mib": 9673.181,
    },
    {
        "label": "GPU bank\nRI b32",
        "policy": "RI b32",
        "bank": "GPU",
        "wall_seconds": 711,
        "retrieval_seconds": 2.705,
        "sampling_seconds": 685.963,
        "peak_bank_mib": 0.826,
        "peak_bank_frames": 32,
        "peak_gpu_mib": 10917.0,
        "peak_torch_allocated_mib": 9642.367,
    },
]

COLORS = {
    ("CPU", "Unbounded"): "#111111",
    ("CPU", "RI b32"): "#0072b2",
    ("GPU", "Unbounded"): "#d55e00",
    ("GPU", "RI b32"): "#009e73",
}

MARKERS = {"CPU": "o", "GPU": "s"}


def annotate(ax, x, y, text, dx=6, dy=6):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.75},
    )


def plot_pareto(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    for item in PROFILE_60S:
        color = COLORS[(item["bank"], item["policy"])]
        ax.scatter(
            item["wall_seconds"],
            item["peak_gpu_mib"],
            s=130,
            marker=MARKERS[item["bank"]],
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
            label=item["label"].replace("\n", " "),
        )
        annotate(ax, item["wall_seconds"], item["peak_gpu_mib"], item["label"])

    ax.set_title("WorldMem 60s Pareto Profile", fontsize=15, pad=12)
    ax.set_xlabel("Wall time per generated video (seconds)", fontsize=12)
    ax.set_ylabel("Peak GPU memory, nvidia-smi (MiB)", fontsize=12)
    ax.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.set_xlim(705, 752)
    ax.set_ylim(10895, 10965)

    ax.text(
        0.0,
        -0.18,
        "CECSL, one 60s WorldMem rollout. RI b32 improves latency and quality at essentially unchanged total GPU peak.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        va="top",
    )

    fig.subplots_adjust(bottom=0.2)
    for suffix in ("png", "pdf"):
        path = output_dir / f"worldmem_pareto_walltime_gpu_memory_60s.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(path)
    plt.close(fig)


def plot_retrieval_bank(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    for item in PROFILE_60S:
        color = COLORS[(item["bank"], item["policy"])]
        bank_frames = item["peak_bank_frames"]
        x = bank_frames if bank_frames > 0 else 1
        ax.scatter(
            x,
            item["retrieval_seconds"],
            s=130,
            marker=MARKERS[item["bank"]],
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        annotate(ax, x, item["retrieval_seconds"], item["label"])

    ax.set_title("WorldMem 60s Retrieval Cost vs Resident Bank", fontsize=15, pad=12)
    ax.set_xlabel("Resident GPU-bank frames (log scale; CPU-bank shown at 1)", fontsize=12)
    ax.set_ylabel("Retrieval time per generated video (seconds)", fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 1800)
    ax.set_ylim(0, 48)
    ax.grid(True, which="both", color="#d7d7d7", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)

    ax.text(
        0.0,
        -0.18,
        "Unbounded retrieval searches a growing candidate bank; RI b32 caps candidates and cuts retrieval from ~44s to ~3s.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        va="top",
    )

    fig.subplots_adjust(bottom=0.2)
    for suffix in ("png", "pdf"):
        path = output_dir / f"worldmem_retrieval_vs_bank_frames_60s.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(path)
    plt.close(fig)


def plot_combined(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))
    pareto_ax, retrieval_ax = axes

    for item in PROFILE_60S:
        color = COLORS[(item["bank"], item["policy"])]
        marker = MARKERS[item["bank"]]
        pareto_ax.scatter(
            item["wall_seconds"],
            item["peak_gpu_mib"],
            s=115,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        annotate(pareto_ax, item["wall_seconds"], item["peak_gpu_mib"], item["label"], dx=5, dy=5)

        bank_frames = item["peak_bank_frames"]
        x = bank_frames if bank_frames > 0 else 1
        retrieval_ax.scatter(
            x,
            item["retrieval_seconds"],
            s=115,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        annotate(retrieval_ax, x, item["retrieval_seconds"], item["label"], dx=5, dy=5)

    pareto_ax.set_title("Latency vs Peak GPU Memory", fontsize=13, pad=10)
    pareto_ax.set_xlabel("Wall time (seconds)", fontsize=11)
    pareto_ax.set_ylabel("Peak GPU memory (MiB)", fontsize=11)
    pareto_ax.set_xlim(705, 752)
    pareto_ax.set_ylim(10895, 10965)
    pareto_ax.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.75)
    pareto_ax.set_axisbelow(True)

    retrieval_ax.set_title("Retrieval Cost vs Resident Bank", fontsize=13, pad=10)
    retrieval_ax.set_xlabel("Resident GPU-bank frames (log scale; CPU=1)", fontsize=11)
    retrieval_ax.set_ylabel("Retrieval time (seconds)", fontsize=11)
    retrieval_ax.set_xscale("log")
    retrieval_ax.set_xlim(0.7, 1800)
    retrieval_ax.set_ylim(0, 48)
    retrieval_ax.grid(True, which="both", color="#d7d7d7", linewidth=0.8, alpha=0.75)
    retrieval_ax.set_axisbelow(True)

    fig.suptitle("WorldMem 60s Memory-Bank Systems Profile", fontsize=16, y=1.03)
    fig.text(
        0.02,
        -0.02,
        "CECSL, one 60s rollout. Bounded RI b32 improves quality, caps resident bank size, and reduces retrieval time; total GPU peak is model-dominated.",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_dir / f"worldmem_memory_bank_pareto_combined_60s.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(path)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "assets" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_pareto(output_dir)
    plot_retrieval_bank(output_dir)
    plot_combined(output_dir)


if __name__ == "__main__":
    main()
