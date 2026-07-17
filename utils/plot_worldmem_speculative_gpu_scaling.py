#!/usr/bin/env python3
"""Speculative WorldMem peak-GPU scaling curves from CECSL profile anchors."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worldmem_mplconfig")

import matplotlib.pyplot as plt


FPS = 10
CONTEXT_FRAMES = 600
DURATIONS_SEC = [10, 20, 30, 60, 90, 120, 180]

# Measured CECSL one-video profile anchors from 2026-07-17.
CPU_BASELINE_10S_MIB = 8555.0
CPU_BASELINE_60S_MIB = (10921.0 + 10915.0) / 2.0

GPU_LATENT_UNBOUNDED_MEASURED = {10: 8571.0, 60: 10953.0}
GPU_LATENT_RI_B32_MEASURED = {10: 8557.0, 60: 10917.0}
CPU_BASELINE_MEASURED = {10: CPU_BASELINE_10S_MIB, 60: CPU_BASELINE_60S_MIB}

# Resident-bank measurements from the GPU-bank analysis mode.
LATENT_UNBOUNDED_BANK_10S_MIB = 14.062
LATENT_UNBOUNDED_BANK_60S_MIB = 31.641
LATENT_RI_B32_BANK_MIB = 0.826

OUTPUT_STEM = "worldmem_speculative_peak_gpu_scaling"


def frame_count(duration_sec: int) -> int:
    return CONTEXT_FRAMES + duration_sec * FPS


def mib_to_gib(value_mib: float) -> float:
    return value_mib / 1024.0


def rgb_frame_mib(resolution: int, bytes_per_channel: int) -> float:
    return resolution * resolution * 3 * bytes_per_channel / (1024.0 * 1024.0)


def fitted_model_peak_mib(duration_sec: int) -> float:
    slope = (CPU_BASELINE_60S_MIB - CPU_BASELINE_10S_MIB) / (60 - 10)
    return CPU_BASELINE_10S_MIB + slope * (duration_sec - 10)


def latent_unbounded_bank_mib(duration_sec: int) -> float:
    slope = (LATENT_UNBOUNDED_BANK_60S_MIB - LATENT_UNBOUNDED_BANK_10S_MIB) / (60 - 10)
    return LATENT_UNBOUNDED_BANK_10S_MIB + slope * (duration_sec - 10)


def build_rows() -> list[dict[str, float | int]]:
    rgb128_fp16 = rgb_frame_mib(128, bytes_per_channel=2)
    rgb512_fp16 = rgb_frame_mib(512, bytes_per_channel=2)

    rows: list[dict[str, float | int]] = []
    for duration in DURATIONS_SEC:
        model_peak = fitted_model_peak_mib(duration)
        frames = frame_count(duration)
        latent_unbounded_bank = latent_unbounded_bank_mib(duration)
        latent_ri_bank = LATENT_RI_B32_BANK_MIB
        rgb128_bank = frames * rgb128_fp16
        rgb512_bank = frames * rgb512_fp16
        rgb512_ri_bank = 32 * rgb512_fp16

        rows.append(
            {
                "duration_sec": duration,
                "frames_context_plus_future": frames,
                "model_peak_cpu_offload_mib": model_peak,
                "latent_unbounded_bank_mib": latent_unbounded_bank,
                "latent_ri_b32_bank_mib": latent_ri_bank,
                "rgb128_fp16_unbounded_bank_mib": rgb128_bank,
                "rgb512_fp16_unbounded_bank_mib": rgb512_bank,
                "rgb512_fp16_ri_b32_bank_mib": rgb512_ri_bank,
                "peak_cpu_offload_mib": model_peak,
                "peak_latent_unbounded_mib": model_peak + latent_unbounded_bank,
                "peak_latent_ri_b32_mib": model_peak + latent_ri_bank,
                "peak_rgb128_fp16_unbounded_mib": model_peak + rgb128_bank,
                "peak_rgb512_fp16_unbounded_mib": model_peak + rgb512_bank,
                "peak_rgb512_fp16_ri_b32_mib": model_peak + rgb512_ri_bank,
            }
        )
    return rows


def series(rows: list[dict[str, float | int]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def plot(rows: list[dict[str, float | int]], output_dir: Path) -> None:
    durations = [int(row["duration_sec"]) for row in rows]

    curves = [
        {
            "key": "peak_cpu_offload_mib",
            "bank_key": None,
            "label": "CPU offload / fitted model peak",
            "color": "#4d4d4d",
            "linestyle": "-",
            "linewidth": 2.4,
        },
        {
            "key": "peak_latent_unbounded_mib",
            "bank_key": "latent_unbounded_bank_mib",
            "label": "GPU latent bank, unbounded",
            "color": "#0072b2",
            "linestyle": "-",
            "linewidth": 2.4,
        },
        {
            "key": "peak_latent_ri_b32_mib",
            "bank_key": "latent_ri_b32_bank_mib",
            "label": "GPU latent bank, RI b32",
            "color": "#56b4e9",
            "linestyle": "--",
            "linewidth": 2.4,
        },
        {
            "key": "peak_rgb128_fp16_unbounded_mib",
            "bank_key": "rgb128_fp16_unbounded_bank_mib",
            "label": "Spec RGB fp16 128^2 bank, unbounded",
            "color": "#e69f00",
            "linestyle": "-",
            "linewidth": 2.2,
        },
        {
            "key": "peak_rgb512_fp16_unbounded_mib",
            "bank_key": "rgb512_fp16_unbounded_bank_mib",
            "label": "Spec RGB fp16 512^2 bank, unbounded",
            "color": "#d55e00",
            "linestyle": "-",
            "linewidth": 2.6,
        },
        {
            "key": "peak_rgb512_fp16_ri_b32_mib",
            "bank_key": "rgb512_fp16_ri_b32_bank_mib",
            "label": "Spec RGB fp16 512^2 bank, RI b32",
            "color": "#009e73",
            "linestyle": "--",
            "linewidth": 2.4,
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    total_ax, bank_ax = axes

    for curve in curves:
        y_gib = [mib_to_gib(value) for value in series(rows, curve["key"])]
        total_ax.plot(
            durations,
            y_gib,
            label=curve["label"],
            color=curve["color"],
            linestyle=curve["linestyle"],
            linewidth=curve["linewidth"],
        )

        if curve["bank_key"] is not None:
            bank_ax.plot(
                durations,
                series(rows, curve["bank_key"]),
                label=curve["label"].replace("GPU ", "").replace("Spec ", ""),
                color=curve["color"],
                linestyle=curve["linestyle"],
                linewidth=curve["linewidth"],
            )

    total_ax.scatter(
        list(CPU_BASELINE_MEASURED),
        [mib_to_gib(v) for v in CPU_BASELINE_MEASURED.values()],
        color="#4d4d4d",
        edgecolor="white",
        linewidth=0.9,
        s=75,
        marker="o",
        zorder=4,
        label="Measured CPU-bank anchors",
    )
    total_ax.scatter(
        list(GPU_LATENT_UNBOUNDED_MEASURED),
        [mib_to_gib(v) for v in GPU_LATENT_UNBOUNDED_MEASURED.values()],
        color="#0072b2",
        edgecolor="white",
        linewidth=0.9,
        s=80,
        marker="s",
        zorder=4,
        label="Measured latent unbounded",
    )
    total_ax.scatter(
        list(GPU_LATENT_RI_B32_MEASURED),
        [mib_to_gib(v) for v in GPU_LATENT_RI_B32_MEASURED.values()],
        color="#56b4e9",
        edgecolor="white",
        linewidth=0.9,
        s=80,
        marker="D",
        zorder=4,
        label="Measured latent RI b32",
    )

    total_ax.set_title("Projected Total Peak GPU Usage", fontsize=14, pad=10)
    total_ax.set_xlabel("Generated future duration (seconds)", fontsize=11)
    total_ax.set_ylabel("Peak GPU usage (GiB)", fontsize=11)
    total_ax.set_xticks(DURATIONS_SEC)
    total_ax.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.75)
    total_ax.set_axisbelow(True)

    bank_ax.set_title("Resident Bank Component", fontsize=14, pad=10)
    bank_ax.set_xlabel("Generated future duration (seconds)", fontsize=11)
    bank_ax.set_ylabel("Resident bank memory (MiB, log scale)", fontsize=11)
    bank_ax.set_xticks(DURATIONS_SEC)
    bank_ax.set_yscale("log")
    bank_ax.set_ylim(0.5, 5000)
    bank_ax.grid(True, which="both", color="#d7d7d7", linewidth=0.8, alpha=0.75)
    bank_ax.set_axisbelow(True)

    fig.suptitle("WorldMem Speculative Peak-GPU Scaling", fontsize=16, y=1.02)
    fig.text(
        0.02,
        -0.03,
        "Measured anchors: CECSL one-video profiles at 10s and 60s. Curves beyond those points extrapolate a fitted model peak plus resident-bank size. "
        "WorldMem's actual bank is compact latent state at 128x128; RGB curves are speculative what-if storage variants.",
        fontsize=9,
        color="#555555",
    )

    handles, labels = total_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.16),
        fontsize=8.7,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28, top=0.88, wspace=0.24)

    for suffix in ("png", "pdf"):
        path = output_dir / f"{OUTPUT_STEM}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(path)
    plt.close(fig)


def write_csv(rows: list[dict[str, float | int]], output_dir: Path) -> None:
    path = output_dir / f"{OUTPUT_STEM}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def main() -> None:
    output_dir = Path("assets/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows, output_dir)
    plot(rows, output_dir)


if __name__ == "__main__":
    main()
