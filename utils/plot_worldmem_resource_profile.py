#!/usr/bin/env python3
"""Build measured WorldMem archive-growth and resource-profile figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/worldmem_mplconfig")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


COLORS = {
    "unbounded": "#3d424a",
    "bounded": "#228b4e",
    "descriptor": "#74b98b",
    "other_update": "#d7e8dc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bounded-policy", default="slam_covisibility")
    parser.add_argument("--bounded-label", default="KEEPSAKE")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--expected-duration-sec", type=float, default=60.0)
    return parser.parse_args()


def number(value, default=math.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def load_summary(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Profile summary not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    successful = [row for row in rows if int(number(row.get("status"), 1)) == 0]
    if not successful:
        raise RuntimeError(f"No successful rows in {path}")
    return successful


def row_budget(row: dict[str, str]) -> int | None:
    value = number(row.get("budget"))
    return None if math.isnan(value) else int(value)


def select_row(
    rows: list[dict[str, str]],
    policy: str,
    budget: int | None,
    bank_device: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get("policy") == policy
        and row_budget(row) == budget
        and row.get("memory_bank_device") == bank_device
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {bank_device}-bank row for policy={policy}, "
            f"budget={budget}; found {len(matches)}"
        )
    return matches[0]


def load_archive_states(trace_path: Path, fps: float) -> list[dict[str, float]]:
    states = []
    legacy_gpu_states = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "memory_archive_state":
                generated_frames = int(record["generated_frames"])
                states.append(
                    {
                        "generated_frame": generated_frames,
                        "generated_time_sec": generated_frames / fps,
                        "archive_frames": int(record["archive_frames"]),
                        "archive_mib": float(record["archive_latent_payload_mib"]),
                        "gpu_bank_mib": float(record["gpu_bank_payload_mib"]),
                        "history_mib": float(record["history_latent_payload_mib"]),
                    }
                )
            elif record.get("event") == "gpu_memory_bank_sync":
                legacy_gpu_states.append(record)

    if not states and legacy_gpu_states:
        for event_index, record in enumerate(legacy_gpu_states):
            generated_frames = int(record.get("generated_frames", event_index))
            bank_mib = float(record["estimated_bank_mib"])
            states.append(
                {
                    "generated_frame": generated_frames,
                    "generated_time_sec": generated_frames / fps,
                    "archive_frames": int(record["stored_memory_size"]),
                    "archive_mib": bank_mib,
                    "gpu_bank_mib": bank_mib,
                    "history_mib": math.nan,
                }
            )
    if not states:
        raise RuntimeError(
            f"No memory_archive_state events in {trace_path}. Re-run the current "
            "resource profiler to obtain measured growth traces."
        )

    deduplicated = {}
    for state in states:
        deduplicated[int(state["generated_frame"])] = state
    return [deduplicated[index] for index in sorted(deduplicated)]


def validate_growth(
    unbounded: list[dict[str, float]],
    bounded: list[dict[str, float]],
    budget: int,
    expected_duration_sec: float,
) -> None:
    for label, states in (("unbounded", unbounded), ("bounded", bounded)):
        final_time = states[-1]["generated_time_sec"]
        if not math.isclose(final_time, expected_duration_sec, abs_tol=0.11):
            raise RuntimeError(
                f"{label} trace ends at {final_time:.3f}s, expected "
                f"{expected_duration_sec:.3f}s"
            )
    unbounded_sizes = [state["archive_frames"] for state in unbounded]
    if any(right < left for left, right in zip(unbounded_sizes, unbounded_sizes[1:])):
        raise RuntimeError("Unbounded archive size is not monotonic")
    if max(state["archive_frames"] for state in bounded) > budget:
        raise RuntimeError(f"Bounded archive exceeded B={budget}")


def method_label(policy: str, bounded_label: str, budget: int) -> str:
    if policy == "unbounded":
        return "Complete retention"
    return f"{bounded_label} (B={budget})"


def write_growth_csv(
    path: Path,
    traces: list[tuple[str, str, Path, list[dict[str, float]]]],
) -> None:
    fields = [
        "method",
        "policy",
        "generated_frame",
        "generated_time_sec",
        "archive_frames",
        "archive_latent_payload_mib",
        "gpu_bank_payload_mib",
        "history_latent_payload_mib",
        "source_trace",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, policy, source, states in traces:
            for state in states:
                writer.writerow(
                    {
                        "method": label,
                        "policy": policy,
                        "generated_frame": int(state["generated_frame"]),
                        "generated_time_sec": f'{state["generated_time_sec"]:.3f}',
                        "archive_frames": int(state["archive_frames"]),
                        "archive_latent_payload_mib": f'{state["archive_mib"]:.6f}',
                        "gpu_bank_payload_mib": f'{state["gpu_bank_mib"]:.6f}',
                        "history_latent_payload_mib": (
                            ""
                            if math.isnan(state["history_mib"])
                            else f'{state["history_mib"]:.6f}'
                        ),
                        "source_trace": str(source),
                    }
                )


def plot_growth(
    output_stem: Path,
    traces: list[tuple[str, str, Path, list[dict[str, float]]]],
    expected_duration_sec: float,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for label, policy, _, states in traces:
        key = "unbounded" if policy == "unbounded" else "bounded"
        times = [state["generated_time_sec"] for state in states]
        axes[0].plot(
            times,
            [state["archive_frames"] for state in states],
            color=COLORS[key],
            linewidth=2.6,
            label=label,
        )
        axes[1].plot(
            times,
            [state["gpu_bank_mib"] for state in states],
            color=COLORS[key],
            linewidth=2.6,
            label=label,
        )

    ticks = list(range(0, int(expected_duration_sec) + 1, 15))
    for axis in axes:
        axis.set_xlim(0, expected_duration_sec)
        axis.set_xticks(ticks)
        axis.set_xlabel("Generated time (s)")
        axis.grid(axis="y", color="#dce0e5", linewidth=0.8)
        axis.set_axisbelow(True)
    axes[0].set_title("Persistent archive size")
    axes[0].set_ylabel("Retained latent frames")
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_title("GPU-resident archive payload (analysis mode)")
    axes[1].set_ylabel("Latent tensor payload (MiB)")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Bounded memory stays flat as complete retention grows",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.025,
        "Archive payload only; total host/CUDA footprints are measured separately.",
        ha="center",
        fontsize=8.5,
        color="#555b63",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def required_number(row: dict[str, str], key: str) -> float:
    value = number(row.get(key))
    if not math.isfinite(value):
        raise RuntimeError(
            f"Profile row {row.get('run_name')} lacks {key}; re-run the current profiler"
        )
    return value


def resource_record(label: str, row: dict[str, str]) -> dict[str, object]:
    update_count = required_number(row, "memory_update_count")
    descriptor_seconds = required_number(row, "descriptor_extraction_seconds")
    return {
        "method": label,
        "run_name": row["run_name"],
        "bank_device": row["memory_bank_device"],
        "final_archive_frames": int(required_number(row, "final_archive_frames")),
        "final_archive_mib": required_number(row, "final_archive_mib"),
        "retrieval_ms_mean": required_number(row, "retrieval_ms_mean"),
        "retrieval_ms_median": required_number(row, "retrieval_ms_median"),
        "retrieval_ms_p95": required_number(row, "retrieval_ms_p95"),
        "descriptor_seconds": descriptor_seconds,
        "descriptor_ms_per_update": 1000.0 * descriptor_seconds / update_count,
        "update_ms_per_update": required_number(row, "memory_update_ms_mean"),
        "peak_process_tree_rss_mib": required_number(
            row, "peak_process_tree_rss_mib"
        ),
        "peak_torch_allocated_mib": required_number(
            row, "peak_torch_allocated_mib"
        ),
        "peak_nvidia_smi_used_mib": required_number(
            row, "peak_nvidia_smi_used_mib"
        ),
        "wall_seconds": required_number(row, "wall_seconds"),
    }


def write_resource_table(path: Path, records: list[dict[str, object]]) -> None:
    fields = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_resource_tex(path: Path, records: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Archive & Archive MiB & Retrieval ms & Update ms & Host GiB & CUDA GiB \\",
        r"\midrule",
    ]
    for row in records:
        lines.append(
            f"{row['method']} & {row['final_archive_frames']:,} & "
            f"{row['final_archive_mib']:.2f} & {row['retrieval_ms_mean']:.3f} & "
            f"{row['update_ms_per_update']:.3f} & "
            f"{row['peak_process_tree_rss_mib'] / 1024:.2f} & "
            f"{row['peak_torch_allocated_mib'] / 1024:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_resources(output_stem: Path, records: list[dict[str, object]]) -> None:
    labels = [str(row["method"]) for row in records]
    colors = [COLORS["unbounded"], COLORS["bounded"]]
    x = list(range(len(records)))
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0))

    panels = [
        (axes[0, 0], "retrieval_ms_mean", "Retrieval latency", "ms / query"),
        (axes[1, 0], "peak_process_tree_rss_mib", "Peak host memory", "GiB"),
        (axes[1, 1], "peak_torch_allocated_mib", "Peak CUDA memory", "GiB"),
    ]
    for axis, key, title, ylabel in panels:
        scale = 1024.0 if ylabel == "GiB" else 1.0
        values = [float(row[key]) / scale for row in records]
        bars = axis.bar(x, values, color=colors, width=0.58)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="#e0e3e7", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        axis.set_ylim(0, max(values) * 1.20 if max(values) else 1)

    update_axis = axes[0, 1]
    descriptor = [float(row["descriptor_ms_per_update"]) for row in records]
    total = [float(row["update_ms_per_update"]) for row in records]
    other = [max(total_value - desc, 0.0) for total_value, desc in zip(total, descriptor)]
    update_axis.bar(
        x,
        other,
        color=COLORS["other_update"],
        width=0.58,
        label="Other update",
    )
    update_axis.bar(
        x,
        descriptor,
        bottom=other,
        color=COLORS["descriptor"],
        width=0.58,
        label="Descriptor extraction",
    )
    update_axis.set_title("Archive-update overhead")
    update_axis.set_ylabel("ms / update (inclusive)")
    update_axis.set_xticks(x, labels)
    update_axis.grid(axis="y", color="#e0e3e7", linewidth=0.8)
    update_axis.set_axisbelow(True)
    update_axis.legend(frameon=False, fontsize=8)
    update_axis.set_ylim(0, max(total) * 1.20 if max(total) else 1)

    fig.suptitle("WorldMem resource profile", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    profile_root = args.profile_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else profile_root / "resource_report"
    )
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary(profile_root / "summary.csv")
    gpu_unbounded = select_row(rows, "unbounded", None, "gpu")
    gpu_bounded = select_row(
        rows,
        args.bounded_policy,
        args.budget,
        "gpu",
    )
    cpu_unbounded = select_row(rows, "unbounded", None, "cpu")
    cpu_bounded = select_row(
        rows,
        args.bounded_policy,
        args.budget,
        "cpu",
    )

    unbounded_trace = Path(gpu_unbounded["trace_path"])
    bounded_trace = Path(gpu_bounded["trace_path"])
    unbounded_states = load_archive_states(unbounded_trace, args.fps)
    bounded_states = load_archive_states(bounded_trace, args.fps)
    validate_growth(
        unbounded_states,
        bounded_states,
        args.budget,
        args.expected_duration_sec,
    )

    bounded_label = method_label(
        args.bounded_policy,
        args.bounded_label,
        args.budget,
    )
    traces = [
        ("Complete retention", "unbounded", unbounded_trace, unbounded_states),
        (bounded_label, args.bounded_policy, bounded_trace, bounded_states),
    ]
    growth_csv = tables_dir / "worldmem_archive_growth.csv"
    write_growth_csv(growth_csv, traces)
    plot_growth(
        figures_dir / "worldmem_archive_growth",
        traces,
        args.expected_duration_sec,
    )

    resource_records = [
        resource_record("Complete retention", cpu_unbounded),
        resource_record(bounded_label, cpu_bounded),
    ]
    write_resource_table(
        tables_dir / "worldmem_resource_summary.csv",
        resource_records,
    )
    write_resource_tex(
        tables_dir / "worldmem_resource_summary.tex",
        resource_records,
    )
    plot_resources(
        figures_dir / "worldmem_resource_summary",
        resource_records,
    )

    provenance = {
        "profile_root": str(profile_root),
        "summary_csv": str(profile_root / "summary.csv"),
        "environment_manifest": str(profile_root / "environment.json"),
        "growth_bank_device": "gpu",
        "resource_summary_bank_device": "cpu",
        "fps": args.fps,
        "expected_duration_sec": args.expected_duration_sec,
        "bounded_policy": args.bounded_policy,
        "budget": args.budget,
        "archive_payload_scope": "latent tensor data only; excludes Python and pose metadata",
        "host_memory_scope": "sum of RSS for the launched inference process tree",
        "gpu_memory_scope": "PyTorch max allocated bytes for the inference process",
        "retrieval_scope": "generate_condition_indices with CUDA synchronization",
        "update_scope": "initial archive construction plus generation-time updates; descriptor time is included",
        "source_traces": [str(unbounded_trace), str(bounded_trace)],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    print("WorldMem measured resource report")
    for row in resource_records:
        print(
            f"{row['method']:<25} archive={row['final_archive_frames']:4d} "
            f"retrieval={row['retrieval_ms_mean']:.3f} ms/query "
            f"host={row['peak_process_tree_rss_mib'] / 1024:.2f} GiB "
            f"CUDA={row['peak_torch_allocated_mib'] / 1024:.2f} GiB"
        )
    print(f"Wrote: {output_dir}")


if __name__ == "__main__":
    main()
