"""Validate and plot WorldMem retrieval deterioration from an exported CSV.

This stage is CPU-only: it does not import Torch, decode videos, or extract
features. Query means are already set-level statistics from the exporter.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = (
    "selected_view_mismatch",
    "selected_memory_corruption",
    "selected_effective_mismatch",
    "full_oracle_best_k_effective_mismatch",
    "full_oracle_effective_mismatch",
)
NAMES = (
    "View mismatch",
    "Stored-content corruption",
    "Selected-set effective mismatch",
    "Best-8 available mismatch",
    "Best-single available mismatch",
)
IDENTITY = (
    "row",
    "scene",
    "dataset_start_frame",
    "duration_sec",
    "generation_seed",
)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_sections(path, run, duration, expected_videos):
    grouped = defaultdict(list)
    times = defaultdict(list)
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            *IDENTITY,
            *FIELDS,
            "run_name",
            "section_idx",
            "target_frame",
            "rollout_frame",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for row in reader:
            if row["run_name"] != run or int(row["duration_sec"]) != duration:
                continue
            identity = tuple(row[field] for field in IDENTITY)
            section = int(row["section_idx"])
            target = int(row["target_frame"])
            key = (*identity, section, target)
            if key in seen:
                raise ValueError(f"Duplicate retrieval query: {key}")
            seen.add(key)
            values = np.asarray([float(row[field]) for field in FIELDS])
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite metric in query {key}")
            selected, best_k, best_single = values[2], values[3], values[4]
            if best_single > best_k + 1e-5 or best_k > selected + 1e-5:
                raise ValueError(f"Oracle ordering is invalid in query {key}")
            if row.get("candidate_count_mismatch") and float(row["candidate_count_mismatch"]) != 0:
                raise ValueError(f"Candidate-count mismatch in query {key}")
            grouped[(identity, section)].append(values)
            times[(identity, section)].append(float(row["rollout_frame"]))

    identities = sorted({key[0] for key in grouped})
    if len(identities) != expected_videos:
        raise ValueError(
            f"Expected {expected_videos} trajectories for {run}/{duration}s; "
            f"found {len(identities)}"
        )
    sections = sorted(section for identity, section in grouped if identity == identities[0])
    if len(sections) < 8 or any(right != left + 1 for left, right in zip(sections, sections[1:])):
        raise ValueError("Need at least eight contiguous sections per trajectory")
    for identity in identities:
        current = sorted(section for item, section in grouped if item == identity)
        if current != sections:
            raise ValueError(f"Section coverage differs for {identity}")
    values = np.asarray(
        [[np.mean(grouped[(identity, section)], axis=0) for section in sections] for identity in identities]
    )
    targets = np.asarray(
        [[np.mean(times[(identity, section)]) for section in sections] for identity in identities]
    )
    return identities, sections, values, targets, len(seen)


def summarize(values, targets, fps, bins, repeats, seed):
    trajectories, sections, _ = values.shape
    if trajectories < 2 or not 2 <= bins <= sections or repeats < 100 or fps <= 0:
        raise ValueError(
            "Need >=2 trajectories, 2..Nsection bins, >=100 bootstrap draws, and positive FPS"
        )
    quarter = sections // 4
    deltas = values[:, -quarter:].mean(axis=1) - values[:, :quarter].mean(axis=1)
    groups = np.array_split(np.arange(sections), bins)
    curves = np.stack([values[:, indices].mean(axis=1) for indices in groups], axis=1)
    time_sec = np.asarray([targets[:, indices].mean() / fps for indices in groups])
    draws = np.random.default_rng(seed).integers(
        0, trajectories, size=(repeats, trajectories)
    )

    def interval(array):
        samples = array[draws].mean(axis=1)
        return np.quantile(samples, [0.025, 0.975], axis=0)

    return {
        "quarter_sections": quarter,
        "deltas": deltas,
        "delta_mean": deltas.mean(axis=0),
        "delta_ci": interval(deltas),
        "curve_mean": curves.mean(axis=0),
        "curve_ci": interval(curves),
        "time_sec": time_sec,
    }


def plot(result, trajectory_count, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 3.8),
        layout="constrained",
        gridspec_kw={"width_ratios": [1, 1.45]},
    )
    left = axes[0]
    for index, color in enumerate(("#238575", "#BD443A")):
        y = 1 - index
        mean = result["delta_mean"][index]
        low, high = result["delta_ci"][:, index]
        left.scatter(
            result["deltas"][:, index],
            y + np.linspace(-0.1, 0.1, trajectory_count),
            s=16,
            color=color,
            alpha=0.3,
        )
        left.plot([low, high], [y, y], color=color, linewidth=2.5)
        left.scatter([mean], [y], color=color, s=48, zorder=3)
        left.annotate(
            f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]",
            (mean, y),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    left.axvline(0, color="#777777", linewidth=1, linestyle="--")
    left.set(
        yticks=[1, 0],
        yticklabels=["View\nmismatch", "Memory\ncorruption"],
        ylim=(-0.45, 1.55),
        xlabel="Late minus early DINO distance",
    )
    left.margins(x=0.3)
    left.set_title("(a) Changes in retrieved memory", loc="left", fontsize=11)

    right = axes[1]
    curves = (
        (2, "Selected 8-memory set", "#BD443A", "-"),
        (3, "Best 8 available", "#2673A8", "-"),
        (4, "Best single available", "#6FA6C8", "--"),
    )
    for index, name, color, style in curves:
        right.plot(
            result["time_sec"],
            result["curve_mean"][:, index],
            marker="o",
            markersize=4,
            color=color,
            linestyle=style,
            label=name,
            linewidth=2,
        )
        right.fill_between(
            result["time_sec"],
            result["curve_ci"][0, :, index],
            result["curve_ci"][1, :, index],
            color=color,
            alpha=0.1,
        )
    right.set(xlabel="Generated rollout time (s)", ylabel="DINO distance to target GT")
    right.set_title("(b) Selected versus eligible evidence", loc="left", fontsize=11)
    right.legend(frameon=False, fontsize=8.5)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"retrieval_deterioration.{extension}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run", default="worldmem_memquality_unbounded_60s_n15_seed101"
    )
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Missing source CSV: {args.input}")
    if args.expected_videos < 2:
        parser.error("At least two trajectories are required")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error("Use an empty output directory or pass --overwrite")

    identities, sections, values, targets, queries = load_sections(
        args.input, args.run, args.duration, args.expected_videos
    )
    result = summarize(
        values, targets, args.fps, args.bins, args.bootstrap, args.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    plot(result, len(identities), args.output)
    summary = [
        {
            "metric": field,
            "late_minus_early": float(result["delta_mean"][index]),
            "ci_low": float(result["delta_ci"][0, index]),
            "ci_high": float(result["delta_ci"][1, index]),
        }
        for index, field in enumerate(FIELDS)
    ]
    write_csv(args.output / "changes.csv", summary)
    write_csv(
        args.output / "trajectory_changes.csv",
        [
            dict(zip(IDENTITY, identity), **dict(zip(FIELDS, delta)))
            for identity, delta in zip(identities, result["deltas"])
        ],
    )
    write_csv(
        args.output / "curves.csv",
        [
            {
                "time_sec": float(time),
                "metric": field,
                "mean": float(result["curve_mean"][bin_index, field_index]),
                "ci_low": float(result["curve_ci"][0, bin_index, field_index]),
                "ci_high": float(result["curve_ci"][1, bin_index, field_index]),
            }
            for bin_index, time in enumerate(result["time_sec"])
            for field_index, field in enumerate(FIELDS)
        ],
    )
    provenance = {
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "trajectories": [dict(zip(IDENTITY, identity)) for identity in identities],
        "queries": queries,
        "sections_total": len(identities) * len(sections),
        "section_indices": sections,
        "early_sections": sections[: result["quarter_sections"]],
        "late_sections": sections[-result["quarter_sections"] :],
        "method": "One query per WorldMem generated frame. Each query is the uniform mean over all eight discrete selected memories. Sections and trajectories receive equal weight. First/last floor(Nsections/4) sections define early/late. Eight equal-count temporal bins. Paired trajectory bootstrap with pointwise percentile 95% intervals.",
        "interpretation": "Observational temporal diagnostic, not a causal intervention. Best-single is optimistic; best-8 matches selected-set cardinality. Generated-memory DINO features come from saved-MP4 RGB proxies as documented by export_provenance.json.",
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "caption.txt").write_text(
        f"Unbounded WorldMem retrieval diagnostics on {len(identities)} trajectories "
        f"({queries:,} frame-level retrieval queries; eight selected memories per query). "
        "(a) Late-minus-early changes in clean-view mismatch and stored-content corruption; "
        "faint points are trajectory changes. (b) Effective mismatch of the selected memory "
        "set versus the hindsight-best eligible eight-item set and best single item. All "
        "distances use DINO features; lower is better. Intervals resample complete "
        "trajectories (95%); curve bands are pointwise. Generated memory RGB is a proxy "
        "decoded from the saved prediction MP4. This is observational, not causal.\n",
        encoding="utf-8",
    )
    print(
        f"Source: {args.input}\n{len(identities)} trajectories; {queries:,} queries; "
        f"{len(identities) * len(sections):,} sections"
    )
    for name, row in zip(NAMES, summary):
        print(
            f"{name:34s} {row['late_minus_early']:+.4f}  "
            f"95% CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        )
    print(f"Figure: {args.output / 'retrieval_deterioration.pdf'}")


if __name__ == "__main__":
    main()
