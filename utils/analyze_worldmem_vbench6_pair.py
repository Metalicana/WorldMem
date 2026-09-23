"""Paired video-level VBench-6 analysis for two WorldMem runs.

The six-dimension score follows the custom aggregate used for MemCam. It is not
the official full VBench Quality or Total Score.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.build_worldmem_final_metric_status import (  # noqa: E402
    DIMENSIONS,
    latest_result,
    load_json,
)
from utils.calculate_worldmem_vbench6 import SPEC, calculate_vbench6  # noqa: E402


VIDEO_RE = re.compile(r"^video_batch(?P<batch>\d+)_0_rank0(?:_step.*)?\.mp4$")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_contribution(dimension, value):
    low, high, weight = SPEC[dimension]
    total_weight = sum(item[2] for item in SPEC.values())
    return 100.0 * weight * (float(value) - low) / (high - low) / total_weight


def detail_score(dimension, detail):
    if not isinstance(detail, dict) or "video_results" not in detail:
        raise ValueError(f"Malformed {dimension} per-video result")
    value = float(detail["video_results"])
    # VBench's imaging details are MUSIQ scores on 0-100, while its aggregate
    # is divided by 100 before being returned.
    if dimension == "imaging_quality":
        value /= 100.0
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid {dimension} per-video score: {value}")
    return value


def batch_by_source_name(manifest, limit):
    if manifest.get("selected_batch_ids") != list(range(limit)):
        raise ValueError("Manifest does not contain the required matched batch prefix")
    selected = manifest.get("selected_videos")
    if not isinstance(selected, list) or len(selected) != limit:
        raise ValueError("Manifest has the wrong selected-video count")
    result = {}
    for row in selected:
        name = row.get("source_name")
        batch = row.get("batch_id")
        match = VIDEO_RE.match(str(name))
        if match is None or int(match.group("batch")) != int(batch):
            raise ValueError(f"Manifest filename/batch mismatch: {row}")
        if name in result:
            raise ValueError(f"Duplicate manifest source name: {name}")
        result[name] = int(batch)
    return result


def load_run(root, run_name, limit, tolerance=1e-5):
    run_dir = Path(root) / run_name
    result_path = latest_result(run_dir)
    manifest_path = run_dir / "input_selection.json"
    if result_path is None or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing VBench result or manifest for {run_name}")
    manifest = load_json(manifest_path)
    source_to_batch = batch_by_source_name(manifest, limit)
    payload = load_json(result_path)
    if set(payload) != set(DIMENSIONS):
        raise ValueError(
            f"{run_name}: expected exactly six dimensions; found {sorted(payload)}"
        )

    by_batch = {batch: {} for batch in range(limit)}
    run_means = {}
    for dimension in DIMENSIONS:
        value = payload[dimension]
        if not isinstance(value, list) or len(value) not in (2, 3):
            raise ValueError(f"{run_name}: malformed aggregate for {dimension}")
        run_means[dimension] = float(value[0])
        details = value[-1]
        if not isinstance(details, list) or len(details) != limit:
            raise ValueError(f"{run_name}: wrong per-video count for {dimension}")
        seen = set()
        for detail in details:
            name = Path(str(detail.get("video_path", ""))).name
            if name not in source_to_batch:
                raise ValueError(f"{run_name}: unexpected {dimension} video {name}")
            batch = source_to_batch[name]
            if batch in seen:
                raise ValueError(f"{run_name}: duplicate {dimension} batch {batch}")
            seen.add(batch)
            by_batch[batch][dimension] = detail_score(dimension, detail)
        per_video_mean = float(np.mean([by_batch[b][dimension] for b in range(limit)]))
        if not math.isclose(per_video_mean, run_means[dimension], abs_tol=tolerance, rel_tol=0):
            raise ValueError(
                f"{run_name}: {dimension} per-video mean {per_video_mean:.9f} "
                f"does not reproduce aggregate {run_means[dimension]:.9f}"
            )

    rows = []
    for batch in range(limit):
        if set(by_batch[batch]) != set(DIMENSIONS):
            raise ValueError(f"{run_name}: incomplete dimensions for batch {batch}")
        rows.append({
            "run_name": run_name,
            "batch_id": batch,
            **by_batch[batch],
            "vbench6_percent": calculate_vbench6(by_batch[batch]),
        })
    aggregate = calculate_vbench6(run_means)
    if not math.isclose(
        aggregate,
        float(np.mean([row["vbench6_percent"] for row in rows])),
        abs_tol=tolerance * 100,
        rel_tol=0,
    ):
        raise ValueError(f"{run_name}: video-level aggregate does not reproduce run aggregate")
    return rows, run_means, aggregate, result_path, manifest_path


def percentile_interval(values, bootstrap_indices):
    values = np.asarray(values, dtype=np.float64)
    draws = values[bootstrap_indices].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def analyze_pair(left_rows, right_rows, samples=10000, seed=17):
    left = {int(row["batch_id"]): row for row in left_rows}
    right = {int(row["batch_id"]): row for row in right_rows}
    if set(left) != set(right):
        raise ValueError("Runs do not contain the same batch IDs")
    batches = sorted(left)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(batches), size=(samples, len(batches)))
    left_scores = np.asarray([left[batch]["vbench6_percent"] for batch in batches])
    right_scores = np.asarray([right[batch]["vbench6_percent"] for batch in batches])
    difference = right_scores - left_scores
    difference_low, difference_high = percentile_interval(difference, indices)
    left_low, left_high = percentile_interval(left_scores, indices)
    right_low, right_high = percentile_interval(right_scores, indices)
    return {
        "batches": batches,
        "left_mean": float(left_scores.mean()),
        "left_ci_low": left_low,
        "left_ci_high": left_high,
        "right_mean": float(right_scores.mean()),
        "right_ci_low": right_low,
        "right_ci_high": right_high,
        "right_minus_left": float(difference.mean()),
        "difference_ci_low": difference_low,
        "difference_ci_high": difference_high,
        "right_wins": int(np.sum(difference > 0)),
        "left_wins": int(np.sum(difference < 0)),
        "ties": int(np.sum(difference == 0)),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--left-run", required=True)
    parser.add_argument("--right-run", required=True)
    parser.add_argument("--left-label", default="RI")
    parser.add_argument("--right-label", default="KEEPSAKE")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory to preserve previous results")
    args.output.mkdir(parents=True, exist_ok=True)

    left_rows, left_means, left_aggregate, left_source, left_manifest = load_run(
        args.root, args.left_run, args.limit
    )
    right_rows, right_means, right_aggregate, right_source, right_manifest = load_run(
        args.root, args.right_run, args.limit
    )
    paired = analyze_pair(
        left_rows, right_rows, samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )

    per_video = []
    left_by_batch = {row["batch_id"]: row for row in left_rows}
    right_by_batch = {row["batch_id"]: row for row in right_rows}
    for batch in paired["batches"]:
        left = left_by_batch[batch]
        right = right_by_batch[batch]
        row = {
            "batch_id": batch,
            "left_label": args.left_label,
            "right_label": args.right_label,
            "left_vbench6_percent": left["vbench6_percent"],
            "right_vbench6_percent": right["vbench6_percent"],
            "right_minus_left": right["vbench6_percent"] - left["vbench6_percent"],
        }
        for dimension in DIMENSIONS:
            row[f"left_{dimension}"] = left[dimension]
            row[f"right_{dimension}"] = right[dimension]
            row[f"delta_{dimension}"] = right[dimension] - left[dimension]
        per_video.append(row)
    write_csv(args.output / "per_video.csv", per_video)

    contributions = []
    for dimension in DIMENSIONS:
        delta = right_means[dimension] - left_means[dimension]
        contributions.append({
            "dimension": dimension,
            "left_mean": left_means[dimension],
            "right_mean": right_means[dimension],
            "right_minus_left_raw": delta,
            "right_minus_left_vbench6_points": (
                normalized_contribution(dimension, right_means[dimension])
                - normalized_contribution(dimension, left_means[dimension])
            ),
        })
    write_csv(args.output / "dimension_contributions.csv", contributions)

    summary = {
        "protocol": "paired_video_level_vbench6_v1",
        "definition": "normalized weighted aggregate of six custom-input VBench dimensions; not full official VBench",
        "limit": args.limit,
        "left": {
            "run": args.left_run,
            "label": args.left_label,
            "aggregate_from_run_means": left_aggregate,
            "source": str(left_source),
            "manifest": str(left_manifest),
            "dynamic_positive_videos": int(sum(row["dynamic_degree"] for row in left_rows)),
        },
        "right": {
            "run": args.right_run,
            "label": args.right_label,
            "aggregate_from_run_means": right_aggregate,
            "source": str(right_source),
            "manifest": str(right_manifest),
            "dynamic_positive_videos": int(sum(row["dynamic_degree"] for row in right_rows)),
        },
        "paired": paired,
        "spec": {
            dimension: {"minimum": low, "maximum": high, "weight": weight}
            for dimension, (low, high, weight) in SPEC.items()
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "summary.csv", [{
        "left_run": args.left_run,
        "right_run": args.right_run,
        "videos": args.limit,
        **paired,
    }])

    print(f"{args.left_label:<12} {paired['left_mean']:.3f} "
          f"[{paired['left_ci_low']:.3f}, {paired['left_ci_high']:.3f}]")
    print(f"{args.right_label:<12} {paired['right_mean']:.3f} "
          f"[{paired['right_ci_low']:.3f}, {paired['right_ci_high']:.3f}]")
    print(
        f"{args.right_label} - {args.left_label}: {paired['right_minus_left']:+.3f} "
        f"[{paired['difference_ci_low']:+.3f}, {paired['difference_ci_high']:+.3f}]"
    )
    print(
        f"Per-video wins: {args.right_label}={paired['right_wins']} "
        f"{args.left_label}={paired['left_wins']} ties={paired['ties']}"
    )
    print(
        f"Dynamic positives: {args.left_label}={summary['left']['dynamic_positive_videos']}/{args.limit}, "
        f"{args.right_label}={summary['right']['dynamic_positive_videos']}/{args.limit}"
    )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
