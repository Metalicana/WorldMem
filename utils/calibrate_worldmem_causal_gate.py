"""Calibrate WorldMem's causal-consistency admission gate from shadow traces.

The deployable score uses only generated pooled-latent descriptors, the actual
retrieved parent identities and camera geometry. Ground truth appears here only
to fit the pose-conditioned expectation and to label held-out frame quality.
"""

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.zeros(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def quality_auc(scores, bad_labels):
    """AUC where larger gate scores predict clean targets."""
    scores = np.asarray(scores, dtype=np.float64)
    bad = np.asarray(bad_labels, dtype=bool)
    clean = ~bad
    n_clean = int(clean.sum())
    n_bad = int(bad.sum())
    if n_clean == 0 or n_bad == 0:
        return None
    ranks = average_ranks(scores) + 1.0
    return float(
        (ranks[clean].sum() - n_clean * (n_clean + 1) / 2.0)
        / (n_clean * n_bad)
    )


def classification_metrics(scores, bad_labels, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    bad = np.asarray(bad_labels, dtype=bool)
    rejected = scores < float(threshold)
    tp = int(np.sum(rejected & bad))
    fp = int(np.sum(rejected & (~bad)))
    fn = int(np.sum((~rejected) & bad))
    tn = int(np.sum((~rejected) & (~bad)))
    return {
        "samples": int(len(scores)),
        "bad_targets": int(bad.sum()),
        "rejected_targets": int(rejected.sum()),
        "rejected_fraction": float(rejected.mean()) if len(scores) else None,
        "bad_precision": tp / (tp + fp) if tp + fp else None,
        "bad_recall": tp / (tp + fn) if tp + fn else None,
        "clean_false_reject_rate": fp / (fp + tn) if fp + tn else None,
    }


def fit_conservative_threshold(scores, bad_labels, max_clean_false_reject):
    scores = np.asarray(scores, dtype=np.float64)
    bad = np.asarray(bad_labels, dtype=bool)
    if len(scores) == 0 or not np.any(bad) or np.all(bad):
        raise ValueError("threshold fitting requires both clean and bad targets")
    unique = np.unique(scores)
    candidates = [float(unique[0] - 1e-12)]
    candidates.extend(float(0.5 * (a + b)) for a, b in zip(unique[:-1], unique[1:]))
    candidates.append(float(unique[-1] + 1e-12))
    best = None
    for threshold in candidates:
        metrics = classification_metrics(scores, bad, threshold)
        clean_fpr = float(metrics["clean_false_reject_rate"])
        if clean_fpr > float(max_clean_false_reject) + 1e-12:
            continue
        candidate = (float(metrics["bad_recall"]), -clean_fpr, threshold)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None or not math.isfinite(best[2]):
        raise ValueError("could not fit a finite causal-gate threshold")
    return float(best[2])


def trajectory_split(trajectory_ids, test_fraction, seed):
    trajectory_ids = sorted(set(trajectory_ids))
    if len(trajectory_ids) < 2:
        raise ValueError("at least two trajectories are required for calibration")
    shuffled = list(trajectory_ids)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    test_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * test_fraction)))
    return set(shuffled[test_count:]), set(shuffled[:test_count])


def median_positive(values):
    values = np.asarray(list(values), dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 1e-12)]
    return float(np.median(positive)) if positive.size else 1.0


def fit_binned_expectation(parent_rows, bins):
    distances = np.asarray([row["pose_distance"] for row in parent_rows], dtype=np.float64)
    values = np.asarray([row["gt_similarity"] for row in parent_rows], dtype=np.float64)
    if len(values) < int(bins) * 2:
        raise ValueError(f"need at least {int(bins) * 2} parent pairs for {bins} bins")
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)[1:-1]
    edges = np.unique(np.quantile(distances, quantiles))
    assignments = np.searchsorted(edges, distances, side="right")
    global_mean = float(values.mean())
    means = []
    counts = []
    for bin_index in range(len(edges) + 1):
        selected = values[assignments == bin_index]
        means.append(float(selected.mean()) if len(selected) else global_mean)
        counts.append(int(len(selected)))
    return {
        "edges": [float(value) for value in edges],
        "means": means,
        "counts": counts,
    }


def expected_similarity(expectation, distance):
    index = int(np.searchsorted(expectation["edges"], float(distance), side="right"))
    return float(expectation["means"][index])


def parse_trace_paths(values):
    paths = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            matches = sorted(glob.glob(str(Path(part).expanduser())))
            paths.extend(Path(match) for match in matches)
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    if not unique:
        raise FileNotFoundError("no shadow trace files matched --trace_paths")
    return unique


def load_observations(paths):
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event") != "causal_gate_observation":
                    continue
                if row.get("causal_gate_mode") != "shadow":
                    continue
                required = (
                    "target_generated_to_gt_similarity",
                    "target_generated_to_gt_mse",
                    "causal_gate_parents",
                )
                if any(row.get(field) is None for field in required):
                    continue
                row["trace_path"] = str(path)
                row["trajectory_id"] = (
                    f"{path.stem}:batch{int(row.get('global_batch_idx', 0)):05d}"
                )
                rows.append(row)
    if not rows:
        raise RuntimeError("shadow traces contain no usable causal_gate_observation events")
    return rows


def add_within_trajectory_labels(rows, bad_quantile):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["trajectory_id"]].append(row)
    for group in grouped.values():
        similarity = [row["target_generated_to_gt_similarity"] for row in group]
        negative_mse = [-row["target_generated_to_gt_mse"] for row in group]
        denominator = max(len(group) - 1, 1)
        composite = 0.5 * (
            average_ranks(similarity) + average_ranks(negative_mse)
        ) / denominator
        cutoff = float(np.quantile(composite, float(bad_quantile)))
        for row, percentile in zip(group, composite):
            row["gt_quality_percentile"] = float(percentile)
            row["gt_bad_target"] = bool(percentile <= cutoff)


def add_pose_model(rows, train_trajectories, bins):
    train_parents = []
    for row in rows:
        if row["trajectory_id"] not in train_trajectories:
            continue
        train_parents.extend(row["causal_gate_parents"])
    pose_scales = {
        "translation": median_positive(row["translation"] for row in train_parents),
        "rotation_rad": median_positive(row["rotation_rad"] for row in train_parents),
    }
    for row in rows:
        for parent in row["causal_gate_parents"]:
            parent["pose_distance"] = float(
                float(parent["translation"]) / pose_scales["translation"]
                + float(parent["rotation_rad"]) / pose_scales["rotation_rad"]
            )
    train_parents = [
        parent
        for row in rows
        if row["trajectory_id"] in train_trajectories
        for parent in row["causal_gate_parents"]
    ]
    expectation = fit_binned_expectation(train_parents, bins)
    for row in rows:
        residuals = []
        weights = []
        for parent in row["causal_gate_parents"]:
            expected = expected_similarity(expectation, parent["pose_distance"])
            parent["expected_similarity"] = expected
            parent["residual"] = float(parent["generated_similarity"]) - expected
            residuals.append(parent["residual"])
            weights.append(float(parent["weight"]))
        row["causal_gate_score"] = float(np.dot(weights, residuals))
    return pose_scales, expectation


def write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def serializable_observation(row):
    return {
        key: value
        for key, value in row.items()
        if key != "causal_gate_parents"
    }


def make_plot(path, train_rows, test_rows, threshold):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, title, rows in zip(
        axes, ("Calibration trajectories", "Held-out trajectories"), (train_rows, test_rows)
    ):
        clean = [row["causal_gate_score"] for row in rows if not row["gt_bad_target"]]
        bad = [row["causal_gate_score"] for row in rows if row["gt_bad_target"]]
        axis.hist(clean, bins=35, alpha=0.7, label="clean", color="#287271")
        axis.hist(bad, bins=35, alpha=0.7, label="bottom 20%", color="#d1495b")
        axis.axvline(threshold, color="black", linestyle="--", linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("causal consistency residual")
        axis.set_ylabel("targets")
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_paths", nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pose_bins", type=int, default=4)
    parser.add_argument("--bad_quantile", type=float, default=0.20)
    parser.add_argument("--test_fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--split_seed", type=int, default=17)
    parser.add_argument("--max_train_clean_false_reject", type=float, default=0.10)
    parser.add_argument("--min_heldout_auc", type=float, default=0.70)
    parser.add_argument("--min_bad_precision", type=float, default=0.50)
    parser.add_argument("--min_bad_recall", type=float, default=0.20)
    parser.add_argument("--max_heldout_clean_false_reject", type=float, default=0.15)
    args = parser.parse_args()

    paths = parse_trace_paths(args.trace_paths)
    rows = load_observations(paths)
    add_within_trajectory_labels(rows, args.bad_quantile)
    train_ids, test_ids = trajectory_split(
        [row["trajectory_id"] for row in rows], args.test_fraction, args.split_seed
    )
    pose_scales, expectation = add_pose_model(rows, train_ids, args.pose_bins)
    train = [row for row in rows if row["trajectory_id"] in train_ids]
    test = [row for row in rows if row["trajectory_id"] in test_ids]
    threshold = fit_conservative_threshold(
        [row["causal_gate_score"] for row in train],
        [row["gt_bad_target"] for row in train],
        args.max_train_clean_false_reject,
    )
    train_metrics = classification_metrics(
        [row["causal_gate_score"] for row in train],
        [row["gt_bad_target"] for row in train],
        threshold,
    )
    test_metrics = classification_metrics(
        [row["causal_gate_score"] for row in test],
        [row["gt_bad_target"] for row in test],
        threshold,
    )
    test_auc = quality_auc(
        [row["causal_gate_score"] for row in test],
        [row["gt_bad_target"] for row in test],
    )
    checks = {
        "heldout_auc": test_auc is not None and test_auc >= args.min_heldout_auc,
        "heldout_bad_precision": (
            test_metrics["bad_precision"] is not None
            and test_metrics["bad_precision"] >= args.min_bad_precision
        ),
        "heldout_bad_recall": (
            test_metrics["bad_recall"] is not None
            and test_metrics["bad_recall"] >= args.min_bad_recall
        ),
        "heldout_clean_false_reject": (
            test_metrics["clean_false_reject_rate"] is not None
            and test_metrics["clean_false_reject_rate"]
            <= args.max_heldout_clean_false_reject
        ),
    }
    calibration = {
        "version": 1,
        "feature_backend": "latent",
        "threshold": threshold,
        "pose_scales": pose_scales,
        "expectation": expectation,
        "deployment_approved": bool(all(checks.values())),
        "checks": checks,
        "train_trajectory_ids": sorted(train_ids),
        "test_trajectory_ids": sorted(test_ids),
        "train_metrics": train_metrics,
        "heldout_metrics": {"quality_auc": test_auc, **test_metrics},
        "config": vars(args) | {"trace_paths": [str(path) for path in paths]},
    }
    calibration["config"]["output_dir"] = str(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    write_csv(
        args.output_dir / "scored_observations.csv",
        [serializable_observation(row) for row in rows],
    )
    make_plot(args.output_dir / "heldout_validation.png", train, test, threshold)

    print("WorldMem causal-gate calibration")
    print(f"Shadow observations: {len(rows)}")
    print(f"Train trajectories: {len(train_ids)}")
    print(f"Held-out trajectories: {len(test_ids)}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Held-out AUC: {test_auc:.4f}" if test_auc is not None else "Held-out AUC: n/a")
    print(f"Held-out bad precision: {test_metrics['bad_precision']}")
    print(f"Held-out bad recall: {test_metrics['bad_recall']}")
    print(f"Held-out clean false reject: {test_metrics['clean_false_reject_rate']}")
    print(f"Deployment approved: {calibration['deployment_approved']}")
    print(f"Wrote: {args.output_dir / 'calibration.json'}")


if __name__ == "__main__":
    main()
