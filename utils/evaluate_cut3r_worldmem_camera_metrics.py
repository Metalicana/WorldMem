import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from worldmem_eval_common import mean_or_none, parse_csv, safe_round, write_json


def parse_rows(value):
    if not value:
        return None
    rows = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            rows.update(range(int(start_text), int(end_text) + 1))
        else:
            rows.add(int(part))
    return rows


def percentile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


def numeric_stats(prefix, values):
    values = [value for value in values if value is not None]
    return {
        f"{prefix}_mean": safe_round(mean_or_none(values)),
        f"{prefix}_median": safe_round(percentile(values, 0.5)),
        f"{prefix}_p90": safe_round(percentile(values, 0.9)),
        f"{prefix}_max": safe_round(max(values)) if values else None,
    }


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_predicted_poses(reconstruction_dir):
    camera_dir = reconstruction_dir / "camera"
    paths = sorted(camera_dir.glob("*.npz"))
    poses = []
    for path in paths:
        with np.load(path) as data:
            poses.append(np.asarray(data["pose"], dtype=np.float64))
    if not poses:
        raise RuntimeError(f"No CUT3R camera poses found under {camera_dir}")
    return np.stack(poses, axis=0)


def relative_poses(c2ws):
    first_inv = np.linalg.inv(c2ws[0])
    return np.stack([first_inv @ pose for pose in c2ws], axis=0)


def rotation_error_deg(pred_rot, gt_rot):
    delta = pred_rot @ np.swapaxes(gt_rot, -1, -2)
    traces = np.trace(delta, axis1=1, axis2=2)
    cos_values = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_values))


def path_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def fit_scalar(pred_points, gt_points):
    denominator = float(np.sum(pred_points * pred_points))
    if denominator <= 1e-12:
        return 1.0
    return float(np.sum(gt_points * pred_points) / denominator)


def fit_umeyama(pred_points, gt_points):
    if len(pred_points) < 2:
        return pred_points.copy(), 1.0

    pred_mean = pred_points.mean(axis=0)
    gt_mean = gt_points.mean(axis=0)
    pred_centered = pred_points - pred_mean
    gt_centered = gt_points - gt_mean

    variance = float(np.mean(np.sum(pred_centered * pred_centered, axis=1)))
    if variance <= 1e-12:
        return pred_points.copy(), 1.0

    covariance = (gt_centered.T @ pred_centered) / len(pred_points)
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(u_mat) * np.linalg.det(vt_mat))
    correction = np.eye(3)
    correction[-1, -1] = sign if sign != 0 else 1.0
    rotation = u_mat @ correction @ vt_mat
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    aligned = (scale * (rotation @ pred_centered.T)).T + gt_mean
    return aligned, scale


def worldscore_camera_control_score(rotation_error_mean, translation_error_mean):
    rotation_component = 1.0 - min(max(rotation_error_mean, 0.0), 15.0) / 15.0
    translation_component = 1.0 - min(max(translation_error_mean, 0.0), 0.5) / 0.5
    return 100.0 * math.sqrt(max(rotation_component, 0.0) * max(translation_component, 0.0))


def score_reconstruction(reconstruction_dir, metadata):
    pred = load_predicted_poses(reconstruction_dir)
    gt = np.load(metadata["gt_c2w_path"]).astype(np.float64)

    count = min(len(pred), len(gt))
    pred = pred[:count]
    gt = gt[:count]
    if count < 2:
        raise RuntimeError(f"Need at least two poses to score {reconstruction_dir}")

    pred_rel = relative_poses(pred)
    gt_rel = relative_poses(gt)
    rot_errors = rotation_error_deg(pred_rel[:, :3, :3], gt_rel[:, :3, :3])

    pred_points = pred_rel[:, :3, 3]
    gt_points = gt_rel[:, :3, 3]

    scalar = fit_scalar(pred_points, gt_points)
    pred_scaled = scalar * pred_points
    scale_only_errors = np.linalg.norm(pred_scaled - gt_points, axis=1)
    pred_sim3, sim3_scale = fit_umeyama(pred_points, gt_points)
    sim3_errors = np.linalg.norm(pred_sim3 - gt_points, axis=1)

    rotation_mean = float(rot_errors.mean())
    translation_mean = float(scale_only_errors.mean())
    gt_path_length = path_length(gt_points)
    pred_path_length_scaled = path_length(pred_scaled)
    pred_endpoint = float(np.linalg.norm(pred_scaled[-1] - pred_scaled[0]))
    gt_endpoint = float(np.linalg.norm(gt_points[-1] - gt_points[0]))

    return {
        "run_name": metadata["run_name"],
        "batch_idx": metadata["batch_idx"],
        "duration_sec": metadata["duration_sec"],
        "sampled_frames": count,
        "frame_stride": metadata.get("frame_stride"),
        "total_video_frames": metadata.get("total_video_frames"),
        "rotation_error_deg_mean": safe_round(rotation_mean),
        "rotation_error_deg_median": safe_round(percentile(rot_errors, 0.5)),
        "rotation_error_deg_p90": safe_round(percentile(rot_errors, 0.9)),
        "rotation_error_deg_max": safe_round(float(rot_errors.max())),
        "translation_error_scale_only_mean": safe_round(translation_mean),
        "translation_error_scale_only_median": safe_round(percentile(scale_only_errors, 0.5)),
        "translation_error_scale_only_p90": safe_round(percentile(scale_only_errors, 0.9)),
        "translation_error_scale_only_max": safe_round(float(scale_only_errors.max())),
        "translation_error_sim3_mean": safe_round(float(sim3_errors.mean())),
        "translation_error_sim3_median": safe_round(percentile(sim3_errors, 0.5)),
        "translation_error_sim3_p90": safe_round(percentile(sim3_errors, 0.9)),
        "translation_error_sim3_max": safe_round(float(sim3_errors.max())),
        "endpoint_rotation_error_deg": safe_round(float(rot_errors[-1])),
        "endpoint_translation_error_scale_only": safe_round(float(scale_only_errors[-1])),
        "endpoint_translation_error_sim3": safe_round(float(sim3_errors[-1])),
        "gt_path_length": safe_round(gt_path_length),
        "pred_path_length_scale_only": safe_round(pred_path_length_scaled),
        "pred_path_length_sim3": safe_round(path_length(pred_sim3)),
        "path_length_ratio_scale_only": safe_round(
            pred_path_length_scaled / gt_path_length if gt_path_length > 1e-12 else None
        ),
        "loop_gt_endpoint_distance": safe_round(gt_endpoint),
        "loop_pred_endpoint_distance_scale_only": safe_round(pred_endpoint),
        "loop_endpoint_distance_error": safe_round(abs(pred_endpoint - gt_endpoint)),
        "scale_only_factor": safe_round(scalar),
        "sim3_scale": safe_round(sim3_scale),
        "worldscore_camera_control_score": safe_round(
            worldscore_camera_control_score(rotation_mean, translation_mean)
        ),
        "reconstruction_dir": str(reconstruction_dir),
    }


def discover_metadata_files(cut3r_dir, runs=None):
    metadata_files = sorted(Path(cut3r_dir).glob("*/*/video_batch*/metadata.json"))
    if runs is None:
        return metadata_files
    run_set = set(runs)
    return [path for path in metadata_files if path.parents[2].name in run_set]


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_run(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["run_name"]].append(row)

    metric_fields = [
        "rotation_error_deg_mean",
        "rotation_error_deg_p90",
        "translation_error_scale_only_mean",
        "translation_error_scale_only_p90",
        "translation_error_sim3_mean",
        "translation_error_sim3_p90",
        "endpoint_rotation_error_deg",
        "endpoint_translation_error_scale_only",
        "loop_endpoint_distance_error",
        "worldscore_camera_control_score",
    ]
    output = []
    for run_name, group in sorted(grouped.items()):
        row = {
            "run_name": run_name,
            "videos": len(group),
            "sampled_frames_mean": safe_round(mean_or_none(item["sampled_frames"] for item in group)),
        }
        for field in metric_fields:
            row.update(numeric_stats(field, [item.get(field) for item in group]))
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Score CUT3R camera trajectories for WorldMem generated videos."
    )
    parser.add_argument("--cut3r_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--runs", type=str, default=None)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--duration_sec", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_filter = parse_csv(args.runs)
    row_filter = parse_rows(args.rows)

    rows = []
    failures = []
    for metadata_path in discover_metadata_files(args.cut3r_dir, runs=run_filter):
        metadata = load_json(metadata_path)
        if metadata.get("status") != "completed":
            continue
        if row_filter is not None and int(metadata["batch_idx"]) not in row_filter:
            continue
        if args.duration_sec is not None and int(metadata["duration_sec"]) != args.duration_sec:
            continue
        try:
            rows.append(score_reconstruction(metadata_path.parent, metadata))
        except Exception as exc:
            failures.append(
                {
                    "metadata_path": str(metadata_path),
                    "run_name": metadata.get("run_name"),
                    "batch_idx": metadata.get("batch_idx"),
                    "error": repr(exc),
                }
            )

    if not rows:
        raise RuntimeError(f"No CUT3R reconstructions scored under {args.cut3r_dir}")

    summary = summarize_by_run(rows)
    write_csv(args.output_dir / "cut3r_camera_metrics.csv", rows)
    write_csv(args.output_dir / "cut3r_camera_summary.csv", summary)
    write_json(args.output_dir / "cut3r_camera_metrics.json", rows)
    write_json(args.output_dir / "cut3r_camera_summary.json", summary)
    write_json(args.output_dir / "cut3r_camera_failures.json", failures)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {args.output_dir / 'cut3r_camera_summary.csv'}")


if __name__ == "__main__":
    main()
