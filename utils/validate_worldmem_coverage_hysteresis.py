"""Test whether older generated representatives beat later covered rewrites.

This is an offline validation on completed unbounded WorldMem videos. Runtime
policy code must not consume the GT quality values produced here.
"""

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from worldmem_eval_common import (
    list_prediction_videos,
    load_pose_c2ws_for_batch,
    read_worldmem_gt_frames,
    video_frame_count,
    write_json,
)


def parse_float_csv(value):
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def discover_trace_paths(run_dir, patterns=None):
    patterns = patterns or [str(Path(run_dir) / "access_traces" / "*.jsonl")]
    paths = []
    for pattern in patterns:
        paths.extend(Path(path) for path in sorted(glob.glob(str(pattern))))
    paths = sorted(set(path.resolve() for path in paths))
    if not paths:
        raise FileNotFoundError(
            f"No access traces found for {run_dir}. Actual chunk indexing is "
            "required; this validator will not assume MemCam's chunk stride."
        )
    return paths


def load_chunk_samples(trace_paths, default_context_frames=600):
    """Return one sampled generated frame per actual WorldMem chunk."""
    context_by_batch = {}
    chunks_by_batch = defaultdict(dict)
    for path in trace_paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[warn] malformed trace line {path}:{line_number}: {exc}")
                    continue
                batch_idx = row.get("global_batch_idx")
                if batch_idx is None:
                    continue
                batch_idx = int(batch_idx)
                if row.get("event") == "memory_run_start":
                    context_by_batch[batch_idx] = int(
                        row.get("context_frames", default_context_frames)
                    )
                if (
                    row.get("event") == "memory_retrieval"
                    and int(row.get("context_slot", 0)) == 0
                ):
                    target = int(row["target_frame"])
                    horizon = int(row.get("target_horizon", 1))
                    chunks_by_batch[batch_idx][target] = horizon

    output = {}
    for batch_idx, chunks in chunks_by_batch.items():
        context_frames = context_by_batch.get(batch_idx, int(default_context_frames))
        samples = []
        for chunk_index, (target_frame, horizon) in enumerate(sorted(chunks.items())):
            if horizon < 1:
                raise ValueError(
                    f"Invalid target_horizon={horizon} for batch {batch_idx}"
                )
            local_start = int(target_frame - context_frames)
            if local_start < 0:
                continue
            samples.append(
                {
                    "chunk_index": int(chunk_index),
                    "target_frame": int(target_frame),
                    "target_horizon": int(horizon),
                    # The final frame represents the completed output of a
                    # multi-frame chunk. Current WorldMem uses horizon one.
                    "generated_frame": int(local_start + horizon - 1),
                }
            )
        output[batch_idx] = {
            "context_frames": context_frames,
            "samples": samples,
        }
    return output


def worldmem_fov_similarity(
    c2ws,
    query_indices,
    memory_indices,
    radius=30.0,
    fov_half_h=52.5,
    fov_half_v=37.5,
):
    """Physical FOV affinity adapted to WorldMem's +z-forward convention."""
    query_indices = list(query_indices)
    memory_indices = list(memory_indices)
    if not query_indices or not memory_indices:
        return np.zeros((len(query_indices), len(memory_indices)), dtype=np.float64)
    if radius <= 0 or fov_half_h <= 0 or fov_half_v <= 0:
        raise ValueError("radius and FOV half angles must be positive")

    c2ws = np.asarray(c2ws, dtype=np.float64)
    query = c2ws[query_indices]
    memory = c2ws[memory_indices]
    position_distance = np.linalg.norm(
        query[:, None, :3, 3] - memory[None, :, :3, 3], axis=-1
    )
    position_similarity = np.clip(
        1.0 - position_distance / (2.0 * float(radius)), 0.0, 1.0
    )

    query_forward = query[:, :3, 2]
    memory_forward = memory[:, :3, 2]
    query_forward /= np.maximum(
        np.linalg.norm(query_forward, axis=1, keepdims=True), 1e-12
    )
    memory_forward /= np.maximum(
        np.linalg.norm(memory_forward, axis=1, keepdims=True), 1e-12
    )
    query_yaw = np.arctan2(query_forward[:, 0], query_forward[:, 2])
    memory_yaw = np.arctan2(memory_forward[:, 0], memory_forward[:, 2])
    yaw_distance = np.abs(query_yaw[:, None] - memory_yaw[None, :])
    yaw_distance = np.minimum(yaw_distance, 2.0 * np.pi - yaw_distance)
    query_pitch = np.arctan2(
        query_forward[:, 1],
        np.linalg.norm(query_forward[:, [0, 2]], axis=1),
    )
    memory_pitch = np.arctan2(
        memory_forward[:, 1],
        np.linalg.norm(memory_forward[:, [0, 2]], axis=1),
    )
    pitch_distance = np.abs(query_pitch[:, None] - memory_pitch[None, :])

    horizontal = np.clip(
        1.0 - yaw_distance / (2.0 * np.deg2rad(float(fov_half_h))),
        0.0,
        1.0,
    )
    vertical = np.clip(
        1.0 - pitch_distance / (2.0 * np.deg2rad(float(fov_half_v))),
        0.0,
        1.0,
    )
    return position_similarity * horizontal * vertical


def find_oldest_covered_pairs(samples, c2ws, thresholds, min_chunk_separation, **geometry):
    frame_indices = [int(sample["generated_frame"]) for sample in samples]
    affinity = worldmem_fov_similarity(
        c2ws,
        frame_indices,
        frame_indices,
        **geometry,
    )
    pairs = []
    for threshold in thresholds:
        for later_row, later in enumerate(samples):
            eligible = [
                earlier_row
                for earlier_row, earlier in enumerate(samples[:later_row])
                if int(later["chunk_index"]) - int(earlier["chunk_index"])
                >= int(min_chunk_separation)
                and affinity[later_row, earlier_row] >= float(threshold)
            ]
            if not eligible:
                continue
            earlier_row = eligible[0]
            earlier = samples[earlier_row]
            pairs.append(
                {
                    "coverage_threshold": float(threshold),
                    "earlier_chunk_index": int(earlier["chunk_index"]),
                    "later_chunk_index": int(later["chunk_index"]),
                    "earlier_generated_frame": int(earlier["generated_frame"]),
                    "later_generated_frame": int(later["generated_frame"]),
                    "temporal_gap_chunks": int(
                        later["chunk_index"] - earlier["chunk_index"]
                    ),
                    "temporal_gap_frames": int(
                        later["generated_frame"] - earlier["generated_frame"]
                    ),
                    "geometric_affinity": float(affinity[later_row, earlier_row]),
                }
            )
    return pairs


def _gaussian_window(torch, window_size, sigma, channels, device):
    coordinates = torch.arange(window_size, device=device, dtype=torch.float32)
    coordinates -= (window_size - 1) / 2
    kernel = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    kernel = torch.outer(kernel, kernel)
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


class FrameQualityRunner:
    def __init__(self, device="cuda", batch_size=64, psnr_cap=100.0):
        import torch

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested for quality metrics but unavailable; using CPU.")
            device = "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = max(int(batch_size), 1)
        self.psnr_cap = float(psnr_cap)

    def _ssim(self, prediction, target):
        torch = self.torch
        channels = prediction.shape[1]
        window_size = min(11, min(prediction.shape[-2:]))
        if window_size % 2 == 0:
            window_size -= 1
        window_size = max(window_size, 1)
        window = _gaussian_window(
            torch,
            window_size,
            1.5 * window_size / 11,
            channels,
            prediction.device,
        )
        padding = window_size // 2
        conv = torch.nn.functional.conv2d
        mu_pred = conv(prediction, window, padding=padding, groups=channels)
        mu_gt = conv(target, window, padding=padding, groups=channels)
        mu_pred_sq = mu_pred.square()
        mu_gt_sq = mu_gt.square()
        mu_cross = mu_pred * mu_gt
        sigma_pred = (
            conv(prediction.square(), window, padding=padding, groups=channels)
            - mu_pred_sq
        ).clamp_min(0)
        sigma_gt = (
            conv(target.square(), window, padding=padding, groups=channels)
            - mu_gt_sq
        ).clamp_min(0)
        sigma_cross = conv(
            prediction * target, window, padding=padding, groups=channels
        ) - mu_cross
        numerator = (2 * mu_cross + 0.01**2) * (2 * sigma_cross + 0.03**2)
        denominator = (mu_pred_sq + mu_gt_sq + 0.01**2) * (
            sigma_pred + sigma_gt + 0.03**2
        )
        return (numerator / denominator.clamp_min(1e-12)).flatten(1).mean(1)

    def compute(self, prediction_frames, gt_frames, frame_indices):
        torch = self.torch
        output = {}
        frame_indices = sorted(set(int(index) for index in frame_indices))
        with torch.inference_mode():
            for start in range(0, len(frame_indices), self.batch_size):
                indices = frame_indices[start : start + self.batch_size]
                pred = np.stack([prediction_frames[index] for index in indices])
                gt = np.stack([gt_frames[index] for index in indices])
                pred = (
                    torch.from_numpy(pred.astype(np.float32) / 255.0)
                    .permute(0, 3, 1, 2)
                    .to(self.device)
                )
                gt = (
                    torch.from_numpy(gt.astype(np.float32) / 255.0)
                    .permute(0, 3, 1, 2)
                    .to(self.device)
                )
                if gt.shape[-2:] != pred.shape[-2:]:
                    gt = torch.nn.functional.interpolate(
                        gt,
                        size=pred.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                mse = (pred - gt).square().flatten(1).mean(1)
                psnr = -10 * torch.log10(
                    mse.clamp_min(10 ** (-self.psnr_cap / 10))
                )
                ssim = self._ssim(pred, gt)
                for offset, frame_index in enumerate(indices):
                    output[frame_index] = {
                        "psnr": float(psnr[offset].clamp_max(self.psnr_cap).cpu()),
                        "ssim": float(ssim[offset].cpu()),
                    }
        return output


def read_prediction_frames(video_path, indices):
    from worldmem_eval_common import read_video_frames

    return read_video_frames(video_path, indices)


def bootstrap_trajectory_means(values, samples=10000, seed=0):
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(values):
        return None, None, None
    if len(values) == 1:
        value = float(values[0])
        return value, value, value
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def write_csv(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_pairs(pair_rows, thresholds, bootstrap_samples, seed):
    trajectory_rows = []
    summary_rows = []
    for threshold in thresholds:
        threshold_rows = [
            row for row in pair_rows if row["coverage_threshold"] == float(threshold)
        ]
        for subset, subset_rows in (
            ("all", threshold_rows),
            ("final_quarter", [row for row in threshold_rows if row["is_final_quarter"]]),
        ):
            by_trajectory = defaultdict(list)
            for row in subset_rows:
                by_trajectory[int(row["batch_idx"])].append(row)
            for batch_idx, rows in sorted(by_trajectory.items()):
                trajectory_rows.append(
                    {
                        "coverage_threshold": float(threshold),
                        "subset": subset,
                        "batch_idx": batch_idx,
                        "pairs": len(rows),
                        "psnr_delta_mean": float(np.mean([row["psnr_delta"] for row in rows])),
                        "ssim_delta_mean": float(np.mean([row["ssim_delta"] for row in rows])),
                        "older_psnr_win_fraction": float(
                            np.mean([row["psnr_delta"] > 0 for row in rows])
                        ),
                        "older_ssim_win_fraction": float(
                            np.mean([row["ssim_delta"] > 0 for row in rows])
                        ),
                    }
                )
            matching_trajectories = [
                row
                for row in trajectory_rows
                if row["coverage_threshold"] == float(threshold)
                and row["subset"] == subset
            ]
            psnr_estimate, psnr_low, psnr_high = bootstrap_trajectory_means(
                [row["psnr_delta_mean"] for row in matching_trajectories],
                samples=bootstrap_samples,
                seed=seed + int(round(float(threshold) * 1000)),
            )
            ssim_estimate, ssim_low, ssim_high = bootstrap_trajectory_means(
                [row["ssim_delta_mean"] for row in matching_trajectories],
                samples=bootstrap_samples,
                seed=seed + 10000 + int(round(float(threshold) * 1000)),
            )
            summary_rows.append(
                {
                    "coverage_threshold": float(threshold),
                    "subset": subset,
                    "pairs": len(subset_rows),
                    "trajectories": len(matching_trajectories),
                    "psnr_delta_trajectory_mean": psnr_estimate,
                    "psnr_delta_ci_low": psnr_low,
                    "psnr_delta_ci_high": psnr_high,
                    "ssim_delta_trajectory_mean": ssim_estimate,
                    "ssim_delta_ci_low": ssim_low,
                    "ssim_delta_ci_high": ssim_high,
                    "trajectories_psnr_positive": sum(
                        row["psnr_delta_mean"] > 0 for row in matching_trajectories
                    ),
                    "trajectories_ssim_positive": sum(
                        row["ssim_delta_mean"] > 0 for row in matching_trajectories
                    ),
                    "pair_weighted_psnr_delta": (
                        float(np.mean([row["psnr_delta"] for row in subset_rows]))
                        if subset_rows
                        else None
                    ),
                    "pair_weighted_ssim_delta": (
                        float(np.mean([row["ssim_delta"] for row in subset_rows]))
                        if subset_rows
                        else None
                    ),
                    "median_temporal_gap_chunks": (
                        float(np.median([row["temporal_gap_chunks"] for row in subset_rows]))
                        if subset_rows
                        else None
                    ),
                }
            )
    return trajectory_rows, summary_rows


def make_plot(path, summary_rows):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    colors = {"all": "#287271", "final_quarter": "#d1495b"}
    labels = {"all": "all 60s", "final_quarter": "45-60s"}
    for metric, axis in zip(("psnr", "ssim"), axes[:2]):
        for subset in ("all", "final_quarter"):
            rows = sorted(
                [row for row in summary_rows if row["subset"] == subset],
                key=lambda row: row["coverage_threshold"],
            )
            x = np.asarray([row["coverage_threshold"] for row in rows])
            y = np.asarray(
                [
                    np.nan if row[f"{metric}_delta_trajectory_mean"] is None
                    else row[f"{metric}_delta_trajectory_mean"]
                    for row in rows
                ],
                dtype=np.float64,
            )
            low = np.asarray(
                [
                    np.nan if row[f"{metric}_delta_ci_low"] is None
                    else row[f"{metric}_delta_ci_low"]
                    for row in rows
                ],
                dtype=np.float64,
            )
            high = np.asarray(
                [
                    np.nan if row[f"{metric}_delta_ci_high"] is None
                    else row[f"{metric}_delta_ci_high"]
                    for row in rows
                ],
                dtype=np.float64,
            )
            valid = np.isfinite(y) & np.isfinite(low) & np.isfinite(high)
            axis.errorbar(
                x[valid],
                y[valid],
                yerr=np.vstack((y[valid] - low[valid], high[valid] - y[valid])),
                marker="o",
                linewidth=2,
                capsize=3,
                color=colors[subset],
                label=labels[subset],
            )
        axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
        axis.set_xlabel("coverage threshold")
        axis.set_ylabel(f"older - later {metric.upper()}")
        axis.legend(frameon=False)
    for subset in ("all", "final_quarter"):
        rows = sorted(
            [row for row in summary_rows if row["subset"] == subset],
            key=lambda row: row["coverage_threshold"],
        )
        axes[2].plot(
            [row["coverage_threshold"] for row in rows],
            [row["pairs"] for row in rows],
            marker="o",
            linewidth=2,
            color=colors[subset],
            label=labels[subset],
        )
    axes[2].set_xlabel("coverage threshold")
    axes[2].set_ylabel("matched rewrite events")
    axes[2].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/minecraft"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--trace_paths", nargs="*")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--context_frames", type=int, default=600)
    parser.add_argument("--future_seconds", type=int, default=60)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--initial_skip_frames", type=int, default=100)
    parser.add_argument("--coverage_thresholds", default="0.80,0.85,0.90,0.95")
    parser.add_argument("--min_chunk_separation", type=int, default=2)
    parser.add_argument("--radius", type=float, default=30.0)
    parser.add_argument("--fov_half_h", type=float, default=52.5)
    parser.add_argument("--fov_half_v", type=float, default=37.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric_batch_size", type=int, default=64)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=17)
    args = parser.parse_args()

    thresholds = sorted(set(parse_float_csv(args.coverage_thresholds)))
    if not thresholds or min(thresholds) < 0 or max(thresholds) > 1:
        raise ValueError("coverage thresholds must be in [0, 1]")
    if args.min_chunk_separation < 2:
        raise ValueError("min_chunk_separation must be at least two")
    expected_frames = int(args.future_seconds * args.fps)
    trace_paths = discover_trace_paths(args.run_dir, args.trace_paths)
    chunks = load_chunk_samples(trace_paths, args.context_frames)
    prediction_videos = list_prediction_videos(args.run_dir, limit=args.limit)
    quality_runner = FrameQualityRunner(args.device, args.metric_batch_size)

    pair_rows = []
    video_rows = []
    for batch_idx, prediction_path in prediction_videos:
        if batch_idx not in chunks:
            raise RuntimeError(f"No chunk trace found for generated batch {batch_idx}")
        samples = [
            row
            for row in chunks[batch_idx]["samples"]
            if int(row["generated_frame"]) < expected_frames
        ]
        if not samples:
            raise RuntimeError(f"No generated chunks found for batch {batch_idx}")
        generated_indices = [row["generated_frame"] for row in samples]
        if video_frame_count(prediction_path) <= max(generated_indices):
            raise RuntimeError(f"Prediction video is shorter than its trace: {prediction_path}")
        c2ws, dataset_video_path = load_pose_c2ws_for_batch(
            data_dir=args.data_dir,
            batch_idx=batch_idx,
            generated_frame_indices=generated_indices,
            seed=args.dataset_seed,
            context_frames=chunks[batch_idx]["context_frames"],
            initial_skip_frames=args.initial_skip_frames,
            n_frames_valid=chunks[batch_idx]["context_frames"] + expected_frames,
        )
        # c2ws is ordered like generated_indices; the matcher indexes this
        # compact array, so remap sampled generated frame IDs to local rows.
        compact_samples = [
            {**row, "generated_frame": local_index}
            for local_index, row in enumerate(samples)
        ]
        pairs = find_oldest_covered_pairs(
            compact_samples,
            c2ws,
            thresholds,
            args.min_chunk_separation,
            radius=args.radius,
            fov_half_h=args.fov_half_h,
            fov_half_v=args.fov_half_v,
        )
        for row in pairs:
            row["earlier_generated_frame"] = generated_indices[
                row["earlier_generated_frame"]
            ]
            row["later_generated_frame"] = generated_indices[
                row["later_generated_frame"]
            ]
            row["temporal_gap_frames"] = (
                row["later_generated_frame"] - row["earlier_generated_frame"]
            )
        needed = sorted(
            {
                frame
                for row in pairs
                for frame in (
                    row["earlier_generated_frame"],
                    row["later_generated_frame"],
                )
            }
        )
        prediction_frames = read_prediction_frames(prediction_path, needed)
        gt_frames, _ = read_worldmem_gt_frames(
            data_dir=args.data_dir,
            batch_idx=batch_idx,
            generated_frame_indices=needed,
            seed=args.dataset_seed,
            context_frames=chunks[batch_idx]["context_frames"],
            initial_skip_frames=args.initial_skip_frames,
        )
        missing = sorted(
            (set(needed) - set(prediction_frames))
            | (set(needed) - set(gt_frames))
        )
        if missing:
            raise RuntimeError(f"Missing prediction/GT frames for batch {batch_idx}: {missing[:10]}")
        quality = quality_runner.compute(
            prediction_frames, gt_frames, needed
        ) if needed else {}
        final_quarter_start = int(math.floor(0.75 * expected_frames))
        for row in pairs:
            earlier = row["earlier_generated_frame"]
            later = row["later_generated_frame"]
            pair_rows.append(
                {
                    "batch_idx": int(batch_idx),
                    "prediction_video": str(prediction_path),
                    "dataset_video": str(dataset_video_path),
                    **row,
                    "later_seconds": float(later / args.fps),
                    "is_final_quarter": bool(later >= final_quarter_start),
                    "earlier_psnr": quality[earlier]["psnr"],
                    "later_psnr": quality[later]["psnr"],
                    "psnr_delta": quality[earlier]["psnr"] - quality[later]["psnr"],
                    "earlier_ssim": quality[earlier]["ssim"],
                    "later_ssim": quality[later]["ssim"],
                    "ssim_delta": quality[earlier]["ssim"] - quality[later]["ssim"],
                }
            )
        video_rows.append(
            {
                "batch_idx": int(batch_idx),
                "prediction_video": str(prediction_path),
                "dataset_video": str(dataset_video_path),
                "actual_chunks": len(samples),
                "unique_chunk_horizons": ",".join(
                    str(value) for value in sorted({row["target_horizon"] for row in samples})
                ),
                **{
                    f"pairs_threshold_{threshold:.2f}": sum(
                        row["coverage_threshold"] == threshold for row in pairs
                    )
                    for threshold in thresholds
                },
            }
        )
        print(
            f"[coverage] batch={batch_idx:02d} chunks={len(samples)} "
            + " ".join(
                f"t{threshold:.2f}={sum(row['coverage_threshold'] == threshold for row in pairs)}"
                for threshold in thresholds
            )
        )

    trajectory_rows, summary_rows = summarize_pairs(
        pair_rows,
        thresholds,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pair_details.csv", pair_rows)
    write_csv(args.output_dir / "trajectory_summary.csv", trajectory_rows)
    write_csv(args.output_dir / "video_inventory.csv", video_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_json(
        args.output_dir / "summary.json",
        {
            "config": {
                **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "trace_paths": [str(path) for path in trace_paths],
                "clean_input_frame_in_prediction_video": False,
                "chunk_sample": "final generated frame of each traced chunk",
            },
            "summary": summary_rows,
        },
    )
    make_plot(args.output_dir / "coverage_hysteresis_validation.png", summary_rows)

    print("\nCoverage-hysteresis validation, older minus later")
    for row in summary_rows:
        print(
            f"threshold={row['coverage_threshold']:.2f} "
            f"subset={row['subset']:<13} pairs={row['pairs']:4d} "
            f"trajectories={row['trajectories']:2d} "
            f"PSNR={row['psnr_delta_trajectory_mean']} "
            f"CI=[{row['psnr_delta_ci_low']},{row['psnr_delta_ci_high']}] "
            f"SSIM={row['ssim_delta_trajectory_mean']} "
            f"CI=[{row['ssim_delta_ci_low']},{row['ssim_delta_ci_high']}]"
        )
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
