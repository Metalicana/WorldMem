#!/usr/bin/env python3
"""Evaluate WorldMem reconstruction FID on a matched frame cohort.

The WorldMem paper calls this metric reconstruction FID (rFID). The released
``calculate_fid.py`` computes ordinary image FID between generated and exact-GT
frames; this utility applies that protocol to memory-policy runs while fixing
the trajectory and frame sample across every policy.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.worldmem_eval_common import (  # noqa: E402
    list_prediction_videos,
    parse_csv,
    resolve_dataset_video_for_batch,
    video_frame_count,
)


DEFAULT_RUNS = [
    "worldmem_unbounded_60s_n30",
    "worldmem_fifo_b32_60s_n30",
    "worldmem_mce_b32_60s_n15",
    "worldmem_kcenter_coreset_b32_60s_n15",
    "worldmem_rarity_irreplaceability_b32_60s_n30",
    "worldmem_slam_covisibility_b32_60s_n30",
]

LABELS = {
    "worldmem_unbounded_60s_n30": "WorldMem",
    "worldmem_fifo_b32_60s_n30": "WorldMem + FIFO",
    "worldmem_mce_b32_60s_n15": "WorldMem + MCE",
    "worldmem_kcenter_coreset_b32_60s_n15": "WorldMem + K-center",
    "worldmem_rarity_irreplaceability_b32_60s_n30": "WorldMem + RI",
    "worldmem_slam_covisibility_b32_60s_n30": "WorldMem + KEEPSAKE",
}


def distribute_frame_counts(total_frames, videos, frames_per_video):
    """Distribute an exact sample count as evenly as possible over videos."""
    total_frames = int(total_frames)
    videos = int(videos)
    frames_per_video = int(frames_per_video)
    if videos <= 0 or frames_per_video <= 0:
        raise ValueError("videos and frames_per_video must be positive")
    if total_frames <= 1:
        raise ValueError("rFID requires at least two sampled frames")
    if total_frames > videos * frames_per_video:
        raise ValueError(
            f"Requested {total_frames} frames, but only "
            f"{videos * frames_per_video} are available"
        )
    base, remainder = divmod(total_frames, videos)
    counts = [base + (index < remainder) for index in range(videos)]
    if max(counts) > frames_per_video or sum(counts) != total_frames:
        raise AssertionError("Invalid frame-count allocation")
    return counts


def evenly_spaced_indices(length, count):
    """Return ``count`` unique indices spanning ``[0, length - 1]``."""
    length = int(length)
    count = int(count)
    if length <= 0 or count <= 0 or count > length:
        raise ValueError(f"Cannot select {count} indices from length {length}")
    if count == 1:
        return [length // 2]
    indices = [index * (length - 1) // (count - 1) for index in range(count)]
    if len(indices) != len(set(indices)) or indices[0] != 0 or indices[-1] != length - 1:
        raise AssertionError("Even frame selection was not unique and endpoint-covering")
    return indices


def build_frame_index_plan(
    batch_ids,
    duration_frames,
    total_frames,
    gt_frame_offset,
):
    """Build the policy-independent generated/GT frame-index plan."""
    batch_ids = [int(batch) for batch in batch_ids]
    counts = distribute_frame_counts(total_frames, len(batch_ids), duration_frames)
    rows = []
    for batch_idx, count in zip(batch_ids, counts):
        generated_indices = evenly_spaced_indices(duration_frames, count)
        rows.append(
            {
                "batch_idx": batch_idx,
                "sampled_frames": count,
                "generated_frame_indices": generated_indices,
                "gt_frame_indices": [gt_frame_offset + index for index in generated_indices],
            }
        )
    return rows


def iter_selected_frame_batches(video_path, indices, batch_size):
    """Decode a video once and yield selected RGB frames in contiguous batches."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for rFID evaluation") from exc

    indices = [int(index) for index in indices]
    if indices != sorted(set(indices)):
        raise ValueError("Frame indices must be sorted and unique")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    selected = []
    selected_position = 0
    target = indices[selected_position] if indices else None
    try:
        for frame_index in range(indices[-1] + 1 if indices else 0):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index != target:
                continue
            selected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            selected_position += 1
            if len(selected) == batch_size:
                yield np.stack(selected)
                selected = []
            if selected_position == len(indices):
                target = None
                break
            target = indices[selected_position]
    finally:
        cap.release()

    if selected:
        yield np.stack(selected)
    if selected_position != len(indices):
        raise RuntimeError(
            f"Decoded {selected_position}/{len(indices)} requested frames from {video_path}"
        )


def frame_batch_to_tensor(torch, frames, device):
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device=device, dtype=torch.float32).div_(255.0)


def update_metric_from_plan(metric, plan, path_key, index_key, batch_size, torch, device):
    frame_count = 0
    real = path_key == "gt_path"
    with torch.inference_mode():
        for row in plan:
            for frames in iter_selected_frame_batches(
                row[path_key], row[index_key], batch_size=batch_size
            ):
                tensor = frame_batch_to_tensor(torch, frames, device)
                metric.update(tensor, real=real)
                frame_count += len(frames)
                del tensor
    return frame_count


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--duration-frames", type=int, default=600)
    parser.add_argument("--total-frames", type=int, default=5000)
    parser.add_argument("--context-frames", type=int, default=600)
    parser.add_argument("--initial-skip-frames", type=int, default=100)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    runs = parse_csv(args.runs) or []
    if not runs:
        raise ValueError("At least one run is required")
    expected_batches = list(range(args.limit))
    gt_offset = args.initial_skip_frames + args.context_frames
    index_plan = build_frame_index_plan(
        batch_ids=expected_batches,
        duration_frames=args.duration_frames,
        total_frames=args.total_frames,
        gt_frame_offset=gt_offset,
    )

    prediction_paths = {}
    for run_name in runs:
        run_dir = args.output_root / run_name
        videos = dict(
            list_prediction_videos(run_dir, limit=args.limit, require_prefix=True)
        )
        for batch_idx in expected_batches:
            available = video_frame_count(videos[batch_idx])
            if available < args.duration_frames:
                raise RuntimeError(
                    f"{run_name} batch {batch_idx} has {available} frames; "
                    f"expected at least {args.duration_frames}"
                )
        prediction_paths[run_name] = videos

    plan = []
    for row in index_plan:
        batch_idx = row["batch_idx"]
        gt_path = resolve_dataset_video_for_batch(
            data_dir=args.data_dir,
            batch_idx=batch_idx,
            seed=args.dataset_seed,
            split="test",
            wo_updown=False,
        )
        available_gt = video_frame_count(gt_path)
        if row["gt_frame_indices"][-1] >= available_gt:
            raise RuntimeError(
                f"GT batch {batch_idx} has {available_gt} frames, but frame "
                f"{row['gt_frame_indices'][-1]} was requested"
            )
        plan.append({**row, "gt_path": str(gt_path)})

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    (args.metrics_dir / "frame_sampling_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )

    try:
        import torch
        import torchmetrics
        from torchmetrics.image.fid import FrechetInceptionDistance
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "rFID requires torchmetrics[image] and torch-fidelity in the active environment"
        ) from exc

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for rFID, but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    metric = FrechetInceptionDistance(
        feature=2048,
        normalize=True,
        reset_real_features=False,
    ).to(device)

    print(f"[rFID] extracting {args.total_frames} shared GT frames")
    gt_frames = update_metric_from_plan(
        metric=metric,
        plan=plan,
        path_key="gt_path",
        index_key="gt_frame_indices",
        batch_size=args.batch_size,
        torch=torch,
        device=device,
    )
    if gt_frames != args.total_frames:
        raise AssertionError(f"Expected {args.total_frames} GT frames; got {gt_frames}")

    summary_rows = []
    for run_name in runs:
        run_plan = [
            {
                **row,
                "pred_path": str(prediction_paths[run_name][row["batch_idx"]]),
            }
            for row in plan
        ]
        print(f"[rFID] run={run_name} frames={args.total_frames}")
        generated_frames = update_metric_from_plan(
            metric=metric,
            plan=run_plan,
            path_key="pred_path",
            index_key="generated_frame_indices",
            batch_size=args.batch_size,
            torch=torch,
            device=device,
        )
        if generated_frames != args.total_frames:
            raise AssertionError(
                f"Expected {args.total_frames} generated frames; got {generated_frames}"
            )
        value = float(metric.compute().detach().cpu().item())
        summary_rows.append(
            {
                "run_name": run_name,
                "policy_label": LABELS.get(run_name, run_name),
                "videos": args.limit,
                "frames": args.total_frames,
                "rfid": value,
            }
        )
        print(f"[rFID] {LABELS.get(run_name, run_name)}: {value:.6f}")
        metric.reset()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(args.metrics_dir / "summary.csv", summary_rows)
    payload = {
        "metric": "reconstruction_fid",
        "definition": (
            "Inception-v3 FID between generated and exact-index Minecraft GT frames; "
            "the WorldMem paper names this rFID"
        ),
        "summary": summary_rows,
        "config": {
            "runs": runs,
            "videos": args.limit,
            "duration_frames": args.duration_frames,
            "sampled_frames": args.total_frames,
            "sampling": "equal-per-trajectory deterministic temporal coverage",
            "dataset_seed": args.dataset_seed,
            "context_frames": args.context_frames,
            "initial_skip_frames": args.initial_skip_frames,
            "inception_feature": 2048,
            "torchmetrics_normalize": True,
            "torchmetrics_version": torchmetrics.__version__,
            "device": str(device),
        },
    }
    (args.metrics_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote: {args.metrics_dir / 'summary.csv'}")
    print(f"Wrote: {args.metrics_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
