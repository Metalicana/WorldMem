import argparse
import csv
from pathlib import Path

import numpy as np

from worldmem_eval_common import (
    load_pose_c2ws_for_batch,
    parse_int_csv,
    safe_round,
    write_json,
    write_jsonl,
)


def rotation_angle_degrees(rotation_a, rotation_b):
    relative = rotation_a.T @ rotation_b
    trace = np.trace(relative)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def find_revisits(c2ws, context_frames, future_frames, pos_threshold, yaw_threshold, min_separation):
    context_positions = c2ws[:context_frames, :3, 3]
    future_positions = c2ws[context_frames : context_frames + future_frames, :3, 3]
    context_rotations = c2ws[:context_frames, :3, :3]
    future_rotations = c2ws[context_frames : context_frames + future_frames, :3, :3]

    rows = []
    for future_offset in range(len(future_positions)):
        future_global = context_frames + future_offset
        max_context = min(context_frames, future_global - min_separation)
        if max_context <= 0:
            continue

        position_dists = np.linalg.norm(
            context_positions[:max_context] - future_positions[future_offset],
            axis=-1,
        )
        candidate_indices = np.flatnonzero(position_dists <= pos_threshold)
        if candidate_indices.size == 0:
            continue

        best = None
        for context_idx in candidate_indices:
            angle = rotation_angle_degrees(
                context_rotations[context_idx],
                future_rotations[future_offset],
            )
            if angle > yaw_threshold:
                continue
            score = float(position_dists[context_idx]) + float(angle) / max(float(yaw_threshold), 1e-6)
            if best is None or score < best["score"]:
                best = {
                    "future_frame": int(future_global),
                    "future_offset": int(future_offset),
                    "context_frame": int(context_idx),
                    "temporal_gap": int(future_global - context_idx),
                    "position_distance": float(position_dists[context_idx]),
                    "rotation_degrees": float(angle),
                    "score": score,
                }

        if best is not None:
            rows.append(best)
    return rows


def select_non_overlapping_revisits(rows, max_pairs, min_future_gap):
    selected = []
    for row in sorted(rows, key=lambda item: (item["score"], item["future_offset"])):
        if len(selected) >= max_pairs:
            break
        if any(abs(row["future_offset"] - kept["future_offset"]) < min_future_gap for kept in selected):
            continue
        selected.append(row)
    return sorted(selected, key=lambda item: item["future_offset"])


def write_csv(path, rows):
    rows = list(rows)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Find pose-based context revisits in WorldMem Minecraft test trajectories."
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data/minecraft"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--num_videos", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_frames", type=int, default=600)
    parser.add_argument("--future_seconds", type=int, default=60)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--initial_skip_frames", type=int, default=100)
    parser.add_argument("--pos_threshold", type=float, default=1.0)
    parser.add_argument("--yaw_threshold", type=float, default=20.0)
    parser.add_argument("--min_separation", type=int, default=60)
    parser.add_argument("--max_pairs_per_video", type=int, default=20)
    parser.add_argument("--min_future_gap", type=int, default=15)
    args = parser.parse_args()

    future_frames = int(args.future_seconds * args.fps)
    n_frames_valid = int(args.context_frames + future_frames)
    output_dir = args.output_dir or Path("outputs") / "worldmem_revisits"
    row_filter = set(parse_int_csv(args.rows)) if args.rows else set(range(args.num_videos))

    detail_rows = []
    summary_rows = []
    selected_rows = []
    for batch_idx in range(args.num_videos):
        if batch_idx not in row_filter:
            continue
        generated_indices = list(range(future_frames))
        c2ws, dataset_video_path = load_pose_c2ws_for_batch(
            data_dir=args.data_dir,
            batch_idx=batch_idx,
            generated_frame_indices=generated_indices,
            seed=args.seed,
            context_frames=args.context_frames,
            initial_skip_frames=args.initial_skip_frames,
            n_frames_valid=n_frames_valid,
        )
        rows = find_revisits(
            c2ws=c2ws,
            context_frames=args.context_frames,
            future_frames=future_frames,
            pos_threshold=args.pos_threshold,
            yaw_threshold=args.yaw_threshold,
            min_separation=args.min_separation,
        )
        selected = select_non_overlapping_revisits(
            rows,
            max_pairs=args.max_pairs_per_video,
            min_future_gap=args.min_future_gap,
        )

        for row in rows:
            detail_rows.append(
                {
                    "batch_idx": batch_idx,
                    "dataset_video": str(dataset_video_path),
                    **row,
                }
            )
        for row in selected:
            selected_rows.append(
                {
                    "batch_idx": batch_idx,
                    "dataset_video": str(dataset_video_path),
                    **row,
                }
            )
        summary_rows.append(
            {
                "batch_idx": batch_idx,
                "dataset_video": str(dataset_video_path),
                "future_frames": future_frames,
                "revisit_frames": len(rows),
                "selected_revisit_pairs": len(selected),
                "revisit_frame_fraction": safe_round(len(rows) / future_frames),
                "best_position_distance": safe_round(min((row["position_distance"] for row in rows), default=None)),
                "best_rotation_degrees": safe_round(min((row["rotation_degrees"] for row in rows), default=None)),
            }
        )

    overall = {
        "videos": len(summary_rows),
        "future_seconds": args.future_seconds,
        "future_frames": future_frames,
        "context_frames": args.context_frames,
        "pos_threshold": args.pos_threshold,
        "yaw_threshold": args.yaw_threshold,
        "min_separation": args.min_separation,
        "videos_with_revisits": sum(row["revisit_frames"] > 0 for row in summary_rows),
        "total_revisit_frames": sum(row["revisit_frames"] for row in summary_rows),
        "total_selected_revisit_pairs": len(selected_rows),
        "mean_revisit_frame_fraction": safe_round(
            np.mean([row["revisit_frame_fraction"] for row in summary_rows]) if summary_rows else None
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "revisit_summary.csv", summary_rows)
    write_csv(output_dir / "revisit_pairs.csv", selected_rows)
    write_jsonl(output_dir / "revisit_details.jsonl", detail_rows)
    write_json(output_dir / "revisit_summary.json", {"overall": overall, "by_video": summary_rows})

    print("WorldMem revisit analysis")
    print(f"Output dir: {output_dir}")
    print(f"Videos with revisits: {overall['videos_with_revisits']}/{overall['videos']}")
    print(f"Total selected revisit pairs: {overall['total_selected_revisit_pairs']}")
    print(f"Mean revisit frame fraction: {overall['mean_revisit_frame_fraction']}")
    for row in summary_rows:
        print(
            f"batch={row['batch_idx']:02d} "
            f"revisit_frames={row['revisit_frames']:4d} "
            f"selected={row['selected_revisit_pairs']:2d} "
            f"fraction={row['revisit_frame_fraction']}"
        )


if __name__ == "__main__":
    main()
