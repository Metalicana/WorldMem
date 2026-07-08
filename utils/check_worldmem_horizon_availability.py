import argparse
import random
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether WorldMem Minecraft clips can support a requested evaluation horizon."
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data/minecraft"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--future_seconds", type=int, default=180)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--context_frames", type=int, default=600)
    parser.add_argument("--initial_skip_frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_videos", type=int, default=15)
    parser.add_argument("--show", type=int, default=25)
    parser.add_argument("--wo_updown", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def discover_paths(data_dir, split, wo_updown=False):
    split_dir = Path(data_dir) / split
    paths = sorted(split_dir.glob("**/*.mp4"), key=lambda path: path.name)

    if wo_updown:
        paths = [path for path in paths if "w_updown" not in str(path)]
    elif split in {"validation", "test"}:
        paths = [path for path in paths if "w_updown" in str(path)]

    if not paths and split_dir.exists():
        for sub_dir in sorted(split_dir.iterdir()):
            if sub_dir.is_dir():
                paths.extend(sorted(sub_dir.glob("**/*.mp4"), key=lambda path: path.name))
    return paths


def video_frame_count(path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for this availability check.") from exc

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def npz_lengths(path):
    npz_path = path.with_suffix(".npz")
    if not npz_path.exists():
        return 0, 0
    with np.load(npz_path) as data:
        actions = len(data["actions"]) if "actions" in data else 0
        poses = len(data["poses"]) if "poses" in data else 0
    if poses < actions:
        poses += 1
    return actions, poses


def shuffled_indices(length, seed):
    indices = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices


def main():
    args = parse_args()
    required = args.initial_skip_frames + args.context_frames + args.future_seconds * args.fps
    n_frames_valid = args.context_frames + args.future_seconds * args.fps
    paths = discover_paths(args.data_dir, args.split, wo_updown=args.wo_updown)
    if not paths:
        raise RuntimeError(f"No videos found under {args.data_dir / args.split}")

    rows = []
    for path in paths:
        video_frames = video_frame_count(path)
        action_frames, pose_frames = npz_lengths(path)
        usable_frames = min(video_frames, action_frames, pose_frames)
        rows.append(
            {
                "path": path,
                "video_frames": video_frames,
                "action_frames": action_frames,
                "pose_frames": pose_frames,
                "usable_frames": usable_frames,
                "ok": usable_frames >= required,
            }
        )

    ordered_rows = [rows[index] for index in shuffled_indices(len(rows), args.seed)]
    requested_rows = ordered_rows[: args.num_videos]
    ok_total = sum(row["ok"] for row in rows)
    ok_requested = sum(row["ok"] for row in requested_rows)

    print("WorldMem horizon availability")
    print(f"Data dir: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Future seconds: {args.future_seconds}")
    print(f"Context frames: {args.context_frames}")
    print(f"FPS: {args.fps}")
    print(f"N_FRAMES_VALID: {n_frames_valid}")
    print(f"Required source frames/actions/poses: {required}")
    print(f"Candidate videos: {len(rows)}")
    print(f"Videos meeting horizon: {ok_total}")
    print(f"Requested first videos meeting horizon: {ok_requested}/{len(requested_rows)}")
    print()
    print("First selected videos in dataset order:")
    for batch_idx, row in enumerate(requested_rows[: args.show]):
        status = "OK" if row["ok"] else "SHORT"
        print(
            f"{batch_idx:04d} {status:5s} "
            f"usable={row['usable_frames']:5d} "
            f"video={row['video_frames']:5d} "
            f"actions={row['action_frames']:5d} "
            f"poses={row['pose_frames']:5d} "
            f"{row['path']}"
        )

    if ok_requested < len(requested_rows):
        print()
        print(
            "WARNING: The current GT-backed test loader will have to retry past short clips. "
            "That can make generated batch IDs diverge from metric batch IDs."
        )

    if args.strict and ok_requested < len(requested_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
