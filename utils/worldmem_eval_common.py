import json
import math
import random
import re
from pathlib import Path

import numpy as np


PRED_VIDEO_RE = re.compile(r"^video_batch(?P<batch>\d+)_(?P<sample>\d+)_rank(?P<rank>\d+)(?:_step.*)?\.mp4$")


def parse_csv(value):
    if value is None:
        return None
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_int_csv(value):
    values = parse_csv(value)
    return [int(value) for value in values] if values else []


def safe_round(value, digits=6):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), digits)


def mean_or_none(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def discover_run_dirs(output_root, runs=None):
    output_root = Path(output_root)
    if runs:
        return [(run, output_root / run) for run in runs]
    return [
        (path.name, path)
        for path in sorted(output_root.glob("worldmem_*_*s_n*"))
        if (path / "videos" / "test_vis" / "pred").is_dir()
    ]


def prediction_dir(run_dir):
    return Path(run_dir) / "videos" / "test_vis" / "pred"


def parse_prediction_name(path):
    match = PRED_VIDEO_RE.match(Path(path).name)
    if not match:
        return None
    return {
        "batch_idx": int(match.group("batch")),
        "sample_idx": int(match.group("sample")),
        "rank": int(match.group("rank")),
    }


def list_prediction_videos(run_dir, limit=None):
    videos = {}
    for path in sorted(prediction_dir(run_dir).glob("video_batch*.mp4")):
        parsed = parse_prediction_name(path)
        if parsed is None or parsed["sample_idx"] != 0:
            continue
        batch_idx = parsed["batch_idx"]
        videos.setdefault(batch_idx, path)
    items = sorted(videos.items())
    if limit is not None:
        items = items[: int(limit)]
    return items


def discover_minecraft_paths(data_dir, split="test", wo_updown=False):
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    paths = sorted(list(split_dir.glob("**/*.mp4")), key=lambda path: path.name)

    if wo_updown:
        paths = [path for path in paths if "w_updown" not in str(path)]
    elif split in {"validation", "test"}:
        paths = [path for path in paths if "w_updown" in str(path)]

    if not paths and split_dir.exists():
        for sub_dir in sorted(split_dir.iterdir()):
            if sub_dir.is_dir():
                paths += sorted(list(sub_dir.glob("**/*.mp4")), key=lambda path: path.name)
    return paths


def shuffled_dataset_indices(num_items, seed=42):
    indices = list(range(num_items))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices


def resolve_dataset_video_for_batch(data_dir, batch_idx, seed=42, split="test", wo_updown=False):
    paths = discover_minecraft_paths(data_dir, split=split, wo_updown=wo_updown)
    if not paths:
        raise RuntimeError(f"No Minecraft videos found under {Path(data_dir) / split}")
    remap = shuffled_dataset_indices(len(paths), seed=seed)
    if batch_idx < 0 or batch_idx >= len(remap):
        raise IndexError(f"Batch index {batch_idx} is outside dataset length {len(remap)}")
    return paths[remap[batch_idx]]


def normalize_worldmem_pose_segment(poses_pool, frame_start, n_frames):
    poses = np.asarray(poses_pool, dtype=np.float64).copy()
    if len(poses) > 1:
        poses[0, 1] = poses[1, 1]
    segment = np.copy(poses[frame_start : frame_start + n_frames])
    if len(segment) < n_frames:
        raise RuntimeError(
            f"Pose segment too short: requested {n_frames} from {frame_start}, got {len(segment)}"
        )
    ref = segment[:1]
    segment[:, :3] -= ref[:, :3]
    segment[:, -1] = -segment[:, -1]
    segment[:, 3:] %= 360
    return segment


def euler_to_c2w_numpy(poses):
    poses = np.asarray(poses, dtype=np.float64)
    x, y, z, pitch_deg, yaw_deg = [poses[:, idx] for idx in range(5)]
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    r_pitch = np.zeros((len(poses), 3, 3), dtype=np.float64)
    r_pitch[:, 0, 0] = 1.0
    r_pitch[:, 1, 1] = cos_pitch
    r_pitch[:, 1, 2] = -sin_pitch
    r_pitch[:, 2, 1] = sin_pitch
    r_pitch[:, 2, 2] = cos_pitch

    r_yaw = np.zeros((len(poses), 3, 3), dtype=np.float64)
    r_yaw[:, 0, 0] = cos_yaw
    r_yaw[:, 0, 2] = sin_yaw
    r_yaw[:, 1, 1] = 1.0
    r_yaw[:, 2, 0] = -sin_yaw
    r_yaw[:, 2, 2] = cos_yaw

    rotations = r_yaw @ r_pitch
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    c2w[:, :3, :3] = rotations
    c2w[:, :3, 3] = np.stack([x, y, z], axis=-1)
    return c2w


def load_pose_c2ws_for_batch(
    data_dir,
    batch_idx,
    generated_frame_indices,
    seed=42,
    context_frames=600,
    initial_skip_frames=100,
    n_frames_valid=1200,
    split="test",
    wo_updown=False,
):
    dataset_video_path = resolve_dataset_video_for_batch(
        data_dir=data_dir,
        batch_idx=batch_idx,
        seed=seed,
        split=split,
        wo_updown=wo_updown,
    )
    npz_path = dataset_video_path.with_suffix(".npz")
    with np.load(npz_path) as data:
        poses_pool = data["poses"]
        actions_pool = data["actions"]
        if len(poses_pool) < len(actions_pool):
            poses_pool = np.pad(poses_pool, ((1, 0), (0, 0)))

    pose_segment = normalize_worldmem_pose_segment(
        poses_pool=poses_pool,
        frame_start=initial_skip_frames,
        n_frames=n_frames_valid,
    )
    local_indices = [context_frames + int(index) for index in generated_frame_indices]
    if max(local_indices, default=-1) >= len(pose_segment):
        raise RuntimeError(
            f"Generated frame index exceeds pose segment for batch {batch_idx}: "
            f"max local {max(local_indices)} vs segment length {len(pose_segment)}"
        )
    return euler_to_c2w_numpy(pose_segment[local_indices]), dataset_video_path


def get_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for WorldMem video evaluation.") from exc
    return cv2


def video_frame_count(video_path):
    cv2 = get_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def read_video_frames(video_path, indices):
    cv2 = get_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames = {}
    try:
        for frame_index in sorted(set(int(index) for index in indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            frames[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return frames


def read_worldmem_gt_frames(
    data_dir,
    batch_idx,
    generated_frame_indices,
    seed=42,
    context_frames=600,
    initial_skip_frames=100,
    split="test",
    wo_updown=False,
):
    dataset_video_path = resolve_dataset_video_for_batch(
        data_dir=data_dir,
        batch_idx=batch_idx,
        seed=seed,
        split=split,
        wo_updown=wo_updown,
    )
    original_indices = [
        initial_skip_frames + context_frames + int(index)
        for index in generated_frame_indices
    ]
    frames = read_video_frames(dataset_video_path, original_indices)
    return {
        generated_index: frames[original_index]
        for generated_index, original_index in zip(generated_frame_indices, original_indices)
        if original_index in frames
    }, dataset_video_path
