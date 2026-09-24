#!/usr/bin/env python3
"""Export pose-verified WorldMem revisit candidates and editable previews.

The generated MP4 indices used here are always zero-based indices into the
saved prediction video.  They deliberately exclude the initial context.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from worldmem_eval_common import (  # noqa: E402
    euler_to_c2w_numpy,
    get_cv2,
    normalize_worldmem_pose_segment,
    read_video_frames,
)


CSV_FIELDS = [
    "video_filename",
    "trajectory_id",
    "first_frame",
    "middle_frame",
    "revisit_frame",
    "first_time_sec",
    "middle_time_sec",
    "revisit_time_sec",
    "endpoint_position_distance",
    "position_unit",
    "endpoint_rotation_deg",
    "middle_position_distance",
    "middle_rotation_deg",
    "away_duration_sec",
    "pose_match",
    "away_verified",
    "gt_checked",
    "notes",
]

POLICY_KEYS = ("unbounded", "fifo_b32", "keepsake_b32")
TRACE_BATCH_KEYS = ("global_batch_idx", "output_batch_idx", "batch_idx")
VIDEO_RE = re.compile(r"^video_batch(?P<batch>\d+)_0_rank0\.mp4$")
NUMERIC_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/minecraft"))
    parser.add_argument("--unbounded-run-dir", type=Path, required=True)
    parser.add_argument("--fifo-run-dir", type=Path, required=True)
    parser.add_argument("--keepsake-run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--wo-updown", action="store_true")
    parser.add_argument("--trajectory-fps", type=float, default=10.0)
    parser.add_argument("--expected-output-frames", type=int, default=600)
    parser.add_argument("--context-frames", type=int, default=600)
    parser.add_argument("--source-start-offset", type=int, default=100)
    parser.add_argument("--min-separation-sec", type=float, default=5.0)
    parser.add_argument("--endpoint-position-threshold", type=float, default=0.75)
    parser.add_argument("--endpoint-rotation-threshold-deg", type=float, default=15.0)
    parser.add_argument("--departure-position-threshold", type=float, default=2.0)
    parser.add_argument("--departure-rotation-threshold-deg", type=float, default=45.0)
    parser.add_argument("--min-away-sec", type=float, default=1.0)
    parser.add_argument("--max-candidates-per-trajectory", type=int, default=12)
    parser.add_argument("--revisit-suppression-sec", type=float, default=1.0)
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_value) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def rotation_angles_deg(rotations: np.ndarray, reference: np.ndarray) -> np.ndarray:
    traces = np.einsum("nij,ij->n", rotations, reference)
    cosine = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs [start, end) for a one-dimensional boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def find_verified_returns(
    c2ws: np.ndarray,
    *,
    trajectory_fps: float,
    min_separation_sec: float,
    endpoint_position_threshold: float,
    endpoint_rotation_threshold_deg: float,
    departure_position_threshold: float,
    departure_rotation_threshold_deg: float,
    min_away_sec: float,
) -> list[dict[str, Any]]:
    c2ws = np.asarray(c2ws, dtype=np.float64)
    if c2ws.ndim != 3 or c2ws.shape[1:] != (4, 4):
        raise ValueError(f"Expected [T,4,4] c2ws, got {c2ws.shape}")

    rotations = c2ws[:, :3, :3]
    positions = c2ws[:, :3, 3]
    position_available = np.isfinite(positions).all(axis=1)
    min_gap = int(math.ceil(min_separation_sec * trajectory_fps))
    min_away = int(math.ceil(min_away_sec * trajectory_fps))
    if min_gap <= 0 or min_away <= 0:
        raise ValueError("Temporal thresholds must correspond to at least one frame")

    rotation_traces = np.einsum("aij,bij->ab", rotations, rotations)
    rotation_distances = np.degrees(
        np.arccos(np.clip((rotation_traces - 1.0) / 2.0, -1.0, 1.0))
    )
    position_distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1
    )
    valid_position_pairs = position_available[:, None] & position_available[None, :]
    position_distances[~valid_position_pairs] = np.nan

    candidates: list[dict[str, Any]] = []

    for first in range(0, len(c2ws) - min_gap):
        endpoint_rotation_ok = (
            rotation_distances[first]
            <= endpoint_rotation_threshold_deg + NUMERIC_TOLERANCE
        )
        endpoint_position_ok = (
            position_distances[first]
            <= endpoint_position_threshold + NUMERIC_TOLERANCE
        )
        endpoint_ok = endpoint_rotation_ok & (
            endpoint_position_ok | ~valid_position_pairs[first]
        )
        endpoint_ok[: first + min_gap] = False
        endpoint_errors = (
            rotation_distances[first] / endpoint_rotation_threshold_deg
        )
        endpoint_errors = endpoint_errors.copy()
        endpoint_errors[valid_position_pairs[first]] += (
            position_distances[first, valid_position_pairs[first]]
            / endpoint_position_threshold
        )
        revisit_episodes = true_runs(endpoint_ok)
        revisit_records = []
        for episode_start, episode_end in revisit_episodes:
            episode_indices = np.arange(episode_start, episode_end, dtype=np.int64)
            representative = int(
                episode_indices[
                    int(np.argmin(endpoint_errors[episode_indices]))
                ]
            )
            revisit_records.append(
                {
                    "frame": representative,
                    "episode_start": episode_start,
                    "episode_end": episode_end,
                }
            )
        if not revisit_records:
            continue

        away = (
            rotation_distances[first]
            >= departure_rotation_threshold_deg - NUMERIC_TOLERANCE
        )
        if position_available[first]:
            away |= (
                np.nan_to_num(position_distances[first], nan=-np.inf)
                >= departure_position_threshold - NUMERIC_TOLERANCE
            )
        away[: first + 1] = False

        position_severity = np.nan_to_num(
            position_distances[first] / departure_position_threshold,
            nan=-np.inf,
        )
        rotation_severity = (
            rotation_distances[first] / departure_rotation_threshold_deg
        )
        severity = np.maximum(position_severity, rotation_severity)
        qualifying_runs = [
            run for run in true_runs(away) if run[1] - run[0] >= min_away
        ]
        if not qualifying_runs:
            continue

        run_records = []
        for run_start, run_end in qualifying_runs:
            middle = run_start + int(np.argmax(severity[run_start:run_end]))
            run_records.append(
                {
                    "start": run_start,
                    "end": run_end,
                    "middle": middle,
                    "key": (
                        float(severity[middle]),
                        run_end - run_start,
                        -run_start,
                    ),
                }
            )

        run_cursor = 0
        best_run: dict[str, Any] | None = None
        for revisit_record in revisit_records:
            revisit = int(revisit_record["frame"])
            while (
                run_cursor < len(run_records)
                and int(run_records[run_cursor]["end"]) <= revisit
            ):
                proposed = run_records[run_cursor]
                if best_run is None or proposed["key"] > best_run["key"]:
                    best_run = proposed
                run_cursor += 1
            if best_run is None:
                continue

            middle = int(best_run["middle"])
            run_start = int(best_run["start"])
            run_end = int(best_run["end"])
            endpoint_position = float(position_distances[first, revisit])
            endpoint_rotation = float(rotation_distances[first, revisit])
            orientation_only = not bool(valid_position_pairs[first, revisit])
            if orientation_only:
                endpoint_error = endpoint_rotation / endpoint_rotation_threshold_deg
            else:
                endpoint_error = (
                    endpoint_position / endpoint_position_threshold
                    + endpoint_rotation / endpoint_rotation_threshold_deg
                )

            candidates.append(
                {
                    "first_frame": first,
                    "middle_frame": middle,
                    "revisit_frame": revisit,
                    "revisit_episode_start_frame": int(
                        revisit_record["episode_start"]
                    ),
                    "revisit_episode_end_frame_exclusive": int(
                        revisit_record["episode_end"]
                    ),
                    "endpoint_position_distance": endpoint_position,
                    "endpoint_rotation_deg": endpoint_rotation,
                    "middle_position_distance": float(
                        position_distances[first, middle]
                    ),
                    "middle_rotation_deg": float(
                        rotation_distances[first, middle]
                    ),
                    "away_duration_sec": (run_end - run_start) / trajectory_fps,
                    "temporal_separation_sec": (revisit - first) / trajectory_fps,
                    "endpoint_error_normalized": endpoint_error,
                    "orientation_only": orientation_only,
                }
            )

    return candidates


def select_ranked_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_candidates: int,
    revisit_suppression_frames: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["endpoint_error_normalized"],
            -row["temporal_separation_sec"],
            -row["away_duration_sec"],
            row["first_frame"],
            row["revisit_frame"],
        ),
    )
    selected: list[dict[str, Any]] = []
    for row in ranked:
        if any(
            abs(row["revisit_frame"] - prior["revisit_frame"])
            < revisit_suppression_frames
            for prior in selected
        ):
            continue
        selected.append(dict(row))
        if len(selected) >= max_candidates:
            break
    return selected


def video_metadata(path: Path) -> dict[str, Any]:
    cv2 = get_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    metadata = {
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        "encoded_fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    }
    capture.release()
    fps = metadata["encoded_fps"]
    if fps <= 0:
        raise RuntimeError(f"Video has no valid encoded frame rate: {path}")
    metadata["encoded_playback_duration_sec"] = metadata["frame_count"] / fps
    return metadata


def extract_batch_number(path: Path) -> int:
    match = VIDEO_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected prediction filename: {path.name}")
    return int(match.group("batch"))


def canonical_prediction_video(run_dir: Path, batch: int) -> Path:
    path = (
        run_dir
        / "videos"
        / "test_vis"
        / "pred"
        / f"video_batch{batch:05d}_0_rank0.mp4"
    )
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing canonical prediction video: {path}")
    return path


def exact_dataset_paths(data_dir: Path, *, split: str, wo_updown: bool) -> list[Path]:
    """Reproduce MinecraftVideoDataset.get_data_paths, including fallback order."""
    split_dir = data_dir / split
    paths = sorted(split_dir.glob("**/*.mp4"), key=lambda path: path.name)
    if wo_updown:
        paths = [path for path in paths if "w_updown" not in str(path)]
    elif split in {"validation", "test"}:
        paths = [path for path in paths if "w_updown" in str(path)]

    if not paths and split_dir.exists():
        for sub_dir in os.listdir(split_dir):
            sub_path = split_dir / sub_dir
            paths.extend(sorted(sub_path.glob("**/*.mp4"), key=lambda path: path.name))
    return paths


def resolve_exact_dataset_video(
    data_dir: Path,
    *,
    batch: int,
    seed: int,
    split: str = "test",
    wo_updown: bool = False,
) -> Path:
    paths = exact_dataset_paths(data_dir, split=split, wo_updown=wo_updown)
    if not paths:
        raise RuntimeError(f"No Minecraft videos found under {data_dir / split}")
    remap = list(range(len(paths)))
    random.Random(seed).shuffle(remap)
    if batch < 0 or batch >= len(remap):
        raise IndexError(f"Batch {batch} is outside dataset length {len(remap)}")
    return paths[remap[batch]]


def load_exact_generated_c2ws(
    dataset_video: Path,
    *,
    output_frames: int,
    context_frames: int,
    source_start_offset: int,
) -> np.ndarray:
    npz_path = dataset_video.with_suffix(".npz")
    with np.load(npz_path) as data:
        poses_pool = data["poses"]
        actions_pool = data["actions"]
    if len(poses_pool) < len(actions_pool):
        poses_pool = np.pad(poses_pool, ((1, 0), (0, 0)))
    segment = normalize_worldmem_pose_segment(
        poses_pool=poses_pool,
        frame_start=source_start_offset,
        n_frames=context_frames + output_frames,
    )
    return euler_to_c2w_numpy(segment[context_frames : context_frames + output_frames])


def dataset_source_metadata(dataset_video: Path, required_frames: int) -> dict[str, Any]:
    metadata = video_metadata(dataset_video)
    with np.load(dataset_video.with_suffix(".npz")) as data:
        action_frames = len(data["actions"])
        pose_frames_raw = len(data["poses"])
    pose_frames_effective = pose_frames_raw + int(pose_frames_raw < action_frames)
    usable_frames = min(
        int(metadata["frame_count"]),
        action_frames,
        pose_frames_effective,
    )
    return {
        **metadata,
        "action_frames": action_frames,
        "pose_frames_raw": pose_frames_raw,
        "pose_frames_effective": pose_frames_effective,
        "usable_frames": usable_frames,
        "required_frames": required_frames,
        "has_required_horizon": usable_frames >= required_frames,
    }


def trace_batch_idx(record: dict[str, Any]) -> int | None:
    for key in TRACE_BATCH_KEYS:
        value = record.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def collect_trace_evidence(run_dir: Path, batches: set[int]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {
        batch: {"start_records": [], "run_end_records": 0} for batch in batches
    }
    trace_root = run_dir / "access_traces"
    if not trace_root.exists():
        return result
    for path in sorted(trace_root.glob("*.jsonl")):
        with path.open(errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                batch = trace_batch_idx(record)
                if batch not in result:
                    continue
                event = record.get("event")
                if event == "memory_run_start":
                    keep = {
                        key: record.get(key)
                        for key in (
                            "event",
                            "global_batch_idx",
                            "dataset_batch_idx",
                            "output_batch_idx",
                            "generation_seed",
                            "memory_policy",
                            "memory_budget",
                            "memory_reference_source",
                            "context_frames",
                            "n_frames",
                        )
                        if key in record
                    }
                    keep["trace_file"] = str(path.resolve())
                    result[batch]["start_records"].append(keep)
                elif event == "memory_run_end":
                    result[batch]["run_end_records"] += 1
    return result


def pooled_motion_statistics(c2ws_by_batch: dict[int, np.ndarray]) -> dict[str, Any]:
    translations: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    for c2ws in c2ws_by_batch.values():
        positions = c2ws[:, :3, 3]
        translations.append(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        rotations.append(
            np.asarray(
                [
                    rotation_angle_deg(c2ws[i, :3, :3], c2ws[i + 1, :3, :3])
                    for i in range(len(c2ws) - 1)
                ]
            )
        )

    def summarize(parts: list[np.ndarray]) -> dict[str, float]:
        values = np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)
        if values.size == 0:
            return {}
        return {
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "maximum": float(np.max(values)),
        }

    return {
        "one_step_translation_minecraft_blocks": summarize(translations),
        "one_step_rotation_degrees": summarize(rotations),
    }


def save_rgb(path: Path, image: np.ndarray) -> None:
    cv2 = get_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write preview: {path}")


def export_previews(
    *,
    output_root: Path,
    selected_by_batch: dict[int, list[dict[str, Any]]],
    videos_by_batch: dict[int, dict[str, Path]],
    dataset_video_by_batch: dict[int, Path],
    context_frames: int,
    source_start_offset: int,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for batch, candidates in selected_by_batch.items():
        if not candidates:
            continue
        output_indices = sorted(
            {
                int(row[key])
                for row in candidates
                for key in ("first_frame", "middle_frame", "revisit_frame")
            }
        )
        generated_frames = {
            policy: read_video_frames(path, output_indices)
            for policy, path in videos_by_batch[batch].items()
        }
        source_indices = [
            source_start_offset + context_frames + frame for frame in output_indices
        ]
        gt_frames: dict[int, np.ndarray] = {}
        gt_error: str | None = None
        try:
            decoded = read_video_frames(dataset_video_by_batch[batch], source_indices)
            gt_frames = {
                output_frame: decoded[source_frame]
                for output_frame, source_frame in zip(output_indices, source_indices)
                if source_frame in decoded
            }
        except Exception as exc:  # provenance preserves the exact missing-data failure
            gt_error = f"{type(exc).__name__}: {exc}"

        for rank, row in enumerate(candidates, start=1):
            case_name = f"candidate_{rank:02d}"
            case_root = output_root / "previews" / f"trajectory_{batch:05d}" / case_name
            roles = {
                "first": int(row["first_frame"]),
                "middle": int(row["middle_frame"]),
                "revisit": int(row["revisit_frame"]),
            }
            candidate_gt_complete = True
            for policy in POLICY_KEYS:
                for role, frame in roles.items():
                    if frame not in generated_frames[policy]:
                        raise RuntimeError(
                            f"Could not decode {policy} batch {batch} output frame {frame}"
                        )
                    image = generated_frames[policy][frame]
                    path = case_root / policy / f"{role}_frame_{frame:05d}.png"
                    save_rgb(path, image)
                    manifest.append(
                        {
                            "trajectory_id": batch,
                            "candidate_rank": rank,
                            "source": policy,
                            "role": role,
                            "output_frame": frame,
                            "source_video_frame": source_start_offset + context_frames + frame,
                            "path": str(path.resolve()),
                            "status": "exported",
                        }
                    )
            for role, frame in roles.items():
                if frame in gt_frames:
                    path = case_root / "ground_truth" / f"{role}_frame_{frame:05d}.png"
                    save_rgb(path, gt_frames[frame])
                    status = "exported_unchecked"
                    path_value = str(path.resolve())
                else:
                    candidate_gt_complete = False
                    status = (
                        f"missing: {gt_error}"
                        if gt_error
                        else "missing: source frame did not decode"
                    )
                    path_value = ""
                manifest.append(
                    {
                        "trajectory_id": batch,
                        "candidate_rank": rank,
                        "source": "ground_truth",
                        "role": role,
                        "output_frame": frame,
                        "source_video_frame": source_start_offset + context_frames + frame,
                        "path": path_value,
                        "status": status,
                    }
                )
            row["gt_preview_status"] = (
                "exported_unchecked"
                if candidate_gt_complete
                else f"missing: {gt_error or 'one or more source frames did not decode'}"
            )
    return manifest


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.trajectory_fps <= 0:
        raise ValueError("--trajectory-fps must be positive")
    if args.max_candidates_per_trajectory <= 0:
        raise ValueError("--max-candidates-per-trajectory must be positive")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dirs = {
        "unbounded": args.unbounded_run_dir.resolve(),
        "fifo_b32": args.fifo_run_dir.resolve(),
        "keepsake_b32": args.keepsake_run_dir.resolve(),
    }
    data_dir = args.data_dir.resolve()
    expected_batches = list(range(args.limit))
    missing_cases: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    videos_by_batch: dict[int, dict[str, Path]] = {}
    video_metadata_by_batch: dict[int, dict[str, Any]] = {}
    c2ws_by_batch: dict[int, np.ndarray] = {}
    dataset_video_by_batch: dict[int, Path] = {}

    trace_evidence = {
        policy: collect_trace_evidence(run_dir, set(expected_batches))
        for policy, run_dir in run_dirs.items()
    }

    for batch in expected_batches:
        policy_videos: dict[str, Path] = {}
        policy_metadata: dict[str, Any] = {}
        batch_valid = True
        for policy, run_dir in run_dirs.items():
            try:
                video = canonical_prediction_video(run_dir, batch)
                if extract_batch_number(video) != batch:
                    raise RuntimeError(f"Filename batch mismatch: {video}")
                metadata = video_metadata(video)
                if metadata["frame_count"] != args.expected_output_frames:
                    raise RuntimeError(
                        f"Expected {args.expected_output_frames} frames, "
                        f"found {metadata['frame_count']}"
                    )
                policy_videos[policy] = video
                policy_metadata[policy] = metadata
            except Exception as exc:
                batch_valid = False
                missing_cases.append(
                    {
                        "batch": batch,
                        "policy": policy,
                        "stage": "prediction_video",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if not batch_valid:
            if args.allow_incomplete:
                continue
            raise RuntimeError(f"Batch {batch} does not have a valid three-policy video set")

        dataset_video = resolve_exact_dataset_video(
            data_dir,
            batch=batch,
            seed=args.dataset_seed,
            wo_updown=args.wo_updown,
        )
        source_metadata = dataset_source_metadata(
            dataset_video,
            required_frames=(
                args.source_start_offset
                + args.context_frames
                + args.expected_output_frames
            ),
        )
        if not source_metadata["has_required_horizon"]:
            missing_cases.append(
                {
                    "batch": batch,
                    "policy": "shared_requested_trajectory",
                    "stage": "source_horizon",
                    "error": (
                        f"Only {source_metadata['usable_frames']} usable source frames; "
                        f"requires {source_metadata['required_frames']}"
                    ),
                }
            )
            if args.allow_incomplete:
                continue
            raise RuntimeError(
                f"Batch {batch} requested source cannot support the 60-second mapping"
            )
        c2ws = load_exact_generated_c2ws(
            dataset_video,
            output_frames=args.expected_output_frames,
            context_frames=args.context_frames,
            source_start_offset=args.source_start_offset,
        )
        pose_path = dataset_video.with_suffix(".npz")
        if len(c2ws) != args.expected_output_frames:
            raise RuntimeError(
                f"Batch {batch}: expected {args.expected_output_frames} requested poses, "
                f"found {len(c2ws)}"
            )

        videos_by_batch[batch] = policy_videos
        video_metadata_by_batch[batch] = policy_metadata
        c2ws_by_batch[batch] = c2ws
        dataset_video_by_batch[batch] = dataset_video
        mappings.append(
            {
                "trajectory_id": batch,
                "requested_dataset_video": str(dataset_video.resolve()),
                "requested_pose_file": str(pose_path.resolve()),
                "requested_source_metadata": source_metadata,
                "prediction_videos": {
                    policy: str(path.resolve()) for policy, path in policy_videos.items()
                },
                "same_requested_trajectory": True,
                "actual_post_retry_source_verified": False,
                "requested_source_has_required_horizon": source_metadata[
                    "has_required_horizon"
                ],
                "requested_source_usable_frames": source_metadata["usable_frames"],
                "trace_evidence": {
                    policy: trace_evidence[policy].get(batch, {}) for policy in POLICY_KEYS
                },
            }
        )

    mapping_rows = []
    for mapping in mappings:
        batch = int(mapping["trajectory_id"])
        metadata = video_metadata_by_batch[batch]
        mapping_rows.append(
            {
                "trajectory_id": batch,
                "video_filename": videos_by_batch[batch]["unbounded"].name,
                "unbounded_video": mapping["prediction_videos"]["unbounded"],
                "fifo_b32_video": mapping["prediction_videos"]["fifo_b32"],
                "keepsake_b32_video": mapping["prediction_videos"]["keepsake_b32"],
                "requested_dataset_video": mapping["requested_dataset_video"],
                "camera_state_npz": mapping["requested_pose_file"],
                "same_requested_trajectory": True,
                "actual_post_retry_source_verified": False,
                "requested_source_has_required_horizon": mapping[
                    "requested_source_has_required_horizon"
                ],
                "requested_source_usable_frames": mapping[
                    "requested_source_usable_frames"
                ],
                "output_frames": args.expected_output_frames,
                "trajectory_fps": args.trajectory_fps,
                "unbounded_encoded_fps": metadata["unbounded"]["encoded_fps"],
                "fifo_b32_encoded_fps": metadata["fifo_b32"]["encoded_fps"],
                "keepsake_b32_encoded_fps": metadata["keepsake_b32"]["encoded_fps"],
            }
        )
    mapping_path = output_root / "trajectory_mapping.csv"
    write_csv(
        mapping_path,
        mapping_rows,
        [
            "trajectory_id",
            "video_filename",
            "unbounded_video",
            "fifo_b32_video",
            "keepsake_b32_video",
            "requested_dataset_video",
            "camera_state_npz",
            "same_requested_trajectory",
            "actual_post_retry_source_verified",
            "requested_source_has_required_horizon",
            "requested_source_usable_frames",
            "output_frames",
            "trajectory_fps",
            "unbounded_encoded_fps",
            "fifo_b32_encoded_fps",
            "keepsake_b32_encoded_fps",
        ],
    )

    selected_by_batch: dict[int, list[dict[str, Any]]] = {}
    all_candidate_counts: dict[str, int] = {}
    suppression_frames = int(math.ceil(args.revisit_suppression_sec * args.trajectory_fps))
    for batch, c2ws in c2ws_by_batch.items():
        all_candidates = find_verified_returns(
            c2ws,
            trajectory_fps=args.trajectory_fps,
            min_separation_sec=args.min_separation_sec,
            endpoint_position_threshold=args.endpoint_position_threshold,
            endpoint_rotation_threshold_deg=args.endpoint_rotation_threshold_deg,
            departure_position_threshold=args.departure_position_threshold,
            departure_rotation_threshold_deg=args.departure_rotation_threshold_deg,
            min_away_sec=args.min_away_sec,
        )
        selected = select_ranked_candidates(
            all_candidates,
            max_candidates=args.max_candidates_per_trajectory,
            revisit_suppression_frames=suppression_frames,
        )
        selected_by_batch[batch] = selected
        all_candidate_counts[str(batch)] = len(all_candidates)

    preview_manifest: list[dict[str, Any]] = []
    if not args.no_previews:
        preview_manifest = export_previews(
            output_root=output_root,
            selected_by_batch=selected_by_batch,
            videos_by_batch=videos_by_batch,
            dataset_video_by_batch=dataset_video_by_batch,
            context_frames=args.context_frames,
            source_start_offset=args.source_start_offset,
        )
        write_csv(
            output_root / "preview_manifest.csv",
            preview_manifest,
            [
                "trajectory_id",
                "candidate_rank",
                "source",
                "role",
                "output_frame",
                "source_video_frame",
                "path",
                "status",
            ],
        )
        missing_cases.extend(
            {
                "batch": int(row["trajectory_id"]),
                "policy": "ground_truth",
                "stage": "preview_export",
                "output_frame": int(row["output_frame"]),
                "error": row["status"],
            }
            for row in preview_manifest
            if row["source"] == "ground_truth"
            and str(row["status"]).startswith("missing:")
        )

    rows: list[dict[str, Any]] = []
    for batch in sorted(selected_by_batch):
        filename = videos_by_batch[batch]["unbounded"].name
        for rank, candidate in enumerate(selected_by_batch[batch], start=1):
            orientation_only = bool(candidate["orientation_only"])
            gt_status = candidate.get("gt_preview_status", "not_exported")
            notes = [
                f"geometry_rank={rank}",
                f"temporal_separation_sec={candidate['temporal_separation_sec']:.3f}",
                (
                    "revisit_episode_frames="
                    f"[{candidate['revisit_episode_start_frame']},"
                    f"{candidate['revisit_episode_end_frame_exclusive']})"
                ),
                f"gt_preview={gt_status}",
                "GT not visually checked",
            ]
            if orientation_only:
                notes.append("orientation-only pose match; position unavailable")
            rows.append(
                {
                    "video_filename": filename,
                    "trajectory_id": batch,
                    "first_frame": candidate["first_frame"],
                    "middle_frame": candidate["middle_frame"],
                    "revisit_frame": candidate["revisit_frame"],
                    "first_time_sec": f"{candidate['first_frame'] / args.trajectory_fps:.3f}",
                    "middle_time_sec": f"{candidate['middle_frame'] / args.trajectory_fps:.3f}",
                    "revisit_time_sec": f"{candidate['revisit_frame'] / args.trajectory_fps:.3f}",
                    "endpoint_position_distance": (
                        ""
                        if not np.isfinite(candidate["endpoint_position_distance"])
                        else f"{candidate['endpoint_position_distance']:.6f}"
                    ),
                    "position_unit": (
                        "unavailable"
                        if orientation_only
                        else "Minecraft block-coordinate unit"
                    ),
                    "endpoint_rotation_deg": f"{candidate['endpoint_rotation_deg']:.6f}",
                    "middle_position_distance": (
                        ""
                        if not np.isfinite(candidate["middle_position_distance"])
                        else f"{candidate['middle_position_distance']:.6f}"
                    ),
                    "middle_rotation_deg": f"{candidate['middle_rotation_deg']:.6f}",
                    "away_duration_sec": f"{candidate['away_duration_sec']:.3f}",
                    "pose_match": "orientation_only" if orientation_only else True,
                    "away_verified": True,
                    "gt_checked": False,
                    "notes": "; ".join(notes),
                    "candidate_rank": rank,
                }
            )

    rows.sort(
        key=lambda row: (
            int(row["trajectory_id"]),
            int(row["candidate_rank"]),
        )
    )
    csv_path = output_root / "revisit_candidates.csv"
    write_csv(csv_path, rows, CSV_FIELDS)

    encoded_fps_values = sorted(
        {
            round(float(metadata["encoded_fps"]), 6)
            for policy_metadata in video_metadata_by_batch.values()
            for metadata in policy_metadata.values()
        }
    )
    provenance = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "status": "verified_candidates_found" if rows else "no_verified_returns",
        "request_scope": {
            "new_generation_launched": False,
            "aggregate_quality_metrics_computed": False,
            "matched_trajectory_count": len(videos_by_batch),
            "requested_limit": args.limit,
            "policies": {
                "unbounded": str(run_dirs["unbounded"]),
                "fifo_b32": str(run_dirs["fifo_b32"]),
                "keepsake_b32": str(run_dirs["keepsake_b32"]),
            },
        },
        "clock_and_index_mapping": {
            "trajectory_fps": args.trajectory_fps,
            "encoded_mp4_fps_values": encoded_fps_values,
            "expected_mp4_frame_count": args.expected_output_frames,
            "trajectory_duration_sec": args.expected_output_frames / args.trajectory_fps,
            "encoded_playback_duration_sec_if_15_fps": args.expected_output_frames / 15.0,
            "time_columns_use": "trajectory/request clock, output_frame / trajectory_fps",
            "trajectory_clock_evidence": {
                "source_file": str(
                    (
                        SCRIPT_DIR.parent
                        / "datasets/video/minecraft_video_dataset.py"
                    ).resolve()
                ),
                "evidence": "MinecraftVideoDataset.load_data fixes source clip fps to 10",
            },
            "encoding_clock_evidence": {
                "source_file": str((SCRIPT_DIR / "logging_utils.py").resolve()),
                "evidence": "log_video defaults saved videos to 15 fps",
                "actual_values_are_read_from_each_mp4": True,
            },
            "output_frame_index_base": 0,
            "output_mp4_contains_initial_context": False,
            "history_id_formula": f"history_id = output_frame + {args.context_frames}",
            "dataset_clip_index_formula": f"clip_index = output_frame + {args.context_frames}",
            "source_video_frame_formula": (
                "source_video_frame = output_frame + "
                f"{args.context_frames + args.source_start_offset}"
            ),
            "source_start_offset": args.source_start_offset,
            "context_frames": args.context_frames,
            "example": {
                "output_frame": 0,
                "history_id": args.context_frames,
                "dataset_clip_index": args.context_frames,
                "source_video_frame": args.context_frames + args.source_start_offset,
                "trajectory_time_sec": 0.0,
                "encoded_playback_time_sec_at_15_fps": 0.0,
            },
        },
        "coordinate_conventions": {
            "pose_source": "dataset NPZ poses aligned to the requested MineDojo trajectory",
            "raw_pose_layout": "[x, y, z, pitch_degrees, yaw_degrees]",
            "position": "camera translation from camera-to-world matrix",
            "position_unit": (
                "Minecraft block-coordinate unit from MineDojo location_stats.pos; "
                "reported as blocks, not metres"
            ),
            "orientation": (
                "full SO(3) geodesic angle from camera-to-world rotation matrices; "
                "no Euler subtraction, so yaw wraparound is handled"
            ),
            "rotation_construction": (
                "the dataset normalization negates yaw and wraps pitch/yaw to [0,360); "
                "camera-to-world rotation is R_yaw @ R_pitch"
            ),
        },
        "thresholds": {
            "minimum_first_to_revisit_sec": args.min_separation_sec,
            "endpoint_position_distance_max_blocks": args.endpoint_position_threshold,
            "endpoint_rotation_max_deg": args.endpoint_rotation_threshold_deg,
            "departure_position_distance_min_blocks": args.departure_position_threshold,
            "departure_rotation_min_deg": args.departure_rotation_threshold_deg,
            "minimum_contiguous_away_duration_sec": args.min_away_sec,
            "floating_point_comparison_tolerance": NUMERIC_TOLERANCE,
            "rationale": (
                "The endpoint tolerance is sub-block in Minecraft coordinates, not a reused "
                "MemCam metre threshold. Departure requires either a two-block translation or "
                "a 45-degree orientation change for a contiguous one-second interval."
            ),
        },
        "search_and_sampling": {
            "search_domain": (
                "all zero-based generated-output frame pairs within each requested trajectory"
            ),
            "pose_match": "endpoint position AND full-rotation thresholds",
            "revisit_episode_rule": (
                "contiguous endpoint-match frames form one return episode; the exported "
                "revisit is the earliest minimum-pose-error frame in that episode"
            ),
            "departure_rule": (
                "a contiguous interior run meeting translation OR rotation departure threshold"
            ),
            "middle_frame_rule": (
                "frame of maximum normalized departure severity inside the selected qualifying "
                "contiguous away run; never an arbitrary temporal midpoint"
            ),
            "ranking": (
                "ascending normalized endpoint pose error, then descending temporal separation, "
                "then descending away duration"
            ),
            "deduplication": (
                f"greedy revisit-time suppression within {suppression_frames} frames "
                f"({args.revisit_suppression_sec:.3f}s)"
            ),
            "maximum_candidates_per_trajectory": args.max_candidates_per_trajectory,
            "raw_verified_candidate_counts": all_candidate_counts,
            "selected_candidate_counts": {
                str(batch): len(rows_for_batch)
                for batch, rows_for_batch in selected_by_batch.items()
            },
            "selection_does_not_use_generated_pixels_or_policy_quality": True,
        },
        "mapping_verification": {
            "dataset_seed": args.dataset_seed,
            "wo_updown": args.wo_updown,
            "mapping_method": (
                "reproduce MinecraftVideoDataset.get_data_paths (including its unsorted "
                "os.listdir fallback), apply the deterministic seed shuffle, and map output "
                "batch IDs to requested dataset video/pose files"
            ),
            "same_requested_trajectory_across_policies": True,
            "legacy_retry_limitation": (
                "The historical runs do not log the final dataset source path after the dataset "
                "loader's silent retry-on-read-error behavior. Therefore this export verifies "
                "the shared requested trajectory, not an independently recorded post-retry path."
            ),
            "trajectories": mappings,
        },
        "video_metadata": video_metadata_by_batch,
        "trajectory_motion_scale_audit": pooled_motion_statistics(c2ws_by_batch),
        "ground_truth": {
            "mapping": (
                "GT source frame = output frame + source_start_offset + context_frames"
            ),
            "previews_requested": not args.no_previews,
            "gt_checked_definition": "manual visual validation",
            "gt_checked_value_in_csv": False,
            "reason": (
                "GT images are exported when decodable but are not automatically marked as a "
                "passed visual check. Pose validation and GT visual review remain separate."
            ),
        },
        "missing_data_cases": missing_cases,
        "outputs": {
            "revisit_candidates_csv": str(csv_path.resolve()),
            "trajectory_mapping_csv": str(mapping_path.resolve()),
            "preview_manifest_csv": (
                str((output_root / "preview_manifest.csv").resolve())
                if not args.no_previews
                else None
            ),
            "preview_root": (
                str((output_root / "previews").resolve()) if not args.no_previews else None
            ),
        },
    }
    provenance_path = output_root / "provenance.json"
    write_json(provenance_path, provenance)

    print("WorldMem revisit-candidate export")
    print(f"Matched trajectories: {len(videos_by_batch)}/{args.limit}")
    print(f"Verified selected candidates: {len(rows)}")
    print(f"Trajectories with candidates: {sum(bool(v) for v in selected_by_batch.values())}")
    print(f"CSV: {csv_path}")
    print(f"Trajectory mapping: {mapping_path}")
    print(f"Provenance: {provenance_path}")
    if not args.no_previews:
        print(f"Preview manifest: {output_root / 'preview_manifest.csv'}")
    if not rows:
        print("No verified returns met the declared pose and departure thresholds.")


if __name__ == "__main__":
    main()
