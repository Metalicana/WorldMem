"""Stage an exact, auditable WorldMem batch prefix for VBench evaluation."""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


VIDEO_RE = re.compile(
    r"^video_batch(?P<batch>\d+)_(?P<sample>\d+)_rank(?P<rank>\d+)"
    r"(?:_step.*)?\.mp4$"
)


def git_commit(repo_root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def gpu_models():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def discover_batch_videos(source_dir):
    videos = {}
    duplicates = {}
    for path in sorted(Path(source_dir).glob("video_batch*.mp4")):
        match = VIDEO_RE.match(path.name)
        if match is None:
            continue
        if int(match.group("sample")) != 0 or int(match.group("rank")) != 0:
            continue
        batch_id = int(match.group("batch"))
        if path.stat().st_size <= 4096:
            continue
        if batch_id in videos:
            duplicates.setdefault(batch_id, [videos[batch_id]]).append(path)
        else:
            videos[batch_id] = path
    if duplicates:
        formatted = ", ".join(
            f"{batch}: {[path.name for path in paths]}"
            for batch, paths in sorted(duplicates.items())
        )
        raise RuntimeError(f"Duplicate prediction videos by batch ID: {formatted}")
    return videos


def stage_inputs(source_dir, stage_dir, limit, reset_derived=False):
    source_dir = Path(source_dir).resolve()
    stage_dir = Path(stage_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {source_dir}")
    if limit < 1:
        raise ValueError("limit must be positive")

    videos = discover_batch_videos(source_dir)
    selected_ids = list(range(int(limit)))
    missing = [batch_id for batch_id in selected_ids if batch_id not in videos]
    if missing:
        raise RuntimeError(
            f"Required matched batches are missing from {source_dir}: {missing}"
        )

    stage_dir.mkdir(parents=True, exist_ok=True)
    for path in stage_dir.glob("video_batch*.mp4"):
        if path.is_symlink() or path.is_file():
            path.unlink()
    split_dir = stage_dir / "split_clip"
    if reset_derived and split_dir.exists():
        shutil.rmtree(split_dir)

    selected = []
    for batch_id in selected_ids:
        source = videos[batch_id].resolve()
        staged = stage_dir / source.name
        staged.symlink_to(source)
        selected.append(
            {
                "batch_id": batch_id,
                "source_name": source.name,
                "source_path": str(source),
                "size_bytes": source.stat().st_size,
            }
        )
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--dimensions", required=True)
    parser.add_argument("--reset-derived", action="store_true")
    args = parser.parse_args()

    selected = stage_inputs(
        args.source_dir,
        args.stage_dir,
        args.limit,
        reset_derived=args.reset_derived,
    )
    payload = {
        "protocol": "worldmem_matched_batch_prefix_v1",
        "run_name": args.run_name,
        "limit": int(args.limit),
        "selected_batch_ids": [row["batch_id"] for row in selected],
        "selected_videos": selected,
        "source_dir": str(args.source_dir.resolve()),
        "stage_dir": str(args.stage_dir.resolve()),
        "evaluation_mode": args.mode,
        "dimensions": args.dimensions.split(),
        "git_commit": git_commit(args.repo_root),
        "gpu_models": gpu_models(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"Staged {len(selected)} videos for {args.run_name}: "
        f"batches {payload['selected_batch_ids']}"
    )
    print(f"Selection metadata: {args.metadata_path}")


if __name__ == "__main__":
    main()
