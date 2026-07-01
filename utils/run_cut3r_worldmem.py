import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from worldmem_eval_common import (
    discover_run_dirs,
    list_prediction_videos,
    load_pose_c2ws_for_batch,
    parse_csv,
    video_frame_count,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def output_item_dir(output_dir, run_name, batch_idx, duration_sec):
    return output_dir / run_name / f"{duration_sec}s" / f"video_batch{batch_idx:05d}"


def has_completed_cut3r_output(run_output_dir):
    metadata_path = run_output_dir / "metadata.json"
    camera_dir = run_output_dir / "camera"
    if not metadata_path.exists() or not camera_dir.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected = len(metadata.get("video_frame_indices", []))
    actual = len(list(camera_dir.glob("*.npz")))
    return expected > 0 and actual >= expected


def extract_sampled_frames(video_path, frame_dir, duration_frames, frame_stride, max_frames):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to extract video frames for CUT3R.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open generated video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    usable_frames = min(total_frames, int(duration_frames))
    if usable_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no usable frames: {video_path}")

    indices = list(range(0, usable_frames, frame_stride))
    if max_frames is not None and len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, max_frames)
        indices = sorted({indices[int(round(pos))] for pos in positions})

    frame_paths = []
    frame_dir.mkdir(parents=True, exist_ok=True)
    for output_idx, video_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(video_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = frame_dir / f"{output_idx:06d}_video{video_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(str(frame_path))

    cap.release()
    if not frame_paths:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return {
        "frame_paths": frame_paths,
        "video_frame_indices": indices[: len(frame_paths)],
        "total_video_frames": total_frames,
        "usable_video_frames": usable_frames,
        "video_fps": fps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run CUT3R pose reconstruction on WorldMem generated videos."
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--runs", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--cut3r_root", type=Path, default=Path.home() / "MemCam" / "CUT3R")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path.home() / "MemCam" / "CUT3R" / "src" / "cut3r_512_dpt_4_64.pth",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--duration_sec", type=int, default=60)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--context_frames", type=int, default=600)
    parser.add_argument("--initial_skip_frames", type=int, default=100)
    parser.add_argument("--n_frames_valid", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame_stride", type=int, default=30)
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")
    if not args.model_path.exists():
        raise FileNotFoundError(f"CUT3R checkpoint not found: {args.model_path}")
    if not args.cut3r_root.exists():
        raise FileNotFoundError(f"CUT3R root not found: {args.cut3r_root}")

    output_dir = args.output_dir or (args.output_root / "metrics" / "cut3r_pose_recon")
    output_dir.mkdir(parents=True, exist_ok=True)
    duration_frames = int(args.duration_sec * args.fps)
    row_filter = parse_rows(args.rows)

    sys.path.insert(0, str(args.cut3r_root.resolve()))
    sys.path.insert(0, str((args.cut3r_root / "src" / "croco").resolve()))
    from add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r(str(args.model_path.resolve()))
    if args.device == "cuda":
        try:
            from models.curope import cuRoPE2D  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "CUT3R CUDA inference requires the compiled RoPE2D extension. "
                "Compile it in CUT3R/src/croco/models/curope with "
                "python setup.py build_ext --inplace."
            ) from exc

    from demo import prepare_input, prepare_output
    from src.dust3r.inference import inference
    from src.dust3r.model import ARCroco3DStereo
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUT3R requested CUDA but torch.cuda.is_available() is false.")

    print(f"Loading CUT3R model: {args.model_path}")
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device)
    model.eval()

    status_rows = []
    for run_name, run_dir in discover_run_dirs(args.output_root, runs=parse_csv(args.runs)):
        videos = list_prediction_videos(run_dir, limit=args.limit)
        for batch_idx, video_path in videos:
            if row_filter is not None and batch_idx not in row_filter:
                continue

            run_output_dir = output_item_dir(output_dir, run_name, batch_idx, args.duration_sec)
            metadata_path = run_output_dir / "metadata.json"

            if not args.force and has_completed_cut3r_output(run_output_dir):
                status_rows.append(
                    {
                        "run_name": run_name,
                        "batch_idx": batch_idx,
                        "status": "skipped_existing",
                        "output_dir": str(run_output_dir),
                    }
                )
                print(f"[skip] {run_name} batch={batch_idx} {run_output_dir}")
                continue

            if run_output_dir.exists() and args.force:
                shutil.rmtree(run_output_dir)
            run_output_dir.mkdir(parents=True, exist_ok=True)

            temp_dir = Path(tempfile.mkdtemp(prefix="worldmem_cut3r_frames_"))
            start_time = time.time()
            try:
                sampled = extract_sampled_frames(
                    video_path=video_path,
                    frame_dir=temp_dir,
                    duration_frames=duration_frames,
                    frame_stride=args.frame_stride,
                    max_frames=args.max_frames,
                )
                gt_c2ws, dataset_video_path = load_pose_c2ws_for_batch(
                    data_dir=args.data_dir,
                    batch_idx=batch_idx,
                    generated_frame_indices=sampled["video_frame_indices"],
                    seed=args.seed,
                    context_frames=args.context_frames,
                    initial_skip_frames=args.initial_skip_frames,
                    n_frames_valid=args.n_frames_valid,
                )
                np.save(run_output_dir / "gt_c2w.npy", gt_c2ws)

                print(
                    f"[CUT3R] run={run_name} batch={batch_idx} "
                    f"frames={len(sampled['frame_paths'])} video={video_path.name}"
                )
                views = prepare_input(
                    img_paths=sampled["frame_paths"],
                    img_mask=[True] * len(sampled["frame_paths"]),
                    size=args.size,
                    revisit=1,
                    update=True,
                )
                with torch.inference_mode():
                    outputs, _state_args = inference(views, model, args.device)
                    prepare_output(outputs, str(run_output_dir), revisit=1, use_pose=True)

                elapsed = round(time.time() - start_time, 3)
                metadata = {
                    "status": "completed",
                    "run_name": run_name,
                    "batch_idx": batch_idx,
                    "duration_sec": args.duration_sec,
                    "fps": args.fps,
                    "duration_frames": duration_frames,
                    "pred_video_path": str(video_path),
                    "dataset_video_path": str(dataset_video_path),
                    "cut3r_model_path": str(args.model_path),
                    "cut3r_size": args.size,
                    "frame_stride": args.frame_stride,
                    "max_frames": args.max_frames,
                    "video_frame_indices": sampled["video_frame_indices"],
                    "gt_pose_local_indices": [
                        int(args.context_frames) + int(index)
                        for index in sampled["video_frame_indices"]
                    ],
                    "total_video_frames": video_frame_count(video_path),
                    "usable_video_frames": sampled["usable_video_frames"],
                    "video_fps": sampled["video_fps"],
                    "gt_c2w_path": str(run_output_dir / "gt_c2w.npy"),
                    "time_sec": elapsed,
                }
                write_json(metadata_path, metadata)
                status_rows.append(
                    {
                        "run_name": run_name,
                        "batch_idx": batch_idx,
                        "status": "completed",
                        "output_dir": str(run_output_dir),
                        "time_sec": elapsed,
                    }
                )
            except Exception as exc:
                status_rows.append(
                    {
                        "run_name": run_name,
                        "batch_idx": batch_idx,
                        "status": "failed",
                        "output_dir": str(run_output_dir),
                        "error": repr(exc),
                    }
                )
                write_json(
                    metadata_path,
                    {
                        "status": "failed",
                        "run_name": run_name,
                        "batch_idx": batch_idx,
                        "pred_video_path": str(video_path),
                        "error": repr(exc),
                    },
                )
                raise
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
                if args.device == "cuda":
                    torch.cuda.empty_cache()

    write_json(output_dir / "cut3r_run_status.json", status_rows)
    print(f"Wrote: {output_dir / 'cut3r_run_status.json'}")


if __name__ == "__main__":
    main()
