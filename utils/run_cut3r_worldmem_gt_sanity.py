import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from run_cut3r_worldmem import has_completed_cut3r_output, output_item_dir, parse_rows
from worldmem_eval_common import (
    load_pose_c2ws_for_batch,
    read_video_frames,
    resolve_dataset_video_for_batch,
    write_json,
)


def future_frame_indices(duration_frames, frame_stride, max_frames):
    indices = list(range(0, int(duration_frames), int(frame_stride)))
    if max_frames is not None and len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, int(max_frames))
        indices = sorted({indices[int(round(pos))] for pos in positions})
    return indices


def extract_dataset_frames(video_path, frame_dir, original_indices):
    frames = read_video_frames(video_path, original_indices)
    frame_paths = []
    frame_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to write CUT3R sanity frames.") from exc

    for output_idx, original_index in enumerate(original_indices):
        frame = frames.get(original_index)
        if frame is None:
            continue
        frame_path = frame_dir / f"{output_idx:06d}_gt{original_index:06d}.jpg"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_paths.append(str(frame_path))
    if not frame_paths:
        raise RuntimeError(f"No GT frames extracted from {video_path}")
    return frame_paths


def load_cut3r_model(args):
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

    from src.dust3r.model import ARCroco3DStereo
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUT3R requested CUDA but torch.cuda.is_available() is false.")

    print(f"Loading CUT3R model: {args.model_path}")
    original_torch_load = torch.load

    def trusted_checkpoint_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_checkpoint_load
    try:
        model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device)
    finally:
        torch.load = original_torch_load
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Run CUT3R on GT Minecraft frames to sanity-check pose scoring."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--cut3r_root", type=Path, default=Path.home() / "MemCam" / "CUT3R")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path.home() / "MemCam" / "CUT3R" / "src" / "cut3r_512_dpt_4_64.pth",
    )
    parser.add_argument("--run_name", type=str, default="worldmem_gt_sanity_60s")
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
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")
    if not args.model_path.exists():
        raise FileNotFoundError(f"CUT3R checkpoint not found: {args.model_path}")
    if not args.cut3r_root.exists():
        raise FileNotFoundError(f"CUT3R root not found: {args.cut3r_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    duration_frames = int(args.duration_sec * args.fps)
    generated_indices = future_frame_indices(duration_frames, args.frame_stride, args.max_frames)
    row_filter = parse_rows(args.rows)

    model = load_cut3r_model(args)
    from demo import prepare_input, prepare_output
    from src.dust3r.inference import inference
    import torch

    if row_filter is None:
        batch_indices = list(range(int(args.limit)))
    else:
        batch_indices = sorted(row_filter)
        if args.limit is not None:
            batch_indices = batch_indices[: int(args.limit)]

    status_rows = []
    for batch_idx in batch_indices:
        run_output_dir = output_item_dir(args.output_dir, args.run_name, batch_idx, args.duration_sec)
        metadata_path = run_output_dir / "metadata.json"
        if not args.force and has_completed_cut3r_output(run_output_dir):
            status_rows.append(
                {
                    "run_name": args.run_name,
                    "batch_idx": batch_idx,
                    "status": "skipped_existing",
                    "output_dir": str(run_output_dir),
                }
            )
            print(f"[skip] {args.run_name} batch={batch_idx} {run_output_dir}")
            continue

        if run_output_dir.exists() and args.force:
            shutil.rmtree(run_output_dir)
        run_output_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = Path(tempfile.mkdtemp(prefix="worldmem_cut3r_gt_sanity_frames_"))
        start_time = time.time()
        try:
            dataset_video_path = resolve_dataset_video_for_batch(
                data_dir=args.data_dir,
                batch_idx=batch_idx,
                seed=args.seed,
                split="test",
            )
            original_indices = [
                int(args.initial_skip_frames) + int(args.context_frames) + int(index)
                for index in generated_indices
            ]
            frame_paths = extract_dataset_frames(dataset_video_path, temp_dir, original_indices)
            kept_generated_indices = generated_indices[: len(frame_paths)]
            gt_c2ws, resolved_dataset_video_path = load_pose_c2ws_for_batch(
                data_dir=args.data_dir,
                batch_idx=batch_idx,
                generated_frame_indices=kept_generated_indices,
                seed=args.seed,
                context_frames=args.context_frames,
                initial_skip_frames=args.initial_skip_frames,
                n_frames_valid=args.n_frames_valid,
            )
            np.save(run_output_dir / "gt_c2w.npy", gt_c2ws)

            print(
                f"[CUT3R GT sanity] batch={batch_idx} frames={len(frame_paths)} "
                f"video={resolved_dataset_video_path}"
            )
            views = prepare_input(
                img_paths=frame_paths,
                img_mask=[True] * len(frame_paths),
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
                "run_name": args.run_name,
                "batch_idx": batch_idx,
                "duration_sec": args.duration_sec,
                "fps": args.fps,
                "duration_frames": duration_frames,
                "pred_video_path": str(resolved_dataset_video_path),
                "dataset_video_path": str(resolved_dataset_video_path),
                "cut3r_model_path": str(args.model_path),
                "cut3r_size": args.size,
                "frame_stride": args.frame_stride,
                "max_frames": args.max_frames,
                "video_frame_indices": kept_generated_indices,
                "gt_pose_local_indices": [
                    int(args.context_frames) + int(index)
                    for index in kept_generated_indices
                ],
                "dataset_video_original_indices": original_indices[: len(frame_paths)],
                "gt_c2w_path": str(run_output_dir / "gt_c2w.npy"),
                "time_sec": elapsed,
            }
            write_json(metadata_path, metadata)
            status_rows.append(
                {
                    "run_name": args.run_name,
                    "batch_idx": batch_idx,
                    "status": "completed",
                    "output_dir": str(run_output_dir),
                    "time_sec": elapsed,
                }
            )
        except Exception as exc:
            write_json(
                metadata_path,
                {
                    "status": "failed",
                    "run_name": args.run_name,
                    "batch_idx": batch_idx,
                    "error": repr(exc),
                },
            )
            status_rows.append(
                {
                    "run_name": args.run_name,
                    "batch_idx": batch_idx,
                    "status": "failed",
                    "error": repr(exc),
                    "output_dir": str(run_output_dir),
                }
            )
            print(f"[failed] {args.run_name} batch={batch_idx}: {exc!r}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    write_json(args.output_dir / "cut3r_gt_sanity_status.json", status_rows)
    print(f"Wrote: {args.output_dir / 'cut3r_gt_sanity_status.json'}")


if __name__ == "__main__":
    main()
