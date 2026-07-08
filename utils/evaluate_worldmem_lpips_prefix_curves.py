import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldmem_eval_common import (
    discover_run_dirs,
    list_prediction_videos,
    mean_or_none,
    parse_csv,
    parse_int_csv,
    resolve_dataset_video_for_batch,
    safe_round,
    video_frame_count,
    write_json,
    write_jsonl,
)


def get_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for WorldMem LPIPS evaluation.") from exc
    return cv2


def read_contiguous_frames(video_path, start_frame, num_frames):
    cv2 = get_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        for _ in range(int(num_frames)):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def frame_batch_to_tensor(torch, frames):
    array = np.stack(frames)
    if array.ndim == 3:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    array = np.clip(array, 0, 255).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def resize_to_match(torch, pred, gt, image_size=None):
    import torch.nn.functional as F

    if image_size is not None:
        size = (int(image_size), int(image_size))
        pred = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
        gt = F.interpolate(gt, size=size, mode="bilinear", align_corners=False)
        return pred, gt

    if pred.shape[-2:] != gt.shape[-2:]:
        gt = F.interpolate(gt, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    return pred, gt


class LPIPSRunner:
    def __init__(self, device="cuda", batch_size=16, image_size=None):
        import torch
        from algorithms.common.metrics import LearnedPerceptualImagePatchSimilarity

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested for LPIPS but unavailable; using CPU.")
            device = "cpu"

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.image_size = int(image_size) if image_size else None
        self.metric = LearnedPerceptualImagePatchSimilarity().to(self.device).eval()

    def compute_video(self, pred_frames, gt_frames):
        if len(pred_frames) != len(gt_frames):
            raise ValueError(f"Frame count mismatch: pred={len(pred_frames)} gt={len(gt_frames)}")
        if not pred_frames:
            return None

        with self.torch.inference_mode():
            self.metric.reset()
            for start in range(0, len(pred_frames), self.batch_size):
                end = min(start + self.batch_size, len(pred_frames))
                pred = frame_batch_to_tensor(self.torch, pred_frames[start:end]).to(self.device)
                gt = frame_batch_to_tensor(self.torch, gt_frames[start:end]).to(self.device)
                pred, gt = resize_to_match(self.torch, pred, gt, image_size=self.image_size)
                self.metric.update(pred, gt)
                del pred, gt
                if self.device.type == "cuda":
                    self.torch.cuda.empty_cache()
            value = float(self.metric.compute().detach().cpu().item())
            self.metric.reset()
        return value


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


def compute_lpips_for_run_duration(
    lpips_runner,
    run_dir,
    data_dir,
    duration_frames,
    seed,
    context_frames,
    initial_skip_frames,
    limit,
    strict=False,
):
    rows = []
    weighted_scores = []
    total_frames = 0

    for batch_idx, pred_path in list_prediction_videos(run_dir, limit=limit):
        try:
            available_frames = video_frame_count(pred_path)
            num_frames = min(int(duration_frames), available_frames)
            if num_frames <= 0:
                rows.append(
                    {
                        "batch_idx": batch_idx,
                        "status": "too_short",
                        "pred_path": str(pred_path),
                        "frames_available": available_frames,
                        "frames": 0,
                    }
                )
                continue

            gt_path = resolve_dataset_video_for_batch(
                data_dir=data_dir,
                batch_idx=batch_idx,
                seed=seed,
                split="test",
                wo_updown=False,
            )
            pred_frames = read_contiguous_frames(pred_path, 0, num_frames)
            gt_start = int(initial_skip_frames) + int(context_frames)
            gt_frames = read_contiguous_frames(gt_path, gt_start, num_frames)
            usable_frames = min(len(pred_frames), len(gt_frames))
            pred_frames = pred_frames[:usable_frames]
            gt_frames = gt_frames[:usable_frames]

            if usable_frames <= 0:
                rows.append(
                    {
                        "batch_idx": batch_idx,
                        "status": "too_short",
                        "pred_path": str(pred_path),
                        "gt_path": str(gt_path),
                        "frames_available": available_frames,
                        "frames": 0,
                    }
                )
                continue

            value = lpips_runner.compute_video(pred_frames, gt_frames)
            weighted_scores.append(value * usable_frames)
            total_frames += usable_frames
            rows.append(
                {
                    "batch_idx": batch_idx,
                    "status": "completed",
                    "pred_path": str(pred_path),
                    "gt_path": str(gt_path),
                    "frames_available": available_frames,
                    "frames": usable_frames,
                    "lpips": safe_round(value),
                    "lpips_raw": value,
                }
            )
            print(
                f"[LPIPS] batch={batch_idx} frames={usable_frames} "
                f"value={value:.6f} video={pred_path.name}"
            )
        except Exception as exc:
            row = {
                "batch_idx": batch_idx,
                "status": "failed",
                "pred_path": str(pred_path),
                "error": repr(exc),
            }
            rows.append(row)
            if strict:
                raise
            print(f"[warn] LPIPS failed for {pred_path}: {exc}")

    value = float(sum(weighted_scores) / total_frames) if total_frames else None
    return value, total_frames, rows


def main():
    parser = argparse.ArgumentParser(
        description="Compute LPIPS prefix curves for WorldMem generated videos."
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--metrics_dir", type=Path, default=None)
    parser.add_argument("--runs", type=str, default=None)
    parser.add_argument("--eval_durations", type=str, default="10,20,30,60")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--context_frames", type=int, default=600)
    parser.add_argument("--initial_skip_frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metric_device", type=str, default="cuda")
    parser.add_argument("--metric_batch_size", type=int, default=16)
    parser.add_argument("--lpips_image_size", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    metrics_dir = args.metrics_dir or (args.output_root / "metrics" / "lpips_prefix")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    lpips_runner = LPIPSRunner(
        device=args.metric_device,
        batch_size=args.metric_batch_size,
        image_size=args.lpips_image_size,
    )

    run_dirs = discover_run_dirs(args.output_root, runs=parse_csv(args.runs))
    if not run_dirs:
        raise RuntimeError(f"No WorldMem run dirs found under {args.output_root}")

    summary_rows = []
    detail_rows = []
    summary = {
        "metric": "lpips",
        "by_run": {},
        "config": {
            "output_root": str(args.output_root),
            "data_dir": str(args.data_dir),
            "eval_durations": parse_int_csv(args.eval_durations),
            "fps": args.fps,
            "context_frames": args.context_frames,
            "initial_skip_frames": args.initial_skip_frames,
            "seed": args.seed,
            "limit": args.limit,
            "metric_batch_size": args.metric_batch_size,
            "lpips_image_size": args.lpips_image_size,
            "lpips_input_range": "[0, 1]",
        },
    }

    for run_name, run_dir in run_dirs:
        summary["by_run"][run_name] = {}
        for duration in parse_int_csv(args.eval_durations):
            duration_frames = int(duration * args.fps)
            print(f"[LPIPS] run={run_name} duration={duration}s frames={duration_frames}")
            value, frames, rows = compute_lpips_for_run_duration(
                lpips_runner=lpips_runner,
                run_dir=run_dir,
                data_dir=args.data_dir,
                duration_frames=duration_frames,
                seed=args.seed,
                context_frames=args.context_frames,
                initial_skip_frames=args.initial_skip_frames,
                limit=args.limit,
                strict=args.strict,
            )
            completed = sum(row.get("status") == "completed" for row in rows)
            failed = sum(row.get("status") == "failed" for row in rows)
            row = {
                "run_name": run_name,
                "duration_sec": duration,
                "lpips": safe_round(value),
                "lpips_raw": value,
                "frames": frames,
                "videos": len(rows),
                "completed_videos": completed,
                "failed_videos": failed,
            }
            summary_rows.append(row)
            detail_rows.extend({**item, "run_name": run_name, "duration_sec": duration} for item in rows)
            summary["by_run"][run_name][str(duration)] = row

    summary["overall"] = {
        run_name: {
            "lpips_mean_over_durations": safe_round(
                mean_or_none(
                    summary["by_run"][run_name][str(duration)]["lpips_raw"]
                    for duration in parse_int_csv(args.eval_durations)
                )
            )
        }
        for run_name, _run_dir in run_dirs
    }

    write_json(metrics_dir / "summary.json", summary)
    write_jsonl(metrics_dir / "details.jsonl", detail_rows)
    write_csv(metrics_dir / "summary.csv", summary_rows)
    write_csv(metrics_dir / "details.csv", detail_rows)
    print(f"Wrote: {metrics_dir / 'summary.csv'}")
    print(f"Wrote: {metrics_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
