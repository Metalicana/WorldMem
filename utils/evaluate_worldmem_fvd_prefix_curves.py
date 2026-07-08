import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

from worldmem_eval_common import (
    discover_run_dirs,
    list_prediction_videos,
    mean_or_none,
    parse_csv,
    parse_int_csv,
    read_video_frames,
    read_worldmem_gt_frames,
    safe_round,
    video_frame_count,
    write_json,
    write_jsonl,
)


FVD_I3D_DETECTOR_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
FVD_I3D_BACKENDS = {"styleganv_i3d", "i3d_torchscript"}


def normalize_video_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


class FVDRunner:
    def __init__(
        self,
        device="cuda",
        batch_size=8,
        image_size=224,
        clip_length=16,
        clips_per_video=4,
        frame_stride=4,
        backend="styleganv_i3d",
        detector_path=None,
        detector_url=FVD_I3D_DETECTOR_URL,
        cache_dir=None,
        allow_download=True,
    ):
        import torch

        backend = "styleganv_i3d" if backend == "i3d_torchscript" else backend
        if backend not in FVD_I3D_BACKENDS:
            raise ValueError(f"Unsupported FVD backend: {backend}")
        if clip_length < 2:
            raise ValueError("--fvd_clip_length must be >= 2")
        if clips_per_video < 1:
            raise ValueError("--fvd_clips_per_video must be >= 1")
        if frame_stride < 1:
            raise ValueError("--fvd_frame_stride must be >= 1")

        self.torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested for FVD but unavailable; using CPU.")
            device = "cpu"
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.clip_length = int(clip_length)
        self.clips_per_video = int(clips_per_video)
        self.frame_stride = int(frame_stride)
        self.backend = backend
        self.detector_path = Path(detector_path).expanduser() if detector_path else None
        self.detector_url = detector_url
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else self._default_cache_dir()
        self.allow_download = bool(allow_download)
        self.resolved_detector_path = None
        self.feature_model = self._build_feature_model()

    def _default_cache_dir(self):
        cache_root = os.environ.get("XDG_CACHE_HOME")
        if cache_root:
            return Path(cache_root) / "worldmem"
        return Path.home() / ".cache" / "worldmem"

    def _resolve_i3d_detector_path(self):
        if self.detector_path is not None:
            if not self.detector_path.exists():
                raise FileNotFoundError(f"FVD detector not found: {self.detector_path}")
            return self.detector_path

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        detector_path = self.cache_dir / "i3d_torchscript.pt"
        if detector_path.exists():
            return detector_path
        if not self.allow_download:
            raise FileNotFoundError(
                "FVD I3D detector is not cached. Pass --fvd_detector_path or allow download."
            )
        print(f"Downloading FVD I3D detector to {detector_path}")
        self._download_file(self.detector_url, detector_path)
        return detector_path

    def _download_file(self, url, destination):
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Downloading the FVD detector requires requests.") from exc

        destination = Path(destination)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(tmp_path, destination)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download FVD detector from {url}. "
                "Download manually and pass --fvd_detector_path."
            ) from exc

    def _build_feature_model(self):
        detector_path = self._resolve_i3d_detector_path()
        self.resolved_detector_path = detector_path
        model = self.torch.jit.load(str(detector_path), map_location=self.device)
        return model.eval().to(self.device)

    def sample_starts(self, num_frames):
        span = (self.clip_length - 1) * self.frame_stride + 1
        if num_frames < span:
            return []
        max_start = num_frames - span
        if self.clips_per_video == 1:
            return [max_start // 2]
        starts = np.linspace(0, max_start, self.clips_per_video)
        return sorted({int(round(start)) for start in starts})

    def clip_indices(self, num_frames):
        return [
            [start + offset * self.frame_stride for offset in range(self.clip_length)]
            for start in self.sample_starts(num_frames)
        ]

    def _clip_tensor(self, clips):
        tensors = []
        for clip in clips:
            tensor = self.torch.from_numpy(np.asarray(clip)).float().to(self.device)
            tensor = tensor.permute(1, 0, 2, 3).contiguous()
            frames = tensor.permute(1, 0, 2, 3)
            frames = self.torch.nn.functional.interpolate(
                frames,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            tensors.append(frames.permute(1, 0, 2, 3))
        return self.torch.stack(tensors)

    def encode_clips(self, clips):
        tensor = self._clip_tensor(clips)
        with self.torch.inference_mode():
            features = self.feature_model(
                tensor,
                rescale=True,
                resize=False,
                return_features=True,
            )
        return features.float().detach().cpu().numpy()

    def append_features(self, feature_rows, clips):
        for start in range(0, len(clips), self.batch_size):
            feature_rows.append(self.encode_clips(clips[start : start + self.batch_size]))

    @staticmethod
    def _symmetric_matrix(matrix):
        return 0.5 * (matrix + matrix.T)

    def _trace_sqrt_product(self, sigma_a, sigma_b):
        sigma_a = self._symmetric_matrix(sigma_a)
        sigma_b = self._symmetric_matrix(sigma_b)
        eigvals, eigvecs = np.linalg.eigh(sigma_a)
        eigvals = np.clip(eigvals, 0.0, None)
        sigma_a_sqrt = (eigvecs * np.sqrt(eigvals)).dot(eigvecs.T)
        product = sigma_a_sqrt.dot(sigma_b).dot(sigma_a_sqrt)
        product = self._symmetric_matrix(product)
        product_eigvals = np.linalg.eigvalsh(product)
        return float(np.sum(np.sqrt(np.clip(product_eigvals, 0.0, None))))

    def frechet_distance(self, real_features, generated_features):
        real_features = np.asarray(real_features, dtype=np.float64)
        generated_features = np.asarray(generated_features, dtype=np.float64)
        real_mu = np.mean(real_features, axis=0)
        generated_mu = np.mean(generated_features, axis=0)
        real_sigma = np.atleast_2d(np.cov(real_features, rowvar=False))
        generated_sigma = np.atleast_2d(np.cov(generated_features, rowvar=False))
        real_sigma = self._symmetric_matrix(real_sigma)
        generated_sigma = self._symmetric_matrix(generated_sigma)
        diff = real_mu - generated_mu
        value = diff.dot(diff) + np.trace(real_sigma) + np.trace(generated_sigma)
        value -= 2.0 * self._trace_sqrt_product(real_sigma, generated_sigma)
        return float(max(value, 0.0))


def frames_to_chw_clip(frame_dict, indices):
    return np.stack(
        [
            np.transpose(normalize_video_frame(frame_dict[index]), (2, 0, 1))
            for index in indices
        ]
    )


def compute_fvd_for_run_duration(
    fvd_runner,
    run_dir,
    data_dir,
    duration_frames,
    seed,
    context_frames,
    initial_skip_frames,
    limit,
    strict=False,
):
    gen_batch = []
    gt_batch = []
    gen_features = []
    gt_features = []
    rows = []
    clip_count = 0

    for batch_idx, pred_path in list_prediction_videos(run_dir, limit=limit):
        try:
            available_frames = video_frame_count(pred_path)
            num_frames = min(int(duration_frames), available_frames)
            clip_indices = fvd_runner.clip_indices(num_frames)
            if not clip_indices:
                rows.append(
                    {
                        "batch_idx": batch_idx,
                        "status": "too_short",
                        "pred_path": str(pred_path),
                        "frames_available": available_frames,
                        "clips": 0,
                    }
                )
                continue

            flat_indices = sorted({index for indices in clip_indices for index in indices})
            gen_frames = read_video_frames(pred_path, flat_indices)
            gt_frames, dataset_video_path = read_worldmem_gt_frames(
                data_dir=data_dir,
                batch_idx=batch_idx,
                generated_frame_indices=flat_indices,
                seed=seed,
                context_frames=context_frames,
                initial_skip_frames=initial_skip_frames,
            )

            item_clips = 0
            for indices in clip_indices:
                if all(index in gen_frames and index in gt_frames for index in indices):
                    gen_batch.append(frames_to_chw_clip(gen_frames, indices))
                    gt_batch.append(frames_to_chw_clip(gt_frames, indices))
                    clip_count += 1
                    item_clips += 1
                    if len(gen_batch) >= fvd_runner.batch_size:
                        fvd_runner.append_features(gen_features, gen_batch)
                        fvd_runner.append_features(gt_features, gt_batch)
                        gen_batch.clear()
                        gt_batch.clear()

            rows.append(
                {
                    "batch_idx": batch_idx,
                    "status": "completed" if item_clips else "no_complete_clips",
                    "pred_path": str(pred_path),
                    "dataset_video_path": str(dataset_video_path),
                    "frames_available": available_frames,
                    "frames_evaluated": num_frames,
                    "clips": item_clips,
                }
            )
        except Exception as exc:
            if strict:
                raise
            rows.append(
                {
                    "batch_idx": batch_idx,
                    "status": "failed",
                    "pred_path": str(pred_path),
                    "error": repr(exc),
                    "clips": 0,
                }
            )

    if gen_batch:
        fvd_runner.append_features(gen_features, gen_batch)
        fvd_runner.append_features(gt_features, gt_batch)

    if clip_count < 2:
        return None, clip_count, rows
    value = fvd_runner.frechet_distance(np.concatenate(gt_features), np.concatenate(gen_features))
    return value, clip_count, rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute FVD prefix curves for WorldMem generated videos."
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
    parser.add_argument("--metric_batch_size", type=int, default=8)
    parser.add_argument("--fvd_clip_length", type=int, default=16)
    parser.add_argument("--fvd_clips_per_video", type=int, default=4)
    parser.add_argument("--fvd_frame_stride", type=int, default=4)
    parser.add_argument("--fvd_image_size", type=int, default=224)
    parser.add_argument("--fvd_detector_path", type=Path, default=None)
    parser.add_argument("--fvd_detector_url", type=str, default=FVD_I3D_DETECTOR_URL)
    parser.add_argument("--fvd_cache_dir", type=Path, default=None)
    parser.add_argument("--no_fvd_download", action="store_false", dest="fvd_allow_download")
    parser.set_defaults(fvd_allow_download=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    metrics_dir = args.metrics_dir or (args.output_root / "metrics" / "fvd_prefix")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    fvd_runner = FVDRunner(
        device=args.metric_device,
        batch_size=args.metric_batch_size,
        image_size=args.fvd_image_size,
        clip_length=args.fvd_clip_length,
        clips_per_video=args.fvd_clips_per_video,
        frame_stride=args.fvd_frame_stride,
        detector_path=args.fvd_detector_path,
        detector_url=args.fvd_detector_url,
        cache_dir=args.fvd_cache_dir,
        allow_download=args.fvd_allow_download,
    )

    run_dirs = discover_run_dirs(args.output_root, runs=parse_csv(args.runs))
    if not run_dirs:
        raise RuntimeError(f"No WorldMem run dirs found under {args.output_root}")

    summary_rows = []
    detail_rows = []
    summary = {
        "metric": "fvd",
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
            "fvd_clip_length": args.fvd_clip_length,
            "fvd_clips_per_video": args.fvd_clips_per_video,
            "fvd_frame_stride": args.fvd_frame_stride,
            "fvd_image_size": args.fvd_image_size,
            "fvd_detector_path": str(fvd_runner.resolved_detector_path),
        },
    }

    for run_name, run_dir in run_dirs:
        summary["by_run"][run_name] = {}
        for duration in parse_int_csv(args.eval_durations):
            duration_frames = int(duration * args.fps)
            print(f"[FVD] run={run_name} duration={duration}s frames={duration_frames}")
            value, clips, rows = compute_fvd_for_run_duration(
                fvd_runner=fvd_runner,
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
                "fvd": safe_round(value),
                "fvd_raw": value,
                "fvd_clips": clips,
                "videos": len(rows),
                "completed_videos": completed,
                "failed_videos": failed,
            }
            summary_rows.append(row)
            detail_rows.extend({**item, "run_name": run_name, "duration_sec": duration} for item in rows)
            summary["by_run"][run_name][str(duration)] = row

    summary["overall"] = {
        run_name: {
            "fvd_mean_over_durations": safe_round(
                mean_or_none(
                    summary["by_run"][run_name][str(duration)]["fvd_raw"]
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
