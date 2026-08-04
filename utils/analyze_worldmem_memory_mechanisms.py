import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldmem_eval_common import (  # noqa: E402
    list_prediction_videos,
    resolve_dataset_video_for_batch,
    write_json,
)


def get_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for mechanism analysis") from exc
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


def frames_to_tensor(torch, frames):
    array = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


class FrameErrorRunner:
    def __init__(self, device="cuda", batch_size=32, compute_lpips=True):
        import torch
        import torch.nn.functional as F

        self.torch = torch
        self.F = F
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.batch_size = max(int(batch_size), 1)
        self.metric = None
        if compute_lpips:
            from algorithms.common.metrics import LearnedPerceptualImagePatchSimilarity

            self.metric = LearnedPerceptualImagePatchSimilarity().to(self.device).eval()

    def _lpips_per_frame(self, pred, gt):
        if self.metric is None:
            return np.full(len(pred), np.nan, dtype=np.float64)
        normalize = bool(getattr(self.metric, "normalize", False))
        try:
            values = self.metric.net(pred, gt, normalize=normalize)
            values = values.detach().float().reshape(len(pred), -1).mean(1)
            return values.cpu().numpy().astype(np.float64)
        except (AttributeError, TypeError):
            values = []
            for index in range(len(pred)):
                self.metric.reset()
                self.metric.update(pred[index : index + 1], gt[index : index + 1])
                values.append(float(self.metric.compute().detach().cpu().item()))
            self.metric.reset()
            return np.asarray(values, dtype=np.float64)

    def compute(self, pred_frames, gt_frames):
        if len(pred_frames) != len(gt_frames):
            raise ValueError("Prediction and GT frame counts must match")
        lpips_values = []
        mse_values = []
        with self.torch.inference_mode():
            for start in range(0, len(pred_frames), self.batch_size):
                end = min(start + self.batch_size, len(pred_frames))
                pred = frames_to_tensor(self.torch, pred_frames[start:end]).to(self.device)
                gt = frames_to_tensor(self.torch, gt_frames[start:end]).to(self.device)
                if gt.shape[-2:] != pred.shape[-2:]:
                    gt = self.F.interpolate(
                        gt,
                        size=pred.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                mse = (pred - gt).square().flatten(1).mean(1)
                mse_values.extend(mse.detach().cpu().numpy().astype(np.float64))
                lpips_values.extend(self._lpips_per_frame(pred, gt))
                del pred, gt, mse
                if self.device.type == "cuda":
                    self.torch.cuda.empty_cache()
        return np.asarray(lpips_values), np.asarray(mse_values)


def read_trace(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] skipping malformed trace line {path}:{line_number}: {exc}")
    return rows


def discover_runs(output_root, runs=None):
    output_root = Path(output_root)
    if runs:
        names = [value.strip() for value in runs.split(",") if value.strip()]
        return [(name, output_root / name) for name in names]
    return [
        (path.name, path)
        for path in sorted(output_root.glob("worldmem_diag_*"))
        if path.is_dir()
    ]


def trace_path_for_run(run_dir):
    paths = sorted((Path(run_dir) / "access_traces").glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No access trace found under {run_dir}")
    return paths[0]


def write_csv(path, rows):
    rows = list(rows)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_or_none(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def percentile_or_none(values, percentile):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.percentile(values, percentile)) if values else None


def correlation_or_none(left, right, rank=False):
    import pandas as pd

    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    if rank:
        frame = frame.rank(method="average")
    return float(frame["left"].corr(frame["right"]))


def bootstrap_mean_interval(values, samples=2000, seed=0):
    values = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if values.size == 0:
        return None, None, None
    if values.size == 1:
        value = float(values[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def clustered_bootstrap_mean_interval(values, clusters, samples=2000, seed=0):
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        if value is not None and np.isfinite(value):
            grouped[cluster].append(float(value))
    cluster_means = [float(np.mean(group)) for group in grouped.values() if group]
    return (
        *bootstrap_mean_interval(cluster_means, samples=samples, seed=seed),
        len(cluster_means),
    )


def policy_label(row):
    policy = row.get("memory_policy", "unknown")
    budget = row.get("memory_budget")
    backend = row.get("memory_feature_backend", "latent")
    source = row.get("memory_reference_source", "predicted")
    candidate_cap = row.get("retrieval_candidate_cap")
    label = policy
    if candidate_cap not in {None, "", "None"}:
        label = f"candidate-cap {int(candidate_cap)}"
    if budget not in {None, "", "None"}:
        label += f" b{int(budget)}"
    if backend != "latent":
        label += f" {backend}"
    if source != "predicted":
        label += f" ({source})"
    return label


def metadata_by_batch(trace_rows):
    metadata = {}
    for row in trace_rows:
        if row.get("event") != "memory_run_start":
            continue
        batch_idx = int(row.get("global_batch_idx", row.get("batch_idx", 0)))
        metadata[batch_idx] = row
    return metadata


def frame_error_for_memory(errors, frame_idx, context_frames, key):
    if frame_idx < context_frames:
        return 0.0
    local_idx = int(frame_idx) - int(context_frames)
    values = errors.get(key)
    if values is None or local_idx < 0 or local_idx >= len(values):
        return None
    return float(values[local_idx])


def chunk_error(errors, target_frame, horizon, context_frames, key):
    start = int(target_frame) - int(context_frames)
    end = start + int(horizon)
    values = errors.get(key)
    if values is None or start < 0 or start >= len(values):
        return None
    segment = values[start : min(end, len(values))]
    return mean_or_none(segment)


def build_error_tables(args, runs, traces, runner):
    frame_errors = {}
    video_rows = []
    cache_dir = args.metrics_dir / "frame_error_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for run_name, run_dir in runs:
        batch_meta = metadata_by_batch(traces[run_name])
        for batch_idx, pred_path in list_prediction_videos(run_dir, limit=args.limit):
            cache_path = cache_dir / run_name / f"batch_{batch_idx:05d}.npz"
            if cache_path.exists() and not args.recompute_frame_errors:
                cached = np.load(cache_path)
                lpips_values = cached["lpips"]
                mse_values = cached["mse"]
                gt_path = str(cached["gt_path"].item())
            else:
                gt_video = resolve_dataset_video_for_batch(
                    data_dir=args.data_dir,
                    batch_idx=batch_idx,
                    seed=args.dataset_seed,
                    split="test",
                    wo_updown=False,
                )
                pred_frames = read_contiguous_frames(pred_path, 0, args.max_frames)
                gt_start = args.initial_skip_frames + args.context_frames
                gt_frames = read_contiguous_frames(gt_video, gt_start, len(pred_frames))
                usable = min(len(pred_frames), len(gt_frames))
                pred_frames = pred_frames[:usable]
                gt_frames = gt_frames[:usable]
                lpips_values, mse_values = runner.compute(pred_frames, gt_frames)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    lpips=lpips_values,
                    mse=mse_values,
                    pred_path=str(pred_path),
                    gt_path=str(gt_video),
                )
                gt_path = str(gt_video)

            frame_errors[(run_name, batch_idx, "lpips")] = lpips_values
            frame_errors[(run_name, batch_idx, "mse")] = mse_values
            meta = batch_meta.get(batch_idx, {})
            row = {
                "run_name": run_name,
                "batch_idx": batch_idx,
                "memory_policy": meta.get("memory_policy"),
                "memory_budget": meta.get("memory_budget"),
                "memory_reference_source": meta.get(
                    "memory_reference_source",
                    "predicted",
                ),
                "memory_feature_backend": meta.get(
                    "memory_feature_backend",
                    "latent",
                ),
                "retrieval_candidate_cap": meta.get("retrieval_candidate_cap"),
                "generation_seed": meta.get("generation_seed"),
                "frames": len(mse_values),
                "video_lpips": mean_or_none(lpips_values),
                "video_mse": mean_or_none(mse_values),
                "pred_path": str(pred_path),
                "gt_path": gt_path,
            }
            for duration_sec in args.prefix_seconds:
                prefix_frames = min(int(duration_sec * args.fps), len(mse_values))
                row[f"lpips_{duration_sec}s"] = mean_or_none(
                    lpips_values[:prefix_frames]
                )
                row[f"mse_{duration_sec}s"] = mean_or_none(
                    mse_values[:prefix_frames]
                )
            row["policy_label"] = policy_label(row)
            video_rows.append(row)
            print(
                f"[errors] run={run_name} batch={batch_idx} frames={len(mse_values)} "
                f"lpips={row['video_lpips']} mse={row['video_mse']:.6f}"
            )
    return frame_errors, video_rows


def enrich_trace_tables(args, runs, traces, frame_errors):
    retrieval_rows = []
    candidate_rows = []
    bank_summary_rows = []
    bank_state_rows = []
    runtime_rows = []
    for run_name, _run_dir in runs:
        for raw in traces[run_name]:
            event = raw.get("event")
            if event not in {
                "memory_retrieval",
                "memory_candidate",
                "memory_bank_summary",
                "memory_bank_state",
                "runtime_breakdown",
            }:
                continue
            row = {"run_name": run_name, **raw}
            row["policy_label"] = policy_label(row)
            batch_idx = int(row.get("global_batch_idx", row.get("batch_idx", 0)))
            row["batch_idx"] = batch_idx
            if event == "memory_retrieval":
                selected = int(row["selected_memory_frame"])
                target = int(row["target_frame"])
                horizon = int(row["target_horizon"])
                row["selected_memory_lpips"] = frame_error_for_memory(
                    frame_errors,
                    selected,
                    args.context_frames,
                    (run_name, batch_idx, "lpips"),
                )
                row["selected_memory_mse"] = frame_error_for_memory(
                    frame_errors,
                    selected,
                    args.context_frames,
                    (run_name, batch_idx, "mse"),
                )
                if row.get("memory_reference_source") == "ground_truth":
                    row["selected_memory_lpips"] = 0.0
                    row["selected_memory_mse"] = 0.0
                row["next_chunk_lpips"] = chunk_error(
                    frame_errors,
                    target,
                    horizon,
                    args.context_frames,
                    (run_name, batch_idx, "lpips"),
                )
                row["next_chunk_mse"] = chunk_error(
                    frame_errors,
                    target,
                    horizon,
                    args.context_frames,
                    (run_name, batch_idx, "mse"),
                )
                row["horizon_sec"] = (target - args.context_frames) / args.fps
                retrieval_rows.append(row)
            elif event == "memory_candidate":
                candidate = int(row["candidate_frame"])
                target = int(row["target_frame"])
                row["candidate_memory_lpips"] = frame_error_for_memory(
                    frame_errors,
                    candidate,
                    args.context_frames,
                    (run_name, batch_idx, "lpips"),
                )
                row["candidate_memory_mse"] = frame_error_for_memory(
                    frame_errors,
                    candidate,
                    args.context_frames,
                    (run_name, batch_idx, "mse"),
                )
                if row.get("memory_reference_source") == "ground_truth":
                    row["candidate_memory_lpips"] = 0.0
                    row["candidate_memory_mse"] = 0.0
                row["horizon_sec"] = (target - args.context_frames) / args.fps
                candidate_rows.append(row)
            elif event == "memory_bank_summary":
                row["horizon_sec"] = (
                    int(row["chunk_end_frame"]) - args.context_frames
                ) / args.fps
                bank_summary_rows.append(row)
            elif event == "memory_bank_state":
                row["horizon_sec"] = (
                    int(row["chunk_end_frame"]) - args.context_frames
                ) / args.fps
                bank_state_rows.append(row)
            elif event == "runtime_breakdown":
                runtime_rows.append(row)
    return retrieval_rows, candidate_rows, bank_summary_rows, bank_state_rows, runtime_rows


def build_chunk_rows(retrieval_rows):
    import pandas as pd

    if not retrieval_rows:
        return []
    frame = pd.DataFrame(retrieval_rows)
    group_cols = [
        "run_name",
        "policy_label",
        "memory_policy",
        "memory_budget",
        "memory_reference_source",
        "memory_feature_backend",
        "retrieval_candidate_cap",
        "generation_seed",
        "batch_idx",
        "target_frame",
        "target_horizon",
        "horizon_sec",
    ]
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "retrieval_slots": len(group),
                "unique_retrieved_frames": int(group["selected_memory_frame"].nunique()),
                "candidate_count": float(group["candidate_count"].max()),
                "selected_memory_age_mean": float(group["selected_memory_age"].mean()),
                "selected_context_fraction": float(group["source_is_initial_context"].mean()),
                "selected_overlap_mean": float(group["selected_overlap"].mean()),
                "selected_confidence_mean": float(group["selected_confidence"].mean()),
                "selected_memory_lpips_mean": mean_or_none(group["selected_memory_lpips"]),
                "selected_memory_mse_mean": mean_or_none(group["selected_memory_mse"]),
                "next_chunk_lpips": mean_or_none(group["next_chunk_lpips"]),
                "next_chunk_mse": mean_or_none(group["next_chunk_mse"]),
            }
        )
        rows.append(row)
    return rows


def add_paired_effects(video_rows):
    baseline = {}
    for row in video_rows:
        if (
            row.get("memory_policy") == "unbounded"
            and row.get("memory_feature_backend", "latent") == "latent"
            and row.get("retrieval_candidate_cap") in {None, "", "None"}
        ):
            key = (
                row.get("generation_seed"),
                row.get("batch_idx"),
                row.get("memory_reference_source", "predicted"),
            )
            baseline[key] = row
    paired = []
    for row in video_rows:
        key = (
            row.get("generation_seed"),
            row.get("batch_idx"),
            row.get("memory_reference_source", "predicted"),
        )
        reference = baseline.get(key)
        enriched = dict(row)
        enriched["baseline_run_name"] = reference.get("run_name") if reference else None
        enriched["lpips_delta_vs_unbounded"] = (
            row["video_lpips"] - reference["video_lpips"]
            if reference and row.get("video_lpips") is not None
            else None
        )
        enriched["mse_delta_vs_unbounded"] = (
            row["video_mse"] - reference["video_mse"]
            if reference and row.get("video_mse") is not None
            else None
        )
        for key_name, value in row.items():
            if not key_name.startswith("lpips_") or not key_name.endswith("s"):
                continue
            reference_value = reference.get(key_name) if reference else None
            duration = key_name.removeprefix("lpips_").removesuffix("s")
            enriched[f"lpips_delta_{duration}s_vs_unbounded"] = (
                value - reference_value
                if value is not None and reference_value is not None
                else None
            )
        paired.append(enriched)
    return paired


def build_summary_rows(video_rows, chunk_rows, candidate_rows, runtime_rows):
    import pandas as pd

    videos = pd.DataFrame(video_rows)
    chunks = pd.DataFrame(chunk_rows)
    candidates = pd.DataFrame(candidate_rows)
    runtimes = pd.DataFrame(runtime_rows)
    rows = []
    for run_name, group in videos.groupby("run_name", dropna=False):
        first = group.iloc[0]
        row = {
            "run_name": run_name,
            "policy_label": first["policy_label"],
            "memory_policy": first["memory_policy"],
            "memory_budget": first["memory_budget"],
            "memory_reference_source": first["memory_reference_source"],
            "memory_feature_backend": first["memory_feature_backend"],
            "retrieval_candidate_cap": first.get("retrieval_candidate_cap"),
            "videos": len(group),
            "video_lpips_mean": float(group["video_lpips"].mean()),
            "video_mse_mean": float(group["video_mse"].mean()),
        }
        run_chunks = chunks[chunks["run_name"] == run_name] if not chunks.empty else chunks
        if not run_chunks.empty:
            row.update(
                {
                    "candidate_count_mean": float(run_chunks["candidate_count"].mean()),
                    "selected_memory_lpips_mean": float(
                        run_chunks["selected_memory_lpips_mean"].mean()
                    ),
                    "selected_memory_mse_mean": float(
                        run_chunks["selected_memory_mse_mean"].mean()
                    ),
                    "selected_memory_age_mean": float(
                        run_chunks["selected_memory_age_mean"].mean()
                    ),
                    "selected_context_fraction_mean": float(
                        run_chunks["selected_context_fraction"].mean()
                    ),
                    "memory_error_next_chunk_lpips_pearson": correlation_or_none(
                        run_chunks["selected_memory_lpips_mean"],
                        run_chunks["next_chunk_lpips"],
                    ),
                    "memory_error_next_chunk_lpips_spearman": correlation_or_none(
                        run_chunks["selected_memory_lpips_mean"],
                        run_chunks["next_chunk_lpips"],
                        rank=True,
                    ),
                }
            )
        run_candidates = (
            candidates[candidates["run_name"] == run_name]
            if not candidates.empty
            else candidates
        )
        if not run_candidates.empty:
            row["confidence_vs_negative_memory_error_spearman"] = correlation_or_none(
                run_candidates["candidate_confidence"],
                -run_candidates["candidate_memory_lpips"],
                rank=True,
            )
        run_runtime = runtimes[runtimes["run_name"] == run_name] if not runtimes.empty else runtimes
        if not run_runtime.empty:
            for field in (
                "total_seconds",
                "retrieval_seconds",
                "sampling_seconds",
                "memory_update_seconds",
                "decode_seconds",
            ):
                row[f"{field}_mean"] = float(run_runtime[field].mean())
        rows.append(row)
    return rows


def aggregate_paired_effects(paired_rows, prefix_seconds, bootstrap_samples):
    import pandas as pd

    frame = pd.DataFrame(paired_rows)
    rows = []
    for duration_sec in prefix_seconds:
        field = f"lpips_delta_{duration_sec}s_vs_unbounded"
        if field not in frame:
            continue
        valid = frame[frame[field].notna()]
        for label, group in valid.groupby("policy_label", dropna=False):
            mean, low, high, independent_videos = clustered_bootstrap_mean_interval(
                group[field],
                group["batch_idx"],
                samples=bootstrap_samples,
                seed=17 + int(duration_sec),
            )
            rows.append(
                {
                    "policy_label": label,
                    "duration_sec": int(duration_sec),
                    "pairs": len(group),
                    "independent_videos": independent_videos,
                    "lpips_delta_mean": mean,
                    "lpips_delta_ci95_low": low,
                    "lpips_delta_ci95_high": high,
                }
            )
    return rows


def plot_results(
    args,
    video_rows,
    paired_summary,
    chunk_rows,
    candidate_rows,
    bank_state_rows,
    runtime_rows,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    figure_dir = args.metrics_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "unbounded": "#30343B",
        "random_cap": "#2F6B9A",
        "fifo": "#D17A22",
        "rarity_irreplaceability": "#7A8F32",
        "slam_covisibility": "#B64E72",
    }

    video = pd.DataFrame(video_rows)
    chunks = pd.DataFrame(chunk_rows)
    runtimes = pd.DataFrame(runtime_rows)

    if not video.empty and not chunks.empty:
        run_candidate = chunks.groupby("run_name")["candidate_count"].mean()
        plot_frame = (
            video.groupby(
                ["run_name", "policy_label", "memory_policy", "memory_budget"],
                dropna=False,
            )["video_lpips"]
            .mean()
            .reset_index()
        )
        plot_frame["candidate_count"] = plot_frame["run_name"].map(run_candidate)
        if not runtimes.empty:
            retrieval_time = runtimes.groupby("run_name")["retrieval_seconds"].mean()
            memory_update_time = runtimes.groupby("run_name")["memory_update_seconds"].mean()
            plot_frame["retrieval_seconds"] = plot_frame["run_name"].map(retrieval_time)
            plot_frame["memory_update_seconds"] = plot_frame["run_name"].map(
                memory_update_time
            )
        aggregate_fields = [
            "video_lpips",
            "candidate_count",
            *(
                ["retrieval_seconds", "memory_update_seconds"]
                if "retrieval_seconds" in plot_frame
                else []
            ),
        ]
        plot_frame = (
            plot_frame.groupby(
                ["policy_label", "memory_policy", "memory_budget"],
                dropna=False,
            )[aggregate_fields]
            .mean()
            .reset_index()
        )
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
        quality_axis, latency_axis = axes
        for policy, group in plot_frame.groupby("memory_policy"):
            group = group.sort_values("candidate_count")
            quality_axis.plot(
                group["candidate_count"],
                group["video_lpips"],
                marker="o",
                linewidth=1.8,
                color=colors.get(policy, "#666666"),
                label=policy.replace("_", " "),
            )
            for _, row in group.iterrows():
                if policy != "random_cap":
                    quality_axis.annotate(
                        row["policy_label"],
                        (row["candidate_count"], row["video_lpips"]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                    )
            if "retrieval_seconds" in group:
                latency_axis.plot(
                    group["candidate_count"],
                    group["retrieval_seconds"],
                    marker="o",
                    linewidth=1.8,
                    color=colors.get(policy, "#666666"),
                    label=policy.replace("_", " "),
                )
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xlabel("Mean retrieval candidate count (log scale)")
            axis.grid(axis="y", color="#D9DDE2", linewidth=0.8)
        quality_axis.set_ylabel("Mean per-video LPIPS")
        quality_axis.set_title("Candidate Count and Long-Horizon Error")
        quality_axis.legend(frameon=False)
        latency_axis.set_ylabel("Retrieval time per video (seconds)")
        latency_axis.set_title("Candidate Count and Retrieval Cost")
        fig.tight_layout()
        fig.savefig(figure_dir / "candidate_count_vs_lpips.png", dpi=220)
        plt.close(fig)

        if {"retrieval_seconds", "memory_update_seconds"}.issubset(plot_frame):
            plot_frame["policy_overhead_seconds"] = (
                plot_frame["retrieval_seconds"]
                + plot_frame["memory_update_seconds"]
            )
            fig, ax = plt.subplots(figsize=(8.2, 5.6))
            for policy, group in plot_frame.groupby("memory_policy"):
                ax.scatter(
                    group["policy_overhead_seconds"],
                    group["video_lpips"],
                    s=48,
                    alpha=0.85,
                    color=colors.get(policy, "#666666"),
                    label=policy.replace("_", " "),
                )
                for _, row in group.iterrows():
                    ax.annotate(
                        row["policy_label"],
                        (row["policy_overhead_seconds"], row["video_lpips"]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                    )
            ax.set_xlabel("Retrieval + memory update time per video (seconds)")
            ax.set_ylabel("Mean per-video LPIPS")
            ax.set_title("Memory-Policy Quality and Runtime Pareto View")
            ax.grid(color="#E2E5E9", linewidth=0.7)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(figure_dir / "quality_vs_policy_overhead.png", dpi=220)
            plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.0), sharex=True)
        metrics = [
            ("candidate_count", "Candidate count"),
            ("selected_memory_lpips_mean", "Retrieved-memory LPIPS"),
            ("next_chunk_lpips", "Next-chunk LPIPS"),
        ]
        for label, group in chunks.groupby("policy_label"):
            policy = group["memory_policy"].iloc[0]
            curve = group.groupby("horizon_sec")[[item[0] for item in metrics]].mean()
            for axis, (field, ylabel) in zip(axes, metrics):
                axis.plot(
                    curve.index,
                    curve[field],
                    label=label,
                    color=colors.get(policy, "#666666"),
                    alpha=0.85,
                    linewidth=1.5,
                )
                axis.set_ylabel(ylabel)
                axis.grid(axis="y", color="#E2E5E9", linewidth=0.7)
        axes[0].set_title("Memory Candidate Growth, Pollution, and Output Error")
        axes[-1].set_xlabel("Generated duration (seconds)")
        axes[0].legend(frameon=False, ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "mechanism_over_horizon.png", dpi=220)
        plt.close(fig)

        scatter = chunks.dropna(
            subset=["selected_memory_lpips_mean", "next_chunk_lpips"]
        )
        if len(scatter) > 6000:
            scatter = scatter.sample(6000, random_state=7)
        fig, ax = plt.subplots(figsize=(7.4, 5.5))
        for policy, group in scatter.groupby("memory_policy"):
            ax.scatter(
                group["selected_memory_lpips_mean"],
                group["next_chunk_lpips"],
                s=15,
                alpha=0.24,
                color=colors.get(policy, "#666666"),
                label=policy.replace("_", " "),
            )
        ax.set_xlabel("Mean LPIPS of retrieved generated memories")
        ax.set_ylabel("LPIPS of following chunk")
        ax.set_title("Retrieved-Memory Error and Subsequent Generation Error")
        ax.grid(color="#E2E5E9", linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "memory_feedback_scatter.png", dpi=220)
        plt.close(fig)

    paired = pd.DataFrame(paired_summary)
    if not paired.empty:
        final_duration = paired["duration_sec"].max()
        final_paired = paired[paired["duration_sec"] == final_duration].sort_values(
            "lpips_delta_mean"
        )
        fig_height = max(4.0, 0.38 * len(final_paired) + 1.8)
        fig, ax = plt.subplots(figsize=(8.4, fig_height))
        positions = np.arange(len(final_paired))
        errors = np.vstack(
            [
                final_paired["lpips_delta_mean"]
                - final_paired["lpips_delta_ci95_low"],
                final_paired["lpips_delta_ci95_high"]
                - final_paired["lpips_delta_mean"],
            ]
        )
        ax.errorbar(
            final_paired["lpips_delta_mean"],
            positions,
            xerr=errors,
            fmt="o",
            color="#2F6B9A",
            ecolor="#8AAAC3",
            capsize=3,
        )
        ax.axvline(0, color="#30343B", linewidth=1)
        ax.set_yticks(positions, final_paired["policy_label"])
        ax.set_xlabel("Paired LPIPS difference versus unbounded (lower is better)")
        ax.set_title(f"Paired Memory-Policy Effects at {int(final_duration)} Seconds")
        ax.grid(axis="x", color="#E2E5E9", linewidth=0.7)
        fig.tight_layout()
        fig.savefig(figure_dir / "paired_lpips_effects.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        for label, group in paired.groupby("policy_label"):
            group = group.sort_values("duration_sec")
            ax.plot(
                group["duration_sec"],
                group["lpips_delta_mean"],
                marker="o",
                linewidth=1.5,
                label=label,
            )
            ax.fill_between(
                group["duration_sec"].to_numpy(dtype=float),
                group["lpips_delta_ci95_low"].to_numpy(dtype=float),
                group["lpips_delta_ci95_high"].to_numpy(dtype=float),
                alpha=0.1,
            )
        ax.axhline(0, color="#30343B", linewidth=1)
        ax.set_xlabel("Generated duration (seconds)")
        ax.set_ylabel("Paired LPIPS difference versus unbounded")
        ax.set_title("Policy Advantage Over the Generation Horizon")
        ax.grid(axis="y", color="#E2E5E9", linewidth=0.7)
        ax.legend(frameon=False, ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "paired_lpips_over_horizon.png", dpi=220)
        plt.close(fig)

    bank = pd.DataFrame(bank_state_rows)
    retrieval = pd.DataFrame(
        [row for row in chunk_rows]
    )
    if not bank.empty:
        first_seed = bank["generation_seed"].dropna().min()
        subset = bank[bank["generation_seed"] == first_seed]
        selected_runs = []
        for policy in (
            "unbounded",
            "random_cap",
            "rarity_irreplaceability",
            "slam_covisibility",
        ):
            matches = subset[subset["memory_policy"] == policy]
            if not matches.empty:
                selected_runs.append(matches["run_name"].iloc[0])
        if selected_runs:
            fig, axes = plt.subplots(
                len(selected_runs),
                1,
                figsize=(10.0, 2.4 * len(selected_runs)),
                squeeze=False,
            )
            for axis, run_name in zip(axes[:, 0], selected_runs):
                run = subset[subset["run_name"] == run_name]
                chunk_values = sorted(run["chunk_end_frame"].unique())
                frame_max = int(run["chunk_end_frame"].max())
                matrix = np.zeros((len(chunk_values), frame_max), dtype=np.float32)
                chunk_lookup = {value: idx for idx, value in enumerate(chunk_values)}
                if run["memory_policy"].iloc[0] == "unbounded":
                    for chunk_end, row_idx in chunk_lookup.items():
                        matrix[row_idx, : int(chunk_end)] = 1.0
                else:
                    for _, row in run.iterrows():
                        matrix[
                            chunk_lookup[row["chunk_end_frame"]],
                            int(row["retained_memory_frame"]),
                        ] = 1.0
                axis.imshow(
                    matrix,
                    aspect="auto",
                    origin="lower",
                    interpolation="nearest",
                    cmap="Blues",
                    vmin=0,
                    vmax=1,
                )
                axis.set_ylabel("Chunk")
                axis.set_title(run["policy_label"].iloc[0], loc="left", fontsize=10)
            axes[-1, 0].set_xlabel("Source frame index")
            fig.suptitle("Retained Memory Across Generation", y=1.0)
            fig.tight_layout()
            fig.savefig(figure_dir / "retention_heatmap.png", dpi=220)
            plt.close(fig)

    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.dropna(
            subset=["candidate_confidence", "candidate_memory_lpips"]
        )
        if len(candidates) > 8000:
            candidates = candidates.sample(8000, random_state=11)
        fig, ax = plt.subplots(figsize=(7.4, 5.5))
        for policy, group in candidates.groupby("memory_policy"):
            ax.scatter(
                group["candidate_confidence"],
                group["candidate_memory_lpips"],
                s=12,
                alpha=0.2,
                color=colors.get(policy, "#666666"),
                label=policy.replace("_", " "),
            )
        ax.set_xlabel("WorldMem FOV retrieval confidence")
        ax.set_ylabel("Candidate memory LPIPS at insertion timestep")
        ax.set_title("Retrieval Confidence Versus Visual Memory Quality")
        ax.grid(color="#E2E5E9", linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "confidence_vs_memory_error.png", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose why bounded WorldMem memory can beat unbounded memory."
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--metrics_dir", type=Path, default=None)
    parser.add_argument("--runs", type=str, default=None)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--context_frames", type=int, default=600)
    parser.add_argument("--initial_skip_frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--future_seconds", type=int, default=60)
    parser.add_argument("--prefix_seconds", type=str, default="10,20,30,60")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metric_device", type=str, default="cuda")
    parser.add_argument("--metric_batch_size", type=int, default=32)
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--recompute_frame_errors", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    args = parser.parse_args()

    args.metrics_dir = args.metrics_dir or (args.output_root / "mechanism_analysis")
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.max_frames = int(args.future_seconds * args.fps)
    args.prefix_seconds = sorted(
        {
            int(value.strip())
            for value in args.prefix_seconds.split(",")
            if value.strip() and int(value.strip()) <= args.future_seconds
        }
    )
    if not args.prefix_seconds:
        args.prefix_seconds = [args.future_seconds]

    runs = discover_runs(args.output_root, args.runs)
    if not runs:
        raise RuntimeError(f"No diagnostic runs found under {args.output_root}")
    traces = {
        run_name: read_trace(trace_path_for_run(run_dir))
        for run_name, run_dir in runs
    }
    runner = FrameErrorRunner(
        device=args.metric_device,
        batch_size=args.metric_batch_size,
        compute_lpips=not args.skip_lpips,
    )
    frame_errors, video_rows = build_error_tables(args, runs, traces, runner)
    (
        retrieval_rows,
        candidate_rows,
        bank_summary_rows,
        bank_state_rows,
        runtime_rows,
    ) = enrich_trace_tables(args, runs, traces, frame_errors)
    chunk_rows = build_chunk_rows(retrieval_rows)
    paired_rows = add_paired_effects(video_rows)
    summary_rows = build_summary_rows(
        video_rows,
        chunk_rows,
        candidate_rows,
        runtime_rows,
    )
    paired_summary = aggregate_paired_effects(
        paired_rows,
        prefix_seconds=args.prefix_seconds,
        bootstrap_samples=args.bootstrap_samples,
    )

    outputs = {
        "video_diagnostics.csv": paired_rows,
        "retrieval_diagnostics.csv": retrieval_rows,
        "chunk_diagnostics.csv": chunk_rows,
        "candidate_diagnostics.csv": candidate_rows,
        "bank_summary.csv": bank_summary_rows,
        "bank_state.csv": bank_state_rows,
        "runtime.csv": runtime_rows,
        "run_summary.csv": summary_rows,
        "paired_effects.csv": paired_summary,
    }
    for filename, rows in outputs.items():
        write_csv(args.metrics_dir / filename, rows)

    plot_results(
        args,
        video_rows=paired_rows,
        paired_summary=paired_summary,
        chunk_rows=chunk_rows,
        candidate_rows=candidate_rows,
        bank_state_rows=bank_state_rows,
        runtime_rows=runtime_rows,
    )
    write_json(
        args.metrics_dir / "summary.json",
        {
            "config": {
                "output_root": str(args.output_root),
                "data_dir": str(args.data_dir),
                "dataset_seed": args.dataset_seed,
                "context_frames": args.context_frames,
                "future_seconds": args.future_seconds,
                "prefix_seconds": args.prefix_seconds,
                "fps": args.fps,
                "runs": [name for name, _ in runs],
            },
            "run_summary": summary_rows,
            "paired_effects": paired_summary,
        },
    )
    print(f"Wrote mechanism analysis: {args.metrics_dir}")
    print(f"Figures: {args.metrics_dir / 'figures'}")


if __name__ == "__main__":
    main()
