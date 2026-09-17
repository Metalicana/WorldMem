"""Export WorldMem retrieval-deterioration measurements from an existing run.

WorldMem makes one discrete retrieval decision per generated frame and supplies
eight selected latent frames to the generator.  The saved prediction MP4 is the
only complete RGB representation of generated history, so this diagnostic uses
its decoded frames as a proxy for the stored generated latents.  Initial-context
items use their exact dataset frames.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.worldmem_eval_common import (
    list_prediction_videos,
    resolve_dataset_video_for_batch,
    video_frame_count,
)


CACHE_SCHEMA = 1


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Refusing to write an empty retrieval decomposition")
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cosine_distances(left, right):
    return 1.0 - np.clip(np.asarray(left) @ np.asarray(right), -1.0, 1.0)


def query_measurements(generated, ground_truth, selected, target, candidates):
    selected = np.asarray(selected, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    if selected.size == 0 or candidates.size == 0:
        raise ValueError("Selected and eligible candidate sets must be nonempty")
    if len(np.unique(selected)) != len(selected):
        raise ValueError(f"Selected memory set contains duplicates: {selected.tolist()}")
    if not np.isin(selected, candidates).all():
        raise ValueError("A selected memory frame was absent from eligible history")
    if target < 0 or target >= min(len(generated), len(ground_truth)):
        raise IndexError(f"Target frame {target} is outside the feature arrays")

    target_gt = ground_truth[target]
    selected_view = cosine_distances(ground_truth[selected], target_gt)
    selected_corruption = 1.0 - np.clip(
        np.sum(generated[selected] * ground_truth[selected], axis=1), -1.0, 1.0
    )
    selected_effective = cosine_distances(generated[selected], target_gt)
    eligible_effective = cosine_distances(generated[candidates], target_gt)
    order = np.argsort(eligible_effective, kind="stable")
    best_k = order[: len(selected)]
    return {
        "selected_view_mismatch": float(np.mean(selected_view)),
        "selected_memory_corruption": float(np.mean(selected_corruption)),
        "selected_effective_mismatch": float(np.mean(selected_effective)),
        "full_oracle_effective_mismatch": float(eligible_effective[order[0]]),
        "full_oracle_best_k_effective_mismatch": float(
            np.mean(eligible_effective[best_k])
        ),
        "full_oracle_frame": int(candidates[order[0]]),
        "full_oracle_best_k_frames": [int(candidates[index]) for index in best_k],
    }


def load_retrieval_trace(path, expected_policy="unbounded"):
    metadata = {}
    retrievals = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed trace row {path}:{line_number}") from exc
            batch = row.get("global_batch_idx", row.get("batch_idx"))
            if batch is None:
                continue
            batch = int(batch)
            if row.get("event") == "memory_run_start":
                metadata[batch] = row
            elif row.get("event") == "memory_retrieval":
                key = (batch, int(row["target_frame"]), int(row["context_slot"]))
                # Resume-aware: the last trace exposure corresponds to the final MP4.
                retrievals[key] = row

    grouped = {}
    for (batch, target, _slot), row in retrievals.items():
        grouped.setdefault((batch, target), []).append(row)
    for key, rows in grouped.items():
        rows.sort(key=lambda item: int(item["context_slot"]))
        meta = metadata.get(key[0])
        if meta is None:
            raise ValueError(f"Missing memory_run_start for batch {key[0]}")
        if meta.get("memory_policy", expected_policy) != expected_policy:
            raise ValueError(
                f"Expected {expected_policy} trace; batch {key[0]} uses "
                f"{meta.get('memory_policy')}"
            )
    return grouped, metadata


def validate_batch_trace(groups, metadata, batch, context_frames, n_frames):
    meta = metadata.get(batch)
    if meta is None:
        raise ValueError(f"Missing run metadata for batch {batch}")
    if int(meta.get("context_frames", -1)) != int(context_frames):
        raise ValueError(f"Context mismatch for batch {batch}: {meta.get('context_frames')}")
    if int(meta.get("n_frames", -1)) != int(n_frames):
        raise ValueError(f"Frame-count mismatch for batch {batch}: {meta.get('n_frames')}")
    if meta.get("memory_reference_source") not in {None, "predicted"}:
        raise ValueError("Diagnostic requires generated/predicted memory content")
    memory_length = int(meta.get("memory_condition_length", 0))
    if memory_length <= 0:
        raise ValueError("Trace has no discrete memory references")

    expected_targets = list(range(int(context_frames), int(n_frames)))
    targets = sorted(target for candidate_batch, target in groups if candidate_batch == batch)
    if targets != expected_targets:
        missing = sorted(set(expected_targets) - set(targets))
        extra = sorted(set(targets) - set(expected_targets))
        raise ValueError(
            f"Incomplete/noncontiguous retrieval coverage for batch {batch}: "
            f"missing={missing[:8]} extra={extra[:8]}"
        )

    for target in targets:
        rows = groups[(batch, target)]
        slots = [int(row["context_slot"]) for row in rows]
        if slots != list(range(memory_length)):
            raise ValueError(f"Bad slot coverage at batch {batch}, target {target}: {slots}")
        selected = [int(row["selected_memory_frame"]) for row in rows]
        if len(set(selected)) != memory_length or not all(0 <= item < target for item in selected):
            raise ValueError(
                f"Invalid selected set at batch {batch}, target {target}: {selected}"
            )
        for row in rows:
            if int(row.get("candidate_count", -1)) != target:
                raise ValueError(
                    f"Unbounded candidate count mismatch at batch {batch}, target {target}: "
                    f"logged={row.get('candidate_count')} expected={target}"
                )
    return targets, memory_length


class DinoEncoder:
    def __init__(self, model_name, revision, device, batch_size):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for DINO but is unavailable")
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.processor = AutoImageProcessor.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).eval().to(self.device)
        processor_config = self.processor.to_dict()
        self.identity = {
            "model_name": model_name,
            "requested_revision": revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "model_type": getattr(self.model.config, "model_type", None),
            "processor_sha256": stable_hash(processor_config),
            "feature_source": "pooler_output_or_cls",
            "normalization": "float32_l2",
        }

    def encode(self, images, label):
        features = []
        batch = []
        count = 0
        with self.torch.inference_mode():
            for image in images:
                batch.append(image)
                if len(batch) == self.batch_size:
                    features.append(self._encode_batch(batch))
                    count += len(batch)
                    batch = []
                    if count % (self.batch_size * 10) == 0:
                        print(f"  encoded {count} {label}")
            if batch:
                features.append(self._encode_batch(batch))
                count += len(batch)
        if not features:
            raise RuntimeError(f"No frames encoded for {label}")
        print(f"  encoded {count} {label}")
        return np.concatenate(features, axis=0)

    def _encode_batch(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        features = getattr(outputs, "pooler_output", None)
        if features is None:
            features = outputs.last_hidden_state[:, 0]
        features = self.torch.nn.functional.normalize(features.float(), dim=-1)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)


def iter_video_range(path, start, count):
    from PIL import Image
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    try:
        for offset in range(int(count)):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Could not decode frame {int(start) + offset} from {path}"
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(rgb)
    finally:
        cap.release()


def load_or_extract_features(
    cache_path,
    encoder,
    gt_path,
    pred_path,
    initial_skip_frames,
    context_frames,
    n_frames,
    force=False,
):
    future_frames = int(n_frames) - int(context_frames)
    identity = {
        "schema": CACHE_SCHEMA,
        "encoder": encoder.identity,
        "ground_truth": {
            "path": str(Path(gt_path).resolve()),
            "size": Path(gt_path).stat().st_size,
            "sha256": sha256_file(gt_path),
        },
        "prediction": {
            "path": str(Path(pred_path).resolve()),
            "size": Path(pred_path).stat().st_size,
            "sha256": sha256_file(pred_path),
        },
        "mapping": {
            "initial_skip_frames": int(initial_skip_frames),
            "context_frames": int(context_frames),
            "n_frames": int(n_frames),
            "prediction_frame_zero_maps_to_local": int(context_frames),
        },
    }
    cache_path = Path(cache_path)
    if cache_path.is_file() and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_identity = json.loads(str(cached["identity_json"].item()))
            gt_features = np.asarray(cached["ground_truth"], dtype=np.float32)
            generated_features = np.asarray(cached["generated"], dtype=np.float32)
        if cached_identity == identity and gt_features.shape[0] == n_frames and generated_features.shape == gt_features.shape:
            if np.isfinite(gt_features).all() and np.isfinite(generated_features).all():
                print(f"  verified feature cache: {cache_path}")
                return generated_features, gt_features, identity
        print(f"  cache identity/shape mismatch; rebuilding {cache_path}")

    if video_frame_count(pred_path) != future_frames:
        raise RuntimeError(
            f"Prediction frame count mismatch for {pred_path}: "
            f"{video_frame_count(pred_path)} != {future_frames}"
        )
    gt_features = encoder.encode(
        iter_video_range(gt_path, initial_skip_frames, n_frames),
        label=f"GT frames from {Path(gt_path).name}",
    )
    pred_features = encoder.encode(
        iter_video_range(pred_path, 0, future_frames),
        label=f"generated frames from {Path(pred_path).name}",
    )
    if len(gt_features) != n_frames or len(pred_features) != future_frames:
        raise RuntimeError("DINO feature extraction returned an incomplete sequence")
    generated_features = np.concatenate(
        [gt_features[:context_frames], pred_features], axis=0
    ).astype(np.float32, copy=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        generated=generated_features,
        ground_truth=gt_features.astype(np.float32, copy=False),
        identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
    )
    return generated_features, gt_features, identity


def scene_identity(video_path, data_dir):
    try:
        return str(Path(video_path).relative_to(Path(data_dir) / "test"))
    except ValueError:
        return str(Path(video_path).resolve())


def find_trace(run_dir):
    paths = sorted((Path(run_dir) / "access_traces").glob("*.jsonl"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one access trace under {run_dir}; found {paths}")
    return paths[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument(
        "--run-name", default="worldmem_memquality_unbounded_60s_n15_seed101"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--duration-sec", type=int, default=60)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--context-frames", type=int, default=600)
    parser.add_argument("--initial-skip-frames", type=int, default=100)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    if args.limit <= 0 or args.fps <= 0 or args.duration_sec <= 0:
        parser.error("limit, fps, and duration-sec must be positive")
    n_frames = args.context_frames + round(args.duration_sec * args.fps)
    run_dir = args.quality_root / args.run_name
    trace = find_trace(run_dir)
    groups, metadata = load_retrieval_trace(trace)
    videos = dict(list_prediction_videos(run_dir, limit=args.limit, require_prefix=True))
    encoder = DinoEncoder(
        args.model_name, args.model_revision, args.device, args.batch_size
    )

    output_rows = []
    cache_identities = []
    for batch in range(args.limit):
        targets, memory_length = validate_batch_trace(
            groups, metadata, batch, args.context_frames, n_frames
        )
        pred_path = videos[batch]
        gt_path = resolve_dataset_video_for_batch(
            args.data_dir, batch, seed=args.dataset_seed, split="test", wo_updown=False
        )
        print(f"[batch {batch:02d}] {gt_path} | {pred_path}")
        generated, ground_truth, cache_identity = load_or_extract_features(
            args.cache_dir / f"batch_{batch:05d}.npz",
            encoder,
            gt_path,
            pred_path,
            args.initial_skip_frames,
            args.context_frames,
            n_frames,
            force=args.force_features,
        )
        cache_identities.append({"batch": batch, **cache_identity})
        meta = metadata[batch]
        scene = scene_identity(gt_path, args.data_dir)
        for target in targets:
            trace_rows = groups[(batch, target)]
            selected = [int(row["selected_memory_frame"]) for row in trace_rows]
            candidates = range(target)
            values = query_measurements(
                generated, ground_truth, selected, target, candidates
            )
            logged_count = int(trace_rows[0]["candidate_count"])
            output_rows.append(
                {
                    "run_name": args.run_name,
                    "row": batch,
                    "scene": scene,
                    "dataset_start_frame": args.initial_skip_frames,
                    "duration_sec": args.duration_sec,
                    "generation_seed": meta.get("generation_seed"),
                    "section_idx": target - args.context_frames,
                    "target_frame": target,
                    "rollout_frame": target - args.context_frames,
                    "target_original_frame": args.initial_skip_frames + target,
                    "target_horizon": int(trace_rows[0].get("target_horizon", 1)),
                    "selected_set_size": memory_length,
                    "selected_memory_frames": json.dumps(selected),
                    "selected_original_frames": json.dumps(
                        [args.initial_skip_frames + value for value in selected]
                    ),
                    "eligible_candidate_count": target,
                    "logged_candidate_count": logged_count,
                    "candidate_count_mismatch": logged_count - target,
                    "eligible_history_start": 0,
                    "eligible_history_end_exclusive": target,
                    "prediction_video": str(pred_path.resolve()),
                    "dataset_video": str(gt_path.resolve()),
                    "rgb_proxy": "saved_prediction_mp4_plus_exact_gt_context",
                    **values,
                    "full_oracle_best_k_frames": json.dumps(
                        values["full_oracle_best_k_frames"]
                    ),
                }
            )

    output_csv = args.output_dir / "query_decomposition.csv"
    write_csv(output_csv, output_rows)
    provenance = {
        "schema": 1,
        "run_name": args.run_name,
        "trace": str(trace.resolve()),
        "trace_sha256": sha256_file(trace),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "encoder": encoder.identity,
        "queries": len(output_rows),
        "trajectories": args.limit,
        "queries_per_trajectory": len(output_rows) // args.limit,
        "selected_set_reduction": "uniform mean over all eight discrete memory references actually supplied to the generator",
        "eligible_history": "unbounded pre-read archive range(0, target_frame)",
        "single_oracle": "minimum generated-to-target-GT DINO distance over eligible history",
        "best_k_oracle": "mean of the K lowest generated-to-target-GT DINO distances over eligible history, where K equals the selected set size",
        "rgb_proxy_limit": "Generated memory items are represented by frames decoded from the saved prediction MP4. The exact stored latent was not persisted. Initial-context items are exact dataset frames.",
        "target_definition": "The trace target_frame is one WorldMem generated frame because target_horizon=1. rollout_frame=target_frame-context_frames.",
        "cache_identities": cache_identities,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "export_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {output_csv}\n"
        f"Trajectories: {args.limit}; queries: {len(output_rows)}; "
        f"selected items/query: {output_rows[0]['selected_set_size']}"
    )


if __name__ == "__main__":
    main()
