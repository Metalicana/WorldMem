"""CPU-only input audit for the WorldMem common-source retention/selection study.

Does not generate video, load neural models, extract features, or compute gaps.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.export_worldmem_retrieval_deterioration import sha256_file
from utils.worldmem_eval_common import parse_prediction_name


CONFIGS = [("worldmem_unbounded_60s_n30", "unbounded", None)] + [
    (f"worldmem_{policy}_b{budget}_60s_n{count}", policy, budget)
    for policy, count in (
        ("fifo", 30), ("rarity_irreplaceability", 30),
        ("slam_covisibility", 30), ("kcenter_coreset", 15), ("mce", 15)
    )
    for budget in (16, 32, 64, 128)
]


def read_attempts(paths):
    """Attach initialization evictions preceding run_start to the following attempt."""
    attempts = defaultdict(list)
    for path in paths:
        pending = []
        active = None
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                kind = event.get("event")
                if kind == "memory_eviction" and event.get("phase") == "initial_context":
                    pending.append(event)
                    continue
                if kind == "memory_run_start":
                    batch = event.get("global_batch_idx", event.get("dataset_batch_idx", event.get("batch_idx")))
                    if batch is None:
                        raise ValueError(f"No trajectory identifier at {path}:{line_number}")
                    batch = int(batch)
                    for eviction in pending:
                        if eviction.get("global_batch_idx", batch) != batch:
                            raise ValueError("Initialization evictions have inconsistent trajectory IDs")
                    active = {"metadata": event, "events": pending, "trace": str(path), "complete": False}
                    pending = []
                    attempts[batch].append(active)
                    continue
                if kind in {"memory_retrieval", "memory_eviction", "memory_run_end"}:
                    if active is None:
                        raise ValueError(f"Unscoped {kind} at {path}:{line_number}")
                    batch = active["metadata"].get("global_batch_idx", active["metadata"].get("dataset_batch_idx", active["metadata"].get("batch_idx")))
                    if event.get("global_batch_idx", batch) != batch:
                        raise ValueError("Event belongs to a different trajectory")
                    active["events"].append(event)
                    if kind == "memory_run_end":
                        active["complete"] = True
                        active = None
        if pending:
            raise ValueError("Trailing initialization evictions have no run_start")
    return attempts


def reconstruct_attempt(attempt, policy, budget, context=600, total=1200, k=8):
    """Reconstruct add-all-frames policies from their actual eviction/read logs."""
    meta = attempt["metadata"]
    if meta.get("memory_policy") != policy:
        raise ValueError("Policy differs from requested configuration")
    if budget is not None and int(meta.get("memory_budget", -1)) != budget:
        raise ValueError("Budget differs from requested configuration")
    for key, expected in (("context_frames", context), ("n_frames", total), ("memory_condition_length", k), ("batch_size", 1)):
        if int(meta.get(key, -1)) != expected:
            raise ValueError(f"Unexpected {key}: {meta.get(key)}")
    if meta.get("memory_reference_source") not in {None, "predicted"}:
        raise ValueError("Memory source is explicitly non-predicted")
    if meta.get("retrieval_candidate_cap") not in {None, ""}:
        raise ValueError("Candidate-cap override requires a separately audited eligibility adapter")
    if meta.get("gt_memory_replay_target_frame") not in {None, ""}:
        raise ValueError("GT memory-cleaning replay is not a primary policy rollout")
    if policy not in {"unbounded", "fifo", "rarity_irreplaceability", "slam_covisibility", "kcenter_coreset", "mce"}:
        raise ValueError("Admission rules are not supported by this add-all-frames adapter")

    bank = set(range(context))
    reads = defaultdict(list)
    snapshots = {}
    evictions = set()
    last_written_end = context - 1
    for event in attempt["events"]:
        if int(event.get("batch_index", 0)) != 0:
            raise ValueError("Adapter requires one sample per batch")
        if event["event"] == "memory_eviction":
            frame = int(event["evicted_memory_frame"])
            if policy == "unbounded":
                raise ValueError("Unbounded trace contains evictions")
            if event.get("phase") == "initial_context":
                if snapshots:
                    raise ValueError("Initialization eviction after a read")
                if frame not in bank:
                    raise ValueError("Invalid or duplicate initialization eviction")
            elif event.get("phase") == "generation":
                end = int(event["section_end_frame"])
                if end < last_written_end or end >= total:
                    raise ValueError("Invalid write/eviction timestamp")
                bank.update(range(last_written_end + 1, end + 1))
                last_written_end = end
                if frame not in bank:
                    raise ValueError("Generation eviction removed an absent item")
            else:
                raise ValueError("Unknown eviction phase")
            bank.remove(frame)
            evictions.add(frame)
        elif event["event"] == "memory_retrieval":
            target = int(event["target_frame"])
            if not context <= target < total:
                raise ValueError("Query outside the generated rollout")
            if int(event.get("target_horizon", -1)) != 1:
                raise ValueError("Expected actual one-frame reads; chunk protocol differs")
            if event.get("fallback_reason"):
                raise ValueError("Fallback/padded read is not an unconstrained distinct set")
            if target not in snapshots:
                bank.update(range(last_written_end + 1, target))
                last_written_end = target - 1
                retained = sorted(bank)
                if not retained or retained[0] < 0 or retained[-1] >= target:
                    raise ValueError("M is not a subset of eligible historical H")
                if len(retained) < k or (budget is not None and len(retained) > budget):
                    raise ValueError("Insufficient candidates or bank exceeds B")
                if policy != "fifo" and policy != "unbounded" and 0 not in bank:
                    raise ValueError("Expected pinned initial frame 0 is absent")
                if target - 1 not in bank:
                    raise ValueError("Expected latest observed endpoint is absent")
                snapshots[target] = retained
            retained = snapshots[target]
            if int(event.get("candidate_count", -1)) != len(retained):
                raise ValueError(f"Bank/candidate-count mismatch at target {target}")
            if int(event.get("stored_memory_size", -1)) != len(retained):
                raise ValueError(f"Stored-bank count mismatch at target {target}")
            if int(event["selected_memory_frame"]) not in retained:
                raise ValueError("R is not a subset of M")
            reads[target].append(event)

    if sorted(reads) != list(range(context, total)):
        raise ValueError(f"Incomplete query coverage: {len(reads)}/{total-context}")
    selected = {}
    for target, records in reads.items():
        records.sort(key=lambda row: int(row["context_slot"]))
        if [int(row["context_slot"]) for row in records] != list(range(k)):
            raise ValueError(f"Duplicate or missing slots at target {target}")
        indices = [int(row["selected_memory_frame"]) for row in records]
        if len(set(indices)) != k:
            raise ValueError(f"Repeated selected IDs at target {target}")
        selected[target] = indices
    return snapshots, selected, evictions


def choose_attempt(attempts):
    """Select one unambiguous read stream; completion is a separate audit fact."""
    if not attempts:
        raise ValueError("No run_start-scoped attempt found for this trajectory")
    completed = [attempt for attempt in attempts if attempt["complete"]]
    if len(completed) > 1:
        raise ValueError("Multiple completed attempts: video/read-stream association is ambiguous")
    if completed:
        if attempts[-1] is not completed[0]:
            raise ValueError("A later unfinished attempt makes video/read-stream association ambiguous")
        return completed[0], "end_marker_observed"
    # Structural validation below still requires every query and every selected slot.
    # Do not turn this missing marker into an asserted completed generation.
    return attempts[-1], "end_marker_missing"


def audit_cache(path, source_video, source_run, batch):
    with np.load(path, allow_pickle=False) as cached:
        identity = json.loads(str(cached["identity_json"].item()))
        source = np.asarray(cached["generated"])
        gt = np.asarray(cached["ground_truth"])
    for features in (source, gt):
        if features.shape != (1200, 768) or features.dtype != np.float32:
            raise ValueError(f"Unexpected DINO shape/dtype: {features.shape}/{features.dtype}")
        if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=.002, rtol=0):
            raise ValueError("Nonfinite or non-normalized features")
    mapping = identity["mapping"]
    expected = {"initial_skip_frames": 100, "context_frames": 600, "n_frames": 1200, "prediction_frame_zero_maps_to_local": 600}
    if mapping != expected:
        raise ValueError("Feature history/dataset mapping differs")
    encoder = identity["encoder"]
    if encoder.get("model_name") != "facebook/dinov2-base" or not encoder.get("resolved_revision") or not encoder.get("processor_sha256"):
        raise ValueError("Missing or incompatible encoder provenance")
    if encoder.get("normalization") != "float32_l2" or encoder.get("feature_source") != "pooler_output_or_cls":
        raise ValueError("Incompatible feature definition")
    prediction = identity["prediction"]
    cached_prediction = Path(prediction["path"])
    if cached_prediction.resolve() != source_video.resolve() or prediction["sha256"] != sha256_file(source_video):
        raise ValueError("Cache belongs to a different or changed common-source video")
    if source_run not in source_video.parts:
        raise ValueError("Source video does not belong to declared run")
    gt_video = Path(identity["ground_truth"]["path"])
    if identity["ground_truth"]["sha256"] != sha256_file(gt_video):
        raise ValueError("GT cache source changed")
    if not np.allclose(source[:600], gt[:600], atol=1e-6, rtol=0):
        raise ValueError("Initial observed-context features do not match GT/input features")
    return {"batch": batch, "cache": str(path), "cache_sha256": sha256_file(path), "identity": identity}


def videos_by_batch(run_dir):
    result = defaultdict(list)
    for path in (run_dir / "videos/test_vis/pred").glob("video_batch*.mp4"):
        parsed = parse_prediction_name(path)
        if parsed and parsed["sample_idx"] == 0 and path.stat().st_size > 0:
            result[parsed["batch_idx"]].append(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-run", default="worldmem_memquality_unbounded_60s_n15_seed101")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-videos", type=int, default=15)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory to preserve previous audit evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_root / args.source_run
    source_videos = videos_by_batch(source_dir)
    cache_records, source_metadata = {}, {}
    source_errors = {}
    try:
        source_attempts = read_attempts(sorted((source_dir / "access_traces").glob("*.jsonl")))
        for batch in range(args.expected_videos):
            completed = [attempt for attempt in source_attempts[batch] if attempt["complete"]]
            if len(completed) != 1:
                raise ValueError(f"Common source batch {batch}: expected one completed attempt")
            reconstruct_attempt(completed[0], "unbounded", None)
            source_metadata[batch] = completed[0]["metadata"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        source_errors["trace"] = str(exc)
    for batch in range(args.expected_videos):
        try:
            if len(source_videos[batch]) != 1:
                raise ValueError("Expected one unambiguous source video")
            cache_records[batch] = audit_cache(
                args.cache_dir / f"batch_{batch:05d}.npz", source_videos[batch][0], args.source_run, batch
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            source_errors[str(batch)] = str(exc)

    coverage, manifests = [], []
    all_trace_paths = set((source_dir / "access_traces").glob("*.jsonl"))
    for run, policy, budget in CONFIGS:
        directory = args.suite_root / run
        videos = videos_by_batch(directory)
        traces = sorted((directory / "access_traces").glob("*.jsonl"))
        all_trace_paths.update(traces)
        parse_error = ""
        try:
            attempts = read_attempts(traces)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            attempts, parse_error = {}, str(exc)
        valid = 0
        for batch in range(args.expected_videos):
            record = {"run_name": run, "policy": policy, "budget": budget, "trajectory_id": batch,
                      "run_path": str(directory), "video_paths": json.dumps([str(path) for path in videos[batch]]),
                      "trace_paths": json.dumps([str(path) for path in traces]),
                      "video_present": len(videos[batch]) == 1, "trace_status": "invalid", "queries": 0,
                      "raw_query_coverage": len({str(event.get("target_frame")) for attempt in attempts.get(batch, []) for event in attempt["events"] if event.get("event") == "memory_retrieval" and event.get("target_frame") is not None}),
                      "cache_status": "verified" if batch in cache_records else "missing_or_invalid",
                      "source_seed_status": "unverified", "policy_generation_seed": "",
                      "source_generation_seed": source_metadata.get(batch, {}).get("generation_seed"),
                      "memory_source_status": "unverified", "completion_status": "unverified",
                      "attempt_count": len(attempts.get(batch, [])), "dataset_identity_status": "not_logged",
                      "checkpoint_config_status": "not_audited", "error": parse_error}
            try:
                if parse_error:
                    raise ValueError(parse_error)
                attempt, completion_status = choose_attempt(attempts.get(batch, []))
                meta = attempt["metadata"]
                memory_source_status = (
                    "unlogged_legacy_metadata"
                    if meta.get("memory_reference_source") is None
                    else f"observed_{meta['memory_reference_source']}"
                )
                source_seed = source_metadata.get(batch, {}).get("generation_seed")
                seed = meta.get("generation_seed")
                seed_status = (
                    "missing" if seed is None or source_seed is None
                    else "matched" if seed == source_seed else "mismatch"
                )
                record.update(source_seed_status=seed_status, policy_generation_seed=seed,
                              memory_source_status=memory_source_status, completion_status=completion_status)
                snapshots, selected, _ = reconstruct_attempt(attempt, policy, budget)
                record.update(trace_status="valid", queries=len(selected), error="")
                manifest = {"run_name": run, "policy": policy, "budget": budget, "trajectory_id": batch,
                            "trace_metadata": meta, "trace_path": attempt["trace"],
                            "memory_source_status": memory_source_status, "completion_status": completion_status,
                            "video_paths": [str(path) for path in videos[batch]],
                            "available_config_paths": [str(path) for path in directory.rglob("config.yaml")],
                            "selected_initial_context_exposures": sum(i < 600 for ids in selected.values() for i in ids),
                            "min_retained": min(map(len, snapshots.values())), "max_retained": max(map(len, snapshots.values())),
                            "identity_limit": "Trace has no actual dataset source path after retry. Batch IDs/seed alone do not establish matched scene identity."}
                manifests.append(manifest)
                valid += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record["error"] = str(exc)
            coverage.append(record)
        matching_seeds = sum(row["source_seed_status"] == "matched" for row in coverage if row["run_name"] == run)
        print(f"{run:<57} videos={sum(len(videos[b]) == 1 for b in range(args.expected_videos)):2}/{args.expected_videos} banks/reads={valid:2}/{args.expected_videos} source-seed={matching_seeds:2}/{args.expected_videos}", flush=True)
    with (args.output / "coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    trace_paths = sorted(str(path) for path in all_trace_paths)
    report = {"stage": "input_audit_only", "source_run": args.source_run, "source_errors": source_errors,
              "source_metadata": source_metadata, "verified_caches": cache_records, "manifests": manifests,
              "trace_hashes": {path: sha256_file(path) for path in trace_paths},
              "adapter_sha256": sha256_file(Path(__file__)),
              "reader_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
                  ROOT / "algorithms/worldmem/df_video.py", ROOT / "algorithms/worldmem/memory_policies.py",
                  ROOT / "datasets/video/minecraft_video_dataset.py", ROOT / "datasets/video/base_video_dataset.py")},
              "limitations": ["Video existence/nonempty-file checks are not full video decoding.", "Current reader hashes are not historical rollout reader revisions.", "Existing legacy DINO cache arrays retain source/GT separation but processor configuration is represented by its saved hash."],
              "historical_source_evidence": {
                  "predicted_only_example_revision": "339722c",
                  "reference_gather": "xs_pred[random_idx[:, range(xs_pred.shape[1])], range(xs_pred.shape[1])]",
                  "configurable_source_introduced_revision": "6a33105",
                  "interpretation": "This establishes legacy predicted-only code behavior, not the exact runtime revision of an unversioned trace. Missing metadata is not silently rewritten. Valid bank/read structure is separate from verified protocol/completion."},
              "cohort": list(range(args.expected_videos)), "query_ids": list(range(600, 1200)),
              "ready_for_analysis": False,
              "remaining_audit": ["Verify actual dataset scene/start identity and checkpoint/config per trajectory; these are not established by the current read trace.", "Resolve source generation-seed mismatches explicitly before claiming a matched source.", "Review coverage.csv for missing/invalid historical bank reconstruction."]}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified common-source caches: {len(cache_records)}/{args.expected_videos}")
    print("Gap computation not launched: cohort/config audit remains required.")
    print(f"Wrote: {args.output / 'coverage.csv'}")


if __name__ == "__main__":
    main()
