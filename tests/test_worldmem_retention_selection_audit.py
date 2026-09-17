import importlib.util
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gap_audit", ROOT / "utils/audit_worldmem_retention_selection.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def synthetic_attempt(policy="fifo", budget=2):
    meta = {"global_batch_idx": 0, "memory_policy": policy, "memory_budget": budget,
            "context_frames": 6, "n_frames": 10, "memory_condition_length": 2,
            "batch_size": 1, "memory_reference_source": "predicted"}
    events = []
    bank = list(range(6))
    if budget:
        for frame in range(6 - budget):
            events.append({"event": "memory_eviction", "phase": "initial_context", "evicted_memory_frame": frame})
            bank.remove(frame)
    for target in range(6, 10):
        for slot, frame in enumerate(bank[-2:]):
            events.append({"event": "memory_retrieval", "target_frame": target,
                           "target_horizon": 1, "context_slot": slot,
                           "selected_memory_frame": frame, "candidate_count": len(bank),
                           "stored_memory_size": len(bank), "fallback_reason": None})
        bank.append(target)
        if budget:
            frame = bank.pop(0)
            events.append({"event": "memory_eviction", "phase": "generation",
                           "section_end_frame": target, "evicted_memory_frame": frame})
    return {"metadata": meta, "events": events, "complete": True}


class RetentionSelectionAuditTests(unittest.TestCase):
    def test_fifo_uses_pre_read_state_and_counts_initial_context_inside_budget(self):
        banks, selected, removed = audit.reconstruct_attempt(
            synthetic_attempt(), "fifo", 2, context=6, total=10, k=2
        )
        self.assertEqual(banks[6], [4, 5])
        self.assertEqual(banks[7], [5, 6])
        self.assertEqual(selected[7], [5, 6])
        self.assertIn(0, removed)

    def test_unbounded_retains_exact_eligible_history(self):
        banks, _, _ = audit.reconstruct_attempt(
            synthetic_attempt("unbounded", None), "unbounded", None, context=6, total=10, k=2
        )
        self.assertEqual(banks[9], list(range(9)))

    def test_rejects_duplicates_missing_slots_future_ids_and_wrong_bank_counts(self):
        for key, value, message in (
            ("selected_memory_frame", 5, "Repeated selected"),
            ("context_slot", 1, "missing slots"),
            ("selected_memory_frame", 6, "subset"),
            ("candidate_count", 99, "candidate-count"),
        ):
            with self.subTest(key=key, value=value):
                attempt = synthetic_attempt()
                read = next(row for row in attempt["events"] if row["event"] == "memory_retrieval")
                read[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    audit.reconstruct_attempt(attempt, "fifo", 2, 6, 10, 2)

    def test_rejects_incomplete_queries_and_candidate_cap(self):
        attempt = synthetic_attempt()
        attempt["events"] = [row for row in attempt["events"] if row.get("target_frame") != 9]
        with self.assertRaisesRegex(ValueError, "Incomplete query"):
            audit.reconstruct_attempt(attempt, "fifo", 2, 6, 10, 2)
        attempt = synthetic_attempt()
        attempt["metadata"]["retrieval_candidate_cap"] = 2
        with self.assertRaisesRegex(ValueError, "Candidate-cap"):
            audit.reconstruct_attempt(attempt, "fifo", 2, 6, 10, 2)

    def test_initial_evictions_before_start_are_scoped_to_following_attempt(self):
        attempt = synthetic_attempt()
        events = attempt["events"]
        initial = [row for row in events if row.get("phase") == "initial_context"]
        remaining = [row for row in events if row not in initial]
        start = {"event": "memory_run_start", **attempt["metadata"]}
        end = {"event": "memory_run_end", "global_batch_idx": 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in initial + [start] + remaining + [end]))
            parsed = audit.read_attempts([path])
            banks, _, _ = audit.reconstruct_attempt(parsed[0][0], "fifo", 2, 6, 10, 2)
            self.assertEqual(banks[6], [4, 5])
            self.assertTrue(parsed[0][0]["complete"])

    def test_stale_source_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source_run"
            root.mkdir()
            source = root / "video_batch00000_0_rank0.mp4"
            source.write_bytes(b"video")
            features = np.zeros((1200, 768), dtype=np.float32)
            features[:, 0] = 1
            identity = {
                "mapping": {"initial_skip_frames": 100, "context_frames": 600, "n_frames": 1200, "prediction_frame_zero_maps_to_local": 600},
                "encoder": {"model_name": "facebook/dinov2-base", "resolved_revision": "revision", "processor_sha256": "processor", "normalization": "float32_l2", "feature_source": "pooler_output_or_cls"},
                "prediction": {"path": str(source), "sha256": "stale"},
            }
            cache = root / "batch_00000.npz"
            np.savez(cache, generated=features, ground_truth=features, identity_json=json.dumps(identity))
            with self.assertRaisesRegex(ValueError, "changed common-source"):
                audit.audit_cache(cache, source, "source_run", 0)

    def test_cli_exports_full_coverage_and_preserves_source_seed_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            source_root = root / "source"
            for parent, run, seed in ((suite, "worldmem_unbounded_60s_n30", 42), (source_root, "source_run", 101)):
                run_dir = parent / run
                (run_dir / "videos/test_vis/pred").mkdir(parents=True)
                (run_dir / "videos/test_vis/pred/video_batch00000_0_rank0.mp4").write_bytes(b"video")
                (run_dir / "access_traces").mkdir()
                meta = {"event": "memory_run_start", "global_batch_idx": 0,
                        "memory_policy": "unbounded", "memory_budget": None,
                        "context_frames": 600, "n_frames": 1200, "memory_condition_length": 8,
                        "batch_size": 1, "memory_reference_source": "predicted", "generation_seed": seed}
                with (run_dir / "access_traces/trace.jsonl").open("w") as handle:
                    handle.write(json.dumps(meta) + "\n")
                    for target in range(600, 1200):
                        for slot in range(8):
                            handle.write(json.dumps({"event": "memory_retrieval", "target_frame": target,
                                "target_horizon": 1, "context_slot": slot, "selected_memory_frame": slot,
                                "candidate_count": target, "stored_memory_size": target}) + "\n")
                    handle.write(json.dumps({"event": "memory_run_end", "global_batch_idx": 0}) + "\n")
            output = root / "audit"
            result = subprocess.run([sys.executable, str(ROOT / "utils/audit_worldmem_retention_selection.py"),
                "--suite-root", str(suite), "--source-root", str(source_root), "--source-run", "source_run",
                "--cache-dir", str(root / "cache"), "--expected-videos", "1", "--output", str(output)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "coverage.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 21)
            self.assertEqual(rows[0]["trace_status"], "valid")
            self.assertEqual(rows[0]["source_seed_status"], "mismatch")
            self.assertEqual(rows[1]["trace_status"], "invalid")
            self.assertFalse(json.loads((output / "audit.json").read_text())["ready_for_analysis"])

    def test_missing_source_field_does_not_destroy_bank_reconstruction(self):
        attempt = synthetic_attempt()
        del attempt["metadata"]["memory_reference_source"]
        banks, _, _ = audit.reconstruct_attempt(attempt, "fifo", 2, 6, 10, 2)
        self.assertEqual(banks[6], [4, 5])
        attempt["metadata"]["memory_reference_source"] = "ground_truth"
        with self.assertRaisesRegex(ValueError, "explicitly non-predicted"):
            audit.reconstruct_attempt(attempt, "fifo", 2, 6, 10, 2)

    def test_missing_end_marker_is_reported_not_promoted_to_completion(self):
        attempt = synthetic_attempt()
        attempt["complete"] = False
        selected, status = audit.choose_attempt([attempt])
        self.assertIs(selected, attempt)
        self.assertEqual(status, "end_marker_missing")
        banks, _, _ = audit.reconstruct_attempt(selected, "fifo", 2, 6, 10, 2)
        self.assertEqual(len(banks), 4)
        completed = synthetic_attempt()
        with self.assertRaisesRegex(ValueError, "later unfinished"):
            audit.choose_attempt([completed, attempt])
        with self.assertRaisesRegex(ValueError, "Multiple completed"):
            audit.choose_attempt([completed, synthetic_attempt()])


if __name__ == "__main__":
    unittest.main()
