import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "latency", ROOT / "utils/summarize_worldmem_retrieval_latency.py"
)
latency = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(latency)


class RetrievalLatencyTests(unittest.TestCase):
    def write_trace(self, path, policy, budget, trajectories=2, frames=4):
        with path.open("w", encoding="utf-8") as handle:
            for batch in range(trajectories):
                handle.write(json.dumps({
                    "event": "memory_run_start", "global_batch_idx": batch
                }) + "\n")
                for frame in range(frames):
                    row = {
                        "event": "retrieval_query_profile",
                        "memory_policy": policy,
                        "memory_budget": budget,
                        "global_batch_idx": batch,
                        "rollout_frame": frame,
                        "candidate_count": 10 + frame,
                        "retrieved_memory_count": 8,
                        "query_milliseconds": 1 + frame + batch,
                        "cuda_synchronized": True,
                        "timing_scope": "generate_condition_indices",
                    }
                    handle.write(json.dumps(row) + "\n")
                handle.write(json.dumps({
                    "event": "memory_run_end", "global_batch_idx": batch
                }) + "\n")

    def test_reads_complete_synchronized_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            self.write_trace(path, "slam_covisibility", 32)
            rows = latency.read_profile(path, "slam_covisibility", 32, 2, 4)
            self.assertEqual(len(rows), 2)
            summary = latency.summarize(
                "Ours", "slam_covisibility", 32, rows, 1, [(0, 2), (2, 4)], 0
            )
            self.assertEqual(summary[0]["candidate_count_trajectory_mean"], 10.5)
            self.assertEqual(summary[0]["query_time_ms_trajectory_mean"], 2.0)

    def test_rejects_incomplete_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            self.write_trace(path, "unbounded", None, trajectories=1, frames=3)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                latency.read_profile(path, "unbounded", None, 1, 4)


if __name__ == "__main__":
    unittest.main()
