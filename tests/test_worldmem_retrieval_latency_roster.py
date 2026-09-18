import json
from pathlib import Path
import tempfile
import unittest

from utils.summarize_worldmem_retrieval_latency_roster import collect


class RosterLatencyTests(unittest.TestCase):
    def write(self, root, policy, budget, samples=10000):
        tag = "" if budget is None else f"_b{budget}"
        name = f"worldmem_retrieval_latency_{policy}{tag}_60s_n1_seed101"
        path = root / name / "access_traces" / f"{name}.jsonl"
        path.parent.mkdir(parents=True)
        rows = [{"event": "memory_run_start", "global_batch_idx": 0}]
        for frame in range(600):
            rows.append({"event": "retrieval_query_profile", "global_batch_idx": 0,
                         "rollout_frame": frame, "memory_policy": policy, "memory_budget": budget,
                         "retrieved_memory_count": 8, "cuda_synchronized": True,
                         "timing_scope": "generate_condition_indices", "query_milliseconds": 2,
                         "candidate_count": 600 + frame if budget is None else budget,
                         "cuda_device_name": "test GPU", "retrieval_fov_samples": samples,
                         "memory_bank_device": "cpu", "memory_reference_source": "predicted",
                         "memory_feature_backend": "latent", "retrieval_candidate_cap": None,
                         "generation_seed": 101})
        rows.append({"event": "memory_run_end", "global_batch_idx": 0})
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_full_rollout_and_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "unbounded", None)
            self.write(root, "fifo", 32)
            rows, provenance = collect(root, "unbounded: fifo:32", 1, 101)
            full = [row for row in rows if row["generated_window"] == "0-60s"]
            self.assertEqual(len(full), 2)
            self.assertTrue(all(row["queries_per_trajectory"] == 600 for row in full))
            self.assertEqual(full[0]["query_time_ms_trajectory_mean"], 2)
            self.assertEqual(full[0]["candidate_count_trajectory_mean"], 899.5)
            self.assertEqual(len(provenance), 2)

    def test_rejects_non_native_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "fifo", 32, samples=100)
            with self.assertRaisesRegex(ValueError, "10000"):
                collect(root, "fifo:32", 1, 101)


if __name__ == "__main__":
    unittest.main()
