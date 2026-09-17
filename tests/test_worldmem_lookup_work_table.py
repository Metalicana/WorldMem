import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lookup_work", ROOT / "utils/build_worldmem_lookup_work_table.py"
)
lookup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lookup)


class LookupWorkTableTests(unittest.TestCase):
    def test_aggregates_logged_candidate_growth(self):
        trajectories = {
            trajectory: [
                {
                    "rollout_frame": frame,
                    "candidate_count": 600 + frame,
                    "selected_set_size": 8,
                }
                for frame in range(8)
            ]
            for trajectory in range(3)
        }
        rows = lookup.aggregate_windows(
            trajectories,
            selected_size=8,
            duration_sec=8,
            fps=1,
            window_sec=2,
            fov_samples=10000,
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["mean_eligible_candidates_per_query"], 600.5)
        self.assertEqual(rows[-1]["mean_eligible_candidates_per_query"], 606.5)
        self.assertEqual(rows[0]["retrieved_memories_per_query"], 8)
        self.assertEqual(
            rows[0]["mean_candidate_fov_point_tests_per_query"], 6_005_000
        )
        self.assertIn("Candidate / retrieved", lookup.markdown(rows))
        self.assertIn("tab:worldmem_lookup_work", lookup.latex(rows))

    def test_loader_rejects_incomplete_or_mismatched_logs(self):
        fields = [
            "run_name",
            "row",
            "duration_sec",
            "rollout_frame",
            "eligible_candidate_count",
            "logged_candidate_count",
            "candidate_count_mismatch",
            "selected_set_size",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.csv"
            path.write_text(
                ",".join(fields)
                + "\n"
                + "run,0,1,0,600,599,1,8\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Candidate-count mismatch"):
                lookup.load_queries(path, "run", 1, 1, 1)


if __name__ == "__main__":
    unittest.main()
