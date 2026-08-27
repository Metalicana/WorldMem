import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "utils"
    / "build_worldmem_final_metric_status.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_worldmem_final_metric_status",
    MODULE_PATH,
)
STATUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATUS)


class FinalMetricStatusTest(unittest.TestCase):
    def test_prefix_metric_requires_exact_matched_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "run_name",
                        "duration_sec",
                        "lpips",
                        "videos",
                        "completed_videos",
                        "failed_videos",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "run_name": "matched",
                        "duration_sec": 60,
                        "lpips": 0.5,
                        "videos": 15,
                        "completed_videos": 15,
                        "failed_videos": 0,
                    }
                )
                writer.writerow(
                    {
                        "run_name": "unmatched",
                        "duration_sec": 60,
                        "lpips": 0.4,
                        "videos": 30,
                        "completed_videos": 30,
                        "failed_videos": 0,
                    }
                )
            values = STATUS.load_prefix_metric(
                path,
                "lpips",
                {"matched", "unmatched"},
                limit=15,
            )
        self.assertEqual(values[("matched", 60)], 0.5)
        self.assertNotIn(("unmatched", 60), values)

    def test_vbench_requires_manifest_and_fifteen_details(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "input_selection.json").write_text(
                json.dumps({"selected_batch_ids": list(range(15))}),
                encoding="utf-8",
            )
            payload = {
                dimension: [0.5, [{} for _ in range(15)]]
                for dimension in STATUS.DIMENSIONS
            }
            (run_dir / "results_eval_results.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            scores, source = STATUS.load_vbench_run(
                Path(directory),
                "run",
                limit=15,
            )
        self.assertEqual(set(scores), set(STATUS.DIMENSIONS))
        self.assertIsNotNone(source)

    def test_cut3r_is_ignored_without_validity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cut3r.csv"
            path.write_text(
                "run_name,videos,rotation_error_deg_mean_mean,"
                "translation_error_scale_only_mean_mean,"
                "worldscore_camera_control_score_mean\n"
                "run,15,1.0,2.0,0.8\n",
                encoding="utf-8",
            )
            values = STATUS.load_cut3r(path, False, {"run"}, limit=15)
        self.assertEqual(values, {})


if __name__ == "__main__":
    unittest.main()
