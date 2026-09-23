import json
from pathlib import Path
import tempfile
import unittest

from utils.analyze_worldmem_vbench6_pair import analyze_pair, detail_score, load_run
from utils.build_worldmem_final_metric_status import DIMENSIONS


class VBench6PairTests(unittest.TestCase):
    def test_imaging_detail_is_rescaled(self):
        self.assertAlmostEqual(
            detail_score("imaging_quality", {"video_results": 65.0}), 0.65
        )
        self.assertEqual(
            detail_score("dynamic_degree", {"video_results": True}), 1.0
        )

    def test_paired_bootstrap_uses_matched_rows(self):
        left = [
            {"batch_id": 0, "vbench6_percent": 70.0},
            {"batch_id": 1, "vbench6_percent": 80.0},
        ]
        right = [
            {"batch_id": 0, "vbench6_percent": 72.0},
            {"batch_id": 1, "vbench6_percent": 79.0},
        ]
        result = analyze_pair(left, right, samples=100, seed=3)
        self.assertAlmostEqual(result["right_minus_left"], 0.5)
        self.assertEqual(result["right_wins"], 1)
        self.assertEqual(result["left_wins"], 1)

    def test_loader_reconstructs_video_level_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            selected = []
            for batch in range(2):
                name = f"video_batch{batch:05d}_0_rank0.mp4"
                selected.append({"batch_id": batch, "source_name": name})
            (run / "input_selection.json").write_text(json.dumps({
                "selected_batch_ids": [0, 1],
                "selected_videos": selected,
            }))
            values = {
                "subject_consistency": [0.7, 0.8],
                "background_consistency": [0.8, 0.9],
                "motion_smoothness": [0.95, 0.97],
                "dynamic_degree": [0.0, 1.0],
                "aesthetic_quality": [0.4, 0.5],
                "imaging_quality": [0.6, 0.7],
            }
            payload = {}
            for dimension in DIMENSIONS:
                details = []
                for batch, value in enumerate(values[dimension]):
                    stored = value * 100 if dimension == "imaging_quality" else value
                    details.append({
                        "video_path": str(run / selected[batch]["source_name"]),
                        "video_results": stored,
                    })
                payload[dimension] = [sum(values[dimension]) / 2, details]
            (run / "results_1_eval_results.json").write_text(json.dumps(payload))
            rows, means, aggregate, _result, _manifest = load_run(root, "run", 2)
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(means["imaging_quality"], 0.65)
            self.assertAlmostEqual(
                aggregate, sum(row["vbench6_percent"] for row in rows) / 2
            )


if __name__ == "__main__":
    unittest.main()
