import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils" / "visualize_worldmem_retrieval_failures.py"
SPEC = importlib.util.spec_from_file_location(
    "visualize_worldmem_retrieval_failures", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def quality_row(frame, slot, psnr, ssim, lpips):
    return {
        "selected_memory_frame": frame,
        "context_slot": slot,
        "source_is_initial_context": False,
        "selected_overlap": 0.9,
        "decoded_memory_psnr": psnr,
        "decoded_memory_ssim": ssim,
        "decoded_memory_lpips": lpips,
    }


class RetrievalFailureVisualizationTest(unittest.TestCase):
    def test_actual_candidates_rank_mean_exposure_not_displayed_item(self):
        reference = {
            (0, 1050): [
                quality_row(700, 0, 4.0, 0.1, 0.95),
                quality_row(701, 1, 10.0, 0.3, 0.65),
            ]
        }
        policy = {
            (0, 1050): [
                quality_row(800, 0, 18.0, 0.6, 0.25),
                quality_row(801, 1, 20.0, 0.7, 0.15),
            ]
        }
        rows = MODULE.build_actual_candidates(
            reference,
            policy,
            context_frames=600,
            fps=10,
            late_start_sec=45,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["mean_psnr_gain"], 12.0)
        self.assertAlmostEqual(row["mean_lpips_gain"], 0.6)
        self.assertEqual(row["unbounded_selected_frame"], 700)
        self.assertEqual(row["bounded_selected_frame"], 800)

    def test_generated_rows_excludes_clean_context(self):
        rows = [
            dict(quality_row(100, 0, 30.0, 0.9, 0.1), source_is_initial_context=True),
            quality_row(600, 1, 15.0, 0.5, 0.5),
        ]
        selected = MODULE.generated_rows(rows, context_frames=600)
        self.assertEqual([row["selected_memory_frame"] for row in selected], [600])

    def test_select_diverse_limits_each_trajectory(self):
        rows = [
            {"batch_idx": 0, "target_frame": 1050, "score": 5.0},
            {"batch_idx": 0, "target_frame": 1150, "score": 4.0},
            {"batch_idx": 1, "target_frame": 1060, "score": 3.0},
        ]
        selected = MODULE.select_diverse(
            rows,
            max_examples=2,
            per_batch=1,
            min_target_gap=20,
            rank_key=lambda row: -row["score"],
        )
        self.assertEqual([row["batch_idx"] for row in selected], [0, 1])

    def test_cpu_metrics_are_perfect_for_identical_frames(self):
        if importlib.util.find_spec("cv2") is None:
            self.skipTest("opencv-python is not installed in the local sandbox")
        frame = np.full((24, 24, 3), 127, dtype=np.uint8)
        self.assertEqual(MODULE.frame_psnr(frame, frame.copy()), 100.0)
        self.assertAlmostEqual(MODULE.frame_ssim(frame, frame.copy()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
