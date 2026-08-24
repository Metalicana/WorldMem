import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
MODULE_PATH = UTILS_DIR / "validate_worldmem_coverage_hysteresis.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_worldmem_coverage_hysteresis", MODULE_PATH
)
VALIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE)


def line_c2ws(positions):
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(positions), axis=0)
    poses[:, 0, 3] = np.asarray(positions, dtype=np.float64)
    return poses


class ChunkTraceTest(unittest.TestCase):
    def test_uses_actual_horizon_and_deduplicates_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            rows = [
                {
                    "event": "memory_run_start",
                    "global_batch_idx": 3,
                    "context_frames": 600,
                }
            ]
            for slot in range(8):
                rows.append(
                    {
                        "event": "memory_retrieval",
                        "global_batch_idx": 3,
                        "context_slot": slot,
                        "target_frame": 600,
                        "target_horizon": 2,
                    }
                )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            chunks = VALIDATE.load_chunk_samples([path])
        self.assertEqual(len(chunks[3]["samples"]), 1)
        self.assertEqual(chunks[3]["samples"][0]["generated_frame"], 1)
        self.assertEqual(chunks[3]["samples"][0]["target_horizon"], 2)

    def test_legacy_trace_uses_run_start_batch_idx(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.jsonl"
            rows = []
            for batch_idx in (0, 1):
                rows.append(
                    {
                        "event": "memory_run_start",
                        "batch_idx": batch_idx,
                        "context_frames": 600,
                    }
                )
                for target in (600, 601):
                    rows.append(
                        {
                            "event": "memory_retrieval",
                            "batch_index": 0,
                            "context_slot": 0,
                            "target_frame": target,
                            "target_horizon": 1,
                        }
                    )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            chunks = VALIDATE.load_chunk_samples([path])
        self.assertEqual(sorted(chunks), [0, 1])
        self.assertEqual(len(chunks[0]["samples"]), 2)
        self.assertEqual(len(chunks[1]["samples"]), 2)

    def test_oldest_trace_detects_target_resets_without_run_start(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oldest.jsonl"
            rows = [
                {
                    "event": "memory_retrieval",
                    "context_slot": 0,
                    "target_frame": target,
                    "target_horizon": 1,
                }
                for target in (600, 601, 600, 601)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            chunks = VALIDATE.load_chunk_samples([path])
        self.assertEqual(sorted(chunks), [0, 1])


class GeometryMatchingTest(unittest.TestCase):
    def test_identity_pose_has_unit_similarity(self):
        poses = line_c2ws([0.0])
        value = VALIDATE.worldmem_fov_similarity(poses, [0], [0])[0, 0]
        self.assertAlmostEqual(value, 1.0)

    def test_threshold_sweep_is_nested_and_preserves_oldest_match(self):
        poses = line_c2ws([0.5, 0.0, 0.0, 0.0])
        samples = [
            {
                "chunk_index": index,
                "generated_frame": index,
                "target_frame": 600 + index,
                "target_horizon": 1,
            }
            for index in range(4)
        ]
        pairs = VALIDATE.find_oldest_covered_pairs(
            samples,
            poses,
            thresholds=[0.95, 0.995],
            min_chunk_separation=2,
        )
        loose = [row for row in pairs if row["coverage_threshold"] == 0.95]
        strict = [row for row in pairs if row["coverage_threshold"] == 0.995]
        self.assertGreaterEqual(len(loose), len(strict))
        later_two = next(row for row in loose if row["later_chunk_index"] == 2)
        # Frame 1 is geometrically closer, but it is not old enough at a
        # two-chunk gap. The oldest eligible covered incumbent is frame 0.
        self.assertEqual(later_two["earlier_chunk_index"], 0)


class TrajectoryBootstrapTest(unittest.TestCase):
    def test_summary_weights_trajectories_not_pairs(self):
        rows = []
        for _ in range(100):
            rows.append(
                {
                    "coverage_threshold": 0.9,
                    "batch_idx": 0,
                    "is_final_quarter": False,
                    "psnr_delta": 10.0,
                    "ssim_delta": 1.0,
                    "temporal_gap_chunks": 2,
                }
            )
        rows.append(
            {
                "coverage_threshold": 0.9,
                "batch_idx": 1,
                "is_final_quarter": False,
                "psnr_delta": 0.0,
                "ssim_delta": 0.0,
                "temporal_gap_chunks": 2,
            }
        )
        _, summary = VALIDATE.summarize_pairs(rows, [0.9], 200, 7)
        overall = next(row for row in summary if row["subset"] == "all")
        self.assertAlmostEqual(overall["psnr_delta_trajectory_mean"], 5.0)
        self.assertEqual(overall["trajectories"], 2)


class FrameQualityTest(unittest.TestCase):
    def test_identical_frames_have_perfect_quality(self):
        frame = np.full((16, 16, 3), 127, dtype=np.uint8)
        runner = VALIDATE.FrameQualityRunner(device="cpu", batch_size=1)
        result = runner.compute({0: frame}, {0: frame.copy()}, [0])[0]
        self.assertAlmostEqual(result["psnr"], 100.0, places=4)
        self.assertAlmostEqual(result["ssim"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
