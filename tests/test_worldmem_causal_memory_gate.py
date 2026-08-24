import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "algorithms"
    / "worldmem"
    / "causal_memory_gate.py"
)
SPEC = importlib.util.spec_from_file_location("worldmem_causal_memory_gate", MODULE_PATH)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def line_c2ws(positions):
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(positions), axis=0)
    poses[:, 0, 3] = np.asarray(positions, dtype=np.float64)
    return poses


class ContextWeightTest(unittest.TestCase):
    def test_positive_overlaps_are_normalized(self):
        weights = GATE.normalized_context_weights([0.2, 0.3, None])
        np.testing.assert_allclose(weights, [0.4, 0.6, 0.0])

    def test_nonpositive_overlaps_fall_back_to_uniform(self):
        weights = GATE.normalized_context_weights([None, 0.0, -1.0])
        np.testing.assert_allclose(weights, [1.0 / 3.0] * 3)


class CausalConsistencyScoreTest(unittest.TestCase):
    def test_multi_parent_score_matches_weighted_residual(self):
        calibration = {
            "version": 1,
            "feature_backend": "latent",
            "threshold": -0.15,
            "pose_scales": {"translation": 1.0, "rotation_rad": 1.0},
            "expectation": {"edges": [1.5], "means": [0.8, 0.5]},
            "deployment_approved": True,
        }
        result = GATE.causal_consistency_score(
            target_feature=np.asarray([1.0, 0.0]),
            context_features=[
                np.asarray([1.0, 0.0]),
                np.asarray([0.0, 1.0]),
            ],
            c2ws=line_c2ws([0.0, 1.0, 2.0]),
            target_frame=2,
            context_frames=[1, 0],
            overlaps=[0.75, 0.25],
            calibration=calibration,
        )
        # Parent 1: 1.0 - 0.8. Parent 0: 0.0 - 0.5.
        expected = 0.75 * 0.2 + 0.25 * -0.5
        self.assertAlmostEqual(result["score"], expected)
        self.assertTrue(result["admitted"])
        self.assertEqual(result["weight_source"], "positive_overlap")


class CalibrationValidationTest(unittest.TestCase):
    def test_unapproved_artifact_is_rejected_by_default(self):
        calibration = {
            "version": 1,
            "feature_backend": "latent",
            "threshold": 0.0,
            "pose_scales": {"translation": 1.0, "rotation_rad": 1.0},
            "expectation": {"edges": [], "means": [0.5]},
            "deployment_approved": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(calibration), encoding="utf-8")
            with self.assertRaises(ValueError):
                GATE.load_calibration(path)
            loaded = GATE.load_calibration(path, require_approved=False)
        self.assertEqual(loaded["threshold"], 0.0)


if __name__ == "__main__":
    unittest.main()
