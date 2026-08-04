import unittest
import importlib.util
from pathlib import Path

import numpy as np


POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "algorithms"
    / "worldmem"
    / "memory_policies.py"
)
SPEC = importlib.util.spec_from_file_location("worldmem_memory_policies", POLICY_PATH)
POLICIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICIES)
FrameMemoryBuffer = POLICIES.FrameMemoryBuffer
compute_rarity_irreplaceability_scores = (
    POLICIES.compute_rarity_irreplaceability_scores
)


class FrameMemoryBufferTest(unittest.TestCase):
    def test_random_cap_is_bounded_pinned_and_deterministic(self):
        left = FrameMemoryBuffer(
            policy="random_cap",
            budget=5,
            pinned_frames={0},
            random_seed=17,
        )
        right = FrameMemoryBuffer(
            policy="random_cap",
            budget=5,
            pinned_frames={0},
            random_seed=17,
        )

        left.update(range(20))
        right.update(range(20))

        self.assertEqual(left.candidates(), right.candidates())
        self.assertEqual(len(left), 5)
        self.assertIn(0, left.candidates())

    def test_random_cap_changes_with_seed(self):
        retained = []
        for seed in (3, 4):
            buffer = FrameMemoryBuffer(
                policy="random_cap",
                budget=6,
                pinned_frames={0},
                random_seed=seed,
            )
            buffer.update(range(30))
            retained.append(buffer.candidates())
        self.assertNotEqual(retained[0], retained[1])


class RarityIrreplaceabilityTest(unittest.TestCase):
    def test_legacy_latent_call_matches_explicit_cosine_backend(self):
        features = {
            0: np.asarray([1.0, 0.0]),
            1: np.asarray([0.9, 0.1]),
            2: np.asarray([0.0, 1.0]),
        }
        legacy = compute_rarity_irreplaceability_scores(
            [0, 1, 2],
            latent_features=features,
        )
        explicit = compute_rarity_irreplaceability_scores(
            [0, 1, 2],
            rarity_features=features,
            irreplaceability_features=features,
            irreplaceability_metric="cosine",
        )
        self.assertEqual(legacy, explicit)

    def test_dino_rgb_backend_uses_rgb_mean_absolute_distance(self):
        dino = {
            0: np.asarray([1.0, 0.0]),
            1: np.asarray([0.95, 0.05]),
            2: np.asarray([0.0, 1.0]),
        }
        rgb = {
            0: np.zeros(12),
            1: np.full(12, 0.1),
            2: np.ones(12),
        }
        scores, details = compute_rarity_irreplaceability_scores(
            [0, 1, 2],
            rarity_features=dino,
            irreplaceability_features=rgb,
            irreplaceability_metric="mean_abs",
            pinned_frames={0},
            return_details=True,
        )

        self.assertTrue(np.isinf(scores[0]))
        self.assertAlmostEqual(details[0]["irreplaceability"], 0.1)
        self.assertAlmostEqual(details[1]["irreplaceability"], 0.1)
        self.assertAlmostEqual(details[2]["irreplaceability"], 0.9)
        self.assertEqual(details[2]["irreplaceability_metric"], "mean_abs")


if __name__ == "__main__":
    unittest.main()
