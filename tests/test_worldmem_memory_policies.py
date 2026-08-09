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
compute_marginal_coverage_eviction_scores = (
    POLICIES.compute_marginal_coverage_eviction_scores
)


def make_line_c2ws(positions):
    c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], len(positions), axis=0)
    c2ws[:, 0, 3] = np.array(positions, dtype=np.float64)
    return c2ws


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


class MarginalCoverageEvictionTest(unittest.TestCase):
    def test_policy_requires_budget(self):
        with self.assertRaises(ValueError):
            FrameMemoryBuffer(policy="mce")

    def test_duplicate_view_counterexample(self):
        # frames 0, 1: near-identical room-A views (close pose, near-identical
        # DINO features). frame 2: a distinct room-B view. Budget 2 should
        # keep one room-A view plus the distinct one, not both room-A views.
        c2ws = make_line_c2ws([0.0, 0.1, 20.0])
        dino = {
            0: np.array([1.0, 0.0]),
            1: np.array([0.98, 0.02]),
            2: np.array([0.0, 1.0]),
        }

        scores, details = compute_marginal_coverage_eviction_scores(
            memory_frame_indices=[0, 1, 2],
            c2ws=c2ws,
            budget=2,
            latent_features=dino,
            return_details=True,
        )

        selected = {frame_idx for frame_idx, row in details.items() if row["mce_selected"]}
        self.assertEqual(selected, {0, 2})
        self.assertLess(scores[1], scores[0])
        self.assertLess(scores[1], scores[2])

        buffer = FrameMemoryBuffer(policy="mce", budget=2)
        evicted = buffer.update([0, 1, 2], eviction_scores=scores)
        self.assertEqual(buffer.candidates(), [0, 2])
        self.assertEqual(evicted, [1])

    def test_kernel_is_additive_not_multiplicative(self):
        # Identical pose, orthogonal DINO features. Under the additive kernel
        # alpha * K_geo + (1 - alpha) * K_vis, K_geo = 1 keeps K(q, m) well
        # above zero even though K_vis = 0 -- a product kernel would collapse
        # it to ~0 instead.
        c2ws = make_line_c2ws([0.0, 0.0])
        dino = {
            0: np.array([1.0, 0.0]),
            1: np.array([0.0, 1.0]),
        }

        _, details = compute_marginal_coverage_eviction_scores(
            memory_frame_indices=[0, 1],
            c2ws=c2ws,
            budget=2,
            latent_features=dino,
            alpha=0.65,
            return_details=True,
        )
        # K(0, 1) = 0.65 * K_geo(1) + 0.35 * K_vis(0.5) = 0.825.
        self.assertGreater(details[0]["mce_coverage_value"], 0.7)

    def test_forced_frames_are_never_evicted(self):
        c2ws = make_line_c2ws([0.0, 0.05, 20.0])
        dino = {
            0: np.array([1.0, 0.0]),
            1: np.array([1.0, 0.0]),
            2: np.array([0.0, 1.0]),
        }

        _, details = compute_marginal_coverage_eviction_scores(
            memory_frame_indices=[0, 1, 2],
            c2ws=c2ws,
            budget=1,
            pinned_frames={1},
            latent_features=dino,
            return_details=True,
        )
        self.assertTrue(details[1]["mce_selected"])
        self.assertTrue(details[1]["mce_forced_keep"])
        self.assertEqual(details[1]["score"], float("inf"))

    def test_large_initial_context_trimmed_to_small_budget_stays_finite(self):
        # Regression test for a real production crash: WorldMem's initial
        # context can hand MCE ~600 candidates to trim down to a budget as
        # small as 16 -- hundreds of reverse-deletion steps, not the "dozens"
        # the direct-space running-product assumed safe. A long enough run of
        # divisions by kernel values near the 1e-6 clip floor compounded the
        # running product past float64's max and tripped the finite-marginals
        # assertion. Autocorrelated positions/features (a slow-moving camera,
        # a random walk in feature space) reproduce the redundant-content
        # shape that triggered it -- independent random features did not.
        n = 600
        rng = np.random.default_rng(0)
        c2ws = np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0)
        c2ws[:, :3, 3] = rng.normal(scale=0.05, size=(n, 3)).cumsum(axis=0)
        feature_walk = rng.normal(scale=0.05, size=(n, 32)).cumsum(axis=0)
        dino = {i: feature_walk[i] for i in range(n)}

        for budget in (16, 32, 64, 128):
            scores, details = compute_marginal_coverage_eviction_scores(
                memory_frame_indices=list(range(n)),
                c2ws=c2ws,
                budget=budget,
                pinned_frames={0},
                latent_features=dino,
                alpha=0.65,
                return_details=True,
            )
            selected = [frame_idx for frame_idx, row in details.items() if row["mce_selected"]]
            self.assertEqual(len(selected), budget)
            for score in scores.values():
                self.assertTrue(np.isfinite(score) or score == float("inf"))

    def test_alpha_zero_drops_geometry_entirely(self):
        # Far-apart poses but identical content: with alpha=0 the geometry
        # cue must not matter at all.
        c2ws = make_line_c2ws([0.0, 1000.0])
        dino = {
            0: np.array([1.0, 0.0]),
            1: np.array([1.0, 0.0]),
        }

        _, details = compute_marginal_coverage_eviction_scores(
            memory_frame_indices=[0, 1],
            c2ws=c2ws,
            budget=2,
            latent_features=dino,
            alpha=0.0,
            return_details=True,
        )
        self.assertAlmostEqual(details[0]["mce_coverage_value"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
