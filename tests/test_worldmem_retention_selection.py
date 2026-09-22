import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "retention_selection", ROOT / "utils/analyze_worldmem_retention_selection.py"
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class RetentionSelectionTests(unittest.TestCase):
    def test_best_eight_decomposition_telescopes(self):
        distances = np.arange(20, dtype=np.float64) / 20
        result = analysis.decompose_query(
            distances,
            np.arange(20),
            np.arange(4, 20),
            np.arange(12, 20),
            k=8,
        )
        self.assertAlmostEqual(result["full_oracle_distance"], np.mean(distances[:8]))
        self.assertAlmostEqual(result["bank_oracle_distance"], np.mean(distances[4:12]))
        self.assertAlmostEqual(result["selected_distance"], np.mean(distances[12:20]))
        self.assertAlmostEqual(
            result["retention_gap"] + result["selection_gap"], result["total_gap"]
        )

    def test_k_one_matches_memcam_definition(self):
        distances = np.asarray([0.4, 0.1, 0.3, 0.2])
        result = analysis.decompose_query(distances, [0, 1, 2, 3], [0, 2, 3], [2], k=1)
        self.assertAlmostEqual(result["retention_gap"], 0.1)
        self.assertAlmostEqual(result["selection_gap"], 0.1)
        self.assertAlmostEqual(result["total_gap"], 0.2)

    def test_zero_retention_and_zero_selection_for_oracle(self):
        distances = np.linspace(0.0, 1.0, 12)
        selected = list(range(8))
        result = analysis.decompose_query(
            distances, np.arange(12), np.arange(12), selected, k=8
        )
        self.assertAlmostEqual(result["retention_gap"], 0.0)
        self.assertAlmostEqual(result["selection_gap"], 0.0)

    def test_precomputed_full_oracle_matches_direct_result(self):
        distances = np.linspace(1.0, 0.0, 20)
        full = np.arange(20)
        bank = np.arange(2, 20)
        selected = np.arange(12, 20)
        direct = analysis.decompose_query(distances, full, bank, selected, k=8)
        oracle = analysis.best_k(full, distances, 8)
        cached = analysis.decompose_query(
            distances, full, bank, selected, k=8, full_oracle=oracle
        )
        for metric in analysis.METRICS:
            self.assertAlmostEqual(direct[metric], cached[metric])

    def test_rejects_invalid_set_relationships_and_duplicates(self):
        distances = np.linspace(0.0, 1.0, 16)
        with self.assertRaisesRegex(ValueError, "subset"):
            analysis.decompose_query(distances, np.arange(12), np.arange(4, 16), np.arange(4, 12), k=8)
        with self.assertRaisesRegex(ValueError, "distinct"):
            analysis.decompose_query(distances, np.arange(16), np.arange(16), [0, 1, 2, 3, 4, 5, 6, 6], k=8)

    def test_ties_have_deterministic_low_id_oracle(self):
        mean, ids = analysis.best_k([9, 4, 7, 1], [0.2, 0.2, 0.2, 0.2], 2)
        self.assertAlmostEqual(mean, 0.2)
        self.assertEqual(ids, [1, 4])


if __name__ == "__main__":
    unittest.main()
