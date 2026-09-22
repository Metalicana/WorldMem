import unittest

from utils.calculate_worldmem_vbench6 import SPEC, calculate_vbench6


class VBench6Tests(unittest.TestCase):
    def test_all_maxima_score_one_hundred(self):
        values = {dimension: high for dimension, (_low, high, _weight) in SPEC.items()}
        self.assertAlmostEqual(calculate_vbench6(values), 100.0)

    def test_worldmem_recorded_unbounded(self):
        values = {
            "subject_consistency": 0.7387,
            "background_consistency": 0.8753,
            "motion_smoothness": 0.9725,
            "dynamic_degree": 0.6667,
            "aesthetic_quality": 0.3837,
            "imaging_quality": 0.6201,
        }
        self.assertAlmostEqual(calculate_vbench6(values), 68.663406, places=5)

    def test_requires_exact_six_dimensions(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            calculate_vbench6({"subject_consistency": 1.0})


if __name__ == "__main__":
    unittest.main()
