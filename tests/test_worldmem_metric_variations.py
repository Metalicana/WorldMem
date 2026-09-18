import unittest

from utils.plot_worldmem_metric_variations import ROOT, label, load_rows, relative_improvement


class MetricVariationTests(unittest.TestCase):
    def test_directional_relative_difference(self):
        self.assertAlmostEqual(relative_improvement(75, 100, True), 25)
        self.assertAlmostEqual(relative_improvement(125, 100, True), -25)
        self.assertAlmostEqual(relative_improvement(125, 100, False), 25)
        self.assertEqual(relative_improvement(100, 100, False), 0)

    def test_saved_grid_and_label(self):
        rows = load_rows(ROOT / "assets/results/worldmem_budget_sweep_60s_n15.csv")
        self.assertEqual(len(rows), 21)
        self.assertTrue(all(row["videos_matched"] == 15 for row in rows))
        self.assertEqual(label("Geometric Coverage"), "KEEPSAKE")


if __name__ == "__main__":
    unittest.main()
