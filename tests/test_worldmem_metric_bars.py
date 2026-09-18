import unittest

from utils.plot_worldmem_complete_metric_sweep import load_rows
from utils.plot_worldmem_metric_bars import ORDER, ROOT, VBENCH, fixed_rows, name


class MetricBarTests(unittest.TestCase):
    def test_fixed_budget_includes_unbounded_and_all_methods(self):
        rows = load_rows(ROOT / "assets/results/worldmem_budget_sweep_60s_n15.csv")
        lookup = {(row["policy"], row["budget"]): row for row in rows}
        group = fixed_rows(lookup)
        self.assertEqual(tuple(row["policy"] for row in group), ORDER)
        self.assertIsNone(group[0]["budget"])
        self.assertTrue(all(row["budget"] == 32 for row in group[1:]))
        self.assertEqual(name(group[-1]["policy"]), "KEEPSAKE")

    def test_vbench_dimensions_are_distinct(self):
        self.assertEqual(len({key for key, _ in VBENCH}), 6)


if __name__ == "__main__":
    unittest.main()
