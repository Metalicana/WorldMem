import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "utils"
    / "plot_worldmem_complete_metric_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("plot_worldmem_complete_metric_sweep", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompleteMetricSweepPlotTest(unittest.TestCase):
    def make_rows(self):
        rows = []
        keys = [("Unbounded", None)] + [
            (policy, budget)
            for policy in MODULE.POLICIES
            for budget in MODULE.BUDGETS
        ]
        for index, (policy, budget) in enumerate(keys):
            row = {
                "run_name": f"run_{index}",
                "policy": policy,
                "budget": "" if budget is None else budget,
                "videos_matched": 15,
            }
            for metric, _, _, _ in MODULE.METRICS:
                row[metric] = 0.5 + index / 100
            rows.append(row)
        return rows

    def write_csv(self, path, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_loads_complete_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            self.write_csv(path, self.make_rows())
            rows = MODULE.load_rows(path)
            self.assertEqual(len(rows), 21)
            self.assertEqual(rows[0]["budget"], None)

    def test_rejects_missing_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            self.write_csv(path, self.make_rows()[:-1])
            with self.assertRaisesRegex(ValueError, "Incomplete metric grid"):
                MODULE.load_rows(path)

    def test_renders_all_figure_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "metrics.csv"
            output = root / "plots"
            self.write_csv(source, self.make_rows())
            argv = [
                "plot_worldmem_complete_metric_sweep.py",
                "--input-csv",
                str(source),
                "--output-dir",
                str(output),
            ]
            with mock.patch("sys.argv", argv):
                MODULE.main()
            self.assertTrue(
                (output / "worldmem_lpips_fvd_budget_sweep_60s_n15.png").is_file()
            )
            self.assertTrue(
                (output / "worldmem_vbench_budget_sweep_60s_n15.pdf").is_file()
            )
            self.assertTrue(
                (
                    output
                    / "worldmem_complete_metric_budget_sweep_60s_n15.png"
                ).is_file()
            )
            self.assertTrue(
                (output / "worldmem_vbench_radar_highlights_60s_n15.pdf").is_file()
            )


if __name__ == "__main__":
    unittest.main()
