import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "plot_worldmem_budget_sweep.py"
SPEC = importlib.util.spec_from_file_location("plot_worldmem_budget_sweep", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

COMMON_PATH = Path(__file__).resolve().parents[1] / "utils" / "worldmem_eval_common.py"
COMMON_SPEC = importlib.util.spec_from_file_location("worldmem_eval_common", COMMON_PATH)
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
COMMON_SPEC.loader.exec_module(COMMON)


class WorldMemBudgetSweepPlotTest(unittest.TestCase):
    def write_summary(self, path, metric, rows):
        fields = [
            "run_name",
            "duration_sec",
            metric,
            "videos",
            "completed_videos",
            "failed_videos",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_classifies_every_budget_policy(self):
        cases = {
            "worldmem_unbounded_60s_n30": ("Unbounded", None),
            "worldmem_fifo_b16_60s_n30": ("FIFO", 16),
            "worldmem_rarity_irreplaceability_b32_60s_n30": ("Latent-RI", 32),
            "worldmem_slam_covisibility_b64_60s_n30": ("Geometric Coverage", 64),
            "worldmem_kcenter_coreset_b128_60s_n15": ("K-center", 128),
            "worldmem_mce_b32_60s_n15": ("MCE", 32),
        }
        for run_name, expected in cases.items():
            self.assertEqual(MODULE.classify_run(run_name), expected)

    def test_load_requires_exact_matched_video_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            self.write_summary(
                path,
                "lpips",
                [
                    {
                        "run_name": "worldmem_fifo_b16_60s_n30",
                        "duration_sec": 60,
                        "lpips": 0.5,
                        "videos": 15,
                        "completed_videos": 15,
                        "failed_videos": 0,
                    },
                    {
                        "run_name": "worldmem_fifo_b32_60s_n30",
                        "duration_sec": 60,
                        "lpips": 0.4,
                        "videos": 30,
                        "completed_videos": 30,
                        "failed_videos": 0,
                    },
                ],
            )
            values, sources = MODULE.load_metric_summary(path, "lpips", 60, 15)
            self.assertEqual(values, {("FIFO", 16): 0.5})
            self.assertEqual(
                sources[("FIFO", 16)], "worldmem_fifo_b16_60s_n30"
            )

    def test_missing_cells_reports_complete_expected_grid(self):
        values = {("Unbounded", None): 1.0}
        missing = MODULE.missing_cells(values)
        self.assertEqual(len(missing), len(MODULE.POLICY_ORDER) * len(MODULE.BUDGETS))
        self.assertIn("Geometric Coverage B32", missing)
        self.assertIn("MCE B128", missing)

    def test_exact_prefix_rejects_a_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "worldmem_fifo_b16_60s_n30"
            pred_dir = run_dir / "videos" / "test_vis" / "pred"
            pred_dir.mkdir(parents=True)
            for batch_idx in (0, 2):
                (pred_dir / f"video_batch{batch_idx:05d}_0_rank0.mp4").touch()
            with self.assertRaisesRegex(RuntimeError, r"batch IDs: \[1\]"):
                COMMON.list_prediction_videos(
                    run_dir,
                    limit=2,
                    require_prefix=True,
                )

    def test_complete_grid_renders_table_and_figures(self):
        run_names = ["worldmem_unbounded_60s_n30"]
        stems = {
            "FIFO": "fifo",
            "Latent-RI": "rarity_irreplaceability",
            "Geometric Coverage": "slam_covisibility",
            "K-center": "kcenter_coreset",
            "MCE": "mce",
        }
        for policy in MODULE.POLICY_ORDER:
            suffix = "n15" if policy in {"K-center", "MCE"} else "n30"
            for budget in MODULE.BUDGETS:
                run_names.append(
                    f"worldmem_{stems[policy]}_b{budget}_60s_{suffix}"
                )

        def rows(metric):
            return [
                {
                    "run_name": run_name,
                    "duration_sec": 60,
                    metric: 0.5 + index,
                    "videos": 15,
                    "completed_videos": 15,
                    "failed_videos": 0,
                }
                for index, run_name in enumerate(run_names)
            ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lpips_path = root / "lpips.csv"
            fvd_path = root / "fvd.csv"
            output_dir = root / "plots"
            self.write_summary(lpips_path, "lpips", rows("lpips"))
            self.write_summary(fvd_path, "fvd", rows("fvd"))
            argv = [
                "plot_worldmem_budget_sweep.py",
                "--lpips-summary",
                str(lpips_path),
                "--fvd-summary",
                str(fvd_path),
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch("sys.argv", argv):
                MODULE.main()
            self.assertTrue(
                (output_dir / "worldmem_budget_sweep_60s_n15.csv").is_file()
            )
            self.assertTrue(
                (output_dir / "worldmem_budget_sweep_60s_n15.png").is_file()
            )
            self.assertTrue(
                (output_dir / "worldmem_lpips_vs_budget_60s_n15.pdf").is_file()
            )


if __name__ == "__main__":
    unittest.main()
