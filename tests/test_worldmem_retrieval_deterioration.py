import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = load_module(
    "worldmem_deterioration_exporter",
    ROOT / "utils/export_worldmem_retrieval_deterioration.py",
)
plotter = load_module(
    "worldmem_deterioration_plotter",
    ROOT / "utils/plot_worldmem_retrieval_deterioration.py",
)


def fixture():
    rows = []
    for trajectory in range(3):
        for section in range(8):
            rows.append(
                {
                    "run_name": "unbounded",
                    "row": trajectory,
                    "scene": f"scene{trajectory}",
                    "dataset_start_frame": 100,
                    "duration_sec": 60,
                    "generation_seed": 101 + trajectory,
                    "section_idx": section,
                    "target_frame": 600 + section,
                    "rollout_frame": section,
                    "candidate_count_mismatch": 0,
                    "selected_view_mismatch": 0.3 - section * 0.01,
                    "selected_memory_corruption": 0.1 + section * 0.02 + trajectory * 0.01,
                    "selected_effective_mismatch": 0.5 + section * 0.03,
                    "full_oracle_best_k_effective_mismatch": 0.3 + section * 0.01,
                    "full_oracle_effective_mismatch": 0.2,
                }
            )
    return rows


class ExporterTests(unittest.TestCase):
    def test_set_metrics_and_cardinality_matched_oracle(self):
        gt = np.eye(4, dtype=np.float32)
        generated = gt.copy()
        generated[1] = np.asarray([0.0, 0.8, 0.6, 0.0], dtype=np.float32)
        values = exporter.query_measurements(
            generated, gt, selected=[0, 1], target=2, candidates=[0, 1, 2, 3]
        )
        self.assertLessEqual(
            values["full_oracle_effective_mismatch"],
            values["full_oracle_best_k_effective_mismatch"],
        )
        self.assertLessEqual(
            values["full_oracle_best_k_effective_mismatch"],
            values["selected_effective_mismatch"],
        )
        self.assertEqual(len(values["full_oracle_best_k_frames"]), 2)

    def test_trace_validation_requires_exact_pre_read_archive(self):
        metadata = {
            0: {
                "context_frames": 2,
                "n_frames": 4,
                "memory_condition_length": 2,
                "memory_reference_source": "predicted",
            }
        }
        groups = {}
        for target in (2, 3):
            groups[(0, target)] = [
                {
                    "context_slot": slot,
                    "selected_memory_frame": slot,
                    "candidate_count": target,
                }
                for slot in range(2)
            ]
        targets, length = exporter.validate_batch_trace(groups, metadata, 0, 2, 4)
        self.assertEqual((targets, length), ([2, 3], 2))
        groups[(0, 3)][0]["candidate_count"] = 2
        with self.assertRaisesRegex(ValueError, "candidate count mismatch"):
            exporter.validate_batch_trace(groups, metadata, 0, 2, 4)


class PlotterTests(unittest.TestCase):
    def test_trajectory_weighted_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.csv"
            plotter.write_csv(path, fixture())
            identities, sections, values, targets, count = plotter.load_sections(
                path, "unbounded", 60, 3
            )
            self.assertEqual((len(identities), len(sections), count), (3, 8, 24))
            result = plotter.summarize(values, targets, 10, 4, 1000, 0)
            np.testing.assert_allclose(
                result["delta_mean"], [-0.06, 0.12, 0.18, 0.06, 0.0], atol=1e-12
            )

    def test_rejects_bad_oracle_duplicate_and_incomplete_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.csv"
            invalid = [fixture() + [fixture()[0]], fixture()[:-1]]
            bad = fixture()
            bad[0]["full_oracle_best_k_effective_mismatch"] = 0.9
            invalid.append(bad)
            for rows in invalid:
                plotter.write_csv(path, rows)
                with self.assertRaises(ValueError):
                    plotter.load_sections(path, "unbounded", 60, 3)

    def test_cli_exports_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "queries.csv"
            output = root / "figure"
            plotter.write_csv(source, fixture())
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "utils/plot_worldmem_retrieval_deterioration.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--run",
                    "unbounded",
                    "--expected-videos",
                    "3",
                    "--bootstrap",
                    "1000",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertIn("3 trajectories; 24 queries; 24 sections", completed.stdout)
            for name in (
                "retrieval_deterioration.png",
                "retrieval_deterioration.pdf",
                "changes.csv",
                "trajectory_changes.csv",
                "curves.csv",
                "caption.txt",
                "provenance.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            with (output / "changes.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 5)
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertEqual(provenance["early_sections"], [0, 1])


if __name__ == "__main__":
    unittest.main()
