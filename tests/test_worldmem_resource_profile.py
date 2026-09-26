import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]

PLOT_SPEC = importlib.util.spec_from_file_location(
    "resource_plot", ROOT / "utils/plot_worldmem_resource_profile.py"
)
RESOURCE_PLOT = importlib.util.module_from_spec(PLOT_SPEC)
PLOT_SPEC.loader.exec_module(RESOURCE_PLOT)

SAMPLER_SPEC = importlib.util.spec_from_file_location(
    "memory_sampler", ROOT / "utils/sample_process_tree_memory.py"
)
MEMORY_SAMPLER = importlib.util.module_from_spec(SAMPLER_SPEC)
SAMPLER_SPEC.loader.exec_module(MEMORY_SAMPLER)


class WorldMemResourceProfileTests(unittest.TestCase):
    def write_trace(self, path, sizes, budgeted):
        with path.open("w", encoding="utf-8") as handle:
            for generated_frame, archive_frames in enumerate(sizes):
                archive_mib = archive_frames * 0.025
                handle.write(
                    json.dumps(
                        {
                            "event": "memory_archive_state",
                            "generated_frames": generated_frame,
                            "archive_frames": archive_frames,
                            "archive_latent_payload_mib": archive_mib,
                            "gpu_bank_payload_mib": archive_mib,
                            "history_latent_payload_mib": (2 + generated_frame)
                            * 0.025,
                            "bounded": budgeted,
                        }
                    )
                    + "\n"
                )

    def summary_row(self, policy, budget, device, trace_path):
        return {
            "run_name": f"{device}_{policy}",
            "policy": policy,
            "budget": "" if budget is None else budget,
            "memory_bank_device": device,
            "status": 0,
            "trace_path": str(trace_path),
            "final_archive_frames": 4 if budget is None else budget,
            "final_archive_mib": 0.1 if budget is None else 0.05,
            "retrieval_ms_mean": 12.0 if budget is None else 2.0,
            "retrieval_ms_median": 11.0 if budget is None else 1.8,
            "retrieval_ms_p95": 20.0 if budget is None else 3.0,
            "memory_update_count": 3,
            "memory_update_ms_mean": 0.2 if budget is None else 0.8,
            "descriptor_extraction_seconds": 0.0 if budget is None else 0.001,
            "peak_process_tree_rss_mib": 2048 if budget is None else 1900,
            "peak_torch_allocated_mib": 9600 if budget is None else 9500,
            "peak_nvidia_smi_used_mib": 11000 if budget is None else 10900,
            "wall_seconds": 100 if budget is None else 90,
        }

    def test_builds_growth_and_resource_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traces = root / "access_traces"
            traces.mkdir()
            unbounded_trace = traces / "gpu_unbounded.jsonl"
            bounded_trace = traces / "gpu_keepsake.jsonl"
            self.write_trace(unbounded_trace, [2, 3, 4], budgeted=False)
            self.write_trace(bounded_trace, [2, 2, 2], budgeted=True)

            rows = []
            for device in ("cpu", "gpu"):
                rows.append(
                    self.summary_row(
                        "unbounded",
                        None,
                        device,
                        unbounded_trace,
                    )
                )
                rows.append(
                    self.summary_row(
                        "slam_covisibility",
                        2,
                        device,
                        bounded_trace,
                    )
                )
            with (root / "summary.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            output = root / "report"
            argv = [
                "plot_worldmem_resource_profile.py",
                "--profile-root",
                str(root),
                "--output-dir",
                str(output),
                "--budget",
                "2",
                "--fps",
                "1",
                "--expected-duration-sec",
                "2",
            ]
            with mock.patch("sys.argv", argv):
                RESOURCE_PLOT.main()

            self.assertTrue(
                (output / "figures/worldmem_archive_growth.png").is_file()
            )
            self.assertTrue(
                (output / "figures/worldmem_resource_summary.pdf").is_file()
            )
            self.assertTrue(
                (output / "tables/worldmem_resource_summary.csv").is_file()
            )
            self.assertTrue((output / "provenance.json").is_file())

    def test_legacy_gpu_trace_uses_event_order_for_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for frames in (600, 601, 602):
                    handle.write(
                        json.dumps(
                            {
                                "event": "gpu_memory_bank_sync",
                                "stored_memory_size": frames,
                                "estimated_bank_mib": frames / 100,
                            }
                        )
                        + "\n"
                    )
            states = RESOURCE_PLOT.load_archive_states(path, fps=10)
            self.assertEqual(states[-1]["generated_time_sec"], 0.2)

    def test_parse_proc_stat_handles_spaces_in_process_name(self):
        fields = [
            "S",
            "42",
            "1",
            "1",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "1",
            "0",
            "98765",
        ]
        ppid, start_ticks = MEMORY_SAMPLER.parse_proc_stat(
            "123 (python worker) " + " ".join(fields)
        )
        self.assertEqual(ppid, 42)
        self.assertEqual(start_ticks, 98765)


if __name__ == "__main__":
    unittest.main()
