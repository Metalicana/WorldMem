import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "utils"
    / "summarize_worldmem_native_eval.py"
)
SPEC = importlib.util.spec_from_file_location("native_eval_summary", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NativeEvalSummaryTests(unittest.TestCase):
    def test_read_batch_metrics_uses_latest_record_per_batch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "trace.jsonl"
            records = [
                {"event": "batch_metrics", "global_batch_idx": 0, "psnr": 1},
                {"event": "other", "global_batch_idx": 0},
                {"event": "batch_metrics", "global_batch_idx": 1, "psnr": 2},
                {"event": "batch_metrics", "global_batch_idx": 0, "psnr": 3},
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            result = MODULE.read_batch_metrics(path, limit=2)
            self.assertEqual([record["psnr"] for record in result], [3, 2])

    def test_read_batch_metrics_rejects_incomplete_cohort(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "trace.jsonl"
            path.write_text(
                json.dumps(
                    {"event": "batch_metrics", "global_batch_idx": 0, "psnr": 1}
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, r"batches: \[1\]"):
                MODULE.read_batch_metrics(path, limit=2)

    def test_read_rfid(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "fid_results.txt"
            path.write_text("FID Score: 15.1300\n", encoding="utf-8")
            self.assertEqual(MODULE.read_rfid(path), 15.13)


if __name__ == "__main__":
    unittest.main()
