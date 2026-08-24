import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "utils"
    / "calibrate_worldmem_causal_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "calibrate_worldmem_causal_gate", MODULE_PATH
)
CALIBRATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATE)


class CalibrationWorkflowTest(unittest.TestCase):
    def test_shadow_trace_produces_approved_disjoint_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "shadow.jsonl"
            output = root / "calibration"
            with trace.open("w", encoding="utf-8") as handle:
                for trajectory in range(6):
                    for step in range(20):
                        quality = step / 19.0
                        generated_similarity = 0.2 + 0.7 * quality
                        row = {
                            "event": "causal_gate_observation",
                            "causal_gate_mode": "shadow",
                            "global_batch_idx": trajectory,
                            "target_frame": 600 + step,
                            "target_generated_to_gt_similarity": quality,
                            "target_generated_to_gt_mse": 1.0 - quality,
                            "causal_gate_parents": [
                                {
                                    "frame": 500 + step,
                                    "weight": 0.6,
                                    "generated_similarity": generated_similarity,
                                    "gt_similarity": 0.8,
                                    "translation": 0.5 + 0.1 * (step % 4),
                                    "rotation_rad": 0.1,
                                },
                                {
                                    "frame": 400 + step,
                                    "weight": 0.4,
                                    "generated_similarity": generated_similarity,
                                    "gt_similarity": 0.8,
                                    "translation": 1.0 + 0.1 * (step % 4),
                                    "rotation_rad": 0.2,
                                },
                            ],
                        }
                        handle.write(json.dumps(row) + "\n")

            argv = [
                "calibrate_worldmem_causal_gate.py",
                "--trace_paths",
                str(trace),
                "--output_dir",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                CALIBRATE.main()

            artifact = json.loads(
                (output / "calibration.json").read_text(encoding="utf-8")
            )
            self.assertTrue(artifact["deployment_approved"])
            self.assertTrue(
                set(artifact["train_trajectory_ids"]).isdisjoint(
                    artifact["test_trajectory_ids"]
                )
            )
            self.assertGreaterEqual(
                artifact["heldout_metrics"]["quality_auc"], 0.99
            )


if __name__ == "__main__":
    unittest.main()
