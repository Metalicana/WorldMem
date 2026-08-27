import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "utils"
    / "stage_worldmem_vbench_input.py"
)
SPEC = importlib.util.spec_from_file_location("stage_worldmem_vbench_input", MODULE_PATH)
STAGING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGING)


def make_video(path):
    path.write_bytes(b"0" * 5000)


class VBenchStagingTest(unittest.TestCase):
    def test_stages_exact_batch_prefix_in_batch_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            stage = root / "stage"
            source.mkdir()
            for batch_id in (2, 0, 1, 3):
                make_video(source / f"video_batch{batch_id:05d}_0_rank0.mp4")

            selected = STAGING.stage_inputs(source, stage, limit=3)

            self.assertEqual([row["batch_id"] for row in selected], [0, 1, 2])
            staged = sorted(stage.glob("video_batch*.mp4"))
            self.assertEqual(len(staged), 3)
            self.assertTrue(all(path.is_symlink() for path in staged))

    def test_refuses_missing_required_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for batch_id in (0, 2):
                make_video(source / f"video_batch{batch_id:05d}_0_rank0.mp4")

            with self.assertRaisesRegex(RuntimeError, r"\[1\]"):
                STAGING.stage_inputs(source, root / "stage", limit=3)

    def test_refuses_duplicate_prediction_for_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            make_video(source / "video_batch00000_0_rank0.mp4")
            make_video(source / "video_batch00000_0_rank0_step9.mp4")
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                STAGING.discover_batch_videos(source)

    def test_force_reset_removes_long_split_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            stage = root / "stage"
            split = stage / "split_clip"
            source.mkdir()
            split.mkdir(parents=True)
            make_video(source / "video_batch00000_0_rank0.mp4")
            (split / "stale.mp4").write_bytes(b"stale")

            STAGING.stage_inputs(source, stage, limit=1, reset_derived=True)

            self.assertFalse(split.exists())


if __name__ == "__main__":
    unittest.main()
