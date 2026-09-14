import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "utils" / "run_worldmem_vbench_long.py"
)
SPEC = importlib.util.spec_from_file_location("run_worldmem_vbench_long", MODULE_PATH)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class VBenchLongGroupingTest(unittest.TestCase):
    def make_rows(self, stage, stem, scores):
        video = stage / f"{stem}.mp4"
        video.write_bytes(b"video")
        clips = stage / "split_clip" / stem
        clips.mkdir(parents=True)
        return [
            {
                "video_path": str(clips / f"{stem}_{index:03d}.mp4"),
                "video_results": score,
            }
            for index, score in enumerate(scores)
        ]

    def test_groups_clips_by_original_video(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            rows = self.make_rows(stage, "video_a", [0.2, 0.4])
            rows += self.make_rows(stage, "video_b", [0.6, 0.8])
            overall, details, videos = ADAPTER.reorganize_clips_results(rows)
        self.assertAlmostEqual(overall, 0.5)
        self.assertEqual(details, rows)
        self.assertEqual(len(videos), 2)

    def test_imaging_quality_normalizes_only_top_level(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            rows = self.make_rows(stage, "video", [40.0, 60.0])
            overall, _, videos = ADAPTER.reorganize_clips_results(
                rows,
                dimension="imaging_quality",
            )
        self.assertAlmostEqual(overall, 0.5)
        self.assertAlmostEqual(videos[0]["video_results"], 50.0)

    def test_rejects_duplicate_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.make_rows(Path(directory), "video", [0.5])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                ADAPTER.reorganize_clips_results(rows + rows)


if __name__ == "__main__":
    unittest.main()
