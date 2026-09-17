import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "utils"
    / "visualize_worldmem_keepsake_snowballing.py"
)
SPEC = importlib.util.spec_from_file_location("visualize_worldmem_keepsake_snowballing", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class KeepsakeSnowballingFigureTest(unittest.TestCase):
    def test_four_columns_share_one_identity(self):
        gt = np.full((32, 48, 3), 128, dtype=np.uint8)
        images = {
            "gt": gt,
            "unbounded": np.full_like(gt, 96),
            "fifo": np.full_like(gt, 112),
            "keepsake": np.full_like(gt, 124),
        }
        case = {"batch_idx": "3", "target_frame": "1050"}
        with mock.patch.object(MODULE, "frame_ssim", return_value=0.9):
            tiles = MODULE.render_case_tiles(case, images, context_frames=600, fps=10)
        self.assertEqual(len(tiles), 4)
        self.assertTrue(all(tile.size == (MODULE.TILE_WIDTH, MODULE.TILE_HEIGHT) for tile in tiles))

    def test_case_table_requires_frame_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.csv"
            path.write_text("batch_idx,value\n0,1\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "target_frame"):
                MODULE.load_cases(path, 5)


if __name__ == "__main__":
    unittest.main()
