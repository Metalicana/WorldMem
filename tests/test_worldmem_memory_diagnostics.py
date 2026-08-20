import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "algorithms"
    / "worldmem"
    / "memory_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location("worldmem_memory_diagnostics", MODULE_PATH)
DIAGNOSTICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTICS)


class ImageQualityTest(unittest.TestCase):
    def test_identical_images_have_perfect_psnr_and_ssim(self):
        images = torch.rand(3, 3, 24, 24)
        metrics = DIAGNOSTICS.image_quality_per_image(
            images,
            images.clone(),
            psnr_cap=100,
        )
        self.assertTrue(torch.equal(metrics["mse"], torch.zeros(3)))
        self.assertTrue(torch.equal(metrics["psnr"], torch.full((3,), 100.0)))
        self.assertTrue(torch.allclose(metrics["ssim"], torch.ones(3), atol=1e-6))
        self.assertTrue(torch.isnan(metrics["lpips"]).all())

    def test_opposite_constant_images_have_zero_psnr(self):
        prediction = torch.zeros(2, 3, 24, 24)
        target = torch.ones_like(prediction)
        metrics = DIAGNOSTICS.image_quality_per_image(prediction, target)
        self.assertTrue(torch.allclose(metrics["mse"], torch.ones(2)))
        self.assertTrue(torch.allclose(metrics["psnr"], torch.zeros(2)))
        self.assertEqual(metrics["ssim"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
