"""Small deterministic checks; no network, pretrained weights or private data."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ensegnet.data import Tiles
from ensegnet.engine import region_loss, rgb_predict
from ensegnet.metrics import aggregate, boundary_f1, confusion, overlap, tile_metrics


class MetricsTests(unittest.TestCase):
    def test_confusion_orientation_and_pooled_iou(self):
        y = np.array([[0, 0, 1], [1, 2, 2]])
        p = np.array([[0, 1, 1], [2, 2, 2]])
        cm = confusion(y, p)
        np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 1, 1], [0, 0, 2]])
        np.testing.assert_allclose(overlap(cm)["iou"], [.5, 1 / 3, 2 / 3])
        combined = aggregate([tile_metrics("one", y, p), tile_metrics("two", y, y)])
        np.testing.assert_allclose(combined["iou"], [.75, 3 / 5, .8])

    def test_coverage_sign_and_percentage_points(self):
        y = np.array([[0, 1], [2, 2]])
        p = np.array([[0, 1], [1, 2]])
        row = tile_metrics("example", y, p)
        self.assertEqual(row["coverage_error_pp"], [0., 25., -25.])
        self.assertEqual(aggregate([row])["coverage_mae_pp"], [0., 25., 25.])

    def test_absent_class_and_absent_boundary(self):
        y = np.zeros((8, 8), np.uint8)
        self.assertEqual(overlap(confusion(y, y))["miou"], 1 / 3)
        self.assertIsNone(boundary_f1(y, y, 1))
        p = y.copy()
        p[3:5, 3:5] = 1
        self.assertEqual(boundary_f1(y, p, 1), 0.)

    def test_invalid_labels_rejected(self):
        with self.assertRaises(ValueError):
            confusion(np.array([[0, 20]]), np.array([[0, 1]]))

    def test_dice_and_ce_are_applied_to_logits(self):
        y = torch.tensor([[[0, 1], [2, 0]]])
        good = torch.nn.functional.one_hot(y, 3).permute(0, 3, 1, 2).float() * 20
        self.assertLess(float(region_loss(good, y)), 1e-5)
        self.assertGreater(float(region_loss(-good, y)), 10.)

    def test_rgb_quantization_endpoints(self):
        image = np.array([[[0, 0, 0], [255, 255, 255]]], np.uint8)
        lut = np.zeros(8 ** 3, np.uint8)
        lut[-1] = 2
        np.testing.assert_array_equal(rgb_predict(image, 8, lut), [[0, 2]])


class DataTests(unittest.TestCase):
    def test_cvd_loading_preserves_orientation_and_label_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test/images").mkdir(parents=True)
            (root / "test/masks").mkdir()
            image = np.zeros((512, 512, 3), np.uint8)
            image[0, 0] = [255, 0, 0]
            mask = np.zeros((512, 512), np.uint8)
            mask[0, 0], mask[-1, -1] = 20, 40
            Image.fromarray(image).save(root / "test/images/a.png")
            Image.fromarray(mask).save(root / "test/masks/a_mask.PNG")
            data = Tiles(root, "test")
            _, x, y = data[0]
            self.assertEqual(float(x[0, 0, 0]), 1.)
            self.assertEqual(int(y[0, 0]), 1)
            self.assertEqual(int(y[-1, -1]), 2)
            with self.assertRaises(ValueError):
                Tiles(root, "test", training=True)

    def test_released_crop_is_not_resized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test/images").mkdir(parents=True)
            (root / "test/labels").mkdir()
            image = np.zeros((100, 100, 3), np.uint8)
            mask = np.ones((100, 100), np.uint8)
            mask[95, 95], mask[96:, :] = 3, 2
            Image.fromarray(image).save(root / "test/images/a.jpg")
            Image.fromarray(mask).save(root / "test/labels/a.png")
            _, x, y = Tiles(root, "test", "exfoliated")[0]
            self.assertEqual(tuple(x.shape), (3, 96, 96))
            self.assertEqual(int(y[-1, -1]), 2)


if __name__ == "__main__":
    unittest.main()
