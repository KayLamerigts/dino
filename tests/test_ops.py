import unittest

import numpy as np

import dino.ops
from testing import faking


class TestResizeImage(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.image = faking.create_fake_image((16, 16, 16))

    def test_same_size(self):
        image_resized = dino.ops.resize(self.image, (16, 16, 16))
        self.assertEqual(image_resized, self.image)

    def test_smaller_size(self):
        image_resized = dino.ops.resize(self.image, (8, 8, 8))
        np.testing.assert_array_equal(image_resized.size, (8, 8, 8))


class TestCropImage(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.image = faking.create_fake_image((16, 16, 16))

    def test_same_size(self):
        image_cropped = dino.ops.crop(self.image, ((0, 0, 0), (16, 16, 16)))
        self.assertEqual(image_cropped, self.image)

    def test_smaller_size(self):
        image_cropped = dino.ops.crop(self.image, ((0, 0, 0), (8, 8, 8)))
        np.testing.assert_array_equal(image_cropped.size, (8, 8, 8))
