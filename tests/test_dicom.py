import unittest

import numpy as np

import dino.dicom
from testing import faking


class TestLoadAffine(unittest.TestCase):
    def test_load_affine_identity(self):
        slice_one = faking.create_empty_pydicom_dataset()
        slice_one.PixelSpacing = [1, 1]
        slice_one.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        slice_one.ImagePositionPatient = [0, 0, 0]

        slice_two = faking.create_empty_pydicom_dataset()
        slice_two.PixelSpacing = [1, 1]
        slice_two.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        slice_two.ImagePositionPatient = [0, 0, 1]

        image = dino.dicom.create_image([slice_one, slice_two])

        np.testing.assert_array_equal(image.affine, np.eye(4))


if __name__ == "__main__":
    unittest.main()
