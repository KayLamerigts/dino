import unittest

import numpy as np
import pydicom

from dino import dicom


class TestLoadAffine(unittest.TestCase):
    def test_load_affine_identity(self):
        slice_one = pydicom.Dataset()
        slice_one.PixelSpacing = [1, 1]
        slice_one.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        slice_one.ImagePositionPatient = [0, 0, 0]

        slice_two = pydicom.Dataset()
        slice_two.PixelSpacing = [1, 1]
        slice_two.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        slice_two.ImagePositionPatient = [0, 0, 1]

        affine = dicom.load_affine([slice_one, slice_two])

        np.testing.assert_array_equal(affine, np.eye(4))


if __name__ == "__main__":
    unittest.main()
