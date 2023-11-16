import unittest

import numpy as np
import pydicom

import dino.dicom


def create_empty_pydicom_dataset() -> pydicom.Dataset:
    dicom_dataset = pydicom.Dataset()

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    dicom_dataset.file_meta = file_meta

    dicom_dataset.is_little_endian = True
    dicom_dataset.is_implicit_VR = True
    dicom_dataset.preamble = b"DICM".rjust(128, bytes([0]))

    dicom_dataset.BitsAllocated = 16
    dicom_dataset.BitsStored = 16
    dicom_dataset.SamplesPerPixel = 1
    dicom_dataset.PhotometricInterpretation = "MONOCHROME1"
    dicom_dataset.PixelRepresentation = 0

    dicom_dataset.PixelSpacing = [1, 1]
    dicom_dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dicom_dataset.ImagePositionPatient = [0, 0, 0]
    dicom_dataset.RescaleSlope = 1
    dicom_dataset.RescaleIntercept = 0

    set_pydicom_pixel_data(dicom_dataset, np.zeros((512, 512), dtype=np.int16))

    dicom_dataset.SOPInstanceUID = pydicom.uid.generate_uid()

    return dicom_dataset


def set_pydicom_pixel_data(slice: pydicom.Dataset, pixel_array: np.ndarray) -> pydicom.Dataset:
    slice.Rows = pixel_array.shape[0]
    slice.Columns = pixel_array.shape[1]
    slice.PixelData = pixel_array.tobytes()
    return slice


class TestLoadAffine(unittest.TestCase):
    def test_load_affine_identity(self):
        slice_one = create_empty_pydicom_dataset()
        slice_one.PixelSpacing = [1, 1]
        slice_one.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        slice_one.ImagePositionPatient = [0, 0, 0]

        slice_two = create_empty_pydicom_dataset()
        slice_two.PixelSpacing = [1, 1]
        slice_two.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        slice_two.ImagePositionPatient = [0, 0, 1]

        image = dino.dicom.create_image([slice_one, slice_two])

        np.testing.assert_array_equal(image.affine, np.eye(4))


if __name__ == "__main__":
    unittest.main()
