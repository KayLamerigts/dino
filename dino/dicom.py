import itertools

import numpy as np
import pydicom


class DicomError(ValueError):
    pass


def _verify_contains_attribute_per_slice(slices: list[pydicom.Dataset], attribute: str) -> bool:
    try:
        attribute_values = (getattr(slice, attribute) for slice in slices)
    except AttributeError:
        raise DicomError(f"Not all slices have {attribute}.")
    if not all(value for value in attribute_values):
        raise DicomError(f"Not all slices have {attribute}.")


def _verify_identical_attribute_per_slice(slices: list[pydicom.Dataset], attribute: str) -> bool:
    attribute_values = (getattr(slice, attribute) for slice in slices)
    if not all(value == attribute_values[0] for value in attribute_values):
        raise DicomError(f"Not all slices have identical {attribute} values.")


def _load_slice(slice: pydicom.Dataset) -> np.ndarray:
    return slice.pixel_array * slice.RescaleSlope + slice.RescaleIntercept


def create_scan(
    slices: list[pydicom.Dataset],
    load_voxels: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if len(slices) < 2:
        raise DicomError("Not enough slices to create scan.")

    _verify_identical_attribute_per_slice(slices, "PixelSpacing")
    _verify_identical_attribute_per_slice(slices, "ImageOrientationPatient")

    _verify_contains_attribute_per_slice(slices, "ImagePositionPatient")
    _verify_contains_attribute_per_slice(slices, "RescaleSlope")
    _verify_contains_attribute_per_slice(slices, "RescaleIntercept")

    # Rotation
    orientation_x = np.array(slices[0].ImageOrientationPatient[:3])
    orientation_y = np.array(slices[0].ImageOrientationPatient[3:])
    orientation_z = np.cross(orientation_x, orientation_y)
    orientation_z /= np.linalg.norm(orientation_z)
    orientation = np.array([orientation_x, orientation_y, orientation_z])

    if not np.isclose(np.linalg.det(orientation), 1, atol=1e-6):
        raise DicomError("Orientation matrix is not orthogonal.")

    slices = sorted(slices, key=lambda s: np.dot(np.array(s.ImagePositionPatient), orientation_z))
    header = slices[0]

    # Translation
    position = np.array(header.ImagePositionPatient)

    # Scale
    spacing_x, spacing_y = header.PixelSpacing
    spacing_z = np.linalg.norm(position - np.array(slices[1].ImagePositionPatient))
    spacing = np.diag([spacing_x, spacing_y, spacing_z])

    for slice_prev, slice_next in itertools.pairwise(slices):
        slice_prev_position = np.array(slice_prev.ImagePositionPatient)
        slice_next_position = np.array(slice_next.ImagePositionPatient)
        inter_slice_vector = slice_prev_position - slice_next_position

        # Check if the spacing is approximately equal for each slice
        slice_spacing = np.linalg.norm(inter_slice_vector)
        if not np.isclose(slice_spacing, spacing_z, atol=1e-6):
            raise DicomError("Spacing between slices is not equal.")

        # Check that the slices are aligned with orientation_z
        slice_spacing_proj = np.abs(np.dot(inter_slice_vector, orientation_z))
        if not np.isclose(slice_spacing_proj, slice_spacing, atol=1e-6):
            raise DicomError("Slices are not aligned along z-axis.")

        # Redundant check, because all IOP are the same.
        # Check orthogonality of slice orientations with orientation_z
        slice_orientation_x = np.array(slice_prev.ImageOrientationPatient[:3])
        slice_orientation_y = np.array(slice_next.ImageOrientationPatient[3:])
        if not np.isclose(np.dot(slice_orientation_x, orientation_z), 0, atol=1e-6):
            raise DicomError("Slice x-orientation is not orthogonal to z-axis.")
        if not np.isclose(np.dot(slice_orientation_y, orientation_z), 0, atol=1e-6):
            raise DicomError("Slice y-orientation is not orthogonal to z-axis.")

    # Voxels & Size
    if load_voxels:
        voxels = np.stack([_load_slice(slice) for slice in slices])
        size = np.array(voxels.shape)
    else:
        _verify_identical_attribute_per_slice(slices, "Rows")
        _verify_identical_attribute_per_slice(slices, "Columns")
        voxels = None
        size = np.array([len(slices), header.Rows, header.Columns])

    # Affine
    rotation_scale = np.dot(orientation, spacing)
    homogeneous_row = np.array([0, 0, 0, 1]).reshape(1, 4)
    affine = np.r_[np.c_[rotation_scale, position], homogeneous_row]

    return affine, size, voxels
