import dataclasses

import numpy as np

import dino.utils


@dataclasses.dataclass(frozen=True)
class Image:
    """A class representing a spatially referenced volumetric image with or without multiple channels.

    Attributes:
        affine: A 4x4 affine transformation matrix that maps voxel coordinates to world coordinates.
        size: A 3-element array representing the size of the image in voxels along each axis.
        voxels: An array representing the image data with (4D) or without (3D) channels.
    """

    affine: np.ndarray  # 4x4
    size: np.ndarray  # 3
    voxels: np.ndarray  # DxHxW[xC]

    def __post_init__(self):
        if self.affine.shape != (4, 4):
            raise ValueError(
                f"Affine matrix must have shape (4, 4), but got shape {self.affine.shape}"
            )
        if self.size.shape != (3,):
            raise ValueError(f"Size must have shape (3,), but got shape {self.size.shape}")
        if len(self.voxels.shape) < 3 or len(self.voxels.shape) > 4:
            raise ValueError(f"Voxels must have shape DxHxW[xC], but got shape {self.voxels.shape}")

        self.affine.setflags(write=False)
        self.size.setflags(write=False)
        self.voxels.setflags(write=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Image):
            return False

        return (
            dino.utils.allclose_with_shape_check(self.voxels, other.voxels)
            and np.allclose(self.size, other.size)
            and np.allclose(self.affine, other.affine)
            # and self.voxels.dtype == other.voxels.dtype
            # and self.size.dtype == other.size.dtype
            # and self.affine.dtype == other.affine.dtype
        )

    @property
    def position(self) -> np.ndarray:
        return get_translation(self.affine)

    @property
    def spacing(self) -> np.ndarray:  # 3
        return get_scaling(self.affine)

    @property
    def orientation(self) -> np.ndarray:
        return get_rotation(self.affine)


def get_translation(affine: np.ndarray) -> np.ndarray:  # 3
    return affine[:3, 3]


def get_rotation(affine: np.ndarray) -> np.ndarray:  # 3x3
    return np.dot(affine[:3, :3], np.diag(1 / get_scaling(affine)))


def get_scaling(affine: np.ndarray) -> np.ndarray:  # 3
    return np.linalg.norm(affine[:3, :3], axis=0)
