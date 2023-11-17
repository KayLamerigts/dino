import dataclasses

import numpy as np

import dino.utils


@dataclasses.dataclass(frozen=True)
class Image:
    """A class representing a spatially referenced volumetric image.

    Attributes:
        affine: A 4x4 affine transformation matrix that maps voxel coordinates to world coordinates.
        voxels: An array representing the volumetric image data.
    """

    affine: np.ndarray  # 4x4
    voxels: np.ndarray  # DxHxW

    def __post_init__(self):
        if self.affine.shape != (4, 4):
            raise ValueError(
                f"Affine matrix must have shape 4x4, but got shape {self.affine.shape}"
            )
        if len(self.voxels.shape) != 3:
            raise ValueError(f"Voxels must have shape DxHxW, but got shape {self.voxels.shape}")

        self.affine.setflags(write=False)
        self.voxels.setflags(write=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Image):
            return False

        return (
            dino.utils.allclose_with_shape_check(self.voxels, other.voxels)
            and np.allclose(self.affine, other.affine)
            # and self.voxels.dtype == other.voxels.dtype
            # and self.affine.dtype == other.affine.dtype
        )

    @property
    def size(self) -> np.ndarray:  # 3
        return np.array(self.voxels.shape)

    @property
    def origin(self) -> np.ndarray:  # 3
        return self.affine[:3, 3]

    @property
    def spacing(self) -> np.ndarray:  # 3
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def orientation(self) -> np.ndarray:  # 3x3
        return self.affine[:3, :3] @ np.diag(1 / self.spacing)
