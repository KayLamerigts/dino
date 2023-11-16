import numpy as np
import numpy.typing as npt

import dino.structs


def create_fake_image(size: npt.ArrayLike = (16, 16, 16)) -> dino.structs.Image:
    affine = np.eye(4)
    voxels = np.zeros(size, dtype=np.float32)
    return dino.structs.Image(affine, voxels)
