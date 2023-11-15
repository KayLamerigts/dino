import dataclasses

import numpy as np


@dataclasses.dataclass
class Image:
    affine: np.ndarray
    size: np.ndarray
    voxels: np.ndarray | None
