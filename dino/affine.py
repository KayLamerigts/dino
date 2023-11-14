import numpy as np


def get_translation(affine: np.ndarray) -> np.ndarray:
    return affine[:3, 3]


def get_rotation(affine: np.ndarray) -> np.ndarray:
    return affine[:3, :2].flatten()


def get_scale(affine: np.ndarray) -> np.ndarray:
    return np.linalg.norm(affine[:3, :2], axis=0)
