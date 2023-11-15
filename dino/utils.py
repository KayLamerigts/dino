import numpy as np


def allclose_with_shape_check(a, b, rtol=1e-05, atol=1e-08):
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, rtol=rtol, atol=atol)
