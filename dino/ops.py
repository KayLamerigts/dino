import dataclasses

import numpy as np
import numpy.typing as npt
import scipy.ndimage

import dino.image


def _resize_voxels(voxels: np.ndarray, factors_zoom: np.ndarray, order: int) -> np.ndarray:
    # Apply anti aliasing when downscaling, sigma formula taken from skimage:
    # https://github.com/scikit-image/scikit-image/blob/39a94a08ef10b1ae4d6e0e04668c45cde94c55b4/skimage/transform/_warps.py#L163
    aa_sigma = np.maximum(0, ((1 / factors_zoom) - 1) / 2)
    voxels_blurred = scipy.ndimage.gaussian_filter(voxels, aa_sigma, mode="nearest")
    return scipy.ndimage.zoom(
        # order 1 means linear, mode refers to how to pad the volume
        # ie, mode="nearest" does _not_ mean nearest neighbour interpolation, see:
        # https://docs.scipy.org/doc/scipy/tutorial/ndimage.html?highlight=spline%20interpolation#interpolation-boundary-handling
        voxels_blurred,
        zoom=factors_zoom,
        order=order,
        mode="nearest",
        grid_mode=False,
    )


def _resize_image(image: dino.image.Image, size: np.ndarray, order: int) -> dino.image.Image:
    if image.voxels is None:
        raise ValueError(
            "Image does not have voxels. Try again after loading with `create_image(..., load_voxels=True)`."
        )

    factors_zoom = size / image.size
    voxels = image.voxels.astype(np.float32)

    if len(voxels.shape) == 3:
        voxels_resized = _resize_voxels(voxels, factors_zoom, order)
    elif len(voxels.shape) == 4:
        voxels_resized = np.empty([*size, voxels.shape[3]])
        for channel_ix in range(voxels.shape[3]):
            voxels_resized[..., channel_ix] = _resize_voxels(
                voxels[..., channel_ix], factors_zoom, order
            )
    else:
        raise ValueError(
            f"Image should be 3D or 4D, ie, a volume or a volume with channels, but got {voxels.shape}."
        )

    if isinstance(image.voxels.flat[0], np.bool_):
        # a boolean array was casted to 0.0 or 1.0, cast it back to bool here
        voxels_resized = voxels_resized > 0.5

    factor_spacing = (size - 1) / (image.size - 1)
    spacing = image.spacing / factor_spacing

    affine = image.affine.copy()
    affine[:3, :3] = np.dot(image.orientation, np.diag(spacing))
    image = dataclasses.replace(
        image,
        affine=affine,
        size=np.array(voxels_resized.shape[:3]),
        voxels=voxels_resized,
    )

    return image


def resize(image: dino.image.Image, size: npt.ArrayLike, order: int = 1) -> dino.image.Image:
    """Creates an images resized to a specific size.

    Args:
        image: the image to be resized
        size: the output size of the image
        order: what order to use for the interpolation, default 1 is linear

    Returns:
        a newly created image with the specified size
    """
    size = np.asarray(size)

    if size.shape != (3,):
        raise ValueError("size should be a 3D vector")
    if not isinstance(size.flat[0], np.integer):
        raise ValueError("size should be an int vector")
    if not np.all(size > 0):
        raise ValueError("size should only have positive values")

    return _resize_image(image, size, order)


def rescale(image: dino.image.Image, spacing: npt.ArrayLike, order: int = 1) -> dino.image.Image:
    """Creates an images rescaled close to a specific spacing.

    It is possible that the specified spacing leads to fractional first or last pixel.
    The resulting coordinate system is then ill-defined. Therefore, we take the closest
    spacing with a well-defined coordinate system.

    Args:
        image: the image to be resized
        spacing: the output spacing of the image
        order: what order to use for the interpolation, default 1 is linear

    Returns:
        a newly created image with the specified spacing
    """
    spacing = np.asarray(spacing)

    if spacing.shape != (3,):
        raise ValueError("spacing should be a 3D vector")
    if not isinstance(spacing.flat[0], np.floating):
        raise ValueError("spacing should be a float vector")
    if not np.all(spacing > 0):
        raise ValueError("spacing should only have positive values")

    # approximate a size close to the desired spacing
    apx_distance = (image.size - 1) * image.spacing
    apx_size = (apx_distance / spacing) + 1

    size = np.round(apx_size).astype(int)

    return _resize_image(image, size, order)
