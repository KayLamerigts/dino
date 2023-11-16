import dataclasses

import numpy as np
import numpy.typing as npt
import scipy.ndimage

import dino.structs


def _resize_image(image: dino.structs.Image, size: np.ndarray, order: int) -> dino.structs.Image:
    if image.voxels is None:
        raise ValueError(
            "Image does not have voxels. Try again after loading with `create_image(..., load_voxels=True)`."
        )
    if len(image.voxels.shape) != 3:
        raise ValueError("Image does not have 3D voxels.")

    factors_zoom = size / image.size
    voxels = image.voxels.astype(np.float32)

    # Apply anti aliasing when downscaling, sigma formula taken from skimage:
    # https://github.com/scikit-image/scikit-image/blob/39a94a08ef10b1ae4d6e0e04668c45cde94c55b4/skimage/transform/_warps.py#L163
    aa_sigma = np.maximum(0, ((1 / factors_zoom) - 1) / 2)
    # Mode refers to how to pad the volume, mode="nearest" does _not_ mean nearest neighbour interpolation, see:
    # https://docs.scipy.org/doc/scipy/tutorial/ndimage.html?highlight=spline%20interpolation#interpolation-boundary-handling
    voxels_blurred = scipy.ndimage.gaussian_filter(voxels, aa_sigma, mode="nearest")
    voxels_resized = scipy.ndimage.zoom(
        voxels_blurred,
        zoom=factors_zoom,
        order=order,
        mode="nearest",
        grid_mode=False,
    )

    if isinstance(image.voxels.flat[0], np.bool_):
        # a boolean array was casted to 0.0 or 1.0, cast it back to bool here
        voxels_resized = voxels_resized > 0.5

    factor_spacing = (size - 1) / (image.size - 1)
    spacing = image.spacing / factor_spacing

    affine = image.affine.copy()
    affine[:3, :3] = image.orientation @ np.diag(spacing)
    image = dataclasses.replace(
        image,
        affine=affine,
        size=np.array(voxels_resized.shape[:3]),
        voxels=voxels_resized,
    )

    return image


def resize(image: dino.structs.Image, size: npt.ArrayLike, order: int = 1) -> dino.structs.Image:
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
    if not np.issubdtype(size.dtype, np.integer):
        raise ValueError("size should be an int vector")
    if not np.all(size > 0):
        raise ValueError("size should only have positive values")

    return _resize_image(image, size, order)


def rescale(
    image: dino.structs.Image, spacing: npt.ArrayLike, order: int = 1
) -> dino.structs.Image:
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
    if not np.all(spacing > 0):
        raise ValueError("spacing should only have positive values")

    # approximate a size close to the desired spacing
    apx_distance = (image.size - 1) * image.spacing
    apx_size = (apx_distance / spacing) + 1

    size = np.round(apx_size).astype(int)

    return _resize_image(image, size, order)

# TODO: different api for crop and pad? (bounds vs before/after)

def crop(image: dino.structs.Image, crop_bounds: npt.ArrayLike) -> dino.structs.Image:
    crop_bounds = np.asarray(crop_bounds)

    if crop_bounds.shape != (2, 3):
        raise ValueError("crop_bounds should be a 2x3 matrix")
    if not np.issubdtype(crop_bounds.dtype, np.integer):
        raise ValueError(f"crop_bounds should be an integer vector")

    start, end = crop_bounds

    if not np.all(start >= 0):
        raise ValueError("crop_bounds should only have positive values")
    if not np.all(end <= image.size):
        raise ValueError("crop_bounds should be smaller than the image size")
    if not np.all(start < end):
        raise ValueError("crop_bounds should have start < end")

    voxels = image.voxels[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
    position = (image.affine @ np.array([*start, 1]))[:3]
    affine = image.affine.copy()
    affine[:3, 3] = position

    return dataclasses.replace(image, voxels=voxels, affine=affine, size=np.array(voxels.shape))


def pad(
    image: dino.structs.Image, pad_width: npt.ArrayLike, pad_value: int | float | None = None
) -> dino.structs.Image:
    pad_width = np.asarray(pad_width)

    if pad_width.shape != (2, 3):
        raise ValueError("pad_width should be a 2x3 matrix")
    if not np.issubdtype(pad_width.dtype, np.integer):
        raise ValueError(f"pad_width should be an integer vector")

    before, after = pad_width

    if not np.all(before >= 0):
        raise ValueError("pad_width should only have positive values")
    if not np.all(after >= 0):
        raise ValueError("pad_width should only have positive values")

    pad_value = image.voxels.min() if pad_value is None else pad_value
    voxels = np.pad(image.voxels, (before, after), mode="constant", constant_values=pad_value)
    position = (image.affine @ np.array([*-before, 1]))[:3]
    affine = image.affine.copy()
    affine[:3, 3] = position

    return dataclasses.replace(image, voxels=voxels, affine=affine, size=np.array(voxels.shape))


def canonicalize_reflected_image(image: dino.structs.Image) -> dino.structs.Image:
    """Canonicalizes an image by reflecting it if necessary.

    Args:
        image: the image to be canonicalized

    Returns:
        a newly created image that is canonicalized
    """
    # TODO: AI written not done
    if np.linalg.det(image.orientation) < 0:
        # flip the image
        voxels = image.voxels[::-1, :, :]
        orientation = image.orientation.copy()
        orientation[0, :] *= -1
        affine = image.affine.copy()
        affine[:3, :3] = orientation @ np.diag(image.spacing)
        image = dataclasses.replace(
            image,
            affine=affine,
            voxels=voxels,
        )

    return image
