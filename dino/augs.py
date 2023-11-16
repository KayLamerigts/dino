import dataclasses

import numpy as np
import numpy.typing as npt

import dino.ops
import dino.structs


def random_crop_or_pad_image(
    image: dino.structs.Image,
    size: npt.ArrayLike,
    *,
    pad_value: int | float | None = None,
    rng: np.random.Generator | None = None,
) -> dino.structs.Image:
    """Randomly crops or pads an image to the given size.

    Args:
        image: The image to crop or pad.
        size: The desired size of the output image.
        pad_value (optional): The value to use for padding. Defaults to min value in voxels.
        rng (optional): The random number generator to use. Defaults to default numpy rng.

    Returns:
        The cropped or padded image.
    """
    size = np.asarray(size)
    if np.any(size <= 0):
        raise ValueError("size should be positive")

    pad_value = pad_value or image.voxels.min()
    rng = rng or np.random.default_rng()

    crop_size = np.minimum(size, image.size)
    pad_size = np.maximum(size, image.size)

    crop_margin = np.maximum(0, image.size - crop_size)
    pad_margin = np.maximum(0, pad_size - crop_size)

    crop_offsets = rng.integers(0, crop_margin, endpoint=True)
    pad_offsets = rng.integers(0, pad_margin, endpoint=True)

    voxels = image.voxels[
        crop_offsets[0] : crop_offsets[0] + crop_size[0],
        crop_offsets[1] : crop_offsets[1] + crop_size[1],
        crop_offsets[2] : crop_offsets[2] + crop_size[2],
    ]

    padding_ends = pad_size - (pad_offsets + crop_size)
    paddings = [(pad_offsets[axis], padding_ends[axis]) for axis in range(3)]
    voxels = np.pad(voxels, paddings, constant_values=pad_value)

    translation = crop_offsets - pad_offsets
    position = (image.affine @ np.array([*translation, 1]))[:3]
    affine = image.affine.copy()
    affine[:3, 3] = position

    return dataclasses.replace(image, voxels=voxels, affine=affine)
