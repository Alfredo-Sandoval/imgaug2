"""Shared helpers for segmentation augmenters."""

from __future__ import annotations

from imgaug2.augmenters._typing import Array
import imgaug2.imgaug as ia


# TODO merge this into imresize?
def _ensure_image_max_size(image: Array, max_size: int | None, interpolation: str | int) -> Array:
    """Ensure that images do not exceed a required maximum sidelength.

    This downscales to `max_size` if any side violates that maximum.
    The other side is downscaled too so that the aspect ratio is maintained.

    **Supported dtypes**:

    See :func:`~imgaug2.imgaug2.imresize_single_image`.

    Parameters
    ----------
    image : ndarray
        Image to potentially downscale.

    max_size : int
        Maximum length of any side of the image.

    interpolation : string or int
        See :func:`~imgaug2.imgaug2.imresize_single_image`.

    """
    if max_size is not None:
        size = max(image.shape[0], image.shape[1])
        if size > max_size:
            resize_factor = max_size / size
            new_height = int(image.shape[0] * resize_factor)
            new_width = int(image.shape[1] * resize_factor)
            image = ia.imresize_single_image(
                image, (new_height, new_width), interpolation=interpolation
            )
    return image
