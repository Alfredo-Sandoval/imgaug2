"""Segment replacement helpers for segmentation augmenters."""

from __future__ import annotations

from typing import cast

import numpy as np

import imgaug2.dtypes as iadt
from imgaug2.augmenters._typing import Array
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _NUMBA_INSTALLED as _NUMBA_INSTALLED_FALLBACK, _numbajit


def _is_numba_installed() -> bool:
    """Check numba availability with segmentation-module overrides."""
    try:
        from imgaug2.augmenters import segmentation as _segmentation

        return bool(getattr(_segmentation, "_NUMBA_INSTALLED", _NUMBA_INSTALLED_FALLBACK))
    except Exception:
        return bool(_NUMBA_INSTALLED_FALLBACK)


# TODO add the old skimage method here for 512x512+ images as it starts to
#      be faster for these areas
# TODO incorporate this dtype support in the dtype sections of docstrings for
#      Superpixels and segment_voronoi()
@legacy(version="0.5.0")
def replace_segments_(image: Array, segments: Array, replace_flags: Array | None) -> Array:
    """Replace segments in images by their average colors in-place.

    This expects an image ``(H,W,[C])`` and an integer segmentation
    map ``(H,W)``. The segmentation map must contain the same id for pixels
    that are supposed to be replaced by the same color ("segments").
    For each segement, the average color is computed and used as the
    replacement.


    **Supported dtypes**:

    * ``uint8``: yes; indirectly tested
    * ``uint16``: yes; indirectly tested
    * ``uint32``: yes; indirectly tested
    * ``uint64``: no; not tested
    * ``int8``: yes; indirectly tested
    * ``int16``: yes; indirectly tested
    * ``int32``: yes; indirectly tested
    * ``int64``: no; not tested
    * ``float16``: ?; not tested
    * ``float32``: ?; not tested
    * ``float64``: ?; not tested
    * ``float128``: ?; not tested
    * ``bool``: yes; indirectly tested

    Parameters
    ----------
    image : ndarray
        An image of shape ``(H,W,[C])``.
        This image may be changed in-place.
        The function is currently not tested for float dtypes.

    segments : ndarray
        A ``(H,W)`` integer array containing the same ids for pixels belonging
        to the same segment.

    replace_flags : ndarray or None
        A boolean array containing at the ``i`` th index a flag denoting
        whether the segment with id ``i`` should be replaced by its average
        color. If the flag is ``False``, the original image pixels will be
        kept unchanged for that flag.
        If this is ``None``, all segments will be replaced.

    Returns
    -------
    ndarray
        The image with replaced pixels.
        Might be the same image as was provided via `image`.

    """
    from imgaug2.mlx._core import is_mlx_array

    if (
        is_mlx_array(image)
        or is_mlx_array(segments)
        or (replace_flags is not None and is_mlx_array(replace_flags))
    ):
        from imgaug2.mlx import segmentation as mlx_segmentation

        return cast(Array, mlx_segmentation.replace_segments_(image, segments, replace_flags))

    assert replace_flags is None or replace_flags.dtype.kind == "b"

    input_shape = image.shape
    if 0 in image.shape:
        return image

    if len(input_shape) == 2:
        image = image[:, :, np.newaxis]

    nb_segments = None
    bad_dtype = image.dtype not in {iadt._UINT8_DTYPE, iadt._INT8_DTYPE}
    if bad_dtype or not _is_numba_installed():
        func = _replace_segments_np_
    else:
        max_id = np.max(segments)
        nb_segments = 1 + max_id
        func = _replace_segments_numba_dispatcher_

    result = func(image, segments, replace_flags, nb_segments)

    if len(input_shape) == 2:
        return result[:, :, 0]
    return result


@legacy(version="0.5.0")
def _replace_segments_np_(
    image: Array, segments: Array, replace_flags: Array | None, _nb_segments: int | None
) -> Array:
    seg_ids = np.unique(segments)
    if replace_flags is None:
        replace_flags = np.ones((len(seg_ids),), dtype=bool)
    for i, seg_id in enumerate(seg_ids):
        if replace_flags[i % len(replace_flags)]:
            mask = segments == seg_id
            mean_color = np.average(image[mask, :], axis=(0,))
            image[mask] = mean_color
    return image


@legacy(version="0.5.0")
def _replace_segments_numba_dispatcher_(
    image: Array, segments: Array, replace_flags: Array | None, nb_segments: int
) -> Array:
    if replace_flags is None:
        replace_flags = np.ones((nb_segments,), dtype=bool)
    elif not np.any(replace_flags[:nb_segments]):
        return image

    average_colors = _replace_segments_numba_collect_avg_colors(
        image, segments, replace_flags, nb_segments, image.dtype
    )
    image = _replace_segments_numba_apply_avg_cols_(image, segments, replace_flags, average_colors)
    return image


@legacy(version="0.5.0")
@_numbajit(nopython=True, nogil=True, cache=True)
def _replace_segments_numba_collect_avg_colors(
    image: Array,
    segments: Array,
    replace_flags: Array,
    nb_segments: int,
    output_dtype: np.dtype[np.generic],
) -> Array:
    height, width, nb_channels = image.shape
    nb_flags = len(replace_flags)

    average_colors = np.zeros((nb_segments, nb_channels), dtype=np.float64)

    counters = np.zeros((nb_segments,), dtype=np.int32)
    for seg_id in range(nb_segments):
        if not replace_flags[seg_id % nb_flags]:
            counters[seg_id] = -1

    for y in range(height):
        for x in range(width):
            seg_id = segments[y, x]
            count = counters[seg_id]

            if count != -1:
                col = image[y, x, :]
                average_colors[seg_id] += col
                counters[seg_id] += 1

    counters = np.maximum(counters, 1)
    counters = counters.reshape((-1, 1))
    average_colors /= counters

    average_colors = average_colors.astype(output_dtype)
    return average_colors


@legacy(version="0.5.0")
@_numbajit(nopython=True, nogil=True, cache=True)
def _replace_segments_numba_apply_avg_cols_(
    image: Array, segments: Array, replace_flags: Array, average_colors: Array
) -> Array:
    height, width = image.shape[0:2]
    nb_flags = len(replace_flags)

    for y in range(height):
        for x in range(width):
            seg_id = segments[y, x]
            if replace_flags[seg_id % nb_flags]:
                image[y, x, :] = average_colors[seg_id]

    return image
