"""PIL-compatible operations for MLX-compatible pipelines.

These wrappers mirror ``imgaug2.augmenters.pillike`` functions but allow MLX
arrays as inputs and preserve MLX outputs for on-device pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

from imgaug2.augmenters import pillike as _pillike

from ._core import is_mlx_array, require, to_mlx, to_numpy

if TYPE_CHECKING:
    import mlx.core as _mx

    MlxArray: TypeAlias = _mx.array
else:
    MlxArray: TypeAlias = object

NumpyArray: TypeAlias = NDArray[np.generic]


def _wrap_pillike(func: object, image: object, *args: object, **kwargs: object) -> object:
    is_input_mlx = is_mlx_array(image)
    if is_input_mlx:
        require()

    image_np = to_numpy(image)
    result = func(image_np, *args, **kwargs)

    if is_input_mlx:
        return to_mlx(result)
    return result


@overload
def autocontrast(image: NumpyArray, cutoff: int = 0, ignore: _pillike.IgnoreValues = None) -> NumpyArray: ...


@overload
def autocontrast(image: MlxArray, cutoff: int = 0, ignore: _pillike.IgnoreValues = None) -> MlxArray: ...


def autocontrast(
    image: object,
    cutoff: int = 0,
    ignore: _pillike.IgnoreValues = None,
) -> object:
    """Maximize image contrast (PIL-compatible)."""
    return _wrap_pillike(_pillike.autocontrast, image, cutoff=cutoff, ignore=ignore)


@overload
def equalize(image: NumpyArray, mask: NumpyArray | None = None) -> NumpyArray: ...


@overload
def equalize(image: MlxArray, mask: MlxArray | None = None) -> MlxArray: ...


def equalize(image: object, mask: object | None = None) -> object:
    """Histogram equalization (PIL-compatible)."""
    is_input_mlx = is_mlx_array(image)
    is_mask_mlx = is_mlx_array(mask) if mask is not None else False
    if is_input_mlx or is_mask_mlx:
        require()

    image_np = to_numpy(image)
    mask_np = to_numpy(mask) if mask is not None else None
    result = _pillike.equalize(image_np, mask=mask_np)

    if is_input_mlx:
        return to_mlx(result)
    return result


def enhance_color(image: object, factor: float) -> object:
    """Enhance color saturation (PIL-compatible)."""
    return _wrap_pillike(_pillike.enhance_color, image, factor)


def enhance_contrast(image: object, factor: float) -> object:
    """Enhance contrast (PIL-compatible)."""
    return _wrap_pillike(_pillike.enhance_contrast, image, factor)


def enhance_brightness(image: object, factor: float) -> object:
    """Enhance brightness (PIL-compatible)."""
    return _wrap_pillike(_pillike.enhance_brightness, image, factor)


def enhance_sharpness(image: object, factor: float) -> object:
    """Enhance sharpness (PIL-compatible)."""
    return _wrap_pillike(_pillike.enhance_sharpness, image, factor)


def filter_blur(image: object) -> object:
    """Apply PIL BLUR filter."""
    return _wrap_pillike(_pillike.filter_blur, image)


def filter_smooth(image: object) -> object:
    """Apply PIL SMOOTH filter."""
    return _wrap_pillike(_pillike.filter_smooth, image)


def filter_smooth_more(image: object) -> object:
    """Apply PIL SMOOTH_MORE filter."""
    return _wrap_pillike(_pillike.filter_smooth_more, image)


def filter_edge_enhance(image: object) -> object:
    """Apply PIL EDGE_ENHANCE filter."""
    return _wrap_pillike(_pillike.filter_edge_enhance, image)


def filter_edge_enhance_more(image: object) -> object:
    """Apply PIL EDGE_ENHANCE_MORE filter."""
    return _wrap_pillike(_pillike.filter_edge_enhance_more, image)


def filter_find_edges(image: object) -> object:
    """Apply PIL FIND_EDGES filter."""
    return _wrap_pillike(_pillike.filter_find_edges, image)


def filter_contour(image: object) -> object:
    """Apply PIL CONTOUR filter."""
    return _wrap_pillike(_pillike.filter_contour, image)


def filter_emboss(image: object) -> object:
    """Apply PIL EMBOSS filter."""
    return _wrap_pillike(_pillike.filter_emboss, image)


def filter_sharpen(image: object) -> object:
    """Apply PIL SHARPEN filter."""
    return _wrap_pillike(_pillike.filter_sharpen, image)


def filter_detail(image: object) -> object:
    """Apply PIL DETAIL filter."""
    return _wrap_pillike(_pillike.filter_detail, image)


__all__ = [
    "autocontrast",
    "equalize",
    "enhance_color",
    "enhance_contrast",
    "enhance_brightness",
    "enhance_sharpness",
    "filter_blur",
    "filter_smooth",
    "filter_smooth_more",
    "filter_edge_enhance",
    "filter_edge_enhance_more",
    "filter_find_edges",
    "filter_contour",
    "filter_emboss",
    "filter_sharpen",
    "filter_detail",
]
