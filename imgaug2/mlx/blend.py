"""Blending utilities for MLX-compatible pipelines.

This module provides alpha blending wrappers that accept MLX arrays while
delegating the actual blending to CPU utilities. It preserves the input type:
if either input is an MLX array, the output is an MLX array.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

from imgaug2.augmenters import _blend_utils

from ._core import is_mlx_array, require, to_mlx, to_numpy

AlphaInput: TypeAlias = _blend_utils.AlphaInput

if TYPE_CHECKING:
    import mlx.core as _mx

    MlxArray: TypeAlias = _mx.array
else:
    MlxArray: TypeAlias = object

NumpyArray: TypeAlias = NDArray[np.generic]


@overload
def blend_alpha(
    image_fg: NumpyArray,
    image_bg: NumpyArray,
    alpha: AlphaInput,
    eps: float = 1e-2,
) -> NumpyArray: ...


@overload
def blend_alpha(
    image_fg: MlxArray,
    image_bg: MlxArray,
    alpha: AlphaInput,
    eps: float = 1e-2,
) -> MlxArray: ...


def blend_alpha(
    image_fg: object,
    image_bg: object,
    alpha: AlphaInput,
    eps: float = 1e-2,
) -> object:
    """Blend two images using an alpha factor.

    Parameters
    ----------
    image_fg : object
        Foreground image as NumPy array or MLX array.
    image_bg : object
        Background image as NumPy array or MLX array.
    alpha : AlphaInput
        Blending factor(s). See ``imgaug2.augmenters._blend_utils.blend_alpha``.
    eps : float, optional
        Threshold for treating alpha values as 0 or 1 (skips computation).

    Returns
    -------
    object
        Blended image. Returns MLX array if either input was MLX, else NumPy.
    """
    is_fg_mlx = is_mlx_array(image_fg)
    is_bg_mlx = is_mlx_array(image_bg)
    if is_fg_mlx or is_bg_mlx:
        require()

    fg_np = to_numpy(image_fg)
    bg_np = to_numpy(image_bg)
    alpha_np = to_numpy(alpha) if is_mlx_array(alpha) else alpha

    blended = _blend_utils.blend_alpha(fg_np, bg_np, alpha_np, eps=eps)

    if is_fg_mlx or is_bg_mlx:
        return to_mlx(blended)
    return blended


__all__ = ["blend_alpha"]
