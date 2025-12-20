"""Artistic effects for MLX-compatible pipelines.

These wrappers provide MLX-friendly entry points for artistic augmentations
implemented on CPU, preserving MLX array inputs/outputs for pipeline use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

from imgaug2.augmenters import artistic as _artistic

from ._core import is_mlx_array, require, to_mlx, to_numpy

if TYPE_CHECKING:
    import mlx.core as _mx

    MlxArray: TypeAlias = _mx.array
else:
    MlxArray: TypeAlias = object

NumpyArray: TypeAlias = NDArray[np.generic]


@overload
def stylize_cartoon(
    image: NumpyArray,
    blur_ksize: int = 3,
    segmentation_size: float = 1.0,
    saturation: float = 2.0,
    edge_prevalence: float = 1.0,
    suppress_edges: bool = True,
    from_colorspace: str = _artistic.colorlib.CSPACE_RGB,
) -> NumpyArray: ...


@overload
def stylize_cartoon(
    image: MlxArray,
    blur_ksize: int = 3,
    segmentation_size: float = 1.0,
    saturation: float = 2.0,
    edge_prevalence: float = 1.0,
    suppress_edges: bool = True,
    from_colorspace: str = _artistic.colorlib.CSPACE_RGB,
) -> MlxArray: ...


def stylize_cartoon(
    image: object,
    blur_ksize: int = 3,
    segmentation_size: float = 1.0,
    saturation: float = 2.0,
    edge_prevalence: float = 1.0,
    suppress_edges: bool = True,
    from_colorspace: str = _artistic.colorlib.CSPACE_RGB,
) -> object:
    """Convert image style to a cartoon-like appearance."""
    is_input_mlx = is_mlx_array(image)
    if is_input_mlx:
        require()

    image_np = to_numpy(image)
    result = _artistic.stylize_cartoon(
        image_np,
        blur_ksize=blur_ksize,
        segmentation_size=segmentation_size,
        saturation=saturation,
        edge_prevalence=edge_prevalence,
        suppress_edges=suppress_edges,
        from_colorspace=from_colorspace,
    )

    if is_input_mlx:
        return to_mlx(result)
    return result


__all__ = ["stylize_cartoon"]
