"""Convolutional utilities for MLX-compatible pipelines.

This module provides a wrapper around the CPU convolution helpers so they can
accept MLX arrays and return MLX arrays for pipeline compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

from imgaug2.augmenters import convolutional as _conv

from ._core import is_mlx_array, require, to_mlx, to_numpy

if TYPE_CHECKING:
    import mlx.core as _mx

    MlxArray: TypeAlias = _mx.array
else:
    MlxArray: TypeAlias = object

NumpyArray: TypeAlias = NDArray[np.generic]
KernelArray: TypeAlias = NDArray[np.generic]
KernelInput: TypeAlias = KernelArray | list[KernelArray | None]


def _to_numpy_kernel(kernel: KernelInput | MlxArray) -> KernelInput:
    if is_mlx_array(kernel):
        return to_numpy(kernel)
    if isinstance(kernel, list):
        converted: list[KernelArray | None] = []
        for k in kernel:
            if k is None:
                converted.append(None)
            elif is_mlx_array(k):
                converted.append(to_numpy(k))
            else:
                converted.append(k)
        return converted
    return kernel


@overload
def convolve(image: NumpyArray, kernel: KernelInput) -> NumpyArray: ...


@overload
def convolve(image: MlxArray, kernel: KernelInput) -> MlxArray: ...


def convolve(image: object, kernel: KernelInput | MlxArray) -> object:
    """Apply a convolution kernel to an image.

    This function delegates to the CPU implementation and preserves the input
    type: MLX inputs are returned as MLX arrays, NumPy inputs as NumPy arrays.
    """
    is_input_mlx = is_mlx_array(image)
    if is_input_mlx or is_mlx_array(kernel):
        require()

    image_np = to_numpy(image)
    kernel_np = _to_numpy_kernel(kernel)

    result = _conv.convolve(image_np, kernel_np)

    if is_input_mlx:
        return to_mlx(result)
    return result


__all__ = ["convolve"]
