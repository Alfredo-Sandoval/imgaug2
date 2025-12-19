"""MLX-friendly wrappers for edge detection ops.

This module currently provides a CPU-backed Canny edge detector that
preserves MLX array inputs/outputs for pipeline compatibility.
"""

from __future__ import annotations

import cv2
import numpy as np

from imgaug2.imgaug import _normalize_cv2_input_arr_

from ._core import is_mlx_array, require, to_mlx, to_numpy


def canny(
    image: object,
    threshold1: float,
    threshold2: float,
    *,
    sobel_kernel_size: int = 3,
    l2_gradient: bool = True,
) -> object:
    """
    Apply Canny edge detection to an image.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Shapes (H, W) or (H, W, C).
    threshold1 : float
        Lower hysteresis threshold.
    threshold2 : float
        Upper hysteresis threshold.
    sobel_kernel_size : int, default 3
        Aperture size for the Sobel operator (must be odd, <= 7).
    l2_gradient : bool, default True
        Whether to use a more accurate L2 gradient norm.

    Returns
    -------
    object
        Boolean edge mask. Returns MLX array if input was MLX, else NumPy.
    """
    is_input_mlx = is_mlx_array(image)
    if is_input_mlx:
        require()
    img_np = to_numpy(image)

    if img_np.dtype != np.uint8:
        raise ValueError("canny expects uint8 input")

    if img_np.ndim == 3:
        img_np = img_np[..., :3]
    elif img_np.ndim != 2:
        raise ValueError(f"canny expects 2D or 3D input, got shape {img_np.shape}")

    sobel_kernel_size = int(sobel_kernel_size)
    if sobel_kernel_size < 0 or sobel_kernel_size > 7:
        raise ValueError("sobel_kernel_size must be in [0, 7]")
    if sobel_kernel_size % 2 == 0 and sobel_kernel_size != 0:
        sobel_kernel_size -= 1

    img_np = _normalize_cv2_input_arr_(img_np)
    edges = cv2.Canny(
        img_np,
        threshold1=float(threshold1),
        threshold2=float(threshold2),
        apertureSize=sobel_kernel_size,
        L2gradient=bool(l2_gradient),
    )
    mask = edges > 0

    if is_input_mlx:
        return to_mlx(mask)
    return mask


__all__ = ["canny"]
