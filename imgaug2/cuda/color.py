"""GPU-accelerated color operations using CUDA/CuPy.

This module provides GPU-accelerated implementations of color transformations,
optimized for batch processing on CUDA-compatible devices.
"""
from __future__ import annotations

from ._core import cp, is_cupy_array, require


def _require_cupy_array(x: object, fn_name: str) -> cp.ndarray:
    require()
    if not is_cupy_array(x):
        raise TypeError(
            f"{fn_name} expects a cupy.ndarray (GPU resident). "
            "Convert first via `cupy.asarray(...)`."
        )
    return x


def _dtype_min_max(dtype: object) -> tuple[float, float]:
    if cp.issubdtype(dtype, cp.integer):
        info = cp.iinfo(dtype)
        return float(info.min), float(info.max)
    return 0.0, 1.0


def grayscale(image: cp.ndarray, alpha: float = 1.0) -> cp.ndarray:
    """Convert RGB image to grayscale on GPU using luminance weights.

    Parameters
    ----------
    image : cupy.ndarray
        Input RGB image of shape (H, W, C), where C is typically 3.
    alpha : float, optional
        Blending factor. 1.0 = full grayscale, 0.0 = original image. Default is 1.0.

    Returns
    -------
    cupy.ndarray
        Grayscale image with same shape and dtype as input.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.

    Notes
    -----
    Uses standard luminance weights: [0.299, 0.587, 0.114] for R, G, B channels.
    """
    img = _require_cupy_array(image, "grayscale")
    dtype = img.dtype
    if img.size == 0:
        return img

    x = img.astype(cp.float32, copy=False)
    weights = cp.asarray([0.299, 0.587, 0.114], dtype=cp.float32)
    gray = cp.sum(x * weights, axis=-1, keepdims=True)
    gray = cp.broadcast_to(gray, x.shape)

    a = float(alpha)
    if a < 1.0:
        y = x * (1.0 - a) + gray * a
    else:
        y = gray

    if cp.issubdtype(dtype, cp.integer):
        mn, mxv = _dtype_min_max(dtype)
        y = cp.floor(y + 0.5)
        y = cp.clip(y, mn, mxv)
        return y.astype(dtype)
    return y.astype(dtype, copy=False)


__all__ = ["grayscale"]
