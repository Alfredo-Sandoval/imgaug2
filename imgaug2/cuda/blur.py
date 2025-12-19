"""GPU-accelerated blur operations using CUDA/CuPy.

This module provides GPU-accelerated implementations of blur operations,
optimized for batch processing on CUDA-compatible devices.
"""
from __future__ import annotations

import importlib

import numpy as np

from imgaug2.errors import DependencyMissingError
from ._core import cp, is_cupy_array, require


def _require_cupy_array(x: object, fn_name: str) -> cp.ndarray:
    require()
    if not is_cupy_array(x):
        raise TypeError(
            f"{fn_name} expects a cupy.ndarray (GPU resident). "
            "Convert first via `cupy.asarray(...)`."
        )
    return x


def _compute_gaussian_blur_ksize(sigma: float) -> int:
    if sigma < 3.0:
        ksize = 3.3 * sigma
    elif sigma < 5.0:
        ksize = 2.9 * sigma
    else:
        ksize = 2.6 * sigma

    ksize = int(max(ksize, 5))
    if ksize % 2 == 0:
        ksize += 1
    return ksize


def _gaussian_kernel_1d(sigma: float, ksize: int) -> np.ndarray:
    radius = ksize // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * (sigma * sigma)))
    k = k / k.sum()
    return k.astype(np.float32)


def _dtype_min_max(dtype: object) -> tuple[float, float]:
    if cp.issubdtype(dtype, cp.integer):
        info = cp.iinfo(dtype)
        return float(info.min), float(info.max)
    return 0.0, 1.0


def gaussian_blur(image: cp.ndarray, sigma: float, kernel_size: int | None = None) -> cp.ndarray:
    """Apply Gaussian blur to an image on GPU.

    Uses separable 1D convolution via cupyx.scipy.ndimage for efficient GPU computation.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of shape (H, W), (H, W, C), or (N, H, W, C).
    sigma : float
        Standard deviation of the Gaussian kernel. Must be >= 0.
    kernel_size : int or None, optional
        Size of the Gaussian kernel. If None, automatically computed from sigma.
        Must be odd. Default is None.

    Returns
    -------
    cupy.ndarray
        Blurred image with same shape and dtype as input.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.
    ValueError
        If sigma < 0 or image has invalid shape.
    ImportError
        If cupyx.scipy.ndimage is not available.
    """
    img = _require_cupy_array(image, "gaussian_blur")
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if img.size == 0 or sigma == 0.0:
        return img

    try:
        cndimage = importlib.import_module("cupyx.scipy.ndimage")
    except Exception as exc:  # pragma: no cover
        raise DependencyMissingError(
            "cupyx.scipy.ndimage is required for gaussian_blur. "
            "Install a CuPy build that includes cupyx."
        ) from exc

    if kernel_size is None:
        kernel_size = _compute_gaussian_blur_ksize(sigma)
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)

    k1 = _gaussian_kernel_1d(sigma, kernel_size)
    k1_gpu = cp.asarray(k1)

    dtype = img.dtype
    x = img.astype(cp.float32, copy=False)

    if x.ndim == 2:
        y = cndimage.convolve1d(x, k1_gpu, axis=1, mode="reflect")
        y = cndimage.convolve1d(y, k1_gpu, axis=0, mode="reflect")
    elif x.ndim == 3:
        y = cndimage.convolve1d(x, k1_gpu, axis=1, mode="reflect")
        y = cndimage.convolve1d(y, k1_gpu, axis=0, mode="reflect")
    elif x.ndim == 4:
        y = cndimage.convolve1d(x, k1_gpu, axis=2, mode="reflect")
        y = cndimage.convolve1d(y, k1_gpu, axis=1, mode="reflect")
    else:
        raise ValueError(f"gaussian_blur expects 2D/3D/4D image, got shape {img.shape}")

    if cp.issubdtype(dtype, cp.integer):
        mn, mxv = _dtype_min_max(dtype)
        y = cp.floor(y + 0.5)
        y = cp.clip(y, mn, mxv)
        return y.astype(dtype)
    return y.astype(dtype, copy=False)


__all__ = ["gaussian_blur"]
