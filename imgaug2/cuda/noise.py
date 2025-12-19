"""GPU-accelerated noise operations using CUDA/CuPy.

This module provides GPU-accelerated implementations of noise operations,
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


def additive_gaussian_noise(image: cp.ndarray, scale: float, seed: int | None = None) -> cp.ndarray:
    """Add Gaussian noise to an image on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of any shape.
    scale : float
        Standard deviation of the Gaussian noise distribution.
    seed : int or None, optional
        Random seed for reproducibility. If None, uses current random state. Default is None.

    Returns
    -------
    cupy.ndarray
        Noisy image with same shape and dtype as input.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.
    """
    img = _require_cupy_array(image, "additive_gaussian_noise")
    dtype = img.dtype
    if img.size == 0:
        return img

    if seed is not None:
        cp.random.seed(int(seed))

    x = img.astype(cp.float32, copy=False)
    noise = cp.random.normal(loc=0.0, scale=float(scale), size=x.shape).astype(
        cp.float32, copy=False
    )
    y = x + noise

    if cp.issubdtype(dtype, cp.integer):
        mn, mxv = _dtype_min_max(dtype)
        y = cp.floor(y + 0.5)
        y = cp.clip(y, mn, mxv)
        return y.astype(dtype)
    return y.astype(dtype, copy=False)


def dropout(image: cp.ndarray, p: float, seed: int | None = None) -> cp.ndarray:
    """Apply dropout by zeroing random pixels with given probability on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of shape (H, W), (H, W, C), or (N, H, W, C).
    p : float
        Dropout probability in range [0, 1]. Fraction of pixels to zero out.
    seed : int or None, optional
        Random seed for reproducibility. If None, uses current random state. Default is None.

    Returns
    -------
    cupy.ndarray
        Image with dropout applied, same shape and dtype as input.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.
    ValueError
        If p is not in range [0, 1].

    Notes
    -----
    For images with channels, the dropout mask is shared across all channels.
    """
    img = _require_cupy_array(image, "dropout")
    dtype = img.dtype
    if img.size == 0:
        return img

    p = float(p)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"p must be in [0,1], got {p}")

    if seed is not None:
        cp.random.seed(int(seed))

    x = img.astype(cp.float32, copy=False)

    if x.ndim == 2:
        mask = (cp.random.random(size=x.shape) > p).astype(cp.float32)
    else:
        mask = (cp.random.random(size=x.shape[:-1]) > p).astype(cp.float32)[..., None]

    y = x * mask

    if cp.issubdtype(dtype, cp.integer):
        mn, mxv = _dtype_min_max(dtype)
        y = cp.clip(y, mn, mxv)
        return y.astype(dtype)
    return y.astype(dtype, copy=False)


__all__ = ["additive_gaussian_noise", "dropout"]
