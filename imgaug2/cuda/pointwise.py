"""GPU-accelerated pointwise operations using CUDA/CuPy.

This module provides GPU-accelerated implementations of pointwise pixel operations
including arithmetic, contrast adjustments, and color transformations, optimized
for batch processing on CUDA-compatible devices.
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


def add(image: cp.ndarray, value: cp.ndarray | float | int) -> cp.ndarray:
    """Add a value to all pixels in an image on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of any shape.
    value : cupy.ndarray, float, or int
        Value to add. Can be a scalar or array of same shape as image.

    Returns
    -------
    cupy.ndarray
        Image with value added, same shape and dtype as input.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.
    """
    img = _require_cupy_array(image, "add")
    dtype = img.dtype

    if is_cupy_array(value):
        val = value
    else:
        val = float(value)

    if cp.issubdtype(dtype, cp.integer):
        mn, mxv = _dtype_min_max(dtype)
        out = img.astype(cp.float32, copy=False) + val
        out = cp.clip(out, mn, mxv).astype(dtype)
        return out

    return img + val


def multiply(image: cp.ndarray, value: cp.ndarray | float | int) -> cp.ndarray:
    """Multiply all pixels in an image by a value on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of any shape.
    value : cupy.ndarray, float, or int
        Multiplicative factor. Can be a scalar or array of same shape as image.

    Returns
    -------
    cupy.ndarray
        Image with value multiplied, same shape and dtype as input.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.
    """
    img = _require_cupy_array(image, "multiply")
    dtype = img.dtype

    if is_cupy_array(value):
        val = value
    else:
        val = float(value)

    if cp.issubdtype(dtype, cp.integer):
        mn, mxv = _dtype_min_max(dtype)
        out = img.astype(cp.float32, copy=False) * val
        out = cp.clip(out, mn, mxv).astype(dtype)
        return out

    return img * val


def linear_contrast(image: cp.ndarray, factor: float) -> cp.ndarray:
    """Apply linear contrast adjustment on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of any shape.
    factor : float
        Contrast factor. Values > 1 increase contrast, < 1 decrease contrast.

    Returns
    -------
    cupy.ndarray
        Contrast-adjusted image with same shape and dtype as input.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.

    Notes
    -----
    Adjusts contrast around midpoint value of 128.0.
    """
    img = _require_cupy_array(image, "linear_contrast")
    dtype = img.dtype

    midpoint = 128.0
    out = img.astype(cp.float32, copy=False)
    out = (out - midpoint) * float(factor) + midpoint

    if cp.issubdtype(dtype, cp.integer):
        mn, mxv = _dtype_min_max(dtype)
        return cp.clip(out, mn, mxv).astype(dtype)
    return out


def invert(
    image: cp.ndarray,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> cp.ndarray:
    """Invert image intensities on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of any shape.
    min_value : float or None, optional
        Minimum value for inversion. If None, inferred from dtype. Default is None.
    max_value : float or None, optional
        Maximum value for inversion. If None, inferred from dtype. Default is None.

    Returns
    -------
    cupy.ndarray
        Inverted image with same shape and dtype as input.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.

    Notes
    -----
    Inversion computed as: (min_value + max_value) - pixel_value
    """
    img = _require_cupy_array(image, "invert")
    dtype = img.dtype

    if min_value is None or max_value is None:
        mn, mxv = _dtype_min_max(dtype)
        min_value = mn if min_value is None else float(min_value)
        max_value = mxv if max_value is None else float(max_value)

    out = (float(min_value) + float(max_value)) - img.astype(cp.float32, copy=False)

    if cp.issubdtype(dtype, cp.integer):
        return cp.clip(out, float(min_value), float(max_value)).astype(dtype)
    return out


def solarize(
    image: cp.ndarray,
    *,
    threshold: float | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> cp.ndarray:
    """Apply solarization effect by inverting pixels above threshold on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of any shape.
    threshold : float or None, optional
        Intensity threshold. Pixels above this are inverted. If None, auto-computed. Default is None.
    min_value : float or None, optional
        Minimum value for inversion. If None, inferred from dtype. Default is None.
    max_value : float or None, optional
        Maximum value for inversion. If None, inferred from dtype. Default is None.

    Returns
    -------
    cupy.ndarray
        Solarized image with same shape and dtype as input.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.

    Notes
    -----
    Pixels >= threshold are inverted, pixels < threshold remain unchanged.
    """
    img = _require_cupy_array(image, "solarize")
    dtype = img.dtype

    if min_value is None or max_value is None:
        mn, mxv = _dtype_min_max(dtype)
        min_value = mn if min_value is None else float(min_value)
        max_value = mxv if max_value is None else float(max_value)

    if threshold is None:
        threshold = 128.0 if float(max_value) >= 255.0 else 0.5

    x = img.astype(cp.float32, copy=False)
    inv = (float(min_value) + float(max_value)) - x
    out = cp.where(x >= float(threshold), inv, x)

    if cp.issubdtype(dtype, cp.integer):
        return cp.clip(out, float(min_value), float(max_value)).astype(dtype)
    return out


def gamma_contrast(
    image: cp.ndarray,
    gamma: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> cp.ndarray:
    """Apply gamma contrast adjustment on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of any shape.
    gamma : float
        Gamma exponent. Values > 1 brighten, < 1 darken the image.
    min_value : float or None, optional
        Minimum value for normalization. If None, inferred from dtype. Default is None.
    max_value : float or None, optional
        Maximum value for normalization. If None, inferred from dtype. Default is None.

    Returns
    -------
    cupy.ndarray
        Gamma-adjusted image with same shape and dtype as input.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.

    Notes
    -----
    Applies power-law transformation: out = min + ((x - min) / (max - min))^gamma * (max - min)
    """
    img = _require_cupy_array(image, "gamma_contrast")
    dtype = img.dtype

    if min_value is None or max_value is None:
        mn, mxv = _dtype_min_max(dtype)
        min_value = mn if min_value is None else float(min_value)
        max_value = mxv if max_value is None else float(max_value)

    if float(max_value) == float(min_value):
        return img

    x = img.astype(cp.float32, copy=False)
    rng = float(max_value) - float(min_value)
    x01 = cp.clip((x - float(min_value)) / rng, 0.0, 1.0)
    out = float(min_value) + cp.power(x01, float(gamma)) * rng

    if cp.issubdtype(dtype, cp.integer):
        return cp.clip(out, float(min_value), float(max_value)).astype(dtype)
    return out


__all__ = [
    "add",
    "gamma_contrast",
    "invert",
    "linear_contrast",
    "multiply",
    "solarize",
]
