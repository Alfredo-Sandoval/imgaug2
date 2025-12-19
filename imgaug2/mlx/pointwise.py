"""MLX-accelerated pointwise intensity operations for image manipulation.

This module provides hardware-accelerated pixel-wise operations using Apple's MLX
framework. These operations modify pixel intensities individually or uniformly
across the image without considering spatial neighborhoods.

Supported operations include addition, multiplication, contrast adjustment, inversion,
solarization, and gamma correction. All functions preserve input array type (NumPy or MLX).

Examples
--------
>>> import numpy as np  # doctest: +SKIP
>>> from imgaug2.mlx.pointwise import add, multiply, linear_contrast, invert  # doctest: +SKIP
>>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)  # doctest: +SKIP
>>> brightened = add(img, 50)  # doctest: +SKIP
>>> darkened = multiply(img, 0.5)  # doctest: +SKIP
>>> contrasted = linear_contrast(img, factor=1.5)  # doctest: +SKIP
>>> inverted = invert(img)  # doctest: +SKIP
"""

from __future__ import annotations

import numpy as np

from ._core import ensure_float32, is_mlx_array, mx, require, to_mlx, to_numpy


def _infer_min_max_for_invert(dtype: object) -> tuple[float, float]:
    """Infer min/max value range from dtype for invert/solarize operations.

    Internal helper that determines appropriate value range based on data type.
    """
    # numpy dtype path
    if isinstance(dtype, np.dtype):
        if dtype.kind in {"u", "i"}:
            info = np.iinfo(dtype)
            return float(info.min), float(info.max)
        if dtype.kind == "f":
            # imgaug uses various float conventions; for these low-level ops
            # we default to [0, 1] unless callers pass explicit bounds.
            return 0.0, 1.0
        raise TypeError(f"Unsupported dtype for invert/solarize: {dtype}")

    # MLX dtype path
    if dtype == mx.uint8:
        return 0.0, 255.0
    if dtype == mx.uint16:
        return 0.0, 65535.0
    if dtype == mx.uint32:
        return 0.0, float(2**32 - 1)
    if dtype == mx.int8:
        return float(-128), float(127)
    if dtype == mx.int16:
        return float(-32768), float(32767)
    if dtype == mx.int32:
        return float(-(2**31)), float(2**31 - 1)
    if dtype in {mx.float16, mx.float32}:
        return 0.0, 1.0

    raise TypeError(f"Unsupported dtype for invert/solarize: {dtype}")


def add(image: object, value: object) -> object:
    """Add a scalar or array value to an image.

    Parameters
    ----------
    image : object
        Input image (HWC or NHWC), NumPy or MLX.
    value : object
        Scalar or array to add.

    Returns
    -------
    object
        Result image; MLX array if input is MLX, otherwise NumPy.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    if isinstance(value, np.ndarray) or is_mlx_array(value):
        value_mlx = ensure_float32(to_mlx(value))
    else:
        value_mlx = mx.array(float(value))

    result = img_mlx + value_mlx

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def multiply(image: object, value: object) -> object:
    """Multiply an image by a scalar or array value.

    Parameters
    ----------
    image : object
        Input image (HWC or NHWC), NumPy or MLX.
    value : object
        Scalar or array multiplier.

    Returns
    -------
    object
        Result image; MLX array if input is MLX, otherwise NumPy.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    if isinstance(value, np.ndarray) or is_mlx_array(value):
        value_mlx = ensure_float32(to_mlx(value))
    else:
        value_mlx = mx.array(float(value))

    result = img_mlx * value_mlx

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def linear_contrast(image: object, factor: float) -> object:
    """Adjust linear contrast around the midpoint.

    Parameters
    ----------
    image : object
        Input image (HWC or NHWC), NumPy or MLX.
    factor : float
        Contrast factor (1.0 keeps input unchanged).

    Returns
    -------
    object
        Result image; MLX array if input is MLX, otherwise NumPy.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    midpoint = 128.0
    if isinstance(factor, (int, float)):
        factor = mx.array(float(factor))
    elif not is_mlx_array(factor):
        factor = to_mlx(factor)

    result = (img_mlx - midpoint) * factor + midpoint

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def invert(
    image: object,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> object:
    """Invert pixel values across a specified value range.

    Inverts pixel intensities by computing (min_value + max_value) - image.
    This creates a photographic negative effect.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array in HWC or NHWC format.
    min_value : float, optional
        Minimum value of the range. If None, inferred from dtype.
    max_value : float, optional
        Maximum value of the range. If None, inferred from dtype.

    Returns
    -------
    object
        Inverted image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    Notes
    -----
    This is a low-level operation for MLX pipelines. It does not implement
    full augmenter semantics (e.g., per-channel probabilities). For uint8 images,
    default range is [0, 255]. For float images, default range is [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> img = np.array([[0, 128, 255]], dtype=np.uint8)
    >>> invert(img)
    array([[255, 127, 0]], dtype=uint8)
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)

    if min_value is None or max_value is None:
        min_d, max_d = _infer_min_max_for_invert(
            original_dtype if original_dtype is not None else img_mlx.dtype
        )
        min_value = min_d if min_value is None else float(min_value)
        max_value = max_d if max_value is None else float(max_value)

    result = (mx.array(float(min_value)) + mx.array(float(max_value))) - ensure_float32(img_mlx)

    if is_input_mlx:
        # Preserve integer dtypes if possible.
        if img_mlx.dtype in {mx.uint8, mx.uint16, mx.uint32, mx.int8, mx.int16, mx.int32}:
            result = mx.clip(result, min_value, max_value).astype(img_mlx.dtype)
        return result

    result_np = to_numpy(result)
    if original_dtype is not None and original_dtype.kind in {"u", "i"}:
        result_np = np.clip(result_np, min_value, max_value).astype(original_dtype)
    return result_np


def solarize(
    image: object,
    *,
    threshold: float | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> object:
    """Apply solarization effect by inverting pixels above a threshold.

    Solarization inverts pixel values that exceed the threshold while leaving
    other pixels unchanged. This creates a surreal, high-contrast effect.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array in HWC or NHWC format.
    threshold : float, optional
        Threshold value for solarization. Pixels >= threshold are inverted.
        If None, defaults to 128 for uint8 images or 0.5 for float images.
    min_value : float, optional
        Minimum value of the range. If None, inferred from dtype.
    max_value : float, optional
        Maximum value of the range. If None, inferred from dtype.

    Returns
    -------
    object
        Solarized image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    Notes
    -----
    The operation is: result = invert(image) if image >= threshold else image.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.array([[50, 128, 200]], dtype=np.uint8)
    >>> solarize(img, threshold=128)
    array([[ 50,  127,  55]], dtype=uint8)
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)

    if min_value is None or max_value is None:
        min_d, max_d = _infer_min_max_for_invert(
            original_dtype if original_dtype is not None else img_mlx.dtype
        )
        min_value = min_d if min_value is None else float(min_value)
        max_value = max_d if max_value is None else float(max_value)

    if threshold is None:
        # common defaults
        threshold = 128.0 if max_value >= 255.0 else 0.5

    x = ensure_float32(img_mlx)
    inv = (mx.array(float(min_value)) + mx.array(float(max_value))) - x
    result = mx.where(x >= float(threshold), inv, x)

    if is_input_mlx:
        if img_mlx.dtype in {mx.uint8, mx.uint16, mx.uint32, mx.int8, mx.int16, mx.int32}:
            result = mx.clip(result, min_value, max_value).astype(img_mlx.dtype)
        return result

    result_np = to_numpy(result)
    if original_dtype is not None and original_dtype.kind in {"u", "i"}:
        result_np = np.clip(result_np, min_value, max_value).astype(original_dtype)
    return result_np


def gamma_contrast(
    image: object,
    gamma: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> object:
    """Apply gamma correction for contrast adjustment.

    Gamma correction adjusts image contrast by applying a power-law transformation.
    Values < 1 increase brightness (especially in dark regions), while values > 1
    decrease brightness.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array in HWC or NHWC format.
    gamma : float
        Gamma correction factor.
        - gamma < 1: Brighten image, expand dark tones
        - gamma = 1: No change
        - gamma > 1: Darken image, compress dark tones
    min_value : float, optional
        Minimum value of the range. If None, inferred from dtype.
    max_value : float, optional
        Maximum value of the range. If None, inferred from dtype.

    Returns
    -------
    object
        Gamma-corrected image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If gamma <= 0.

    Notes
    -----
    The transformation formula is:
        out = min + ((x - min) / (max - min)) ** gamma * (max - min)

    For uint8 inputs, this matches typical gamma correction semantics.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.array([[0, 64, 128, 192, 255]], dtype=np.uint8)
    >>> gamma_contrast(img, gamma=0.5)  # Brighten
    >>> gamma_contrast(img, gamma=2.0)  # Darken
    """
    require()

    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma!r}")

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)

    if min_value is None or max_value is None:
        min_d, max_d = _infer_min_max_for_invert(
            original_dtype if original_dtype is not None else img_mlx.dtype
        )
        min_value = min_d if min_value is None else float(min_value)
        max_value = max_d if max_value is None else float(max_value)

    if max_value == min_value:
        return image

    x = ensure_float32(img_mlx)
    rng = float(max_value - min_value)

    if isinstance(gamma, (int, float)):
        gamma_arr = mx.array(float(gamma))
    else:
        gamma_arr = to_mlx(gamma) if not is_mlx_array(gamma) else gamma

    x01 = mx.clip((x - float(min_value)) / rng, 0.0, 1.0)
    y = float(min_value) + mx.power(x01, gamma_arr) * rng

    if is_input_mlx:
        if img_mlx.dtype in {mx.uint8, mx.uint16, mx.uint32, mx.int8, mx.int16, mx.int32}:
            y = mx.clip(y, min_value, max_value).astype(img_mlx.dtype)
        return y

    y_np = to_numpy(y)
    if original_dtype is not None and original_dtype.kind in {"u", "i"}:
        y_np = np.clip(y_np, min_value, max_value).astype(original_dtype)
    return y_np


def sigmoid_contrast(
    image: object,
    gain: float,
    cutoff: float,
    inv: float | bool = False,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> object:
    """Apply sigmoid contrast adjustment.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array.
    gain : float
        Multiplier for the sigmoid function's output.
    cutoff : float
        Cutoff that shifts the sigmoid function in horizontal direction.
    inv : float
        Whether to invert the sigmoid correction.
    min_value : float, optional
        Minimum value of the range.
    max_value : float, optional
        Maximum value of the range.

    Returns
    -------
    object
        Contrast-adjusted image.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype
    img_mlx = to_mlx(image)

    if min_value is None or max_value is None:
        min_d, max_d = _infer_min_max_for_invert(
            original_dtype if original_dtype is not None else img_mlx.dtype
        )
        min_value = min_d if min_value is None else float(min_value)
        max_value = max_d if max_value is None else float(max_value)

    if max_value == min_value:
        return image

    x = ensure_float32(img_mlx)
    rng = float(max_value - min_value)

    if isinstance(gain, (int, float)):
        gain_arr = mx.array(float(gain))
    else:
        gain_arr = to_mlx(gain) if not is_mlx_array(gain) else gain

    if isinstance(cutoff, (int, float)):
        cutoff_arr = mx.array(float(cutoff))
    else:
        cutoff_arr = to_mlx(cutoff) if not is_mlx_array(cutoff) else cutoff

    x01 = (x - float(min_value)) / rng
    exp_term = mx.exp(gain_arr * (cutoff_arr - x01))
    sig = 1 / (1 + exp_term)

    inv_is_scalar = isinstance(inv, (int, float, bool, np.generic))
    if inv_is_scalar:
        inv_bool = float(inv) > 0.5
        if inv_bool:
            y = float(min_value) + rng * (1 - sig)
        else:
            y = float(min_value) + rng * sig
    else:
        inv_arr = to_mlx(inv) if not is_mlx_array(inv) else inv
        y = float(min_value) + rng * sig
        y_inv = float(min_value) + rng * (1 - sig)
        y = mx.where(inv_arr > 0.5, y_inv, y)

    if is_input_mlx:
        if img_mlx.dtype in {mx.uint8, mx.uint16, mx.uint32, mx.int8, mx.int16, mx.int32}:
            y = mx.clip(y, min_value, max_value).astype(img_mlx.dtype)
        return y

    y_np = to_numpy(y)
    if original_dtype is not None and original_dtype.kind in {"u", "i"}:
        y_np = np.clip(y_np, min_value, max_value).astype(original_dtype)
    return y_np


def log_contrast(
    image: object,
    gain: float,
    inv: float | bool = False,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> object:
    """Apply logarithmic contrast adjustment.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array.
    gain : float
        Multiplier for the logarithm result.
    inv : float
        Whether to invert the logarithmic correction.
    min_value : float, optional
        Minimum value of the range.
    max_value : float, optional
        Maximum value of the range.

    Returns
    -------
    object
        Contrast-adjusted image.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype
    img_mlx = to_mlx(image)

    if min_value is None or max_value is None:
        min_d, max_d = _infer_min_max_for_invert(
            original_dtype if original_dtype is not None else img_mlx.dtype
        )
        min_value = min_d if min_value is None else float(min_value)
        max_value = max_d if max_value is None else float(max_value)

    if max_value == min_value:
        return image

    x = ensure_float32(img_mlx)
    rng = float(max_value - min_value)

    if isinstance(gain, (int, float)):
        gain_arr = mx.array(float(gain))
    else:
        gain_arr = to_mlx(gain) if not is_mlx_array(gain) else gain

    x01 = (x - float(min_value)) / rng
    # Clip to ensure valid log input, though x01 should be >= 0 naturally if x >= min
    x01 = mx.maximum(x01, 0.0)

    inv_is_scalar = isinstance(inv, (int, float, bool, np.generic))
    if inv_is_scalar:
        inv_bool = float(inv) > 0.5
        if inv_bool:
            y = float(min_value) + rng * gain_arr * (2.0**x01 - 1.0)
        else:
            y = float(min_value) + rng * gain_arr * mx.log2(1.0 + x01)
    else:
        inv_arr = to_mlx(inv) if not is_mlx_array(inv) else inv
        y = float(min_value) + rng * gain_arr * mx.log2(1.0 + x01)
        y_inv = float(min_value) + rng * gain_arr * (2.0**x01 - 1.0)
        y = mx.where(inv_arr > 0.5, y_inv, y)

    if is_input_mlx:
        if img_mlx.dtype in {mx.uint8, mx.uint16, mx.uint32, mx.int8, mx.int16, mx.int32}:
            y = mx.clip(y, min_value, max_value).astype(img_mlx.dtype)
        return y

    y_np = to_numpy(y)
    if original_dtype is not None and original_dtype.kind in {"u", "i"}:
        y_np = np.clip(y_np, min_value, max_value).astype(original_dtype)
    return y_np


__all__ = [
    "add",
    "multiply",
    "linear_contrast",
    "invert",
    "solarize",
    "gamma_contrast",
    "sigmoid_contrast",
    "log_contrast",
]
