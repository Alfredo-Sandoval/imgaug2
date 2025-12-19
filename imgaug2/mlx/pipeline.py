"""Pipeline utilities for efficient MLX-accelerated image processing.

This module provides utilities for building efficient MLX pipelines by minimizing
host-device data transfers. The key principle is to convert to MLX once, perform
multiple operations on-device, then convert back once.

The main functions are:
- `to_device`: Convert NumPy to MLX with optional dtype conversion
- `to_host`: Convert MLX to NumPy with dtype restoration
- `chain`: Execute multiple MLX operations in sequence without host roundtrips

Examples
--------
>>> import numpy as np
>>> from imgaug2.mlx.pipeline import chain, to_device, to_host
>>> from imgaug2.mlx.blur import gaussian_blur
>>> from imgaug2.mlx.pointwise import multiply
>>>
>>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
>>>
>>> # Efficient: single host-device roundtrip
>>> result = chain(
...     img,
...     lambda x: gaussian_blur(x, sigma=2.0),
...     lambda x: multiply(x, 1.2),
... )
>>>
>>> # Less efficient: multiple roundtrips
>>> result = gaussian_blur(img, sigma=2.0)
>>> result = multiply(result, 1.2)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._core import ensure_float32, is_mlx_array, mx, require, to_mlx, to_numpy

_DEFAULT_DEVICE_DTYPE = object()


def to_device(image: object, *, dtype: object = _DEFAULT_DEVICE_DTYPE) -> mx.array:
    """Convert an image to MLX array for GPU/accelerator processing.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array.
    dtype : dtype or None, optional
        Target dtype for MLX array. Special values:
        - (default): Convert to float32 (recommended for most operations)
        - None: Keep original dtype
        - mx.float32, mx.float16, etc.: Convert to specified dtype

    Returns
    -------
    mx.array
        Image as MLX array on device.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    See Also
    --------
    to_host : Convert MLX array back to NumPy
    chain : Execute multiple operations on-device

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    >>> img_mlx = to_device(img)  # Convert to float32 MLX array
    >>> img_mlx_u8 = to_device(img, dtype=None)  # Keep uint8
    """
    require()
    x = to_mlx(image)
    if dtype is _DEFAULT_DEVICE_DTYPE:
        return ensure_float32(x)
    if dtype is None:
        return x
    if dtype == mx.float32:
        return ensure_float32(x)
    if x.dtype != dtype:
        return x.astype(dtype)
    return x


def to_host(
    image: object,
    *,
    dtype: np.dtype | None = None,
    clip: bool = True,
    round: bool = True,
) -> np.ndarray:
    """Convert an MLX array to NumPy array for CPU/host processing.

    Parameters
    ----------
    image : object
        Input image as MLX array or NumPy array.
    dtype : np.dtype, optional
        Target NumPy dtype. If None, no dtype conversion is performed.
    clip : bool, optional
        If True, clip values to valid range for target dtype. Default is True.
    round : bool, optional
        If True, round float values when converting to integer dtypes. Default is True.

    Returns
    -------
    np.ndarray
        Image as NumPy array on host.

    See Also
    --------
    to_device : Convert NumPy array to MLX
    chain : Execute multiple operations on-device

    Notes
    -----
    Uses imgaug2 dtype conversion semantics for proper handling of value ranges
    and rounding when converting between dtypes.

    Examples
    --------
    >>> import mlx.core as mx
    >>> img_mlx = mx.random.uniform(0, 1, (100, 100, 3))
    >>> img_np = to_host(img_mlx, dtype=np.uint8)  # Convert to uint8
    """
    arr = to_numpy(image)
    if dtype is None:
        return arr

    # Use imgaug2's dtype restoration semantics.
    import imgaug2.dtypes as iadt

    return iadt.change_dtype_(arr, np.dtype(dtype), clip=clip, round=round)


def chain(
    image: object,
    *ops: Callable[[mx.array], mx.array],
    device_dtype: object = _DEFAULT_DEVICE_DTYPE,
    output_dtype: np.dtype | None = None,
) -> object:
    """Execute a sequence of MLX operations efficiently on-device.

    This is the recommended way to use MLX operations: convert to MLX once,
    run multiple operations on-device, then convert back once. This minimizes
    expensive host-device memory transfers.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array.
    *ops : callable
        Variable number of callables that each take and return an mx.array.
        These operations are executed in sequence on the MLX device.
    device_dtype : dtype or None, optional
        Dtype to use on device. Default converts to float32.
    output_dtype : np.dtype, optional
        Output dtype. Behavior depends on input type:
        - If input is NumPy: defaults to input dtype
        - If input is MLX: defaults to returning MLX array (no conversion)

    Returns
    -------
    object
        Processed image. Returns MLX array if input was MLX and output_dtype is None,
        otherwise returns NumPy array.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    See Also
    --------
    to_device : Convert to MLX array
    to_host : Convert to NumPy array

    Examples
    --------
    >>> import numpy as np
    >>> from imgaug2.mlx.pipeline import chain
    >>> from imgaug2.mlx.blur import gaussian_blur
    >>> from imgaug2.mlx.pointwise import multiply, add
    >>>
    >>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    >>>
    >>> # Chain multiple operations efficiently
    >>> result = chain(
    ...     img,
    ...     lambda x: gaussian_blur(x, sigma=1.5),
    ...     lambda x: multiply(x, 1.1),
    ...     lambda x: add(x, 10),
    ... )
    >>>
    >>> # Equivalent but less efficient (3 host-device roundtrips)
    >>> temp1 = gaussian_blur(img, sigma=1.5)
    >>> temp2 = multiply(temp1, 1.1)
    >>> result = add(temp2, 10)

    Notes
    -----
    This function provides the "fast path" for MLX operations by minimizing
    data transfers. For a single operation, the overhead of chaining is minimal,
    but for multiple operations the performance benefit is significant.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    inferred_output_dtype = None if is_input_mlx else to_numpy(image).dtype

    x = to_device(image, dtype=device_dtype)
    for op in ops:
        x = op(x)

    if is_input_mlx and output_dtype is None:
        return x

    return to_host(x, dtype=(output_dtype or inferred_output_dtype))


__all__ = ["chain", "to_device", "to_host"]
