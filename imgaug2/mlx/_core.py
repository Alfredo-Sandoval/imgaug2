"""
Core utilities for the MLX backend.

This module provides the foundational utilities for MLX-accelerated image
augmentation on Apple Silicon. All MLX operations in this package depend
on these helpers for array conversion, dtype handling, and availability checks.

Device Semantics
----------------
MLX arrays live on the unified memory architecture of Apple Silicon GPUs.
Conversions between NumPy and MLX are zero-copy when possible but may
trigger synchronization.

Notes
-----
- MLX requires Apple Silicon (M1/M2/M3/M4) and macOS 13.3+.
- Install via ``pip install mlx``.

Examples
--------
>>> from imgaug2.mlx import is_available, to_mlx
>>> if is_available():
...     arr = to_mlx(np.zeros((32, 32, 3)))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard, cast

import numpy as np

from imgaug2.errors import BackendUnavailableError, DependencyMissingError

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
    _MLX_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    MLX_AVAILABLE = False
    mx = None
    _MLX_IMPORT_ERROR = exc


if TYPE_CHECKING:
    try:
        from mlx.core import array as MxArray
    except ImportError:
        MxArray: TypeAlias = Any
else:
    MxArray: TypeAlias = Any


def is_available() -> bool:
    """
    Check if the MLX backend is available and functional.

    Returns
    -------
    bool
        True if MLX is installed and can allocate arrays, False otherwise.

    Examples
    --------
    >>> from imgaug2.mlx import is_available
    >>> if is_available():
    ...     print("MLX ready")
    """
    if not MLX_AVAILABLE or mx is None:
        return False
    try:
        _ = mx.zeros((1,), dtype=mx.float32)
    except Exception:
        return False
    return True


def require() -> None:
    """
    Raise an error if MLX is not available.

    Raises
    ------
    DependencyMissingError
        If MLX is not installed.
    BackendUnavailableError
        If MLX is installed but not functional on this system.
    """
    if not MLX_AVAILABLE or mx is None:
        raise DependencyMissingError(
            "MLX backend is not available because `mlx` could not be imported. "
            "Install it on Apple Silicon, e.g. `pip install mlx`."
        ) from _MLX_IMPORT_ERROR

    if not is_available():
        raise BackendUnavailableError(
            "MLX backend is installed but not functional on this system. "
            "Ensure you're running on Apple Silicon with macOS 13.3+ and a working `mlx` install."
        )


def is_mlx_array(arr: object) -> TypeGuard[MxArray]:
    """
    Check if an object is an MLX array.

    Parameters
    ----------
    arr : object
        Any Python object.

    Returns
    -------
    bool
        True if ``arr`` is an ``mlx.core.array``, False otherwise.
    """
    if not MLX_AVAILABLE or mx is None:
        return False
    return isinstance(arr, mx.array)


def to_mlx(arr: object) -> MxArray:
    """
    Convert an array-like object to an MLX array.

    Parameters
    ----------
    arr : array-like
        NumPy array, MLX array, or any object convertible via ``mx.array()``.

    Returns
    -------
    MxArray
        MLX array on the default device.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()
    if is_mlx_array(arr):
        return cast(MxArray, arr)
    if isinstance(arr, np.ndarray):
        return mx.array(arr)
    return mx.array(arr)


def to_numpy(arr: object) -> np.ndarray:
    """
    Convert an array-like object to a NumPy array.

    Parameters
    ----------
    arr : array-like
        NumPy array, MLX array, or any object convertible via ``np.array()``.

    Returns
    -------
    np.ndarray
        NumPy array. If input was MLX, triggers device synchronization.
    """
    if isinstance(arr, np.ndarray):
        return arr
    if MLX_AVAILABLE and is_mlx_array(arr):
        return np.array(arr)
    return np.array(arr)


def ensure_float32(arr: MxArray) -> MxArray:
    """
    Ensure an MLX array has dtype float32.

    Parameters
    ----------
    arr : MxArray
        Input MLX array.

    Returns
    -------
    MxArray
        Array with dtype ``mx.float32``. Returns input unchanged if already float32.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()
    if arr.dtype == mx.float32:
        return arr
    return arr.astype(mx.float32)


def ensure_dtype(dtype: np.dtype | None) -> np.dtype:
    """
    Type-narrow dtype, raising if unexpectedly None.

    This is an internal helper for code paths where we've already
    checked `is_input_mlx` and returned early if True. When we reach
    the numpy return path, `original_dtype` must be set.

    Parameters
    ----------
    dtype : np.dtype | None
        The original dtype, or None if input was MLX.

    Returns
    -------
    np.dtype
        The dtype, guaranteed non-None.

    Raises
    ------
    RuntimeError
        If dtype is None (internal logic error).
    """
    if dtype is None:
        raise RuntimeError("Internal error: dtype expected to be set for numpy input path")
    return dtype


def restore_dtype(
    result: MxArray,
    original_dtype: np.dtype | None,
    is_input_mlx: bool,
    *,
    clip_uint8: bool = True,
) -> np.ndarray | MxArray:
    """
    Restore the original dtype and array type after an MLX operation.

    This helper ensures consistent output semantics: if the input was an MLX
    array, the output stays MLX; if the input was NumPy, the output is NumPy
    with the original dtype restored.

    Parameters
    ----------
    result : MxArray
        The MLX array result from an operation.
    original_dtype : np.dtype or None
        The dtype of the original NumPy input, or None if input was MLX.
    is_input_mlx : bool
        Whether the original input was an MLX array.
    clip_uint8 : bool, default True
        If True and ``original_dtype`` is uint8, clip values to [0, 255]
        before casting. Prevents overflow artifacts.

    Returns
    -------
    np.ndarray or MxArray
        MLX array if input was MLX, otherwise NumPy array with original dtype.
    """
    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        if clip_uint8:
            return np.clip(result_np, 0, 255).astype(np.uint8)
        return result_np.astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


__all__ = [
    "MLX_AVAILABLE",
    "mx",
    "ensure_float32",
    "is_available",
    "is_mlx_array",
    "require",
    "restore_dtype",
    "to_mlx",
    "to_numpy",
]
