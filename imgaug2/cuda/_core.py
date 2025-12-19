"""Core utilities and array conversion functions for the CUDA backend.

This module provides low-level utilities for working with CuPy arrays,
including array type checking, CPU-GPU conversion, and backend availability
detection.
"""
from __future__ import annotations

import importlib
from typing import Any, TypeGuard

import numpy as np

from imgaug2.errors import BackendUnavailableError, DependencyMissingError

try:
    # CuPy is an optional dependency; we import it dynamically so type checking
    # doesn't fail on platforms without it (e.g. macOS / Apple Silicon).
    cp: Any = importlib.import_module("cupy")
    CUDA_AVAILABLE = True
    _CUDA_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    CUDA_AVAILABLE = False
    cp = None
    _CUDA_IMPORT_ERROR = exc


def is_available() -> bool:
    """Check whether the CUDA backend is available and usable.

    Returns
    -------
    bool
        True if CuPy is installed and a CUDA device is accessible, False otherwise.
    """
    if not CUDA_AVAILABLE or cp is None:
        return False
    try:
        _ = cp.cuda.Device(0).compute_capability
    except Exception:
        return False
    return True


def require() -> None:
    """Raise an error if the CUDA backend is unavailable.

    Raises
    ------
    DependencyMissingError
        If CuPy is not installed.
    BackendUnavailableError
        If CuPy is installed but no CUDA device is accessible.
    """
    if not CUDA_AVAILABLE or cp is None:
        raise DependencyMissingError(
            "CUDA backend is not available because `cupy` could not be imported. "
            "Install a CuPy build matching your CUDA runtime, e.g. `pip install cupy-cuda12x`."
        ) from _CUDA_IMPORT_ERROR

    try:
        _ = cp.cuda.Device(0).compute_capability
    except Exception as exc:
        raise BackendUnavailableError(
            "CUDA backend is installed but no CUDA device is accessible."
        ) from exc


def is_cupy_array(arr: object) -> TypeGuard[cp.ndarray]:
    """Check if an object is a CuPy array.

    Parameters
    ----------
    arr : object
        Object to check.

    Returns
    -------
    bool
        True if ``arr`` is a ``cupy.ndarray`` instance, False otherwise.
    """
    if not CUDA_AVAILABLE or cp is None:
        return False
    return isinstance(arr, cp.ndarray)


def to_cupy(arr: object) -> cp.ndarray:
    """Convert input to a CuPy array on GPU.

    Parameters
    ----------
    arr : object
        Input array (NumPy array, CuPy array, or array-like).

    Returns
    -------
    cupy.ndarray
        GPU-resident CuPy array.

    Raises
    ------
    RuntimeError
        If CUDA backend is not available.
    """
    require()
    if is_cupy_array(arr):
        return arr
    if isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return cp.array(arr)


def to_numpy(arr: object) -> np.ndarray:
    """Convert input to a NumPy array, copying from GPU if needed.

    Parameters
    ----------
    arr : object
        Input array (CuPy array, NumPy array, or array-like).

    Returns
    -------
    numpy.ndarray
        CPU-resident NumPy array.
    """
    if isinstance(arr, np.ndarray):
        return arr
    if CUDA_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.array(arr)


__all__ = [
    "CUDA_AVAILABLE",
    "cp",
    "is_available",
    "is_cupy_array",
    "require",
    "to_cupy",
    "to_numpy",
]
