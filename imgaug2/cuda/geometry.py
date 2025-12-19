"""GPU-accelerated geometric transformations using CUDA/CuPy.

This module provides GPU-accelerated implementations of geometric transformations
including affine warping, optimized for batch processing on CUDA-compatible devices.
"""
from __future__ import annotations

import importlib
from typing import Literal

import numpy as np

from imgaug2.errors import DependencyMissingError
from ._core import cp, is_cupy_array, require

_Mode = Literal["constant", "edge", "symmetric", "reflect", "wrap"]


def _require_cupy_array(x: object, fn_name: str) -> cp.ndarray:
    require()
    if not is_cupy_array(x):
        raise TypeError(
            f"{fn_name} expects a cupy.ndarray (GPU resident). "
            "Convert first via `cupy.asarray(...)`."
        )
    return x


def _mode_to_cupyx_ndimage(mode: str) -> str:
    """Map imgaug2 border mode to cupyx.scipy.ndimage mode."""
    mode = str(mode).lower()
    if mode in {"constant", "zeros"}:
        return "constant"
    if mode in {"edge", "nearest"}:
        return "nearest"
    if mode == "symmetric":
        return "reflect"
    if mode == "reflect":
        return "mirror"
    if mode == "wrap":
        return "wrap"
    raise ValueError(f"Unsupported mode={mode!r} for CUDA affine.")


def _dtype_restore(y: cp.ndarray, dtype: object) -> cp.ndarray:
    if cp.issubdtype(dtype, cp.integer):
        info = cp.iinfo(dtype)
        y = cp.floor(y + 0.5)
        y = cp.clip(y, float(info.min), float(info.max))
        return y.astype(dtype)
    if cp.issubdtype(dtype, cp.bool_):
        return y > 0.5
    return y.astype(dtype, copy=False)


def affine_transform(
    image: cp.ndarray,
    matrix: np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    order: int = 1,
    cval: float = 0.0,
    mode: _Mode = "constant",
) -> cp.ndarray:
    """Apply affine transformation to an image on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of shape (H, W), (H, W, C), or (N, H, W, C).
    matrix : numpy.ndarray
        Affine transformation matrix of shape (2, 3) or (3, 3) in (x, y) pixel
        coordinates. Internally inverted for inverse mapping.
    output_shape : tuple of int or None, optional
        Output image shape as (height, width). If None, uses input shape. Default is None.
    order : int, optional
        Interpolation order: 0 (nearest) or 1 (bilinear). Default is 1.
    cval : float, optional
        Fill value for constant mode. Default is 0.0.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Border extrapolation mode. Default is 'constant'.

    Returns
    -------
    cupy.ndarray
        Transformed image with same dtype as input.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.
    ValueError
        If matrix shape is invalid, order is not 0/1, or image shape is invalid.
    ImportError
        If cupyx.scipy.ndimage is not available.
    """
    img = _require_cupy_array(image, "affine_transform")
    dtype = img.dtype
    if img.size == 0:
        return img

    if int(order) not in (0, 1):
        raise ValueError(f"Only order 0/1 supported for CUDA affine, got {order}")

    if img.ndim == 2:
        h_in, w_in = int(img.shape[0]), int(img.shape[1])
        c_in = None
        n_in = None
    elif img.ndim == 3:
        h_in, w_in, c_in = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
        n_in = None
    elif img.ndim == 4:
        n_in, h_in, w_in, c_in = (
            int(img.shape[0]),
            int(img.shape[1]),
            int(img.shape[2]),
            int(img.shape[3]),
        )
    else:
        raise ValueError(f"affine_transform expects 2D/3D/4D image, got shape {img.shape}")

    h_out, w_out = output_shape or (h_in, w_in)

    mat = np.asarray(matrix, dtype=np.float32)
    if mat.shape == (2, 3):
        mat3 = np.eye(3, dtype=np.float32)
        mat3[:2, :] = mat
        mat = mat3
    elif mat.shape != (3, 3):
        raise ValueError(f"Expected matrix shape (2,3) or (3,3), got {mat.shape}.")

    inv = np.linalg.inv(mat).astype(np.float32)
    a, b, c = inv[0]
    d, e, f = inv[1]

    if img.ndim == 2:
        matrix_nd = cp.asarray(np.array([[e, d], [b, a]], dtype=np.float32))
        offset_nd = cp.asarray(np.array([f, c], dtype=np.float32))
        out_shape = (int(h_out), int(w_out))
    elif img.ndim == 3:
        matrix_nd = cp.asarray(
            np.array([[e, d, 0.0], [b, a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        )
        offset_nd = cp.asarray(np.array([f, c, 0.0], dtype=np.float32))
        out_shape = (int(h_out), int(w_out), int(c_in))
    else:
        matrix_nd = cp.asarray(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, e, d, 0.0],
                    [0.0, b, a, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
        offset_nd = cp.asarray(np.array([0.0, f, c, 0.0], dtype=np.float32))
        out_shape = (int(n_in), int(h_out), int(w_out), int(c_in))

    try:
        cndimage = importlib.import_module("cupyx.scipy.ndimage")
    except Exception as exc:  # pragma: no cover
        raise DependencyMissingError(
            "cupyx.scipy.ndimage is required for affine_transform. "
            "Install a CuPy build that includes cupyx."
        ) from exc

    nd_mode = _mode_to_cupyx_ndimage(mode)
    x = img.astype(cp.float32, copy=False)
    y = cndimage.affine_transform(
        x,
        matrix=matrix_nd,
        offset=offset_nd,
        output_shape=out_shape,
        order=int(order),
        mode=nd_mode,
        cval=float(cval),
        prefilter=False,
    )
    return _dtype_restore(y, dtype)


def perspective_transform(*_args: object, **_kwargs: object) -> np.ndarray:
    """Placeholder for perspective transform (not implemented on CUDA).

    Raises
    ------
    NotImplementedError
        This operation is not yet implemented for CUDA backend.
    """
    raise NotImplementedError("CUDA perspective transform not implemented yet")


def elastic_transform(*_args: object, **_kwargs: object) -> np.ndarray:
    """Placeholder for elastic transform (not implemented on CUDA).

    Raises
    ------
    NotImplementedError
        This operation is not yet implemented for CUDA backend.
    """
    raise NotImplementedError("CUDA elastic transform not implemented yet")


def piecewise_affine(*_args: object, **_kwargs: object) -> np.ndarray:
    """Placeholder for piecewise affine transform (not implemented on CUDA).

    Raises
    ------
    NotImplementedError
        This operation is not yet implemented for CUDA backend.
    """
    raise NotImplementedError("CUDA piecewise affine not implemented yet")


__all__ = [
    "affine_transform",
    "elastic_transform",
    "perspective_transform",
    "piecewise_affine",
]
