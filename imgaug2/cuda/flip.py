"""GPU-accelerated flip and rotation operations using CUDA/CuPy.

This module provides GPU-accelerated implementations of geometric flip and
rotation operations, optimized for batch processing on CUDA-compatible devices.
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


def fliplr(image: cp.ndarray) -> cp.ndarray:
    """Flip image left-right (horizontal mirror) on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of shape (H, W), (H, W, C), or (N, H, W, C).

    Returns
    -------
    cupy.ndarray
        Horizontally flipped image with same shape and dtype as input.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.
    ValueError
        If image has invalid shape (not 2D/3D/4D).
    """
    img = _require_cupy_array(image, "fliplr")

    if img.ndim == 2:
        return img[:, ::-1]
    if img.ndim == 3:
        return img[:, ::-1, :]
    if img.ndim == 4:
        return img[:, :, ::-1, :]
    raise ValueError(f"fliplr expects 2D/3D/4D image, got shape {img.shape}")


def flipud(image: cp.ndarray) -> cp.ndarray:
    """Flip image up-down (vertical mirror) on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of shape (H, W), (H, W, C), or (N, H, W, C).

    Returns
    -------
    cupy.ndarray
        Vertically flipped image with same shape and dtype as input.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.
    ValueError
        If image has invalid shape (not 2D/3D/4D).
    """
    img = _require_cupy_array(image, "flipud")

    if img.ndim == 2:
        return img[::-1, :]
    if img.ndim == 3:
        return img[::-1, :, :]
    if img.ndim == 4:
        return img[:, ::-1, :, :]
    raise ValueError(f"flipud expects 2D/3D/4D image, got shape {img.shape}")


def rot90(image: cp.ndarray, k: int = 1) -> cp.ndarray:
    """Rotate image by 90 degrees counter-clockwise on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of shape (H, W), (H, W, C), or (N, H, W, C).
    k : int, optional
        Number of 90-degree rotations. Normalized via modulo 4. Default is 1.

    Returns
    -------
    cupy.ndarray
        Rotated image with potentially transposed spatial dimensions.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.
    ValueError
        If image has invalid shape (not 2D/3D/4D).
    """
    img = _require_cupy_array(image, "rot90")
    k = int(k) % 4
    if k == 0:
        return img

    if img.ndim == 2:
        if k == 1:
            return cp.transpose(img, (1, 0))[::-1, :]
        if k == 2:
            return img[::-1, ::-1]
        return cp.transpose(img, (1, 0))[:, ::-1]

    if img.ndim == 3:
        if k == 1:
            return cp.transpose(img, (1, 0, 2))[::-1, :, :]
        if k == 2:
            return img[::-1, ::-1, :]
        return cp.transpose(img, (1, 0, 2))[:, ::-1, :]

    if img.ndim == 4:
        if k == 1:
            return cp.transpose(img, (0, 2, 1, 3))[:, ::-1, :, :]
        if k == 2:
            return img[:, ::-1, ::-1, :]
        return cp.transpose(img, (0, 2, 1, 3))[:, :, ::-1, :]

    raise ValueError(f"rot90 expects 2D/3D/4D image, got shape {img.shape}")


__all__ = ["fliplr", "flipud", "rot90"]
