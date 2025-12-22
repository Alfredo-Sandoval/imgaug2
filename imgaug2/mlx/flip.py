"""Flip and rotation operations for MLX-accelerated image augmentation.

This module provides hardware-accelerated flip and rotation operations using
Apple's MLX framework. Functions operate on images in HWC (Height, Width, Channel)
or NHWC (Batch, Height, Width, Channel) format and preserve the input array type
(NumPy or MLX).

Examples
--------
>>> # xdoctest: +SKIP
>>> import numpy as np
>>> from imgaug2.mlx.flip import fliplr, flipud, rot90
>>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
>>> flipped = fliplr(img)  # Returns numpy array
>>> rotated = rot90(img, k=2)  # Rotate 180 degrees
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

from ._core import is_mlx_array, mx, require, to_mlx, to_numpy

if TYPE_CHECKING:
    import mlx.core as _mx

    MlxArray: TypeAlias = _mx.array
else:
    MlxArray: TypeAlias = object

NumpyArray: TypeAlias = NDArray[np.generic]


def _as_points(points: mx.array) -> tuple[mx.array, bool]:
    if points.ndim == 1:
        if int(points.shape[0]) != 2:
            raise ValueError(
                "Expected points shape (2,) or (N,2); got "
                f"{tuple(points.shape)!r}."
            )
        return points[None, :], True
    if points.ndim == 2 and int(points.shape[1]) == 2:
        return points, False
    raise ValueError(f"Expected points shape (N,2); got {tuple(points.shape)!r}.")


def fliplr(image: object) -> object:
    """Flip image horizontally (left-right mirror).

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are:
        - (H, W) : Single-channel grayscale image
        - (H, W, C) : Multi-channel image
        - (N, H, W, C) : Batch of multi-channel images

    Returns
    -------
    object
        Horizontally flipped image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If input shape is not 2D, 3D, or 4D.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.arange(12).reshape(3, 4)
    >>> fliplr(img).tolist()
    [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]]
    """
    require()

    is_input_mlx = is_mlx_array(image)
    img = to_mlx(image)

    if img.ndim == 2:
        out = img[:, ::-1]
    elif img.ndim == 3:
        out = img[:, ::-1, :]
    elif img.ndim == 4:
        out = img[:, :, ::-1, :]
    else:
        raise ValueError(f"fliplr expects 2D/3D/4D image, got shape {tuple(img.shape)}")

    if is_input_mlx:
        return out
    return to_numpy(out)


def flipud(image: object) -> object:
    """Flip image vertically (up-down mirror).

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are:
        - (H, W) : Single-channel grayscale image
        - (H, W, C) : Multi-channel image
        - (N, H, W, C) : Batch of multi-channel images

    Returns
    -------
    object
        Vertically flipped image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If input shape is not 2D, 3D, or 4D.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.arange(12).reshape(3, 4)
    >>> flipud(img).tolist()
    [[8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]
    """
    require()

    is_input_mlx = is_mlx_array(image)
    img = to_mlx(image)

    if img.ndim == 2:
        out = img[::-1, :]
    elif img.ndim == 3:
        out = img[::-1, :, :]
    elif img.ndim == 4:
        out = img[:, ::-1, :, :]
    else:
        raise ValueError(f"flipud expects 2D/3D/4D image, got shape {tuple(img.shape)}")

    if is_input_mlx:
        return out
    return to_numpy(out)


def rot90(image: object, k: int = 1) -> object:
    """Rotate image by 90 degrees counter-clockwise.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are:
        - (H, W) : Single-channel grayscale image
        - (H, W, C) : Multi-channel image
        - (N, H, W, C) : Batch of multi-channel images
    k : int, optional
        Number of 90-degree rotations. Positive values rotate counter-clockwise.
        The value is taken modulo 4. Default is 1.

    Returns
    -------
    object
        Rotated image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If input shape is not 2D, 3D, or 4D.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.arange(6).reshape(2, 3)
    >>> rot90(img, k=1).tolist()  # 90 degrees counter-clockwise
    [[2, 5], [1, 4], [0, 3]]
    >>> rot90(img, k=2).tolist()  # 180 degrees
    [[5, 4, 3], [2, 1, 0]]
    """
    require()

    k = int(k) % 4
    if k == 0:
        return image

    is_input_mlx = is_mlx_array(image)
    img = to_mlx(image)

    if img.ndim == 2:
        if k == 1:
            out = mx.transpose(img, (1, 0))[::-1, :]
        elif k == 2:
            out = img[::-1, ::-1]
        else:  # k == 3
            out = mx.transpose(img, (1, 0))[:, ::-1]
    elif img.ndim == 3:
        if k == 1:
            out = mx.transpose(img, (1, 0, 2))[::-1, :, :]
        elif k == 2:
            out = img[::-1, ::-1, :]
        else:  # k == 3
            out = mx.transpose(img, (1, 0, 2))[:, ::-1, :]
    elif img.ndim == 4:
        if k == 1:
            out = mx.transpose(img, (0, 2, 1, 3))[:, ::-1, :, :]
        elif k == 2:
            out = img[:, ::-1, ::-1, :]
        else:  # k == 3
            out = mx.transpose(img, (0, 2, 1, 3))[:, :, ::-1, :]
    else:
        raise ValueError(f"rot90 expects 2D/3D/4D image, got shape {tuple(img.shape)}")

    if is_input_mlx:
        return out
    return to_numpy(out)

@overload
def fliplr_points(points: NumpyArray, shape: tuple[int, int]) -> NumpyArray: ...


@overload
def fliplr_points(points: MlxArray, shape: tuple[int, int]) -> MlxArray: ...


def fliplr_points(points: object, shape: tuple[int, int]) -> object:
    """Flip point coordinates horizontally (left-right mirror).

    Parameters
    ----------
    points : array-like
        Keypoint coordinates as ``(N,2)`` or ``(2,)`` in ``(x, y)`` order.
    shape : tuple of int
        Image shape ``(H, W)`` corresponding to the points.
    """
    require()
    is_input_mlx = is_mlx_array(points)
    pts = to_mlx(points)
    pts, squeeze = _as_points(pts)

    pts_f = pts if mx.issubdtype(pts.dtype, mx.floating) else pts.astype(mx.float32)
    width = float(shape[1])
    out = mx.stack([width - pts_f[:, 0], pts_f[:, 1]], axis=1)
    if squeeze:
        out = out[0]
    return out if is_input_mlx else to_numpy(out)


@overload
def flipud_points(points: NumpyArray, shape: tuple[int, int]) -> NumpyArray: ...


@overload
def flipud_points(points: MlxArray, shape: tuple[int, int]) -> MlxArray: ...


def flipud_points(points: object, shape: tuple[int, int]) -> object:
    """Flip point coordinates vertically (up-down mirror)."""
    require()
    is_input_mlx = is_mlx_array(points)
    pts = to_mlx(points)
    pts, squeeze = _as_points(pts)

    pts_f = pts if mx.issubdtype(pts.dtype, mx.floating) else pts.astype(mx.float32)
    height = float(shape[0])
    out = mx.stack([pts_f[:, 0], height - pts_f[:, 1]], axis=1)
    if squeeze:
        out = out[0]
    return out if is_input_mlx else to_numpy(out)


@overload
def rot90_points(points: NumpyArray, shape: tuple[int, int], k: int = 1) -> NumpyArray: ...


@overload
def rot90_points(points: MlxArray, shape: tuple[int, int], k: int = 1) -> MlxArray: ...


def rot90_points(points: object, shape: tuple[int, int], k: int = 1) -> object:
    """Rotate point coordinates by 90-degree steps counter-clockwise.

    Notes
    -----
    This matches ``imgaug2.augmenters.geometric.Rot90`` keypoint math. The
    output shape depends on ``k`` (odd ``k`` swaps height/width); callers
    should update shapes accordingly when needed.
    """
    require()
    k = int(k) % 4
    if k == 0:
        return points

    is_input_mlx = is_mlx_array(points)
    pts = to_mlx(points)
    pts, squeeze = _as_points(pts)

    pts_f = pts if mx.issubdtype(pts.dtype, mx.floating) else pts.astype(mx.float32)
    h, w = float(shape[0]), float(shape[1])
    x = pts_f[:, 0]
    y = pts_f[:, 1]
    if k == 1:
        out = mx.stack([h - y, x], axis=1)
    elif k == 2:
        out = mx.stack([w - x, h - y], axis=1)
    else:  # k == 3
        out = mx.stack([y, w - x], axis=1)

    if squeeze:
        out = out[0]
    return out if is_input_mlx else to_numpy(out)


__all__ = ["fliplr", "flipud", "rot90", "fliplr_points", "flipud_points", "rot90_points"]
