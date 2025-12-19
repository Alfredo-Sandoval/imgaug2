"""Flip and rotation operations for MLX-accelerated image augmentation.

This module provides hardware-accelerated flip and rotation operations using
Apple's MLX framework. Functions operate on images in HWC (Height, Width, Channel)
or NHWC (Batch, Height, Width, Channel) format and preserve the input array type
(NumPy or MLX).

Examples
--------
>>> import numpy as np
>>> from imgaug2.mlx.flip import fliplr, flipud, rot90
>>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
>>> flipped = fliplr(img)  # Returns numpy array
>>> rotated = rot90(img, k=2)  # Rotate 180 degrees
"""

from __future__ import annotations

from ._core import is_mlx_array, mx, require, to_mlx, to_numpy


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
    >>> fliplr(img)
    array([[ 3,  2,  1,  0],
           [ 7,  6,  5,  4],
           [11, 10,  9,  8]])
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
    >>> flipud(img)
    array([[ 8,  9, 10, 11],
           [ 4,  5,  6,  7],
           [ 0,  1,  2,  3]])
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
    >>> rot90(img, k=1)  # 90 degrees counter-clockwise
    array([[2, 5],
           [1, 4],
           [0, 3]])
    >>> rot90(img, k=2)  # 180 degrees
    array([[5, 4, 3],
           [2, 1, 0]])
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


__all__ = ["fliplr", "flipud", "rot90"]
