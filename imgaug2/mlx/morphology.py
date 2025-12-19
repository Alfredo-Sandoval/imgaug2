"""MLX-accelerated morphological operations for image processing.

This module provides hardware-accelerated morphological image processing operations
using Apple's MLX framework. These operations use structuring elements to probe and
modify image shapes, commonly used for noise removal, shape extraction, and image
segmentation.

Supported operations include erosion, dilation, opening, closing, and morphological
gradient. All functions preserve input array type (NumPy or MLX).

Examples
--------
>>> import numpy as np  # doctest: +SKIP
>>> from imgaug2.mlx.morphology import erosion, dilation, opening  # doctest: +SKIP
>>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)  # doctest: +SKIP
>>> eroded = erosion(img, ksize=3, shape='rect')  # doctest: +SKIP
>>> opened = opening(img, ksize=5, shape='ellipse')  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._core import ensure_float32, is_mlx_array, mx, require, to_mlx, to_numpy


def _get_structuring_element(
    shape: Literal["rect", "cross", "ellipse"],
    ksize: int,
) -> np.ndarray:
    """Create a structuring element for morphological operations.

    Internal helper that generates binary masks used as structuring elements.
    """
    if ksize < 1:
        raise ValueError(f"ksize must be >= 1, got {ksize}")

    if shape == "rect":
        return np.ones((ksize, ksize), dtype=np.float32)
    elif shape == "cross":
        elem = np.zeros((ksize, ksize), dtype=np.float32)
        center = ksize // 2
        elem[center, :] = 1.0
        elem[:, center] = 1.0
        return elem
    elif shape == "ellipse":
        elem = np.zeros((ksize, ksize), dtype=np.float32)
        center = (ksize - 1) / 2.0
        for y in range(ksize):
            for x in range(ksize):
                # Normalized distance from center
                dy = (y - center) / (center + 1e-6)
                dx = (x - center) / (center + 1e-6)
                if dy * dy + dx * dx <= 1.0:
                    elem[y, x] = 1.0
        return elem
    else:
        raise ValueError(f"Unknown shape: {shape}")


def erosion(
    image: object,
    ksize: int = 3,
    shape: Literal["rect", "cross", "ellipse"] = "rect",
    iterations: int = 1,
) -> object:
    """Erode an image using a structuring element.

    Erosion is a morphological operation that shrinks bright regions and enlarges
    dark regions. It computes the local minimum over the neighborhood defined by
    the structuring element.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W), (H, W, C), or (N, H, W, C).
    ksize : int, optional
        Size of the structuring element (ksize x ksize). Default is 3.
    shape : {'rect', 'cross', 'ellipse'}, optional
        Shape of the structuring element. Default is 'rect'.
        - 'rect': Rectangular (box) kernel
        - 'cross': Cross-shaped kernel
        - 'ellipse': Elliptical (circular) kernel
    iterations : int, optional
        Number of times to apply erosion. Default is 1.

    Returns
    -------
    object
        Eroded image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If input shape is invalid.

    See Also
    --------
    dilation : Morphological dilation operation
    opening : Erosion followed by dilation
    closing : Dilation followed by erosion
    """
    require()

    if iterations < 1:
        return image

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)
    if img_mlx.size == 0:
        return image

    squeezed_batch_axis = False
    squeezed_channel_axis = False

    if img_mlx.ndim == 2:
        img_mlx = img_mlx[:, :, None]
        squeezed_channel_axis = True

    if img_mlx.ndim == 3:
        img_mlx = img_mlx[None, :, :, :]
        squeezed_batch_axis = True
    elif img_mlx.ndim == 4:
        pass
    else:
        raise ValueError(
            "erosion expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mlx.shape)}."
        )

    _n, h, w, c = img_mlx.shape

    # Get structuring element
    elem = _get_structuring_element(shape, ksize)
    pad = ksize // 2

    x = ensure_float32(img_mlx)

    for _ in range(iterations):
        # Pad with max value (so edges don't affect min)
        x_padded = mx.pad(
            x,
            [(0, 0), (pad, pad), (pad, pad), (0, 0)],
            constant_values=float("inf"),
        )

        # Collect windows where structuring element is 1
        windows = []
        for dy in range(ksize):
            for dx in range(ksize):
                if elem[dy, dx] > 0:
                    win = x_padded[:, dy : dy + h, dx : dx + w, :]
                    windows.append(win)

        # Stack and take minimum
        stacked = mx.stack(windows, axis=0)
        x = mx.min(stacked, axis=0)

    if squeezed_batch_axis:
        x = x[0]
    if squeezed_channel_axis:
        x = x[:, :, 0]

    if is_input_mlx:
        return x

    result_np = to_numpy(x)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def dilation(
    image: object,
    ksize: int = 3,
    shape: Literal["rect", "cross", "ellipse"] = "rect",
    iterations: int = 1,
) -> object:
    """Dilate an image using a structuring element.

    Dilation is a morphological operation that enlarges bright regions and shrinks
    dark regions. It computes the local maximum over the neighborhood defined by
    the structuring element.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W), (H, W, C), or (N, H, W, C).
    ksize : int, optional
        Size of the structuring element (ksize x ksize). Default is 3.
    shape : {'rect', 'cross', 'ellipse'}, optional
        Shape of the structuring element. Default is 'rect'.
        - 'rect': Rectangular (box) kernel
        - 'cross': Cross-shaped kernel
        - 'ellipse': Elliptical (circular) kernel
    iterations : int, optional
        Number of times to apply dilation. Default is 1.

    Returns
    -------
    object
        Dilated image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If input shape is invalid.

    See Also
    --------
    erosion : Morphological erosion operation
    opening : Erosion followed by dilation
    closing : Dilation followed by erosion
    """
    require()

    if iterations < 1:
        return image

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)
    if img_mlx.size == 0:
        return image

    squeezed_batch_axis = False
    squeezed_channel_axis = False

    if img_mlx.ndim == 2:
        img_mlx = img_mlx[:, :, None]
        squeezed_channel_axis = True

    if img_mlx.ndim == 3:
        img_mlx = img_mlx[None, :, :, :]
        squeezed_batch_axis = True
    elif img_mlx.ndim == 4:
        pass
    else:
        raise ValueError(
            "dilation expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mlx.shape)}."
        )

    _n, h, w, c = img_mlx.shape

    # Get structuring element
    elem = _get_structuring_element(shape, ksize)
    pad = ksize // 2

    x = ensure_float32(img_mlx)

    for _ in range(iterations):
        # Pad with min value (so edges don't affect max)
        x_padded = mx.pad(
            x,
            [(0, 0), (pad, pad), (pad, pad), (0, 0)],
            constant_values=float("-inf"),
        )

        # Collect windows where structuring element is 1
        windows = []
        for dy in range(ksize):
            for dx in range(ksize):
                if elem[dy, dx] > 0:
                    win = x_padded[:, dy : dy + h, dx : dx + w, :]
                    windows.append(win)

        # Stack and take maximum
        stacked = mx.stack(windows, axis=0)
        x = mx.max(stacked, axis=0)

    if squeezed_batch_axis:
        x = x[0]
    if squeezed_channel_axis:
        x = x[:, :, 0]

    if is_input_mlx:
        return x

    result_np = to_numpy(x)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def opening(
    image: object,
    ksize: int = 3,
    shape: Literal["rect", "cross", "ellipse"] = "rect",
) -> object:
    """Apply morphological opening (erosion followed by dilation).

    Opening is useful for removing small bright spots (noise) and thin protrusions
    while preserving the overall shape and size of larger bright regions.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W), (H, W, C), or (N, H, W, C).
    ksize : int, optional
        Size of the structuring element (ksize x ksize). Default is 3.
    shape : {'rect', 'cross', 'ellipse'}, optional
        Shape of the structuring element. Default is 'rect'.

    Returns
    -------
    object
        Opened image. Returns same type as input (NumPy or MLX).

    See Also
    --------
    closing : Morphological closing operation
    erosion : Erosion operation
    dilation : Dilation operation
    """
    eroded = erosion(image, ksize=ksize, shape=shape, iterations=1)
    return dilation(eroded, ksize=ksize, shape=shape, iterations=1)


def closing(
    image: object,
    ksize: int = 3,
    shape: Literal["rect", "cross", "ellipse"] = "rect",
) -> object:
    """Apply morphological closing (dilation followed by erosion).

    Closing is useful for filling small dark holes and thin gaps while preserving
    the overall shape and size of dark regions.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W), (H, W, C), or (N, H, W, C).
    ksize : int, optional
        Size of the structuring element (ksize x ksize). Default is 3.
    shape : {'rect', 'cross', 'ellipse'}, optional
        Shape of the structuring element. Default is 'rect'.

    Returns
    -------
    object
        Closed image. Returns same type as input (NumPy or MLX).

    See Also
    --------
    opening : Morphological opening operation
    erosion : Erosion operation
    dilation : Dilation operation
    """
    dilated = dilation(image, ksize=ksize, shape=shape, iterations=1)
    return erosion(dilated, ksize=ksize, shape=shape, iterations=1)


def morphological_gradient(
    image: object,
    ksize: int = 3,
    shape: Literal["rect", "cross", "ellipse"] = "rect",
) -> object:
    """Compute morphological gradient (dilation minus erosion).

    The morphological gradient highlights edges and boundaries in the image by
    computing the difference between the dilated and eroded versions.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W), (H, W, C), or (N, H, W, C).
    ksize : int, optional
        Size of the structuring element (ksize x ksize). Default is 3.
    shape : {'rect', 'cross', 'ellipse'}, optional
        Shape of the structuring element. Default is 'rect'.

    Returns
    -------
    object
        Morphological gradient image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    See Also
    --------
    erosion : Erosion operation
    dilation : Dilation operation
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    dilated = dilation(image, ksize=ksize, shape=shape, iterations=1)
    eroded = erosion(image, ksize=ksize, shape=shape, iterations=1)

    # Convert to MLX for subtraction
    dilated_mlx = to_mlx(dilated)
    eroded_mlx = to_mlx(eroded)

    result = dilated_mlx - eroded_mlx

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


__all__ = [
    "closing",
    "dilation",
    "erosion",
    "morphological_gradient",
    "opening",
]
