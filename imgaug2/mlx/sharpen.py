"""MLX-accelerated sharpening and emboss operations for image enhancement.

This module provides hardware-accelerated image sharpening operations using Apple's
MLX framework. Sharpening enhances edges and fine details in images through
convolution with specialized kernels.

Supported operations include basic sharpening, emboss effects, and unsharp masking.
All functions preserve input array type (NumPy or MLX).

Examples
--------
>>> import numpy as np
>>> from imgaug2.mlx.sharpen import sharpen, emboss, unsharp_mask
>>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
>>> sharpened = sharpen(img, alpha=1.0, lightness=1.5)
>>> embossed = emboss(img, alpha=0.8, strength=2.0)
>>> unsharp = unsharp_mask(img, sigma=1.0, strength=1.5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

from ._core import ensure_float32, is_mlx_array, mx, require, to_mlx, to_numpy

if TYPE_CHECKING:
    import mlx.core as _mx

    MlxArray: TypeAlias = _mx.array
else:
    MlxArray: TypeAlias = object

NumpyArray: TypeAlias = NDArray[np.generic]


def _apply_kernel_3x3(image: MlxArray, kernel: MlxArray) -> MlxArray:
    """Apply a 3x3 convolution kernel to an image.

    Internal helper using depthwise convolution with edge padding.
    """
    # image: NHWC, kernel: 3x3
    n, h, w, c = image.shape

    # Pad with edge replication to match previous behavior.
    padded = mx.pad(image, [(0, 0), (1, 1), (1, 1), (0, 0)], mode="edge")

    # Depthwise conv: (C, KH, KW, 1), groups=C
    kernel_mx = kernel.astype(mx.float32)
    weight = mx.broadcast_to(kernel_mx[..., None], (3, 3, 1))
    weight = mx.broadcast_to(weight[None, ...], (int(c), 3, 3, 1))

    return mx.conv2d(padded, weight, stride=1, padding=0, dilation=1, groups=int(c))


@overload
def sharpen(image: NumpyArray, alpha: float = 1.0, lightness: float = 1.0) -> NumpyArray: ...


@overload
def sharpen(image: MlxArray, alpha: float = 1.0, lightness: float = 1.0) -> MlxArray: ...


def sharpen(
    image: object,
    alpha: float = 1.0,
    lightness: float = 1.0,
) -> object:
    """Sharpen an image using a Laplacian-based sharpening kernel.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W, C) or (N, H, W, C).
    alpha : float, optional
        Blending factor between original and sharpened image. Default is 1.0.
        - 0.0: Returns original image unchanged
        - 1.0: Returns fully sharpened image
        - Values between blend proportionally
    lightness : float, optional
        Controls the strength of the sharpening effect. Default is 1.0.
        - < 1.0: More subtle sharpening
        - 1.0: Standard sharpening
        - > 1.0: Stronger sharpening

    Returns
    -------
    object
        Sharpened image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    See Also
    --------
    unsharp_mask : Alternative sharpening using Gaussian blur
    emboss : Create embossed edge effect

    Notes
    -----
    Uses a Laplacian-based kernel for edge enhancement. The kernel has the form:
        [  0,      -L,       0  ]
        [ -L,  4*L + 1,     -L  ]
        [  0,      -L,       0  ]
    where L is the lightness parameter.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    squeezed_batch = False
    if img_mlx.ndim == 3:
        img_mlx = img_mlx[None, ...]
        squeezed_batch = True

    # Sharpening kernel (Laplacian-based)
    # Standard sharpen: center = 5, edges = -1
    center = 4 * lightness + 1
    kernel = mx.array(
        [
            [0, -lightness, 0],
            [-lightness, center, -lightness],
            [0, -lightness, 0],
        ],
        dtype=mx.float32,
    )

    sharpened = _apply_kernel_3x3(img_mlx, kernel)

    # Blend with original
    if alpha < 1.0:
        result = img_mlx * (1 - alpha) + sharpened * alpha
    else:
        result = sharpened

    if squeezed_batch:
        result = result[0]

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


@overload
def emboss(image: NumpyArray, alpha: float = 1.0, strength: float = 1.0) -> NumpyArray: ...


@overload
def emboss(image: MlxArray, alpha: float = 1.0, strength: float = 1.0) -> MlxArray: ...


def emboss(
    image: object,
    alpha: float = 1.0,
    strength: float = 1.0,
) -> object:
    """Apply emboss effect to create a raised relief appearance.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W, C) or (N, H, W, C).
    alpha : float, optional
        Blending factor between original and embossed image. Default is 1.0.
        - 0.0: Returns original image unchanged
        - 1.0: Returns fully embossed image
        - Values between blend proportionally
    strength : float, optional
        Strength of the emboss effect. Default is 1.0.
        Higher values create more pronounced relief.

    Returns
    -------
    object
        Embossed image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    See Also
    --------
    sharpen : Sharpen edges without emboss effect

    Notes
    -----
    The emboss filter creates a 3D relief effect by highlighting edges in one
    direction. A midpoint gray value (128) is added to center the output range.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    squeezed_batch = False
    if img_mlx.ndim == 3:
        img_mlx = img_mlx[None, ...]
        squeezed_batch = True

    # Emboss kernel
    kernel = mx.array(
        [
            [-2 * strength, -strength, 0],
            [-strength, 1, strength],
            [0, strength, 2 * strength],
        ],
        dtype=mx.float32,
    )

    embossed = _apply_kernel_3x3(img_mlx, kernel)

    # Add 128 to center the values (for visibility)
    embossed = embossed + 128.0

    # Blend with original
    if alpha < 1.0:
        result = img_mlx * (1 - alpha) + embossed * alpha
    else:
        result = embossed

    if squeezed_batch:
        result = result[0]

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


@overload
def unsharp_mask(image: NumpyArray, sigma: float = 1.0, strength: float = 1.0) -> NumpyArray: ...


@overload
def unsharp_mask(image: MlxArray, sigma: float = 1.0, strength: float = 1.0) -> MlxArray: ...


def unsharp_mask(
    image: object,
    sigma: float = 1.0,
    strength: float = 1.0,
) -> object:
    """Apply unsharp masking for high-quality edge sharpening.

    Unsharp masking sharpens an image by subtracting a blurred version, then
    adding the difference back to the original. This is a standard technique
    in professional image editing.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W, C) or (N, H, W, C).
    sigma : float, optional
        Standard deviation for Gaussian blur. Default is 1.0.
        Larger values blur more, affecting larger-scale features.
    strength : float, optional
        Strength of the sharpening effect. Default is 1.0.
        - 0.0: No sharpening
        - 1.0: Standard sharpening
        - > 1.0: Stronger sharpening

    Returns
    -------
    object
        Sharpened image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    See Also
    --------
    sharpen : Alternative sharpening using Laplacian kernel

    Notes
    -----
    The unsharp mask formula is:
        sharpened = original + strength * (original - blurred)

    This method generally produces higher quality results than simple sharpening
    kernels, especially for photographic images.
    """
    require()

    from .blur import gaussian_blur

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    # Get blurred version
    blurred = gaussian_blur(img_mlx, sigma=sigma)
    if not is_mlx_array(blurred):
        blurred = to_mlx(blurred)

    # Unsharp mask: original + strength * (original - blurred)
    result = img_mlx + strength * (img_mlx - blurred)

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


__all__ = [
    "emboss",
    "sharpen",
    "unsharp_mask",
]
