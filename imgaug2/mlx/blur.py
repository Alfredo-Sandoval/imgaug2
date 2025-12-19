"""
Blur operations for the MLX backend.

This module provides GPU-accelerated blur operations using Apple's MLX framework.
All functions accept both NumPy arrays and MLX arrays, returning the same type
as input.

Shapes
------
All functions support:
- ``(H, W)`` — grayscale
- ``(H, W, C)`` — single image with channels
- ``(N, H, W, C)`` — batch of images

Dtype Handling
--------------
- Input uint8 images are processed in float32 and returned as uint8 with
  values clipped to [0, 255].
- Float inputs are returned as float with the original dtype preserved.

Hybrid Operations
-----------------
``motion_blur`` can generate kernels via OpenCV on the CPU when no precomputed
kernel is provided. All other operations run fully on the MLX device.

See Also
--------
imgaug2.augmenters.blur : High-level augmenter classes using these operations.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

from ._core import ensure_float32, is_mlx_array, mx, require, to_mlx, to_numpy
from ._fast_metal import (
    gaussian_blur2d_reflect101,
    gaussian_blur_sep_reflect101_tg32,
    gaussian_kernel_2d_weights_flat_mx,
)

if TYPE_CHECKING:
    import mlx.core as _mx

    MlxArray: TypeAlias = _mx.array
else:
    MlxArray: TypeAlias = object

NumpyArray: TypeAlias = NDArray[np.generic]


def _compute_gaussian_blur_ksize(sigma: float) -> int:
    """
    Match imgaug2.augmenters.blur._compute_gaussian_blur_ksize().

    This is not "6*sigma" — it uses different multipliers depending on sigma and
    enforces a minimum 5x5.
    """
    if sigma < 3.0:
        ksize = 3.3 * sigma  # 99% of weight
    elif sigma < 5.0:
        ksize = 2.9 * sigma  # 97% of weight
    else:
        ksize = 2.6 * sigma  # 95% of weight

    ksize = int(max(ksize, 5))
    if ksize % 2 == 0:
        ksize += 1
    return ksize


def _sigma_from_ksize(ksize: int) -> float:
    """OpenCV heuristic for sigma when sigmaX=0."""
    return 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8


@lru_cache(maxsize=128)
def _reflect101_indices(length: int, pad: int) -> mx.array:
    """
    Indices for OpenCV BORDER_REFLECT_101 padding.

    Equivalent to "reflect without repeating the edge pixel".
    """
    if pad <= 0:
        return mx.arange(length, dtype=mx.int32)

    if length <= 1:
        return mx.zeros((length + 2 * pad,), dtype=mx.int32)

    period = 2 * (length - 1)
    coords = mx.arange(-pad, length + pad, dtype=mx.int32)
    coords = mx.abs(coords)
    coords = mx.remainder(coords, period)
    coords = mx.where(coords >= length, period - coords, coords)
    return coords.astype(mx.int32)


def _pad_reflect101_hw(x_nhwc: mx.array, pad_h: int, pad_w: int) -> mx.array:
    """Pad NHWC array with BORDER_REFLECT_101 on H and W axes."""
    if pad_h > 0:
        idx_h = _reflect101_indices(int(x_nhwc.shape[1]), pad_h)
        x_nhwc = mx.take(x_nhwc, idx_h, axis=1)
    if pad_w > 0:
        idx_w = _reflect101_indices(int(x_nhwc.shape[2]), pad_w)
        x_nhwc = mx.take(x_nhwc, idx_w, axis=2)
    return x_nhwc


@lru_cache(maxsize=128)
def _edge_indices(length: int, pad: int) -> mx.array:
    """Indices for edge/replicate padding."""
    if pad <= 0:
        return mx.arange(length, dtype=mx.int32)
    if length <= 1:
        return mx.zeros((length + 2 * pad,), dtype=mx.int32)
    coords = mx.arange(-pad, length + pad, dtype=mx.int32)
    return mx.clip(coords, 0, length - 1).astype(mx.int32)


def _pad_edge_hw(x_nhwc: mx.array, pad_h: int, pad_w: int) -> mx.array:
    """Pad NHWC array with edge replication on H and W axes."""
    if pad_h > 0:
        idx_h = _edge_indices(int(x_nhwc.shape[1]), pad_h)
        x_nhwc = mx.take(x_nhwc, idx_h, axis=1)
    if pad_w > 0:
        idx_w = _edge_indices(int(x_nhwc.shape[2]), pad_w)
        x_nhwc = mx.take(x_nhwc, idx_w, axis=2)
    return x_nhwc


@lru_cache(maxsize=32)
def _gaussian_kernel_1d_mlx(sigma: float, ksize: int) -> mx.array:
    radius = ksize // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel_1d = np.exp(-(x * x) / (2.0 * (sigma * sigma)))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return mx.array(kernel_1d.astype(np.float32))


@overload
def gaussian_blur(
    image: NumpyArray, sigma: float, kernel_size: int | None = None
) -> NumpyArray: ...


@overload
def gaussian_blur(image: MlxArray, sigma: float, kernel_size: int | None = None) -> MlxArray: ...


def gaussian_blur(image: object, sigma: float, kernel_size: int | None = None) -> object:
    """
    Apply Gaussian blur to an image.

    Uses separable convolution for efficiency when possible. The kernel size
    is computed automatically from sigma if not specified.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    sigma : float
        Standard deviation of the Gaussian kernel. Must be >= 0.
    kernel_size : int, optional
        Kernel size. If None, computed from sigma. Must be odd if specified.

    Returns
    -------
    array-like
        Blurred image. Same type as input (NumPy or MLX).

    Raises
    ------
    ValueError
        If sigma < 0 or image shape is invalid.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")

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
            "gaussian_blur expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mlx.shape)}."
        )

    _n, _h, _w, c = img_mlx.shape

    if kernel_size is None:
        kernel_size = _compute_gaussian_blur_ksize(float(sigma))
    else:
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)

    pad = kernel_size // 2
    sigma_eff = float(sigma) if sigma > 0 else _sigma_from_ksize(int(kernel_size))

    x_nhwc = ensure_float32(img_mlx)
    sigma_q = float(np.round(float(sigma_eff), 6))

    tile = 32 - 2 * pad
    if tile >= 8:
        w_1d = _gaussian_kernel_1d_mlx(sigma_q, int(kernel_size))
        y = gaussian_blur_sep_reflect101_tg32(x_nhwc, w_1d, ksize=int(kernel_size))
    else:
        w_flat = gaussian_kernel_2d_weights_flat_mx(sigma_q, int(kernel_size))
        y = gaussian_blur2d_reflect101(x_nhwc, w_flat, ksize=int(kernel_size))

    if squeezed_batch_axis:
        y = y[0]
    if squeezed_channel_axis:
        y = y[:, :, 0]

    if is_input_mlx:
        return y

    result_np = to_numpy(y)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def average_blur(
    image: object,
    ksize: int | tuple[int, int],
) -> object:
    """
    Apply average (box) blur to an image.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    ksize : int or tuple of int
        Kernel size. If int, uses square kernel. If tuple, ``(height, width)``.

    Returns
    -------
    array-like
        Blurred image. Same type as input.

    Raises
    ------
    ValueError
        If image shape is invalid or ksize is invalid.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    # Validate ksize
    if isinstance(ksize, int):
        if ksize < 1:
            raise ValueError(f"ksize must be >= 1, got {ksize!r}")
        kh, kw = ksize, ksize
    else:
        if len(ksize) != 2:
            raise ValueError(f"ksize must be int or (h, w) tuple, got {ksize!r}")
        kh, kw = ksize
        if kh < 1 or kw < 1:
            raise ValueError(f"ksize elements must be >= 1, got {ksize!r}")

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
            "average_blur expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mlx.shape)}."
        )

    _n, _h, _w, c = img_mlx.shape

    pad_h = kh // 2
    pad_w = kw // 2

    x_nhwc = ensure_float32(img_mlx)

    weight_val = 1.0 / (kh * kw)
    weight = mx.full((c, kh, kw, 1), weight_val, dtype=mx.float32)

    x_nhwc = _pad_reflect101_hw(x_nhwc, pad_h=pad_h, pad_w=pad_w)
    y = mx.conv2d(x_nhwc, weight, stride=1, padding=0, dilation=1, groups=c)

    if squeezed_batch_axis:
        y = y[0]
    if squeezed_channel_axis:
        y = y[:, :, 0]

    if is_input_mlx:
        return y

    result_np = to_numpy(y)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def median_blur(image: object, ksize: int) -> object:
    """
    Apply median blur to an image.

    Replaces each pixel with the median of its neighborhood. Effective for
    removing salt-and-pepper noise while preserving edges.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    ksize : int
        Kernel size. Must be odd (3, 5, 7, etc.).

    Returns
    -------
    array-like
        Blurred image. Same type as input.

    Raises
    ------
    ValueError
        If ksize is even or image shape is invalid.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    Notes
    -----
    Uses a stack-and-sort approach. Memory intensive for large kernels;
    best for ksize <= 7.
    """
    require()

    if ksize % 2 == 0:
        # imgaug/OpenCV usually enforce odd ksize for median
        raise ValueError(f"median_blur ksize must be odd, got {ksize}")

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
            "median_blur expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mlx.shape)}."
        )

    _n, h, w, _c = img_mlx.shape

    pad = ksize // 2
    x_nhwc = ensure_float32(img_mlx)

    x_padded = _pad_edge_hw(x_nhwc, pad_h=pad, pad_w=pad)

    windows = []
    for dy in range(ksize):
        for dx in range(ksize):
            win = x_padded[:, dy : dy + h, dx : dx + w, :]
            windows.append(win)

    stacked = mx.stack(windows, axis=0)
    sorted_windows = mx.sort(stacked, axis=0)
    mid_idx = (ksize * ksize) // 2
    y = sorted_windows[mid_idx]

    if squeezed_batch_axis:
        y = y[0]
    if squeezed_channel_axis:
        y = y[:, :, 0]

    if is_input_mlx:
        return y

    result_np = to_numpy(y)
    if original_dtype == np.uint8:
        return result_np.astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def motion_blur(
    image: object,
    kernel: np.ndarray | mx.array | None = None,
    k: int | None = None,
    angle: float | None = None,
) -> object:
    """
    Apply motion blur to an image.

    Simulates camera or object motion during exposure.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    kernel : array-like, optional
        Precomputed motion blur kernel. If provided, ``k`` and ``angle`` are ignored.
    k : int, optional
        Kernel size. Required if ``kernel`` is not provided.
    angle : float, optional
        Motion angle in degrees. Required if ``kernel`` is not provided.

    Returns
    -------
    array-like
        Motion-blurred image. Same type as input.

    Raises
    ------
    ValueError
        If neither ``kernel`` nor both ``k`` and ``angle`` are provided.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    Notes
    -----
    When ``kernel`` is not provided, the kernel is generated on the CPU using
    OpenCV. For a fully on-device pipeline, pass a precomputed kernel.
    """
    require()

    if kernel is None:
        if k is None or angle is None:
            raise ValueError("Must provide either 'kernel' or both 'k' and 'angle'.")

        import cv2

        k_int = int(k)
        angle_deg = float(angle)
        center = (k_int - 1) / 2.0

        M = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
        kernel_line = np.zeros((k_int, k_int), dtype=np.float32)
        cv2.line(kernel_line, (0, int(center)), (k_int - 1, int(center)), 1.0)
        kernel_cpu = cv2.warpAffine(kernel_line, M, (k_int, k_int))

        s = kernel_cpu.sum()
        if s > 0:
            kernel_cpu /= s

        kernel_mlx = mx.array(kernel_cpu)
    else:
        if isinstance(kernel, np.ndarray):
            kernel_mlx = mx.array(kernel.astype(np.float32))
        else:
            kernel_mlx = kernel

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
            "motion_blur expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mlx.shape)}."
        )

    _n, _h, _w, c = img_mlx.shape

    kh, kw = kernel_mlx.shape[:2]
    pad_h = kh // 2
    pad_w = kw // 2

    x_nhwc = ensure_float32(img_mlx)

    weight = mx.broadcast_to(kernel_mlx[..., None], (kh, kw, 1))
    weight = mx.broadcast_to(weight[None, ...], (c, kh, kw, 1))

    x_nhwc = _pad_reflect101_hw(x_nhwc, pad_h=pad_h, pad_w=pad_w)
    y = mx.conv2d(x_nhwc, weight, stride=1, padding=0, dilation=1, groups=c)

    if squeezed_batch_axis:
        y = y[0]
    if squeezed_channel_axis:
        y = y[:, :, 0]

    if is_input_mlx:
        return y

    result_np = to_numpy(y)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def downscale(
    image: object,
    scale: float,
    interpolation_down: str = "linear",
    interpolation_up: str = "linear",
    *,
    mode: str = "edge",
) -> object:
    """
    Downscale and upscale an image to simulate low-resolution artifacts.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    scale : float
        Scale factor in (0, 1]. Lower values produce stronger artifacts.
    interpolation_down : {"nearest", "linear"}, default "linear"
        Interpolation method for downscaling.
    interpolation_up : {"nearest", "linear"}, default "linear"
        Interpolation method for upscaling.
    mode : str, default "edge"
        Border mode for interpolation.

    Returns
    -------
    array-like
        Image at original size with downscale artifacts. Same type as input.

    Raises
    ------
    ValueError
        If scale is not in (0, 1] or interpolation method is invalid.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    s = float(scale)
    if not (0.0 < s <= 1.0):
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    if s == 1.0:
        return image

    interp_to_order = {"nearest": 0, "linear": 1}
    try:
        order_down = interp_to_order[str(interpolation_down).lower()]
    except KeyError as exc:
        raise ValueError(
            f"interpolation_down must be one of {sorted(interp_to_order)}, got {interpolation_down!r}"
        ) from exc
    try:
        order_up = interp_to_order[str(interpolation_up).lower()]
    except KeyError as exc:
        raise ValueError(
            f"interpolation_up must be one of {sorted(interp_to_order)}, got {interpolation_up!r}"
        ) from exc

    from .geometry import resize

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
            f"downscale expects (H,W), (H,W,C), or (N,H,W,C), got shape {tuple(img_mlx.shape)}."
        )

    _n, h, w, _c = img_mlx.shape

    new_h = max(1, int(round(int(h) * s)))
    new_w = max(1, int(round(int(w) * s)))

    downscaled = resize(img_mlx, (new_h, new_w), order=order_down, mode=mode)
    upscaled = resize(downscaled, (int(h), int(w)), order=order_up, mode=mode)

    if squeezed_batch_axis:
        upscaled = upscaled[0]
    if squeezed_channel_axis:
        upscaled = upscaled[:, :, 0]

    if is_input_mlx:
        return upscaled

    result_np = to_numpy(upscaled)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def glass_blur(
    image: object,
    sigma: float = 0.7,
    max_delta: int = 4,
    iterations: int = 2,
    seed: int | None = None,
    *,
    mode: str = "edge",
) -> object:
    """
    Apply glass blur effect (frosted glass distortion).

    Simulates looking through textured glass by randomly shuffling nearby
    pixels, then applying a Gaussian blur.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    sigma : float, default 0.7
        Gaussian blur sigma applied after pixel shuffling.
    max_delta : int, default 4
        Maximum pixel displacement distance.
    iterations : int, default 2
        Number of shuffle iterations.
    seed : int, optional
        Random seed for reproducibility.
    mode : str, default "edge"
        Border mode for out-of-bounds pixels.

    Returns
    -------
    array-like
        Glass-blurred image. Same type as input.

    Raises
    ------
    ValueError
        If image shape is invalid.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

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
            "glass_blur expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mlx.shape)}."
        )

    n, h, w, c = img_mlx.shape
    rng = np.random.default_rng(seed)

    result_np = to_numpy(img_mlx).astype(np.float32)

    for _ in range(iterations):
        dx = rng.integers(-max_delta, max_delta + 1, size=(n, h, w))
        dy = rng.integers(-max_delta, max_delta + 1, size=(n, h, w))

        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        yy = np.broadcast_to(yy[None, :, :], (n, h, w))
        xx = np.broadcast_to(xx[None, :, :], (n, h, w))

        src_y = np.clip(yy + dy, 0, h - 1)
        src_x = np.clip(xx + dx, 0, w - 1)

        for b in range(n):
            for ch in range(c):
                result_np[b, :, :, ch] = result_np[b, src_y[b], src_x[b], ch]

    result_mlx = mx.array(result_np)
    if sigma > 0:
        result_mlx = gaussian_blur(result_mlx, sigma=sigma)

    if squeezed_batch_axis:
        result_mlx = result_mlx[0]
    if squeezed_channel_axis:
        result_mlx = result_mlx[:, :, 0]

    if is_input_mlx:
        return result_mlx

    result_np = to_numpy(result_mlx)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def zoom_blur(
    image: object,
    max_factor: float = 0.1,
    step_factor: float = 0.01,
) -> object:
    """
    Apply zoom blur effect (radial blur from center).

    Simulates the effect of zooming a camera during exposure by averaging
    multiple scaled versions of the image.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    max_factor : float, default 0.1
        Maximum zoom factor. 0.1 means 10% zoom in each direction.
    step_factor : float, default 0.01
        Step size between zoom levels.

    Returns
    -------
    array-like
        Zoom-blurred image. Same type as input.

    Raises
    ------
    ValueError
        If image shape is invalid.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

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
            "zoom_blur expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mlx.shape)}."
        )

    _n, h, w, _c = img_mlx.shape

    zoom_factors = np.arange(1.0 - max_factor, 1.0 + max_factor + step_factor, step_factor)
    if len(zoom_factors) == 0:
        zoom_factors = np.array([1.0])

    x_nhwc = ensure_float32(img_mlx)
    accumulated = mx.zeros_like(x_nhwc)

    from .geometry import resize

    for zoom in zoom_factors:
        if abs(zoom - 1.0) < 1e-6:
            accumulated = accumulated + x_nhwc
        else:
            new_h = int(round(h * zoom))
            new_w = int(round(w * zoom))

            if new_h <= 0 or new_w <= 0:
                continue

            scaled = resize(x_nhwc, (new_h, new_w), order=1, mode="edge")

            if zoom > 1.0:
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                cropped = scaled[:, start_y : start_y + h, start_x : start_x + w, :]
                accumulated = accumulated + cropped
            else:
                pad_y = (h - new_h) // 2
                pad_x = (w - new_w) // 2
                padded = mx.pad(
                    scaled,
                    [(0, 0), (pad_y, h - new_h - pad_y), (pad_x, w - new_w - pad_x), (0, 0)],
                    mode="edge",
                )
                accumulated = accumulated + padded

    result = accumulated / float(len(zoom_factors))

    if squeezed_batch_axis:
        result = result[0]
    if squeezed_channel_axis:
        result = result[:, :, 0]

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def defocus_blur(
    image: object,
    radius: int = 5,
    alias_blur: float = 0.1,
) -> object:
    """
    Apply defocus (bokeh/lens) blur effect.

    Simulates out-of-focus blur using a disk-shaped kernel, mimicking
    camera bokeh.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    radius : int, default 5
        Radius of the defocus disk in pixels.
    alias_blur : float, default 0.1
        Additional Gaussian blur sigma to reduce aliasing artifacts.

    Returns
    -------
    array-like
        Defocus-blurred image. Same type as input.

    Raises
    ------
    ValueError
        If image shape is invalid.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    if radius < 1:
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
            "defocus_blur expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mlx.shape)}."
        )

    _n, _h, _w, c = img_mlx.shape

    ksize = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = (x * x + y * y <= radius * radius).astype(np.float32)
    disk = disk / disk.sum()

    if alias_blur > 0:
        from scipy.ndimage import gaussian_filter

        disk = gaussian_filter(disk, sigma=alias_blur)
        disk = disk / disk.sum()

    disk_mx = mx.array(disk)

    weight = mx.broadcast_to(disk_mx[..., None], (ksize, ksize, 1))
    weight = mx.broadcast_to(weight[None, ...], (c, ksize, ksize, 1))

    pad = radius
    x_nhwc = ensure_float32(img_mlx)
    x_nhwc = _pad_reflect101_hw(x_nhwc, pad_h=pad, pad_w=pad)
    y = mx.conv2d(x_nhwc, weight, stride=1, padding=0, dilation=1, groups=c)

    if squeezed_batch_axis:
        y = y[0]
    if squeezed_channel_axis:
        y = y[:, :, 0]

    if is_input_mlx:
        return y

    result_np = to_numpy(y)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


__all__ = [
    "average_blur",
    "defocus_blur",
    "downscale",
    "gaussian_blur",
    "glass_blur",
    "median_blur",
    "motion_blur",
    "zoom_blur",
]
