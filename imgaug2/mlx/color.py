"""
Color operations for the MLX backend.

This module provides GPU-accelerated color manipulation operations using Apple's
MLX framework. All functions accept both NumPy arrays and MLX arrays, returning
the same type as input.

Shapes
------
All functions support:
- ``(H, W, C)`` — single image with channels
- ``(N, H, W, C)`` — batch of images

Some functions also support:
- ``(H, W)`` — grayscale (channel dimension added internally)

Dtype Handling
--------------
- Input uint8 images are processed in float32 and returned as uint8 with
  values clipped to [0, 255].
- Float inputs are returned as float with the original dtype preserved when possible.
- Operations that require float output (e.g., normalize) always return float32.

Color Spaces
------------
- RGB/BGR operations assume last dimension is channels in standard order
- HSV conversions use: H in [0, 360), S in [0, 1], V in [0, 1]
- All color space conversions preserve spatial dimensions

See Also
--------
imgaug2.augmenters.color : High-level augmenter classes using these operations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import DTypeLike

from ._core import ensure_float32, is_mlx_array, mx, require, to_mlx, to_numpy


def _restore_dtype(result: object, original_dtype: np.dtype | None, is_input_mlx: bool) -> object:
    """
    Restore original dtype for NumPy outputs.

    Parameters
    ----------
    result : array-like
        Result array (MLX or NumPy).
    original_dtype : np.dtype or None
        Original dtype to restore (None for MLX input).
    is_input_mlx : bool
        Whether the input was an MLX array.

    Returns
    -------
    array-like
        Result with restored dtype (MLX array or NumPy array).
    """
    if is_input_mlx:
        return result
    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def _mx_dtype_from_numpy(dtype: DTypeLike) -> object | None:
    """
    Convert NumPy dtype to MLX dtype.

    Parameters
    ----------
    dtype : dtype-like
        NumPy dtype or dtype-like object.

    Returns
    -------
    mx.Dtype or None
        Corresponding MLX dtype, or None if not found.
    """
    try:
        np_dtype = np.dtype(dtype)
    except (TypeError, ValueError):
        return None
    return getattr(mx, np_dtype.name, None)


def grayscale(image: object, alpha: float = 1.0) -> object:
    """
    Convert RGB image to grayscale using luminance weights.

    Uses standard ITU-R BT.601 luma coefficients: 0.299*R + 0.587*G + 0.114*B.

    Parameters
    ----------
    image : array-like
        Input RGB image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
    alpha : float, default 1.0
        Blending factor. 1.0 = full grayscale, 0.0 = original image.

    Returns
    -------
    array-like
        Grayscale image (with same number of channels as input). Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    weights = mx.array([0.299, 0.587, 0.114], dtype=mx.float32)
    gray = mx.sum(img_mlx * weights, axis=-1, keepdims=True)
    gray = mx.broadcast_to(gray, img_mlx.shape)

    if alpha < 1.0:
        result = img_mlx * (1 - alpha) + gray * alpha
    else:
        result = gray

    return _restore_dtype(result, original_dtype, is_input_mlx)


def rgb_shift(
    image: object,
    r_shift: float = 0.0,
    g_shift: float = 0.0,
    b_shift: float = 0.0,
) -> object:
    """
    Shift RGB channels by specified amounts.

    Parameters
    ----------
    image : array-like
        Input RGB image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
    r_shift : float, default 0.0
        Amount to add to red channel (in [0, 255] scale for uint8).
    g_shift : float, default 0.0
        Amount to add to green channel.
    b_shift : float, default 0.0
        Amount to add to blue channel.

    Returns
    -------
    array-like
        Image with shifted RGB channels. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    shift = mx.array([r_shift, g_shift, b_shift], dtype=mx.float32)
    result = img_mlx + shift

    return _restore_dtype(result, original_dtype, is_input_mlx)


def channel_shuffle(
    image: object, order: Sequence[int] | None = None, seed: int | None = None
) -> object:
    """
    Shuffle or permute image channels.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
    order : sequence of int, optional
        Explicit channel order, e.g., [2, 0, 1] for BGR to RGB conversion.
        If None, a random permutation is used.
    seed : int, optional
        Random seed for reproducible shuffling. Only used if ``order`` is None.

    Returns
    -------
    array-like
        Image with shuffled channels. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)
    num_channels = img_mlx.shape[-1]

    if order is None:
        rng = np.random.default_rng(seed)
        order = rng.permutation(num_channels).tolist()

    result = mx.concatenate([img_mlx[..., i : i + 1] for i in order], axis=-1)

    return _restore_dtype(result, original_dtype, is_input_mlx)


def normalize(
    image: object,
    mean: Sequence[float] | float = 0.0,
    std: Sequence[float] | float = 1.0,
) -> object:
    """
    Normalize image by subtracting mean and dividing by standard deviation.

    Common use case: ImageNet normalization with mean=[0.485, 0.456, 0.406] and
    std=[0.229, 0.224, 0.225] for images already scaled to [0, 1].

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        Should be float in [0, 1] for typical use.
    mean : float or sequence of float, default 0.0
        Per-channel mean or single value for all channels.
    std : float or sequence of float, default 1.0
        Per-channel standard deviation or single value for all channels.

    Returns
    -------
    array-like
        Normalized image as float32. Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If mean or std sequence length does not match number of channels.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    img_mlx = ensure_float32(to_mlx(image))
    num_channels = int(img_mlx.shape[-1])

    if isinstance(mean, (int, float)):
        mean_arr = mx.full((num_channels,), mean, dtype=mx.float32)
    else:
        mean_list = list(mean)
        if len(mean_list) != num_channels:
            raise ValueError(
                f"mean length must match number of channels ({num_channels}), "
                f"got {len(mean_list)}"
            )
        mean_arr = mx.array(mean_list, dtype=mx.float32)

    if isinstance(std, (int, float)):
        std_arr = mx.full((num_channels,), std, dtype=mx.float32)
    else:
        std_list = list(std)
        if len(std_list) != num_channels:
            raise ValueError(
                f"std length must match number of channels ({num_channels}), "
                f"got {len(std_list)}"
            )
        std_arr = mx.array(std_list, dtype=mx.float32)

    result = (img_mlx - mean_arr) / std_arr

    if is_input_mlx:
        return result
    return to_numpy(result)


def denormalize(
    image: object,
    mean: Sequence[float] | float = 0.0,
    std: Sequence[float] | float = 1.0,
) -> object:
    """
    Reverse normalization by multiplying by std and adding mean.

    Parameters
    ----------
    image : array-like
        Normalized image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
    mean : float or sequence of float, default 0.0
        Per-channel mean or single value.
    std : float or sequence of float, default 1.0
        Per-channel standard deviation or single value.

    Returns
    -------
    array-like
        Denormalized image as float32. Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If mean or std sequence length does not match number of channels.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    img_mlx = ensure_float32(to_mlx(image))
    num_channels = int(img_mlx.shape[-1])

    if isinstance(mean, (int, float)):
        mean_arr = mx.full((num_channels,), mean, dtype=mx.float32)
    else:
        mean_list = list(mean)
        if len(mean_list) != num_channels:
            raise ValueError(
                f"mean length must match number of channels ({num_channels}), "
                f"got {len(mean_list)}"
            )
        mean_arr = mx.array(mean_list, dtype=mx.float32)

    if isinstance(std, (int, float)):
        std_arr = mx.full((num_channels,), std, dtype=mx.float32)
    else:
        std_list = list(std)
        if len(std_list) != num_channels:
            raise ValueError(
                f"std length must match number of channels ({num_channels}), "
                f"got {len(std_list)}"
            )
        std_arr = mx.array(std_list, dtype=mx.float32)

    result = img_mlx * std_arr + mean_arr

    if is_input_mlx:
        return result
    return to_numpy(result)


def to_float(image: object, max_value: float = 255.0) -> object:
    """
    Convert image to float32 in [0, 1] range.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array (typically uint8 in [0, 255]).
    max_value : float, default 255.0
        Maximum value to divide by (255 for uint8).

    Returns
    -------
    array-like
        Float32 image in [0, 1]. Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    img_mlx = to_mlx(image).astype(mx.float32) / max_value

    if is_input_mlx:
        return img_mlx
    return to_numpy(img_mlx)


def from_float(image: object, max_value: float = 255.0, dtype: DTypeLike = np.uint8) -> object:
    """
    Convert float image back to integer range.

    Parameters
    ----------
    image : array-like
        Float image as NumPy array or MLX array (typically in [0, 1]).
    max_value : float, default 255.0
        Maximum value to multiply by.
    dtype : dtype-like, default np.uint8
        Target NumPy dtype.

    Returns
    -------
    array-like
        Image scaled to [0, max_value] and converted to dtype. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    img_mlx = ensure_float32(to_mlx(image)) * max_value

    if is_input_mlx:
        result = mx.clip(img_mlx, 0, float(max_value))
        mx_dtype = _mx_dtype_from_numpy(dtype)
        if mx_dtype is None:
            return result
        return result.astype(mx_dtype)

    result_np = to_numpy(img_mlx)
    return np.clip(result_np, 0, max_value).astype(dtype)


def rgb_to_hsv(image: object) -> object:
    """
    Convert RGB image to HSV color space.

    Parameters
    ----------
    image : array-like
        RGB image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        Values can be in [0, 1] or [0, 255] (automatically detected).

    Returns
    -------
    array-like
        HSV image with H in [0, 360), S in [0, 1], V in [0, 1]. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    img_mlx = ensure_float32(to_mlx(image))

    max_val = float(to_numpy(mx.max(img_mlx)))
    if max_val > 1.0:
        img_mlx = img_mlx / 255.0

    r = img_mlx[..., 0]
    g = img_mlx[..., 1]
    b = img_mlx[..., 2]

    cmax = mx.maximum(mx.maximum(r, g), b)
    cmin = mx.minimum(mx.minimum(r, g), b)
    delta = cmax - cmin

    h = mx.zeros_like(cmax)

    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)

    h = mx.where(mask_r, 60.0 * (((g - b) / (delta + 1e-8)) % 6), h)
    h = mx.where(mask_g, 60.0 * (((b - r) / (delta + 1e-8)) + 2), h)
    h = mx.where(mask_b, 60.0 * (((r - g) / (delta + 1e-8)) + 4), h)

    s = mx.where(cmax > 0, delta / (cmax + 1e-8), mx.zeros_like(cmax))
    v = cmax

    result = mx.stack([h, s, v], axis=-1)

    if is_input_mlx:
        return result
    return to_numpy(result)


def hsv_to_rgb(image: object) -> object:
    """
    Convert HSV image to RGB color space.

    Parameters
    ----------
    image : array-like
        HSV image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        H in [0, 360), S in [0, 1], V in [0, 1].

    Returns
    -------
    array-like
        RGB image with values in [0, 1]. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    img_mlx = ensure_float32(to_mlx(image))

    h = img_mlx[..., 0]
    s = img_mlx[..., 1]
    v = img_mlx[..., 2]

    c = v * s
    h_prime = h / 60.0
    x = c * (1 - mx.abs(h_prime % 2 - 1))
    m = v - c

    zeros = mx.zeros_like(c)
    h_int = (h_prime.astype(mx.int32)) % 6

    r = mx.where(
        h_int == 0,
        c,
        mx.where(
            h_int == 1,
            x,
            mx.where(h_int == 2, zeros, mx.where(h_int == 3, zeros, mx.where(h_int == 4, x, c))),
        ),
    )
    g = mx.where(
        h_int == 0,
        x,
        mx.where(
            h_int == 1,
            c,
            mx.where(h_int == 2, c, mx.where(h_int == 3, x, mx.where(h_int == 4, zeros, zeros))),
        ),
    )
    b = mx.where(
        h_int == 0,
        zeros,
        mx.where(
            h_int == 1,
            zeros,
            mx.where(h_int == 2, x, mx.where(h_int == 3, c, mx.where(h_int == 4, c, x))),
        ),
    )

    result = mx.stack([r + m, g + m, b + m], axis=-1)

    if is_input_mlx:
        return result
    return to_numpy(result)


def hue_saturation_value(
    image: object,
    hue_shift: float = 0.0,
    saturation_scale: float = 1.0,
    value_scale: float = 1.0,
) -> object:
    """
    Adjust hue, saturation, and value of an RGB image.

    Parameters
    ----------
    image : array-like
        RGB image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        Values can be in [0, 1] or [0, 255] (automatically detected).
    hue_shift : float, default 0.0
        Amount to shift hue in degrees (range [-180, 180] typical).
    saturation_scale : float, default 1.0
        Multiplier for saturation.
    value_scale : float, default 1.0
        Multiplier for value/brightness.

    Returns
    -------
    array-like
        Adjusted RGB image in original range. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    max_val = float(to_numpy(mx.max(img_mlx)))
    is_uint8_range = max_val > 1.0

    if is_uint8_range:
        img_mlx = img_mlx / 255.0

    hsv = rgb_to_hsv(img_mlx)
    if not is_mlx_array(hsv):
        hsv = to_mlx(hsv)

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    h = (h + hue_shift) % 360.0
    s = mx.clip(s * saturation_scale, 0.0, 1.0)
    v = mx.clip(v * value_scale, 0.0, 1.0)

    hsv_adjusted = mx.stack([h, s, v], axis=-1)

    result = hsv_to_rgb(hsv_adjusted)
    if not is_mlx_array(result):
        result = to_mlx(result)

    if is_uint8_range:
        result = result * 255.0

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def equalize(image: object) -> object:
    """
    Apply histogram equalization to an image.

    Equalizes the histogram of each channel independently to improve contrast.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        Values in [0, 255] range.

    Returns
    -------
    array-like
        Histogram-equalized image. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)
    ndim = len(img_mlx.shape)

    squeezed_batch = False
    if ndim == 3:
        img_mlx = img_mlx[None, ...]
        squeezed_batch = True

    # Work with float for computation
    img_float = img_mlx.astype(mx.float32)

    n, h, w, c = img_float.shape
    if h == 0 or w == 0:
        return _restore_dtype(img_float, original_dtype, is_input_mlx)

    num_pixels = h * w
    bins = mx.arange(256, dtype=mx.int32)
    result_channels = []

    for ch in range(c):
        channel = img_float[..., ch]  # (n, h, w)
        flat = mx.reshape(channel, (n, -1))  # (n, num_pixels)
        flat_int = mx.clip(mx.round(flat), 0, 255).astype(mx.int32)

        # Build histograms for all batches at once: (n, 256)
        hist = mx.sum(flat_int[:, :, None] == bins[None, None, :], axis=1).astype(mx.float32)
        cdf = mx.cumsum(hist, axis=1)
        cdf_min = mx.min(
            mx.where(cdf > 0, cdf, mx.full(cdf.shape, float("inf"), dtype=cdf.dtype)),
            axis=1,
        )
        cdf_normalized = (
            (cdf - cdf_min[:, None]) / (float(num_pixels) - cdf_min[:, None] + 1e-8) * 255.0
        )
        lut_stack = mx.clip(cdf_normalized, 0, 255)

        # Vectorized apply: compute flat indices for each batch
        lut_flat = mx.reshape(lut_stack, (n * 256,))
        batch_offsets = mx.arange(n, dtype=mx.int32)[:, None] * 256  # (n, 1)
        indices = batch_offsets + flat_int  # (n, num_pixels)
        indices_flat = mx.reshape(indices, (-1,))

        equalized_flat = mx.take(lut_flat, indices_flat)
        equalized = mx.reshape(equalized_flat, (n, h, w))
        result_channels.append(equalized)

    result = mx.stack(result_channels, axis=-1)

    if squeezed_batch:
        result = result[0]

    return _restore_dtype(result, original_dtype, is_input_mlx)


def _compute_tile_lut(tile: object, clip_limit: float, tile_size: int) -> object:
    """
    Compute CLAHE lookup table for a single tile.

    Parameters
    ----------
    tile : mx.array
        Tile data in float32.
    clip_limit : float
        Contrast limiting threshold.
    tile_size : int
        Number of pixels in the tile.

    Returns
    -------
    mx.array
        Lookup table (256 values) for the tile.
    """
    tile_int = mx.clip(mx.round(tile), 0, 255).astype(mx.int32)
    flat = mx.reshape(tile_int, (-1,))

    bins = mx.arange(256, dtype=mx.int32)
    hist = mx.sum(flat[:, None] == bins[None, :], axis=0).astype(mx.float32)

    clip_threshold = clip_limit * tile_size / 256.0

    excess = mx.sum(mx.maximum(hist - clip_threshold, 0))
    hist = mx.minimum(hist, clip_threshold)
    hist = hist + excess / 256.0

    cdf = mx.cumsum(hist)
    cdf_min = mx.min(mx.where(cdf > 0, cdf, mx.full(cdf.shape, float("inf"), dtype=cdf.dtype)))
    cdf_normalized = (cdf - cdf_min) / (tile_size - cdf_min + 1e-8) * 255.0
    return mx.clip(cdf_normalized, 0, 255)


def clahe(
    image: object,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> object:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE divides the image into tiles and applies histogram equalization
    with contrast limiting to each tile, then bilinearly interpolates the results.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        Values in [0, 255] range.
    clip_limit : float, default 2.0
        Threshold for contrast limiting. Higher values give more contrast.
    tile_grid_size : tuple of int, default (8, 8)
        Number of tiles in (rows, cols).

    Returns
    -------
    array-like
        CLAHE-enhanced image. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)
    ndim = len(img_mlx.shape)

    squeezed_batch = False
    if ndim == 3:
        img_mlx = img_mlx[None, ...]
        squeezed_batch = True

    img_float = img_mlx.astype(mx.float32)
    n, h, w, c = img_float.shape

    if h == 0 or w == 0:
        return _restore_dtype(img_float, original_dtype, is_input_mlx)

    tile_rows, tile_cols = tile_grid_size
    tile_rows = max(1, min(int(tile_rows), h))
    tile_cols = max(1, min(int(tile_cols), w))
    tile_h = max(1, h // tile_rows)
    tile_w = max(1, w // tile_cols)

    y_coords = mx.arange(h, dtype=mx.float32)
    x_coords = mx.arange(w, dtype=mx.float32)
    ty = y_coords / float(tile_h) - 0.5
    tx = x_coords / float(tile_w) - 0.5
    ty0 = mx.clip(mx.floor(ty).astype(mx.int32), 0, tile_rows - 1)
    ty1 = mx.clip(ty0 + 1, 0, tile_rows - 1)
    tx0 = mx.clip(mx.floor(tx).astype(mx.int32), 0, tile_cols - 1)
    tx1 = mx.clip(tx0 + 1, 0, tile_cols - 1)
    fy = ty - mx.floor(ty)
    fx = tx - mx.floor(tx)

    tile_idx00 = ty0[:, None] * tile_cols + tx0[None, :]
    tile_idx01 = ty0[:, None] * tile_cols + tx1[None, :]
    tile_idx10 = ty1[:, None] * tile_cols + tx0[None, :]
    tile_idx11 = ty1[:, None] * tile_cols + tx1[None, :]
    fx_b = fx[None, :]
    fy_b = fy[:, None]

    result_channels = []

    for ch in range(c):
        channel = img_float[..., ch]
        equalized_batch = []

        for b in range(n):
            channel_b = channel[b]

            luts = []
            for tr in range(tile_rows):
                row_luts = []
                for tc in range(tile_cols):
                    y0 = tr * tile_h
                    y1 = (tr + 1) * tile_h if tr < tile_rows - 1 else h
                    x0 = tc * tile_w
                    x1 = (tc + 1) * tile_w if tc < tile_cols - 1 else w

                    tile = channel_b[y0:y1, x0:x1]
                    tile_size = int((y1 - y0) * (x1 - x0))
                    lut = _compute_tile_lut(tile, clip_limit, tile_size)
                    row_luts.append(lut)
                luts.append(row_luts)

            lut_array = mx.stack([mx.stack(row, axis=0) for row in luts], axis=0)
            lut_flat = lut_array.reshape(tile_rows * tile_cols * 256)

            pixel_int = mx.clip(mx.round(channel_b), 0, 255).astype(mx.int32)

            idx00 = tile_idx00 * 256 + pixel_int
            idx01 = tile_idx01 * 256 + pixel_int
            idx10 = tile_idx10 * 256 + pixel_int
            idx11 = tile_idx11 * 256 + pixel_int

            v00 = mx.take(lut_flat, idx00)
            v01 = mx.take(lut_flat, idx01)
            v10 = mx.take(lut_flat, idx10)
            v11 = mx.take(lut_flat, idx11)

            v0 = v00 * (1.0 - fx_b) + v01 * fx_b
            v1 = v10 * (1.0 - fx_b) + v11 * fx_b
            result_img = v0 * (1.0 - fy_b) + v1 * fy_b
            equalized_batch.append(result_img)

        result_channels.append(mx.stack(equalized_batch, axis=0))

    result = mx.stack(result_channels, axis=-1)

    if squeezed_batch:
        result = result[0]

    return _restore_dtype(result, original_dtype, is_input_mlx)


def channel_dropout(
    image: object,
    channel_idx: int | Sequence[int] | None = None,
    fill_value: float = 0.0,
    seed: int | None = None,
) -> object:
    """
    Drop (set to fill value) one or more channels.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
    channel_idx : int or sequence of int, optional
        Index or indices of channels to drop. If None, drops a random channel.
    fill_value : float, default 0.0
        Value to fill dropped channels with.
    seed : int, optional
        Random seed for reproducible channel selection (only used if ``channel_idx`` is None).

    Returns
    -------
    array-like
        Image with dropped channels. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)
    num_channels = img_mlx.shape[-1]

    if channel_idx is None:
        rng = np.random.default_rng(seed)
        channel_idx = int(rng.integers(0, num_channels))

    if isinstance(channel_idx, int):
        channel_idx = [channel_idx]

    result = img_mlx
    for idx in channel_idx:
        channels = [img_mlx[..., i : i + 1] for i in range(num_channels)]
        channels[idx] = mx.full(channels[idx].shape, fill_value, dtype=img_mlx.dtype)
        result = mx.concatenate(channels, axis=-1)
        img_mlx = result

    return _restore_dtype(result, original_dtype, is_input_mlx)


def posterize(image: object, bits: int = 4) -> object:
    """
    Reduce the number of bits for each color channel.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        Values in [0, 255] range.
    bits : int, default 4
        Number of bits to keep per channel (1-8).

    Returns
    -------
    array-like
        Posterized image. Same type as input.

    Raises
    ------
    ValueError
        If bits is not in range [1, 8].
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be between 1 and 8, got {bits}")

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image).astype(mx.float32)

    shift = 8 - bits
    divisor = float(2**shift)
    result = mx.floor(img_mlx / divisor) * divisor

    return _restore_dtype(result, original_dtype, is_input_mlx)


def autocontrast(image: object, cutoff: float = 0.0) -> object:
    """
    Maximize image contrast by stretching the histogram to full range.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        Values in [0, 255] range.
    cutoff : float, default 0.0
        Percentage of pixels to cut off from low and high ends (0-50).

    Returns
    -------
    array-like
        Auto-contrasted image. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)
    ndim = len(img_mlx.shape)

    squeezed_batch = False
    if ndim == 3:
        img_mlx = img_mlx[None, ...]
        squeezed_batch = True

    img_float = img_mlx.astype(mx.float32)
    n, _h, _w, c = img_float.shape

    result_channels = []

    for ch in range(c):
        channel = img_float[..., ch]
        adjusted_batch = []

        for b in range(n):
            channel_b = channel[b]
            flat = mx.reshape(channel_b, (-1,))

            if cutoff > 0:
                sorted_vals = mx.sort(flat)
                num_pixels = len(sorted_vals)
                lo_idx = int(num_pixels * cutoff / 100.0)
                hi_idx = int(num_pixels * (100.0 - cutoff) / 100.0) - 1
                lo = float(sorted_vals[max(0, lo_idx)])
                hi = float(sorted_vals[min(num_pixels - 1, hi_idx)])
            else:
                lo = float(mx.min(flat))
                hi = float(mx.max(flat))

            if hi <= lo:
                adjusted_batch.append(channel_b)
            else:
                scale = 255.0 / (hi - lo)
                adjusted = (channel_b - lo) * scale
                adjusted = mx.clip(adjusted, 0, 255)
                adjusted_batch.append(adjusted)

        result_channels.append(mx.stack(adjusted_batch, axis=0))

    result = mx.stack(result_channels, axis=-1)

    if squeezed_batch:
        result = result[0]

    return _restore_dtype(result, original_dtype, is_input_mlx)


def color_jitter(
    image: object,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
) -> object:
    """
    Adjust brightness, contrast, saturation, and hue.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        Values in [0, 255] range.
    brightness : float, default 0.0
        Brightness adjustment factor (0 = no change).
    contrast : float, default 0.0
        Contrast adjustment factor (0 = no change).
    saturation : float, default 0.0
        Saturation adjustment factor (0 = no change).
    hue : float, default 0.0
        Hue adjustment factor in [-0.5, 0.5] (0 = no change).

    Returns
    -------
    array-like
        Color-jittered image. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    if brightness != 0:
        img_mlx = img_mlx * (1 + brightness)

    if contrast != 0:
        mean = mx.mean(img_mlx, axis=(-3, -2), keepdims=True)
        img_mlx = (img_mlx - mean) * (1 + contrast) + mean

    if saturation != 0 or hue != 0:
        result = hue_saturation_value(
            img_mlx,
            hue_shift=hue * 360,
            saturation_scale=1 + saturation,
            value_scale=1.0,
        )
        if not is_mlx_array(result):
            img_mlx = to_mlx(result)
        else:
            img_mlx = result

    result = mx.clip(img_mlx, 0, 255)

    return _restore_dtype(result, original_dtype, is_input_mlx)


def sepia(image: object, strength: float = 1.0) -> object:
    """
    Apply sepia tone effect to an RGB image.

    The sepia transform applies a color matrix that gives images a warm,
    brownish tone reminiscent of old photographs.

    Parameters
    ----------
    image : array-like
        Input RGB image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        Values in [0, 255] range.
    strength : float, default 1.0
        Strength of the effect (0.0 = no effect, 1.0 = full sepia).

    Returns
    -------
    array-like
        Sepia-toned image. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    sepia_matrix = mx.array(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ],
        dtype=mx.float32,
    )

    sepia_result = mx.matmul(img_mlx, sepia_matrix.T)

    if strength < 1.0:
        result = img_mlx * (1.0 - strength) + sepia_result * strength
    else:
        result = sepia_result

    result = mx.clip(result, 0, 255)

    return _restore_dtype(result, original_dtype, is_input_mlx)


def fancy_pca(
    image: object,
    alpha_std: float = 0.1,
    seed: int | None = None,
) -> object:
    """
    Apply PCA-based color augmentation (FancyPCA / AlexNet-style).

    This augmentation adds multiples of the principal components of the
    RGB pixel values, with magnitudes proportional to the eigenvalues.

    Parameters
    ----------
    image : array-like
        Input RGB image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        Values in [0, 255] range.
    alpha_std : float, default 0.1
        Standard deviation for the random alpha values.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Color-augmented image. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    Notes
    -----
    Uses pre-computed ImageNet PCA values from the AlexNet paper.
    For dataset-specific PCA, compute eigenvectors from your data.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    eigenvectors = mx.array(
        [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ],
        dtype=mx.float32,
    )

    eigenvalues = mx.array([0.2175, 0.0188, 0.0045], dtype=mx.float32)

    rng = np.random.default_rng(seed)
    alpha = rng.normal(0, alpha_std, size=3).astype(np.float32)
    alpha_mx = mx.array(alpha)

    delta = mx.sum(alpha_mx * eigenvalues * eigenvectors, axis=1)
    result = img_mlx + delta
    result = mx.clip(result, 0, 255)

    return _restore_dtype(result, original_dtype, is_input_mlx)


_PLANCKIAN_COEFFS = {
    # Temperature (K): (R, G, B) multipliers (normalized)
    3000: (1.0000, 0.7067, 0.4264),
    3500: (1.0000, 0.7671, 0.5328),
    4000: (1.0000, 0.8198, 0.6266),
    4500: (1.0000, 0.8663, 0.7117),
    5000: (1.0000, 0.9076, 0.7901),
    5500: (1.0000, 0.9445, 0.8634),
    6000: (1.0000, 0.9778, 0.9326),
    6500: (1.0000, 1.0000, 1.0000),  # Reference (D65)
    7000: (0.9548, 0.9760, 1.0000),
    7500: (0.9188, 0.9572, 1.0000),
    8000: (0.8890, 0.9418, 1.0000),
    8500: (0.8639, 0.9290, 1.0000),
    9000: (0.8425, 0.9181, 1.0000),
    9500: (0.8240, 0.9087, 1.0000),
    10000: (0.8079, 0.9005, 1.0000),
}


def _interpolate_planckian(temperature: float) -> tuple[float, float, float]:
    """
    Interpolate RGB multipliers for a given color temperature.

    Parameters
    ----------
    temperature : float
        Color temperature in Kelvin.

    Returns
    -------
    tuple of float
        RGB multipliers (r, g, b).
    """
    temps = sorted(_PLANCKIAN_COEFFS.keys())
    if temperature <= temps[0]:
        return _PLANCKIAN_COEFFS[temps[0]]
    if temperature >= temps[-1]:
        return _PLANCKIAN_COEFFS[temps[-1]]

    for i in range(len(temps) - 1):
        if temps[i] <= temperature <= temps[i + 1]:
            t0, t1 = temps[i], temps[i + 1]
            r0, g0, b0 = _PLANCKIAN_COEFFS[t0]
            r1, g1, b1 = _PLANCKIAN_COEFFS[t1]
            alpha = (temperature - t0) / (t1 - t0)
            return (
                r0 + alpha * (r1 - r0),
                g0 + alpha * (g1 - g0),
                b0 + alpha * (b1 - b0),
            )
    return (1.0, 1.0, 1.0)


def planckian_jitter(
    image: object,
    temperature_range: tuple[float, float] = (4500, 7500),
    seed: int | None = None,
) -> object:
    """
    Apply color temperature jitter based on Planckian locus.

    Simulates changes in lighting color temperature, like the difference
    between warm tungsten lighting and cool daylight.

    Parameters
    ----------
    image : array-like
        Input RGB image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        Values in [0, 255] range.
    temperature_range : tuple of float, default (4500, 7500)
        Range of color temperatures in Kelvin (min, max).
        - 3000K: Warm/tungsten (orange)
        - 5500K: Daylight
        - 6500K: D65 reference (neutral)
        - 10000K: Cool/blue sky
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Color temperature adjusted image. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    rng = np.random.default_rng(seed)
    temperature = rng.uniform(temperature_range[0], temperature_range[1])

    r_mult, g_mult, b_mult = _interpolate_planckian(temperature)
    multipliers = mx.array([r_mult, g_mult, b_mult], dtype=mx.float32)

    result = img_mlx * multipliers
    result = mx.clip(result, 0, 255)

    return _restore_dtype(result, original_dtype, is_input_mlx)


def solarize(image: object, threshold: float = 128.0) -> object:
    """
    Invert all pixel values above a threshold.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        Values in [0, 255] range.
    threshold : float, default 128.0
        All pixels at or above this value are inverted.

    Returns
    -------
    array-like
        Solarized image. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))

    result = mx.where(img_mlx >= threshold, 255.0 - img_mlx, img_mlx)

    return _restore_dtype(result, original_dtype, is_input_mlx)


def invert(image: object) -> object:
    """
    Invert all pixel values.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape: (H, W, C) or (N, H, W, C).
        Values in [0, 255] range.

    Returns
    -------
    array-like
        Inverted image. Same type as input.

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))
    result = 255.0 - img_mlx

    return _restore_dtype(result, original_dtype, is_input_mlx)


__all__ = [
    "autocontrast",
    "channel_dropout",
    "channel_shuffle",
    "clahe",
    "color_jitter",
    "denormalize",
    "equalize",
    "fancy_pca",
    "from_float",
    "grayscale",
    "hsv_to_rgb",
    "hue_saturation_value",
    "invert",
    "normalize",
    "planckian_jitter",
    "posterize",
    "rgb_shift",
    "rgb_to_hsv",
    "sepia",
    "solarize",
    "to_float",
]
