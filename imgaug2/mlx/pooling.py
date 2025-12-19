"""MLX-accelerated pooling operations for image downsampling.

This module provides hardware-accelerated pooling operations using Apple's MLX
framework. Pooling reduces spatial dimensions by aggregating values within blocks,
commonly used for downsampling and feature extraction.

The implementation matches imgaug2 semantics with automatic padding to handle
non-divisible dimensions. All functions preserve input array type (NumPy or MLX).

Examples
--------
>>> import numpy as np  # doctest: +SKIP
>>> from imgaug2.mlx.pooling import avg_pool, max_pool, min_pool  # doctest: +SKIP
>>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)  # doctest: +SKIP
>>> downsampled = avg_pool(img, block_size=2)  # Reduce to 50x50  # doctest: +SKIP
>>> max_pooled = max_pool(img, block_size=(2, 4))  # Reduce height by 2, width by 4  # doctest: +SKIP
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np

from ._core import ensure_dtype, is_mlx_array, mx, require, to_mlx, to_numpy

_PadMode = Literal["constant", "edge", "reflect", "symmetric", "wrap"]


def _normalize_block_size(block_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(block_size, tuple):
        if len(block_size) != 2:
            raise ValueError(f"block_size must be int or (h, w), got {block_size!r}")
        bh, bw = int(block_size[0]), int(block_size[1])
    else:
        bh = bw = int(block_size)

    if bh < 0 or bw < 0:
        raise ValueError(f"block_size must be >= 0, got {block_size!r}")
    return bh, bw


def _compute_axis_pad(axis_size: int, multiple: int | None) -> tuple[int, int]:
    # Mirrors imgaug2.augmenters.size.compute_paddings_to_reach_multiples_of().
    if multiple is None:
        return 0, 0
    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}")
    if axis_size == 0:
        to_pad = multiple
    elif axis_size % multiple == 0:
        to_pad = 0
    else:
        to_pad = multiple - (axis_size % multiple)
    return int(np.floor(to_pad / 2.0)), int(np.ceil(to_pad / 2.0))


@lru_cache(maxsize=128)
def _reflect101_indices(length: int, pad_before: int, pad_after: int) -> mx.array:
    """Generate indices for reflect padding without edge duplication.

    Internal helper implementing OpenCV BORDER_REFLECT_101 / numpy.pad(mode="reflect").
    Supports asymmetric padding with different before/after sizes.
    """
    if pad_before <= 0 and pad_after <= 0:
        return mx.arange(length, dtype=mx.int32)

    if length <= 1:
        return mx.zeros((length + pad_before + pad_after,), dtype=mx.int32)

    period = 2 * (length - 1)
    coords = mx.arange(-pad_before, length + pad_after, dtype=mx.int32)
    coords = mx.abs(coords)
    coords = mx.remainder(coords, period)
    coords = mx.where(coords >= length, period - coords, coords)
    return coords.astype(mx.int32)


@lru_cache(maxsize=128)
def _edge_indices(length: int, pad_before: int, pad_after: int) -> mx.array:
    if pad_before <= 0 and pad_after <= 0:
        return mx.arange(length, dtype=mx.int32)
    coords = mx.arange(-pad_before, length + pad_after, dtype=mx.int32)
    coords = mx.clip(coords, 0, max(0, length - 1))
    return coords.astype(mx.int32)


@lru_cache(maxsize=128)
def _wrap_indices(length: int, pad_before: int, pad_after: int) -> mx.array:
    if pad_before <= 0 and pad_after <= 0:
        return mx.arange(length, dtype=mx.int32)
    if length <= 0:
        return mx.zeros((pad_before + pad_after,), dtype=mx.int32)
    coords = mx.arange(-pad_before, length + pad_after, dtype=mx.int32)
    coords = mx.remainder(coords, length)
    coords = mx.where(coords < 0, coords + length, coords)
    return coords.astype(mx.int32)


def _pad_nhwc(
    x_nhwc: mx.array,
    *,
    top: int,
    right: int,
    bottom: int,
    left: int,
    mode: _PadMode,
    cval: float,
) -> mx.array:
    """Pad NHWC array to (N, H+top+bottom, W+left+right, C).

    Internal helper implementing various padding modes needed by pooling operations.
    """
    mode_str = str(mode).lower()
    if mode_str not in {"constant", "edge", "reflect", "symmetric", "wrap"}:
        raise ValueError(f"Unsupported pad mode: {mode_str!r}")

    n, h, w, c = x_nhwc.shape
    h = int(h)
    w = int(w)

    if h == 0 or w == 0 or (top == 0 and right == 0 and bottom == 0 and left == 0):
        return x_nhwc

    if mode_str == "reflect":
        idx_h = _reflect101_indices(h, top, bottom)
        idx_w = _reflect101_indices(w, left, right)
        y = mx.take(x_nhwc, idx_h, axis=1)
        y = mx.take(y, idx_w, axis=2)
        return y

    if mode_str == "symmetric":
        # numpy.pad(mode="symmetric") repeats edge values. We implement this by
        # reusing reflect101 with an off-by-one trick similar to the warp code.
        # This is not currently used by pooling augmenters, but is helpful for
        # keeping parity with padding modes used elsewhere.
        def _symmetric_indices(length: int, before: int, after: int) -> mx.array:
            if before <= 0 and after <= 0:
                return mx.arange(length, dtype=mx.int32)
            if length <= 1:
                return mx.zeros((length + before + after,), dtype=mx.int32)
            period = 2 * length
            coords = mx.arange(-before, length + after, dtype=mx.int32)
            coords = mx.abs(coords)
            coords = mx.remainder(coords, period)
            coords = mx.where(coords >= length, period - coords - 1, coords)
            return coords.astype(mx.int32)

        idx_h = _symmetric_indices(h, top, bottom)
        idx_w = _symmetric_indices(w, left, right)
        y = mx.take(x_nhwc, idx_h, axis=1)
        y = mx.take(y, idx_w, axis=2)
        return y

    if mode_str == "edge":
        idx_h = _edge_indices(h, top, bottom)
        idx_w = _edge_indices(w, left, right)
        y = mx.take(x_nhwc, idx_h, axis=1)
        y = mx.take(y, idx_w, axis=2)
        return y

    if mode_str == "wrap":
        idx_h = _wrap_indices(h, top, bottom)
        idx_w = _wrap_indices(w, left, right)
        y = mx.take(x_nhwc, idx_h, axis=1)
        y = mx.take(y, idx_w, axis=2)
        return y

    if mode_str == "constant":
        # Gather with edge indices (safe) and then mask-fill to cval.
        idx_h = _edge_indices(h, top, bottom)
        idx_w = _edge_indices(w, left, right)
        y = mx.take(x_nhwc, idx_h, axis=1)
        y = mx.take(y, idx_w, axis=2)

        coords_h = mx.arange(-top, h + bottom, dtype=mx.int32)
        coords_w = mx.arange(-left, w + right, dtype=mx.int32)
        mask_h = (coords_h >= 0) & (coords_h < h)
        mask_w = (coords_w >= 0) & (coords_w < w)
        mask_hw = (mask_h[:, None] & mask_w[None, :])[None, :, :, None]
        mask_hw = mx.broadcast_to(
            mask_hw, (int(n), int(h + top + bottom), int(w + left + right), int(c))
        )
        return mx.where(mask_hw, y, mx.array(float(cval), dtype=y.dtype))

    raise RuntimeError("Unreachable: pad mode already validated")


def _as_nhwc(img: mx.array) -> tuple[mx.array, bool, bool]:
    # Accept (H,W), (H,W,C), (N,H,W,C); return (N,H,W,C) + squeeze flags.
    squeeze_batch = False
    squeeze_channel = False
    if img.ndim == 2:
        img = img[:, :, None]
        squeeze_channel = True
    if img.ndim == 3:
        img = img[None, ...]
        squeeze_batch = True
    if img.ndim != 4:
        raise ValueError(f"Expected image ndim 2/3/4, got {img.ndim}")
    return img, squeeze_batch, squeeze_channel


def _restore_from_nhwc(
    y: mx.array,
    *,
    squeeze_batch: bool,
    squeeze_channel: bool,
) -> mx.array:
    if squeeze_batch:
        y = y[0]
    if squeeze_channel:
        y = y[..., 0]
    return y


def _restore_dtype(y_np: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return y_np > 0.5
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        # For typical images, values are non-negative; we do "round half up".
        y_np = np.floor(y_np + 0.5)
        return np.clip(y_np, info.min, info.max).astype(dtype)
    return y_np.astype(dtype, copy=False)


def _pool_reduce(
    x_nhwc: mx.array,
    *,
    block_h: int,
    block_w: int,
    reduce: Literal["mean", "max", "min"],
) -> mx.array:
    n, h, w, c = x_nhwc.shape
    h = int(h)
    w = int(w)
    if block_h == 0 or block_w == 0:
        return x_nhwc

    if h % block_h != 0 or w % block_w != 0:
        raise RuntimeError("Internal error: expected padded shape divisible by block size.")

    out_h = h // block_h
    out_w = w // block_w

    x6 = x_nhwc.reshape(int(n), out_h, block_h, out_w, block_w, int(c))
    if reduce == "mean":
        return mx.mean(x6, axis=(2, 4))
    if reduce == "max":
        return mx.max(x6, axis=(2, 4))
    if reduce == "min":
        return mx.min(x6, axis=(2, 4))
    raise ValueError(f"Unknown reduce={reduce!r}")


def avg_pool(
    image: object,
    block_size: int | tuple[int, int],
    *,
    pad_mode: _PadMode = "reflect",
    pad_cval: float = 128.0,
    preserve_dtype: bool = True,
) -> object:
    """Apply average pooling to downsample an image.

    Average pooling reduces spatial dimensions by computing the mean of each block.
    The stride equals the block size (non-overlapping blocks).

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W), (H, W, C), or (N, H, W, C).
    block_size : int or tuple of int
        Size of pooling blocks. If int, same size used for height and width.
        If tuple, specifies (block_height, block_width).
    pad_mode : {'constant', 'edge', 'reflect', 'symmetric', 'wrap'}, optional
        Padding mode for images not divisible by block_size. Default is 'reflect'.
    pad_cval : float, optional
        Constant value for 'constant' pad mode. Default is 128.0.
    preserve_dtype : bool, optional
        If True, restore original dtype in output. Default is True.

    Returns
    -------
    object
        Pooled image with dimensions reduced by block_size factor.
        Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    See Also
    --------
    max_pool : Maximum pooling operation
    min_pool : Minimum pooling operation

    Notes
    -----
    Images are automatically padded to make dimensions divisible by block_size.
    This matches imgaug2.imgaug.avg_pool() behavior.
    """
    require()
    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else np.asarray(image).dtype

    bh, bw = _normalize_block_size(block_size)
    if bh == 0 or bw == 0:
        return image

    x = to_mlx(image)
    if x.size == 0:
        return image

    x_nhwc, squeeze_batch, squeeze_channel = _as_nhwc(x)
    # Compute symmetric padding to reach multiples.
    top, bottom = _compute_axis_pad(int(x_nhwc.shape[1]), bh)
    left, right = _compute_axis_pad(int(x_nhwc.shape[2]), bw)

    x_pad = _pad_nhwc(
        x_nhwc,
        top=top,
        right=right,
        bottom=bottom,
        left=left,
        mode=pad_mode,
        cval=float(pad_cval),
    )

    y = _pool_reduce(x_pad.astype(mx.float32), block_h=bh, block_w=bw, reduce="mean")
    y = _restore_from_nhwc(y, squeeze_batch=squeeze_batch, squeeze_channel=squeeze_channel)

    if is_input_mlx:
        if preserve_dtype:
            return y.astype(x.dtype)
        return y

    y_np = to_numpy(y)
    if preserve_dtype:
        return _restore_dtype(y_np, ensure_dtype(original_dtype))
    return y_np


def max_pool(
    image: object,
    block_size: int | tuple[int, int],
    *,
    pad_mode: _PadMode = "edge",
    pad_cval: float = 0.0,
    preserve_dtype: bool = True,
) -> object:
    """Apply maximum pooling to downsample an image.

    Maximum pooling reduces spatial dimensions by taking the maximum value from
    each block. The stride equals the block size (non-overlapping blocks).

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W), (H, W, C), or (N, H, W, C).
    block_size : int or tuple of int
        Size of pooling blocks. If int, same size used for height and width.
        If tuple, specifies (block_height, block_width).
    pad_mode : {'constant', 'edge', 'reflect', 'symmetric', 'wrap'}, optional
        Padding mode for images not divisible by block_size. Default is 'edge'.
    pad_cval : float, optional
        Constant value for 'constant' pad mode. Default is 0.0.
    preserve_dtype : bool, optional
        If True, restore original dtype in output. Default is True.

    Returns
    -------
    object
        Pooled image with dimensions reduced by block_size factor.
        Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    See Also
    --------
    avg_pool : Average pooling operation
    min_pool : Minimum pooling operation

    Notes
    -----
    Images are automatically padded to make dimensions divisible by block_size.
    This matches imgaug2.imgaug.max_pool() behavior.
    """
    require()
    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else np.asarray(image).dtype

    bh, bw = _normalize_block_size(block_size)
    if bh == 0 or bw == 0:
        return image

    x = to_mlx(image)
    if x.size == 0:
        return image

    x_nhwc, squeeze_batch, squeeze_channel = _as_nhwc(x)
    top, bottom = _compute_axis_pad(int(x_nhwc.shape[1]), bh)
    left, right = _compute_axis_pad(int(x_nhwc.shape[2]), bw)

    x_pad = _pad_nhwc(
        x_nhwc,
        top=top,
        right=right,
        bottom=bottom,
        left=left,
        mode=pad_mode,
        cval=float(pad_cval),
    )

    y = _pool_reduce(x_pad, block_h=bh, block_w=bw, reduce="max")
    y = _restore_from_nhwc(y, squeeze_batch=squeeze_batch, squeeze_channel=squeeze_channel)

    if is_input_mlx:
        if preserve_dtype:
            return y.astype(x.dtype)
        return y

    y_np = to_numpy(y)
    if preserve_dtype:
        return _restore_dtype(y_np, ensure_dtype(original_dtype))
    return y_np


def min_pool(
    image: object,
    block_size: int | tuple[int, int],
    *,
    pad_mode: _PadMode = "edge",
    pad_cval: float = 255.0,
    preserve_dtype: bool = True,
) -> object:
    """Apply minimum pooling to downsample an image.

    Minimum pooling reduces spatial dimensions by taking the minimum value from
    each block. The stride equals the block size (non-overlapping blocks).

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W), (H, W, C), or (N, H, W, C).
    block_size : int or tuple of int
        Size of pooling blocks. If int, same size used for height and width.
        If tuple, specifies (block_height, block_width).
    pad_mode : {'constant', 'edge', 'reflect', 'symmetric', 'wrap'}, optional
        Padding mode for images not divisible by block_size. Default is 'edge'.
    pad_cval : float, optional
        Constant value for 'constant' pad mode. Default is 255.0.
    preserve_dtype : bool, optional
        If True, restore original dtype in output. Default is True.

    Returns
    -------
    object
        Pooled image with dimensions reduced by block_size factor.
        Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    See Also
    --------
    avg_pool : Average pooling operation
    max_pool : Maximum pooling operation

    Notes
    -----
    Images are automatically padded to make dimensions divisible by block_size.
    This matches imgaug2.imgaug.min_pool() behavior.
    """
    require()
    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else np.asarray(image).dtype

    bh, bw = _normalize_block_size(block_size)
    if bh == 0 or bw == 0:
        return image

    x = to_mlx(image)
    if x.size == 0:
        return image

    x_nhwc, squeeze_batch, squeeze_channel = _as_nhwc(x)
    top, bottom = _compute_axis_pad(int(x_nhwc.shape[1]), bh)
    left, right = _compute_axis_pad(int(x_nhwc.shape[2]), bw)

    x_pad = _pad_nhwc(
        x_nhwc,
        top=top,
        right=right,
        bottom=bottom,
        left=left,
        mode=pad_mode,
        cval=float(pad_cval),
    )

    y = _pool_reduce(x_pad, block_h=bh, block_w=bw, reduce="min")
    y = _restore_from_nhwc(y, squeeze_batch=squeeze_batch, squeeze_channel=squeeze_channel)

    if is_input_mlx:
        if preserve_dtype:
            return y.astype(x.dtype)
        return y

    y_np = to_numpy(y)
    if preserve_dtype:
        return _restore_dtype(y_np, ensure_dtype(original_dtype))
    return y_np


__all__ = ["avg_pool", "max_pool", "min_pool"]
