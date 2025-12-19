"""GPU-accelerated pooling operations using CUDA/CuPy.

This module provides GPU-accelerated implementations of pooling operations
including average, max, and min pooling, optimized for batch processing on
CUDA-compatible devices.
"""
from __future__ import annotations

import numpy as np

from ._core import cp, is_cupy_array, require


def _require_cupy_array(x: object, fn_name: str) -> cp.ndarray:
    require()
    if not is_cupy_array(x):
        raise TypeError(
            f"{fn_name} expects a cupy.ndarray (GPU resident). "
            "Convert first via `cupy.asarray(...)`."
        )
    return x


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


def _pad_hw(
    x: cp.ndarray,
    *,
    top: int,
    right: int,
    bottom: int,
    left: int,
    mode: str,
    cval: float,
) -> cp.ndarray:
    if top == 0 and right == 0 and bottom == 0 and left == 0:
        return x

    mode = str(mode).lower()
    if x.ndim == 2:
        pad_width = ((top, bottom), (left, right))
    elif x.ndim == 3:
        pad_width = ((top, bottom), (left, right), (0, 0))
    elif x.ndim == 4:
        pad_width = ((0, 0), (top, bottom), (left, right), (0, 0))
    else:
        raise ValueError(f"Expected image ndim 2/3/4, got {x.ndim}")

    if mode in {"constant", "zeros"}:
        return cp.pad(x, pad_width, mode="constant", constant_values=float(cval))
    if mode in {"edge", "nearest"}:
        return cp.pad(x, pad_width, mode="edge")
    if mode in {"reflect", "reflect_101"}:
        return cp.pad(x, pad_width, mode="reflect")
    if mode == "symmetric":
        return cp.pad(x, pad_width, mode="symmetric")
    if mode == "wrap":
        return cp.pad(x, pad_width, mode="wrap")
    raise ValueError(f"Unsupported pad mode: {mode!r}")


def _restore_dtype(y: cp.ndarray, dtype: object) -> cp.ndarray:
    if cp.issubdtype(dtype, cp.integer):
        info = cp.iinfo(dtype)
        y = cp.floor(y + 0.5)
        y = cp.clip(y, float(info.min), float(info.max))
        return y.astype(dtype)
    if cp.issubdtype(dtype, cp.bool_):
        return y > 0.5
    return y.astype(dtype, copy=False)


def _pool_reduce(
    x: cp.ndarray,
    *,
    block_h: int,
    block_w: int,
    reduce: str,
) -> cp.ndarray:
    if block_h == 0 or block_w == 0:
        return x

    if x.ndim == 2:
        h, w = x.shape
        c = None
        n = None
        x_nhwc = x[:, :, None]
    elif x.ndim == 3:
        h, w, c = x.shape
        n = None
        x_nhwc = x
    elif x.ndim == 4:
        n, h, w, c = x.shape
        x_nhwc = x
    else:
        raise ValueError(f"Expected image ndim 2/3/4, got {x.ndim}")

    if h % block_h != 0 or w % block_w != 0:
        raise RuntimeError("Internal error: expected padded shape divisible by block size.")

    out_h = h // block_h
    out_w = w // block_w

    if n is None:
        # HWC path
        x6 = x_nhwc.reshape(out_h, block_h, out_w, block_w, int(c))
        if reduce == "mean":
            y = cp.mean(x6.astype(cp.float32, copy=False), axis=(1, 3))
        elif reduce == "max":
            y = cp.max(x6, axis=(1, 3))
        elif reduce == "min":
            y = cp.min(x6, axis=(1, 3))
        else:
            raise ValueError(f"Unknown reduce={reduce!r}")
        if c == 1:
            return y[:, :, 0]
        return y

    # NHWC path
    x6 = x_nhwc.reshape(int(n), out_h, block_h, out_w, block_w, int(c))
    if reduce == "mean":
        return cp.mean(x6.astype(cp.float32, copy=False), axis=(2, 4))
    if reduce == "max":
        return cp.max(x6, axis=(2, 4))
    if reduce == "min":
        return cp.min(x6, axis=(2, 4))
    raise ValueError(f"Unknown reduce={reduce!r}")


def avg_pool(
    image: cp.ndarray,
    block_size: int | tuple[int, int],
    *,
    pad_mode: str = "reflect",
    pad_cval: float = 128.0,
    preserve_dtype: bool = True,
) -> cp.ndarray:
    """Apply average pooling to an image on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of shape (H, W), (H, W, C), or (N, H, W, C).
    block_size : int or tuple of int
        Pooling block size. If int, uses square blocks. If tuple, (height, width).
    pad_mode : str, optional
        Padding mode for making image dimensions divisible by block_size.
        Options: 'constant', 'edge', 'reflect', 'symmetric', 'wrap'. Default is 'reflect'.
    pad_cval : float, optional
        Constant value for padding when pad_mode='constant'. Default is 128.0.
    preserve_dtype : bool, optional
        If True, output has same dtype as input. If False, returns float32. Default is True.

    Returns
    -------
    cupy.ndarray
        Pooled image with reduced spatial dimensions.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.
    ValueError
        If image shape is invalid or block_size is invalid.
    """
    img = _require_cupy_array(image, "avg_pool")
    dtype = img.dtype

    bh, bw = _normalize_block_size(block_size)
    if bh == 0 or bw == 0 or img.size == 0:
        return img

    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, _c = img.shape
    elif img.ndim == 4:
        _n, h, w, _c = img.shape
    else:
        raise ValueError(f"avg_pool expects 2D/3D/4D image, got shape {img.shape}")

    top, bottom = _compute_axis_pad(int(h), bh)
    left, right = _compute_axis_pad(int(w), bw)
    x_pad = _pad_hw(
        img, top=top, right=right, bottom=bottom, left=left, mode=pad_mode, cval=pad_cval
    )

    y = _pool_reduce(x_pad, block_h=bh, block_w=bw, reduce="mean")

    if preserve_dtype:
        return _restore_dtype(y, dtype)
    return y


def max_pool(
    image: cp.ndarray,
    block_size: int | tuple[int, int],
    *,
    pad_mode: str = "edge",
    pad_cval: float = 0.0,
    preserve_dtype: bool = True,
) -> cp.ndarray:
    """Apply max pooling to an image on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of shape (H, W), (H, W, C), or (N, H, W, C).
    block_size : int or tuple of int
        Pooling block size. If int, uses square blocks. If tuple, (height, width).
    pad_mode : str, optional
        Padding mode for making image dimensions divisible by block_size.
        Options: 'constant', 'edge', 'reflect', 'symmetric', 'wrap'. Default is 'edge'.
    pad_cval : float, optional
        Constant value for padding when pad_mode='constant'. Default is 0.0.
    preserve_dtype : bool, optional
        If True, output has same dtype as input. If False, returns float32. Default is True.

    Returns
    -------
    cupy.ndarray
        Pooled image with reduced spatial dimensions.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.
    ValueError
        If image shape is invalid or block_size is invalid.
    """
    img = _require_cupy_array(image, "max_pool")
    dtype = img.dtype

    bh, bw = _normalize_block_size(block_size)
    if bh == 0 or bw == 0 or img.size == 0:
        return img

    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, _c = img.shape
    elif img.ndim == 4:
        _n, h, w, _c = img.shape
    else:
        raise ValueError(f"max_pool expects 2D/3D/4D image, got shape {img.shape}")

    top, bottom = _compute_axis_pad(int(h), bh)
    left, right = _compute_axis_pad(int(w), bw)
    x_pad = _pad_hw(
        img, top=top, right=right, bottom=bottom, left=left, mode=pad_mode, cval=pad_cval
    )

    y = _pool_reduce(x_pad, block_h=bh, block_w=bw, reduce="max")

    if preserve_dtype:
        return _restore_dtype(y.astype(cp.float32, copy=False), dtype)
    return y


def min_pool(
    image: cp.ndarray,
    block_size: int | tuple[int, int],
    *,
    pad_mode: str = "edge",
    pad_cval: float = 255.0,
    preserve_dtype: bool = True,
) -> cp.ndarray:
    """Apply min pooling to an image on GPU.

    Parameters
    ----------
    image : cupy.ndarray
        Input image of shape (H, W), (H, W, C), or (N, H, W, C).
    block_size : int or tuple of int
        Pooling block size. If int, uses square blocks. If tuple, (height, width).
    pad_mode : str, optional
        Padding mode for making image dimensions divisible by block_size.
        Options: 'constant', 'edge', 'reflect', 'symmetric', 'wrap'. Default is 'edge'.
    pad_cval : float, optional
        Constant value for padding when pad_mode='constant'. Default is 255.0.
    preserve_dtype : bool, optional
        If True, output has same dtype as input. If False, returns float32. Default is True.

    Returns
    -------
    cupy.ndarray
        Pooled image with reduced spatial dimensions.

    Raises
    ------
    TypeError
        If input is not a cupy.ndarray.
    ValueError
        If image shape is invalid or block_size is invalid.
    """
    img = _require_cupy_array(image, "min_pool")
    dtype = img.dtype

    bh, bw = _normalize_block_size(block_size)
    if bh == 0 or bw == 0 or img.size == 0:
        return img

    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, _c = img.shape
    elif img.ndim == 4:
        _n, h, w, _c = img.shape
    else:
        raise ValueError(f"min_pool expects 2D/3D/4D image, got shape {img.shape}")

    top, bottom = _compute_axis_pad(int(h), bh)
    left, right = _compute_axis_pad(int(w), bw)
    x_pad = _pad_hw(
        img, top=top, right=right, bottom=bottom, left=left, mode=pad_mode, cval=pad_cval
    )

    y = _pool_reduce(x_pad, block_h=bh, block_w=bw, reduce="min")

    if preserve_dtype:
        return _restore_dtype(y.astype(cp.float32, copy=False), dtype)
    return y


__all__ = ["avg_pool", "max_pool", "min_pool"]
