"""
Geometric image operations for the MLX backend.

This module provides GPU-accelerated geometric transformations using Apple's MLX
framework. All functions accept both NumPy arrays and MLX arrays, returning the
same type as input.

Shapes
------
All functions support:
- ``(H, W)`` — grayscale
- ``(H, W, C)`` — single image with channels
- ``(N, H, W, C)`` — batch of images (where applicable)

Dtype Handling
--------------
- Input integer/boolean images are processed in float32 and returned with their
  original dtype preserved, with values clipped to valid ranges.
- Float inputs are returned as float with the original dtype preserved.

Hybrid Operations
-----------------
Some functions can fall back to CPU-based operations for certain parameter
combinations **only when explicitly enabled** via ``allow_cpu_fallback=True``.
This keeps MLX pipelines predictable and avoids silent host↔device roundtrips.

- ``affine_transform()`` / ``perspective_transform()`` use fast MLX kernels for
  ``order`` in {0, 1}. For higher interpolation orders, they can fall back to
  OpenCV warps on the CPU (explicit opt-in).
- ``elastic_transform()`` primarily uses MLX, but may use OpenCV's ``blur`` on
  the CPU for larger smoothing sigmas, and can fall back to SciPy when
  ``order`` not in {0, 1} (explicit opt-in).
- ``piecewise_affine()`` uses scikit-image on CPU to build the transformation,
  then resamples on MLX when ``order`` in {0, 1}; higher orders are CPU-only
  (explicit opt-in).

See Also
--------
imgaug2.augmenters.geometric : High-level augmenter classes using these operations.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

from ._core import ensure_dtype, ensure_float32, is_mlx_array, mx, require, to_mlx, to_numpy
from ._fast_metal import warp_affine, warp_perspective

if TYPE_CHECKING:
    import mlx.core as _mx

    MlxArray: TypeAlias = _mx.array
else:
    MlxArray: TypeAlias = object

NumpyArray: TypeAlias = NDArray[np.generic]


class _PiecewiseTransform(Protocol):
    def inverse(self, coords: np.ndarray) -> np.ndarray: ...


def _require_cpu_fallback_allowed(allow: bool, *, op: str, detail: str) -> None:
    if allow:
        return
    raise NotImplementedError(
        f"{op} {detail} Set allow_cpu_fallback=True to enable a host↔device roundtrip."
    )


def _as_nhwc(image: mx.array) -> tuple[mx.array, bool, bool]:
    """Convert image to NHWC format, tracking which axes were added."""
    if image.ndim == 2:
        return image[None, ..., None], True, True
    if image.ndim == 3:
        return image[None, ...], True, False
    if image.ndim == 4:
        return image, False, False
    raise ValueError(f"Expected image ndim 2/3/4, got {image.ndim}.")


def _as_nhwc_grid(grid: mx.array, batch_size: int) -> mx.array:
    """Convert grid to NHWC format by broadcasting if needed."""
    if grid.ndim == 3:
        return mx.broadcast_to(grid[None, ...], (batch_size, *grid.shape))
    if grid.ndim == 4:
        return grid
    raise ValueError(f"Expected grid ndim 3/4, got {grid.ndim}.")


def _reflect101_coords(x: mx.array, size: int) -> mx.array:
    """Apply BORDER_REFLECT_101 coordinate wrapping (reflect without repeating edge)."""
    if size <= 1:
        return mx.zeros_like(x)

    period = 2 * (size - 1)
    x = mx.abs(x)
    x = mx.remainder(x, period)
    x = mx.where(x > (size - 1), period - x, x)
    return x


def _reflect_symmetric_coords(x: mx.array, size: int) -> mx.array:
    """Apply symmetric reflection coordinate wrapping (NumPy mode='symmetric')."""
    if size <= 1:
        return mx.zeros_like(x)

    return _reflect101_coords(x + 0.5, size + 1) - 0.5


def _wrap_coords(x: mx.array, size: int) -> mx.array:
    """Apply wrap-around coordinate wrapping."""
    if size <= 0:
        return mx.zeros_like(x)

    x = mx.remainder(x, size)
    return mx.where(x < 0, x + size, x)


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


def _geometric_median(points: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    y = np.mean(points, axis=0)
    while True:
        diff = points - y
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        nonzeros = dist != 0
        if not np.any(nonzeros):
            return y
        inv = 1.0 / dist[nonzeros]
        y1 = np.sum(points[nonzeros] * inv[:, None], axis=0) / np.sum(inv)
        if np.linalg.norm(y - y1) < eps:
            return y1
        y = y1


def _elastic_displacement_fields(
    h: int, w: int, *, alpha: float, sigma: float, seed: int | None
) -> tuple[mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    dx = (rng.random((h, w), dtype=np.float32) - 0.5) * (2.0 * float(alpha))
    dy = (rng.random((h, w), dtype=np.float32) - 0.5) * (2.0 * float(alpha))

    dx_mx = mx.array(dx)
    dy_mx = mx.array(dy)

    eps = 1e-3
    if float(sigma) >= eps:
        if float(sigma) < 1.5:
            from .blur import gaussian_blur as mlx_gaussian_blur

            dx_mx = mlx_gaussian_blur(dx_mx, sigma=float(sigma))
            dy_mx = mlx_gaussian_blur(dy_mx, sigma=float(sigma))
        else:
            import cv2

            ksize = int(round(2.0 * float(sigma)))
            ksize = max(1, ksize)
            dx_mx = mx.array(cv2.blur(dx, (ksize, ksize)).astype(np.float32, copy=False))
            dy_mx = mx.array(cv2.blur(dy, (ksize, ksize)).astype(np.float32, copy=False))

    return dx_mx, dy_mx


def _gather_hw(
    img_nhwc: mx.array,
    x_idx: mx.array,
    y_idx: mx.array,
) -> mx.array:
    """Gather pixels from image using 2D coordinate indices."""
    n, h, w, c = img_nhwc.shape

    idx = (y_idx * w + x_idx).astype(mx.int32)
    idx = idx.reshape(n, -1, 1)

    flat = img_nhwc.reshape(n, h * w, c)
    idx = mx.broadcast_to(idx, (n, idx.shape[1], c))
    out = mx.take_along_axis(flat, idx, axis=1)
    return out.reshape(n, *x_idx.shape[1:], c)


def _squeeze_out(out: mx.array, squeeze_batch: bool, squeeze_channel: bool) -> mx.array:
    """Remove batch and/or channel dimensions that were added during processing."""
    if squeeze_batch:
        out = out[0]
    if squeeze_channel:
        out = out[..., 0]
    return out


def grid_sample(
    image: object,
    grid: mx.array,
    *,
    mode: Literal["bilinear", "nearest"] = "bilinear",
    padding_mode: Literal["zeros", "border", "reflection", "symmetric", "wrap"] = "zeros",
    cval: float = 0.0,
) -> mx.array:
    """
    Sample from image at normalized grid coordinates.

    Uses align_corners=True convention where grid coordinates in [-1, 1] map to
    pixel coordinates via: ``x_pix = (x_norm + 1) * (W - 1) / 2``.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape (H, W), (H, W, C),
        or (N, H, W, C).
    grid : mx.array
        Sampling coordinates in normalized space [-1, 1]. Shape (Hout, Wout, 2)
        or (N, Hout, Wout, 2). Last dimension is (x, y).
    mode : {"bilinear", "nearest"}, default "bilinear"
        Interpolation mode.
    padding_mode : {"zeros", "border", "reflection", "symmetric", "wrap"}, default "zeros"
        How to handle out-of-bounds coordinates.
    cval : float, default 0.0
        Fill value for "zeros" padding mode.

    Returns
    -------
    mx.array
        Sampled image with shape matching grid dimensions.

    Raises
    ------
    ValueError
        If grid shape is invalid or mode is unknown.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    img_mx = ensure_float32(to_mlx(image))
    img_nhwc, squeeze_batch, squeeze_channel = _as_nhwc(img_mx)
    n, h, w, c = img_nhwc.shape

    g = _as_nhwc_grid(grid, n)
    if g.shape[-1] != 2:
        raise ValueError("grid last dimension must be 2 (x,y).")

    if h == 0 or w == 0:
        out = mx.zeros((n, g.shape[1], g.shape[2], c), dtype=img_nhwc.dtype)
        return _squeeze_out(out, squeeze_batch, squeeze_channel)

    gx = g[..., 0].astype(mx.float32)
    gy = g[..., 1].astype(mx.float32)

    x = (gx + 1.0) * 0.5 * float(w - 1)
    y = (gy + 1.0) * 0.5 * float(h - 1)

    if padding_mode == "border":
        x = mx.clip(x, 0.0, float(w - 1))
        y = mx.clip(y, 0.0, float(h - 1))
    elif padding_mode == "reflection":
        x = _reflect101_coords(x, w)
        y = _reflect101_coords(y, h)
    elif padding_mode == "symmetric":
        x = _reflect_symmetric_coords(x, w)
        y = _reflect_symmetric_coords(y, h)
    elif padding_mode == "wrap":
        x = _wrap_coords(x, w)
        y = _wrap_coords(y, h)
    elif padding_mode != "zeros":
        raise ValueError(f"Unknown padding_mode={padding_mode!r}.")

    if mode == "nearest":
        xn = mx.round(x).astype(mx.int32)
        yn = mx.round(y).astype(mx.int32)

        if padding_mode == "wrap":
            xn = mx.remainder(xn, w)
            yn = mx.remainder(yn, h)
            xn = mx.where(xn < 0, xn + w, xn)
            yn = mx.where(yn < 0, yn + h, yn)

        if padding_mode == "zeros":
            inb = (xn >= 0) & (xn < w) & (yn >= 0) & (yn < h)
            xn_c = mx.clip(xn, 0, w - 1)
            yn_c = mx.clip(yn, 0, h - 1)
            out = _gather_hw(img_nhwc, xn_c, yn_c)
            out = mx.where(inb[..., None], out, mx.array(cval, dtype=out.dtype))
        else:
            if padding_mode != "wrap":
                xn = mx.clip(xn, 0, w - 1)
                yn = mx.clip(yn, 0, h - 1)
            out = _gather_hw(img_nhwc, xn, yn)

        return _squeeze_out(out, squeeze_batch, squeeze_channel)

    if mode != "bilinear":
        raise ValueError(f"Unknown mode={mode!r}.")

    x0 = mx.floor(x).astype(mx.int32)
    y0 = mx.floor(y).astype(mx.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0f = x0.astype(mx.float32)
    y0f = y0.astype(mx.float32)
    wx = (x - x0f).astype(mx.float32)
    wy = (y - y0f).astype(mx.float32)

    if padding_mode == "wrap":
        x0 = mx.remainder(x0, w)
        x1 = mx.remainder(x1, w)
        y0 = mx.remainder(y0, h)
        y1 = mx.remainder(y1, h)

        x0 = mx.where(x0 < 0, x0 + w, x0)
        x1 = mx.where(x1 < 0, x1 + w, x1)
        y0 = mx.where(y0 < 0, y0 + h, y0)
        y1 = mx.where(y1 < 0, y1 + h, y1)

        v00 = _gather_hw(img_nhwc, x0, y0)
        v01 = _gather_hw(img_nhwc, x1, y0)
        v10 = _gather_hw(img_nhwc, x0, y1)
        v11 = _gather_hw(img_nhwc, x1, y1)
    elif padding_mode == "zeros":
        m00 = (x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h)
        m01 = (x1 >= 0) & (x1 < w) & (y0 >= 0) & (y0 < h)
        m10 = (x0 >= 0) & (x0 < w) & (y1 >= 0) & (y1 < h)
        m11 = (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h)

        x0c = mx.clip(x0, 0, w - 1)
        x1c = mx.clip(x1, 0, w - 1)
        y0c = mx.clip(y0, 0, h - 1)
        y1c = mx.clip(y1, 0, h - 1)

        v00 = _gather_hw(img_nhwc, x0c, y0c)
        v01 = _gather_hw(img_nhwc, x1c, y0c)
        v10 = _gather_hw(img_nhwc, x0c, y1c)
        v11 = _gather_hw(img_nhwc, x1c, y1c)

        zero = mx.array(cval, dtype=v00.dtype)
        v00 = mx.where(m00[..., None], v00, zero)
        v01 = mx.where(m01[..., None], v01, zero)
        v10 = mx.where(m10[..., None], v10, zero)
        v11 = mx.where(m11[..., None], v11, zero)
    else:
        x0c = mx.clip(x0, 0, w - 1)
        x1c = mx.clip(x1, 0, w - 1)
        y0c = mx.clip(y0, 0, h - 1)
        y1c = mx.clip(y1, 0, h - 1)

        v00 = _gather_hw(img_nhwc, x0c, y0c)
        v01 = _gather_hw(img_nhwc, x1c, y0c)
        v10 = _gather_hw(img_nhwc, x0c, y1c)
        v11 = _gather_hw(img_nhwc, x1c, y1c)

    wx = wx[..., None]
    wy = wy[..., None]
    out = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v01
        + (1.0 - wx) * wy * v10
        + wx * wy * v11
    )
    return _squeeze_out(out, squeeze_batch, squeeze_channel)


def _mode_to_padding(mode: str) -> Literal["zeros", "border", "reflection", "symmetric", "wrap"]:
    """Convert string mode to canonical padding mode name."""
    mode = str(mode).lower()
    if mode in {"constant", "zeros"}:
        return "zeros"
    if mode in {"edge", "nearest", "replicate", "border"}:
        return "border"
    if mode in {"reflect", "reflect_101", "mirror"}:
        return "reflection"
    if mode in {"symmetric"}:
        return "symmetric"
    if mode in {"wrap"}:
        return "wrap"
    raise ValueError(f"Unsupported mode={mode!r} for MLX warps.")


def _restore_dtype(result: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert float32 result back to original dtype with proper clipping."""
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return result > 0.5
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(result), info.min, info.max).astype(dtype)
    return result.astype(dtype, copy=False)


def _hw_grid_mx(h: int, w: int) -> tuple[mx.array, mx.array]:
    """Create HxW coordinate grids (y, x) in float32."""
    yy, xx = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
    return yy.astype(mx.float32), xx.astype(mx.float32)


@lru_cache(maxsize=32)
def _coords_homogeneous_mx(h: int, w: int) -> mx.array:
    """
    Create homogeneous pixel coordinates (3, H*W) for transformation matrices.

    Returns
    -------
    mx.array
        Shape (3, H*W) containing [x, y, 1] coordinates for each pixel.
    """
    yy, xx = _hw_grid_mx(h, w)
    ones = mx.ones_like(xx)
    coords = mx.stack([xx.reshape(-1), yy.reshape(-1), ones.reshape(-1)], axis=0)
    return coords.astype(mx.float32)


@lru_cache(maxsize=512)
def _inv_affine_flat6_mx(mat_bytes: bytes, shape: tuple[int, int]) -> mx.array:
    """Compute and cache inverse affine matrix (flattened 2x3)."""
    mat = np.frombuffer(mat_bytes, dtype=np.float32).reshape(shape)
    if mat.shape == (2, 3):
        m3 = np.eye(3, dtype=np.float32)
        m3[:2, :] = mat
    elif mat.shape == (3, 3):
        m3 = mat
    else:
        raise ValueError(f"Expected (2,3) or (3,3), got {mat.shape}")
    inv = np.linalg.inv(m3).astype(np.float32, copy=False)
    inv2 = inv[:2, :].reshape(-1)
    return mx.array(inv2)


@lru_cache(maxsize=512)
def _inv_persp_flat9_mx(mat_bytes: bytes, shape: tuple[int, int]) -> mx.array:
    """Compute and cache inverse perspective matrix (flattened 3x3)."""
    mat = np.frombuffer(mat_bytes, dtype=np.float32).reshape(shape)
    if mat.shape != (3, 3):
        raise ValueError(f"Expected (3,3), got {mat.shape}")
    inv = np.linalg.inv(mat).astype(np.float32, copy=False).reshape(-1)
    return mx.array(inv)


@overload
def affine_transform(
    image: NumpyArray,
    matrix: np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    order: int = 1,
    cval: float = 0.0,
    mode: str = "constant",
    allow_cpu_fallback: bool = False,
) -> NumpyArray: ...


@overload
def affine_transform(
    image: MlxArray,
    matrix: np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    order: int = 1,
    cval: float = 0.0,
    mode: str = "constant",
    allow_cpu_fallback: bool = False,
) -> MlxArray: ...


def affine_transform(
    image: object,
    matrix: np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    order: int = 1,
    cval: float = 0.0,
    mode: str = "constant",
    allow_cpu_fallback: bool = False,
) -> object:
    """
    Apply an affine transformation to an image.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape (H, W), (H, W, C),
        or (N, H, W, C).
    matrix : np.ndarray
        Affine transformation matrix. Shape (2, 3) or (3, 3).
    output_shape : tuple of int, optional
        Output image shape (H, W). If None, uses input shape.
    order : int, default 1
        Interpolation order. 0=nearest, 1=bilinear. Higher orders require
        ``allow_cpu_fallback=True`` for CPU execution.
    cval : float, default 0.0
        Fill value for constant mode.
    mode : str, default "constant"
        Border mode ("constant", "edge", "reflect", "symmetric", "wrap").
    allow_cpu_fallback : bool, default False
        If True, allow CPU fallback for unsupported interpolation orders. This
        performs a host↔device roundtrip for MLX inputs.

    Returns
    -------
    array-like
        Transformed image. Same type as input (NumPy or MLX).

    Raises
    ------
    ValueError
        If matrix or image shape is invalid.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    Notes
    -----
    For ``order`` > 1, this function only falls back to OpenCV on the CPU when
    ``allow_cpu_fallback=True``.
    """
    require()

    is_input_mlx = is_mlx_array(image)

    if order not in (0, 1):
        if is_input_mlx:
            _require_cpu_fallback_allowed(
                allow_cpu_fallback,
                op="affine_transform:",
                detail="MLX supports only interpolation orders 0/1.",
            )

        import cv2

        img_np = to_numpy(image)
        if img_np.size == 0:
            out_np = img_np.copy()
            return to_mlx(out_np) if is_input_mlx else out_np

        input_dtype = img_np.dtype
        if img_np.dtype.kind == "b":
            img_np = img_np.astype(np.float32, copy=False)
        elif img_np.dtype == np.float16:
            img_np = img_np.astype(np.float32, copy=False)

        mat = np.asarray(matrix, dtype=np.float32)
        if mat.shape == (3, 3):
            mat = mat[:2]
        elif mat.shape != (2, 3):
            raise ValueError(f"Expected matrix shape (2,3) or (3,3), got {mat.shape}.")

        h_out, w_out = output_shape or img_np.shape[:2]
        dsize = (int(w_out), int(h_out))

        border_map = {
            "constant": cv2.BORDER_CONSTANT,
            "edge": cv2.BORDER_REPLICATE,
            "nearest": cv2.BORDER_REPLICATE,
            "symmetric": cv2.BORDER_REFLECT,
            "reflect": cv2.BORDER_REFLECT_101,
            "reflect_101": cv2.BORDER_REFLECT_101,
            "wrap": cv2.BORDER_WRAP,
        }
        border_mode = border_map.get(str(mode).lower(), cv2.BORDER_CONSTANT)

        interp = cv2.INTER_NEAREST if int(order) == 0 else cv2.INTER_CUBIC
        cval_cv = float(cval) if img_np.dtype.kind == "f" else int(cval)

        out_np = cv2.warpAffine(
            img_np,
            mat,
            dsize=dsize,
            flags=interp,
            borderMode=border_mode,
            borderValue=cval_cv,
        )
        if input_dtype.kind == "b":
            out_np = out_np > 0.5
        elif out_np.dtype != input_dtype:
            out_np = _restore_dtype(out_np.astype(np.float32, copy=False), input_dtype)

        return to_mlx(out_np) if is_input_mlx else out_np
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mx = ensure_float32(to_mlx(image))
    if img_mx.size == 0:
        return image

    if img_mx.ndim == 2:
        h_in, w_in = int(img_mx.shape[0]), int(img_mx.shape[1])
    elif img_mx.ndim == 3:
        h_in, w_in = int(img_mx.shape[0]), int(img_mx.shape[1])
    elif img_mx.ndim == 4:
        h_in, w_in = int(img_mx.shape[1]), int(img_mx.shape[2])
    else:
        raise ValueError(
            "affine_transform expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mx.shape)}."
        )

    h_out, w_out = output_shape or (h_in, w_in)

    mat = np.asarray(matrix, dtype=np.float32)
    if mat.shape == (2, 3):
        mat3 = np.eye(3, dtype=np.float32)
        mat3[:2, :] = mat
        mat = mat3
    elif mat.shape != (3, 3):
        raise ValueError(f"Expected matrix shape (2,3) or (3,3), got {mat.shape}.")

    pad_mode_str = _mode_to_padding(mode)
    if order in (0, 1) and pad_mode_str in {"zeros", "border", "reflection"}:
        cval_scalar: float | None
        if pad_mode_str == "zeros":
            if isinstance(cval, (list, tuple, np.ndarray)):
                cval_arr = np.array(cval, dtype=np.float32).reshape(-1)
                if cval_arr.size == 0:
                    cval_scalar = 0.0
                elif np.all(cval_arr == cval_arr.flat[0]):
                    cval_scalar = float(cval_arr.flat[0])
                else:
                    cval_scalar = None
            else:
                cval_scalar = float(cval)
            if cval_scalar is None:
                pad_mode_str = "skip"
        else:
            cval_scalar = 0.0

    if order in (0, 1) and pad_mode_str in {"zeros", "border", "reflection"}:
        img_nhwc, squeeze_batch, squeeze_channel = _as_nhwc(img_mx)
        if img_nhwc.size == 0:
            return image

        mat3 = mat.astype(np.float32, copy=False)
        inv6 = _inv_affine_flat6_mx(mat3.tobytes(), mat3.shape)
        pad_mode_i = {"zeros": 0, "border": 1, "reflection": 2}[pad_mode_str]
        interp_mode = 0 if int(order) == 0 else 1
        out_nhwc = warp_affine(
            img_nhwc,
            inv6,
            out_h=int(h_out),
            out_w=int(w_out),
            pad_mode=pad_mode_i,
            interp_mode=interp_mode,
            cval=float(cval_scalar),
        )
        out_mx = _squeeze_out(out_nhwc, squeeze_batch, squeeze_channel)

        if is_input_mlx:
            return out_mx

        out_np = to_numpy(out_mx)
        return _restore_dtype(out_np, ensure_dtype(original_dtype))

    coords = _coords_homogeneous_mx(int(h_out), int(w_out))  # (3, HW)

    inv = mx.array(np.linalg.inv(mat).astype(np.float32))
    src = inv @ coords
    sx = src[0].reshape(int(h_out), int(w_out))
    sy = src[1].reshape(int(h_out), int(w_out))

    if w_in > 1:
        gx = (sx / float(w_in - 1)) * 2.0 - 1.0
    else:
        gx = mx.zeros_like(sx)
    if h_in > 1:
        gy = (sy / float(h_in - 1)) * 2.0 - 1.0
    else:
        gy = mx.zeros_like(sy)

    grid = mx.stack([gx, gy], axis=-1)
    out_mx = grid_sample(
        img_mx,
        grid,
        mode="nearest" if order == 0 else "bilinear",
        padding_mode=_mode_to_padding(mode),
        cval=cval,
    )

    if is_input_mlx:
        return out_mx

    out_np = to_numpy(out_mx)
    return _restore_dtype(out_np, ensure_dtype(original_dtype))


@overload
def perspective_transform(
    image: NumpyArray,
    matrix: np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    order: int = 1,
    cval: float = 0.0,
    mode: str = "constant",
    allow_cpu_fallback: bool = False,
) -> NumpyArray: ...


@overload
def perspective_transform(
    image: MlxArray,
    matrix: np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    order: int = 1,
    cval: float = 0.0,
    mode: str = "constant",
    allow_cpu_fallback: bool = False,
) -> MlxArray: ...


def perspective_transform(
    image: object,
    matrix: np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    order: int = 1,
    cval: float = 0.0,
    mode: str = "constant",
    allow_cpu_fallback: bool = False,
) -> object:
    """
    Apply a perspective (projective) transformation to an image.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape (H, W), (H, W, C),
        or (N, H, W, C).
    matrix : np.ndarray
        Perspective transformation matrix. Shape (3, 3).
    output_shape : tuple of int, optional
        Output image shape (H, W). If None, uses input shape.
    order : int, default 1
        Interpolation order. 0=nearest, 1=bilinear. Higher orders require
        ``allow_cpu_fallback=True`` for CPU execution.
    cval : float, default 0.0
        Fill value for constant mode.
    mode : str, default "constant"
        Border mode ("constant", "edge", "reflect", "symmetric", "wrap").
    allow_cpu_fallback : bool, default False
        If True, allow CPU fallback for unsupported interpolation orders. This
        performs a host↔device roundtrip for MLX inputs.

    Returns
    -------
    array-like
        Transformed image. Same type as input (NumPy or MLX).

    Raises
    ------
    ValueError
        If matrix or image shape is invalid.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    Notes
    -----
    For ``order`` > 1, this function only falls back to OpenCV on the CPU when
    ``allow_cpu_fallback=True``.
    """
    require()

    is_input_mlx = is_mlx_array(image)

    if order not in (0, 1):
        if is_input_mlx:
            _require_cpu_fallback_allowed(
                allow_cpu_fallback,
                op="perspective_transform:",
                detail="MLX supports only interpolation orders 0/1.",
            )

        import cv2

        img_np = to_numpy(image)
        if img_np.size == 0:
            out_np = img_np.copy()
            return to_mlx(out_np) if is_input_mlx else out_np

        input_dtype = img_np.dtype
        if img_np.dtype.kind == "b":
            img_np = img_np.astype(np.float32, copy=False)
        elif img_np.dtype == np.float16:
            img_np = img_np.astype(np.float32, copy=False)

        mat = np.asarray(matrix, dtype=np.float32)
        if mat.shape != (3, 3):
            raise ValueError(f"Expected matrix shape (3,3), got {mat.shape}.")

        h_out, w_out = output_shape or img_np.shape[:2]
        dsize = (int(w_out), int(h_out))

        border_map = {
            "constant": cv2.BORDER_CONSTANT,
            "edge": cv2.BORDER_REPLICATE,
            "nearest": cv2.BORDER_REPLICATE,
            "symmetric": cv2.BORDER_REFLECT,
            "reflect": cv2.BORDER_REFLECT_101,
            "reflect_101": cv2.BORDER_REFLECT_101,
            "wrap": cv2.BORDER_WRAP,
        }
        border_mode = border_map.get(str(mode).lower(), cv2.BORDER_CONSTANT)

        interp = cv2.INTER_NEAREST if int(order) == 0 else cv2.INTER_CUBIC
        cval_cv = float(cval) if img_np.dtype.kind == "f" else int(cval)

        out_np = cv2.warpPerspective(
            img_np,
            mat,
            dsize=dsize,
            flags=interp,
            borderMode=border_mode,
            borderValue=cval_cv,
        )
        if input_dtype.kind == "b":
            out_np = out_np > 0.5
        elif out_np.dtype != input_dtype:
            out_np = _restore_dtype(out_np.astype(np.float32, copy=False), input_dtype)
        return to_mlx(out_np) if is_input_mlx else out_np
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mx = ensure_float32(to_mlx(image))
    if img_mx.size == 0:
        return image

    if img_mx.ndim == 2:
        h_in, w_in = int(img_mx.shape[0]), int(img_mx.shape[1])
    elif img_mx.ndim == 3:
        h_in, w_in = int(img_mx.shape[0]), int(img_mx.shape[1])
    elif img_mx.ndim == 4:
        h_in, w_in = int(img_mx.shape[1]), int(img_mx.shape[2])
    else:
        raise ValueError(
            "perspective_transform expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mx.shape)}."
        )

    h_out, w_out = output_shape or (h_in, w_in)

    mat = np.asarray(matrix, dtype=np.float32)
    if mat.shape != (3, 3):
        raise ValueError(f"Expected matrix shape (3,3), got {mat.shape}.")

    pad_mode_str = _mode_to_padding(mode)
    if order in (0, 1) and pad_mode_str in {"zeros", "border", "reflection"}:
        cval_scalar: float | None
        if pad_mode_str == "zeros":
            if isinstance(cval, (list, tuple, np.ndarray)):
                cval_arr = np.array(cval, dtype=np.float32).reshape(-1)
                if cval_arr.size == 0:
                    cval_scalar = 0.0
                elif np.all(cval_arr == cval_arr.flat[0]):
                    cval_scalar = float(cval_arr.flat[0])
                else:
                    cval_scalar = None
            else:
                cval_scalar = float(cval)
            if cval_scalar is None:
                pad_mode_str = "skip"
        else:
            cval_scalar = 0.0

    if order in (0, 1) and pad_mode_str in {"zeros", "border", "reflection"}:
        img_nhwc, squeeze_batch, squeeze_channel = _as_nhwc(img_mx)
        if img_nhwc.size == 0:
            return image

        mat3 = mat.astype(np.float32, copy=False)
        inv9 = _inv_persp_flat9_mx(mat3.tobytes(), mat3.shape)
        pad_mode_i = {"zeros": 0, "border": 1, "reflection": 2}[pad_mode_str]
        interp_mode = 0 if int(order) == 0 else 1
        out_nhwc = warp_perspective(
            img_nhwc,
            inv9,
            out_h=int(h_out),
            out_w=int(w_out),
            pad_mode=pad_mode_i,
            interp_mode=interp_mode,
            cval=float(cval_scalar),
        )
        out_mx = _squeeze_out(out_nhwc, squeeze_batch, squeeze_channel)

        if is_input_mlx:
            return out_mx

        out_np = to_numpy(out_mx)
        return _restore_dtype(out_np, ensure_dtype(original_dtype))

    coords = _coords_homogeneous_mx(int(h_out), int(w_out))  # (3, HW)

    inv = mx.array(np.linalg.inv(mat).astype(np.float32))
    src = inv @ coords
    z = src[2]
    z_safe = mx.where(z == 0, mx.ones_like(z), z)
    sx = (src[0] / z_safe).reshape(int(h_out), int(w_out))
    sy = (src[1] / z_safe).reshape(int(h_out), int(w_out))

    if w_in > 1:
        gx = (sx / float(w_in - 1)) * 2.0 - 1.0
    else:
        gx = mx.zeros_like(sx)
    if h_in > 1:
        gy = (sy / float(h_in - 1)) * 2.0 - 1.0
    else:
        gy = mx.zeros_like(sy)

    grid = mx.stack([gx, gy], axis=-1)
    out_mx = grid_sample(
        img_mx,
        grid,
        mode="nearest" if order == 0 else "bilinear",
        padding_mode=_mode_to_padding(mode),
        cval=cval,
    )

    if is_input_mlx:
        return out_mx

    out_np = to_numpy(out_mx)
    return _restore_dtype(out_np, ensure_dtype(original_dtype))


@overload
def affine_points(points: NumpyArray, matrix: np.ndarray) -> NumpyArray: ...


@overload
def affine_points(points: MlxArray, matrix: np.ndarray) -> MlxArray: ...


def affine_points(points: object, matrix: np.ndarray) -> object:
    """Apply an affine transformation to point coordinates."""
    require()
    is_input_mlx = is_mlx_array(points)
    pts = to_mlx(points)
    pts, squeeze = _as_points(pts)

    pts_f = pts if mx.issubdtype(pts.dtype, mx.floating) else pts.astype(mx.float32)

    mat = np.asarray(matrix, dtype=np.float32)
    if mat.shape == (2, 3):
        mat3 = np.eye(3, dtype=np.float32)
        mat3[:2, :] = mat
        mat = mat3
    elif mat.shape != (3, 3):
        raise ValueError(f"Expected matrix shape (2,3) or (3,3), got {mat.shape}.")

    mat_mx = mx.array(mat)
    ones = mx.ones((int(pts_f.shape[0]), 1), dtype=mx.float32)
    coords = mx.concatenate([pts_f, ones], axis=1)
    out = mx.matmul(coords, mx.transpose(mat_mx))
    out_xy = out[:, :2]

    if squeeze:
        out_xy = out_xy[0]
    return out_xy if is_input_mlx else to_numpy(out_xy)


@overload
def perspective_points(points: NumpyArray, matrix: np.ndarray) -> NumpyArray: ...


@overload
def perspective_points(points: MlxArray, matrix: np.ndarray) -> MlxArray: ...


def perspective_points(points: object, matrix: np.ndarray) -> object:
    """Apply a perspective transformation to point coordinates."""
    require()
    is_input_mlx = is_mlx_array(points)
    pts = to_mlx(points)
    pts, squeeze = _as_points(pts)

    pts_f = pts if mx.issubdtype(pts.dtype, mx.floating) else pts.astype(mx.float32)

    mat = np.asarray(matrix, dtype=np.float32)
    if mat.shape != (3, 3):
        raise ValueError(f"Expected matrix shape (3,3), got {mat.shape}.")

    mat_mx = mx.array(mat)
    ones = mx.ones((int(pts_f.shape[0]), 1), dtype=mx.float32)
    coords = mx.concatenate([pts_f, ones], axis=1)
    out = mx.matmul(coords, mx.transpose(mat_mx))
    z = out[:, 2]
    z_safe = mx.where(z == 0, mx.ones_like(z), z)
    out_xy = mx.stack([out[:, 0] / z_safe, out[:, 1] / z_safe], axis=1)

    if squeeze:
        out_xy = out_xy[0]
    return out_xy if is_input_mlx else to_numpy(out_xy)


@overload
def elastic_points(
    points: NumpyArray,
    shape: tuple[int, int],
    *,
    alpha: float,
    sigma: float,
    seed: int | None = None,
    nb_steps: int = 3,
    step_size: float = 1.0,
    iterations: int = 3,
    alpha_thresh: float = 0.05,
    sigma_thresh: float = 1.0,
) -> NumpyArray: ...


@overload
def elastic_points(
    points: MlxArray,
    shape: tuple[int, int],
    *,
    alpha: float,
    sigma: float,
    seed: int | None = None,
    nb_steps: int = 3,
    step_size: float = 1.0,
    iterations: int = 3,
    alpha_thresh: float = 0.05,
    sigma_thresh: float = 1.0,
) -> MlxArray: ...


def elastic_points(
    points: object,
    shape: tuple[int, int],
    *,
    alpha: float,
    sigma: float,
    seed: int | None = None,
    nb_steps: int = 3,
    step_size: float = 1.0,
    iterations: int = 3,
    alpha_thresh: float = 0.05,
    sigma_thresh: float = 1.0,
) -> object:
    """Apply elastic deformation to point coordinates.

    This matches the keypoint path of ``imgaug2.augmenters.geometry.elastic``.
    """
    require()
    is_input_mlx = is_mlx_array(points)
    pts_np = to_numpy(points)

    if pts_np.ndim == 1:
        if pts_np.shape[0] != 2:
            raise ValueError(
                "Expected points shape (2,) or (N,2); got "
                f"{tuple(pts_np.shape)!r}."
            )
        pts_np = pts_np[None, :]
        squeeze = True
    elif pts_np.ndim == 2 and pts_np.shape[1] == 2:
        squeeze = False
    else:
        raise ValueError(f"Expected points shape (N,2); got {tuple(pts_np.shape)!r}.")

    if pts_np.size == 0:
        return points

    dtype_out = pts_np.dtype if pts_np.dtype.kind == "f" else np.float32
    pts_np = pts_np.astype(dtype_out, copy=False)

    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        return points

    if float(alpha) <= float(alpha_thresh) or float(sigma) <= float(sigma_thresh):
        out = pts_np if not squeeze else pts_np[0]
        return to_mlx(out) if is_input_mlx else out

    dx_mx, dy_mx = _elastic_displacement_fields(
        h,
        w,
        alpha=float(alpha),
        sigma=float(sigma),
        seed=seed,
    )
    dx = to_numpy(dx_mx)
    dy = to_numpy(dy_mx)

    out_pts = pts_np.copy()
    for idx, (x0, y0) in enumerate(out_pts):
        if not (0.0 <= x0 < w and 0.0 <= y0 < h):
            continue

        yy = np.linspace(y0 - nb_steps * step_size, y0 + nb_steps * step_size, nb_steps * 2 + 1)
        width = 1
        neighbors = []
        for i_y, y in enumerate(yy):
            if width == 1:
                xx = [x0]
            else:
                xx = np.linspace(
                    x0 - (width - 1) // 2 * step_size,
                    x0 + (width - 1) // 2 * step_size,
                    width,
                )
            for x in xx:
                neighbors.append((x, y))
            if i_y < nb_steps:
                width += 2
            else:
                width -= 2

        neigh = np.array(neighbors, dtype=np.float32)
        xx = np.round(neigh[:, 0]).astype(np.int32)
        yy = np.round(neigh[:, 1]).astype(np.int32)
        inside = (0 <= xx) & (xx < w) & (0 <= yy) & (yy < h)
        xx = xx[inside]
        yy = yy[inside]
        if xx.size == 0:
            continue

        x_in = xx.astype(np.float32)
        y_in = yy.astype(np.float32)
        x_out = x_in
        y_out = y_in
        for _ in range(int(iterations)):
            xx_out = np.clip(np.round(x_out).astype(np.int32), 0, w - 1)
            yy_out = np.clip(np.round(y_out).astype(np.int32), 0, h - 1)
            x_out = x_in + dx[yy_out, xx_out]
            y_out = y_in + dy[yy_out, xx_out]

        med = _geometric_median(np.stack([x_out, y_out], axis=1))
        out_pts[idx, 0] = med[0]
        out_pts[idx, 1] = med[1]

    out_final = out_pts if not squeeze else out_pts[0]
    return to_mlx(out_final) if is_input_mlx else out_final


def elastic_transform(
    image: object,
    alpha: float,
    sigma: float,
    seed: int | None = None,
    *,
    order: int = 1,
    cval: float = 0.0,
    mode: str = "constant",
    allow_cpu_fallback: bool = False,
) -> object:
    """
    Apply elastic deformation to an image.

    Creates a random displacement field, smooths it with Gaussian blur, then
    resamples the image using the displacement field.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape (H, W) or (H, W, C).
    alpha : float
        Displacement magnitude. Larger values produce stronger distortion.
    sigma : float
        Gaussian smoothing sigma for the displacement field. Larger values
        produce smoother, more coherent deformations.
    seed : int, optional
        Random seed for reproducibility.
    order : int, default 1
        Interpolation order. 0=nearest, 1=bilinear. Higher orders require
        ``allow_cpu_fallback=True`` for CPU execution.
    cval : float, default 0.0
        Fill value for constant mode.
    mode : str, default "constant"
        Border mode ("constant", "edge", "reflect", "symmetric", "wrap").
    allow_cpu_fallback : bool, default False
        If True, allow CPU fallback for unsupported interpolation orders. This
        performs a host↔device roundtrip for MLX inputs.

    Returns
    -------
    array-like
        Elastically deformed image. Same type as input (NumPy or MLX).

    Raises
    ------
    ValueError
        If image shape is invalid.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    Notes
    -----
    For large ``sigma`` values, uses OpenCV's blur on CPU. For ``order`` > 1,
    falls back to SciPy on CPU **only** when ``allow_cpu_fallback=True``.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    if is_input_mlx:
        img_mx = to_mlx(image)
        if img_mx.size == 0:
            return img_mx

        if img_mx.ndim not in (2, 3):
            raise ValueError(
                f"elastic_transform expects (H, W) or (H, W, C), got shape {tuple(img_mx.shape)}."
            )

        if float(alpha) == 0.0:
            return img_mx

        original_dtype = None
        h, w = int(img_mx.shape[0]), int(img_mx.shape[1])
    else:
        img_np = to_numpy(image)
        if img_np.size == 0:
            return img_np.copy()

        if img_np.ndim not in (2, 3):
            raise ValueError(
                f"elastic_transform expects (H, W) or (H, W, C), got shape {img_np.shape}."
            )

        if float(alpha) == 0.0:
            return img_np.copy()

        original_dtype = img_np.dtype
        h, w = img_np.shape[:2]

    dx_mx, dy_mx = _elastic_displacement_fields(
        h,
        w,
        alpha=float(alpha),
        sigma=float(sigma),
        seed=seed,
    )

    yy, xx = _hw_grid_mx(h, w)
    x_shifted = xx - dx_mx
    y_shifted = yy - dy_mx

    if w > 1:
        gx = (x_shifted / float(w - 1)) * 2.0 - 1.0
    else:
        gx = mx.zeros_like(x_shifted)
    if h > 1:
        gy = (y_shifted / float(h - 1)) * 2.0 - 1.0
    else:
        gy = mx.zeros_like(y_shifted)

    grid = mx.stack([gx, gy], axis=-1)

    if order not in (0, 1):
        if is_input_mlx:
            _require_cpu_fallback_allowed(
                allow_cpu_fallback,
                op="elastic_transform:",
                detail="MLX supports only interpolation orders 0/1.",
            )
        from scipy import ndimage

        x_shifted_np = np.array(x_shifted).astype(np.float32, copy=False)
        y_shifted_np = np.array(y_shifted).astype(np.float32, copy=False)

        img_np = to_numpy(image)
        img_np3 = img_np if img_np.ndim == 3 else img_np[..., None]
        out_np = np.empty_like(img_np3)
        for ch in range(img_np3.shape[2]):
            remapped = ndimage.map_coordinates(
                img_np3[..., ch],
                (y_shifted_np.ravel(), x_shifted_np.ravel()),
                order=int(order),
                mode=str(mode),
                cval=float(cval),
            ).reshape(h, w)
            out_np[..., ch] = remapped

        if img_np.ndim == 2:
            out_np = out_np[..., 0]
        out_np = _restore_dtype(out_np.astype(np.float32, copy=False), img_np.dtype)
        return to_mlx(out_np) if is_input_mlx else out_np

    out_mx = grid_sample(
        image,
        grid,
        mode="nearest" if order == 0 else "bilinear",
        padding_mode=_mode_to_padding(mode),
        cval=cval,
    )

    if is_input_mlx:
        return out_mx

    out_np = to_numpy(out_mx)
    return _restore_dtype(out_np, ensure_dtype(original_dtype))


def _build_piecewise_tform(
    hw: tuple[int, int],
    scale: float,
    nb_rows: int,
    nb_cols: int,
    seed: int | None,
    absolute_scale: bool,
) -> _PiecewiseTransform | None:
    """Build a piecewise affine transformation from jittered control points."""
    from skimage import transform as tf

    h, w = hw
    y = np.linspace(0, h, nb_rows)
    x = np.linspace(0, w, nb_cols)
    xx_src, yy_src = np.meshgrid(x, y)
    points_src_yx = np.dstack([yy_src.flat, xx_src.flat])[0].astype(np.float32)

    rng = np.random.default_rng(seed)
    jitter = rng.normal(loc=0.0, scale=float(scale), size=points_src_yx.shape).astype(np.float32)

    if absolute_scale:
        if h > 0:
            jitter[:, 0] = jitter[:, 0] / float(h)
        else:
            jitter[:, 0] = 0.0
        if w > 0:
            jitter[:, 1] = jitter[:, 1] / float(w)
        else:
            jitter[:, 1] = 0.0

    jitter[:, 0] = jitter[:, 0] * float(h)
    jitter[:, 1] = jitter[:, 1] * float(w)

    if not np.any(jitter != 0):
        return None

    points_dest_yx = points_src_yx + jitter
    points_dest_yx[:, 0] = np.clip(points_dest_yx[:, 0], 0.0, max(float(h - 1), 0.0))
    points_dest_yx[:, 1] = np.clip(points_dest_yx[:, 1], 0.0, max(float(w - 1), 0.0))

    tform = tf.PiecewiseAffineTransform()
    tform.estimate(points_src_yx[:, ::-1], points_dest_yx[:, ::-1])
    return tform


def piecewise_affine(
    image: object,
    scale: float,
    nb_rows: int = 4,
    nb_cols: int = 4,
    seed: int | None = None,
    *,
    order: int = 1,
    cval: float = 0.0,
    mode: str = "constant",
    absolute_scale: bool = False,
    allow_cpu_fallback: bool = False,
) -> object:
    """
    Apply a piecewise affine transformation to an image.

    Divides the image into a grid and applies local affine transformations
    between grid cells, creating smooth local distortions.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape (H, W) or (H, W, C).
    scale : float
        Jitter magnitude for control points. If ``absolute_scale=False``,
        interpreted as fraction of image dimensions.
    nb_rows : int, default 4
        Number of control point rows.
    nb_cols : int, default 4
        Number of control point columns.
    seed : int, optional
        Random seed for reproducibility.
    order : int, default 1
        Interpolation order. 0=nearest, 1=bilinear. Higher orders require
        ``allow_cpu_fallback=True`` for CPU execution.
    cval : float, default 0.0
        Fill value for constant mode.
    mode : str, default "constant"
        Border mode ("constant", "edge", "reflect", "symmetric", "wrap").
    absolute_scale : bool, default False
        If True, ``scale`` is in pixels. If False, ``scale`` is fraction of image size.
    allow_cpu_fallback : bool, default False
        If True, allow CPU fallback for unsupported interpolation orders. This
        performs a host↔device roundtrip for MLX inputs.

    Returns
    -------
    array-like
        Warped image. Same type as input (NumPy or MLX).

    Raises
    ------
    ValueError
        If image shape is invalid.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    Notes
    -----
    Uses scikit-image on CPU to build the transformation, then resamples on MLX
    when ``order`` in {0, 1}. Higher orders are CPU-only and require
    ``allow_cpu_fallback=True``.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    if is_input_mlx:
        img_mx = to_mlx(image)
        if img_mx.size == 0:
            return img_mx

        if img_mx.ndim not in (2, 3):
            raise ValueError(
                f"piecewise_affine expects (H, W) or (H, W, C), got shape {tuple(img_mx.shape)}."
            )

        if float(scale) == 0.0:
            return img_mx

        h, w = int(img_mx.shape[0]), int(img_mx.shape[1])
        if h <= 1 or w <= 1:
            return img_mx

        if img_mx.ndim == 3 and int(img_mx.shape[2]) == 0:
            return img_mx

        original_dtype = None
    else:
        img_np = to_numpy(image)
        if img_np.size == 0:
            return img_np.copy()

        if img_np.ndim not in (2, 3):
            raise ValueError(
                f"piecewise_affine expects (H, W) or (H, W, C), got shape {img_np.shape}."
            )

        if float(scale) == 0.0:
            return img_np.copy()

        h, w = img_np.shape[:2]
        if h <= 1 or w <= 1:
            return img_np.copy()

        if img_np.ndim == 3 and img_np.shape[2] == 0:
            return img_np.copy()

        original_dtype = img_np.dtype

    tform = _build_piecewise_tform(
        (h, w), float(scale), int(nb_rows), int(nb_cols), seed, absolute_scale
    )
    if tform is None:
        if is_input_mlx:
            return image
        # is_input_mlx is False here since we didn't return above
        return img_np.copy()

    if order not in (0, 1):
        if is_input_mlx:
            _require_cpu_fallback_allowed(
                allow_cpu_fallback,
                op="piecewise_affine:",
                detail="MLX supports only interpolation orders 0/1.",
            )
        from skimage import transform as tf

        img_np = to_numpy(image)
        warped = tf.warp(
            img_np,
            tform,
            order=int(order),
            mode=str(mode),
            cval=float(cval),
            preserve_range=True,
            output_shape=img_np.shape,
        )
        out_np = _restore_dtype(warped.astype(np.float32, copy=False), img_np.dtype)
        return to_mlx(out_np) if is_input_mlx else out_np

    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    coords_out = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (H*W,2) as (x,y)
    coords_in = tform.inverse(coords_out).astype(np.float32, copy=False)

    x_in = coords_in[:, 0].reshape(h, w)
    y_in = coords_in[:, 1].reshape(h, w)

    if w > 1:
        gx = (mx.array(x_in) / float(w - 1)) * 2.0 - 1.0
    else:
        gx = mx.zeros((h, w), dtype=mx.float32)
    if h > 1:
        gy = (mx.array(y_in) / float(h - 1)) * 2.0 - 1.0
    else:
        gy = mx.zeros((h, w), dtype=mx.float32)

    grid = mx.stack([gx, gy], axis=-1)
    out_mx = grid_sample(
        image,
        grid,
        mode="nearest" if order == 0 else "bilinear",
        padding_mode=_mode_to_padding(mode),
        cval=cval,
    )

    if is_input_mlx:
        return out_mx

    out_np = to_numpy(out_mx)
    return _restore_dtype(out_np, ensure_dtype(original_dtype))


def resize(
    image: object,
    output_shape: tuple[int, int],
    *,
    order: int = 1,
    mode: str = "edge",
    cval: float = 0.0,
) -> object:
    """
    Resize an image to specified output shape.

    Uses align_corners=True convention for coordinate mapping.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape (H, W), (H, W, C),
        or (N, H, W, C).
    output_shape : tuple of int
        Target shape (H, W).
    order : int, default 1
        Interpolation order. 0=nearest, 1=bilinear. Higher orders not supported.
    mode : str, default "edge"
        Border mode ("constant", "edge", "reflect", "symmetric", "wrap").
    cval : float, default 0.0
        Fill value for constant mode.

    Returns
    -------
    array-like
        Resized image. Same type as input (NumPy or MLX).

    Raises
    ------
    ValueError
        If output_shape is invalid or order not in {0, 1}.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    order_i = int(order)
    if order_i not in (0, 1):
        raise NotImplementedError(f"resize supports only order 0/1, got {order_i}.")

    h_out, w_out = int(output_shape[0]), int(output_shape[1])
    if h_out < 0 or w_out < 0:
        raise ValueError(f"output_shape must be >=0, got {(h_out, w_out)}.")

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mx = ensure_float32(to_mlx(image))
    if img_mx.size == 0 or h_out == 0 or w_out == 0:
        # Preserve empty shapes.
        if img_mx.ndim == 2:
            out_empty = mx.zeros((h_out, w_out), dtype=img_mx.dtype)
        elif img_mx.ndim == 3:
            out_empty = mx.zeros((h_out, w_out, int(img_mx.shape[2])), dtype=img_mx.dtype)
        elif img_mx.ndim == 4:
            out_empty = mx.zeros(
                (int(img_mx.shape[0]), h_out, w_out, int(img_mx.shape[3])), dtype=img_mx.dtype
            )
        else:
            raise ValueError(
                f"resize expects (H,W), (H,W,C), or (N,H,W,C), got shape {tuple(img_mx.shape)}."
            )

        if is_input_mlx:
            return out_empty
        return _restore_dtype(to_numpy(out_empty), ensure_dtype(original_dtype))

    # Build an output grid in normalized coordinates. With align_corners=True,
    # this implements `x_in = x_out * (w_in-1)/(w_out-1)` (and same for y).
    yy, xx = _hw_grid_mx(h_out, w_out)
    if w_out > 1:
        gx = (xx / float(w_out - 1)) * 2.0 - 1.0
    else:
        gx = mx.zeros_like(xx)
    if h_out > 1:
        gy = (yy / float(h_out - 1)) * 2.0 - 1.0
    else:
        gy = mx.zeros_like(yy)
    grid = mx.stack([gx, gy], axis=-1)

    out_mx = grid_sample(
        img_mx,
        grid,
        mode="nearest" if order_i == 0 else "bilinear",
        padding_mode=_mode_to_padding(mode),
        cval=cval,
    )

    if is_input_mlx:
        return out_mx

    return _restore_dtype(to_numpy(out_mx), ensure_dtype(original_dtype))


def optical_distortion(
    image: object,
    k: float = 0.0,
    dx: float = 0.0,
    dy: float = 0.0,
    *,
    order: int = 1,
    cval: float = 0.0,
    mode: str = "constant",
) -> object:
    """
    Apply optical (lens) distortion to an image.

    Simulates barrel distortion (k > 0) or pincushion distortion (k < 0)
    commonly seen in camera lenses. Uses the radial distortion model:
    ``r_distorted = r * (1 + k * r^2)``.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape (H, W) or (H, W, C).
    k : float, default 0.0
        Distortion coefficient. Positive=barrel, negative=pincushion.
    dx : float, default 0.0
        Horizontal shift of distortion center in normalized coordinates [-1, 1].
    dy : float, default 0.0
        Vertical shift of distortion center in normalized coordinates [-1, 1].
    order : int, default 1
        Interpolation order. 0=nearest, 1=bilinear.
    cval : float, default 0.0
        Fill value for constant mode.
    mode : str, default "constant"
        Border mode ("constant", "edge", "reflect", "symmetric", "wrap").

    Returns
    -------
    array-like
        Distorted image. Same type as input (NumPy or MLX).

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
    if is_input_mlx:
        img_mx = to_mlx(image)
        if img_mx.size == 0:
            return img_mx

        if img_mx.ndim == 2:
            h, w = int(img_mx.shape[0]), int(img_mx.shape[1])
        elif img_mx.ndim == 3:
            h, w = int(img_mx.shape[0]), int(img_mx.shape[1])
        else:
            raise ValueError(
                f"optical_distortion expects (H, W) or (H, W, C), got shape {tuple(img_mx.shape)}."
            )
        original_dtype = None
    else:
        img_np = to_numpy(image)
        if img_np.size == 0:
            return img_np.copy()

        if img_np.ndim not in (2, 3):
            raise ValueError(
                f"optical_distortion expects (H, W) or (H, W, C), got shape {img_np.shape}."
            )
        h, w = img_np.shape[:2]
        original_dtype = img_np.dtype

    if float(k) == 0.0:
        if is_input_mlx:
            return image
        return img_np.copy()

    yy, xx = _hw_grid_mx(h, w)

    cx = (w - 1) / 2.0 + dx * w / 2.0
    cy = (h - 1) / 2.0 + dy * h / 2.0

    x_norm = (xx - cx) / (max(w, h) / 2.0)
    y_norm = (yy - cy) / (max(w, h) / 2.0)

    r = mx.sqrt(x_norm * x_norm + y_norm * y_norm)
    r_distorted = r * (1.0 + k * r * r)

    r_safe = mx.where(r > 1e-8, r, mx.ones_like(r))
    scale = mx.where(r > 1e-8, r_distorted / r_safe, mx.ones_like(r))

    x_src = cx + (xx - cx) * scale
    y_src = cy + (yy - cy) * scale

    if w > 1:
        gx = (x_src / float(w - 1)) * 2.0 - 1.0
    else:
        gx = mx.zeros_like(x_src)
    if h > 1:
        gy = (y_src / float(h - 1)) * 2.0 - 1.0
    else:
        gy = mx.zeros_like(y_src)

    grid = mx.stack([gx, gy], axis=-1)

    out_mx = grid_sample(
        image,
        grid,
        mode="nearest" if order == 0 else "bilinear",
        padding_mode=_mode_to_padding(mode),
        cval=cval,
    )

    if is_input_mlx:
        return out_mx

    out_np = to_numpy(out_mx)
    return _restore_dtype(out_np, ensure_dtype(original_dtype))


def grid_distortion(
    image: object,
    num_steps: int = 5,
    distort_limit: float = 0.3,
    seed: int | None = None,
    *,
    order: int = 1,
    cval: float = 0.0,
    mode: str = "constant",
) -> object:
    """
    Apply grid-based distortion to an image.

    Divides the image into a grid, randomly displaces grid points, then
    interpolates a smooth distortion field across the entire image.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape (H, W) or (H, W, C).
    num_steps : int, default 5
        Number of grid cells in each dimension.
    distort_limit : float, default 0.3
        Maximum displacement as fraction of grid step size.
    seed : int, optional
        Random seed for reproducibility.
    order : int, default 1
        Interpolation order. 0=nearest, 1=bilinear.
    cval : float, default 0.0
        Fill value for constant mode.
    mode : str, default "constant"
        Border mode ("constant", "edge", "reflect", "symmetric", "wrap").

    Returns
    -------
    array-like
        Distorted image. Same type as input (NumPy or MLX).

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
    if is_input_mlx:
        img_mx = to_mlx(image)
        if img_mx.size == 0:
            return img_mx

        if img_mx.ndim == 2:
            h, w = int(img_mx.shape[0]), int(img_mx.shape[1])
        elif img_mx.ndim == 3:
            h, w = int(img_mx.shape[0]), int(img_mx.shape[1])
        else:
            raise ValueError(
                f"grid_distortion expects (H, W) or (H, W, C), got shape {tuple(img_mx.shape)}."
            )
        original_dtype = None
    else:
        img_np = to_numpy(image)
        if img_np.size == 0:
            return img_np.copy()

        if img_np.ndim not in (2, 3):
            raise ValueError(
                f"grid_distortion expects (H, W) or (H, W, C), got shape {img_np.shape}."
            )
        h, w = img_np.shape[:2]
        original_dtype = img_np.dtype

    if distort_limit == 0.0:
        if is_input_mlx:
            return image
        return img_np.copy()

    rng = np.random.default_rng(seed)

    num_steps = max(2, int(num_steps))
    x_steps = np.linspace(0, w, num_steps, dtype=np.float32)
    y_steps = np.linspace(0, h, num_steps, dtype=np.float32)

    step_x = w / (num_steps - 1) if num_steps > 1 else w
    step_y = h / (num_steps - 1) if num_steps > 1 else h

    dx_grid = (
        rng.uniform(-distort_limit, distort_limit, size=(num_steps, num_steps)).astype(np.float32)
        * step_x
    )
    dy_grid = (
        rng.uniform(-distort_limit, distort_limit, size=(num_steps, num_steps)).astype(np.float32)
        * step_y
    )

    dx_grid[0, :] = 0
    dx_grid[-1, :] = 0
    dx_grid[:, 0] = 0
    dx_grid[:, -1] = 0
    dy_grid[0, :] = 0
    dy_grid[-1, :] = 0
    dy_grid[:, 0] = 0
    dy_grid[:, -1] = 0

    from scipy import interpolate

    interp_dx = interpolate.RegularGridInterpolator(
        (y_steps, x_steps), dx_grid, method="linear", bounds_error=False, fill_value=0.0
    )
    interp_dy = interpolate.RegularGridInterpolator(
        (y_steps, x_steps), dy_grid, method="linear", bounds_error=False, fill_value=0.0
    )

    yy_np, xx_np = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    coords = np.stack([yy_np.ravel(), xx_np.ravel()], axis=1)

    dx_full = interp_dx(coords).reshape(h, w).astype(np.float32)
    dy_full = interp_dy(coords).reshape(h, w).astype(np.float32)

    x_src = xx_np + dx_full
    y_src = yy_np + dy_full

    x_src_mx = mx.array(x_src)
    y_src_mx = mx.array(y_src)

    if w > 1:
        gx = (x_src_mx / float(w - 1)) * 2.0 - 1.0
    else:
        gx = mx.zeros_like(x_src_mx)
    if h > 1:
        gy = (y_src_mx / float(h - 1)) * 2.0 - 1.0
    else:
        gy = mx.zeros_like(y_src_mx)

    grid = mx.stack([gx, gy], axis=-1)

    out_mx = grid_sample(
        image,
        grid,
        mode="nearest" if order == 0 else "bilinear",
        padding_mode=_mode_to_padding(mode),
        cval=cval,
    )

    if is_input_mlx:
        return out_mx

    out_np = to_numpy(out_mx)
    return _restore_dtype(out_np, ensure_dtype(original_dtype))


def chromatic_aberration(
    image: object,
    primary_distortion: float = 0.0,
    secondary_distortion: float = 0.0,
    *,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> object:
    """
    Apply chromatic aberration (color fringing) effect.

    Simulates lens chromatic aberration by applying different radial distortions
    to different color channels, causing color separation especially towards edges.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array. Shape (H, W, C) with C >= 3.
    primary_distortion : float, default 0.0
        Radial distortion coefficient for red channel.
    secondary_distortion : float, default 0.0
        Radial distortion coefficient for blue channel.
    order : int, default 1
        Interpolation order. 0=nearest, 1=bilinear.
    mode : str, default "constant"
        Border mode ("constant", "edge", "reflect", "symmetric", "wrap").
    cval : float, default 0.0
        Fill value for constant mode.

    Returns
    -------
    array-like
        Image with chromatic aberration. Same type as input (NumPy or MLX).

    Raises
    ------
    ValueError
        If image shape is invalid.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    Notes
    -----
    Green channel (index 1) is used as the reference and remains undistorted.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    if is_input_mlx:
        img_mx = to_mlx(image)
        if img_mx.size == 0:
            return img_mx
        if img_mx.ndim not in (2, 3):
            raise ValueError(
                f"chromatic_aberration expects (H, W) or (H, W, C), got {tuple(img_mx.shape)}"
            )
        h, w = int(img_mx.shape[0]), int(img_mx.shape[1])
        original_dtype = None
    else:
        img_np = to_numpy(image)
        if img_np.size == 0:
            return img_np.copy()
        if img_np.ndim not in (2, 3):
            raise ValueError(
                f"chromatic_aberration expects (H, W) or (H, W, C), got {img_np.shape}"
            )
        h, w = img_np.shape[:2]
        original_dtype = img_np.dtype

    if primary_distortion == 0.0 and secondary_distortion == 0.0:
        if is_input_mlx:
            return image
        return to_numpy(image).copy()

    img_mlx = to_mlx(image)
    if img_mlx.ndim == 2:
        if is_input_mlx:
            return image
        return to_numpy(image).copy()

    c = int(img_mlx.shape[2])
    if c < 3:
        if is_input_mlx:
            return image
        return to_numpy(image).copy()

    r_channel = img_mlx[:, :, 0:1]
    r_distorted = optical_distortion(
        r_channel, k=primary_distortion, order=order, mode=mode, cval=cval
    )

    g_channel = img_mlx[:, :, 1:2]

    b_channel = img_mlx[:, :, 2:3]
    b_distorted = optical_distortion(
        b_channel, k=secondary_distortion, order=order, mode=mode, cval=cval
    )

    r_mx = to_mlx(r_distorted) if not is_mlx_array(r_distorted) else r_distorted
    b_mx = to_mlx(b_distorted) if not is_mlx_array(b_distorted) else b_distorted

    result = mx.concatenate([r_mx, g_channel, b_mx], axis=-1)

    if c > 3:
        extra = img_mlx[:, :, 3:]
        result = mx.concatenate([result, extra], axis=-1)

    if is_input_mlx:
        return result

    out_np = to_numpy(result)
    return _restore_dtype(out_np, ensure_dtype(original_dtype))


__all__ = [
    "affine_points",
    "affine_transform",
    "chromatic_aberration",
    "elastic_points",
    "elastic_transform",
    "grid_distortion",
    "grid_sample",
    "optical_distortion",
    "perspective_points",
    "perspective_transform",
    "piecewise_affine",
    "resize",
]
