"""Segmentation helpers for MLX-compatible pipelines.

MLX-only execution:
- All inputs (NumPy or MLX) execute through MLX.
- NumPy inputs are converted to MLX internally and the result is converted back
  to NumPy (to preserve the old return-type behavior).

Performance features:
- Fused scatter-add for sums+counts.
- Cached pixel coordinate grid for Voronoi keyed by (H, W).
- Vectorized chunking for Voronoi assignment using batched matmul.
Runtime tuning:
- IMGAUG_MLX_REPLACE_DEVICE: auto|cpu|gpu (default: auto)
- IMGAUG_MLX_VORONOI_DEVICE: auto|cpu|gpu (default: auto)
 - IMGAUG_MLX_REPLACE_BACKEND: auto|scatter|mm (default: mm)
- IMGAUG_MLX_MM_MAX_PK: int (default: 8388608)
- IMGAUG_MLX_MM_MAX_K: int (default: 256)
- IMGAUG_MLX_FUSED_SCATTER: 1|0 (default: 1)
- IMGAUG_MLX_LOG_SEGMENTATION_DECISIONS: 1|0 (default: 0)

Notes:
- IMGAUG_MLX_MM_MAX_PK is a performance threshold, not just a memory cap, and
  is hardware/MLX-version dependent. Treat it as a tuning knob.

Correctness contract (replace_segments_, option A):
- When replace_flags is provided and non-empty, we assume dense segment labels
  in [0, K-1] where K = len(replace_flags). No runtime validation is performed.
- MLX indexing does not perform bounds checking; out-of-bounds is undefined
  behavior. Ensure labels are in range.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
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

_LOGGER = logging.getLogger(__name__)


def _as_bool_mask(mask: MlxArray) -> MlxArray:
    if mask.dtype == mx.bool_:
        return mask
    if mx.issubdtype(mask.dtype, mx.floating):
        return mask > 0.5
    return mask != 0


def _ensure_hw_c(image: MlxArray) -> tuple[MlxArray, bool]:
    if image.ndim == 2:
        return image[..., None], True
    if image.ndim == 3:
        return image, False
    raise ValueError(f"Expected image with ndim=2 or 3, got shape {tuple(image.shape)}.")


def _device_from_env(env_var: str) -> object | None:
    value = os.getenv(env_var, "auto").strip().lower()
    if value == "cpu":
        return mx.cpu
    if value == "gpu":
        return mx.gpu
    return None


def _env_value(env_var: str, default: str = "auto") -> str:
    return os.getenv(env_var, default).strip().lower()


def _select_replace_device(
    height: int,
    width: int,
    num_segments: int | None,
    device_override: object | None,
) -> object:
    if device_override is not None:
        return device_override
    if num_segments is None:
        return mx.cpu
    if num_segments <= 512 and (height * width) <= 1024 * 1024:
        return mx.cpu
    return mx.gpu


def _select_voronoi_device(
    height: int,
    width: int,
    num_cells: int,
    device_override: object | None,
) -> object:
    if device_override is not None:
        return device_override
    return mx.gpu


def _use_fused_scatter() -> bool:
    value = os.getenv("IMGAUG_MLX_FUSED_SCATTER", "1").strip().lower()
    return value not in {"0", "false", "no"}


def _replace_backend() -> tuple[str, bool]:
    value = os.getenv("IMGAUG_MLX_REPLACE_BACKEND")
    if value is None:
        return "mm", False
    return value.strip().lower(), True


def _mm_max_pk() -> int:
    return int(os.getenv("IMGAUG_MLX_MM_MAX_PK", "8388608"))


def _mm_max_k() -> int:
    return int(os.getenv("IMGAUG_MLX_MM_MAX_K", "256"))


def _should_use_mm(num_pixels: int, num_segments: int) -> bool:
    if num_segments <= 0:
        return False
    if num_segments > _mm_max_k():
        return False
    return (num_pixels * num_segments) <= _mm_max_pk()


def _log_decisions_enabled() -> bool:
    value = os.getenv("IMGAUG_MLX_LOG_SEGMENTATION_DECISIONS", "0").strip().lower()
    return value in {"1", "true", "yes"}


@lru_cache(maxsize=512)
def _log_replace_decision_once(message: str) -> None:
    _LOGGER.debug(message)


def _fused_sum_and_count(
    values_f32: MlxArray,
    ids_i32: MlxArray,
    n_bins: int,
) -> tuple[MlxArray, MlxArray]:
    """Compute sums and counts with a single scatter-add.

    values_f32: (P, C) float32
    ids_i32:    (P,) int32, assumed in [0, n_bins-1]
    """
    p, c = values_f32.shape
    ones = mx.ones((p, 1), dtype=mx.float32)
    vals = mx.concatenate([values_f32, ones], axis=1)

    acc = mx.zeros((n_bins, c + 1), dtype=mx.float32)
    acc = acc.at[ids_i32].add(vals)

    sums = acc[:, :c]
    counts = acc[:, c]
    return sums, counts


def _sum_and_count(
    values_f32: MlxArray,
    ids_i32: MlxArray,
    n_bins: int,
) -> tuple[MlxArray, MlxArray]:
    if _use_fused_scatter():
        return _fused_sum_and_count(values_f32, ids_i32, n_bins)

    sums = mx.zeros((n_bins, values_f32.shape[1]), dtype=mx.float32)
    sums = sums.at[ids_i32].add(values_f32)

    counts = mx.zeros((n_bins,), dtype=mx.float32)
    counts = counts.at[ids_i32].add(1.0)
    return sums, counts


def _replace_segments_mm(
    image: MlxArray,
    segments: MlxArray,
    replace_flags: MlxArray,
) -> MlxArray:
    image_hw_c, squeeze_channel = _ensure_hw_c(image)
    h, w, c = image_hw_c.shape

    if segments.ndim != 2 or tuple(segments.shape) != (h, w):
        raise ValueError(
            f"`segments` must have shape {(h, w)!r}, got {tuple(segments.shape)!r}."
        )

    if image_hw_c.size == 0:
        return image_hw_c[..., 0] if squeeze_channel else image_hw_c

    seg_flat = segments.astype(mx.int32).reshape((-1,))
    img_flat = image_hw_c.reshape((-1, c))
    img_f32 = img_flat.astype(mx.float32)

    flags = replace_flags.reshape((-1,))
    k = int(flags.shape[0])
    if k <= 0:
        return image_hw_c[..., 0] if squeeze_channel else image_hw_c

    classes = mx.arange(k, dtype=mx.int32)[:, None]
    onehot = (classes == seg_flat[None, :]).astype(mx.float32)

    sums = mx.matmul(onehot, img_f32)
    counts = mx.sum(onehot, axis=1)
    counts = mx.maximum(counts, 1.0)

    means = (sums / counts[:, None]).astype(image_hw_c.dtype)
    mean_flat = means[seg_flat]

    flags_bool = _as_bool_mask(flags)
    replace_pix = flags_bool[seg_flat]
    if c > 1:
        replace_pix = replace_pix[:, None]

    out_flat = mx.where(replace_pix, mean_flat, img_flat)
    out = out_flat.reshape((h, w, c))
    return out[..., 0] if squeeze_channel else out

@lru_cache(maxsize=32)
def _cached_pixels_xy_norm2(height: int, width: int) -> tuple[MlxArray, MlxArray]:
    """Cache pixel centers and squared norms for a given HxW.

    Materializes once to avoid rebuilding graphs on repeated calls.
    """
    num_pixels = int(height * width)
    lin = mx.arange(num_pixels, dtype=mx.int32)
    xs = (lin % width).astype(mx.float32) + 0.5
    ys = (lin // width).astype(mx.float32) + 0.5
    pixels_xy = mx.stack([xs, ys], axis=1)
    pixels_norm2 = xs * xs + ys * ys

    mx.eval(pixels_xy, pixels_norm2)
    return pixels_xy, pixels_norm2


def _nearest_cell_ids(
    pixels_xy: MlxArray,
    pixels_norm2: MlxArray,
    cell_xy: MlxArray,
    *,
    max_pairs: int = 8_000_000,
) -> MlxArray:
    """Assign each pixel to its nearest cell using squared Euclidean distance.

    Uses batched matmul and vectorized chunking (no Python chunk loop).
    """
    p = int(pixels_xy.shape[0])
    n = int(cell_xy.shape[0])
    if n <= 0:
        return mx.zeros((p,), dtype=mx.int32)

    cell_f = cell_xy.astype(mx.float32)
    cell_t = mx.transpose(cell_f)
    cell_norm2 = mx.sum(cell_f * cell_f, axis=1)

    if p * n <= max_pairs:
        dot = mx.matmul(pixels_xy.astype(mx.float32), cell_t)
        dist2 = pixels_norm2[:, None] + cell_norm2[None, :] - 2.0 * dot
        return mx.argmin(dist2, axis=1).astype(mx.int32)

    chunk = max(1024, max_pairs // max(1, n))
    pad = (chunk - (p % chunk)) % chunk
    if pad:
        pixels_xy_p = mx.pad(pixels_xy, ((0, pad), (0, 0)))
        pixels_norm2_p = mx.pad(pixels_norm2, (0, pad))
    else:
        pixels_xy_p = pixels_xy
        pixels_norm2_p = pixels_norm2

    p2 = int(pixels_xy_p.shape[0])
    b = p2 // chunk

    pix_xy_blk = pixels_xy_p.reshape((b, chunk, 2)).astype(mx.float32)
    pix_n2_blk = pixels_norm2_p.reshape((b, chunk)).astype(mx.float32)

    dot = mx.matmul(pix_xy_blk, cell_t)
    dist2 = pix_n2_blk[..., None] + cell_norm2[None, None, :] - 2.0 * dot
    ids_blk = mx.argmin(dist2, axis=2).astype(mx.int32)

    return ids_blk.reshape((-1,))[:p]


def _replace_segments_mlx(
    image: MlxArray,
    segments: MlxArray,
    replace_flags: MlxArray | None,
) -> MlxArray:
    """MLX-only replace_segments_.

    Fast path (option A):
    - if replace_flags is provided and non-empty, K = len(replace_flags),
      and segments must be dense labels in [0, K-1] (unchecked).
    Slow path:
    - if replace_flags is None, K must be derived from data, requiring a host
      sync to allocate accumulator arrays.
    """
    image_hw_c, squeeze_channel = _ensure_hw_c(image)
    h, w, c = image_hw_c.shape

    if segments.ndim != 2 or tuple(segments.shape) != (h, w):
        raise ValueError(
            f"`segments` must have shape {(h, w)!r}, got {tuple(segments.shape)!r}."
        )

    if image_hw_c.size == 0:
        return image_hw_c[..., 0] if squeeze_channel else image_hw_c

    seg_flat = segments.astype(mx.int32).reshape((-1,))
    img_flat = image_hw_c.reshape((-1, c))
    img_f32 = img_flat.astype(mx.float32)

    if replace_flags is not None:
        flags = replace_flags.reshape((-1,))
        k = int(flags.shape[0])
        if k <= 0:
            return image_hw_c[..., 0] if squeeze_channel else image_hw_c

        sums, counts = _sum_and_count(img_f32, seg_flat, k)
        counts = mx.maximum(counts, 1.0)
        means = (sums / counts[:, None]).astype(image_hw_c.dtype)

        mean_flat = means[seg_flat]
        flags_bool = _as_bool_mask(flags)
        replace_pix = flags_bool[seg_flat]
        if c > 1:
            replace_pix = replace_pix[:, None]

        out_flat = mx.where(replace_pix, mean_flat, img_flat)
        out = out_flat.reshape((h, w, c))
        return out[..., 0] if squeeze_channel else out

    max_label = int(mx.max(seg_flat).item())
    if max_label < 0:
        raise ValueError("`segments` must contain non-negative labels.")
    k = max_label + 1
    if k <= 0:
        return image_hw_c[..., 0] if squeeze_channel else image_hw_c

    sums, counts = _sum_and_count(img_f32, seg_flat, k)
    counts = mx.maximum(counts, 1.0)
    means = (sums / counts[:, None]).astype(image_hw_c.dtype)

    out_flat = means[seg_flat]
    out = out_flat.reshape((h, w, c))
    return out[..., 0] if squeeze_channel else out


def _segment_voronoi_mlx(
    image: MlxArray,
    cell_coordinates: MlxArray,
    replace_mask: MlxArray | None,
) -> MlxArray:
    """MLX-only Voronoi color averaging."""
    image_hw_c, squeeze_channel = _ensure_hw_c(image)
    h, w, c = image_hw_c.shape

    if cell_coordinates.ndim != 2 or int(cell_coordinates.shape[1]) != 2:
        raise ValueError(
            "`cell_coordinates` must have shape (N, 2) containing (x, y) coordinates."
        )

    n_cells = int(cell_coordinates.shape[0])
    if n_cells == 0 or image_hw_c.size == 0:
        return image_hw_c[..., 0] if squeeze_channel else image_hw_c

    pixels_xy, pixels_norm2 = _cached_pixels_xy_norm2(int(h), int(w))
    ids = _nearest_cell_ids(pixels_xy, pixels_norm2, cell_coordinates.astype(mx.float32))

    img_flat = image_hw_c.reshape((-1, c))
    img_f32 = img_flat.astype(mx.float32)

    sums, counts = _sum_and_count(img_f32, ids, n_cells)
    counts = mx.maximum(counts, 1.0)
    means = (sums / counts[:, None]).astype(image_hw_c.dtype)

    mean_flat = means[ids]

    if replace_mask is None:
        out_flat = mean_flat
    else:
        mask = replace_mask.reshape((-1,))
        m = int(mask.shape[0])
        if m <= 0:
            out_flat = img_flat
        else:
            mask_bool = _as_bool_mask(mask)
            if m == n_cells:
                replace_pix = mask_bool[ids]
            else:
                replace_pix = mask_bool[ids % m]
            if c > 1:
                replace_pix = replace_pix[:, None]
            out_flat = mx.where(replace_pix, mean_flat, img_flat)

    out = out_flat.reshape((h, w, c))
    return out[..., 0] if squeeze_channel else out


@lru_cache(maxsize=1)
def _compiled_replace_segments():
    require()
    return mx.compile(_replace_segments_mlx)


@lru_cache(maxsize=1)
def _compiled_replace_segments_mm():
    require()
    return mx.compile(_replace_segments_mm)


@lru_cache(maxsize=1)
def _compiled_segment_voronoi():
    require()
    return mx.compile(_segment_voronoi_mlx)


@overload
def replace_segments_(
    image: NumpyArray,
    segments: NumpyArray,
    replace_flags: NumpyArray | None,
) -> NumpyArray: ...


@overload
def replace_segments_(
    image: MlxArray,
    segments: MlxArray,
    replace_flags: MlxArray | None,
) -> MlxArray: ...


def replace_segments_(
    image: object,
    segments: object,
    replace_flags: object | None,
) -> object:
    """Replace segments with their average colors (MLX-only execution)."""
    require()

    is_out_mlx = (
        is_mlx_array(image)
        or is_mlx_array(segments)
        or is_mlx_array(replace_flags)
    )

    image_mx = to_mlx(image)
    segments_mx = to_mlx(segments)
    flags_mx = to_mlx(replace_flags) if replace_flags is not None else None

    height, width = int(segments_mx.shape[0]), int(segments_mx.shape[1])
    num_segments = int(flags_mx.shape[0]) if flags_mx is not None else None
    num_pixels = int(height * width)

    backend_req, backend_forced = _replace_backend()
    device_req = _env_value("IMGAUG_MLX_REPLACE_DEVICE")
    device_env = _device_from_env("IMGAUG_MLX_REPLACE_DEVICE")
    device = _select_replace_device(height, width, num_segments, device_env)

    use_mm = False
    reason = "replace_flags_none" if flags_mx is None else ""

    if flags_mx is None and backend_req == "mm" and backend_forced:
        raise ValueError(
            "IMGAUG_MLX_REPLACE_BACKEND=mm requested, but replace_flags is None. "
            "Provide replace_flags to use the mm backend."
        )

    if flags_mx is not None:
        mm_feasible = _should_use_mm(num_pixels, num_segments or 0)
        if backend_req == "mm":
            if not mm_feasible and backend_forced:
                raise ValueError(
                    "IMGAUG_MLX_REPLACE_BACKEND=mm requested, but P*K exceeds "
                    "IMGAUG_MLX_MM_MAX_PK or K exceeds IMGAUG_MLX_MM_MAX_K. "
                    "Increase caps to proceed."
                )
            if mm_feasible:
                use_mm = True
                if device_env is None:
                    device = mx.gpu
                elif device_env is mx.cpu:
                    reason = "device_forced_cpu"
            else:
                reason = "mm_infeasible"
        elif backend_req == "scatter":
            reason = "backend_forced_scatter"
        else:
            if mm_feasible and device_env is not mx.cpu:
                use_mm = True
                if device_env is None:
                    device = mx.gpu
            elif device_env is mx.cpu:
                reason = "device_forced_cpu"
            elif num_segments is not None and num_segments > _mm_max_k():
                reason = "k_exceeds"
            elif not mm_feasible:
                reason = "pk_exceeds"

    backend_chosen = "mm" if use_mm else "scatter"
    device_chosen = "cpu" if device is mx.cpu else "gpu"
    if _log_decisions_enabled() and _LOGGER.isEnabledFor(logging.DEBUG):
        message = (
            "replace_segments_ decision "
            f"backend_req={backend_req} backend_forced={backend_forced} "
            f"backend_chosen={backend_chosen} "
            f"device_req={device_req} device_chosen={device_chosen} "
            f"p={num_pixels} k={num_segments} pk={num_pixels * (num_segments or 0)} "
            f"mm_max_pk={_mm_max_pk()} mm_max_k={_mm_max_k()} reason={reason or 'selected'}"
        )
        _log_replace_decision_once(message)

    with mx.stream(device):
        if flags_mx is None:
            out_mx = _replace_segments_mlx(image_mx, segments_mx, None)
        elif use_mm:
            out_mx = _compiled_replace_segments_mm()(image_mx, segments_mx, flags_mx)
        else:
            out_mx = _compiled_replace_segments()(image_mx, segments_mx, flags_mx)

    return out_mx if is_out_mlx else to_numpy(out_mx)


@overload
def segment_voronoi(
    image: NumpyArray,
    cell_coordinates: NumpyArray,
    replace_mask: NumpyArray | None = None,
) -> NumpyArray: ...


@overload
def segment_voronoi(
    image: MlxArray,
    cell_coordinates: MlxArray,
    replace_mask: MlxArray | None = None,
) -> MlxArray: ...


def segment_voronoi(
    image: object,
    cell_coordinates: object,
    replace_mask: object | None = None,
) -> object:
    """Average colors within Voronoi cells (MLX-only execution)."""
    require()

    is_out_mlx = (
        is_mlx_array(image)
        or is_mlx_array(cell_coordinates)
        or is_mlx_array(replace_mask)
    )

    image_mx = to_mlx(image)
    coords_mx = to_mlx(cell_coordinates)
    mask_mx = to_mlx(replace_mask) if replace_mask is not None else None

    height, width = int(image_mx.shape[0]), int(image_mx.shape[1])
    num_cells = int(coords_mx.shape[0])
    device_env = _device_from_env("IMGAUG_MLX_VORONOI_DEVICE")
    device = _select_voronoi_device(height, width, num_cells, device_env)

    with mx.stream(device):
        out_mx = _compiled_segment_voronoi()(image_mx, coords_mx, mask_mx)

    return out_mx if is_out_mlx else to_numpy(out_mx)


__all__ = ["replace_segments_", "segment_voronoi"]
