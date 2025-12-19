"""Shape-aware backend router for CPU vs MLX operations.

This module provides automatic backend selection based on workload characteristics.
The routing decisions are derived from empirical benchmarks on Apple Silicon,
targeting the break-even points where MLX becomes faster than CPU implementations.

Routing Philosophy
------------------
1. For small tensors (low B*H*W), CPU wins due to MLX launch/sync overhead.
2. For larger tensors, MLX wins due to GPU parallelism.
3. Geometric transforms (affine, perspective) favor MLX earlier than blur ops.
4. Batch size matters: B>=4 often tips the balance toward MLX.

The thresholds below are conservative - they aim for MLX to be at least
competitive (speedup >= 0.9) rather than strictly faster.

Usage
-----
>>> from imgaug2.mlx.router import should_use_mlx, get_backend  # doctest: +SKIP
>>>  # doctest: +SKIP
>>> # Check if MLX should be used for a specific op  # doctest: +SKIP
>>> if should_use_mlx("affine_transform", batch=16, height=256, width=256):  # doctest: +SKIP
...     result = mlx_affine_transform(...)  # doctest: +SKIP
... else:  # doctest: +SKIP
...     result = cv2_affine_transform(...)  # doctest: +SKIP
>>>  # doctest: +SKIP
>>> # Or get the recommended backend name  # doctest: +SKIP
>>> backend = get_backend("gaussian_blur", batch=1, height=512, width=512)  # doctest: +SKIP
>>> # backend == "mlx" or "cpu"  # doctest: +SKIP

See Also
--------
benchmarks.analyze : Tool for deriving these thresholds from benchmark data.
imgaug2.mlx._fast_metal : The Metal kernels these thresholds are tuned for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._core import is_available

# Operation categories for routing purposes
OP_CATEGORY_GEOMETRIC = {"affine_transform", "perspective_transform", "resize"}
OP_CATEGORY_BLUR = {"gaussian_blur", "average_blur", "motion_blur"}
OP_CATEGORY_FAST = {"fliplr", "flipud", "rot90", "add", "multiply", "linear_contrast"}
OP_CATEGORY_NOISE = {"additive_gaussian_noise", "dropout", "coarse_dropout"}


@dataclass(frozen=True)
class RoutingThreshold:
    """Defines when MLX becomes favorable for an operation.

    Attributes
    ----------
    min_total_pixels : int
        Minimum B*H*W for MLX to be considered.
    min_hw : int
        Minimum H*W (per image) for MLX to be considered.
    min_batch : int
        Minimum batch size for MLX to be considered.
    prefer_mlx_large_batch : bool
        If True, strongly prefer MLX when batch >= 8.
    """
    min_total_pixels: int = 65536  # 256x256 or 1x256x256
    min_hw: int = 16384  # 128x128
    min_batch: int = 1
    prefer_mlx_large_batch: bool = True


# Routing thresholds derived from benchmark analysis
# These are tuned for Apple Silicon (M1/M2/M3/M4) with the current Metal kernels
ROUTING_THRESHOLDS: dict[str, RoutingThreshold] = {
    # Geometric transforms - MLX wins early
    "affine_transform": RoutingThreshold(
        min_total_pixels=65536,  # B*H*W >= 256*256
        min_hw=65536,  # H*W >= 256x256 at batch=1
        min_batch=1,
        prefer_mlx_large_batch=True,
    ),
    "perspective_transform": RoutingThreshold(
        min_total_pixels=65536,
        min_hw=65536,
        min_batch=1,
        prefer_mlx_large_batch=True,
    ),
    "resize": RoutingThreshold(
        min_total_pixels=32768,
        min_hw=16384,
        min_batch=1,
        prefer_mlx_large_batch=True,
    ),
    # Blur - CPU is strong, need more work for MLX
    "gaussian_blur": RoutingThreshold(
        min_total_pixels=262144,  # B*H*W >= 512*512
        min_hw=262144,  # H*W >= 512x512 at batch=1
        min_batch=1,
        prefer_mlx_large_batch=True,
    ),
    "average_blur": RoutingThreshold(
        min_total_pixels=131072,
        min_hw=65536,
        min_batch=2,
        prefer_mlx_large_batch=True,
    ),
    "motion_blur": RoutingThreshold(
        min_total_pixels=131072,
        min_hw=65536,
        min_batch=2,
        prefer_mlx_large_batch=True,
    ),
    # Fast ops - always use MLX if available (negligible overhead)
    "fliplr": RoutingThreshold(min_total_pixels=1, min_hw=1, min_batch=1),
    "flipud": RoutingThreshold(min_total_pixels=1, min_hw=1, min_batch=1),
    "rot90": RoutingThreshold(min_total_pixels=1, min_hw=1, min_batch=1),
    # Pointwise - prefer MLX for large tensors
    "add": RoutingThreshold(min_total_pixels=65536, min_hw=16384, min_batch=1),
    "multiply": RoutingThreshold(min_total_pixels=65536, min_hw=16384, min_batch=1),
    "linear_contrast": RoutingThreshold(min_total_pixels=65536, min_hw=16384, min_batch=1),
    # Noise - moderate thresholds
    "additive_gaussian_noise": RoutingThreshold(
        min_total_pixels=65536,
        min_hw=16384,
        min_batch=1,
    ),
    "dropout": RoutingThreshold(min_total_pixels=32768, min_hw=16384, min_batch=1),
    "coarse_dropout": RoutingThreshold(min_total_pixels=32768, min_hw=16384, min_batch=1),
}

# Default threshold for unknown ops
DEFAULT_THRESHOLD = RoutingThreshold(
    min_total_pixels=131072,  # Conservative default: 512x256 or 2x256x256
    min_hw=65536,
    min_batch=2,
    prefer_mlx_large_batch=True,
)


def should_use_mlx(
    op: str,
    *,
    batch: int,
    height: int,
    width: int,
    force_cpu: bool = False,
    force_mlx: bool = False,
) -> bool:
    """Determine if MLX should be used for the given operation and shape.

    Parameters
    ----------
    op : str
        Operation name (e.g., "affine_transform", "gaussian_blur").
    batch : int
        Batch size (N in NHWC).
    height : int
        Image height.
    width : int
        Image width.
    force_cpu : bool, default False
        If True, always return False (use CPU).
    force_mlx : bool, default False
        If True, always return True if MLX is available.

    Returns
    -------
    bool
        True if MLX should be used, False for CPU.

    Examples
    --------
    >>> should_use_mlx("affine_transform", batch=16, height=256, width=256)
    True
    >>> should_use_mlx("gaussian_blur", batch=1, height=64, width=64)
    False
    """
    if force_cpu:
        return False

    if not is_available():
        return False

    if force_mlx:
        return True

    threshold = ROUTING_THRESHOLDS.get(op, DEFAULT_THRESHOLD)

    total_pixels = batch * height * width
    hw = height * width

    # Check minimum requirements
    if total_pixels < threshold.min_total_pixels:
        return False

    if hw < threshold.min_hw:
        return False

    if batch < threshold.min_batch:
        return False

    # Strongly prefer MLX for large batches
    if threshold.prefer_mlx_large_batch and batch >= 8:
        return True

    return True


def get_backend(
    op: str,
    *,
    batch: int,
    height: int,
    width: int,
    force_cpu: bool = False,
    force_mlx: bool = False,
) -> Literal["mlx", "cpu"]:
    """Get the recommended backend for an operation.

    Parameters
    ----------
    op : str
        Operation name.
    batch : int
        Batch size.
    height : int
        Image height.
    width : int
        Image width.
    force_cpu : bool, default False
        Force CPU backend.
    force_mlx : bool, default False
        Force MLX backend (if available).

    Returns
    -------
    Literal["mlx", "cpu"]
        The recommended backend.
    """
    use_mlx = should_use_mlx(
        op,
        batch=batch,
        height=height,
        width=width,
        force_cpu=force_cpu,
        force_mlx=force_mlx,
    )
    return "mlx" if use_mlx else "cpu"


def estimate_speedup(
    op: str,
    *,
    batch: int,
    height: int,
    width: int,
) -> float:
    """Estimate the MLX/CPU speedup ratio for an operation.

    This is a rough estimate based on observed patterns:
    - Speedup increases with batch size and image size.
    - Geometric ops scale better than blur ops.
    - Very small tensors have negative speedup (MLX is slower).

    Parameters
    ----------
    op : str
        Operation name.
    batch : int
        Batch size.
    height : int
        Image height.
    width : int
        Image width.

    Returns
    -------
    float
        Estimated speedup ratio. >1 means MLX is faster, <1 means CPU is faster.
    """
    if not is_available():
        return 0.0

    total_pixels = batch * height * width
    hw = height * width

    # Base overhead cost (MLX loses at tiny sizes)
    overhead_penalty = min(1.0, total_pixels / 65536)

    # Batch scaling
    batch_factor = 1.0 + 0.1 * min(batch, 32)

    # Size scaling
    size_factor = min(2.0, hw / 65536)

    # Op-specific multiplier
    if op in OP_CATEGORY_GEOMETRIC:
        op_mult = 1.5  # Geometric ops scale well
    elif op in OP_CATEGORY_BLUR:
        op_mult = 0.8  # Blur is CPU-optimized
    elif op in OP_CATEGORY_FAST:
        op_mult = 1.0  # Fast ops are neutral
    else:
        op_mult = 1.0

    estimated = overhead_penalty * batch_factor * size_factor * op_mult

    # Clamp to reasonable range
    return max(0.1, min(estimated, 30.0))


def get_routing_info(op: str) -> dict:
    """Get routing information for an operation.

    Parameters
    ----------
    op : str
        Operation name.

    Returns
    -------
    dict
        Dictionary with threshold info and category.
    """
    threshold = ROUTING_THRESHOLDS.get(op, DEFAULT_THRESHOLD)

    category = "unknown"
    if op in OP_CATEGORY_GEOMETRIC:
        category = "geometric"
    elif op in OP_CATEGORY_BLUR:
        category = "blur"
    elif op in OP_CATEGORY_FAST:
        category = "fast"
    elif op in OP_CATEGORY_NOISE:
        category = "noise"

    return {
        "op": op,
        "category": category,
        "min_total_pixels": threshold.min_total_pixels,
        "min_hw": threshold.min_hw,
        "min_batch": threshold.min_batch,
        "prefer_mlx_large_batch": threshold.prefer_mlx_large_batch,
        "mlx_available": is_available(),
    }


def update_threshold(
    op: str,
    *,
    min_total_pixels: int | None = None,
    min_hw: int | None = None,
    min_batch: int | None = None,
    prefer_mlx_large_batch: bool | None = None,
) -> None:
    """Update routing threshold for an operation.

    This allows runtime customization of routing decisions, for example
    based on user preferences or runtime profiling.

    Parameters
    ----------
    op : str
        Operation name.
    min_total_pixels : int, optional
        New minimum total pixels threshold.
    min_hw : int, optional
        New minimum H*W threshold.
    min_batch : int, optional
        New minimum batch threshold.
    prefer_mlx_large_batch : bool, optional
        New large batch preference.
    """
    current = ROUTING_THRESHOLDS.get(op, DEFAULT_THRESHOLD)

    ROUTING_THRESHOLDS[op] = RoutingThreshold(
        min_total_pixels=(
            min_total_pixels if min_total_pixels is not None else current.min_total_pixels
        ),
        min_hw=min_hw if min_hw is not None else current.min_hw,
        min_batch=min_batch if min_batch is not None else current.min_batch,
        prefer_mlx_large_batch=(
            prefer_mlx_large_batch
            if prefer_mlx_large_batch is not None
            else current.prefer_mlx_large_batch
        ),
    )


__all__ = [
    "DEFAULT_THRESHOLD",
    "OP_CATEGORY_BLUR",
    "OP_CATEGORY_FAST",
    "OP_CATEGORY_GEOMETRIC",
    "OP_CATEGORY_NOISE",
    "ROUTING_THRESHOLDS",
    "RoutingThreshold",
    "estimate_speedup",
    "get_backend",
    "get_routing_info",
    "should_use_mlx",
    "update_threshold",
]