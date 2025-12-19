"""CUDA/GPU-accelerated image augmentation backend using CuPy.

This module provides GPU-accelerated implementations of common image augmentation
operations using CuPy (CUDA). All functions expect ``cupy.ndarray`` inputs and
return GPU-resident arrays for maximum performance.

Notes
-----
- This backend requires a CUDA-compatible GPU and CuPy installation
- For Apple Silicon (M1/M2/M3), use ``imgaug2.mlx`` instead
- For CPU-only systems, use the main ``imgaug2`` package
- Keep images on GPU across multiple operations to avoid transfer overhead

Available Operations
--------------------
- Blur: gaussian_blur
- Color: grayscale
- Flip: fliplr, flipud, rot90
- Geometry: affine_transform
- Noise: additive_gaussian_noise, dropout
- Pointwise: add, multiply, linear_contrast, gamma_contrast, invert, solarize
- Pooling: avg_pool, max_pool, min_pool
"""
from __future__ import annotations

from ._core import CUDA_AVAILABLE, cp, is_available, require
from .blur import gaussian_blur
from .color import grayscale
from .flip import fliplr, flipud, rot90
from .geometry import affine_transform
from .noise import additive_gaussian_noise, dropout
from .pointwise import add, gamma_contrast, invert, linear_contrast, multiply, solarize
from .pooling import avg_pool, max_pool, min_pool

__all__ = [
    "CUDA_AVAILABLE",
    "add",
    "avg_pool",
    "cp",
    "dropout",
    "affine_transform",
    "fliplr",
    "flipud",
    "gaussian_blur",
    "gamma_contrast",
    "grayscale",
    "invert",
    "is_available",
    "linear_contrast",
    "max_pool",
    "min_pool",
    "multiply",
    "additive_gaussian_noise",
    "require",
    "rot90",
    "solarize",
]
