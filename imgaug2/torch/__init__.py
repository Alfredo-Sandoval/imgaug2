"""PyTorch Integration for imgaug2.

This module provides zero-copy tensor conversion utilities between PyTorch
CUDA tensors and CuPy arrays using the DLPack protocol. These utilities
enable efficient interoperability with PyTorch when performing GPU-accelerated
image augmentations.

The conversion functions leverage DLPack to share memory between frameworks
without copying data, making them suitable for high-performance pipelines.

Notes
-----
- All conversions require CUDA tensors (CPU tensors are not supported)
- Both PyTorch and CuPy must be installed
- Functions are opt-in and not imported by default to avoid dependency issues
"""
from __future__ import annotations

from .dlpack import cupy_array_to_torch_tensor, torch_tensor_to_cupy_array

__all__ = [
    "cupy_array_to_torch_tensor",
    "torch_tensor_to_cupy_array",
]

