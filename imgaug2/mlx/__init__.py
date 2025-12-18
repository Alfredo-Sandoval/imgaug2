"""Dummy MLX module to satisfy imports."""
from __future__ import annotations
from typing import Any
from ._core import is_mlx_array, to_numpy, mx, geometry

__all__ = [
    "is_mlx_array", "to_numpy", "mx", "geometry",
    "add", "multiply", "affine_transform", "perspective_transform",
    "grid_sample", "rot90", "gaussian_blur", "fliplr", "flipud"
]

def add(a: Any, b: Any) -> Any:
    return a

def multiply(a: Any, b: Any) -> Any:
    return a

def affine_transform(*args: Any, **kwargs: Any) -> Any:
    return args[0]

def perspective_transform(*args: Any, **kwargs: Any) -> Any:
    return args[0]

def grid_sample(*args: Any, **kwargs: Any) -> Any:
    return args[0]

def rot90(*args: Any, **kwargs: Any) -> Any:
    return args[0]

def gaussian_blur(*args: Any, **kwargs: Any) -> Any:
    return args[0]

def fliplr(arr: Any) -> Any:
    return arr

def flipud(arr: Any) -> Any:
    return arr
