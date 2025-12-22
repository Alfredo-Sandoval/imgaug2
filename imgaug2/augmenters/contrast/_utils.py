"""Shared helpers for contrast augmenters."""

from __future__ import annotations

from collections.abc import Sequence

from imgaug2.mlx._core import is_mlx_array


def _is_mlx_list(images: object) -> bool:
    return (
        isinstance(images, Sequence)
        and len(images) > 0
        and is_mlx_array(images[0])
    )
