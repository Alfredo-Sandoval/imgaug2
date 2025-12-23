"""Legacy shim for size/padding helpers.

Prefer `imgaug2.augmenters.size._utils` in new code.
"""

from __future__ import annotations

from imgaug2.augmenters.size._utils import (
    compute_paddings_to_reach_aspect_ratio,
    pad,
    pad_to_aspect_ratio,
)

__all__ = [
    "compute_paddings_to_reach_aspect_ratio",
    "pad",
    "pad_to_aspect_ratio",
]
