"""Augmenters that modify the size of images.

This module provides augmenters for resizing, cropping, and padding images
to various target sizes, aspect ratios, or multiples.

Key Augmenters:
    - `Resize`: Resize images to given sizes.
    - `CropAndPad`: Crop and/or pad images by specified amounts.
    - `Crop`, `Pad`: Crop or pad images on specific sides.
    - `PadToFixedSize`, `CropToFixedSize`: Pad/crop to exact pixel dimensions.
    - `PadToAspectRatio`, `CropToAspectRatio`: Adjust to target aspect ratios.
    - `PadToMultiplesOf`, `CropToMultiplesOf`: Ensure dimensions are multiples.
    - `KeepSizeByResize`: Wrap augmenters to restore original size after.
"""

from __future__ import annotations

from ._utils import (
    compute_croppings_to_reach_aspect_ratio,
    compute_croppings_to_reach_multiples_of,
    compute_croppings_to_reach_powers_of,
    compute_paddings_to_reach_aspect_ratio,
    compute_paddings_to_reach_multiples_of,
    compute_paddings_to_reach_powers_of,
    pad,
    pad_to_aspect_ratio,
    pad_to_multiples_of,
)
from .aspect_ratio import (
    CenterCropToAspectRatio,
    CenterPadToAspectRatio,
    CropToAspectRatio,
    PadToAspectRatio,
)
from .crop_pad import Crop, CropAndPad, Pad
from .crop_pad import _prevent_zero_sizes_after_crops_  # noqa: F401
from .fixed_size import CenterCropToFixedSize, CenterPadToFixedSize, CropToFixedSize, PadToFixedSize
from .keep_size import KeepSizeByResize
from .multiples import (
    CenterCropToMultiplesOf,
    CenterPadToMultiplesOf,
    CropToMultiplesOf,
    PadToMultiplesOf,
)
from .powers import (
    CenterCropToPowersOf,
    CenterPadToPowersOf,
    CropToPowersOf,
    PadToPowersOf,
)
from .resize import Resize, Scale
from .square import CenterCropToSquare, CenterPadToSquare, CropToSquare, PadToSquare

__all__ = [
    "pad",
    "pad_to_aspect_ratio",
    "pad_to_multiples_of",
    "compute_paddings_to_reach_aspect_ratio",
    "compute_croppings_to_reach_aspect_ratio",
    "compute_paddings_to_reach_multiples_of",
    "compute_croppings_to_reach_multiples_of",
    "compute_paddings_to_reach_powers_of",
    "compute_croppings_to_reach_powers_of",
    "Resize",
    "Scale",
    "CropAndPad",
    "Crop",
    "Pad",
    "PadToFixedSize",
    "CenterPadToFixedSize",
    "CropToFixedSize",
    "CenterCropToFixedSize",
    "CropToMultiplesOf",
    "CenterCropToMultiplesOf",
    "PadToMultiplesOf",
    "CenterPadToMultiplesOf",
    "CropToPowersOf",
    "CenterCropToPowersOf",
    "PadToPowersOf",
    "CenterPadToPowersOf",
    "CropToAspectRatio",
    "CenterCropToAspectRatio",
    "PadToAspectRatio",
    "CenterPadToAspectRatio",
    "CropToSquare",
    "CenterCropToSquare",
    "PadToSquare",
    "CenterPadToSquare",
    "KeepSizeByResize",
]
