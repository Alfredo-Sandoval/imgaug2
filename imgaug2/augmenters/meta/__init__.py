"""Augmenters for meta usage and pipeline control.

This module provides the base `Augmenter` class and augmenters for controlling
pipeline flow, sequencing, and conditional execution.

Key Augmenters:
    - `Sequential`, `SomeOf`, `OneOf`: Combine and select augmenters.
    - `Sometimes`: Apply augmenters probabilistically.
    - `WithChannels`, `ChannelShuffle`: Channel-specific operations.
    - `Noop`, `Lambda`, `Identity`: Utility augmenters.
"""

from __future__ import annotations

# TODO: Run full meta tests: conda run -n imgaug pytest test/augmenters/test_meta.py

from .base import Augmenter, _maybe_deterministic_ctx  # noqa: F401
from .cba_ops import ClipCBAsToImagePlanes, RemoveCBAsByOutOfImageFraction
from .channels import ChannelShuffle, WithChannels, shuffle_channels
from .containers import OneOf, Sequential, SomeOf, Sometimes, handle_children_list
from .identity import Identity, Noop
from .lambda_assert import AssertLambda, AssertShape, Lambda
from .utils import (
    Number,
    clip_augmented_image,
    clip_augmented_image_,
    clip_augmented_images,
    clip_augmented_images_,
    copy_arrays,
    estimate_max_number_of_channels,
    invert_reduce_to_nonempty,
    reduce_to_nonempty,
)

from .utils import _HasShape, _add_channel_axis, _remove_added_channel_axis  # noqa: F401

from .lambda_assert import (  # noqa: F401
    _AssertLambdaCallback,
    _AssertShapeBoundingBoxesCheck,
    _AssertShapeHeatmapsCheck,
    _AssertShapeImagesCheck,
    _AssertShapeKeypointsCheck,
    _AssertShapeLineStringsCheck,
    _AssertShapePolygonsCheck,
    _AssertShapeSegmapCheck,
)

__all__ = [
    "AssertLambda",
    "AssertShape",
    "Augmenter",
    "ChannelShuffle",
    "ClipCBAsToImagePlanes",
    "Identity",
    "Lambda",
    "Noop",
    "Number",
    "OneOf",
    "RemoveCBAsByOutOfImageFraction",
    "Sequential",
    "SomeOf",
    "Sometimes",
    "WithChannels",
    "clip_augmented_image",
    "clip_augmented_image_",
    "clip_augmented_images",
    "clip_augmented_images_",
    "copy_arrays",
    "estimate_max_number_of_channels",
    "handle_children_list",
    "invert_reduce_to_nonempty",
    "reduce_to_nonempty",
    "shuffle_channels",
]
