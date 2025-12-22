"""Augmenters that blend two images with each other.

This module provides augmenters for alpha blending, frequency-domain blending,
and other image combination techniques.

Key Augmenters:
    - `BlendAlpha`, `BlendAlphaElementwise`: Blend with constant/per-pixel alpha.
    - `BlendAlphaSimplexNoise`, `BlendAlphaFrequencyNoise`: Blend using noise masks.
    - `BlendAlphaSomeColors`, `BlendAlphaRegularGrid`, `BlendAlphaCheckerboard`:
      Blend based on colors, grids, or checkerboard patterns.
    - `BlendAlphaBoundingBoxes`, `BlendAlphaSegMapClassIds`: Blend using augmentables.
"""

from __future__ import annotations

import imgaug2.augmenters._blend_utils as blend_utils
import imgaug2.imgaug as ia

from .base import (
    AlphaInput,
    AggregationMethodInput,
    ChildrenInput,
    CoordinateAugmentable,
    LabelInput,
    PerChannelInput,
    SigmoidInput,
    UpscaleMethodInput,
    _blend_alpha_non_uint8,
    _blend_alpha_uint8_channelwise_alphas_,
    _blend_alpha_uint8_elementwise_,
    _blend_alpha_uint8_single_alpha_,
    _merge_channels,
    blend_alpha,
    blend_alpha_,
)
from .base import BlendAlpha
from .gradients import BlendAlphaHorizontalLinearGradient, BlendAlphaVerticalLinearGradient
from .grids import BlendAlphaCheckerboard, BlendAlphaRegularGrid
from .mask_generators import (
    IBatchwiseMaskGenerator,
    SomeColorsMaskGen,
    StochasticParameterMaskGen,
    _LinearGradientMaskGen,
    BoundingBoxesMaskGen,
    CheckerboardMaskGen,
    HorizontalLinearGradientMaskGen,
    InvertMaskGen,
    RegularGridMaskGen,
    SegMapClassIdsMaskGen,
    VerticalLinearGradientMaskGen,
)
from .masks import BlendAlphaElementwise, BlendAlphaMask, BlendAlphaSomeColors
from .noise_blend import BlendAlphaFrequencyNoise, BlendAlphaSimplexNoise
from .segmap_bbs import BlendAlphaBoundingBoxes, BlendAlphaSegMapClassIds


@ia.deprecated(alt_func="BlendAlpha")
def Alpha(*args: object, **kwargs: object) -> BlendAlpha:
    """Deprecated alias for :class:`BlendAlpha`."""
    return BlendAlpha(*args, **kwargs)


@ia.deprecated(alt_func="BlendAlphaElementwise")
def AlphaElementwise(*args: object, **kwargs: object) -> BlendAlphaElementwise:
    """Deprecated alias for :class:`BlendAlphaElementwise`."""
    return BlendAlphaElementwise(*args, **kwargs)


@ia.deprecated(alt_func="BlendAlphaSimplexNoise")
def SimplexNoiseAlpha(*args: object, **kwargs: object) -> BlendAlphaSimplexNoise:
    """Deprecated alias for :class:`BlendAlphaSimplexNoise`."""
    return BlendAlphaSimplexNoise(*args, **kwargs)


@ia.deprecated(alt_func="BlendAlphaFrequencyNoise")
def FrequencyNoiseAlpha(*args: object, **kwargs: object) -> BlendAlphaFrequencyNoise:
    """Deprecated alias for :class:`BlendAlphaFrequencyNoise`."""
    return BlendAlphaFrequencyNoise(*args, **kwargs)


__all__ = [
    "blend_alpha",
    "blend_alpha_",
    "BlendAlpha",
    "BlendAlphaMask",
    "BlendAlphaElementwise",
    "BlendAlphaSimplexNoise",
    "BlendAlphaFrequencyNoise",
    "BlendAlphaSomeColors",
    "BlendAlphaHorizontalLinearGradient",
    "BlendAlphaVerticalLinearGradient",
    "BlendAlphaRegularGrid",
    "BlendAlphaCheckerboard",
    "BlendAlphaSegMapClassIds",
    "BlendAlphaBoundingBoxes",
    "IBatchwiseMaskGenerator",
    "StochasticParameterMaskGen",
    "SomeColorsMaskGen",
    "HorizontalLinearGradientMaskGen",
    "VerticalLinearGradientMaskGen",
    "RegularGridMaskGen",
    "CheckerboardMaskGen",
    "SegMapClassIdsMaskGen",
    "BoundingBoxesMaskGen",
    "InvertMaskGen",
    "Alpha",
    "AlphaElementwise",
    "SimplexNoiseAlpha",
    "FrequencyNoiseAlpha",
    "AlphaInput",
    "PerChannelInput",
    "ChildrenInput",
    "CoordinateAugmentable",
    "UpscaleMethodInput",
    "AggregationMethodInput",
    "SigmoidInput",
    "LabelInput",
    "blend_utils",
]
