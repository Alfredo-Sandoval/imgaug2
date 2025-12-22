"""Augmenters that affect image colors or image colorspaces.

This module provides augmenters for color manipulation including channel-wise
operations, colorspace conversions, and color-based adjustments.

Key Augmenters:
    - `WithColorspace`, `WithBrightnessChannels`, `WithHueAndSaturation`:
      Apply augmentations in specific colorspaces.
    - `MultiplyAndAddToBrightness`, `MultiplyHue`, `MultiplySaturation`:
      Modify specific color components.
    - `Grayscale`, `RemoveSaturation`, `ChangeColorTemperature`: Color effects.
    - `KMeansColorQuantization`, `UniformColorQuantization`: Reduce colors.
"""

from __future__ import annotations

from ._utils import (
    CSPACE_ALL,
    CSPACE_BGR,
    CSPACE_CIE,
    CSPACE_GRAY,
    CSPACE_HLS,
    CSPACE_HSV,
    CSPACE_Lab,
    CSPACE_Luv,
    CSPACE_RGB,
    CSPACE_YCrCb,
    CSPACE_YUV,
    ChildrenInput,
    ColorSpace,
    ColorSpaceInput,
    KelvinInput,
    PerChannelInput,
    ToColorspaceChoiceInput,
    ToColorspaceParamInput,
)
from .brightness import (
    AddToBrightness,
    MultiplyAndAddToBrightness,
    MultiplyBrightness,
    WithBrightnessChannels,
)
from .colorspace import (
    _CHANGE_COLORSPACE_INPLACE,
    _CSPACE_OPENCV_CONV_VARS,
    ChangeColorspace,
    Grayscale,
    InColorspace,
    WithColorspace,
    change_colorspace_,
    change_colorspaces_,
)
from .hue_saturation import (
    AddToHue,
    AddToHueAndSaturation,
    AddToSaturation,
    MultiplyHue,
    MultiplyHueAndSaturation,
    MultiplySaturation,
    RemoveSaturation,
    WithHueAndSaturation,
)
from .quantization import (
    KMeansColorQuantization,
    Posterize,
    UniformColorQuantization,
    UniformColorQuantizationToNBits,
    quantize_colors_kmeans,
    quantize_colors_uniform,
    quantize_kmeans,
    quantize_uniform,
    quantize_uniform_,
    quantize_uniform_to_n_bits,
    quantize_uniform_to_n_bits_,
    posterize,
)
from .temperature import (
    ChangeColorTemperature,
    change_color_temperature,
    change_color_temperatures_,
)

# Private helpers
from ._utils import _get_arithmetic, _is_mlx_list  # noqa: F401
from .luts import (
    _QuantizeUniformCenterizedLUTTableSingleton,
    _QuantizeUniformLUTTable,
    _QuantizeUniformNotCenterizedLUTTableSingleton,
)  # noqa: F401
from .quantization import _AbstractColorQuantization  # noqa: F401
from .temperature import _KelvinToRGBTable, _KelvinToRGBTableSingleton  # noqa: F401

__all__ = [
    "AddToBrightness",
    "AddToHue",
    "AddToHueAndSaturation",
    "AddToSaturation",
    "CSPACE_ALL",
    "CSPACE_BGR",
    "CSPACE_CIE",
    "CSPACE_GRAY",
    "CSPACE_HLS",
    "CSPACE_HSV",
    "CSPACE_Lab",
    "CSPACE_Luv",
    "CSPACE_RGB",
    "CSPACE_YCrCb",
    "CSPACE_YUV",
    "ChangeColorTemperature",
    "ChangeColorspace",
    "ChildrenInput",
    "ColorSpace",
    "ColorSpaceInput",
    "Grayscale",
    "InColorspace",
    "KelvinInput",
    "KMeansColorQuantization",
    "MultiplyAndAddToBrightness",
    "MultiplyBrightness",
    "MultiplyHue",
    "MultiplyHueAndSaturation",
    "MultiplySaturation",
    "PerChannelInput",
    "Posterize",
    "RemoveSaturation",
    "ToColorspaceChoiceInput",
    "ToColorspaceParamInput",
    "UniformColorQuantization",
    "UniformColorQuantizationToNBits",
    "WithBrightnessChannels",
    "WithColorspace",
    "WithHueAndSaturation",
    "change_color_temperature",
    "change_color_temperatures_",
    "change_colorspace_",
    "change_colorspaces_",
    "posterize",
    "quantize_colors_kmeans",
    "quantize_colors_uniform",
    "quantize_kmeans",
    "quantize_uniform",
    "quantize_uniform_",
    "quantize_uniform_to_n_bits",
    "quantize_uniform_to_n_bits_",
]
