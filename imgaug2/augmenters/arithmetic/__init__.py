"""Augmenters that perform simple arithmetic changes.

This module provides augmenters for pixel-level arithmetic operations including
adding/multiplying values, dropout, noise, and compression artifacts.

Key Augmenters:
    - `Add`, `Multiply`, `AddElementwise`, `MultiplyElementwise`: Arithmetic ops.
    - `Dropout`, `CoarseDropout`, `Cutout`, `ChannelShuffle`: Random masking.
    - `SaltAndPepper`, `AdditiveGaussianNoise`, `AdditiveLaplaceNoise`: Noise.
    - `JpegCompression`: Add JPEG compression artifacts.
    - `Solarize`, `Invert`, `Posterize`: Pixel value transformations.
"""

from __future__ import annotations

from ._utils import (
    CValInput,
    FillModeInput,
    IntParamInput,
    PerChannelInput,
    PositionInput,
    ScalarInput,
    SizePercentInput,
    SizePxInput,
)
from .add import (
    Add,
    AddElementwise,
    AdditiveGaussianNoise,
    AdditiveLaplaceNoise,
    AdditivePoissonNoise,
    add_elementwise,
    add_scalar,
    add_scalar_,
)
from .dropout import (
    CoarseDropout,
    Cutout,
    Dropout,
    Dropout2d,
    TotalDropout,
    cutout,
    cutout_,
)
from .invert import (
    ContrastNormalization,
    Invert,
    Solarize,
    invert,
    invert_,
    solarize,
    solarize_,
)
from .jpeg import JpegCompression, compress_jpeg
from .multiply import (
    Multiply,
    MultiplyElementwise,
    multiply_elementwise,
    multiply_elementwise_,
    multiply_scalar,
    multiply_scalar_,
)
from .replace import (
    CoarsePepper,
    CoarseSalt,
    CoarseSaltAndPepper,
    ImpulseNoise,
    Pepper,
    ReplaceElementwise,
    Salt,
    SaltAndPepper,
    replace_elementwise_,
)

# Private helpers used in tests or internal callers
from .add import (
    _add_elementwise_cv2_to_uint8,
    _add_elementwise_np_to_non_uint8,
    _add_elementwise_np_to_uint8,
    _add_scalar_to_non_uint8,
    _add_scalar_to_uint8_,
)  # noqa: F401
from .dropout import (
    _CUTOUT_FILL_MODES,
    _CutoutSamples,
    _fill_rectangle_constant_,
    _fill_rectangle_gaussian_,
    _handle_dropout_probability_param,
)  # noqa: F401
from .invert import (
    _InvertSamples,
    _InvertTables,
    _InvertTablesSingleton,
    _generate_table_for_invert_uint8,
    _invert_bool,
    _invert_by_distance,
    _invert_float,
    _invert_int_,
    _invert_uint16_or_larger_,
    _invert_uint8_,
    _invert_uint8_lut_pregenerated_,
    _invert_uint8_subtract_,
)  # noqa: F401
from .multiply import (
    _multiply_elementwise_to_non_uint8,
    _multiply_elementwise_to_uint8_,
    _multiply_scalar_to_non_uint8,
    _multiply_scalar_to_uint8_cv2_mul_,
    _multiply_scalar_to_uint8_lut_,
)  # noqa: F401

__all__ = [
    "Add",
    "AddElementwise",
    "AdditiveGaussianNoise",
    "AdditiveLaplaceNoise",
    "AdditivePoissonNoise",
    "CValInput",
    "ContrastNormalization",
    "CoarseDropout",
    "CoarsePepper",
    "CoarseSalt",
    "CoarseSaltAndPepper",
    "Cutout",
    "Dropout",
    "Dropout2d",
    "FillModeInput",
    "ImpulseNoise",
    "IntParamInput",
    "Invert",
    "JpegCompression",
    "Multiply",
    "MultiplyElementwise",
    "Pepper",
    "PerChannelInput",
    "PositionInput",
    "ReplaceElementwise",
    "Salt",
    "SaltAndPepper",
    "ScalarInput",
    "Solarize",
    "SizePercentInput",
    "SizePxInput",
    "TotalDropout",
    "add_elementwise",
    "add_scalar",
    "add_scalar_",
    "compress_jpeg",
    "cutout",
    "cutout_",
    "invert",
    "invert_",
    "multiply_elementwise",
    "multiply_elementwise_",
    "multiply_scalar",
    "multiply_scalar_",
    "replace_elementwise_",
    "solarize",
    "solarize_",
]
