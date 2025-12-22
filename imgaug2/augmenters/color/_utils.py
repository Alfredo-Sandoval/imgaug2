from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Literal, TypeAlias

import imgaug2.parameters as iap
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array
from imgaug2.mlx._core import is_mlx_array

CSPACE_RGB = "RGB"
CSPACE_BGR = "BGR"
CSPACE_GRAY = "GRAY"
CSPACE_YCrCb = "YCrCb"
CSPACE_HSV = "HSV"
CSPACE_HLS = "HLS"
CSPACE_Lab = "Lab"  # aka CIELAB
# TODO add Luv to various color/contrast augmenters as random default choice?
CSPACE_Luv = "Luv"  # aka CIE 1976, aka CIELUV
CSPACE_YUV = "YUV"  # aka CIE 1960
CSPACE_CIE = "CIE"  # aka CIE 1931, aka XYZ in OpenCV
CSPACE_ALL = {
    CSPACE_RGB,
    CSPACE_BGR,
    CSPACE_GRAY,
    CSPACE_YCrCb,
    CSPACE_HSV,
    CSPACE_HLS,
    CSPACE_Lab,
    CSPACE_Luv,
    CSPACE_YUV,
    CSPACE_CIE,
}

ColorSpace: TypeAlias = Literal[
    "RGB",
    "BGR",
    "GRAY",
    "YCrCb",
    "HSV",
    "HLS",
    "Lab",
    "Luv",
    "YUV",
    "CIE",
]
ColorSpaceInput: TypeAlias = ColorSpace | Sequence[ColorSpace]
KelvinInput: TypeAlias = float | int | Sequence[float | int] | Array
PerChannelInput: TypeAlias = bool | float | iap.StochasticParameter
ChildrenInput: TypeAlias = meta.Augmenter | Sequence[meta.Augmenter] | None
ToColorspaceParamInput: TypeAlias = Literal["ALL"] | ColorSpaceInput | iap.StochasticParameter
ToColorspaceChoiceInput: TypeAlias = ColorSpaceInput | iap.StochasticParameter

_ARITHMETIC: ModuleType | None = None


def _is_mlx_list(images: object) -> bool:
    return isinstance(images, Sequence) and len(images) > 0 and is_mlx_array(images[0])


def _get_arithmetic() -> ModuleType:
    global _ARITHMETIC
    if _ARITHMETIC is None:
        from imgaug2.augmenters import arithmetic as arithmetic_lib

        _ARITHMETIC = arithmetic_lib
    return _ARITHMETIC
