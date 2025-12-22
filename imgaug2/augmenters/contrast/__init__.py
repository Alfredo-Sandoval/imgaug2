"""Augmenters that perform contrast changes.

This module provides augmenters for adjusting image contrast using various
algorithms including histogram-based and curve-based methods.

Key Augmenters:
    - `GammaContrast`, `SigmoidContrast`, `LogContrast`, `LinearContrast`:
      Apply various contrast curve transformations.
    - `HistogramEqualization`, `AllChannelsHistogramEqualization`: Equalize histograms.
    - `CLAHE`, `AllChannelsCLAHE`: Contrast Limited Adaptive Histogram Equalization.
"""

from __future__ import annotations

from ._intensity import _IntensityChannelBasedApplier
from ._types import (
    ContrastFunc,
    DTypeStrs,
    FloatArray,
    IntensityChannelFunc,
    KernelSize,
    KernelSizeParamInput,
    KernelSizeParamInput2D,
)
from ._utils import _is_mlx_list
from .clahe import AllChannelsCLAHE, CLAHE
from .curves import (
    _ContrastFuncWrapper,
    GammaContrast,
    LinearContrast,
    LogContrast,
    SigmoidContrast,
    adjust_contrast_gamma,
    adjust_contrast_linear,
    adjust_contrast_log,
    adjust_contrast_sigmoid,
)
from .histogram import AllChannelsHistogramEqualization, HistogramEqualization

__all__ = [
    "KernelSize",
    "FloatArray",
    "DTypeStrs",
    "ContrastFunc",
    "KernelSizeParamInput",
    "KernelSizeParamInput2D",
    "IntensityChannelFunc",
    "adjust_contrast_gamma",
    "adjust_contrast_sigmoid",
    "adjust_contrast_log",
    "adjust_contrast_linear",
    "GammaContrast",
    "SigmoidContrast",
    "LogContrast",
    "LinearContrast",
    "AllChannelsCLAHE",
    "CLAHE",
    "AllChannelsHistogramEqualization",
    "HistogramEqualization",
]
