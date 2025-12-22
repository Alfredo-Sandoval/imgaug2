"""Augmenters that wrap methods from the ``imagecorruptions`` package.

This module provides augmenters for common image corruptions used to benchmark
neural network robustness. Derived from the imagecorruptions package based on
Hendrycks & Dietterich's research on robustness benchmarking.

Key Augmenters:
    - Noise: `GaussianNoise`, `ShotNoise`, `ImpulseNoise`, `SpeckleNoise`
    - Blur: `DefocusBlur`, `GlassBlur`, `MotionBlur`, `ZoomBlur`, `GaussianBlur`
    - Weather: `Snow`, `Frost`, `Fog`, `Spatter`
    - Digital: `Contrast`, `Brightness`, `Saturate`, `JpegCompression`, `Pixelate`
    - Special: `ElasticTransform`

Note:
    Outputs are identical to ``imagecorruptions.corrupt()`` (always ``uint8``).

References:
    - Package: https://github.com/bethgelab/imagecorruptions
    - Paper: https://arxiv.org/abs/1903.12261

Example::

    >>> import numpy as np
    >>> import imgaug2.augmenters as iaa
    >>> images = [np.zeros((64, 64, 3), dtype=np.uint8)]
    >>> aug = iaa.imgcorruptlike.GaussianNoise(severity=2)
    >>> image_aug = aug(images=images)
"""

from __future__ import annotations

from ._types import CorruptionFunc, IntArray, UIntArray
from .core import (
    _MISSING_PACKAGE_ERROR_MSG,
    _call_imgcorrupt_func,
    _clipped_zoom_no_scipy_warning,
    _gaussian_skimage_compat,
    _patch_imagecorruptions_modules_,
)
from .registry import get_corruption_names
from .noise import (
    GaussianNoise,
    ImpulseNoise,
    ShotNoise,
    SpeckleNoise,
    apply_gaussian_noise,
    apply_impulse_noise,
    apply_shot_noise,
    apply_speckle_noise,
)
from .blur import (
    DefocusBlur,
    GaussianBlur,
    GlassBlur,
    MotionBlur,
    ZoomBlur,
    _apply_glass_blur_imgaug,
    _apply_glass_blur_imgaug_loop,
    apply_defocus_blur,
    apply_gaussian_blur,
    apply_glass_blur,
    apply_motion_blur,
    apply_zoom_blur,
)
from .weather import (
    Fog,
    Frost,
    Snow,
    Spatter,
    apply_fog,
    apply_frost,
    apply_snow,
    apply_spatter,
)
from .digital import (
    Brightness,
    Contrast,
    JpegCompression,
    Pixelate,
    Saturate,
    apply_brightness,
    apply_contrast,
    apply_jpeg_compression,
    apply_pixelate,
    apply_saturate,
)
from .special import ElasticTransform, apply_elastic_transform
from .base import _ImgcorruptAugmenterBase

__all__ = [
    "get_corruption_names",
    "apply_gaussian_noise",
    "apply_shot_noise",
    "apply_impulse_noise",
    "apply_speckle_noise",
    "apply_gaussian_blur",
    "apply_glass_blur",
    "apply_defocus_blur",
    "apply_motion_blur",
    "apply_zoom_blur",
    "apply_fog",
    "apply_frost",
    "apply_snow",
    "apply_spatter",
    "apply_contrast",
    "apply_brightness",
    "apply_saturate",
    "apply_jpeg_compression",
    "apply_pixelate",
    "apply_elastic_transform",
    "GaussianNoise",
    "ShotNoise",
    "ImpulseNoise",
    "SpeckleNoise",
    "GaussianBlur",
    "GlassBlur",
    "DefocusBlur",
    "MotionBlur",
    "ZoomBlur",
    "Fog",
    "Frost",
    "Snow",
    "Spatter",
    "Contrast",
    "Brightness",
    "Saturate",
    "JpegCompression",
    "Pixelate",
    "ElasticTransform",
]
