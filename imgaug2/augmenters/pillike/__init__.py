"""Augmenters that have identical outputs to well-known PIL functions.

This module provides augmenters with outputs identical to corresponding PIL
functions, though they may use faster internal techniques. Use these when
exact PIL compatibility is required.

Key Augmenters:
    - `Autocontrast`: Normalize image contrast like PIL's autocontrast.
    - `Equalize`: Histogram equalization like PIL's equalize.
    - `EnhanceColor`, `EnhanceContrast`, `EnhanceBrightness`, `EnhanceSharpness`:
      Image enhancement operations matching PIL's ImageEnhance module.
    - `FilterBlur`, `FilterSharpen`, `FilterEdgeEnhance`, `FilterEmboss`:
      Filter operations matching PIL's ImageFilter module.
    - `Affine`: Affine transformations compatible with PIL's transform.
"""

from __future__ import annotations

from ._types import AffineParam, AffineParamOrNone, FillColor, IgnoreValues
from ._utils import _ensure_valid_shape, _maybe_mlx
from .affine import Affine, _create_affine_matrix, warp_affine
from .autocontrast import Autocontrast, _autocontrast_no_pil, _autocontrast_pil, autocontrast
from .color_ops import Posterize, Solarize, posterize, posterize_, solarize, solarize_
from .enhance import (
    EnhanceBrightness,
    EnhanceColor,
    EnhanceContrast,
    EnhanceSharpness,
    _EnhanceBase,
    _EnhanceCtor,
    _apply_enhance_func,
    enhance_brightness,
    enhance_color,
    enhance_contrast,
    enhance_sharpness,
)
from .equalize import Equalize, _equalize_no_pil_, _equalize_pil_, equalize, equalize_
from .filters import (
    FilterBlur,
    FilterContour,
    FilterDetail,
    FilterEdgeEnhance,
    FilterEdgeEnhanceMore,
    FilterEmboss,
    FilterFindEdges,
    FilterSharpen,
    FilterSmooth,
    FilterSmoothMore,
    _FilterBase,
    _filter_by_kernel,
    filter_blur,
    filter_contour,
    filter_detail,
    filter_edge_enhance,
    filter_edge_enhance_more,
    filter_emboss,
    filter_find_edges,
    filter_sharpen,
    filter_smooth,
    filter_smooth_more,
)

__all__ = [
    "Affine",
    "AffineParam",
    "AffineParamOrNone",
    "Autocontrast",
    "EnhanceBrightness",
    "EnhanceColor",
    "EnhanceContrast",
    "EnhanceSharpness",
    "Equalize",
    "FillColor",
    "FilterBlur",
    "FilterContour",
    "FilterDetail",
    "FilterEdgeEnhance",
    "FilterEdgeEnhanceMore",
    "FilterEmboss",
    "FilterFindEdges",
    "FilterSharpen",
    "FilterSmooth",
    "FilterSmoothMore",
    "IgnoreValues",
    "Posterize",
    "Solarize",
    "autocontrast",
    "enhance_brightness",
    "enhance_color",
    "enhance_contrast",
    "enhance_sharpness",
    "equalize",
    "equalize_",
    "filter_blur",
    "filter_contour",
    "filter_detail",
    "filter_edge_enhance",
    "filter_edge_enhance_more",
    "filter_emboss",
    "filter_find_edges",
    "filter_sharpen",
    "filter_smooth",
    "filter_smooth_more",
    "posterize",
    "posterize_",
    "solarize",
    "solarize_",
    "warp_affine",
]
