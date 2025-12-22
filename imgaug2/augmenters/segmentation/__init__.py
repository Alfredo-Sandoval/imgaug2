"""Augmenters that apply changes to images based on segmentation methods.

This module contains augmenters that use image segmentation techniques
to create stylized or abstract versions of images.

Key Augmenters:
    - `Superpixels`: Replace superpixel regions with their average color.
    - `Voronoi`: Average colors within Voronoi cells.
    - `UniformVoronoi`, `RegularGridVoronoi`: Voronoi variants with different
      point sampling strategies.
"""

from __future__ import annotations

from imgaug2.imgaug import _NUMBA_INSTALLED

from ._utils import _ensure_image_max_size
from .replace import replace_segments_
from .samplers import (
    IPointsSampler,
    DropoutPointsSampler,
    RegularGridPointsSampler,
    RelativeRegularGridPointsSampler,
    SubsamplingPointsSampler,
    UniformPointsSampler,
)
from .superpixels import Superpixels
from .voronoi import (
    RegularGridVoronoi,
    RelativeRegularGridVoronoi,
    UniformVoronoi,
    Voronoi,
    segment_voronoi,
)

__all__ = [
    "Superpixels",
    "segment_voronoi",
    "replace_segments_",
    "Voronoi",
    "UniformVoronoi",
    "RegularGridVoronoi",
    "RelativeRegularGridVoronoi",
    "IPointsSampler",
    "RegularGridPointsSampler",
    "RelativeRegularGridPointsSampler",
    "DropoutPointsSampler",
    "UniformPointsSampler",
    "SubsamplingPointsSampler",
]
