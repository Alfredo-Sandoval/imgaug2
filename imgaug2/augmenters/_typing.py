"""Shared typing helpers for :mod:`imgaug2.augmenters`.

This module is intentionally small and avoids importing heavy third-party
libraries (e.g. ``cv2``). It provides a handful of common aliases used across
augmenter modules while the type-hints migration is in progress.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

import imgaug2.parameters as iap

# ---- NumPy arrays / images -------------------------------------------------

# Generic numpy array with unknown dtype/shape. We prefer `np.generic` over
# `Any` here as most augmenter code is dtype-polymorphic.
Array: TypeAlias = NDArray[np.generic]

# Images are numpy arrays of shape `(H, W)` or `(H, W, C)` (and related).
Image: TypeAlias = Array

# Batches are often `(N, H, W, C)` but we keep the shape generic.
ImageBatch: TypeAlias = Array

# Many augmenter APIs accept either a numpy batch array or a list/tuple of
# per-image arrays.
Images: TypeAlias = ImageBatch | Sequence[Image]


# ---- Common parameter union types -----------------------------------------

Number: TypeAlias = float | int

# Historically, many augmenters accept raw numbers or stochastic parameters.
Numberish: TypeAlias = Number | iap.StochasticParameter

# Common pattern for "scalar or (min,max) or list or stochastic param".
ParamInput: TypeAlias = Number | tuple[Number, Number] | list[Number] | iap.StochasticParameter


# ---- Geometrical Types ----------------------------------------------------

# (H, W) or (H, W, C)
Shape2D: TypeAlias = tuple[int, int]
Shape: TypeAlias = tuple[int, int] | tuple[int, int, int]

# 3x3 transformation matrix
Matrix: TypeAlias = NDArray[np.floating]

if TYPE_CHECKING:
    from imgaug2.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
    from imgaug2.augmentables.heatmaps import HeatmapsOnImage
    from imgaug2.augmentables.kps import Keypoint, KeypointsOnImage
    from imgaug2.augmentables.lines import LineString, LineStringsOnImage
    from imgaug2.augmentables.polys import Polygon, PolygonsOnImage
    from imgaug2.augmentables.segmaps import SegmentationMapsOnImage
    from imgaug2.augmenters.geometric import _AffineSamplingResult

    _AffineSamplingResultVar = _AffineSamplingResult
else:
    _AffineSamplingResultVar = object

_AffineSamplingResultType: TypeAlias = _AffineSamplingResultVar


# ---- RNG inputs -----------------------------------------------------------

if TYPE_CHECKING:
    from numpy.random import BitGenerator, Generator, SeedSequence

    from imgaug2.random import RNG

    RNGInput: TypeAlias = None | int | RNG | Generator | BitGenerator | SeedSequence
else:
    # Used only for runtime type-alias binding; due to `from __future__ import annotations`,
    # function annotations are stored as strings and won't evaluate this at runtime.
    RNGInput: TypeAlias = object
