"""Shared typing helpers for :mod:`imgaug2.augmenters`.

This module is intentionally small and avoids importing heavy third-party
libraries (e.g. ``cv2``). It provides a handful of common aliases used across
augmenter modules while the type-hints migration is in progress.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

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

