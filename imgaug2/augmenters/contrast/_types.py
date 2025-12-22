"""Shared type aliases for contrast augmenters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmenters._typing import Array

KernelSize: TypeAlias = int | tuple[int, int]
FloatArray: TypeAlias = NDArray[np.floating]
DTypeStrs: TypeAlias = str | Sequence[str]
ContrastFunc: TypeAlias = Callable[..., Array]
KernelSizeParamInput: TypeAlias = int | tuple[int, int] | list[int] | iap.StochasticParameter
KernelSizeParamInput2D: TypeAlias = (
    KernelSizeParamInput | tuple[KernelSizeParamInput, KernelSizeParamInput]
)
IntensityChannelFunc: TypeAlias = Callable[[list[Array | None], iarandom.RNG], list[Array]]
