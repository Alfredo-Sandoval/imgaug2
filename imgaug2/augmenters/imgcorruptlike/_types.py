from __future__ import annotations

from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from imgaug2.augmenters._typing import Array

IntArray: TypeAlias = NDArray[np.integer]
UIntArray: TypeAlias = NDArray[np.unsignedinteger]


class CorruptionFunc(Protocol):
    def __call__(self, x: Array, severity: int = 1, seed: int | None = None) -> Array: ...
