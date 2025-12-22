from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import imgaug2.parameters as iap
from imgaug2.augmenters._typing import Array, Number

ScalarInput: TypeAlias = Number | Array
PerChannelInput: TypeAlias = bool | float | iap.StochasticParameter
CValInput: TypeAlias = Number | Sequence[Number] | Array
FillModeInput: TypeAlias = Literal["constant", "gaussian"]
IntParamInput: TypeAlias = int | tuple[int, int] | list[int] | iap.StochasticParameter
SizePxInput: TypeAlias = None | int | tuple[int, int] | iap.StochasticParameter
SizePercentInput: TypeAlias = None | float | tuple[float, float] | iap.StochasticParameter
PositionInput: TypeAlias = (
    str
    | tuple[float | int | iap.StochasticParameter, float | int | iap.StochasticParameter]
    | iap.StochasticParameter
)
