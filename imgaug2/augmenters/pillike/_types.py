from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from imgaug2.augmenters._typing import ParamInput

FillColor: TypeAlias = int | tuple[int, ...] | None
IgnoreValues: TypeAlias = int | Sequence[int] | None
AffineParam: TypeAlias = ParamInput | dict[str, ParamInput]
AffineParamOrNone: TypeAlias = AffineParam | None
