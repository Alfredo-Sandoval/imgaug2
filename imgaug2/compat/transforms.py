"""Dict-based Compose transform wrappers.

Each compat transform:
  - accepts a uniform `p=` probability
  - converts to an imgaug2 augmenter (always-apply)

`compat.Compose` handles probability application via `iaa.Sometimes(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

import imgaug2.augmenters as iaa


def _ensure_tuple2(val: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(val, tuple):
        if len(val) != 2:
            raise ValueError(f"Expected tuple of length 2, got {val!r}")
        return float(val[0]), float(val[1])
    return -float(val), float(val)


def _maybe_tuple(val: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(val, tuple):
        if len(val) != 2:
            raise ValueError(f"Expected tuple of length 2, got {val!r}")
        return float(val[0]), float(val[1])
    return float(val), float(val)


def _to_affine_mode(border_mode: Any) -> str:
    # imgaug2.Affine expects a "mode" string (see docs in geometric.Affine).
    if isinstance(border_mode, str):
        return border_mode

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ValueError("border_mode must be a string when cv2 is unavailable") from exc

    mapping = {
        cv2.BORDER_CONSTANT: "constant",
        cv2.BORDER_REPLICATE: "edge",
        cv2.BORDER_REFLECT: "symmetric",
        cv2.BORDER_REFLECT_101: "reflect",
        cv2.BORDER_WRAP: "wrap",
    }
    if border_mode not in mapping:
        raise ValueError(f"Unsupported border_mode: {border_mode!r}")
    return mapping[border_mode]


@dataclass(frozen=True, slots=True)
class BasicTransform:
    p: float = 0.5

    def to_iaa(self) -> iaa.Augmenter:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class HorizontalFlip(BasicTransform):
    def to_iaa(self) -> iaa.Augmenter:
        return iaa.Fliplr(1.0)


@dataclass(frozen=True, slots=True)
class VerticalFlip(BasicTransform):
    def to_iaa(self) -> iaa.Augmenter:
        return iaa.Flipud(1.0)


@dataclass(frozen=True, slots=True)
class Rotate(BasicTransform):
    limit: float | tuple[float, float] = 90.0
    border_mode: str | int = "reflect"
    value: float | int = 0

    def to_iaa(self) -> iaa.Augmenter:
        low, high = _ensure_tuple2(self.limit)
        return iaa.Affine(
            rotate=(low, high), mode=_to_affine_mode(self.border_mode), cval=self.value
        )


@dataclass(frozen=True, slots=True)
class ShiftScaleRotate(BasicTransform):
    shift_limit: float | tuple[float, float] = 0.0625
    scale_limit: float | tuple[float, float] = 0.1
    rotate_limit: float | tuple[float, float] = 45.0
    border_mode: str | int = "reflect"
    value: float | int = 0

    def to_iaa(self) -> iaa.Augmenter:
        shift_low, shift_high = _ensure_tuple2(self.shift_limit)

        # Compat semantics: float means +/- value around 1.0.
        if isinstance(self.scale_limit, tuple):
            a, b = _maybe_tuple(self.scale_limit)
            scale = (1.0 + a, 1.0 + b)
        else:
            scale = (1.0 - float(self.scale_limit), 1.0 + float(self.scale_limit))

        rot_low, rot_high = _ensure_tuple2(self.rotate_limit)

        return iaa.Affine(
            scale=scale,
            translate_percent={"x": (shift_low, shift_high), "y": (shift_low, shift_high)},
            rotate=(rot_low, rot_high),
            mode=_to_affine_mode(self.border_mode),
            cval=self.value,
        )


@dataclass(frozen=True, slots=True)
class RandomBrightnessContrast(BasicTransform):
    brightness_limit: float | tuple[float, float] = 0.2
    contrast_limit: float | tuple[float, float] = 0.2

    def to_iaa(self) -> iaa.Augmenter:
        # Common convention: beta in [-b, b] * 255, alpha in [1-c, 1+c].
        if isinstance(self.brightness_limit, tuple):
            bl, bh = _maybe_tuple(self.brightness_limit)
            add = (bl * 255.0, bh * 255.0)
        else:
            b = float(self.brightness_limit)
            add = (-b * 255.0, b * 255.0)

        if isinstance(self.contrast_limit, tuple):
            cl, ch = _maybe_tuple(self.contrast_limit)
            mul = (1.0 + cl, 1.0 + ch)
        else:
            c = float(self.contrast_limit)
            mul = (1.0 - c, 1.0 + c)

        # Multiply then add (common fast path).
        return iaa.Sequential([iaa.Multiply(mul), iaa.Add(add)])
