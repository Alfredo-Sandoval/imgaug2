"""Dict-based Compose compatibility API (dict-based I/O).

This module is intentionally *small* and pragmatic: it wraps core imgaug2
augmenters behind a dict-based Compose interface:

    - Dict-based input/output (single call keeps everything in sync)
    - Uniform `p=` probability on transforms
    - Common bbox formats (Pascal VOC / COCO / YOLO / normalized xyxy)

Example:

    from imgaug2 import compat as A

    transform = A.Compose(
        [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2)],
        bbox_params=A.BboxParams(format="coco"),
        keypoint_params=A.KeypointParams(),
    )

    out = transform(image=image, bboxes=bboxes, keypoints=keypoints)
    image_aug = out["image"]
    bboxes_aug = out["bboxes"]
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BboxParams",
    "Compose",
    "HorizontalFlip",
    "KeypointParams",
    "RandomBrightnessContrast",
    "Rotate",
    "ShiftScaleRotate",
    "VerticalFlip",
    # Markers for legacy vs new code
    "get_marker",
    "is_legacy",
    "is_new",
    "legacy",
    "new",
]

# NOTE: Keep this module import-light.
#
# `imgaug2.imgaug` imports `imgaug2.compat.markers` to decorate legacy/new
# functions. Importing any heavy compat modules here would trigger a circular
# import (compat -> augmentables -> imgaug) before `imgaug2.imgaug.deprecated`
# is defined. Hence we lazily import everything via __getattr__.

_LAZY_IMPORTS: dict[str, str] = {
    # Core API
    "BboxParams": ".bbox",
    "Compose": ".compose",
    "KeypointParams": ".keypoint",
    # Transforms
    "HorizontalFlip": ".transforms",
    "RandomBrightnessContrast": ".transforms",
    "Rotate": ".transforms",
    "ShiftScaleRotate": ".transforms",
    "VerticalFlip": ".transforms",
    # Markers
    "get_marker": ".markers",
    "is_legacy": ".markers",
    "is_new": ".markers",
    "legacy": ".markers",
    "new": ".markers",
}


def __getattr__(name: str) -> Any:
    """Lazily import compat submodules to avoid circular imports."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return all exported names, including lazy imports."""
    return sorted(set(list(globals().keys()) + list(_LAZY_IMPORTS.keys())))
