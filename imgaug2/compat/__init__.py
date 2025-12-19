"""Compatibility layer for dict-based augmentation APIs.

This module provides Albumentations-compatible interface for imgaug2, enabling
seamless migration and interoperability between libraries. It wraps core imgaug2
augmenters behind a dict-based Compose interface with the following features:

- Dict-based input/output for synchronized augmentation
- Uniform probability parameter (p) across all transforms
- Common bounding box formats (Pascal VOC, COCO, YOLO, normalized xyxy)
- Keypoint augmentation with extra fields preservation
- Label field synchronization during filtering

Examples
--------
Basic usage with bounding boxes and keypoints:

    >>> from imgaug2 import compat as A
    >>> transform = A.Compose(
    ...     [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2)],
    ...     bbox_params=A.BboxParams(format="coco"),
    ...     keypoint_params=A.KeypointParams(),
    ... )
    >>> out = transform(image=image, bboxes=bboxes, keypoints=keypoints)
    >>> image_aug = out["image"]
    >>> bboxes_aug = out["bboxes"]

Notes
-----
This module uses lazy imports to avoid circular dependencies. Heavy imports
are deferred until first access via __getattr__.
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


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazily import compat submodules to avoid circular imports.

    Parameters
    ----------
    name : str
        Name of the attribute to import.

    Returns
    -------
    Any
        The imported module or attribute.

    Raises
    ------
    AttributeError
        If the requested attribute is not in _LAZY_IMPORTS.
    """
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return all exported names, including lazy imports.

    Returns
    -------
    list of str
        Sorted list of all available attribute names.
    """
    return sorted(set(list(globals().keys()) + list(_LAZY_IMPORTS.keys())))
