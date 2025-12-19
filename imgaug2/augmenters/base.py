"""Base classes and functions used by all/most augmenters.

This module provides foundational utilities for the augmenters package,
including input validation and warning functions for suspicious array shapes.

Note:
    The main `Augmenter` base class is in `meta.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

import imgaug2.imgaug as ia

Array = NDArray[np.generic]
Images = Array | Sequence[Array]


class SuspiciousMultiImageShapeWarning(UserWarning):
    """Warning multi-image inputs that look like a single image."""


class SuspiciousSingleImageShapeWarning(UserWarning):
    """Warning for single-image inputs that look like multiple images."""


def _warn_on_suspicious_multi_image_shapes(images: Images | None) -> None:
    if images is None:
        return

    # check if it looks like (H, W, C) instead of (N, H, W)
    if ia.is_np_array(images):
        if images.ndim == 3 and images.shape[-1] in [1, 3]:
            ia.warn(
                f"You provided a numpy array of shape {images.shape} as a "
                "multi-image augmentation input, which was interpreted as "
                "(N, H, W). The last dimension however has value 1 or "
                "3, which indicates that you provided a single image "
                "with shape (H, W, C) instead. If that is the case, "
                "you should use e.g. augmenter(image=<your input>) or "
                "augment_image(<your input>) -- note the singular 'image' "
                "instead of 'imageS'. Otherwise your single input image "
                "will be interpreted as multiple images of shape (H, W) "
                "during augmentation.",
                category=SuspiciousMultiImageShapeWarning,
            )


def _warn_on_suspicious_single_image_shape(image: Array | None) -> None:
    if image is None:
        return

    # Check if it looks like (N, H, W) instead of (H, W, C).
    # We don't react to (1, 1, C) though, mostly because that is used in many
    # unittests.
    if image.ndim == 3 and image.shape[-1] >= 32 and image.shape[0:2] != (1, 1):
        ia.warn(
            f"You provided a numpy array of shape {image.shape} as a "
            "single-image augmentation input, which was interpreted as "
            "(H, W, C). The last dimension however has a size of >=32, "
            "which indicates that you provided a multi-image array "
            "with shape (N, H, W) instead. If that is the case, "
            "you should use e.g. augmenter(imageS=<your input>) or "
            "augment_imageS(<your input>). Otherwise your multi-image "
            "input will be interpreted as a single image during "
            "augmentation.",
            category=SuspiciousSingleImageShapeWarning,
        )
