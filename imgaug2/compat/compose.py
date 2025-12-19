"""Dict-based composition for imgaug2 compatibility layer.

This module implements the Compose class that orchestrates dict-based transforms,
handles parameter conversion, and manages synchronized augmentation of images,
bounding boxes, and keypoints with their associated label fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import imgaug2.augmenters as iaa

from .bbox import BboxParams, convert_bboxes_from_imgaug, convert_bboxes_to_imgaug, filter_bboxes
from .keypoint import (
    KeypointParams,
    convert_keypoints_from_imgaug,
    convert_keypoints_to_imgaug,
    filter_keypoints,
)
from .transforms import BasicTransform


def _normalize_p(p: float) -> float:
    """Validate and normalize probability parameter.

    Parameters
    ----------
    p : float
        Probability value to validate.

    Returns
    -------
    float
        Validated probability in range [0, 1].

    Raises
    ------
    ValueError
        If p is not in range [0, 1].
    """
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    return p


@dataclass(frozen=True, slots=True)
class Compose:
    """Compose multiple transforms with dict-based I/O.

    This class provides an Albumentations-compatible interface for chaining
    transforms and applying them to images, bboxes, and keypoints in a single call.

    Parameters
    ----------
    transforms : sequence of BasicTransform
        List of transforms to apply in sequence.
    bbox_params : BboxParams or None, default=None
        Bounding box format and filtering parameters.
    keypoint_params : KeypointParams or None, default=None
        Keypoint filtering parameters.
    p : float, default=1.0
        Probability of applying the entire composition.

    Examples
    --------
    Basic usage with probability control:

        >>> import numpy as np
        >>> from imgaug2.compat import Compose, HorizontalFlip, Rotate
        >>> transform = Compose([
        ...     HorizontalFlip(p=0.5),
        ...     Rotate(limit=45, p=0.3)
        ... ], p=0.8)
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> bboxes = []
        >>> result = transform(image=img, bboxes=bboxes)

    Notes
    -----
    This compatibility layer is designed for single-sample augmentation.
    For batch processing, use the native imgaug2 API.
    """

    transforms: tuple[BasicTransform, ...]
    bbox_params: BboxParams | None = None
    keypoint_params: KeypointParams | None = None
    p: float = 1.0

    def __init__(
        self,
        transforms: Sequence[BasicTransform],
        bbox_params: BboxParams | None = None,
        keypoint_params: KeypointParams | None = None,
        p: float = 1.0,
    ) -> None:
        object.__setattr__(self, "transforms", tuple(transforms))
        object.__setattr__(self, "bbox_params", bbox_params)
        object.__setattr__(self, "keypoint_params", keypoint_params)
        object.__setattr__(self, "p", _normalize_p(p))

    def _build_iaa(self) -> iaa.Augmenter:
        """Build imgaug2 augmenter from transform list.

        Returns
        -------
        iaa.Augmenter
            Sequential augmenter with probability wrapping applied.
        """
        augs: list[iaa.Augmenter] = []

        for t in self.transforms:
            p = _normalize_p(t.p)
            if p == 0.0:
                continue
            aug = t.to_iaa()
            if p < 1.0:
                aug = iaa.Sometimes(p, aug)
            augs.append(aug)

        seq: iaa.Augmenter = iaa.Sequential(augs) if len(augs) > 0 else iaa.Noop()
        if self.p < 1.0:
            seq = iaa.Sometimes(self.p, seq)
        return seq

    def __call__(self, **data: Any) -> dict[str, Any]:  # noqa: ANN401, ANN003
        """Apply transforms to input data.

        Parameters
        ----------
        **data : dict
            Input data containing 'image' (required) and optional 'bboxes',
            'keypoints', and label fields. Unknown keys are passed through.

        Returns
        -------
        dict
            Augmented data with same structure as input. Contains augmented
            'image' and any provided bboxes/keypoints with synchronized labels.

        Raises
        ------
        KeyError
            If 'image' key is not provided or required label fields are missing.
        TypeError
            If bboxes or keypoints are not list/tuple.
        ValueError
            If inline labels and label_fields are both specified for bboxes.

        Notes
        -----
        Label fields are automatically filtered to match kept bboxes/keypoints.
        Pass-through keys preserve user metadata across augmentation.
        """
        if "image" not in data:
            raise KeyError("Compose expects an `image=` keyword argument.")

        image = data["image"]

        out: dict[str, Any] = {
            k: v for k, v in data.items() if k not in {"image", "bboxes", "keypoints"}
        }

        bbox_params = self.bbox_params
        keypoint_params = self.keypoint_params

        bboxes_in = data.get("bboxes", None)
        keypoints_in = data.get("keypoints", None)

        label_fields = bbox_params.label_fields if bbox_params is not None else ()
        label_field_values: dict[str, Any] = {}
        for field in label_fields:
            if field in data:
                label_field_values[field] = data[field]

        if bboxes_in is not None and bbox_params is None:
            bbox_params = BboxParams()

        if keypoints_in is not None and keypoint_params is None:
            keypoint_params = KeypointParams()

        bbox_params_for_bboxes: BboxParams | None = None
        bbs = None
        bbox_inline_labels = None
        if bboxes_in is not None:
            if not isinstance(bboxes_in, (list, tuple)):
                raise TypeError("bboxes must be a list/tuple of bboxes")
            bbox_params_for_bboxes = bbox_params or BboxParams()
            bbs, bbox_inline_labels = convert_bboxes_to_imgaug(
                bboxes_in,
                image.shape,
                format=bbox_params_for_bboxes.format,
            )

        kps = None
        kps_extras = None
        if keypoints_in is not None:
            if not isinstance(keypoints_in, (list, tuple)):
                raise TypeError("keypoints must be a list/tuple of keypoints")
            kps, kps_extras = convert_keypoints_to_imgaug(keypoints_in, image.shape)

        aug = self._build_iaa().to_deterministic()

        image_aug = aug(image=image)
        out["image"] = image_aug

        if bbs is not None:
            bbs_aug = aug(bounding_boxes=bbs)

            if bbox_params_for_bboxes is not None:
                external_labels: list[tuple[Any, ...]] | None = None
                if label_fields:
                    per_field_lists: list[list[Any]] = []
                    for field in label_fields:
                        values = data.get(field, None)
                        if values is None:
                            raise KeyError(
                                f"Expected `{field}` because bbox_params.label_fields contains it."
                            )
                        if not isinstance(values, (list, tuple)):
                            raise TypeError(
                                f"Expected `{field}` to be a list/tuple, got {type(values)!r}"
                            )
                        per_field_lists.append(list(values))
                    external_labels = list(zip(*per_field_lists, strict=True))

                if bbox_inline_labels is not None and external_labels is not None:
                    raise ValueError(
                        "Specify bbox labels either inline (5th element in bboxes) OR via label_fields, not both."
                    )

                labels_for_filter: list[Any] | None = None
                if bbox_inline_labels is not None:
                    labels_for_filter = list(bbox_inline_labels)
                elif external_labels is not None:
                    labels_for_filter = list(external_labels)

                bbs_f, labels_f, keep_mask = filter_bboxes(
                    bbs_aug, bbox_params, inline_labels=labels_for_filter
                )

                if label_fields and labels_f is not None:
                    per_field: list[list[Any]] = [[] for _ in label_fields]
                    for lbl in labels_f:
                        if not isinstance(lbl, tuple) or len(lbl) != len(label_fields):
                            raise ValueError("Internal label_fields shape mismatch")
                        for j in range(len(label_fields)):
                            per_field[j].append(lbl[j])
                    for j, field in enumerate(label_fields):
                        out[field] = per_field[j]
                    out["bboxes"] = convert_bboxes_from_imgaug(
                        bbs_f, format=bbox_params_for_bboxes.format
                    )
                else:
                    out["bboxes"] = convert_bboxes_from_imgaug(
                        bbs_f,
                        format=bbox_params_for_bboxes.format,
                        inline_labels=labels_f,
                    )
            else:
                out["bboxes"] = convert_bboxes_from_imgaug(bbs_aug, format="pascal_voc")

        if kps is not None:
            kps_aug = aug(keypoints=kps)
            if keypoint_params is not None:
                kps_f, extras_f = filter_keypoints(
                    kps_aug, keypoint_params, extras=list(kps_extras)
                )
                out["keypoints"] = convert_keypoints_from_imgaug(kps_f, extras=extras_f)
            else:
                out["keypoints"] = convert_keypoints_from_imgaug(kps_aug, extras=kps_extras)

        return out
