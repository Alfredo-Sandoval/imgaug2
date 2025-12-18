"""Dict-based composition for imgaug2 (`imgaug2.compat`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    return p


@dataclass(frozen=True, slots=True)
class Compose:
    """Compose a list of `compat` transforms.

    This follows a dict-based calling convention:

        out = transform(image=image, bboxes=bboxes, keypoints=keypoints)
        image_aug = out["image"]

    Notes:
    - This compat layer is meant for *single samples* (one image per call).
      If you need batched augmentation, use the native imgaug2 API.
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

    def __call__(self, **data: Any) -> dict[str, Any]:
        if "image" not in data:
            raise KeyError("Compose expects an `image=` keyword argument.")

        image = data["image"]

        # Pass-through for unknown keys (keeps user metadata intact).
        out: dict[str, Any] = {
            k: v for k, v in data.items() if k not in {"image", "bboxes", "keypoints"}
        }

        bbox_params = self.bbox_params
        keypoint_params = self.keypoint_params

        bboxes_in = data.get("bboxes", None)
        keypoints_in = data.get("keypoints", None)

        # Labels can be stored inline or in separate fields.
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
                # Combine inline labels and external label fields into one per-bbox label payload.
                external_labels: list[tuple[Any, ...]] | None = None
                if label_fields:
                    # Expect a list-like per field.
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
                    # Zip into per-bbox tuple, kept separate from inline labels.
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

                # Restore label_fields if used.
                if label_fields and labels_f is not None:
                    # labels_f entries are tuples of label_fields values.
                    per_field: list[list[Any]] = [[] for _ in label_fields]
                    for lbl in labels_f:
                        if not isinstance(lbl, tuple) or len(lbl) != len(label_fields):
                            raise ValueError("Internal label_fields shape mismatch")
                        for j in range(len(label_fields)):
                            per_field[j].append(lbl[j])
                    for j, field in enumerate(label_fields):
                        out[field] = per_field[j]
                    # bboxes returned without inline labels when using label_fields.
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
