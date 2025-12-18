"""Bounding box helpers for the `imgaug2.compat` layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypeAlias, overload

import numpy as np

from imgaug2.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

BboxFormat: TypeAlias = Literal["pascal_voc", "coco", "yolo", "xyxy_norm"]


@dataclass(frozen=True, slots=True)
class BboxParams:
    """Parameters for bbox handling in `compat.Compose`.

    Formats:
    - "pascal_voc": (x_min, y_min, x_max, y_max) in absolute pixels
    - "coco": (x_min, y_min, width, height) in absolute pixels
    - "yolo": (x_center, y_center, width, height) normalized to [0, 1]
    - "xyxy_norm": (x_min, y_min, x_max, y_max) normalized to [0, 1]

    Notes:
    - `label_fields` stores labels in separate lists (e.g. `category_ids`)
      instead of embedding them in the bbox tuples.
    - Filtering is applied after augmentation based on `min_*` settings.
    """

    format: BboxFormat = "pascal_voc"
    label_fields: tuple[str, ...] = ()
    min_area: float = 0.0
    min_visibility: float = 0.0
    min_width: float = 0.0
    min_height: float = 0.0


def _split_inline_labels(
    bboxes: Sequence[Sequence[Any]],
) -> tuple[list[tuple[float, float, float, float]], list[Any] | None]:
    coords: list[tuple[float, float, float, float]] = []
    labels: list[Any] = []
    has_any_label = False

    for bbox in bboxes:
        if len(bbox) < 4:
            raise ValueError(f"Expected bbox with >=4 values, got {bbox!r}")
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        coords.append((float(x1), float(y1), float(x2), float(y2)))
        if len(bbox) >= 5:
            has_any_label = True
            labels.append(bbox[4])
        else:
            labels.append(None)

    return coords, labels if has_any_label else None


def _join_inline_labels(
    coords: Sequence[tuple[float, float, float, float]],
    labels: Sequence[Any] | None,
) -> list[tuple[Any, ...]]:
    if labels is None:
        return [tuple(c) for c in coords]
    if len(coords) != len(labels):
        raise ValueError("coords/labels length mismatch")
    return [(*c, labels[i]) for i, c in enumerate(coords)]


def convert_bboxes_to_imgaug(
    bboxes: Sequence[Sequence[Any]],
    image_shape: tuple[int, int, int] | tuple[int, int],
    *,
    format: BboxFormat = "pascal_voc",
) -> tuple[BoundingBoxesOnImage, list[Any] | None]:
    """Convert bbox sequences to `BoundingBoxesOnImage`.

    Returns `(bbs_on_image, inline_labels_or_none)`.

    The compat layer supports either:
    - inline labels (bbox tuples have 5th element), or
    - external label fields (handled by Compose).
    """

    h, w = int(image_shape[0]), int(image_shape[1])

    if len(bboxes) == 0:
        return BoundingBoxesOnImage([], shape=(h, w, 3)), None

    if format in ("pascal_voc", "xyxy_norm"):
        coords_xyxy, inline_labels = _split_inline_labels(bboxes)
    else:
        inline_labels = None
        coords_xyxy = []
        for bbox in bboxes:
            if len(bbox) < 4:
                raise ValueError(f"Expected bbox with >=4 values, got {bbox!r}")
            if len(bbox) >= 5:
                # Inline bbox labels are only supported for xyxy-style formats.
                # Keep this explicit to avoid silent label dropping.
                raise ValueError(
                    "Inline bbox labels are only supported for format='pascal_voc' or "
                    "'xyxy_norm'. For COCO/YOLO, pass labels via label_fields."
                )
            coords_xyxy.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))

    bb_list: list[BoundingBox] = []

    for xyxy in coords_xyxy:
        x1, y1, x2, y2 = xyxy

        if format == "pascal_voc":
            pass
        elif format == "coco":
            x1, y1, bw, bh = x1, y1, x2, y2
            x2 = x1 + bw
            y2 = y1 + bh
        elif format == "yolo":
            xc, yc, bw, bh = x1, y1, x2, y2
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
        elif format == "xyxy_norm":
            # normalized xyxy
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
        else:
            raise ValueError(f"Unknown bbox format: {format!r}")

        bb_list.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))

    return BoundingBoxesOnImage(bb_list, shape=(h, w, 3)), inline_labels


def convert_bboxes_from_imgaug(
    bbs: BoundingBoxesOnImage,
    *,
    format: BboxFormat = "pascal_voc",
    inline_labels: Sequence[Any] | None = None,
) -> list[tuple[Any, ...]]:
    """Convert `BoundingBoxesOnImage` back to a list format.

    If `inline_labels` is provided, it is appended as the 5th element of each
    bbox tuple (Pascal VOC / normalized xyxy formats only).
    """

    h, w = int(bbs.shape[0]), int(bbs.shape[1])

    coords: list[tuple[float, float, float, float]] = []
    for bb in bbs.bounding_boxes:
        if format == "pascal_voc":
            coords.append((float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2)))
        elif format == "coco":
            coords.append((float(bb.x1), float(bb.y1), float(bb.x2 - bb.x1), float(bb.y2 - bb.y1)))
        elif format == "yolo":
            bw = float(bb.x2 - bb.x1)
            bh = float(bb.y2 - bb.y1)
            xc = float(bb.x1 + bw / 2) / w
            yc = float(bb.y1 + bh / 2) / h
            coords.append((xc, yc, bw / w, bh / h))
        elif format == "xyxy_norm":
            coords.append((float(bb.x1) / w, float(bb.y1) / h, float(bb.x2) / w, float(bb.y2) / h))
        else:
            raise ValueError(f"Unknown bbox format: {format!r}")

    if format not in ("pascal_voc", "xyxy_norm"):
        # Inline labels only supported for xyxy-style formats.
        if inline_labels is not None:
            raise ValueError("inline_labels only supported for pascal_voc/xyxy_norm output")
        return [tuple(c) for c in coords]

    return _join_inline_labels(coords, list(inline_labels) if inline_labels is not None else None)


def filter_bboxes(
    bbs: BoundingBoxesOnImage,
    params: BboxParams,
    *,
    inline_labels: list[Any] | None = None,
) -> tuple[BoundingBoxesOnImage, list[Any] | None, list[bool]]:
    """Filter + clip bboxes and keep labels in sync.

    Returns `(filtered_bbs, filtered_inline_labels, keep_mask)`.
    """

    if inline_labels is not None and len(inline_labels) != len(bbs.bounding_boxes):
        raise ValueError("inline_labels length mismatch with bboxes")

    h, w = int(bbs.shape[0]), int(bbs.shape[1])
    image_shape = (h, w, 3)

    keep: list[bool] = []
    kept_bbs: list[BoundingBox] = []
    kept_labels: list[Any] = [] if inline_labels is not None else []

    for idx, bb in enumerate(bbs.bounding_boxes):
        area = float(bb.area)
        if area <= 0:
            keep.append(False)
            continue

        # Visibility: fraction of bbox area that remains inside the image.
        out_frac = float(bb.compute_out_of_image_fraction(image_shape))
        visible_frac = 1.0 - out_frac

        if visible_frac < params.min_visibility:
            keep.append(False)
            continue

        if params.min_area > 0.0:
            # Approximate visible area using clipped bbox.
            bb_clipped = bb.clip_out_of_image(image_shape)
            if float(bb_clipped.area) < params.min_area:
                keep.append(False)
                continue

        if params.min_width > 0.0 and float(bb.width) < params.min_width:
            keep.append(False)
            continue

        if params.min_height > 0.0 and float(bb.height) < params.min_height:
            keep.append(False)
            continue

        keep.append(True)
        kept_bbs.append(bb)
        if inline_labels is not None:
            kept_labels.append(inline_labels[idx])

    filtered = BoundingBoxesOnImage(kept_bbs, shape=bbs.shape).clip_out_of_image()
    if inline_labels is None:
        return filtered, None, keep
    return filtered, kept_labels, keep
