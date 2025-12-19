"""Bounding box conversion and filtering utilities for compatibility layer.

This module provides format conversion between common bounding box representations
(Pascal VOC, COCO, YOLO, normalized xyxy) and imgaug2's native BoundingBoxesOnImage
format. It supports inline labels and external label fields with synchronized filtering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, overload

import numpy as np

from imgaug2.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

BboxFormat: TypeAlias = Literal["pascal_voc", "coco", "yolo", "xyxy_norm"]


@dataclass(frozen=True, slots=True)
class BboxParams:
    """Parameters for bounding box handling in Compose transforms.

    Parameters
    ----------
    format : {'pascal_voc', 'coco', 'yolo', 'xyxy_norm'}, default='pascal_voc'
        Bounding box format specification:
        - 'pascal_voc': (x_min, y_min, x_max, y_max) in absolute pixels
        - 'coco': (x_min, y_min, width, height) in absolute pixels
        - 'yolo': (x_center, y_center, width, height) normalized to [0, 1]
        - 'xyxy_norm': (x_min, y_min, x_max, y_max) normalized to [0, 1]
    label_fields : tuple of str, default=()
        Names of fields containing bbox labels (e.g., ('category_ids', 'labels')).
        Labels are stored in separate lists instead of inline in bbox tuples.
    min_area : float, default=0.0
        Minimum visible area in pixels to keep bbox after augmentation.
    min_visibility : float, default=0.0
        Minimum fraction of bbox area visible [0, 1] to keep bbox.
    min_width : float, default=0.0
        Minimum bbox width in pixels to keep bbox.
    min_height : float, default=0.0
        Minimum bbox height in pixels to keep bbox.

    Notes
    -----
    Filtering is applied after augmentation based on min_* settings.
    Bboxes failing any criterion are removed along with their labels.
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
    """Split bbox coordinates from inline labels.

    Parameters
    ----------
    bboxes : sequence of sequence
        Bounding boxes with at least 4 values. Optional 5th value is label.

    Returns
    -------
    coords : list of tuple
        Bbox coordinates as (x1, y1, x2, y2) tuples.
    labels : list or None
        Inline labels if any bbox has 5+ values, otherwise None.

    Raises
    ------
    ValueError
        If any bbox has fewer than 4 values.
    """
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
    """Combine bbox coordinates with inline labels.

    Parameters
    ----------
    coords : sequence of tuple
        Bbox coordinates as (x1, y1, x2, y2) tuples.
    labels : sequence or None
        Optional inline labels to append as 5th element.

    Returns
    -------
    list of tuple
        Bboxes with labels appended if provided, otherwise just coords.

    Raises
    ------
    ValueError
        If coords and labels have mismatched lengths.
    """
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
    """Convert bbox sequences to BoundingBoxesOnImage format.

    Parameters
    ----------
    bboxes : sequence of sequence
        Bounding boxes in specified format. Each bbox has 4+ values.
    image_shape : tuple of int
        Image shape as (height, width) or (height, width, channels).
    format : {'pascal_voc', 'coco', 'yolo', 'xyxy_norm'}, default='pascal_voc'
        Format of input bboxes.

    Returns
    -------
    bbs_on_image : BoundingBoxesOnImage
        Converted bounding boxes in imgaug2 format.
    inline_labels : list or None
        Inline labels extracted from 5th element, or None if not present.
        Only supported for pascal_voc and xyxy_norm formats.

    Raises
    ------
    ValueError
        If bbox format is unknown or inline labels used with unsupported format.

    Notes
    -----
    Inline labels (5th element) are only supported for pascal_voc and xyxy_norm
    formats. For COCO/YOLO, use label_fields instead.
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
    """Convert BoundingBoxesOnImage back to list format.

    Parameters
    ----------
    bbs : BoundingBoxesOnImage
        Bounding boxes in imgaug2 format.
    format : {'pascal_voc', 'coco', 'yolo', 'xyxy_norm'}, default='pascal_voc'
        Target format for output bboxes.
    inline_labels : sequence or None, default=None
        Labels to append as 5th element. Only supported for pascal_voc
        and xyxy_norm formats.

    Returns
    -------
    list of tuple
        Bboxes in specified format, with optional labels as 5th element.

    Raises
    ------
    ValueError
        If format is unknown or inline_labels used with unsupported format.
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
    """Filter and clip bboxes based on size and visibility criteria.

    Parameters
    ----------
    bbs : BoundingBoxesOnImage
        Bounding boxes to filter.
    params : BboxParams
        Filtering parameters with min_* thresholds.
    inline_labels : list or None, default=None
        Labels synchronized with bboxes. Filtered along with bboxes.

    Returns
    -------
    filtered_bbs : BoundingBoxesOnImage
        Bboxes that pass all filtering criteria, clipped to image bounds.
    filtered_labels : list or None
        Labels for kept bboxes, or None if no labels provided.
    keep_mask : list of bool
        Boolean mask indicating which bboxes were kept.

    Raises
    ------
    ValueError
        If inline_labels length doesn't match number of bboxes.

    Notes
    -----
    Filtering criteria are applied in order: area > 0, visibility, min_area,
    min_width, min_height. All criteria must pass for bbox to be kept.
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

        out_frac = float(bb.compute_out_of_image_fraction(image_shape))
        visible_frac = 1.0 - out_frac

        if visible_frac < params.min_visibility:
            keep.append(False)
            continue

        if params.min_area > 0.0:
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
