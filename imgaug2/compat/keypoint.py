"""Keypoint conversion and filtering utilities for compatibility layer.

This module provides conversion between list-based keypoint representations and
imgaug2's native KeypointsOnImage format. It preserves extra fields beyond (x, y)
coordinates and supports filtering of out-of-bounds keypoints.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from imgaug2.augmentables.kps import Keypoint, KeypointsOnImage


@dataclass(frozen=True, slots=True)
class KeypointParams:
    """Parameters for keypoint handling in Compose transforms.

    Parameters
    ----------
    remove_invisible : bool, default=True
        If True, keypoints outside image bounds are removed after augmentation.
        Coordinates must satisfy 0 <= x < width and 0 <= y < height.
    """

    remove_invisible: bool = True


def convert_keypoints_to_imgaug(
    keypoints: Sequence[Sequence[Any]],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[KeypointsOnImage, list[tuple[Any, ...]]]:
    """Convert keypoint sequences to KeypointsOnImage format.

    Parameters
    ----------
    keypoints : sequence of sequence
        Keypoints with at least 2 values (x, y). Additional values preserved.
    image_shape : tuple of int
        Image shape as (height, width) or (height, width, channels).

    Returns
    -------
    kps_on_image : KeypointsOnImage
        Converted keypoints in imgaug2 format.
    extras : list of tuple
        Additional fields from each keypoint (elements beyond x, y).

    Raises
    ------
    ValueError
        If any keypoint has fewer than 2 values.
    """

    h, w = int(image_shape[0]), int(image_shape[1])

    kp_list: list[Keypoint] = []
    extras: list[tuple[Any, ...]] = []

    for kp in keypoints:
        if len(kp) < 2:
            raise ValueError(f"Expected keypoint with >=2 values, got {kp!r}")
        x, y = float(kp[0]), float(kp[1])
        kp_list.append(Keypoint(x=x, y=y))
        extras.append(tuple(kp[2:]))

    return KeypointsOnImage(kp_list, shape=(h, w, 3)), extras


def convert_keypoints_from_imgaug(
    kps: KeypointsOnImage,
    *,
    extras: Sequence[tuple[Any, ...]] | None = None,
) -> list[tuple[Any, ...]]:
    """Convert KeypointsOnImage back to list format.

    Parameters
    ----------
    kps : KeypointsOnImage
        Keypoints in imgaug2 format.
    extras : sequence of tuple or None, default=None
        Additional fields to append to each keypoint tuple.

    Returns
    -------
    list of tuple
        Keypoints as (x, y, *extras) tuples.

    Raises
    ------
    ValueError
        If extras length doesn't match number of keypoints.
    """
    if extras is not None and len(extras) != len(kps.keypoints):
        raise ValueError("extras length mismatch with keypoints")

    out: list[tuple[Any, ...]] = []
    for idx, kp in enumerate(kps.keypoints):
        tail: tuple[Any, ...] = () if extras is None else tuple(extras[idx])
        out.append((float(kp.x), float(kp.y), *tail))
    return out


def filter_keypoints(
    kps: KeypointsOnImage,
    params: KeypointParams,
    *,
    extras: list[tuple[Any, ...]] | None = None,
) -> tuple[KeypointsOnImage, list[tuple[Any, ...]] | None]:
    """Filter out-of-bounds keypoints based on visibility.

    Parameters
    ----------
    kps : KeypointsOnImage
        Keypoints to filter.
    params : KeypointParams
        Filtering parameters.
    extras : list of tuple or None, default=None
        Extra fields synchronized with keypoints. Filtered along with keypoints.

    Returns
    -------
    filtered_kps : KeypointsOnImage
        Keypoints within image bounds if remove_invisible=True.
    filtered_extras : list of tuple or None
        Extras for kept keypoints, or None if no extras provided.

    Raises
    ------
    ValueError
        If extras length doesn't match number of keypoints.

    Notes
    -----
    Keypoints are considered visible if 0 <= x < width and 0 <= y < height.
    """

    if extras is not None and len(extras) != len(kps.keypoints):
        raise ValueError("extras length mismatch with keypoints")

    if not params.remove_invisible:
        return kps, extras

    h, w = int(kps.shape[0]), int(kps.shape[1])

    kept_kps: list[Keypoint] = []
    kept_extras: list[tuple[Any, ...]] = [] if extras is not None else []

    for idx, kp in enumerate(kps.keypoints):
        x, y = float(kp.x), float(kp.y)
        if 0.0 <= x < w and 0.0 <= y < h:
            kept_kps.append(kp)
            if extras is not None:
                kept_extras.append(extras[idx])

    filtered = KeypointsOnImage(kept_kps, shape=kps.shape)
    if extras is None:
        return filtered, None
    return filtered, kept_extras
