"""Keypoint helpers for the `imgaug2.compat` layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from imgaug2.augmentables.kps import Keypoint, KeypointsOnImage


@dataclass(frozen=True, slots=True)
class KeypointParams:
    """Parameters for keypoint handling in `compat.Compose`."""

    remove_invisible: bool = True


def convert_keypoints_to_imgaug(
    keypoints: Sequence[Sequence[Any]],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[KeypointsOnImage, list[tuple[Any, ...]]]:
    """Convert keypoints to `KeypointsOnImage`.

    Supports keypoints with >=2 values:
    - first two are interpreted as (x, y)
    - remaining values are preserved and returned unchanged
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
    """Optionally remove invisible keypoints and keep extras in sync."""

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
