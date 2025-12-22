"""Shared helpers for meta augmenters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias

import numpy as np

import imgaug2.imgaug as ia
from imgaug2.augmenters._typing import Array, Images
from imgaug2.compat.markers import legacy

Number: TypeAlias = float | int


class _HasShape(Protocol):
    shape: tuple[int, ...]


@ia.deprecated("imgaug2.dtypes.clip_")
def clip_augmented_image_(image: Array, min_value: Number, max_value: Number) -> Array:
    """Clip image in-place."""
    np.clip(image, min_value, max_value, out=image)
    return image


@ia.deprecated("imgaug2.dtypes.clip_")
def clip_augmented_image(image: Array, min_value: Number, max_value: Number) -> Array:
    """Clip image."""
    image_copy = np.copy(image)
    np.clip(image_copy, min_value, max_value, out=image_copy)
    return image_copy


@ia.deprecated("imgaug2.dtypes.clip_")
def clip_augmented_images_(images: Images, min_value: Number, max_value: Number) -> Images:
    """Clip images in-place."""
    if ia.is_np_array(images):
        return np.clip(images, min_value, max_value, out=images)
    return [np.clip(image, min_value, max_value, out=image) for image in images]


@ia.deprecated("imgaug2.dtypes.clip_")
def clip_augmented_images(images: Images, min_value: Number, max_value: Number) -> Images:
    """Clip images."""
    if ia.is_np_array(images):
        images = np.copy(images)
    else:
        images = [np.copy(image) for image in images]
    return clip_augmented_images_(images, min_value, max_value)


@legacy
def reduce_to_nonempty(objs: Sequence[object]) -> tuple[list[object], list[int]]:
    """Remove from a list all objects that don't follow ``obj.empty==True``."""
    objs_reduced = []
    ids = []
    for i, obj in enumerate(objs):
        assert hasattr(obj, "empty"), (
            f"Expected object with property 'empty'. Got type {type(obj)}."
        )
        if not obj.empty:
            objs_reduced.append(obj)
            ids.append(i)
    return objs_reduced, ids


def invert_reduce_to_nonempty(
    objs: Sequence[object], ids: Sequence[int], objs_reduced: Sequence[object]
) -> list[object]:
    """Inverse of :func:`reduce_to_nonempty`."""
    objs_inv = list(objs)
    for idx, obj_from_reduced in zip(ids, objs_reduced, strict=True):
        objs_inv[idx] = obj_from_reduced
    return objs_inv


def estimate_max_number_of_channels(images: Images) -> int | None:
    """Compute the maximum number of image channels among a list of images."""
    if ia.is_np_array(images):
        assert images.ndim == 4, (
            f"Expected 'images' to be 4-dimensional if provided as array. "
            f"Got {images.ndim} dimensions."
        )
        return images.shape[3]

    assert ia.is_iterable(images), (
        f"Expected 'images' to be an array or iterable, got {type(images)}."
    )
    if len(images) == 0:
        return None
    channels = [el.shape[2] if len(el.shape) >= 3 else 1 for el in images]
    return max(channels)


def copy_arrays(arrays: Images) -> Images:
    """Copy the arrays of a single input array or list of input arrays."""
    if ia.is_np_array(arrays):
        return np.copy(arrays)
    return [np.copy(array) for array in arrays]


def _add_channel_axis(arrs: Images) -> Images:
    if ia.is_np_array(arrs):
        if arrs.ndim == 3:  # (N,H,W)
            return arrs[..., np.newaxis]  # (N,H,W) -> (N,H,W,1)
        return arrs
    return [
        arr[..., np.newaxis]  # (H,W) -> (H,W,1)
        if arr.ndim == 2
        else arr
        for arr in arrs
    ]


def _remove_added_channel_axis(arrs_added: Images, arrs_orig: Images) -> Images:
    if ia.is_np_array(arrs_orig):
        if arrs_orig.ndim == 3:  # (N,H,W)
            if ia.is_np_array(arrs_added):
                return arrs_added[..., 0]  # (N,H,W,1) -> (N,H,W)
            # (N,H,W) -> (N,H,W,1) -> <augmentation> -> list of (H,W,1)
            return [arr[..., 0] for arr in arrs_added]
        return arrs_added
    return [
        arr_added[..., 0] if arr_orig.ndim == 2 else arr_added  # (H,W,1) -> (H,W)
        for arr_added, arr_orig in zip(arrs_added, arrs_orig, strict=True)
    ]
