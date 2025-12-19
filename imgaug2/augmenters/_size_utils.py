"""Import-safe size/padding helpers without augmentables dependencies."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
from imgaug2.augmenters._typing import Array, Shape
from imgaug2.imgaug import _assert_two_or_three_dims, _normalize_cv2_input_arr_


def pad(
    arr: Array,
    top: int = 0,
    right: int = 0,
    bottom: int = 0,
    left: int = 0,
    mode: str = "constant",
    cval: float | int | Sequence[float | int] | Array = 0,
) -> Array:
    _assert_two_or_three_dims(arr)
    assert all([v >= 0 for v in [top, right, bottom, left]]), (
        f"Expected padding amounts that are >=0, but got {top}, {right}, {bottom}, {left} "
        "(top, right, bottom, left)"
    )

    is_multi_cval = ia.is_iterable(cval)

    if top > 0 or right > 0 or bottom > 0 or left > 0:
        min_value, _, max_value = iadt.get_value_range_of_dtype(arr.dtype)

        if iadt._FLOAT128_DTYPE is not None and arr.dtype == iadt._FLOAT128_DTYPE:
            cval = np.float128(cval)

        if is_multi_cval:
            cval = np.clip(cval, min_value, max_value)
        else:
            cval = max(min(cval, max_value), min_value)

        has_zero_sized_axis = any([axis == 0 for axis in arr.shape])
        if has_zero_sized_axis:
            mode = "constant"

        mapping_mode_np_to_cv2 = {
            "constant": cv2.BORDER_CONSTANT,
            "edge": cv2.BORDER_REPLICATE,
            "linear_ramp": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "minimum": None,
            "reflect": cv2.BORDER_REFLECT_101,
            "symmetric": cv2.BORDER_REFLECT,
            "wrap": None,
            cv2.BORDER_CONSTANT: cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE: cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT_101: cv2.BORDER_REFLECT_101,
            cv2.BORDER_REFLECT: cv2.BORDER_REFLECT,
        }
        bad_mode_cv2 = mapping_mode_np_to_cv2.get(mode, None) is None

        bad_datatype_cv2 = arr.dtype in iadt._convert_dtype_strs_to_types(
            "uint32 uint64 int64 float16 float128 bool"
        )

        bad_shape_cv2 = arr.ndim == 3 and arr.shape[-1] == 0

        if not bad_datatype_cv2 and not bad_mode_cv2 and not bad_shape_cv2:
            kind = arr.dtype.kind
            if is_multi_cval:
                cval = [float(cval_c) if kind == "f" else int(cval_c) for cval_c in cval]
            else:
                cval = float(cval) if kind == "f" else int(cval)

            if arr.ndim == 2 or arr.shape[2] <= 4:
                if arr.ndim == 3 and not is_multi_cval:
                    cval = tuple([cval] * arr.shape[2])

                arr = _normalize_cv2_input_arr_(arr)
                arr_pad = cv2.copyMakeBorder(
                    arr,
                    top=int(top),
                    bottom=int(bottom),
                    left=int(left),
                    right=int(right),
                    borderType=mapping_mode_np_to_cv2[mode],
                    value=cval,
                )
                if arr.ndim == 3 and arr_pad.ndim == 2:
                    arr_pad = arr_pad[..., np.newaxis]
            else:
                result = []
                channel_start_idx = 0
                cval = cval if is_multi_cval else tuple([cval] * arr.shape[2])
                while channel_start_idx < arr.shape[2]:
                    arr_c = arr[..., channel_start_idx : channel_start_idx + 4]
                    cval_c = cval[channel_start_idx : channel_start_idx + 4]
                    arr_pad_c = cv2.copyMakeBorder(
                        _normalize_cv2_input_arr_(arr_c),
                        top=top,
                        bottom=bottom,
                        left=left,
                        right=right,
                        borderType=mapping_mode_np_to_cv2[mode],
                        value=cval_c,
                    )
                    arr_pad_c = np.atleast_3d(arr_pad_c)
                    result.append(arr_pad_c)
                    channel_start_idx += 4
                arr_pad = np.concatenate(result, axis=2)
        else:
            paddings_np = [(top, bottom), (left, right)]
            if arr.ndim == 3:
                paddings_np.append((0, 0))

            if mode == "constant":
                if arr.ndim > 2 and is_multi_cval:
                    arr_pad_chans = [
                        np.pad(arr[..., c], paddings_np[0:2], mode=mode, constant_values=cval[c])
                        for c in np.arange(arr.shape[2])
                    ]
                    arr_pad = np.stack(arr_pad_chans, axis=-1)
                else:
                    arr_pad = np.pad(arr, paddings_np, mode=mode, constant_values=cval)
            elif mode == "linear_ramp":
                if arr.ndim > 2 and is_multi_cval:
                    arr_pad_chans = [
                        np.pad(arr[..., c], paddings_np[0:2], mode=mode, end_values=cval[c])
                        for c in np.arange(arr.shape[2])
                    ]
                    arr_pad = np.stack(arr_pad_chans, axis=-1)
                else:
                    arr_pad = np.pad(arr, paddings_np, mode=mode, end_values=cval)
            else:
                arr_pad = np.pad(arr, paddings_np, mode=mode)

        return arr_pad

    return np.copy(arr)


def compute_paddings_to_reach_aspect_ratio(arr: Array | Shape, aspect_ratio: float) -> tuple[int, int, int, int]:
    _assert_two_or_three_dims(arr)
    assert aspect_ratio > 0, f"Expected to get an aspect ratio >0, got {aspect_ratio:.4f}."

    pad_top = 0
    pad_right = 0
    pad_bottom = 0
    pad_left = 0

    shape = arr.shape if hasattr(arr, "shape") else arr
    height, width = shape[0:2]

    if height == 0:
        height = 1
        pad_bottom += 1
    if width == 0:
        width = 1
        pad_right += 1

    aspect_ratio_current = width / height

    if aspect_ratio_current < aspect_ratio:
        diff = (aspect_ratio * height) - width
        pad_right += int(np.ceil(diff / 2))
        pad_left += int(np.floor(diff / 2))
    elif aspect_ratio_current > aspect_ratio:
        diff = ((1 / aspect_ratio) * width) - height
        pad_top += int(np.floor(diff / 2))
        pad_bottom += int(np.ceil(diff / 2))

    return pad_top, pad_right, pad_bottom, pad_left


def pad_to_aspect_ratio(
    arr: Array,
    aspect_ratio: float,
    mode: str = "constant",
    cval: float | int | Sequence[float | int] | Array = 0,
    return_pad_amounts: bool = False,
) -> Array | tuple[Array, tuple[int, int, int, int]]:
    pad_top, pad_right, pad_bottom, pad_left = compute_paddings_to_reach_aspect_ratio(
        arr, aspect_ratio
    )
    arr_padded = pad(
        arr, top=pad_top, right=pad_right, bottom=pad_bottom, left=pad_left, mode=mode, cval=cval
    )

    if return_pad_amounts:
        return arr_padded, (pad_top, pad_right, pad_bottom, pad_left)
    return arr_padded
