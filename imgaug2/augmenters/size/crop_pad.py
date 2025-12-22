"""Crop/pad augmenters and shared helpers."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Literal

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, Images, RNGInput
from imgaug2.compat.markers import legacy

from ._utils import (
    CropAndPadMode,
    CropAndPadParamReturn,
    CropAndPadPercentInput,
    CropAndPadPercentSingleParam,
    CropAndPadPxInput,
    CropAndPadPxSingleParam,
    Shape,
    TRBL,
    XYXY,
    _handle_pad_mode_param,
    pad,
)
def _crop_trbl_to_xyxy(
    shape: Shape,
    top: int,
    right: int,
    bottom: int,
    left: int,
    prevent_zero_size: bool = True,
) -> XYXY:
    if prevent_zero_size:
        top, bottom = _prevent_zero_size_after_crop_(shape[0], top, bottom)
        left, right = _prevent_zero_size_after_crop_(shape[1], left, right)

    height, width = shape[0:2]
    x1 = left
    x2 = width - right
    y1 = top
    y2 = height - bottom

    # these steps prevent negative sizes
    # if x2==x1 or y2==y1 then the output arr has size 0 for the respective axis
    # note that if height/width of arr is zero, then y2==y1 or x2==x1, which
    # is still valid, even if height/width is zero and results in a zero-sized
    # axis
    x2 = max(x2, x1)
    y2 = max(y2, y1)

    return x1, y1, x2, y2


def _crop_arr_(
    arr: Array,
    top: int,
    right: int,
    bottom: int,
    left: int,
    prevent_zero_size: bool = True,
) -> Array:
    x1, y1, x2, y2 = _crop_trbl_to_xyxy(
        arr.shape, top, right, bottom, left, prevent_zero_size=prevent_zero_size
    )
    return arr[y1:y2, x1:x2, ...]


def _crop_and_pad_arr(
    arr: Array,
    croppings: TRBL,
    paddings: TRBL,
    pad_mode: str = "constant",
    pad_cval: float | int | Sequence[float | int] | Array = 0,
    keep_size: bool = False,
) -> Array:
    height, width = arr.shape[0:2]

    image_cr = _crop_arr_(arr, *croppings)

    image_cr_pa = pad(
        image_cr,
        top=paddings[0],
        right=paddings[1],
        bottom=paddings[2],
        left=paddings[3],
        mode=pad_mode,
        cval=pad_cval,
    )

    if keep_size:
        from imgaug2.mlx._core import is_mlx_array

        if is_mlx_array(image_cr_pa):
            import imgaug2.mlx as mlx

            image_cr_pa = mlx.geometry.resize(image_cr_pa, (height, width), order=1)
        else:
            image_cr_pa = ia.imresize_single_image(image_cr_pa, (height, width))

    return image_cr_pa


def _crop_and_pad_heatmap_(
    heatmap: ia.HeatmapsOnImage,
    croppings_img: TRBL,
    paddings_img: TRBL,
    pad_mode: str = "constant",
    pad_cval: float = 0.0,
    keep_size: bool = False,
) -> ia.HeatmapsOnImage:
    return _crop_and_pad_hms_or_segmaps_(
        heatmap, croppings_img, paddings_img, pad_mode, pad_cval, keep_size
    )


def _crop_and_pad_segmap_(
    segmap: ia.SegmentationMapsOnImage,
    croppings_img: TRBL,
    paddings_img: TRBL,
    pad_mode: str = "constant",
    pad_cval: int = 0,
    keep_size: bool = False,
) -> ia.SegmentationMapsOnImage:
    return _crop_and_pad_hms_or_segmaps_(
        segmap, croppings_img, paddings_img, pad_mode, pad_cval, keep_size
    )


def _crop_and_pad_hms_or_segmaps_(
    augmentable: ia.HeatmapsOnImage | ia.SegmentationMapsOnImage,
    croppings_img: TRBL,
    paddings_img: TRBL,
    pad_mode: str | np.generic = "constant",
    pad_cval: float | int | np.generic | None = None,
    keep_size: bool = False,
) -> ia.HeatmapsOnImage | ia.SegmentationMapsOnImage:
    if isinstance(augmentable, ia.HeatmapsOnImage):
        arr_attr_name = "arr_0to1"
        pad_cval = pad_cval if pad_cval is not None else 0.0
    else:
        assert isinstance(augmentable, ia.SegmentationMapsOnImage), (
            f"Expected HeatmapsOnImage or SegmentationMapsOnImage, got {type(augmentable)}."
        )
        arr_attr_name = "arr"
        pad_cval = pad_cval if pad_cval is not None else 0

    arr = getattr(augmentable, arr_attr_name)
    arr_shape_orig = arr.shape
    augm_shape = augmentable.shape

    croppings_proj = _project_size_changes(croppings_img, augm_shape, arr.shape)
    paddings_proj = _project_size_changes(paddings_img, augm_shape, arr.shape)

    croppings_proj = _prevent_zero_size_after_crop_trbl_(arr.shape[0], arr.shape[1], croppings_proj)

    arr_cr = _crop_arr_(
        arr, croppings_proj[0], croppings_proj[1], croppings_proj[2], croppings_proj[3]
    )
    arr_cr_pa = pad(
        arr_cr,
        top=paddings_proj[0],
        right=paddings_proj[1],
        bottom=paddings_proj[2],
        left=paddings_proj[3],
        mode=pad_mode,
        cval=pad_cval,
    )

    setattr(augmentable, arr_attr_name, arr_cr_pa)

    if keep_size:
        augmentable = augmentable.resize(arr_shape_orig[0:2])
    else:
        augmentable.shape = _compute_shape_after_crop_and_pad(
            augmentable.shape, croppings_img, paddings_img
        )
    return augmentable


def _crop_and_pad_kpsoi_(
    kpsoi: ia.KeypointsOnImage, croppings_img: TRBL, paddings_img: TRBL, keep_size: bool
) -> ia.KeypointsOnImage:
    # using the trbl function instead of croppings_img has the advantage
    # of incorporating prevent_zero_size, dealing with zero-sized input image
    # axis and dealing the negative crop amounts
    x1, y1, _x2, _y2 = _crop_trbl_to_xyxy(kpsoi.shape, *croppings_img)
    crop_left = x1
    crop_top = y1

    shape_orig = kpsoi.shape
    shifted = kpsoi.shift_(x=-crop_left + paddings_img[3], y=-crop_top + paddings_img[0])
    shifted.shape = _compute_shape_after_crop_and_pad(shape_orig, croppings_img, paddings_img)
    if keep_size:
        shifted = shifted.on_(shape_orig)
    return shifted


def _compute_shape_after_crop_and_pad(old_shape: Shape, croppings: TRBL, paddings: TRBL) -> Shape:
    x1, y1, x2, y2 = _crop_trbl_to_xyxy(old_shape, *croppings)
    new_shape = list(old_shape)
    new_shape[0] = y2 - y1 + paddings[0] + paddings[2]
    new_shape[1] = x2 - x1 + paddings[1] + paddings[3]
    return tuple(new_shape)


def _prevent_zero_size_after_crop_trbl_(height: int, width: int, crop_trbl: TRBL) -> TRBL:
    crop_top = crop_trbl[0]
    crop_right = crop_trbl[1]
    crop_bottom = crop_trbl[2]
    crop_left = crop_trbl[3]

    crop_top, crop_bottom = _prevent_zero_size_after_crop_(height, crop_top, crop_bottom)
    crop_left, crop_right = _prevent_zero_size_after_crop_(width, crop_left, crop_right)
    return (crop_top, crop_right, crop_bottom, crop_left)


def _prevent_zero_size_after_crop_(
    axis_size: int, crop_start: int, crop_end: int
) -> tuple[int, int]:
    crops_start, crops_end = _prevent_zero_sizes_after_crops_(
        np.array([axis_size], dtype=np.int32),
        np.array([crop_start], dtype=np.int32),
        np.array([crop_end], dtype=np.int32),
    )
    return int(crops_start[0]), int(crops_end[0])


def _prevent_zero_sizes_after_crops_(
    axis_sizes: Array, crops_start: Array, crops_end: Array
) -> tuple[Array, Array]:
    remaining_sizes = axis_sizes - (crops_start + crops_end)

    mask_bad_sizes = remaining_sizes < 1
    regains = mask_bad_sizes * (np.abs(remaining_sizes) + 1)
    regains_half = regains.astype(np.float32) / 2
    regains_start = np.ceil(regains_half).astype(np.int32)
    regains_end = np.floor(regains_half).astype(np.int32)

    crops_start -= regains_start
    crops_end -= regains_end

    mask_too_much_start = crops_start < 0
    crops_end[mask_too_much_start] += crops_start[mask_too_much_start]
    crops_start = np.maximum(crops_start, 0)

    mask_too_much_end = crops_end < 0
    crops_start[mask_too_much_end] += crops_end[mask_too_much_end]
    crops_end = np.maximum(crops_end, 0)

    crops_start = np.maximum(crops_start, 0)

    return crops_start, crops_end


def _project_size_changes(trbl: TRBL, from_shape: Shape, to_shape: Shape) -> TRBL:
    if from_shape[0:2] == to_shape[0:2]:
        return trbl

    height_to = to_shape[0]
    width_to = to_shape[1]
    height_from = from_shape[0]
    width_from = from_shape[1]

    top = trbl[0]
    right = trbl[1]
    bottom = trbl[2]
    left = trbl[3]

    # Adding/subtracting 1e-4 here helps for the case where a heatmap/segmap
    # is exactly half the size of an image and the size change on an axis is
    # an odd value. Then the projected value would end up being <something>.5
    # and the rounding would always round up to the next integer. If both
    # sides then have the same change, they are both rounded up, resulting
    # in more change than expected.
    # E.g. image height is 8, map height is 4, change is 3 at the top and 3 at
    # the bottom. The changes are projected to 4*(3/8) = 1.5 and both rounded
    # up to 2.0. Hence, the maps are changed by 4 (100% of the map height,
    # vs. 6 for images, which is 75% of the image height).
    top = _int_r(height_to * (top / height_from) - 1e-4)
    right = _int_r(width_to * (right / width_from) + 1e-4)
    bottom = _int_r(height_to * (bottom / height_from) + 1e-4)
    left = _int_r(width_to * (left / width_from) - 1e-4)

    return top, right, bottom, left


def _int_r(value: float) -> int:
    return int(np.round(value))


# TODO somehow integrate this with pad()
@iap._prefetchable_str
def _handle_pad_mode_param(
    pad_mode: str | list[str] | iap.StochasticParameter | Literal["ALL"],
) -> iap.StochasticParameter:
    pad_modes_available = {
        "constant",
        "edge",
        "linear_ramp",
        "maximum",
        "mean",
        "median",
        "minimum",
        "reflect",
        "symmetric",
        "wrap",
    }
    if pad_mode == ia.ALL:
        return iap.Choice(list(pad_modes_available))
    if ia.is_string(pad_mode):
        assert pad_mode in pad_modes_available, (
            "Value '{}' is not a valid pad mode. Valid pad modes are: {}.".format(
                pad_mode, ", ".join(pad_modes_available)
            )
        )
        return iap.Deterministic(pad_mode)
    if isinstance(pad_mode, list):
        assert all([v in pad_modes_available for v in pad_mode]), (
            "At least one in list {} is not a valid pad mode. Valid pad modes are: {}.".format(
                str(pad_mode), ", ".join(pad_modes_available)
            )
        )
        return iap.Choice(pad_mode)
    if isinstance(pad_mode, iap.StochasticParameter):
        return pad_mode
    raise Exception(
        "Expected pad_mode to be ia.ALL or string or list of strings or "
        f"StochasticParameter, got {type(pad_mode)}."
    )


@legacy(version="0.4.0")
def pad(
    arr: Array,
    top: int = 0,
    right: int = 0,
    bottom: int = 0,
    left: int = 0,
    mode: str = "constant",
    cval: float | int | Sequence[float | int] | Array = 0,
) -> Array:
    """Pad an image-like array on its top/right/bottom/left side.

    This function is a wrapper around :func:`numpy.pad`.

    Previously named ``imgaug2.imgaug2.pad()``.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested (1)
        * ``uint16``: yes; fully tested (1)
        * ``uint32``: yes; fully tested (2) (3)
        * ``uint64``: yes; fully tested (2) (3)
        * ``int8``: yes; fully tested (1)
        * ``int16``: yes; fully tested (1)
        * ``int32``: yes; fully tested (1)
        * ``int64``: yes; fully tested (2) (3)
        * ``float16``: yes; fully tested (2) (3)
        * ``float32``: yes; fully tested (1)
        * ``float64``: yes; fully tested (1)
        * ``float128``: yes; fully tested (2) (3)
        * ``bool``: yes; tested (2) (3)

        - (1) Uses ``cv2`` if `mode` is one of: ``"constant"``, ``"edge"``,
              ``"reflect"``, ``"symmetric"``. Otherwise uses ``numpy``.
        - (2) Uses ``numpy``.
        - (3) Rejected by ``cv2``.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pad.

    top : int, optional
        Amount of pixels to add to the top side of the image.
        Must be ``0`` or greater.

    right : int, optional
        Amount of pixels to add to the right side of the image.
        Must be ``0`` or greater.

    bottom : int, optional
        Amount of pixels to add to the bottom side of the image.
        Must be ``0`` or greater.

    left : int, optional
        Amount of pixels to add to the left side of the image.
        Must be ``0`` or greater.

    mode : str, optional
        Padding mode to use. See :func:`numpy.pad` for details.
        In case of mode ``constant``, the parameter `cval` will be used as
        the ``constant_values`` parameter to :func:`numpy.pad`.
        In case of mode ``linear_ramp``, the parameter `cval` will be used as
        the ``end_values`` parameter to :func:`numpy.pad`.

    cval : number or iterable of number, optional
        Value to use for padding if `mode` is ``constant``.
        See :func:`numpy.pad` for details. The cval is expected to match the
        input array's dtype and value range. If an iterable is used, it is
        expected to contain one value per channel. The number of values
        and number of channels are expected to match.

    Returns
    -------
    (H',W') ndarray or (H',W',C) ndarray
        Padded array with height ``H'=H+top+bottom`` and width
        ``W'=W+left+right``.

    """

    _assert_two_or_three_dims(arr)
    assert all([v >= 0 for v in [top, right, bottom, left]]), (
        f"Expected padding amounts that are >=0, but got {top}, {right}, {bottom}, {left} "
        "(top, right, bottom, left)"
    )

    is_multi_cval = ia.is_iterable(cval)
    from imgaug2.mlx._core import is_mlx_array
    if is_mlx_array(arr) and not is_multi_cval:
        if mode in {"constant", "edge", "reflect", "symmetric", "wrap"}:
            import imgaug2.mlx as mlx

            arr_mlx = arr
            squeeze_channel = False
            if arr.ndim == 2:
                arr_mlx = arr[..., np.newaxis]
                squeeze_channel = True

            arr_pad = mlx.pad(
                arr_mlx,
                pad_top=int(top),
                pad_bottom=int(bottom),
                pad_left=int(left),
                pad_right=int(right),
                mode=str(mode),
                value=float(cval),
            )
            if squeeze_channel:
                arr_pad = arr_pad[..., 0]
            return arr_pad

    if top > 0 or right > 0 or bottom > 0 or left > 0:
        min_value, _, max_value = iadt.get_value_range_of_dtype(arr.dtype)

        # Without the if here there are crashes for float128, e.g. if
        # cval is an int (just using float(cval) seems to not be accurate
        # enough).
        # Note: If float128 is not available on the system, _FLOAT128_DTYPE is
        # None, but 'np.dtype("float64") == None' actually equates to True
        # for whatever reason, so we check first if the constant is not None
        # (i.e. if float128 exists).
        if iadt._FLOAT128_DTYPE is not None and arr.dtype == iadt._FLOAT128_DTYPE:
            cval = np.float128(cval)

        if is_multi_cval:
            cval = np.clip(cval, min_value, max_value)
        else:
            cval = max(min(cval, max_value), min_value)

        # Note that copyMakeBorder() hangs/runs endlessly if arr has an
        # axis of size 0 and mode is "reflect".
        # Numpy also complains in these cases if mode is not "constant".
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

        # these datatypes all simply generate a "TypeError: src data type = X
        # is not supported" error
        bad_datatype_cv2 = arr.dtype in iadt._convert_dtype_strs_to_types(
            "uint32 uint64 int64 float16 float128 bool"
        )

        # OpenCV turns the channel axis for arrays with 0 channels to 512
        bad_shape_cv2 = arr.ndim == 3 and arr.shape[-1] == 0

        if not bad_datatype_cv2 and not bad_mode_cv2 and not bad_shape_cv2:
            # convert cval to expected type, as otherwise we get TypeError
            # for np inputs
            kind = arr.dtype.kind
            if is_multi_cval:
                cval = [float(cval_c) if kind == "f" else int(cval_c) for cval_c in cval]
            else:
                cval = float(cval) if kind == "f" else int(cval)

            if arr.ndim == 2 or arr.shape[2] <= 4:
                # without this, only the first channel is padded with the cval,
                # all following channels with 0
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
            # paddings for 2d case
            paddings_np = [(top, bottom), (left, right)]

            # add paddings for 3d case
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


@legacy(version="0.4.0")
def pad_to_aspect_ratio(
    arr: Array,
    aspect_ratio: float,
    mode: str = "constant",
    cval: float | int | Sequence[float | int] | Array = 0,
    return_pad_amounts: bool = False,
) -> Array | tuple[Array, TRBL]:
    """Pad an image array on its sides so that it matches a target aspect ratio.

    See :func:`~imgaug2.imgaug2.compute_paddings_for_aspect_ratio` for an
    explanation of how the required padding amounts are distributed per
    image axis.

    Previously named ``imgaug2.imgaug2.pad_to_aspect_ratio()``.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.size.pad`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pad.

    aspect_ratio : float
        Target aspect ratio, given as width/height. E.g. ``2.0`` denotes the
        image having twice as much width as height.

    mode : str, optional
        Padding mode to use. See :func:`~imgaug2.imgaug2.pad` for details.

    cval : number, optional
        Value to use for padding if `mode` is ``constant``.
        See :func:`numpy.pad` for details.

    return_pad_amounts : bool, optional
        If ``False``, then only the padded image will be returned. If
        ``True``, a ``tuple`` with two entries will be returned, where the
        first entry is the padded image and the second entry are the amounts
        by which each image side was padded. These amounts are again a
        ``tuple`` of the form ``(top, right, bottom, left)``, with each value
        being an ``int``.

    Returns
    -------
    (H',W') ndarray or (H',W',C) ndarray
        Padded image as ``(H',W')`` or ``(H',W',C)`` ndarray, fulfilling the
        given `aspect_ratio`.

    tuple of int
        Amounts by which the image was padded on each side, given as a
        ``tuple`` ``(top, right, bottom, left)``.
        This ``tuple`` is only returned if `return_pad_amounts` was set to
        ``True``.

    """
    pad_top, pad_right, pad_bottom, pad_left = compute_paddings_to_reach_aspect_ratio(
        arr, aspect_ratio
    )
    arr_padded = pad(
        arr, top=pad_top, right=pad_right, bottom=pad_bottom, left=pad_left, mode=mode, cval=cval
    )

    if return_pad_amounts:
        return arr_padded, (pad_top, pad_right, pad_bottom, pad_left)
    return arr_padded


@legacy(version="0.4.0")
def pad_to_multiples_of(
    arr: Array,
    height_multiple: int | None,
    width_multiple: int | None,
    mode: str = "constant",
    cval: float | int | Sequence[float | int] | Array = 0,
    return_pad_amounts: bool = False,
) -> Array | tuple[Array, TRBL]:
    """Pad an image array until its side lengths are multiples of given values.

    See :func:`~imgaug2.imgaug2.compute_paddings_for_aspect_ratio` for an
    explanation of how the required padding amounts are distributed per
    image axis.

    Previously named ``imgaug2.imgaug2.pad_to_multiples_of()``.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.size.pad`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pad.

    height_multiple : None or int
        The desired multiple of the height. The computed padding amount will
        reflect a padding that increases the y axis size until it is a multiple
        of this value.

    width_multiple : None or int
        The desired multiple of the width. The computed padding amount will
        reflect a padding that increases the x axis size until it is a multiple
        of this value.

    mode : str, optional
        Padding mode to use. See :func:`~imgaug2.imgaug2.pad` for details.

    cval : number, optional
        Value to use for padding if `mode` is ``constant``.
        See :func:`numpy.pad` for details.

    return_pad_amounts : bool, optional
        If ``False``, then only the padded image will be returned. If
        ``True``, a ``tuple`` with two entries will be returned, where the
        first entry is the padded image and the second entry are the amounts
        by which each image side was padded. These amounts are again a
        ``tuple`` of the form ``(top, right, bottom, left)``, with each value
        being an integer.

    Returns
    -------
    (H',W') ndarray or (H',W',C) ndarray
        Padded image as ``(H',W')`` or ``(H',W',C)`` ndarray.

    tuple of int
        Amounts by which the image was padded on each side, given as a
        ``tuple`` ``(top, right, bottom, left)``.
        This ``tuple`` is only returned if `return_pad_amounts` was set to
        ``True``.

    """
    pad_top, pad_right, pad_bottom, pad_left = compute_paddings_to_reach_multiples_of(
        arr, height_multiple, width_multiple
    )
    arr_padded = pad(
        arr, top=pad_top, right=pad_right, bottom=pad_bottom, left=pad_left, mode=mode, cval=cval
    )

    if return_pad_amounts:
        return arr_padded, (pad_top, pad_right, pad_bottom, pad_left)
    return arr_padded


@legacy(version="0.4.0")
def compute_paddings_to_reach_aspect_ratio(arr: Array | Shape, aspect_ratio: float) -> TRBL:
    """Compute pad amounts required to fulfill an aspect ratio.

    "Pad amounts" here denotes the number of pixels that have to be added to
    each side to fulfill the desired constraint.

    The aspect ratio is given as ``ratio = width / height``.
    Depending on which dimension is smaller (height or width), only the
    corresponding sides (top/bottom or left/right) will be padded.

    The axis-wise padding amounts are always distributed equally over the
    sides of the respective axis (i.e. left and right, top and bottom). For
    odd pixel amounts, one pixel will be left over after the equal
    distribution and could be added to either side of the axis. This function
    will always add such a left over pixel to the bottom (y-axis) or
    right (x-axis) side.

    (Previously named
    ``imgaug2.imgaug2.compute_paddings_to_reach_aspect_ratio()``.)

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray or tuple of int
        Image-like array or shape tuple for which to compute pad amounts.

    aspect_ratio : float
        Target aspect ratio, given as width/height. E.g. ``2.0`` denotes the
        image having twice as much width as height.

    Returns
    -------
    tuple of int
        Required padding amounts to reach the target aspect ratio, given as a
        ``tuple`` of the form ``(top, right, bottom, left)``.

    """
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
        # image is more vertical than desired, width needs to be increased
        diff = (aspect_ratio * height) - width
        pad_right += int(np.ceil(diff / 2))
        pad_left += int(np.floor(diff / 2))
    elif aspect_ratio_current > aspect_ratio:
        # image is more horizontal than desired, height needs to be increased
        diff = ((1 / aspect_ratio) * width) - height
        pad_top += int(np.floor(diff / 2))
        pad_bottom += int(np.ceil(diff / 2))

    return pad_top, pad_right, pad_bottom, pad_left


@legacy(version="0.4.0")
def compute_croppings_to_reach_aspect_ratio(arr: Array | Shape, aspect_ratio: float) -> TRBL:
    """Compute crop amounts required to fulfill an aspect ratio.

    "Crop amounts" here denotes the number of pixels that have to be removed
    from each side to fulfill the desired constraint.

    The aspect ratio is given as ``ratio = width / height``.
    Depending on which dimension is smaller (height or width), only the
    corresponding sides (top/bottom or left/right) will be cropped.

    The axis-wise padding amounts are always distributed equally over the
    sides of the respective axis (i.e. left and right, top and bottom). For
    odd pixel amounts, one pixel will be left over after the equal
    distribution and could be added to either side of the axis. This function
    will always add such a left over pixel to the bottom (y-axis) or
    right (x-axis) side.

    If an aspect ratio cannot be reached exactly, this function will return
    rather one pixel too few than one pixel too many.


    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray or tuple of int
        Image-like array or shape tuple for which to compute crop amounts.

    aspect_ratio : float
        Target aspect ratio, given as width/height. E.g. ``2.0`` denotes the
        image having twice as much width as height.

    Returns
    -------
    tuple of int
        Required cropping amounts to reach the target aspect ratio, given as a
        ``tuple`` of the form ``(top, right, bottom, left)``.

    """
    _assert_two_or_three_dims(arr)
    assert aspect_ratio > 0, f"Expected to get an aspect ratio >0, got {aspect_ratio:.4f}."

    shape = arr.shape if hasattr(arr, "shape") else arr
    assert shape[0] > 0, f"Expected to get an array with height >0, got shape {shape}."

    height, width = shape[0:2]
    aspect_ratio_current = width / height

    top = 0
    right = 0
    bottom = 0
    left = 0

    if aspect_ratio_current < aspect_ratio:
        # image is more vertical than desired, height needs to be reduced
        # c = H - W/r
        crop_amount = height - (width / aspect_ratio)
        crop_amount = min(crop_amount, height - 1)
        top = int(np.floor(crop_amount / 2))
        bottom = int(np.ceil(crop_amount / 2))
    elif aspect_ratio_current > aspect_ratio:
        # image is more horizontal than desired, width needs to be reduced
        # c = W - Hr
        crop_amount = width - height * aspect_ratio
        crop_amount = min(crop_amount, width - 1)
        left = int(np.floor(crop_amount / 2))
        right = int(np.ceil(crop_amount / 2))

    return top, right, bottom, left


@legacy(version="0.4.0")
def compute_paddings_to_reach_multiples_of(
    arr: Array | Shape, height_multiple: int | None, width_multiple: int | None
) -> TRBL:
    """Compute pad amounts until img height/width are multiples of given values.

    See :func:`~imgaug2.imgaug2.compute_paddings_for_aspect_ratio` for an
    explanation of how the required padding amounts are distributed per
    image axis.

    (Previously named
    ``imgaug2.imgaug2.compute_paddings_to_reach_multiples_of()``.)

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray or tuple of int
        Image-like array or shape tuple for which to compute pad amounts.

    height_multiple : None or int
        The desired multiple of the height. The computed padding amount will
        reflect a padding that increases the y axis size until it is a multiple
        of this value.

    width_multiple : None or int
        The desired multiple of the width. The computed padding amount will
        reflect a padding that increases the x axis size until it is a multiple
        of this value.

    Returns
    -------
    tuple of int
        Required padding amounts to reach multiples of the provided values,
        given as a ``tuple`` of the form ``(top, right, bottom, left)``.

    """

    def _compute_axis_value(axis_size: int, multiple: int | None) -> tuple[int, int]:
        if multiple is None:
            return 0, 0
        if axis_size == 0:
            to_pad = multiple
        elif axis_size % multiple == 0:
            to_pad = 0
        else:
            to_pad = multiple - (axis_size % multiple)
        return int(np.floor(to_pad / 2)), int(np.ceil(to_pad / 2))

    _assert_two_or_three_dims(arr)

    if height_multiple is not None:
        assert height_multiple > 0, (
            f"Can only pad to multiples of 1 or larger, got {height_multiple}."
        )
    if width_multiple is not None:
        assert width_multiple > 0, (
            f"Can only pad to multiples of 1 or larger, got {width_multiple}."
        )

    shape = arr.shape if hasattr(arr, "shape") else arr
    height, width = shape[0:2]

    top, bottom = _compute_axis_value(height, height_multiple)
    left, right = _compute_axis_value(width, width_multiple)

    return top, right, bottom, left


@legacy(version="0.4.0")
def compute_croppings_to_reach_multiples_of(
    arr: Array | Shape, height_multiple: int | None, width_multiple: int | None
) -> TRBL:
    """Compute croppings to reach multiples of given heights/widths.

    See :func:`~imgaug2.imgaug2.compute_paddings_for_aspect_ratio` for an
    explanation of how the required cropping amounts are distributed per
    image axis.


    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray or tuple of int
        Image-like array or shape tuple for which to compute crop amounts.

    height_multiple : None or int
        The desired multiple of the height. The computed croppings will
        reflect a crop operation that decreases the y axis size until it is
        a multiple of this value.

    width_multiple : None or int
        The desired multiple of the width. The computed croppings amount will
        reflect a crop operation that decreases the x axis size until it is
        a multiple of this value.

    Returns
    -------
    tuple of int
        Required cropping amounts to reach multiples of the provided values,
        given as a ``tuple`` of the form ``(top, right, bottom, left)``.

    """

    def _compute_axis_value(axis_size: int, multiple: int | None) -> tuple[int, int]:
        if multiple is None:
            return 0, 0
        if axis_size == 0:
            to_crop = 0
        elif axis_size % multiple == 0:
            to_crop = 0
        else:
            to_crop = axis_size % multiple
        return int(np.floor(to_crop / 2)), int(np.ceil(to_crop / 2))

    _assert_two_or_three_dims(arr)

    if height_multiple is not None:
        assert height_multiple > 0, (
            f"Can only crop to multiples of 1 or larger, got {height_multiple}."
        )
    if width_multiple is not None:
        assert width_multiple > 0, (
            f"Can only crop to multiples of 1 or larger, got {width_multiple}."
        )

    shape = arr.shape if hasattr(arr, "shape") else arr
    height, width = shape[0:2]

    top, bottom = _compute_axis_value(height, height_multiple)
    left, right = _compute_axis_value(width, width_multiple)

    return top, right, bottom, left


@legacy(version="0.4.0")
def compute_paddings_to_reach_powers_of(
    arr: Array | Shape,
    height_base: int | None,
    width_base: int | None,
    allow_zero_exponent: bool = False,
) -> TRBL:
    """Compute paddings to reach powers of given base values.

    For given axis size ``S``, padded size ``S'`` (``S' >= S``) and base ``B``
    this function computes paddings that fulfill ``S' = B^E``, where ``E``
    is any exponent from the discrete interval ``[0 .. inf)``.

    See :func:`~imgaug2.imgaug2.compute_paddings_for_aspect_ratio` for an
    explanation of how the required padding amounts are distributed per
    image axis.

    (Previously named
    ``imgaug2.imgaug2.compute_paddings_to_reach_exponents_of()``.)

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray or tuple of int
        Image-like array or shape tuple for which to compute pad amounts.

    height_base : None or int
        The desired base of the height.

    width_base : None or int
        The desired base of the width.

    allow_zero_exponent : bool, optional
        Whether ``E=0`` in ``S'=B^E`` is a valid value. If ``True``, axes
        with size ``0`` or ``1`` will be padded up to size ``B^0=1`` and
        axes with size ``1 < S <= B`` will be padded up to ``B^1=B``.
        If ``False``, the minimum output axis size is always at least ``B``.

    Returns
    -------
    tuple of int
        Required padding amounts to fulfill ``S' = B^E`` given as a
        ``tuple`` of the form ``(top, right, bottom, left)``.

    """

    def _compute_axis_value(axis_size: int, base: int | None) -> tuple[int, int]:
        if base is None:
            return 0, 0
        if axis_size == 0:
            to_pad = 1 if allow_zero_exponent else base
        elif axis_size <= base:
            to_pad = base - axis_size
        else:
            # log_{base}(axis_size) in numpy
            exponent = np.log(axis_size) / np.log(base)

            to_pad = (base ** int(np.ceil(exponent))) - axis_size

        return int(np.floor(to_pad / 2)), int(np.ceil(to_pad / 2))

    _assert_two_or_three_dims(arr)

    if height_base is not None:
        assert height_base > 1, f"Can only pad to base larger than 1, got {height_base}."
    if width_base is not None:
        assert width_base > 1, f"Can only pad to base larger than 1, got {width_base}."

    shape = arr.shape if hasattr(arr, "shape") else arr
    height, width = shape[0:2]

    top, bottom = _compute_axis_value(height, height_base)
    left, right = _compute_axis_value(width, width_base)

    return top, right, bottom, left


@legacy(version="0.4.0")
def compute_croppings_to_reach_powers_of(
    arr: Array | Shape,
    height_base: int | None,
    width_base: int | None,
    allow_zero_exponent: bool = False,
) -> TRBL:
    """Compute croppings to reach powers of given base values.

    For given axis size ``S``, cropped size ``S'`` (``S' <= S``) and base ``B``
    this function computes croppings that fulfill ``S' = B^E``, where ``E``
    is any exponent from the discrete interval ``[0 .. inf)``.

    See :func:`~imgaug2.imgaug2.compute_paddings_for_aspect_ratio` for an
    explanation of how the required cropping amounts are distributed per
    image axis.

    .. note::

        For axes where ``S == 0``, this function alwayws returns zeros as
        croppings.

        For axes where ``1 <= S < B`` see parameter `allow_zero_exponent`.


    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray or tuple of int
        Image-like array or shape tuple for which to compute crop amounts.

    height_base : None or int
        The desired base of the height.

    width_base : None or int
        The desired base of the width.

    allow_zero_exponent : bool
        Whether ``E=0`` in ``S'=B^E`` is a valid value. If ``True``, axes
        with size ``1 <= S < B`` will be cropped to size ``B^0=1``.
        If ``False``, axes with sizes ``S < B`` will not be changed.

    Returns
    -------
    tuple of int
        Required cropping amounts to fulfill ``S' = B^E`` given as a
        ``tuple`` of the form ``(top, right, bottom, left)``.

    """

    def _compute_axis_value(axis_size: int, base: int | None) -> tuple[int, int]:
        if base is None:
            return 0, 0
        if axis_size == 0:
            to_crop = 0
        elif axis_size < base:
            # crop down to B^0 = 1
            to_crop = axis_size - 1 if allow_zero_exponent else 0
        else:
            # log_{base}(axis_size) in numpy
            exponent = np.log(axis_size) / np.log(base)

            to_crop = axis_size - (base ** int(exponent))

        return int(np.floor(to_crop / 2)), int(np.ceil(to_crop / 2))

    _assert_two_or_three_dims(arr)

    if height_base is not None:
        assert height_base > 1, f"Can only crop to base larger than 1, got {height_base}."
    if width_base is not None:
        assert width_base > 1, f"Can only crop to base larger than 1, got {width_base}."

    shape = arr.shape if hasattr(arr, "shape") else arr
    height, width = shape[0:2]

    top, bottom = _compute_axis_value(height, height_base)
    left, right = _compute_axis_value(width, width_base)

    return top, right, bottom, left


@ia.deprecated(alt_func="Resize", comment="Resize has the exactly same interface as Scale.")
def Scale(*args: object, **kwargs: object) -> Resize:
    """Augmenter that resizes images to specified heights and widths."""
    return Resize(*args, **kwargs)


class Resize(meta.Augmenter):
    """Augmenter that resizes images to specified heights and widths.

    **Supported dtypes**:

    See :func:`~imgaug2.imgaug2.imresize_many_images`.

    Parameters
    ----------
    size : 'keep' or int or float or tuple of int or tuple of float or list of int or list of float or imgaug2.parameters.StochasticParameter or dict
        The new size of the images.

            * If this has the string value ``keep``, the original height and
              width values will be kept (image is not resized).
            * If this is an ``int``, this value will always be used as the new
              height and width of the images.
            * If this is a ``float`` ``v``, then per image the image's height
              ``H`` and width ``W`` will be changed to ``H*v`` and ``W*v``.
            * If this is a ``tuple``, it is expected to have two entries
              ``(a, b)``. If at least one of these are ``float`` s, a value
              will be sampled from range ``[a, b]`` and used as the ``float``
              value to resize the image (see above). If both are ``int`` s, a
              value will be sampled from the discrete range ``[a..b]`` and
              used as the integer value to resize the image (see above).
            * If this is a ``list``, a random value from the ``list`` will be
              picked to resize the image. All values in the ``list`` must be
              ``int`` s or ``float`` s (no mixture is possible).
            * If this is a ``StochasticParameter``, then this parameter will
              first be queried once per image. The resulting value will be used
              for both height and width.
            * If this is a ``dict``, it may contain the keys ``height`` and
              ``width`` or the keys ``shorter-side`` and ``longer-side``. Each
              key may have the same datatypes as above and describes the
              scaling on x and y-axis or the shorter and longer axis,
              respectively. Both axis are sampled independently. Additionally,
              one of the keys may have the value ``keep-aspect-ratio``, which
              means that the respective side of the image will be resized so
              that the original aspect ratio is kept. This is useful when only
              resizing one image size by a pixel value (e.g. resize images to
              a height of ``64`` pixels and resize the width so that the
              overall aspect ratio is maintained).

    interpolation : imgaug2.ALL or int or str or list of int or list of str or imgaug2.parameters.StochasticParameter, optional
        Interpolation to use.

            * If ``imgaug2.ALL``, then a random interpolation from ``nearest``,
              ``linear``, ``area`` or ``cubic`` will be picked (per image).
            * If ``int``, then this interpolation will always be used.
              Expected to be any of the following:
              ``cv2.INTER_NEAREST``, ``cv2.INTER_LINEAR``, ``cv2.INTER_AREA``,
              ``cv2.INTER_CUBIC``
            * If string, then this interpolation will always be used.
              Expected to be any of the following:
              ``nearest``, ``linear``, ``area``, ``cubic``
            * If ``list`` of ``int`` / ``str``, then a random one of the values
              will be picked per image as the interpolation.
            * If a ``StochasticParameter``, then this parameter will be
              queried per image and is expected to return an ``int`` or
              ``str``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.Resize(32)

    Resize all images to ``32x32`` pixels.

    >>> aug = iaa.Resize(0.5)

    Resize all images to ``50`` percent of their original size.

    >>> aug = iaa.Resize((16, 22))

    Resize all images to a random height and width within the discrete
    interval ``[16..22]`` (uniformly sampled per image).

    >>> aug = iaa.Resize((0.5, 0.75))

    Resize all any input image so that its height (``H``) and width (``W``)
    become ``H*v`` and ``W*v``, where ``v`` is uniformly sampled from the
    interval ``[0.5, 0.75]``.

    >>> aug = iaa.Resize([16, 32, 64])

    Resize all images either to ``16x16``, ``32x32`` or ``64x64`` pixels.

    >>> aug = iaa.Resize({"height": 32})

    Resize all images to a height of ``32`` pixels and keeps the original
    width.

    >>> aug = iaa.Resize({"height": 32, "width": 48})

    Resize all images to a height of ``32`` pixels and a width of ``48``.

    >>> aug = iaa.Resize({"height": 32, "width": "keep-aspect-ratio"})

    Resize all images to a height of ``32`` pixels and resizes the
    x-axis (width) so that the aspect ratio is maintained.

    >>> aug = iaa.Resize(
    >>>     {"shorter-side": 224, "longer-side": "keep-aspect-ratio"})

    Resize all images to a height/width of ``224`` pixels, depending on which
    axis is shorter and resize the other axis so that the aspect ratio is
    maintained.

    >>> aug = iaa.Resize({"height": (0.5, 0.75), "width": [16, 32, 64]})

    Resize all images to a height of ``H*v``, where ``H`` is the original
    height and ``v`` is a random value sampled from the interval
    ``[0.5, 0.75]``. The width/x-axis of each image is resized to either
    ``16`` or ``32`` or ``64`` pixels.

    >>> aug = iaa.Resize(32, interpolation=["linear", "cubic"])

    Resize all images to ``32x32`` pixels. Randomly use either ``linear``
    or ``cubic`` interpolation.

    """

    def __init__(
        self,
        size: ResizeSizeInput,
        interpolation: ResizeInterpolationInput = "cubic",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.size, self.size_order = self._handle_size_arg(size, False)
        self.interpolation = self._handle_interpolation_arg(interpolation)

    @classmethod
    @iap._prefetchable_str
    def _handle_size_arg(
        cls, size: ResizeSizeInput, subcall: bool
    ) -> ResizeSizeParam | tuple[ResizeSizeParam, str]:
        def _dict_to_size_tuple(
            val1: ResizeSizeDictValue, val2: ResizeSizeDictValue
        ) -> tuple[iap.StochasticParameter, iap.StochasticParameter]:
            kaa = "keep-aspect-ratio"
            not_both_kaa = val1 != kaa or val2 != kaa
            assert not_both_kaa, (
                "Expected at least one value to not be \"keep-aspect-ratio\", but got it two times."
            )

            size_tuple = []
            for k in [val1, val2]:
                if k in ["keep-aspect-ratio", "keep"]:
                    entry = iap.Deterministic(k)
                else:
                    entry = cls._handle_size_arg(k, True)
                size_tuple.append(entry)
            return tuple(size_tuple)

        def _contains_any_key(dict_: dict[str, object], keys: Sequence[str]) -> bool:
            return any([key in dict_ for key in keys])

        # HW = height, width
        # SL = shorter, longer
        size_order = "HW"

        if size == "keep":
            result = iap.Deterministic("keep")
        elif ia.is_single_number(size):
            assert size > 0, f"Expected only values > 0, got {size}"
            result = iap.Deterministic(size)
        elif not subcall and isinstance(size, dict):
            if len(size.keys()) == 0:
                result = iap.Deterministic("keep")
            elif _contains_any_key(size, ["height", "width"]):
                height = size.get("height", "keep")
                width = size.get("width", "keep")
                result = _dict_to_size_tuple(height, width)
            elif _contains_any_key(size, ["shorter-side", "longer-side"]):
                shorter = size.get("shorter-side", "keep")
                longer = size.get("longer-side", "keep")
                result = _dict_to_size_tuple(shorter, longer)
                size_order = "SL"
            else:
                raise ValueError(
                    "Expected dictionary containing no keys, "
                    "the keys \"height\" and/or \"width\", "
                    "or the keys \"shorter-side\" and/or \"longer-side\". "
                    f"Got keys: {str(size.keys())}."
                )
        elif isinstance(size, tuple):
            assert len(size) == 2, (
                f"Expected size tuple to contain exactly 2 values, got {len(size)}."
            )
            assert size[0] > 0 and size[1] > 0, (
                f"Expected size tuple to only contain values >0, got {size[0]} and {size[1]}."
            )
            if ia.is_single_float(size[0]) or ia.is_single_float(size[1]):
                result = iap.Uniform(size[0], size[1])
            else:
                result = iap.DiscreteUniform(size[0], size[1])
        elif isinstance(size, list):
            if len(size) == 0:
                result = iap.Deterministic("keep")
            else:
                all_int = all([ia.is_single_integer(v) for v in size])
                all_float = all([ia.is_single_float(v) for v in size])
                assert all_int or all_float, "Expected to get only integers or floats."
                assert all([v > 0 for v in size]), "Expected all values to be >0."
                result = iap.Choice(size)
        elif isinstance(size, iap.StochasticParameter):
            result = size
        else:
            raise ValueError(
                "Expected number, tuple of two numbers, list of numbers, "
                "dictionary of form "
                "{'height': number/tuple/list/'keep-aspect-ratio'/'keep', "
                "'width': <analogous>}, dictionary of form "
                "{'shorter-side': number/tuple/list/'keep-aspect-ratio'/"
                "'keep', 'longer-side': <analogous>} "
                f"or StochasticParameter, got {type(size)}."
            )

        if subcall:
            return result
        return result, size_order

    @classmethod
    def _handle_interpolation_arg(
        cls, interpolation: ResizeInterpolationInput
    ) -> iap.StochasticParameter:
        if interpolation == ia.ALL:
            interpolation = iap.Choice(["nearest", "linear", "area", "cubic"])
        elif ia.is_single_integer(interpolation):
            interpolation = iap.Deterministic(interpolation)
        elif ia.is_string(interpolation):
            interpolation = iap.Deterministic(interpolation)
        elif ia.is_iterable(interpolation):
            interpolation = iap.Choice(interpolation)
        elif isinstance(interpolation, iap.StochasticParameter):
            pass
        else:
            raise Exception(
                "Expected int or string or iterable or StochasticParameter, "
                f"got {type(interpolation)}."
            )
        return interpolation

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        nb_rows = batch.nb_rows
        samples = self._draw_samples(nb_rows, random_state)

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images, samples)

        if batch.heatmaps is not None:
            # TODO this uses the same interpolation as for images for heatmaps
            #      while other augmenters resort to cubic
            batch.heatmaps = self._augment_maps_by_samples(batch.heatmaps, "arr_0to1", samples)

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps, "arr", (samples[0], samples[1], [None] * nb_rows)
            )

        for augm_name in ["keypoints", "bounding_boxes", "polygons", "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(self._augment_keypoints_by_samples, samples=samples)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    @legacy(version="0.4.0")
    def _augment_images_by_samples(self, images: Images, samples: ResizeSamplingResult) -> Images:
        input_was_array = False
        input_dtype = None
        if ia.is_np_array(images):
            input_was_array = True
            input_dtype = images.dtype

        samples_a, samples_b, samples_ip = samples
        result = []
        for i, image in enumerate(images):
            h, w = self._compute_height_width(
                image.shape, samples_a[i], samples_b[i], self.size_order
            )
            from imgaug2.mlx._core import is_mlx_array

            if is_mlx_array(image):
                import imgaug2.mlx as mlx

                if image.size == 0:
                    image_rs = image
                else:
                    order = _mlx_resize_order_from_interpolation(samples_ip[i])
                    if order is None:
                        image_np = mlx.to_numpy(image)
                        image_np = ia.imresize_single_image(
                            image_np, (h, w), interpolation=samples_ip[i]
                        )
                        image_rs = mlx.to_mlx(image_np)
                    else:
                        image_rs = mlx.geometry.resize(image, (h, w), order=order)
            else:
                image_rs = ia.imresize_single_image(image, (h, w), interpolation=samples_ip[i])
            result.append(image_rs)

        if input_was_array:
            all_same_size = len({image.shape for image in result}) == 1
            if all_same_size:
                result = np.array(result, dtype=input_dtype)

        return result

    @legacy(version="0.4.0")
    def _augment_maps_by_samples(
        self,
        augmentables: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage],
        arr_attr_name: str,
        samples: ResizeSamplingResult,
    ) -> list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage]:
        result = []
        samples_h, samples_w, samples_ip = samples

        for i, augmentable in enumerate(augmentables):
            arr = getattr(augmentable, arr_attr_name)
            arr_shape = arr.shape
            img_shape = augmentable.shape
            h_img, w_img = self._compute_height_width(
                img_shape, samples_h[i], samples_w[i], self.size_order
            )
            h = int(np.round(h_img * (arr_shape[0] / img_shape[0])))
            w = int(np.round(w_img * (arr_shape[1] / img_shape[1])))
            h = max(h, 1)
            w = max(w, 1)
            if samples_ip[0] is not None:
                # TODO change this for heatmaps to always have cubic or
                #      automatic interpolation?
                augmentable_resize = augmentable.resize((h, w), interpolation=samples_ip[i])
            else:
                augmentable_resize = augmentable.resize((h, w))
            augmentable_resize.shape = (h_img, w_img) + img_shape[2:]
            result.append(augmentable_resize)

        return result

    @legacy(version="0.4.0")
    def _augment_keypoints_by_samples(
        self, kpsois: list[ia.KeypointsOnImage], samples: ResizeSamplingResult
    ) -> list[ia.KeypointsOnImage]:
        result = []
        samples_a, samples_b, _samples_ip = samples
        for i, kpsoi in enumerate(kpsois):
            h, w = self._compute_height_width(
                kpsoi.shape, samples_a[i], samples_b[i], self.size_order
            )
            new_shape = (h, w) + kpsoi.shape[2:]
            keypoints_on_image_rs = kpsoi.on_(new_shape)

            result.append(keypoints_on_image_rs)

        return result

    def _draw_samples(self, nb_images: int, random_state: iarandom.RNG) -> ResizeSamplingResult:
        rngs = random_state.duplicate(3)
        if isinstance(self.size, tuple):
            samples_h = self.size[0].draw_samples(nb_images, random_state=rngs[0])
            samples_w = self.size[1].draw_samples(nb_images, random_state=rngs[1])
        else:
            samples_h = self.size.draw_samples(nb_images, random_state=rngs[0])
            samples_w = samples_h

        samples_ip = self.interpolation.draw_samples(nb_images, random_state=rngs[2])
        return samples_h, samples_w, samples_ip

    @classmethod
    def _compute_height_width(
        cls,
        image_shape: Shape,
        sample_a: float | int | str,
        sample_b: float | int | str,
        size_order: str,
    ) -> tuple[int, int]:
        imh, imw = image_shape[0:2]

        if size_order == 'SL':
            # size order: short, long
            if imh < imw:
                h, w = sample_a, sample_b
            else:
                w, h = sample_a, sample_b
        else:
            # size order: height, width
            h, w = sample_a, sample_b

        if ia.is_single_float(h):
            assert h > 0, f"Expected 'h' to be >0, got {h:.4f}"
            h = int(np.round(imh * h))
            h = h if h > 0 else 1
        elif h == "keep":
            h = imh
        if ia.is_single_float(w):
            assert w > 0, f"Expected 'w' to be >0, got {w:.4f}"
            w = int(np.round(imw * w))
            w = w if w > 0 else 1
        elif w == "keep":
            w = imw

        # at least the checks for keep-aspect-ratio must come after
        # the float checks, as they are dependent on the results
        # this is also why these are not written as elifs
        if h == "keep-aspect-ratio":
            h_per_w_orig = imh / imw
            h = int(np.round(w * h_per_w_orig))
        if w == "keep-aspect-ratio":
            w_per_h_orig = imw / imh
            w = int(np.round(h * w_per_h_orig))

        return int(h), int(w)

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.size, self.interpolation, self.size_order]


class _CropAndPadSamplingResult:
    def __init__(
        self,
        crop_top: Array,
        crop_right: Array,
        crop_bottom: Array,
        crop_left: Array,
        pad_top: Array,
        pad_right: Array,
        pad_bottom: Array,
        pad_left: Array,
        pad_mode: Array,
        pad_cval: Array,
    ) -> None:
        self.crop_top = crop_top
        self.crop_right = crop_right
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.pad_top = pad_top
        self.pad_right = pad_right
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_mode = pad_mode
        self.pad_cval = pad_cval

    def croppings(self, i: int) -> TRBL:
        """Get absolute pixel amounts of croppings as a TRBL tuple."""
        return (
            int(self.crop_top[i]),
            int(self.crop_right[i]),
            int(self.crop_bottom[i]),
            int(self.crop_left[i]),
        )

    def paddings(self, i: int) -> TRBL:
        """Get absolute pixel amounts of paddings as a TRBL tuple."""
        return (
            int(self.pad_top[i]),
            int(self.pad_right[i]),
            int(self.pad_bottom[i]),
            int(self.pad_left[i]),
        )


class CropAndPad(meta.Augmenter):
    """Crop/pad images by pixel amounts or fractions of image sizes.

    Cropping removes pixels at the sides (i.e. extracts a subimage from
    a given full image). Padding adds pixels to the sides (e.g. black pixels).

    This augmenter will never crop images below a height or width of ``1``.

    .. note::

        This augmenter automatically resizes images back to their original size
        after it has augmented them. To deactivate this, add the
        parameter ``keep_size=False``.

    **Supported dtypes**:

    if (keep_size=False):

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    if (keep_size=True):

        minimum of (
            ``imgaug2.augmenters.size.CropAndPad(keep_size=False)``,
            :func:`~imgaug2.imgaug2.imresize_many_images`
        )

    Parameters
    ----------
    px : None or int or imgaug2.parameters.StochasticParameter or tuple, optional
        The number of pixels to crop (negative values) or pad (positive values)
        on each side of the image. Either this or the parameter `percent` may
        be set, not both at the same time.

            * If ``None``, then pixel-based cropping/padding will not be used.
            * If ``int``, then that exact number of pixels will always be
              cropped/padded.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left), unless `sample_independently` is set to ``False``,
              as then only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``int`` s with values ``a`` and ``b``,
              then each side will be cropped/padded by a random amount sampled
              uniformly per image and side from the inteval ``[a, b]``. If
              however `sample_independently` is set to ``False``, only one
              value will be sampled per image and used for all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``int`` (always
              crop/pad by exactly that value), a ``tuple`` of two ``int`` s
              ``a`` and ``b`` (crop/pad by an amount within ``[a, b]``), a
              ``list`` of ``int`` s (crop/pad by a random value that is
              contained in the ``list``) or a ``StochasticParameter`` (sample
              the amount to crop/pad from that parameter).

    percent : None or number or imgaug2.parameters.StochasticParameter or tuple, optional
        The number of pixels to crop (negative values) or pad (positive values)
        on each side of the image given as a *fraction* of the image
        height/width. E.g. if this is set to ``-0.1``, the augmenter will
        always crop away ``10%`` of the image's height at both the top and the
        bottom (both ``10%`` each), as well as ``10%`` of the width at the
        right and left.
        Expected value range is ``(-1.0, inf)``.
        Either this or the parameter `px` may be set, not both
        at the same time.

            * If ``None``, then fraction-based cropping/padding will not be
              used.
            * If ``number``, then that fraction will always be cropped/padded.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left). If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``float`` s with values ``a`` and ``b``,
              then each side will be cropped/padded by a random fraction
              sampled uniformly per image and side from the interval
              ``[a, b]``. If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``float``
              (always crop/pad by exactly that percent value), a ``tuple`` of
              two ``float`` s ``a`` and ``b`` (crop/pad by a fraction from
              ``[a, b]``), a ``list`` of ``float`` s (crop/pad by a random
              value that is contained in the list) or a ``StochasticParameter``
              (sample the percentage to crop/pad from that parameter).

    pad_mode : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        Padding mode to use. The available modes match the numpy padding modes,
        i.e. ``constant``, ``edge``, ``linear_ramp``, ``maximum``, ``median``,
        ``minimum``, ``reflect``, ``symmetric``, ``wrap``. The modes
        ``constant`` and ``linear_ramp`` use extra values, which are provided
        by ``pad_cval`` when necessary. See :func:`~imgaug2.imgaug2.pad` for
        more details.

            * If ``imgaug2.ALL``, then a random mode from all available modes
              will be sampled per image.
            * If a ``str``, it will be used as the pad mode for all images.
            * If a ``list`` of ``str``, a random one of these will be sampled
              per image and used as the mode.
            * If ``StochasticParameter``, a random mode will be sampled from
              this parameter per image.

    pad_cval : number or tuple of number list of number or imgaug2.parameters.StochasticParameter, optional
        The constant value to use if the pad mode is ``constant`` or the end
        value to use if the mode is ``linear_ramp``.
        See :func:`~imgaug2.imgaug2.pad` for more details.

            * If ``number``, then that value will be used.
            * If a ``tuple`` of two ``number`` s and at least one of them is
              a ``float``, then a random number will be uniformly sampled per
              image from the continuous interval ``[a, b]`` and used as the
              value. If both ``number`` s are ``int`` s, the interval is
              discrete.
            * If a ``list`` of ``number``, then a random value will be chosen
              from the elements of the ``list`` and used as the value.
            * If ``StochasticParameter``, a random value will be sampled from
              that parameter per image.

    keep_size : bool, optional
        After cropping and padding, the result image will usually have a
        different height/width compared to the original input image. If this
        parameter is set to ``True``, then the cropped/padded image will be
        resized to the input image's size, i.e. the augmenter's output shape
        is always identical to the input shape.

    sample_independently : bool, optional
        If ``False`` *and* the values for `px`/`percent` result in exactly
        *one* probability distribution for all image sides, only one single
        value will be sampled from that probability distribution and used for
        all sides. I.e. the crop/pad amount then is the same for all sides.
        If ``True``, four values will be sampled independently, one per side.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.CropAndPad(px=(-10, 0))

    Crop each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[-10..0]``.

    >>> aug = iaa.CropAndPad(px=(0, 10))

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. The padding happens by
    zero-padding, i.e. it adds black pixels (default setting).

    >>> aug = iaa.CropAndPad(px=(0, 10), pad_mode="edge")

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. The padding uses the
    ``edge`` mode from numpy's pad function, i.e. the pixel colors around
    the image sides are repeated.

    >>> aug = iaa.CropAndPad(px=(0, 10), pad_mode=["constant", "edge"])

    Similar to the previous example, but uses zero-padding (``constant``) for
    half of the images and ``edge`` padding for the other half.

    >>> aug = iaa.CropAndPad(px=(0, 10), pad_mode=ia.ALL, pad_cval=(0, 255))

    Similar to the previous example, but uses any available padding mode.
    In case the padding mode ends up being ``constant`` or ``linear_ramp``,
    and random intensity is uniformly sampled (once per image) from the
    discrete interval ``[0..255]`` and used as the intensity of the new
    pixels.

    >>> aug = iaa.CropAndPad(px=(0, 10), sample_independently=False)

    Pad each side by a random pixel value sampled uniformly once per image
    from the discrete interval ``[0..10]``. Each sampled value is used
    for *all* sides of the corresponding image.

    >>> aug = iaa.CropAndPad(px=(0, 10), keep_size=False)

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. Afterwards, do **not**
    resize the padded image back to the input image's size. This will increase
    the image's height and width by a maximum of ``20`` pixels.

    >>> aug = iaa.CropAndPad(px=((0, 10), (0, 5), (0, 10), (0, 5)))

    Pad the top and bottom by a random pixel value sampled uniformly from the
    discrete interval ``[0..10]``. Pad the left and right analogously by
    a random value sampled from ``[0..5]``. Each value is always sampled
    independently.

    >>> aug = iaa.CropAndPad(percent=(0, 0.1))

    Pad each side by a random fraction sampled uniformly from the continuous
    interval ``[0.0, 0.10]``. The fraction is sampled once per image and
    side. E.g. a sampled fraction of ``0.1`` for the top side would pad by
    ``0.1*H``, where ``H`` is the height of the input image.

    >>> aug = iaa.CropAndPad(
    >>>     percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))

    Pads each side by either ``5%`` or ``10%``. The values are sampled
    once per side and image.

    >>> aug = iaa.CropAndPad(px=(-10, 10))

    Sample uniformly per image and side a value ``v`` from the discrete range
    ``[-10..10]``. Then either crop (negative sample) or pad (positive sample)
    the side by ``v`` pixels.

    """

    def __init__(
        self,
        px: CropAndPadPxInput | None = None,
        percent: CropAndPadPercentInput | None = None,
        pad_mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        pad_cval: ParamInput = 0,
        keep_size: bool = True,
        sample_independently: bool = True,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        if px is None and percent is None:
            percent = (-0.1, 0.1)

        self.mode, self.all_sides, self.top, self.right, self.bottom, self.left = (
            self._handle_px_and_percent_args(px, percent)
        )

        self.pad_mode = _handle_pad_mode_param(pad_mode)
        # TODO enable ALL here, like in e.g. Affine
        self.pad_cval = iap.handle_discrete_param(
            pad_cval,
            "pad_cval",
            value_range=None,
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=True,
        )

        self.keep_size = keep_size
        self.sample_independently = sample_independently

        # set these to None to use the same values as sampled for the
        # images (not tested)
        self._pad_mode_heatmaps = "constant"
        self._pad_mode_segmentation_maps = "constant"
        self._pad_cval_heatmaps = 0.0
        self._pad_cval_segmentation_maps = 0

    @classmethod
    @iap._prefetchable
    def _handle_px_and_percent_args(
        cls, px: CropAndPadPxInput | None, percent: CropAndPadPercentInput | None
    ) -> CropAndPadParamReturn:
        all_sides = None
        top, right, bottom, left = None, None, None, None

        if px is None and percent is None:
            mode = "noop"
        elif px is not None and percent is not None:
            raise Exception("Can only pad by pixels or percent, not both.")
        elif px is not None:
            mode = "px"
            all_sides, top, right, bottom, left = cls._handle_px_arg(px)
        else:  # = elif percent is not None:
            mode = "percent"
            all_sides, top, right, bottom, left = cls._handle_percent_arg(percent)
        return mode, all_sides, top, right, bottom, left

    @classmethod
    def _handle_px_arg(
        cls, px: CropAndPadPxInput
    ) -> tuple[
        iap.StochasticParameter | None,
        iap.StochasticParameter | None,
        iap.StochasticParameter | None,
        iap.StochasticParameter | None,
        iap.StochasticParameter | None,
    ]:
        all_sides = None
        top, right, bottom, left = None, None, None, None

        if ia.is_single_integer(px):
            all_sides = iap.Deterministic(px)
        elif isinstance(px, tuple):
            assert len(px) in [2, 4], (
                f"Expected 'px' given as a tuple to contain 2 or 4 entries, got {len(px)}."
            )

            def handle_param(
                p: CropAndPadPxSingleParam | tuple[int, int],
            ) -> iap.StochasticParameter:
                if ia.is_single_integer(p):
                    return iap.Deterministic(p)
                if isinstance(p, tuple):
                    assert len(p) == 2, f"Expected tuple of 2 values, got {len(p)}."
                    only_ints = ia.is_single_integer(p[0]) and ia.is_single_integer(p[1])
                    assert only_ints, (
                        f"Expected tuple of integers, got {type(p[0])} and {type(p[1])}."
                    )
                    return iap.DiscreteUniform(p[0], p[1])
                if isinstance(p, list):
                    assert len(p) > 0, "Expected non-empty list, but got empty one."
                    assert all([ia.is_single_integer(val) for val in p]), (
                        "Expected list of ints, got types {}.".format(
                            ", ".join([str(type(v)) for v in p])
                        )
                    )
                    return iap.Choice(p)
                if isinstance(p, iap.StochasticParameter):
                    return p
                raise Exception(
                    "Expected int, tuple of two ints, list of ints or "
                    f"StochasticParameter, got type {type(p)}."
                )

            if len(px) == 2:
                all_sides = handle_param(px)
            else:  # len == 4
                top = handle_param(px[0])
                right = handle_param(px[1])
                bottom = handle_param(px[2])
                left = handle_param(px[3])
        elif isinstance(px, iap.StochasticParameter):
            top = right = bottom = left = px
        else:
            raise Exception(
                "Expected int, tuple of 4 "
                "ints/tuples/lists/StochasticParameters or "
                f"StochasticParameter, got type {type(px)}."
            )
        return all_sides, top, right, bottom, left

    @classmethod
    def _handle_percent_arg(
        cls, percent: CropAndPadPercentInput
    ) -> tuple[
        iap.StochasticParameter | None,
        iap.StochasticParameter | None,
        iap.StochasticParameter | None,
        iap.StochasticParameter | None,
        iap.StochasticParameter | None,
    ]:
        all_sides = None
        top, right, bottom, left = None, None, None, None

        if ia.is_single_number(percent):
            assert percent > -1.0, f"Expected 'percent' to be >-1.0, got {percent:.4f}."
            all_sides = iap.Deterministic(percent)
        elif isinstance(percent, tuple):
            assert len(percent) in [2, 4], (
                "Expected 'percent' given as a tuple to contain 2 or 4 "
                f"entries, got {len(percent)}."
            )

            def handle_param(
                p: CropAndPadPercentSingleParam | tuple[float | int, float | int],
            ) -> iap.StochasticParameter:
                if ia.is_single_number(p):
                    return iap.Deterministic(p)
                if isinstance(p, tuple):
                    assert len(p) == 2, f"Expected tuple of 2 values, got {len(p)}."
                    only_numbers = ia.is_single_number(p[0]) and ia.is_single_number(p[1])
                    assert only_numbers, (
                        f"Expected tuple of numbers, got {type(p[0])} and {type(p[1])}."
                    )
                    assert p[0] > -1.0 and p[1] > -1.0, (
                        f"Expected tuple of values >-1.0, got {p[0]:.4f} and {p[1]:.4f}."
                    )
                    return iap.Uniform(p[0], p[1])
                if isinstance(p, list):
                    assert len(p) > 0, "Expected non-empty list, but got empty one."
                    assert all([ia.is_single_number(val) for val in p]), (
                        "Expected list of numbers, got types {}.".format(
                            ", ".join([str(type(v)) for v in p])
                        )
                    )
                    assert all([val > -1.0 for val in p]), (
                        "Expected list of values >-1.0, got values {}.".format(
                            ", ".join([f"{v:.4f}" for v in p])
                        )
                    )
                    return iap.Choice(p)
                if isinstance(p, iap.StochasticParameter):
                    return p
                raise Exception(
                    "Expected int, tuple of two ints, list of ints or "
                    f"StochasticParameter, got type {type(p)}."
                )

            if len(percent) == 2:
                all_sides = handle_param(percent)
            else:  # len == 4
                top = handle_param(percent[0])
                right = handle_param(percent[1])
                bottom = handle_param(percent[2])
                left = handle_param(percent[3])
        elif isinstance(percent, iap.StochasticParameter):
            top = right = bottom = left = percent
        else:
            raise Exception(
                "Expected number, tuple of 4 "
                "numbers/tuples/lists/StochasticParameters or "
                f"StochasticParameter, got type {type(percent)}."
            )
        return all_sides, top, right, bottom, left

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        shapes = batch.get_rowwise_shapes()
        samples = self._draw_samples(random_state, shapes)

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images, samples)

        if batch.heatmaps is not None:
            batch.heatmaps = self._augment_maps_by_samples(
                batch.heatmaps, self._pad_mode_heatmaps, self._pad_cval_heatmaps, samples
            )

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps,
                self._pad_mode_segmentation_maps,
                self._pad_cval_segmentation_maps,
                samples,
            )

        for augm_name in ["keypoints", "bounding_boxes", "polygons", "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(self._augment_keypoints_by_samples, samples=samples)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    @legacy(version="0.4.0")
    def _augment_images_by_samples(
        self, images: Images, samples: _CropAndPadSamplingResult
    ) -> Images:
        result = []
        for i, image in enumerate(images):
            image_cr_pa = _crop_and_pad_arr(
                image,
                samples.croppings(i),
                samples.paddings(i),
                samples.pad_mode[i],
                samples.pad_cval[i],
                self.keep_size,
            )

            result.append(image_cr_pa)

        if ia.is_np_array(images):
            if self.keep_size:
                result = np.array(result, dtype=images.dtype)
            else:
                nb_shapes = len({image.shape for image in result})
                if nb_shapes == 1:
                    result = np.array(result, dtype=images.dtype)

        return result

    @legacy(version="0.4.0")
    def _augment_maps_by_samples(
        self,
        augmentables: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage],
        pad_mode: str | None,
        pad_cval: float | int | np.number | None,
        samples: _CropAndPadSamplingResult,
    ) -> list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage]:
        result = []
        for i, augmentable in enumerate(augmentables):
            augmentable = _crop_and_pad_hms_or_segmaps_(
                augmentable,
                croppings_img=samples.croppings(i),
                paddings_img=samples.paddings(i),
                pad_mode=(pad_mode if pad_mode is not None else samples.pad_mode[i]),
                pad_cval=(pad_cval if pad_cval is not None else samples.pad_cval[i]),
                keep_size=self.keep_size,
            )

            result.append(augmentable)

        return result

    @legacy(version="0.4.0")
    def _augment_keypoints_by_samples(
        self, keypoints_on_images: list[ia.KeypointsOnImage], samples: _CropAndPadSamplingResult
    ) -> list[ia.KeypointsOnImage]:
        result = []
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            kpsoi_aug = _crop_and_pad_kpsoi_(
                keypoints_on_image,
                croppings_img=samples.croppings(i),
                paddings_img=samples.paddings(i),
                keep_size=self.keep_size,
            )
            result.append(kpsoi_aug)

        return result

    def _draw_samples(
        self, random_state: iarandom.RNG, shapes: Sequence[Shape]
    ) -> _CropAndPadSamplingResult:
        nb_rows = len(shapes)

        shapes_arr = np.array([shape[0:2] for shape in shapes], dtype=np.int32)
        heights = shapes_arr[:, 0]
        widths = shapes_arr[:, 1]

        if self.mode == "noop":
            top = right = bottom = left = np.full((nb_rows,), 0, dtype=np.int32)
        else:
            if self.all_sides is not None:
                if self.sample_independently:
                    samples = self.all_sides.draw_samples((nb_rows, 4), random_state=random_state)
                    top = samples[:, 0]
                    right = samples[:, 1]
                    bottom = samples[:, 2]
                    left = samples[:, 3]
                else:
                    sample = self.all_sides.draw_samples((nb_rows,), random_state=random_state)
                    top = right = bottom = left = sample
            else:
                top = self.top.draw_samples((nb_rows,), random_state=random_state)
                right = self.right.draw_samples((nb_rows,), random_state=random_state)
                bottom = self.bottom.draw_samples((nb_rows,), random_state=random_state)
                left = self.left.draw_samples((nb_rows,), random_state=random_state)

            if self.mode == "px":
                # no change necessary for pixel values
                pass
            elif self.mode == "percent":
                # percentage values have to be transformed to pixel values
                heights_f = heights.astype(np.float32)
                widths_f = widths.astype(np.float32)
                top = np.round(heights_f * top).astype(np.int32)
                right = np.round(widths_f * right).astype(np.int32)
                bottom = np.round(heights_f * bottom).astype(np.int32)
                left = np.round(widths_f * left).astype(np.int32)
            else:
                raise Exception("Invalid mode")

        # np.maximum(., 0) is a bit faster than arr[arr < 0] = 0 and
        # significantly faster than clip. The masks could be computed once
        # along each side, but it doesn't look like that would improve things
        # very much.
        crop_top = np.maximum((-1) * top, 0)
        crop_right = np.maximum((-1) * right, 0)
        crop_bottom = np.maximum((-1) * bottom, 0)
        crop_left = np.maximum((-1) * left, 0)

        crop_top, crop_bottom = _prevent_zero_sizes_after_crops_(heights, crop_top, crop_bottom)
        crop_left, crop_right = _prevent_zero_sizes_after_crops_(widths, crop_left, crop_right)

        pad_top = np.maximum(top, 0)
        pad_right = np.maximum(right, 0)
        pad_bottom = np.maximum(bottom, 0)
        pad_left = np.maximum(left, 0)

        pad_mode = self.pad_mode.draw_samples((nb_rows,), random_state=random_state)
        pad_cval = self.pad_cval.draw_samples((nb_rows,), random_state=random_state)

        return _CropAndPadSamplingResult(
            crop_top=crop_top,
            crop_right=crop_right,
            crop_bottom=crop_bottom,
            crop_left=crop_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
        )

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [
            self.all_sides,
            self.top,
            self.right,
            self.bottom,
            self.left,
            self.pad_mode,
            self.pad_cval,
        ]


class Pad(CropAndPad):
    """Pad images, i.e. adds columns/rows of pixels to them.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.CropAndPad`.

    Parameters
    ----------
    px : None or int or imgaug2.parameters.StochasticParameter or tuple, optional
        The number of pixels to pad on each side of the image.
        Expected value range is ``[0, inf)``.
        Either this or the parameter `percent` may be set, not both at the same
        time.

            * If ``None``, then pixel-based padding will not be used.
            * If ``int``, then that exact number of pixels will always be
              padded.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left), unless `sample_independently` is set to ``False``,
              as then only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``int`` s with values ``a`` and ``b``,
              then each side will be padded by a random amount sampled
              uniformly per image and side from the inteval ``[a, b]``. If
              however `sample_independently` is set to ``False``, only one
              value will be sampled per image and used for all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``int`` (always
              pad by exactly that value), a ``tuple`` of two ``int`` s
              ``a`` and ``b`` (pad by an amount within ``[a, b]``), a
              ``list`` of ``int`` s (pad by a random value that is
              contained in the ``list``) or a ``StochasticParameter`` (sample
              the amount to pad from that parameter).

    percent : None or int or float or imgaug2.parameters.StochasticParameter or tuple, optional
        The number of pixels to pad
        on each side of the image given as a *fraction* of the image
        height/width. E.g. if this is set to ``0.1``, the augmenter will
        always pad ``10%`` of the image's height at both the top and the
        bottom (both ``10%`` each), as well as ``10%`` of the width at the
        right and left.
        Expected value range is ``[0.0, inf)``.
        Either this or the parameter `px` may be set, not both
        at the same time.

            * If ``None``, then fraction-based padding will not be
              used.
            * If ``number``, then that fraction will always be padded.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left). If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``float`` s with values ``a`` and ``b``,
              then each side will be padded by a random fraction
              sampled uniformly per image and side from the interval
              ``[a, b]``. If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``float``
              (always pad by exactly that fraction), a ``tuple`` of
              two ``float`` s ``a`` and ``b`` (pad by a fraction from
              ``[a, b]``), a ``list`` of ``float`` s (pad by a random
              value that is contained in the list) or a ``StochasticParameter``
              (sample the percentage to pad from that parameter).

    pad_mode : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        Padding mode to use. The available modes match the numpy padding modes,
        i.e. ``constant``, ``edge``, ``linear_ramp``, ``maximum``, ``median``,
        ``minimum``, ``reflect``, ``symmetric``, ``wrap``. The modes
        ``constant`` and ``linear_ramp`` use extra values, which are provided
        by ``pad_cval`` when necessary. See :func:`~imgaug2.imgaug2.pad` for
        more details.

            * If ``imgaug2.ALL``, then a random mode from all available modes
              will be sampled per image.
            * If a ``str``, it will be used as the pad mode for all images.
            * If a ``list`` of ``str``, a random one of these will be sampled
              per image and used as the mode.
            * If ``StochasticParameter``, a random mode will be sampled from
              this parameter per image.

    pad_cval : number or tuple of number list of number or imgaug2.parameters.StochasticParameter, optional
        The constant value to use if the pad mode is ``constant`` or the end
        value to use if the mode is ``linear_ramp``.
        See :func:`~imgaug2.imgaug2.pad` for more details.

            * If ``number``, then that value will be used.
            * If a ``tuple`` of two ``number`` s and at least one of them is
              a ``float``, then a random number will be uniformly sampled per
              image from the continuous interval ``[a, b]`` and used as the
              value. If both ``number`` s are ``int`` s, the interval is
              discrete.
            * If a ``list`` of ``number``, then a random value will be chosen
              from the elements of the ``list`` and used as the value.
            * If ``StochasticParameter``, a random value will be sampled from
              that parameter per image.

    keep_size : bool, optional
        After padding, the result image will usually have a
        different height/width compared to the original input image. If this
        parameter is set to ``True``, then the padded image will be
        resized to the input image's size, i.e. the augmenter's output shape
        is always identical to the input shape.

    sample_independently : bool, optional
        If ``False`` *and* the values for `px`/`percent` result in exactly
        *one* probability distribution for all image sides, only one single
        value will be sampled from that probability distribution and used for
        all sides. I.e. the pad amount then is the same for all sides.
        If ``True``, four values will be sampled independently, one per side.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.Pad(px=(0, 10))

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. The padding happens by
    zero-padding, i.e. it adds black pixels (default setting).

    >>> aug = iaa.Pad(px=(0, 10), pad_mode="edge")

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. The padding uses the
    ``edge`` mode from numpy's pad function, i.e. the pixel colors around
    the image sides are repeated.

    >>> aug = iaa.Pad(px=(0, 10), pad_mode=["constant", "edge"])

    Similar to the previous example, but uses zero-padding (``constant``) for
    half of the images and ``edge`` padding for the other half.

    >>> aug = iaa.Pad(px=(0, 10), pad_mode=ia.ALL, pad_cval=(0, 255))

    Similar to the previous example, but uses any available padding mode.
    In case the padding mode ends up being ``constant`` or ``linear_ramp``,
    and random intensity is uniformly sampled (once per image) from the
    discrete interval ``[0..255]`` and used as the intensity of the new
    pixels.

    >>> aug = iaa.Pad(px=(0, 10), sample_independently=False)

    Pad each side by a random pixel value sampled uniformly once per image
    from the discrete interval ``[0..10]``. Each sampled value is used
    for *all* sides of the corresponding image.

    >>> aug = iaa.Pad(px=(0, 10), keep_size=False)

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. Afterwards, do **not**
    resize the padded image back to the input image's size. This will increase
    the image's height and width by a maximum of ``20`` pixels.

    >>> aug = iaa.Pad(px=((0, 10), (0, 5), (0, 10), (0, 5)))

    Pad the top and bottom by a random pixel value sampled uniformly from the
    discrete interval ``[0..10]``. Pad the left and right analogously by
    a random value sampled from ``[0..5]``. Each value is always sampled
    independently.

    >>> aug = iaa.Pad(percent=(0, 0.1))

    Pad each side by a random fraction sampled uniformly from the continuous
    interval ``[0.0, 0.10]``. The fraction is sampled once per image and
    side. E.g. a sampled fraction of ``0.1`` for the top side would pad by
    ``0.1*H``, where ``H`` is the height of the input image.

    >>> aug = iaa.Pad(
    >>>     percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))

    Pads each side by either ``5%`` or ``10%``. The values are sampled
    once per side and image.

    """

    def __init__(
        self,
        px: CropAndPadPxInput | None = None,
        percent: CropAndPadPercentInput | None = None,
        pad_mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        pad_cval: ParamInput = 0,
        keep_size: bool = True,
        sample_independently: bool = True,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        def recursive_validate(value: object) -> object:
            if value is None:
                return value
            if ia.is_single_number(value):
                assert value >= 0, f"Expected value >0, got {value:.4f}"
                return value
            if isinstance(value, iap.StochasticParameter):
                return value
            if isinstance(value, tuple):
                return tuple([recursive_validate(v_) for v_ in value])
            if isinstance(value, list):
                return [recursive_validate(v_) for v_ in value]
            raise Exception(
                "Expected None or int or float or StochasticParameter or "
                f"list or tuple, got {type(value)}."
            )

        if px is None and percent is None:
            percent = (0.0, 0.1)

        px = recursive_validate(px)
        percent = recursive_validate(percent)

        super().__init__(
            px=px,
            percent=percent,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            keep_size=keep_size,
            sample_independently=sample_independently,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class Crop(CropAndPad):
    """Crop images, i.e. remove columns/rows of pixels at the sides of images.

    This augmenter allows to extract smaller-sized subimages from given
    full-sized input images. The number of pixels to cut off may be defined
    in absolute values or as fractions of the image sizes.

    This augmenter will never crop images below a height or width of ``1``.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.CropAndPad`.

    Parameters
    ----------
    px : None or int or imgaug2.parameters.StochasticParameter or tuple, optional
        The number of pixels to crop on each side of the image.
        Expected value range is ``[0, inf)``.
        Either this or the parameter `percent` may be set, not both at the same
        time.

            * If ``None``, then pixel-based cropping will not be used.
            * If ``int``, then that exact number of pixels will always be
              cropped.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left), unless `sample_independently` is set to ``False``,
              as then only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``int`` s with values ``a`` and ``b``,
              then each side will be cropped by a random amount sampled
              uniformly per image and side from the inteval ``[a, b]``. If
              however `sample_independently` is set to ``False``, only one
              value will be sampled per image and used for all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``int`` (always
              crop by exactly that value), a ``tuple`` of two ``int`` s
              ``a`` and ``b`` (crop by an amount within ``[a, b]``), a
              ``list`` of ``int`` s (crop by a random value that is
              contained in the ``list``) or a ``StochasticParameter`` (sample
              the amount to crop from that parameter).

    percent : None or int or float or imgaug2.parameters.StochasticParameter or tuple, optional
        The number of pixels to crop
        on each side of the image given as a *fraction* of the image
        height/width. E.g. if this is set to ``0.1``, the augmenter will
        always crop ``10%`` of the image's height at both the top and the
        bottom (both ``10%`` each), as well as ``10%`` of the width at the
        right and left.
        Expected value range is ``[0.0, 1.0)``.
        Either this or the parameter `px` may be set, not both
        at the same time.

            * If ``None``, then fraction-based cropping will not be
              used.
            * If ``number``, then that fraction will always be cropped.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left). If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``float`` s with values ``a`` and ``b``,
              then each side will be cropped by a random fraction
              sampled uniformly per image and side from the interval
              ``[a, b]``. If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``float``
              (always crop by exactly that fraction), a ``tuple`` of
              two ``float`` s ``a`` and ``b`` (crop by a fraction from
              ``[a, b]``), a ``list`` of ``float`` s (crop by a random
              value that is contained in the list) or a ``StochasticParameter``
              (sample the percentage to crop from that parameter).

    keep_size : bool, optional
        After cropping, the result image will usually have a
        different height/width compared to the original input image. If this
        parameter is set to ``True``, then the cropped image will be
        resized to the input image's size, i.e. the augmenter's output shape
        is always identical to the input shape.

    sample_independently : bool, optional
        If ``False`` *and* the values for `px`/`percent` result in exactly
        *one* probability distribution for all image sides, only one single
        value will be sampled from that probability distribution and used for
        all sides. I.e. the crop amount then is the same for all sides.
        If ``True``, four values will be sampled independently, one per side.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.Crop(px=(0, 10))

    Crop each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``.

    >>> aug = iaa.Crop(px=(0, 10), sample_independently=False)

    Crop each side by a random pixel value sampled uniformly once per image
    from the discrete interval ``[0..10]``. Each sampled value is used
    for *all* sides of the corresponding image.

    >>> aug = iaa.Crop(px=(0, 10), keep_size=False)

    Crop each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. Afterwards, do **not**
    resize the cropped image back to the input image's size. This will decrease
    the image's height and width by a maximum of ``20`` pixels.

    >>> aug = iaa.Crop(px=((0, 10), (0, 5), (0, 10), (0, 5)))

    Crop the top and bottom by a random pixel value sampled uniformly from the
    discrete interval ``[0..10]``. Crop the left and right analogously by
    a random value sampled from ``[0..5]``. Each value is always sampled
    independently.

    >>> aug = iaa.Crop(percent=(0, 0.1))

    Crop each side by a random fraction sampled uniformly from the continuous
    interval ``[0.0, 0.10]``. The fraction is sampled once per image and
    side. E.g. a sampled fraction of ``0.1`` for the top side would crop by
    ``0.1*H``, where ``H`` is the height of the input image.

    >>> aug = iaa.Crop(
    >>>     percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))

    Crops each side by either ``5%`` or ``10%``. The values are sampled
    once per side and image.

    """

    def __init__(
        self,
        px: CropAndPadPxInput | None = None,
        percent: CropAndPadPercentInput | None = None,
        keep_size: bool = True,
        sample_independently: bool = True,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        def recursive_negate(value: object) -> object:
            if value is None:
                return value
            if ia.is_single_number(value):
                assert value >= 0, f"Expected value >0, got {value:.4f}."
                return -value
            if isinstance(value, iap.StochasticParameter):
                return iap.Multiply(value, -1)
            if isinstance(value, tuple):
                return tuple([recursive_negate(v_) for v_ in value])
            if isinstance(value, list):
                return [recursive_negate(v_) for v_ in value]
            raise Exception(
                "Expected None or int or float or StochasticParameter or "
                f"list or tuple, got {type(value)}."
            )

        if px is None and percent is None:
            percent = (0.0, 0.1)

        px = recursive_negate(px)
        percent = recursive_negate(percent)

        super().__init__(
            px=px,
            percent=percent,
            keep_size=keep_size,
            sample_independently=sample_independently,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

