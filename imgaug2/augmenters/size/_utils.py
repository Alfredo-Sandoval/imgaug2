"""Shared helpers and type aliases for size augmenters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import cv2
import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
from imgaug2.augmenters._typing import Array
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _assert_two_or_three_dims, _normalize_cv2_input_arr_
Shape: TypeAlias = tuple[int, ...]
Shape2D: TypeAlias = tuple[int, int]
TRBL: TypeAlias = tuple[int, int, int, int]
XYXY: TypeAlias = tuple[int, int, int, int]

ResizeSizeScalar: TypeAlias = float | int


def _mlx_resize_order_from_interpolation(interpolation: object) -> int | None:
    if interpolation is None:
        return 1
    if ia.is_string(interpolation):
        interp_str = str(interpolation).lower()
        if interp_str == "nearest":
            return 0
        if interp_str == "linear":
            return 1
        return None
    if ia.is_single_integer(interpolation):
        interp_int = int(interpolation)
        if interp_int == cv2.INTER_NEAREST:
            return 0
        if interp_int == cv2.INTER_LINEAR:
            return 1
        return None
    return None
ResizeSizeTuple: TypeAlias = tuple[ResizeSizeScalar, ResizeSizeScalar]
ResizeSizeList: TypeAlias = list[ResizeSizeScalar]
ResizeSizeDictValue: TypeAlias = (
    Literal["keep", "keep-aspect-ratio"] | ResizeSizeScalar | ResizeSizeTuple | ResizeSizeList
)
ResizeSizeDict: TypeAlias = dict[str, ResizeSizeDictValue]
ResizeSizeParam: TypeAlias = (
    iap.StochasticParameter | tuple[iap.StochasticParameter, iap.StochasticParameter]
)
ResizeSizeInput: TypeAlias = (
    Literal["keep"]
    | ResizeSizeScalar
    | ResizeSizeTuple
    | ResizeSizeList
    | iap.StochasticParameter
    | ResizeSizeDict
)
ResizeInterpolationInput: TypeAlias = (
    Literal["ALL"] | int | str | Sequence[int | str] | iap.StochasticParameter
)
ResizeSamplingResult: TypeAlias = tuple[Array, Array, Array]

CropAndPadPxSingleParam: TypeAlias = int | tuple[int, int] | list[int] | iap.StochasticParameter
CropAndPadPxInput: TypeAlias = (
    int
    | tuple[int, int]
    | tuple[
        CropAndPadPxSingleParam,
        CropAndPadPxSingleParam,
        CropAndPadPxSingleParam,
        CropAndPadPxSingleParam,
    ]
    | iap.StochasticParameter
)
CropAndPadPercentSingleParam: TypeAlias = (
    float | int | tuple[float | int, float | int] | list[float | int] | iap.StochasticParameter
)
CropAndPadPercentInput: TypeAlias = (
    float
    | int
    | tuple[float | int, float | int]
    | tuple[
        CropAndPadPercentSingleParam,
        CropAndPadPercentSingleParam,
        CropAndPadPercentSingleParam,
        CropAndPadPercentSingleParam,
    ]
    | iap.StochasticParameter
)
CropAndPadMode: TypeAlias = Literal["noop", "px", "percent"]
CropAndPadParamReturn: TypeAlias = tuple[
    CropAndPadMode,
    iap.StochasticParameter | None,
    iap.StochasticParameter | None,
    iap.StochasticParameter | None,
    iap.StochasticParameter | None,
    iap.StochasticParameter | None,
]

MinSizeWH: TypeAlias = tuple[int | None, int | None]
PadToFixedSizeSamplingResult: TypeAlias = tuple[list[MinSizeWH], Array, Array, Array, Array]
CropToFixedSizeSamplingResult: TypeAlias = tuple[list[MinSizeWH], Array, Array]

KeepSizeByResizeInterpolationInput: TypeAlias = (
    Literal["NO_RESIZE"] | str | int | list[str | int] | iap.StochasticParameter
)
KeepSizeByResizeInterpolationMapsInput: TypeAlias = (
    KeepSizeByResizeInterpolationInput | Literal["SAME_AS_IMAGES"]
)
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
