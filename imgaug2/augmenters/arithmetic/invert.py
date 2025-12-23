from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, Images, Number, ParamInput, RNGInput
from imgaug2.compat.markers import legacy

from ._utils import PerChannelInput

def invert(
    image: Array,
    min_value: Number | None = None,
    max_value: Number | None = None,
    threshold: Number | None = None,
    invert_above_threshold: bool = True,
) -> Array:
    """Invert an array.

    Parameters
    ----------
    image : ndarray
        See `invert_()`.

    min_value : None or number, optional
        See `invert_()`.

    max_value : None or number, optional
        See `invert_()`.

    threshold : None or number, optional
        See `invert_()`.

    invert_above_threshold : bool, optional
        See `invert_()`.

    Returns
    -------
    ndarray
        Inverted image.

    """
    from . import invert_ as invert_fn

    return invert_fn(
        np.copy(image),
        min_value=min_value,
        max_value=max_value,
        threshold=threshold,
        invert_above_threshold=invert_above_threshold,
    )

@legacy(version="0.4.0")
def invert_(
    image: Array,
    min_value: Number | None = None,
    max_value: Number | None = None,
    threshold: Number | None = None,
    invert_above_threshold: bool = True,
) -> Array:
    """Invert an array in-place.

    By default (``min_value=None`` and ``max_value=None``), the value range is
    derived from the input dtype and most numeric dtypes are supported.

    Notes
    -----
    When using custom ``min_value`` and/or ``max_value``, dtype support is more
    restricted, as the implementation may need to clip values and/or increase
    intermediate resolution.

    In particular:

    - ``bool`` is not supported with custom min/max.
    - ``uint64`` is not supported with custom min/max due to precision loss in
      ``numpy.clip``.
    - 64-bit signed integers and 64-bit floats are not supported with custom
      min/max.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.
        The array *might* be modified in-place.

    min_value : None or number, optional
        Minimum of the value range of input images, e.g. ``0`` for ``uint8``
        images. If set to ``None``, the value will be automatically derived
        from the image's dtype.

    max_value : None or number, optional
        Maximum of the value range of input images, e.g. ``255`` for ``uint8``
        images. If set to ``None``, the value will be automatically derived
        from the image's dtype.

    threshold : None or number, optional
        A threshold to use in order to invert only numbers above or below
        the threshold. If ``None`` no thresholding will be used.

    invert_above_threshold : bool, optional
        If ``True``, only values ``>=threshold`` will be inverted.
        Otherwise, only values ``<threshold`` will be inverted.
        If `threshold` is ``None`` this parameter has no effect.

    Returns
    -------
    ndarray
        Inverted image. This *can* be the same array as input in `image`,
        modified in-place.

    """
    if image.size == 0:
        return image

    # MLX fast-path (B1): only when input is already on device.
    from imgaug2.mlx._core import is_mlx_array

    if is_mlx_array(image):
        import imgaug2.mlx as mlx

        return cast(
            Array,
            mlx.invert(
                image,
                min_value=min_value,
                max_value=max_value,
                threshold=threshold,
                invert_above_threshold=invert_above_threshold,
            ),
        )

    # when no custom min/max are chosen, all bool, uint, int and float dtypes
    # should be invertable (float tested only up to 64bit)
    # when chosing custom min/max:
    # - bool makes no sense, not allowed
    # - int and float must be increased in resolution if custom min/max values
    #   are chosen, hence they are limited to 32 bit and below
    # - uint64 is converted by numpy's clip to float64, hence loss of accuracy
    # - float16 seems to not be perfectly accurate, but still ok-ish -- was
    #   off by 10 for center value of range (float 16 min, 16), where float
    #   16 min is around -65500
    allow_dtypes_custom_minmax = iadt._convert_dtype_strs_to_types(
        "uint8 uint16 uint32 int8 int16 int32 float16 float32"
    )

    min_value_dt, _, max_value_dt = iadt.get_value_range_of_dtype(image.dtype)
    min_value = min_value_dt if min_value is None else min_value
    max_value = max_value_dt if max_value is None else max_value
    assert min_value >= min_value_dt, (
        "Expected min_value to be above or equal to dtype's min "
        f"value, got {str(min_value)} (vs. min possible {str(min_value_dt)} for {image.dtype.name})"
    )
    assert max_value <= max_value_dt, (
        "Expected max_value to be below or equal to dtype's max "
        f"value, got {str(max_value)} (vs. max possible {str(max_value_dt)} for {image.dtype.name})"
    )
    assert min_value < max_value, (
        f"Expected min_value to be below max_value, got {str(min_value)} and {str(max_value)}"
    )

    if min_value != min_value_dt or max_value != max_value_dt:
        assert image.dtype in allow_dtypes_custom_minmax, (
            "Can use custom min/max values only with the following dtypes: {}. Got: {}.".format(
                ", ".join(allow_dtypes_custom_minmax), image.dtype.name
            )
        )

    if image.dtype == iadt._UINT8_DTYPE:
        return _invert_uint8_(image, min_value, max_value, threshold, invert_above_threshold)

    dtype_kind_to_invert_func = {
        "b": _invert_bool,
        "u": _invert_uint16_or_larger_,  # uint8 handled above
        "i": _invert_int_,
        "f": _invert_float,
    }

    func = dtype_kind_to_invert_func[image.dtype.kind]

    if threshold is None:
        return func(image, min_value, max_value)

    arr_inv = func(np.copy(image), min_value, max_value)
    if invert_above_threshold:
        mask = image >= threshold
    else:
        mask = image < threshold
    image[mask] = arr_inv[mask]
    return image

def _invert_bool(arr: Array, min_value: Number, max_value: Number) -> Array:
    assert min_value == 0 and max_value == 1, (
        "min_value and max_value must be 0 and 1 for bool arrays. "
        f"Got {min_value:.4f} and {max_value:.4f}."
    )
    return ~arr

@legacy(version="0.4.0")
def _invert_uint8_(
    arr: Array,
    min_value: int,
    max_value: int,
    threshold: Number | None,
    invert_above_threshold: bool,
) -> Array:
    shape = arr.shape
    nb_channels = shape[-1] if len(shape) == 3 else 1
    valid_for_cv2 = (
        threshold is None
        and min_value == 0
        and len(shape) >= 2
        and shape[0] * shape[1] * nb_channels != 4
    )
    if valid_for_cv2:
        return _invert_uint8_subtract_(arr, max_value)
    return _invert_uint8_lut_pregenerated_(
        arr, min_value, max_value, threshold, invert_above_threshold
    )

@legacy(version="0.5.0")
def _invert_uint8_lut_pregenerated_(
    arr: Array,
    min_value: int,
    max_value: int,
    threshold: Number | None,
    invert_above_threshold: bool,
) -> Array:
    table = _InvertTablesSingleton.get_instance().get_table(
        min_value=min_value,
        max_value=max_value,
        threshold=threshold,
        invert_above_threshold=invert_above_threshold,
    )
    arr = ia.apply_lut_(arr, table)
    return arr

@legacy(version="0.5.0")
def _invert_uint8_subtract_(arr: Array, max_value: int) -> Array:
    if arr.size == 0:
        return arr

    # seems to work with arr.base.shape[0] > 1
    if arr.base is not None and arr.base.shape[0] == 1:
        arr = np.copy(arr)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    input_shape = arr.shape
    if len(input_shape) > 2 and input_shape[-1] > 1:
        arr_flat = arr.ravel()
    # This also supports a mask, which would help for thresholded invert, but
    # it seems that all non-masked components are set to zero in the output
    # array. Tackling this issue seems to rather require more time than just
    # using a LUT.
    if len(input_shape) > 2 and input_shape[-1] > 1:
        # OpenCV treats short 1D arrays (length <4) as scalars and will change
        # the output shape. Using column-vectors keeps output shapes stable.
        arr_mat = arr_flat.reshape((-1, 1))
        arr_mat = cv2.subtract(int(max_value), arr_mat, dst=arr_mat)
        return arr_mat.reshape(input_shape)

    arr = cv2.subtract(int(max_value), arr, dst=arr)
    return arr

@legacy(version="0.4.0")
def _invert_uint16_or_larger_(arr: Array, min_value: int, max_value: int) -> Array:
    min_max_is_vr = min_value == 0 and max_value == np.iinfo(arr.dtype).max
    if min_max_is_vr:
        return max_value - arr
    return _invert_by_distance(np.clip(arr, min_value, max_value), min_value, max_value)

@legacy(version="0.4.0")
def _invert_int_(arr: Array, min_value: int, max_value: int) -> Array:
    # note that for int dtypes the max value is
    #   (-1) * min_value - 1
    # e.g. -128 and 127 (min/max) for int8
    # mapping example:
    #  [-4, -3, -2, -1,  0,  1,  2,  3]
    # will be mapped to
    #  [ 3,  2,  1,  0, -1, -2, -3, -4]
    # hence we can not simply compute the inverse as:
    #  after = (-1) * before
    # but instead need
    #  after = (-1) * before - 1
    # however, this exceeds the value range for the minimum value, e.g.
    # for int8: -128 -> 128 -> 127, where 128 exceeds it. Hence, we must
    # compute the inverse via a mask (extra step for the minimum)
    # or we have to increase the resolution of the array. Here, a
    # two-step approach is used.

    if min_value == (-1) * max_value - 1:
        arr_inv = np.copy(arr)
        mask = arr_inv == min_value

        # there is probably a one-liner here to do this, but
        #  ((-1) * (arr_inv * ~mask) - 1) + mask * max_value
        # has the disadvantage of inverting min_value to max_value - 1
        # while
        #  ((-1) * (arr_inv * ~mask) - 1) + mask * (max_value+1)
        #  ((-1) * (arr_inv * ~mask) - 1) + mask * max_value + mask
        # both sometimes increase the dtype resolution (e.g. int32 to int64)
        arr_inv[mask] = max_value
        arr_inv[~mask] = (-1) * arr_inv[~mask] - 1

        return arr_inv

    return _invert_by_distance(np.clip(arr, min_value, max_value), min_value, max_value)

def _invert_float(arr: Array, min_value: float, max_value: float) -> Array:
    if np.isclose(max_value, (-1) * min_value, rtol=0):
        return (-1) * arr
    return _invert_by_distance(np.clip(arr, min_value, max_value), min_value, max_value)

def _invert_by_distance(arr: Array, min_value: Number, max_value: Number) -> Array:
    arr_inv = arr
    if arr.dtype.kind in ["i", "f"]:
        arr_inv = iadt.increase_array_resolutions_([np.copy(arr)], 2)[0]
    distance_from_min = np.abs(arr_inv - min_value)  # d=abs(v-min)
    arr_inv = max_value - distance_from_min  # v'=MAX-d
    # due to floating point inaccuracies, we might exceed the min/max
    # values for floats here, hence clip this happens especially for
    # values close to the float dtype's maxima
    if arr.dtype.kind == "f":
        arr_inv = np.clip(arr_inv, min_value, max_value)
    if arr.dtype.kind in ["i", "f"]:
        arr_inv = iadt.restore_dtypes_(arr_inv, arr.dtype, clip=False)
    return arr_inv

@legacy(version="0.4.0")
def _generate_table_for_invert_uint8(
    min_value: int, max_value: int, threshold: Number | None, invert_above_threshold: bool
) -> Array:
    table = np.arange(256).astype(np.int32)
    full_value_range = min_value == 0 and max_value == 255
    if full_value_range:
        table_inv = table[::-1]
    else:
        distance_from_min = np.abs(table - min_value)
        table_inv = max_value - distance_from_min
    table_inv = np.clip(table_inv, min_value, max_value).astype(np.uint8)

    if threshold is not None:
        table = table.astype(np.uint8)
        if invert_above_threshold:
            table_inv = np.concatenate(
                [table[0 : int(threshold)], table_inv[int(threshold) :]], axis=0
            )
        else:
            table_inv = np.concatenate(
                [table_inv[0 : int(threshold)], table[int(threshold) :]], axis=0
            )

    return table_inv

@legacy(version="0.5.0")
class _InvertTables:
    @legacy(version="0.5.0")
    def __init__(self) -> None:
        self.tables = {}

    @legacy(version="0.5.0")
    def get_table(
        self, min_value: int, max_value: int, threshold: Number | None, invert_above_threshold: bool
    ) -> Array:
        if min_value == 0 and max_value == 255:
            key = (threshold, invert_above_threshold)
            table = self.tables.get(key, None)
            if table is None:
                table = _generate_table_for_invert_uint8(
                    min_value, max_value, threshold, invert_above_threshold
                )
                self.tables[key] = table
            return table
        return _generate_table_for_invert_uint8(
            min_value, max_value, threshold, invert_above_threshold
        )

@legacy(version="0.5.0")
class _InvertTablesSingleton:
    _INSTANCE = None

    @classmethod
    @legacy(version="0.5.0")
    def get_instance(cls) -> _InvertTables:
        if cls._INSTANCE is None:
            cls._INSTANCE = _InvertTables()
        return cls._INSTANCE

@legacy(version="0.4.0")
def solarize(image: Array, threshold: Number | None = 128) -> Array:
    """Invert pixel values above a threshold.

    Parameters
    ----------
    image : ndarray
        See `solarize_()`.

    threshold : None or number, optional
        See `solarize_()`.

    Returns
    -------
    ndarray
        Inverted image.

    """
    from . import solarize_ as solarize_fn

    return solarize_fn(np.copy(image), threshold=threshold)

@legacy(version="0.4.0")
def solarize_(image: Array, threshold: Number | None = 128) -> Array:
    """Invert pixel values above a threshold in-place.

    This function is a wrapper around `invert_()`.

    This function performs the same transformation as
    `solarize()`.
    Parameters
    ----------
    image : ndarray
        See `invert_()`.

    threshold : None or number, optional
        See `invert_()`.
        Note: The default threshold is optimized for ``uint8`` images.
    Returns
    -------
    ndarray
        Inverted image. This *can* be the same array as input in `image`,
        modified in-place.

    """
    from . import invert_ as invert_fn

    return invert_fn(image, threshold=threshold)

class Invert(meta.Augmenter):
    """
    Invert all values in images, e.g. turn ``5`` into ``255-5=250``.

    For the standard value range of 0-255 it converts ``0`` to ``255``,
    ``255`` to ``0`` and ``10`` to ``(255-10)=245``.
    Let ``M`` be the maximum value possible, ``m`` the minimum value possible,
    ``v`` a value. Then the distance of ``v`` to ``m`` is ``d=abs(v-m)`` and
    the new value is given by ``v'=M-d``.

    Parameters
    ----------
    p : float or imgaug2.parameters.StochasticParameter, optional
        The probability of an image to be inverted.

    per_channel : bool or float or imgaug2.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    min_value : None or number, optional
        Minimum of the value range of input images, e.g. ``0`` for ``uint8``
        images. If set to ``None``, the value will be automatically derived
        from the image's dtype.

    max_value : None or number, optional
        Maximum of the value range of input images, e.g. ``255`` for ``uint8``
        images. If set to ``None``, the value will be automatically derived
        from the image's dtype.

    threshold : None or number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        A threshold to use in order to invert only numbers above or below
        the threshold. If ``None`` no thresholding will be used.

    invert_above_threshold : bool or float or imgaug2.parameters.StochasticParameter, optional
        If ``True``, only values ``>=threshold`` will be inverted.
        Otherwise, only values ``<threshold`` will be inverted.
        If a ``number``, then expected to be in the interval ``[0.0, 1.0]`` and
        denoting an imagewise probability. If a ``StochasticParameter`` then
        ``(N,)`` values will be sampled from the parameter per batch of size
        ``N`` and interpreted as ``True`` if ``>0.5``.
        If `threshold` is ``None`` this parameter has no effect.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See `__init__()`.

    name : None or str, optional
        See `__init__()`.

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
    >>> aug = iaa.Invert(0.1)

    Inverts the colors in ``10`` percent of all images.

    >>> aug = iaa.Invert(0.1, per_channel=True)

    Inverts the colors in ``10`` percent of all image channels. This may or
    may not lead to multiple channels in an image being inverted.

    >>> aug = iaa.Invert(0.1, per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    # when no custom min/max are chosen, all bool, uint, int and float dtypes
    # should be invertable (float tested only up to 64bit)
    # when chosing custom min/max:
    # - bool makes no sense, not allowed
    # - int and float must be increased in resolution if custom min/max values
    #   are chosen, hence they are limited to 32 bit and below
    # - uint64 is converted by numpy's clip to float64, hence loss of accuracy
    # - float16 seems to not be perfectly accurate, but still ok-ish -- was
    #   off by 10 for center value of range (float 16 min, 16), where float
    #   16 min is around -65500
    ALLOW_DTYPES_CUSTOM_MINMAX = [
        np.dtype(dt)
        for dt in [
            np.uint8,
            np.uint16,
            np.uint32,
            np.int8,
            np.int16,
            np.int32,
            np.float16,
            np.float32,
        ]
    ]

    def __init__(
        self,
        p: PerChannelInput = 1,
        per_channel: PerChannelInput = False,
        min_value: Number | None = None,
        max_value: Number | None = None,
        threshold: ParamInput | None = None,
        invert_above_threshold: PerChannelInput = 0.5,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        # TODO allow list and tuple for p
        self.p = iap.handle_probability_param(p, "p")
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")
        self.min_value = min_value
        self.max_value = max_value

        if threshold is None:
            self.threshold = None
        else:
            self.threshold = iap.handle_continuous_param(
                threshold, "threshold", value_range=None, tuple_to_uniform=True, list_to_choice=True
            )
        self.invert_above_threshold = iap.handle_probability_param(
            invert_above_threshold, "invert_above_threshold"
        )

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        if batch.images is None:
            return batch

        samples = self._draw_samples(batch, random_state)

        for i, image in enumerate(batch.images):
            if 0 in image.shape:
                continue

            kwargs = {
                "min_value": samples.min_value[i],
                "max_value": samples.max_value[i],
                "threshold": samples.threshold[i],
                "invert_above_threshold": samples.invert_above_threshold[i],
            }

            if samples.per_channel[i]:
                nb_channels = image.shape[2]
                mask = samples.p[i, :nb_channels]
                image[..., mask] = invert_(image[..., mask], **kwargs)
            else:
                if samples.p[i, 0]:
                    image[:, :, :] = invert_(image, **kwargs)

        return batch

    @legacy(version="0.4.0")
    def _draw_samples(
        self, batch: _BatchInAugmentation, random_state: iarandom.RNG
    ) -> _InvertSamples:
        nb_images = batch.nb_rows
        nb_channels = meta.estimate_max_number_of_channels(batch.images)
        p = self.p.draw_samples((nb_images, nb_channels), random_state=random_state)
        p = p > 0.5
        per_channel = self.per_channel.draw_samples((nb_images,), random_state=random_state)
        per_channel = per_channel > 0.5
        min_value = [self.min_value] * nb_images
        max_value = [self.max_value] * nb_images

        if self.threshold is None:
            threshold = [None] * nb_images
        else:
            threshold = self.threshold.draw_samples((nb_images,), random_state=random_state)

        invert_above_threshold = self.invert_above_threshold.draw_samples(
            (nb_images,), random_state=random_state
        )
        invert_above_threshold = invert_above_threshold > 0.5

        return _InvertSamples(
            p=p,
            per_channel=per_channel,
            min_value=min_value,
            max_value=max_value,
            threshold=threshold,
            invert_above_threshold=invert_above_threshold,
        )

    def get_parameters(self) -> list[object]:
        """See `get_parameters()`."""
        return [
            self.p,
            self.per_channel,
            self.min_value,
            self.max_value,
            self.threshold,
            self.invert_above_threshold,
        ]

@legacy(version="0.4.0")
class _InvertSamples:
    @legacy(version="0.4.0")
    def __init__(
        self,
        p: Array,
        per_channel: Array,
        min_value: list[Number | None],
        max_value: list[Number | None],
        threshold: list[Number | None] | Array,
        invert_above_threshold: Array,
    ) -> None:
        self.p = p
        self.per_channel = per_channel
        self.min_value = min_value
        self.max_value = max_value
        self.threshold = threshold
        self.invert_above_threshold = invert_above_threshold

@legacy(version="0.4.0")
class Solarize(Invert):
    """Invert all pixel values above a threshold.

    This is the same as `Invert`, but sets a default threshold around
    ``128`` (+/- 64, decided per image) and default `invert_above_threshold`
    to ``True`` (i.e. only values above the threshold will be inverted).

    See `Invert` for more details.

    Parameters
    ----------
    p : float or imgaug2.parameters.StochasticParameter
        See `Invert`.

    per_channel : bool or float or imgaug2.parameters.StochasticParameter, optional
        See `Invert`.

    min_value : None or number, optional
        See `Invert`.

    max_value : None or number, optional
        See `Invert`.

    threshold : None or number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See `Invert`.

    invert_above_threshold : bool or float or imgaug2.parameters.StochasticParameter, optional
        See `Invert`.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See `__init__()`.

    name : None or str, optional
        See `__init__()`.

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
    >>> aug = iaa.Solarize(0.5, threshold=(32, 128))

    Invert the colors in ``50`` percent of all images for pixels with a
    value between ``32`` and ``128`` or more. The threshold is sampled once
    per image. The thresholding operation happens per channel.

    """

    def __init__(
        self,
        p: PerChannelInput = 1,
        per_channel: PerChannelInput = False,
        min_value: Number | None = None,
        max_value: Number | None = None,
        threshold: ParamInput | None = (128 - 64, 128 + 64),
        invert_above_threshold: PerChannelInput = True,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            p=p,
            per_channel=per_channel,
            min_value=min_value,
            max_value=max_value,
            threshold=threshold,
            invert_above_threshold=invert_above_threshold,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

# TODO try adding per channel somehow
