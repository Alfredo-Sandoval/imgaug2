from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.compat.markers import legacy
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.imgaug import _normalize_cv2_input_arr_
from ._utils import PerChannelInput, ScalarInput


def multiply_scalar(image: Array, multiplier: ScalarInput) -> Array:
    """Multiply an image by a single scalar or one scalar per channel.

    This method ensures that ``uint8`` does not overflow during the
    multiplication.

    note::

        Tests were only conducted for rather small multipliers, around
        ``-10.0`` to ``+10.0``.

        In general, the multipliers sampled from `multiplier` must be in a
        value range that corresponds to the input image's dtype. E.g. if the
        input image has dtype ``uint16`` and the samples generated from
        `multiplier` are ``float64``, this function will still force all
        samples to be within the value range of ``float16``, as it has the
        same number of bytes (two) as ``uint16``. This is done to make
        overflows less likely to occur.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: limited; tested (1)
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: limited; tested (1)
        * ``int16``: limited; tested (1)
        * ``int32``: no
        * ``int64``: no
        * ``float16``: limited; tested (1)
        * ``float32``: limited; tested (1)
        * ``float64``: no
        * ``float128``: no
        * ``bool``: limited; tested (1)

        - (1) Non-uint8 dtypes can overflow. For floats, this can result in
              +/-inf.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.
        If `value` contains more than one value, the shape of the image is
        expected to be ``(H,W,C)``.

    multiplier : number or ndarray
        The multiplier to use. Either a single value or an array
        containing exactly one component per channel, i.e. ``C`` components.

    Returns
    -------
    ndarray
        Image, multiplied by `multiplier`.

    """
    return multiply_scalar_(np.copy(image), multiplier)


@legacy(version="0.5.0")
def multiply_scalar_(image: Array, multiplier: ScalarInput) -> Array:
    """Multiply in-place an image by a single scalar or one scalar per channel.

    This method ensures that ``uint8`` does not overflow during the
    multiplication.

    note::

        Tests were only conducted for rather small multipliers, around
        ``-10.0`` to ``+10.0``.

        In general, the multipliers sampled from `multiplier` must be in a
        value range that corresponds to the input image's dtype. E.g. if the
        input image has dtype ``uint16`` and the samples generated from
        `multiplier` are ``float64``, this function will still force all
        samples to be within the value range of ``float16``, as it has the
        same number of bytes (two) as ``uint16``. This is done to make
        overflows less likely to occur.


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: limited; tested (1)
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: limited; tested (1)
        * ``int16``: limited; tested (1)
        * ``int32``: no
        * ``int64``: no
        * ``float16``: limited; tested (1)
        * ``float32``: limited; tested (1)
        * ``float64``: no
        * ``float128``: no
        * ``bool``: limited; tested (1)

        - (1) Non-uint8 dtypes can overflow. For floats, this can result in
              +/-inf.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.
        If `value` contains more than one value, the shape of the image is
        expected to be ``(H,W,C)``.
        May be changed in-place.

    multiplier : number or ndarray
        The multiplier to use. Either a single value or an array
        containing exactly one component per channel, i.e. ``C`` components.

    Returns
    -------
    ndarray
        Image, multiplied by `multiplier`.
        Might be the same image instance as was provided in `image`.

    """
    # MLX fast-path (B1): only when input is already on device.
    from imgaug2.mlx._core import is_mlx_array

    if is_mlx_array(image):
        import imgaug2.mlx as mlx

        return cast(Array, mlx.multiply(image, multiplier))

    size = image.size
    if size == 0:
        return image

    iadt.gate_dtypes_strs(
        {image.dtype},
        allowed="bool uint8 uint16 int8 int16 float16 float32",
        disallowed="uint32 uint64 int32 int64 float64 float128",
        augmenter=None,
    )

    if image.dtype == iadt._UINT8_DTYPE:
        if size >= 224 * 224 * 3:
            return _multiply_scalar_to_uint8_lut_(image, multiplier)
        return _multiply_scalar_to_uint8_cv2_mul_(image, multiplier)
    return _multiply_scalar_to_non_uint8(image, multiplier)


# TODO add a c++/cython method here to compute the LUT tables
@legacy(version="0.5.0")
def _multiply_scalar_to_uint8_lut_(image: Array, multiplier: ScalarInput) -> Array:
    is_single_value = (
        ia.is_single_number(multiplier)
        or ia.is_np_scalar(multiplier)
        or (ia.is_np_array(multiplier) and multiplier.size == 1)
    )
    is_channelwise = not is_single_value
    nb_channels = 1 if image.ndim == 2 else image.shape[-1]

    multiplier = np.float32(multiplier)
    value_range = np.arange(0, 256, dtype=np.float32)

    if is_channelwise:
        assert multiplier.ndim == 1, (
            f"Expected `multiplier` to be 1-dimensional, got {multiplier.ndim}-dimensional "
            f"data with shape {multiplier.shape}."
        )
        assert image.ndim == 3, (
            "Expected `image` to be 3-dimensional when multiplying by one "
            f"value per channel, got {image.ndim}-dimensional data with shape {image.shape}."
        )
        assert image.shape[-1] == multiplier.size, (
            "Expected number of channels in `image` and number of components "
            f"in `multiplier` to be identical. Got {image.shape[-1]} vs. {multiplier.size}."
        )

        value_range = np.broadcast_to(value_range[:, np.newaxis], (256, nb_channels))
        value_range = value_range * multiplier[np.newaxis, :]
    else:
        value_range = value_range * multiplier
    value_range = np.clip(value_range, 0, 255).astype(image.dtype)
    return ia.apply_lut_(image, value_range)


@legacy(version="0.5.0")
def _multiply_scalar_to_uint8_cv2_mul_(image: Array, multiplier: ScalarInput) -> Array:
    # multiplier must already be an array_like
    if multiplier.size > 1:
        multiplier = multiplier[np.newaxis, np.newaxis, :]
        multiplier = np.broadcast_to(multiplier, image.shape)
    else:
        multiplier = np.full(image.shape, multiplier, dtype=np.float32)

    image = _normalize_cv2_input_arr_(image)
    result = cv2.multiply(image, multiplier, dtype=cv2.CV_8U, dst=image)

    return result


def _multiply_scalar_to_non_uint8(image: Array, multiplier: ScalarInput) -> Array:
    # TODO estimate via image min/max values whether a resolution
    #      increase is necessary
    input_dtype = image.dtype

    is_single_value = (
        ia.is_single_number(multiplier)
        or ia.is_np_scalar(multiplier)
        or (ia.is_np_array(multiplier) and multiplier.size == 1)
    )
    is_channelwise = not is_single_value
    nb_channels = 1 if image.ndim == 2 else image.shape[-1]

    shape = (1, 1, nb_channels if is_channelwise else 1)
    multiplier = np.array(multiplier).reshape(shape)

    # deactivated itemsize increase due to clip causing problems
    # with int64, see Add
    # mul_min = np.min(mul)
    # mul_max = np.max(mul)
    # is_not_increasing_value_range = (
    #         (-1 <= mul_min <= 1)
    #         and (-1 <= mul_max <= 1))

    # We limit here the value range of the mul parameter to the
    # bytes in the image's dtype. This prevents overflow problems
    # and makes it less likely that the image has to be up-casted,
    # which again improves performance and saves memory. Note that
    # this also enables more dtypes for image inputs.
    # The downside is that the mul parameter is limited in its
    # value range.
    itemsize = max(
        image.dtype.itemsize, 2 if multiplier.dtype.kind == "f" else 1
    )  # float min itemsize is 2 not 1
    dtype_target = np.dtype(f"{multiplier.dtype.kind}{itemsize}")
    multiplier = iadt.clip_to_dtype_value_range_(multiplier, dtype_target, validate=True)

    image, multiplier = iadt.promote_arrays_to_minimal_dtype_(
        [image, multiplier],
        dtypes=[image.dtype, dtype_target],
        # increase_itemsize_factor=(
        #     1 if is_not_increasing_value_range else 2)
        increase_itemsize_factor=1,
    )
    image = np.multiply(image, multiplier, out=image, casting="no")

    return iadt.restore_dtypes_(image, input_dtype)


def multiply_elementwise(image: Array, multipliers: Array) -> Array:
    """Multiply an image with an array of values.

    This method ensures that ``uint8`` does not overflow during the addition.

    note::

        Tests were only conducted for rather small multipliers, around
        ``-10.0`` to ``+10.0``.

        In general, the multipliers sampled from `multipliers` must be in a
        value range that corresponds to the input image's dtype. E.g. if the
        input image has dtype ``uint16`` and the samples generated from
        `multipliers` are ``float64``, this function will still force all
        samples to be within the value range of ``float16``, as it has the
        same number of bytes (two) as ``uint16``. This is done to make
        overflows less likely to occur.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: limited; tested (1)
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: limited; tested (1)
        * ``int16``: limited; tested (1)
        * ``int32``: no
        * ``int64``: no
        * ``float16``: limited; tested (1)
        * ``float32``: limited; tested (1)
        * ``float64``: no
        * ``float128``: no
        * ``bool``: limited; tested (1)

        - (1) Non-uint8 dtypes can overflow. For floats, this can result
              in +/-inf.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.

    multipliers : ndarray
        The multipliers with which to multiply the image. Expected to have
        the same height and width as `image` and either no channels or one
        channel or the same number of channels as `image`.

    Returns
    -------
    ndarray
        Image, multiplied by `multipliers`.

    """
    return multiply_elementwise_(np.copy(image), multipliers)


@legacy(version="0.5.0")
def multiply_elementwise_(image: Array, multipliers: Array) -> Array:
    """Multiply in-place an image with an array of values.

    This method ensures that ``uint8`` does not overflow during the addition.

    note::

        Tests were only conducted for rather small multipliers, around
        ``-10.0`` to ``+10.0``.

        In general, the multipliers sampled from `multipliers` must be in a
        value range that corresponds to the input image's dtype. E.g. if the
        input image has dtype ``uint16`` and the samples generated from
        `multipliers` are ``float64``, this function will still force all
        samples to be within the value range of ``float16``, as it has the
        same number of bytes (two) as ``uint16``. This is done to make
        overflows less likely to occur.


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: limited; tested (1)
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: limited; tested (1)
        * ``int16``: limited; tested (1)
        * ``int32``: no
        * ``int64``: no
        * ``float16``: limited; tested (1)
        * ``float32``: limited; tested (1)
        * ``float64``: no
        * ``float128``: no
        * ``bool``: limited; tested (1)

        - (1) Non-uint8 dtypes can overflow. For floats, this can result
              in +/-inf.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.

    multipliers : ndarray
        The multipliers with which to multiply the image. Expected to have
        the same height and width as `image` and either no channels or one
        channel or the same number of channels as `image`.

    Returns
    -------
    ndarray
        Image, multiplied by `multipliers`.

    """
    iadt.gate_dtypes_strs(
        {image.dtype},
        allowed="bool uint8 uint16 int8 int16 float16 float32",
        disallowed="uint32 uint64 int32 int64 float64 float128",
        augmenter=None,
    )

    if 0 in image.shape:
        return image

    if multipliers.dtype.kind == "b":
        # TODO extend this with some shape checks
        image *= multipliers
        return image
    if image.dtype == iadt._UINT8_DTYPE:
        return _multiply_elementwise_to_uint8_(image, multipliers)
    return _multiply_elementwise_to_non_uint8(image, multipliers)


@legacy(version="0.5.0")
def _multiply_elementwise_to_uint8_(image: Array, multipliers: Array) -> Array:
    dt = multipliers.dtype
    kind = dt.kind
    if kind == "f" and dt != iadt._FLOAT32_DTYPE:
        multipliers = multipliers.astype(np.float32)
    elif kind == "i" and dt != iadt._INT32_DTYPE:
        multipliers = multipliers.astype(np.int32)
    elif kind == "u" and dt != iadt._UINT8_DTYPE:
        multipliers = multipliers.astype(np.uint8)

    if multipliers.ndim < image.ndim:
        multipliers = multipliers[:, :, np.newaxis]
    if multipliers.shape != image.shape:
        multipliers = np.broadcast_to(multipliers, image.shape)

    assert image.shape == multipliers.shape, (
        "Expected multipliers to have shape (H,W) or (H,W,1) or (H,W,C) "
        "(H = image height, W = image width, C = image channels). Reached "
        f"shape {multipliers.shape} after broadcasting, compared to image shape {image.shape}."
    )

    # views seem to be fine here
    if image.flags["C_CONTIGUOUS"] is False:
        image = np.ascontiguousarray(image)

    result = cv2.multiply(image, multipliers, dst=image, dtype=cv2.CV_8U)

    return result


def _multiply_elementwise_to_non_uint8(image: Array, multipliers: Array) -> Array:
    input_dtype = image.dtype

    # TODO maybe introduce to stochastic parameters some way to
    #      get the possible min/max values, could make things
    #      faster for dropout to get 0/1 min/max from the binomial
    # itemsize decrease is currently deactivated due to issues
    # with clip and int64, see Add
    mul_min = np.min(multipliers)
    mul_max = np.max(multipliers)
    # is_not_increasing_value_range = (
    #     (-1 <= mul_min <= 1) and (-1 <= mul_max <= 1))

    # We limit here the value range of the mul parameter to the
    # bytes in the image's dtype. This prevents overflow problems
    # and makes it less likely that the image has to be up-casted,
    # which again improves performance and saves memory. Note that
    # this also enables more dtypes for image inputs.
    # The downside is that the mul parameter is limited in its
    # value range.
    itemsize = max(
        image.dtype.itemsize, 2 if multipliers.dtype.kind == "f" else 1
    )  # float min itemsize is 2
    dtype_target = np.dtype(f"{multipliers.dtype.kind}{itemsize}")
    multipliers = iadt.clip_to_dtype_value_range_(
        multipliers, dtype_target, validate=True, validate_values=(mul_min, mul_max)
    )

    if multipliers.shape[2] == 1:
        # TODO check if tile() is here actually needed
        nb_channels = image.shape[-1]
        multipliers = np.tile(multipliers, (1, 1, nb_channels))

    image, multipliers = iadt.promote_arrays_to_minimal_dtype_(
        [image, multipliers],
        dtypes=[image, dtype_target],
        increase_itemsize_factor=1,
        # increase_itemsize_factor=(
        #     1 if is_not_increasing_value_range else 2)
    )
    image = np.multiply(image, multipliers, out=image, casting="no")
    return iadt.restore_dtypes_(image, input_dtype)


class Multiply(meta.Augmenter):
    """
    Multiply all pixels in an image with a random value sampled once per image.

    This augmenter can be used to make images lighter or darker.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.arithmetic.multiply_scalar`.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        The value with which to multiply the pixel values in each image.

            * If a number, then that value will always be used.
            * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
              will be sampled per image and used for all pixels.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, then that parameter will be used to
              sample a new value per image.

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
    >>> aug = iaa.Multiply(2.0)

    Multiplies all images by a factor of ``2``, making the images significantly
    brighter.

    >>> aug = iaa.Multiply((0.5, 1.5))

    Multiplies images by a random value sampled uniformly from the interval
    ``[0.5, 1.5]``, making some images darker and others brighter.

    >>> aug = iaa.Multiply((0.5, 1.5), per_channel=True)

    Identical to the previous example, but the sampled multipliers differ by
    image *and* channel, instead of only by image.

    >>> aug = iaa.Multiply((0.5, 1.5), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(
        self,
        mul: ParamInput = (0.8, 1.2),
        per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.mul = iap.handle_continuous_param(
            mul, "mul", value_range=None, tuple_to_uniform=True, list_to_choice=True
        )
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

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

        images = batch.images
        nb_images = len(images)
        nb_channels_max = meta.estimate_max_number_of_channels(images)
        rss = random_state.duplicate(2)
        per_channel_samples = self.per_channel.draw_samples((nb_images,), random_state=rss[0])
        mul_samples = self.mul.draw_samples((nb_images, nb_channels_max), random_state=rss[1])

        gen = enumerate(zip(images, mul_samples, per_channel_samples, strict=True))
        for i, (image, mul_samples_i, per_channel_samples_i) in gen:
            nb_channels = image.shape[2]

            # Example code to directly multiply images via image*sample
            # (uint8 only) -- apparently slower than LUT
            # if per_channel_samples_i > 0.5:
            #     result = []
            #     image = image.astype(np.float32)
            #     mul_samples_i = mul_samples_i.astype(np.float32)
            #     for c, mul in enumerate(mul_samples_i[0:nb_channels]):
            #         result.append(
            #             np.clip(
            #                 image[..., c:c+1] * mul, 0, 255
            #             ).astype(np.uint8))
            #     images[i] = np.concatenate(result, axis=2)
            # else:
            #     images[i] = np.clip(
            #         image.astype(np.float32)
            #         * mul_samples_i[0].astype(np.float32),
            #         0, 255
            #     ).astype(np.uint8)

            if per_channel_samples_i > 0.5:
                mul = mul_samples_i[0:nb_channels]
            else:
                # the if/else here catches the case of the channel axis being 0
                mul = mul_samples_i[0] if mul_samples_i.size > 0 else []
            batch.images[i] = multiply_scalar_(image, mul)

        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.mul, self.per_channel]


# TODO merge with Multiply
class MultiplyElementwise(meta.Augmenter):
    """
    Multiply image pixels with values that are pixelwise randomly sampled.

    While the ``Multiply`` Augmenter uses a constant multiplier *per
    image* (and optionally channel), this augmenter samples the multipliers
    to use per image and *per pixel* (and optionally per channel).

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.arithmetic.multiply_elementwise`.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        The value with which to multiply pixel values in the image.

            * If a number, then that value will always be used.
            * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
              will be sampled per image and pixel.
            * If a list, then a random value will be sampled from that list
              per image and pixel.
            * If a ``StochasticParameter``, then that parameter will be used to
              sample a new value per image and pixel.

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
    >>> aug = iaa.MultiplyElementwise(2.0)

    Multiply all images by a factor of ``2.0``, making them significantly
    bighter.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5))

    Samples per image and pixel uniformly a value from the interval
    ``[0.5, 1.5]`` and multiplies the pixel with that value.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=True)

    Samples per image and pixel *and channel* uniformly a value from the
    interval ``[0.5, 1.5]`` and multiplies the pixel with that value. Therefore,
    used multipliers may differ between channels of the same pixel.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(
        self,
        mul: ParamInput = (0.8, 1.2),
        per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.mul = iap.handle_continuous_param(
            mul, "mul", value_range=None, tuple_to_uniform=True, list_to_choice=True
        )
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

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

        images = batch.images
        nb_images = len(images)
        rss = random_state.duplicate(1 + nb_images)
        per_channel_samples = self.per_channel.draw_samples((nb_images,), random_state=rss[0])
        is_mul_binomial = isinstance(self.mul, iap.Binomial) or (
            isinstance(self.mul, iap.FromLowerResolution)
            and isinstance(self.mul.other_param, iap.Binomial)
        )

        gen = enumerate(zip(images, per_channel_samples, rss[1:], strict=True))
        for i, (image, per_channel_samples_i, rs) in gen:
            height, width, nb_channels = image.shape
            sample_shape = (height, width, nb_channels if per_channel_samples_i > 0.5 else 1)
            mul = self.mul.draw_samples(sample_shape, random_state=rs)
            # TODO let Binomial return boolean mask directly instead of [0, 1]
            #      integers?

            # hack to improve performance for Dropout and CoarseDropout
            # converts mul samples to mask if mul is binomial
            if mul.dtype.kind != "b" and is_mul_binomial:
                mul = mul.astype(bool, copy=False)

            batch.images[i] = multiply_elementwise_(image, mul)

        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.mul, self.per_channel]
