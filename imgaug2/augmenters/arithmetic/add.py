from __future__ import annotations

from typing import Literal, cast

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
from ._utils import PerChannelInput, ScalarInput


def add_scalar(image: Array, value: ScalarInput) -> Array:
    """Add a scalar value (or one scalar per channel) to an image.

    This method ensures that ``uint8`` does not overflow during the addition.

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
        If `value` contains more than one value, the shape of the image is
        expected to be ``(H,W,C)``.

    value : number or ndarray
        The value to add to the image. Either a single value or an array
        containing exactly one component per channel, i.e. ``C`` components.

    Returns
    -------
    ndarray
        Image with value added to it.

    """
    return add_scalar_(np.copy(image), value)


def add_scalar_(image: Array, value: ScalarInput) -> Array:
    """Add in-place a scalar value (or one scalar per channel) to an image.

    This method ensures that ``uint8`` does not overflow during the addition.

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
        If `value` contains more than one value, the shape of the image is
        expected to be ``(H,W,C)``.
        The image might be changed in-place.

    value : number or ndarray
        The value to add to the image. Either a single value or an array
        containing exactly one component per channel, i.e. ``C`` components.

    Returns
    -------
    ndarray
        Image with value added to it.
        This might be the input `image`, changed in-place.

    """
    # MLX fast-path (B1): only when input is already on device.
    from imgaug2.mlx._core import is_mlx_array

    if is_mlx_array(image):
        import imgaug2.mlx as mlx

        return cast(Array, mlx.add(image, value))

    if image.size == 0:
        return np.copy(image)

    iadt.gate_dtypes_strs(
        {image.dtype},
        allowed="bool uint8 uint16 int8 int16 float16 float32",
        disallowed="uint32 uint64 int32 int64 float64 float128",
        augmenter=None,
    )

    if image.dtype == iadt._UINT8_DTYPE:
        return _add_scalar_to_uint8_(image, value)
    return _add_scalar_to_non_uint8(image, value)


def _add_scalar_to_uint8_(image: Array, value: ScalarInput) -> Array:
    if ia.is_single_number(value):
        is_single_value = True
        value = round(value)
    elif ia.is_np_scalar(value) or ia.is_np_array(value):
        is_single_value = value.size == 1
        value = np.round(value) if value.dtype.kind == "f" else value
    else:
        is_single_value = False
    is_channelwise = not is_single_value

    if image.ndim == 2 and is_single_value:
        return cv2.add(image, value, dst=image, dtype=cv2.CV_8U)

    input_shape = image.shape
    image_flat = image.ravel()
    values = np.array(value)
    if not is_channelwise:
        values = np.broadcast_to(values, image_flat.shape)
    else:
        values = np.tile(values, image_flat.size // len(values))

    # OpenCV treats short 1D arrays (length <4) as scalars, which would change
    # the output shape and break reshaping back to the input image shape.
    # Converting to column-vectors ensures consistent behavior for all sizes.
    image_mat = image_flat.reshape((-1, 1))
    values_mat = np.asarray(values).reshape((-1, 1))
    image_add = cv2.add(image_mat, values_mat, dst=image_mat, dtype=cv2.CV_8U)

    return image_add.reshape(input_shape)


def _add_scalar_to_non_uint8(image: Array, value: ScalarInput) -> Array:
    input_dtype = image.dtype

    is_single_value = (
        ia.is_single_number(value)
        or ia.is_np_scalar(value)
        or (ia.is_np_array(value) and value.size == 1)
    )
    is_channelwise = not is_single_value
    nb_channels = 1 if image.ndim == 2 else image.shape[-1]

    shape = (1, 1, nb_channels if is_channelwise else 1)
    value = np.array(value).reshape(shape)

    # We limit here the value range of the value parameter to the
    # bytes in the image's dtype. This prevents overflow problems
    # and makes it less likely that the image has to be up-casted,
    # which again improves performance and saves memory. Note that
    # this also enables more dtypes for image inputs.
    # The downside is that the mul parameter is limited in its
    # value range.
    #
    # We need 2* the itemsize of the image here to allow to shift
    # the image's max value to the lowest possible value, e.g. for
    # uint8 it must allow for -255 to 255.
    itemsize = image.dtype.itemsize * 2
    dtype_target = np.dtype(f"{value.dtype.kind}{itemsize}")
    value = iadt.clip_to_dtype_value_range_(value, dtype_target, validate=True)

    # Itemsize is currently reduced from 2 to 1 due to clip no
    # longer supporting int64, which can cause issues with int32
    # samples (32*2 = 64bit).
    # TODO limit value ranges of samples to int16/uint16 for
    #      security
    image, value = iadt.promote_arrays_to_minimal_dtype_(
        [image, value], dtypes=[image.dtype, dtype_target], increase_itemsize_factor=1
    )
    image = np.add(image, value, out=image, casting="no")

    return iadt.restore_dtypes_(image, input_dtype)


def add_elementwise(image: Array, values: Array) -> Array:
    """Add an array of values to an image.

    This method ensures that ``uint8`` does not overflow during the addition.

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

    values : ndarray
        The values to add to the image. Expected to have the same height
        and width as `image` and either no channels or one channel or
        the same number of channels as `image`.
        This array is expected to have dtype ``int8``, ``int16``, ``int32``,
        ``uint8``, ``uint16``, ``float32``, ``float64``. Other dtypes may
        or may not work.
        For ``uint8`` inputs, only `value` arrays with values in the interval
        ``[-1000, 1000]`` are supported. Values beyond that interval may
        result in an output array of zeros (no error is raised due to
        performance reasons).

    Returns
    -------
    ndarray
        Image with values added to it.

    """
    iadt.gate_dtypes_strs(
        {image.dtype},
        allowed="bool uint8 uint16 int8 int16 float16 float32",
        disallowed="uint32 uint64 int32 int64 float64 float128",
        augmenter=None,
    )

    if image.dtype == iadt._UINT8_DTYPE:
        vdt = values.dtype
        valid_value_dtypes_cv2 = iadt._convert_dtype_strs_to_types(
            "int8 int16 int32 uint8 uint16 float32 float64"
        )

        ishape = image.shape

        is_image_valid_shape_cv2 = (
            len(ishape) == 2 or (len(ishape) == 3 and ishape[-1] <= 512)
        ) and 0 not in ishape

        use_cv2 = is_image_valid_shape_cv2 and vdt in valid_value_dtypes_cv2
        if use_cv2:
            return _add_elementwise_cv2_to_uint8(image, values)
        return _add_elementwise_np_to_uint8(image, values)
    return _add_elementwise_np_to_non_uint8(image, values)


def _add_elementwise_cv2_to_uint8(image: Array, values: Array) -> Array:
    ind, vnd = image.ndim, values.ndim
    valid_vnd = [ind] if ind == 2 else [ind - 1, ind]
    assert vnd in valid_vnd, (
        f"Expected values with any of {valid_vnd} dimensions, "
        f"got {vnd} dimensions (shape {values.shape} vs. image shape {image.shape})."
    )

    if vnd == ind - 1:
        values = values[:, :, np.newaxis]
    if values.shape[-1] == 1:
        values = np.broadcast_to(values, image.shape)
    # add does not seem to require normalization
    result = cv2.add(image, values, dtype=cv2.CV_8U)
    if result.ndim == 2 and ind == 3:
        return result[:, :, np.newaxis]
    return result


def _add_elementwise_np_to_uint8(image: Array, values: Array) -> Array:
    # This special uint8 block is around 60-100% faster than the
    # corresponding non-uint8 function further below (more speedup
    # for smaller images).
    #
    # Also tested to instead compute min/max of image and value
    # and then only convert image/value dtype if actually
    # necessary, but that was like 20-30% slower, even for 224x224
    # images.
    #
    if values.dtype.kind == "f":
        values = np.round(values)

    image = image.astype(np.int16)
    values = np.clip(values, -255, 255).astype(np.int16)

    image_aug = image + values
    image_aug = np.clip(image_aug, 0, 255).astype(np.uint8)

    return image_aug


def _add_elementwise_np_to_non_uint8(image: Array, values: Array) -> Array:
    # We limit here the value range of the value parameter to the
    # bytes in the image's dtype. This prevents overflow problems
    # and makes it less likely that the image has to be up-casted,
    # which again improves performance and saves memory. Note that
    # this also enables more dtypes for image inputs.
    # The downside is that the mul parameter is limited in its
    # value range.
    #
    # We need 2* the itemsize of the image here to allow to shift
    # the image's max value to the lowest possible value, e.g. for
    # uint8 it must allow for -255 to 255.
    if image.dtype.kind != "f" and values.dtype.kind == "f":
        values = np.round(values)

    input_shape = image.shape
    input_dtype = image.dtype

    if image.ndim == 2:
        image = image[..., np.newaxis]
    if values.ndim == 2:
        values = values[..., np.newaxis]
    nb_channels = image.shape[-1]

    itemsize = image.dtype.itemsize * 2
    dtype_target = np.dtype(f"{values.dtype.kind}{itemsize}")
    values = iadt.clip_to_dtype_value_range_(values, dtype_target, validate=100)

    if values.shape[2] == 1:
        values = np.tile(values, (1, 1, nb_channels))

    # Decreased itemsize from 2 to 1 here, see explanation in Add.
    image, values = iadt.promote_arrays_to_minimal_dtype_(
        [image, values], dtypes=[image.dtype, dtype_target], increase_itemsize_factor=1
    )
    image = np.add(image, values, out=image, casting="no")
    image = iadt.restore_dtypes_(image, input_dtype)

    if len(input_shape) == 2:
        return image[..., 0]
    return image


class Add(meta.Augmenter):
    """
    Add a value to all pixels in an image.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.arithmetic.add_scalar`.

    Parameters
    ----------
    value : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Value to add to all pixels.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

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
    >>> aug = iaa.Add(10)

    Always adds a value of 10 to all channels of all pixels of all input
    images.

    >>> aug = iaa.Add((-10, 10))

    Adds a value from the discrete interval ``[-10..10]`` to all pixels of
    input images. The exact value is sampled per image.

    >>> aug = iaa.Add((-10, 10), per_channel=True)

    Adds a value from the discrete interval ``[-10..10]`` to all pixels of
    input images. The exact value is sampled per image *and* channel,
    i.e. to a red-channel it might add 5 while subtracting 7 from the
    blue channel of the same image.

    >>> aug = iaa.Add((-10, 10), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(
        self,
        value: ParamInput = (-20, 20),
        per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.value = iap.handle_continuous_param(
            value,
            "value",
            value_range=None,
            tuple_to_uniform=True,
            list_to_choice=True,
            prefetch=True,
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
        value_samples = self.value.draw_samples((nb_images, nb_channels_max), random_state=rss[1])

        gen = enumerate(zip(images, value_samples, per_channel_samples, strict=True))
        for i, (image, value_samples_i, per_channel_samples_i) in gen:
            nb_channels = image.shape[2]

            # Example code to directly add images via image+sample (uint8 only)
            # if per_channel_samples_i > 0.5:
            #     result = []
            #     image = image.astype(np.int16)
            #     value_samples_i = value_samples_i.astype(np.int16)
            #     for c, value in enumerate(value_samples_i[0:nb_channels]):
            #         result.append(
            #             np.clip(
            #                 image[..., c:c+1] + value, 0, 255
            #             ).astype(np.uint8))
            #     images[i] = np.concatenate(result, axis=2)
            # else:
            #     images[i] = np.clip(
            #         image.astype(np.int16)
            #         + value_samples_i[0].astype(np.int16),
            #         0, 255
            #     ).astype(np.uint8)

            if per_channel_samples_i > 0.5:
                value = value_samples_i[0:nb_channels]
            else:
                # the if/else here catches the case of the channel axis being 0
                value = value_samples_i[0] if value_samples_i.size > 0 else []

            batch.images[i] = add_scalar_(image, value)

        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.value, self.per_channel]


# TODO merge this with Add
class AddElementwise(meta.Augmenter):
    """
    Add to the pixels of images values that are pixelwise randomly sampled.

    While the ``Add`` Augmenter samples one value to add *per image* (and
    optionally per channel), this augmenter samples different values per image
    and *per pixel* (and optionally per channel), i.e. intensities of
    neighbouring pixels may be increased/decreased by different amounts.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.arithmetic.add_elementwise`.

    Parameters
    ----------
    value : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Value to add to the pixels.

            * If an int, exactly that value will always be used.
            * If a tuple ``(a, b)``, then values from the discrete interval
              ``[a..b]`` will be sampled per image and pixel.
            * If a list of integers, a random value will be sampled from the
              list per image and pixel.
            * If a ``StochasticParameter``, then values will be sampled per
              image and pixel from that parameter.

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
    >>> aug = iaa.AddElementwise(10)

    Always adds a value of 10 to all channels of all pixels of all input
    images.

    >>> aug = iaa.AddElementwise((-10, 10))

    Samples per image and pixel a value from the discrete interval
    ``[-10..10]`` and adds that value to the respective pixel.

    >>> aug = iaa.AddElementwise((-10, 10), per_channel=True)

    Samples per image, pixel *and also channel* a value from the discrete
    interval ``[-10..10]`` and adds it to the respective pixel's channel value.
    Therefore, added values may differ between channels of the same pixel.

    >>> aug = iaa.AddElementwise((-10, 10), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(
        self,
        value: ParamInput = (-20, 20),
        per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.value = iap.handle_continuous_param(
            value, "value", value_range=None, tuple_to_uniform=True, list_to_choice=True
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

        gen = enumerate(zip(images, per_channel_samples, rss[1:], strict=True))
        for i, (image, per_channel_samples_i, rs) in gen:
            height, width, nb_channels = image.shape
            sample_shape = (height, width, nb_channels if per_channel_samples_i > 0.5 else 1)
            values = self.value.draw_samples(sample_shape, random_state=rs)

            batch.images[i] = add_elementwise(image, values)

        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.value, self.per_channel]


# TODO rename to AddGaussianNoise?
# TODO examples say that iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)) samples
#      the scale from the uniform dist. per image, but is that still the case?
#      AddElementwise seems to now sample once for all images, which should
#      lead to a single scale value.
class AdditiveGaussianNoise(AddElementwise):
    """
    Add noise sampled from gaussian distributions elementwise to images.

    This augmenter samples and adds noise elementwise, i.e. it can add
    different noise values to neighbouring pixels and is comparable
    to ``AddElementwise``.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.arithmetic.AddElementwise`.

    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Mean of the normal distribution from which the noise is sampled.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    scale : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Standard deviation of the normal distribution that generates the noise.
        Must be ``>=0``. If ``0`` then `loc` will simply be added to all
        pixels.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

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
    >>> aug = iaa.AdditiveGaussianNoise(scale=0.1*255)

    Adds gaussian noise from the distribution ``N(0, 0.1*255)`` to images.
    The samples are drawn per image and pixel.

    >>> aug = iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))

    Adds gaussian noise from the distribution ``N(0, s)`` to images,
    where ``s`` is sampled per image from the interval ``[0, 0.1*255]``.

    >>> aug = iaa.AdditiveGaussianNoise(scale=0.1*255, per_channel=True)

    Adds gaussian noise from the distribution ``N(0, 0.1*255)`` to images,
    where the noise value is different per image and pixel *and* channel (e.g.
    a different one for red, green and blue channels of the same pixel).
    This leads to "colorful" noise.

    >>> aug = iaa.AdditiveGaussianNoise(scale=0.1*255, per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(
        self,
        loc: ParamInput = 0,
        scale: ParamInput = (0, 15),
        per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        loc2 = iap.handle_continuous_param(
            loc, "loc", value_range=None, tuple_to_uniform=True, list_to_choice=True
        )
        scale2 = iap.handle_continuous_param(
            scale, "scale", value_range=(0, None), tuple_to_uniform=True, list_to_choice=True
        )

        value = iap.Normal(loc=loc2, scale=scale2)

        super().__init__(
            value,
            per_channel=per_channel,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class AdditiveLaplaceNoise(AddElementwise):
    """
    Add noise sampled from laplace distributions elementwise to images.

    The laplace distribution is similar to the gaussian distribution, but
    puts more weight on the long tail. Hence, this noise will add more
    outliers (very high/low values). It is somewhere between gaussian noise and
    salt and pepper noise.

    Values of around ``255 * 0.05`` for `scale` lead to visible noise (for
    ``uint8``).
    Values of around ``255 * 0.10`` for `scale` lead to very visible
    noise (for ``uint8``).
    It is recommended to usually set `per_channel` to ``True``.

    This augmenter samples and adds noise elementwise, i.e. it can add
    different noise values to neighbouring pixels and is comparable
    to ``AddElementwise``.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.arithmetic.AddElementwise`.

    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Mean of the laplace distribution that generates the noise.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    scale : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Standard deviation of the laplace distribution that generates the noise.
        Must be ``>=0``. If ``0`` then only `loc` will be used.
        Recommended to be around ``255*0.05``.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

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
    >>> aug = iaa.AdditiveLaplaceNoise(scale=0.1*255)

    Adds laplace noise from the distribution ``Laplace(0, 0.1*255)`` to images.
    The samples are drawn per image and pixel.

    >>> aug = iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255))

    Adds laplace noise from the distribution ``Laplace(0, s)`` to images,
    where ``s`` is sampled per image from the interval ``[0, 0.1*255]``.

    >>> aug = iaa.AdditiveLaplaceNoise(scale=0.1*255, per_channel=True)

    Adds laplace noise from the distribution ``Laplace(0, 0.1*255)`` to images,
    where the noise value is different per image and pixel *and* channel (e.g.
    a different one for the red, green and blue channels of the same pixel).
    This leads to "colorful" noise.

    >>> aug = iaa.AdditiveLaplaceNoise(scale=0.1*255, per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(
        self,
        loc: ParamInput = 0,
        scale: ParamInput = (0, 15),
        per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        loc2 = iap.handle_continuous_param(
            loc, "loc", value_range=None, tuple_to_uniform=True, list_to_choice=True
        )
        scale2 = iap.handle_continuous_param(
            scale, "scale", value_range=(0, None), tuple_to_uniform=True, list_to_choice=True
        )

        value = iap.Laplace(loc=loc2, scale=scale2)

        super().__init__(
            value,
            per_channel=per_channel,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class AdditivePoissonNoise(AddElementwise):
    """
    Add noise sampled from poisson distributions elementwise to images.

    Poisson noise is comparable to gaussian noise, as e.g. generated via
    ``AdditiveGaussianNoise``. As poisson distributions produce only positive
    numbers, the sign of the sampled values are here randomly flipped.

    Values of around ``10.0`` for `lam` lead to visible noise (for ``uint8``).
    Values of around ``20.0`` for `lam` lead to very visible noise (for
    ``uint8``).
    It is recommended to usually set `per_channel` to ``True``.

    This augmenter samples and adds noise elementwise, i.e. it can add
    different noise values to neighbouring pixels and is comparable
    to ``AddElementwise``.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.arithmetic.AddElementwise`.

    Parameters
    ----------
    lam : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Lambda parameter of the poisson distribution. Must be ``>=0``.
        Recommended values are around ``0.0`` to ``10.0``.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

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
    >>> aug = iaa.AdditivePoissonNoise(lam=5.0)

    Adds poisson noise sampled from a poisson distribution with a ``lambda``
    parameter of ``5.0`` to images.
    The samples are drawn per image and pixel.

    >>> aug = iaa.AdditivePoissonNoise(lam=(0.0, 15.0))

    Adds poisson noise sampled from ``Poisson(x)`` to images, where ``x`` is
    randomly sampled per image from the interval ``[0.0, 15.0]``.

    >>> aug = iaa.AdditivePoissonNoise(lam=5.0, per_channel=True)

    Adds poisson noise sampled from ``Poisson(5.0)`` to images,
    where the values are different per image and pixel *and* channel (e.g. a
    different one for red, green and blue channels for the same pixel).

    >>> aug = iaa.AdditivePoissonNoise(lam=(0.0, 15.0), per_channel=True)

    Adds poisson noise sampled from ``Poisson(x)`` to images,
    with ``x`` being sampled from ``uniform(0.0, 15.0)`` per image and
    channel. This is the *recommended* configuration.

    >>> aug = iaa.AdditivePoissonNoise(lam=(0.0, 15.0), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(
        self,
        lam: ParamInput = (0.0, 15.0),
        per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        lam2 = iap.handle_continuous_param(
            lam, "lam", value_range=(0, None), tuple_to_uniform=True, list_to_choice=True
        )

        value = iap.RandomSign(iap.Poisson(lam=lam2))

        super().__init__(
            value,
            per_channel=per_channel,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )
