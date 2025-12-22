from __future__ import annotations

from typing import Literal

import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.compat.markers import legacy
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, Images, ParamInput, RNGInput
from ._utils import PerChannelInput, SizePercentInput, SizePxInput


def replace_elementwise_(image: Array, mask: Array, replacements: Array) -> Array:
    """Replace components in an image array with new values.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: no (1)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: no (2)
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no
        * ``bool``: yes; tested

        - (1) ``uint64`` is currently not supported, because
              :func:`~imgaug2.dtypes.clip_to_dtype_value_range_()` does not
              support it, which again is because numpy.clip() seems to not
              support it.
        - (2) `int64` is disallowed due to being converted to `float64`
              by :func:`numpy.clip` since 1.17 (possibly also before?).

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.

    mask : ndarray
        Mask of shape ``(H,W,[C])`` denoting which components to replace.
        If ``C`` is provided, it must be ``1`` or match the ``C`` of `image`.
        May contain floats in the interval ``[0.0, 1.0]``.

    replacements : iterable
        Replacements to place in `image` at the locations defined by `mask`.
        This 1-dimensional iterable must contain exactly as many values
        as there are replaced components in `image`.

    Returns
    -------
    ndarray
        Image with replaced components.

    """
    iadt.gate_dtypes_strs(
        {image.dtype},
        allowed="bool uint8 uint16 uint32 int8 int16 int32 float16 float32 float64",
        disallowed="uint64 int64 float128",
        augmenter=None,
    )

    # This is slightly faster (~20%) for masks that are True at many
    # locations, but slower (~50%) for masks with few Trues, which is
    # probably the more common use-case:
    #
    # replacement_samples = self.replacement.draw_samples(
    #     sampling_shape, random_state=rs_replacement)
    #
    # # round, this makes 0.2 e.g. become 0 in case of boolean
    # # image (otherwise replacing values with 0.2 would
    # # lead to True instead of False).
    # if (image.dtype.kind in ["i", "u", "b"]
    #         and replacement_samples.dtype.kind == "f"):
    #     replacement_samples = np.round(replacement_samples)
    #
    # replacement_samples = iadt.clip_to_dtype_value_range_(
    #     replacement_samples, image.dtype, validate=False)
    # replacement_samples = replacement_samples.astype(
    #     image.dtype, copy=False)
    #
    # if sampling_shape[2] == 1:
    #     mask_samples = np.tile(mask_samples, (1, 1, nb_channels))
    #     replacement_samples = np.tile(
    #         replacement_samples, (1, 1, nb_channels))
    # mask_thresh = mask_samples > 0.5
    # image[mask_thresh] = replacement_samples[mask_thresh]
    input_shape = image.shape
    if image.ndim == 2:
        image = image[..., np.newaxis]
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]

    mask_thresh = mask > 0.5
    if mask.shape[2] == 1:
        nb_channels = image.shape[-1]
        # TODO verify if tile() is here really necessary
        mask_thresh = np.tile(mask_thresh, (1, 1, nb_channels))

    # round, this makes 0.2 e.g. become 0 in case of boolean
    # image (otherwise replacing values with 0.2 would lead to True
    # instead of False).
    if image.dtype.kind in ["i", "u", "b"] and replacements.dtype.kind == "f":
        replacements = np.round(replacements)

    replacement_samples = iadt.clip_to_dtype_value_range_(replacements, image.dtype, validate=False)
    replacement_samples = replacement_samples.astype(image.dtype, copy=False)

    image[mask_thresh] = replacement_samples
    if len(input_shape) == 2:
        return image[..., 0]
    return image


class ReplaceElementwise(meta.Augmenter):
    """
    Replace pixels in an image with new values.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.arithmetic.replace_elementwise_`.

    Parameters
    ----------
    mask : float or tuple of float or list of float or imgaug2.parameters.StochasticParameter
        Mask that indicates the pixels that are supposed to be replaced.
        The mask will be binarized using a threshold of ``0.5``. A value
        of ``1`` then indicates a pixel that is supposed to be replaced.

            * If this is a float, then that value will be used as the
              probability of being a ``1`` in the mask (sampled per image and
              pixel) and hence being replaced.
            * If a tuple ``(a, b)``, then the probability will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image and pixel.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample a mask per image.

    replacement : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        The replacement to use at all locations that are marked as ``1`` in
        the mask.

            * If this is a number, then that value will always be used as the
              replacement.
            * If a tuple ``(a, b)``, then the replacement will be sampled
              uniformly per image and pixel from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image and pixel.
            * If a ``StochasticParameter``, then this parameter will be used
              sample replacement values per image and pixel.

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
    >>> aug = ReplaceElementwise(0.05, [0, 255])

    Replaces ``5`` percent of all pixels in each image by either ``0``
    or ``255``.

    >>> import imgaug2.augmenters as iaa
    >>> aug = ReplaceElementwise(0.1, [0, 255], per_channel=0.5)

    For ``50%`` of all images, replace ``10%`` of all pixels with either the
    value ``0`` or the value ``255`` (same as in the previous example). For
    the other ``50%`` of all images, replace *channelwise* ``10%`` of all
    pixels with either the value ``0`` or the value ``255``. So, it will be
    very rare for each pixel to have all channels replaced by ``255`` or
    ``0``.

    >>> import imgaug2.augmenters as iaa
    >>> import imgaug2.parameters as iap
    >>> aug = ReplaceElementwise(0.1, iap.Normal(128, 0.4*128), per_channel=0.5)

    Replace ``10%`` of all pixels by gaussian noise centered around ``128``.
    Both the replacement mask and the gaussian noise are sampled channelwise
    for ``50%`` of all images.

    >>> import imgaug2.augmenters as iaa
    >>> import imgaug2.parameters as iap
    >>> aug = ReplaceElementwise(
    >>>     iap.FromLowerResolution(iap.Binomial(0.1), size_px=8),
    >>>     iap.Normal(128, 0.4*128),
    >>>     per_channel=0.5)

    Replace ``10%`` of all pixels by gaussian noise centered around ``128``.
    Sample the replacement mask at a lower resolution (``8x8`` pixels) and
    upscale it to the image size, resulting in coarse areas being replaced by
    gaussian noise.

    """

    def __init__(
        self,
        mask: ParamInput,
        replacement: ParamInput,
        per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.mask = iap.handle_probability_param(
            mask, "mask", tuple_to_uniform=True, list_to_choice=True
        )
        self.replacement = iap.handle_continuous_param(replacement, "replacement")
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
        rss = random_state.duplicate(1 + 2 * nb_images)
        per_channel_samples = self.per_channel.draw_samples((nb_images,), random_state=rss[0])

        gen = enumerate(zip(images, per_channel_samples, rss[1::2], rss[2::2], strict=True))
        for i, (image, per_channel_i, rs_mask, rs_replacement) in gen:
            height, width, nb_channels = image.shape
            sampling_shape = (height, width, nb_channels if per_channel_i > 0.5 else 1)
            mask_samples = self.mask.draw_samples(sampling_shape, random_state=rs_mask)

            # TODO add separate per_channels for mask and replacement
            if per_channel_i <= 0.5:
                nb_channels = image.shape[-1]
                replacement_samples = self.replacement.draw_samples(
                    (int(np.sum(mask_samples[:, :, 0])),), random_state=rs_replacement
                )
                # important here to use repeat instead of tile. repeat
                # converts e.g. [0, 1, 2] to [0, 0, 1, 1, 2, 2], while tile
                # leads to [0, 1, 2, 0, 1, 2]. The assignment below iterates
                # over each channel and pixel simultaneously, *not* first
                # over all pixels of channel 0, then all pixels in
                # channel 1, ...
                replacement_samples = np.repeat(replacement_samples, nb_channels)
            else:
                replacement_samples = self.replacement.draw_samples(
                    (int(np.sum(mask_samples)),), random_state=rs_replacement
                )

            batch.images[i] = replace_elementwise_(image, mask_samples, replacement_samples)

        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.mask, self.replacement, self.per_channel]


class SaltAndPepper(ReplaceElementwise):
    """
    Replace pixels in images with salt/pepper noise (white/black-ish colors).

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.arithmetic.ReplaceElementwise`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug2.parameters.StochasticParameter, optional
        Probability of replacing a pixel to salt/pepper noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a image-sized mask will be
              sampled from that parameter per image. Any value ``>0.5`` in
              that mask will be replaced with salt and pepper noise.

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
    >>> aug = iaa.SaltAndPepper(0.05)

    Replace ``5%`` of all pixels with salt and pepper noise.

    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.SaltAndPepper(0.05, per_channel=True)

    Replace *channelwise* ``5%`` of all pixels with salt and pepper
    noise.

    """

    def __init__(
        self,
        p: ParamInput = (0.0, 0.03),
        per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            mask=p,
            replacement=iap.Beta(0.5, 0.5) * 255,
            per_channel=per_channel,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class ImpulseNoise(SaltAndPepper):
    """
    Add impulse noise to images.

    This is identical to ``SaltAndPepper``, except that `per_channel` is
    always set to ``True``.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.arithmetic.SaltAndPepper`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug2.parameters.StochasticParameter, optional
        Probability of replacing a pixel to impulse noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a image-sized mask will be
              sampled from that parameter per image. Any value ``>0.5`` in
              that mask will be replaced with impulse noise noise.

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
    >>> aug = iaa.ImpulseNoise(0.1)

    Replace ``10%`` of all pixels with impulse noise.

    """

    def __init__(
        self,
        p: ParamInput = (0.0, 0.03),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            p=p,
            per_channel=True,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class CoarseSaltAndPepper(ReplaceElementwise):
    """
    Replace rectangular areas in images with white/black-ish pixel noise.

    This adds salt and pepper noise (noisy white-ish and black-ish pixels) to
    rectangular areas within the image. Note that this means that within these
    rectangular areas the color varies instead of each rectangle having only
    one color.

    See also the similar ``CoarseDropout``.

    TODO replace dtype support with uint8 only, because replacement is
         geared towards that value range

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.arithmetic.ReplaceElementwise`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug2.parameters.StochasticParameter, optional
        Probability of changing a pixel to salt/pepper noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a lower-resolution mask will
              be sampled from that parameter per image. Any value ``>0.5`` in
              that mask will denote a spatial location that is to be replaced
              by salt and pepper noise.

    size_px : int or tuple of int or imgaug2.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask in absolute pixel dimensions.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_percent` must be set.
            * If an integer, then that size will always be used for both height
              and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
              which is then upsampled to ``HxW``, where ``H`` is the image size
              and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
              sampled from the discrete interval ``[a..b]``. The mask
              will then be generated at size ``MxN`` and upsampled to ``HxW``.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug2.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask *in percent* of the input image.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_px` must be set.
            * If a float, then that value will always be used as the percentage
              of the height and width (relative to the original size). E.g. for
              value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
              later upsampled to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
              sampled from the interval ``(a, b)`` and used as the size
              fractions, i.e the mask size will be ``(m*H)x(n*W)``.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

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

    min_size : int, optional
        Minimum height and width of the low resolution mask. If
        `size_percent` or `size_px` leads to a lower value than this,
        `min_size` will be used instead. This should never have a value of
        less than ``2``, otherwise one may end up with a ``1x1`` low resolution
        mask, leading easily to the whole image being replaced.

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
    >>> aug = iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1))

    Marks ``5%`` of all pixels in a mask to be replaced by salt/pepper
    noise. The mask has ``1%`` to ``10%`` the size of the input image.
    The mask is then upscaled to the input image size, leading to large
    rectangular areas being marked as to be replaced. These areas are then
    replaced in the input image by salt/pepper noise.

    >>> aug = iaa.CoarseSaltAndPepper(0.05, size_px=(4, 16))

    Same as in the previous example, but the replacement mask before upscaling
    has a size between ``4x4`` and ``16x16`` pixels (the axis sizes are sampled
    independently, i.e. the mask may be rectangular).

    >>> aug = iaa.CoarseSaltAndPepper(
    >>>    0.05, size_percent=(0.01, 0.1), per_channel=True)

    Same as in the first example, but mask and replacement are each sampled
    independently per image channel.

    """

    def __init__(
        self,
        p: ParamInput = (0.02, 0.1),
        size_px: SizePxInput = None,
        size_percent: SizePercentInput = None,
        per_channel: PerChannelInput = False,
        min_size: int = 3,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        mask = iap.handle_probability_param(p, "p", tuple_to_uniform=True, list_to_choice=True)

        if size_px is not None:
            mask_low = iap.FromLowerResolution(other_param=mask, size_px=size_px, min_size=min_size)
        elif size_percent is not None:
            mask_low = iap.FromLowerResolution(
                other_param=mask, size_percent=size_percent, min_size=min_size
            )
        else:
            mask_low = iap.FromLowerResolution(other_param=mask, size_px=(3, 8), min_size=min_size)

        replacement = iap.Beta(0.5, 0.5) * 255

        super().__init__(
            mask=mask_low,
            replacement=replacement,
            per_channel=per_channel,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class Salt(ReplaceElementwise):
    """
    Replace pixels in images with salt noise, i.e. white-ish pixels.

    This augmenter is similar to ``SaltAndPepper``, but adds no pepper noise to
    images.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.arithmetic.ReplaceElementwise`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug2.parameters.StochasticParameter, optional
        Probability of replacing a pixel with salt noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a image-sized mask will be
              sampled from that parameter per image. Any value ``>0.5`` in
              that mask will be replaced with salt noise.

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
    >>> aug = iaa.Salt(0.05)

    Replace ``5%`` of all pixels with salt noise (white-ish colors).

    """

    def __init__(
        self,
        p: ParamInput = (0.0, 0.03),
        per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        replacement01 = iap.ForceSign(iap.Beta(0.5, 0.5) - 0.5, positive=True, mode="invert") + 0.5
        # `replacement01` is in [0.5, 1.0) with floating-point sampling, so
        # scaling by 255 makes values near 255 very rare after rounding.
        # Scale by 256 and rely on later dtype clipping to allow pure white.
        replacement = replacement01 * 256

        super().__init__(
            mask=p,
            replacement=replacement,
            per_channel=per_channel,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class CoarseSalt(ReplaceElementwise):
    """
    Replace rectangular areas in images with white-ish pixel noise.

    See also the similar ``CoarseSaltAndPepper``.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.arithmetic.ReplaceElementwise`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug2.parameters.StochasticParameter, optional
        Probability of changing a pixel to salt noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a lower-resolution mask will
              be sampled from that parameter per image. Any value ``>0.5`` in
              that mask will denote a spatial location that is to be replaced
              by salt noise.

    size_px : int or tuple of int or imgaug2.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask in absolute pixel dimensions.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_percent` must be set.
            * If an integer, then that size will always be used for both height
              and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
              which is then upsampled to ``HxW``, where ``H`` is the image size
              and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
              sampled from the discrete interval ``[a..b]``. The mask
              will then be generated at size ``MxN`` and upsampled to ``HxW``.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug2.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask *in percent* of the input image.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_px` must be set.
            * If a float, then that value will always be used as the percentage
              of the height and width (relative to the original size). E.g. for
              value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
              later upsampled to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
              sampled from the interval ``(a, b)`` and used as the size
              fractions, i.e the mask size will be ``(m*H)x(n*W)``.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

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

    min_size : int, optional
        Minimum height and width of the low resolution mask. If
        `size_percent` or `size_px` leads to a lower value than this,
        `min_size` will be used instead. This should never have a value of
        less than ``2``, otherwise one may end up with a ``1x1`` low resolution
        mask, leading easily to the whole image being replaced.

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
    >>> aug = iaa.CoarseSalt(0.05, size_percent=(0.01, 0.1))

    Mark ``5%`` of all pixels in a mask to be replaced by salt
    noise. The mask has ``1%`` to ``10%`` the size of the input image.
    The mask is then upscaled to the input image size, leading to large
    rectangular areas being marked as to be replaced. These areas are then
    replaced in the input image by salt noise.

    """

    def __init__(
        self,
        p: ParamInput = (0.02, 0.1),
        size_px: SizePxInput = None,
        size_percent: SizePercentInput = None,
        per_channel: PerChannelInput = False,
        min_size: int = 3,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        mask = iap.handle_probability_param(p, "p", tuple_to_uniform=True, list_to_choice=True)

        if size_px is not None:
            mask_low = iap.FromLowerResolution(other_param=mask, size_px=size_px, min_size=min_size)
        elif size_percent is not None:
            mask_low = iap.FromLowerResolution(
                other_param=mask, size_percent=size_percent, min_size=min_size
            )
        else:
            mask_low = iap.FromLowerResolution(other_param=mask, size_px=(3, 8), min_size=min_size)

        replacement01 = iap.ForceSign(iap.Beta(0.5, 0.5) - 0.5, positive=True, mode="invert") + 0.5
        replacement = replacement01 * 255

        super().__init__(
            mask=mask_low,
            replacement=replacement,
            per_channel=per_channel,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class Pepper(ReplaceElementwise):
    """
    Replace pixels in images with pepper noise, i.e. black-ish pixels.

    This augmenter is similar to ``SaltAndPepper``, but adds no salt noise to
    images.

    This augmenter is similar to ``Dropout``, but slower and the black pixels
    are not uniformly black.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.arithmetic.ReplaceElementwise`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug2.parameters.StochasticParameter, optional
        Probability of replacing a pixel with pepper noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a image-sized mask will be
              sampled from that parameter per image. Any value ``>0.5`` in
              that mask will be replaced with pepper noise.

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
    >>> aug = iaa.Pepper(0.05)

    Replace ``5%`` of all pixels with pepper noise (black-ish colors).

    """

    def __init__(
        self,
        p: ParamInput = (0.0, 0.05),
        per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        replacement01 = iap.ForceSign(iap.Beta(0.5, 0.5) - 0.5, positive=False, mode="invert") + 0.5
        replacement = replacement01 * 255

        super().__init__(
            mask=p,
            replacement=replacement,
            per_channel=per_channel,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class CoarsePepper(ReplaceElementwise):
    """
    Replace rectangular areas in images with black-ish pixel noise.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.arithmetic.ReplaceElementwise`.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug2.parameters.StochasticParameter, optional
        Probability of changing a pixel to pepper noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a lower-resolution mask will
              be sampled from that parameter per image. Any value ``>0.5`` in
              that mask will denote a spatial location that is to be replaced
              by pepper noise.

    size_px : int or tuple of int or imgaug2.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask in absolute pixel dimensions.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_percent` must be set.
            * If an integer, then that size will always be used for both height
              and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
              which is then upsampled to ``HxW``, where ``H`` is the image size
              and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
              sampled from the discrete interval ``[a..b]``. The mask
              will then be generated at size ``MxN`` and upsampled to ``HxW``.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug2.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask *in percent* of the input image.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_px` must be set.
            * If a float, then that value will always be used as the percentage
              of the height and width (relative to the original size). E.g. for
              value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
              later upsampled to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
              sampled from the interval ``(a, b)`` and used as the size
              fractions, i.e the mask size will be ``(m*H)x(n*W)``.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

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

    min_size : int, optional
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a ``1x1`` low resolution mask, leading
        easily to the whole image being replaced.

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
    >>> aug = iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1))

    Mark ``5%`` of all pixels in a mask to be replaced by pepper
    noise. The mask has ``1%`` to ``10%`` the size of the input image.
    The mask is then upscaled to the input image size, leading to large
    rectangular areas being marked as to be replaced. These areas are then
    replaced in the input image by pepper noise.

    """

    def __init__(
        self,
        p: ParamInput = (0.02, 0.1),
        size_px: SizePxInput = None,
        size_percent: SizePercentInput = None,
        per_channel: PerChannelInput = False,
        min_size: int = 3,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        mask = iap.handle_probability_param(p, "p", tuple_to_uniform=True, list_to_choice=True)

        if size_px is not None:
            mask_low = iap.FromLowerResolution(other_param=mask, size_px=size_px, min_size=min_size)
        elif size_percent is not None:
            mask_low = iap.FromLowerResolution(
                other_param=mask, size_percent=size_percent, min_size=min_size
            )
        else:
            mask_low = iap.FromLowerResolution(other_param=mask, size_px=(3, 8), min_size=min_size)

        replacement01 = iap.ForceSign(iap.Beta(0.5, 0.5) - 0.5, positive=False, mode="invert") + 0.5
        replacement = replacement01 * 255

        super().__init__(
            mask=mask_low,
            replacement=replacement,
            per_channel=per_channel,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )
