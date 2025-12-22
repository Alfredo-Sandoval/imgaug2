from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.compat.markers import legacy
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, Images, ParamInput, RNGInput

from ._utils import (
    CSPACE_HLS,
    CSPACE_HSV,
    CSPACE_Lab,
    CSPACE_Luv,
    CSPACE_RGB,
    CSPACE_YCrCb,
    CSPACE_YUV,
    ChildrenInput,
    ColorSpace,
    ToColorspaceParamInput,
    _get_arithmetic,
)
from .colorspace import change_colorspaces_


@legacy(version="0.4.0")
class WithBrightnessChannels(meta.Augmenter):
    """Augmenter to apply child augmenters to brightness-related image channels.

    This augmenter first converts an image to a random colorspace containing a
    brightness-related channel (e.g. ``V`` in ``HSV``), then extracts that
    channel and applies its child augmenters to this one channel. Afterwards,
    it reintegrates the augmented channel into the full image and converts
    back to the input colorspace.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.color.change_colorspaces_`.

    Parameters
    ----------
    children : imgaug2.augmenters.meta.Augmenter or list of imgaug2.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to the brightness channels.
        They receive images with a single channel and have to modify these.

    to_colorspace : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        Colorspace in which to extract the brightness-related channels.
        Currently, ``imgaug2.augmenters.color.CSPACE_YCrCb``, ``CSPACE_HSV``,
        ``CSPACE_HLS``, ``CSPACE_Lab``, ``CSPACE_Luv``, ``CSPACE_YUV``,
        ``CSPACE_CIE`` are supported.

            * If ``imgaug2.ALL``: Will pick imagewise a random colorspace from
              all supported colorspaces.
            * If ``str``: Will always use this colorspace.
            * If ``list`` or ``str``: Will pick imagewise a random colorspace
              from this list.
            * If :class:`~imgaug2.parameters.StochasticParameter`:
              A parameter that will be queried once per batch to generate
              all target colorspaces. Expected to return strings matching the
              ``CSPACE_*`` constants.

    from_colorspace : str, optional
        See :func:`~imgaug2.augmenters.color.change_colorspace_`.

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
    >>> aug = iaa.WithBrightnessChannels(iaa.Add((-50, 50)))

    Add ``-50`` to ``50`` to the brightness-related channels of each image.

    >>> aug = iaa.WithBrightnessChannels(
    >>>     iaa.Add((-50, 50)), to_colorspace=[iaa.CSPACE_Lab, iaa.CSPACE_HSV])

    Add ``-50`` to ``50`` to the brightness-related channels of each image,
    but pick those brightness-related channels only from ``Lab`` (``L``) and
    ``HSV`` (``V``) colorspaces.

    >>> aug = iaa.WithBrightnessChannels(
    >>>     iaa.Add((-50, 50)), from_colorspace=iaa.CSPACE_BGR)

    Add ``-50`` to ``50`` to the brightness-related channels of each image,
    where the images are provided in ``BGR`` colorspace instead of the
    standard ``RGB``.

    """

    # Usually one would think that CSPACE_CIE (=XYZ) would also work, as
    # wikipedia says that Y denotes luminance, but this resulted in strong
    # color changes (tried also the other channels).
    _CSPACE_TO_CHANNEL_ID = {
        CSPACE_YCrCb: 0,
        CSPACE_HSV: 2,
        CSPACE_HLS: 1,
        CSPACE_Lab: 0,
        CSPACE_Luv: 0,
        CSPACE_YUV: 0,
    }

    _DEFAULT_COLORSPACES = [
        CSPACE_YCrCb,
        CSPACE_HSV,
        CSPACE_HLS,
        CSPACE_Lab,
        CSPACE_Luv,
        CSPACE_YUV,
    ]

    _VALID_COLORSPACES = set(_CSPACE_TO_CHANNEL_ID.keys())

    @legacy(version="0.4.0")
    def __init__(
        self,
        children: ChildrenInput = None,  # type: ignore
        to_colorspace: ToColorspaceParamInput = None,  # type: ignore
        from_colorspace: ColorSpace = "RGB",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.children = meta.handle_children_list(children, self.name, "then")
        if to_colorspace is None:
            to_colorspace = list(self._DEFAULT_COLORSPACES)
        self.to_colorspace = iap.handle_categorical_string_param(
            to_colorspace, "to_colorspace", valid_values=self._VALID_COLORSPACES
        )
        self.from_colorspace = from_colorspace

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        with batch.propagation_hooks_ctx(self, hooks, parents):
            images_cvt = None
            to_colorspaces = None

            if batch.images is not None:
                to_colorspaces = self.to_colorspace.draw_samples((len(batch.images),), random_state)
                images_cvt = change_colorspaces_(
                    batch.images,
                    from_colorspaces=self.from_colorspace,
                    to_colorspaces=to_colorspaces,
                )
                brightness_channels = self._extract_brightness_channels(images_cvt, to_colorspaces)

                batch.images = brightness_channels

            batch = self.children.augment_batch_(batch, parents=parents + [self], hooks=hooks)

            if batch.images is not None:
                batch.images = self._invert_extract_brightness_channels(
                    batch.images, images_cvt, to_colorspaces
                )

                batch.images = change_colorspaces_(
                    batch.images,
                    from_colorspaces=to_colorspaces,
                    to_colorspaces=self.from_colorspace,
                )

        return batch

    @legacy(version="0.4.0")
    def _extract_brightness_channels(
        self, images: Images, colorspaces: Sequence[ColorSpace]
    ) -> list[Array]:
        result = []
        for image, colorspace in zip(images, colorspaces, strict=True):
            channel_id = self._CSPACE_TO_CHANNEL_ID[colorspace]
            # Note that augmenters expect (H,W,C) and not (H,W), so cannot
            # just use image[:, :, channel_id] here.
            channel = image[:, :, channel_id : channel_id + 1]
            result.append(channel)
        return result

    @legacy(version="0.4.0")
    def _invert_extract_brightness_channels(
        self, channels: Sequence[Array], images: Images, colorspaces: Sequence[ColorSpace]
    ) -> Images:
        for channel, image, colorspace in zip(channels, images, colorspaces, strict=True):
            channel_id = self._CSPACE_TO_CHANNEL_ID[colorspace]
            image[:, :, channel_id : channel_id + 1] = channel
        return images

    @legacy(version="0.4.0")
    def _to_deterministic(self) -> meta.Augmenter:
        aug = self.copy()
        aug.children = aug.children.to_deterministic()
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.to_colorspace, self.from_colorspace]

    @legacy(version="0.4.0")
    def get_children_lists(self) -> list[list[meta.Augmenter]]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_children_lists`."""
        return cast(list[list[meta.Augmenter]], [self.children])

    @legacy(version="0.4.0")
    def __str__(self) -> str:
        return (
            "WithBrightnessChannels("
            f"to_colorspace={self.to_colorspace}, "
            f"from_colorspace={self.from_colorspace}, "
            f"name={self.name}, "
            f"children={self.children}, "
            f"deterministic={self.deterministic})"
        )


@legacy(version="0.4.0")
class MultiplyAndAddToBrightness(WithBrightnessChannels):
    """Multiply and add to the brightness channels of input images.

    This is a wrapper around :class:`WithBrightnessChannels` and hence
    performs internally the same projection to random colorspaces.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.color.WithBrightnessChannels`.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.airthmetic.Multiply`.

    add : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.airthmetic.Add`.

    to_colorspace : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.color.WithBrightnessChannels`.

    from_colorspace : str, optional
        See :class:`~imgaug2.augmenters.color.WithBrightnessChannels`.

    random_order : bool, optional
        Whether to apply the add and multiply operations in random
        order (``True``). If ``False``, this augmenter will always first
        multiply and then add.

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
    >>> aug = iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))

    Convert each image to a colorspace with a brightness-related channel,
    extract that channel, multiply it by a factor between ``0.5`` and ``1.5``,
    add a value between ``-30`` and ``30`` and convert back to the original
    colorspace.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        mul: ParamInput = (0.7, 1.3),
        add: ParamInput = (-30, 30),
        to_colorspace: ToColorspaceParamInput = None,  # type: ignore
        from_colorspace: ColorSpace = "RGB",
        random_order: bool = True,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        if to_colorspace is None:
            to_colorspace = [
                CSPACE_YCrCb,
                CSPACE_HSV,
                CSPACE_HLS,
                CSPACE_Lab,
                CSPACE_Luv,
                CSPACE_YUV,
            ]
        arithmetic_lib = _get_arithmetic()
        mul = (
            meta.Identity()
            if ia.is_single_number(mul) and np.isclose(mul, 1.0)
            else arithmetic_lib.Multiply(mul)
        )
        add = meta.Identity() if add == 0 else arithmetic_lib.Add(add)

        super().__init__(
            children=meta.Sequential([mul, add], random_order=random_order),
            to_colorspace=to_colorspace,
            from_colorspace=from_colorspace,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

    @legacy(version="0.4.0")
    def __str__(self) -> str:
        return (
            "MultiplyAndAddToBrightness("
            f"mul={str(self.children[0])}, "
            f"add={str(self.children[1])}, "
            f"to_colorspace={self.to_colorspace}, "
            f"from_colorspace={self.from_colorspace}, "
            f"random_order={self.children.random_order}, "
            f"name={self.name}, "
            f"deterministic={self.deterministic})"
        )


@legacy(version="0.4.0")
class MultiplyBrightness(MultiplyAndAddToBrightness):
    """Multiply the brightness channels of input images.

    This is a wrapper around :class:`WithBrightnessChannels` and hence
    performs internally the same projection to random colorspaces.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.color.MultiplyAndAddToBrightness`.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.airthmetic.Multiply`.

    to_colorspace : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.color.WithBrightnessChannels`.

    from_colorspace : str, optional
        See :class:`~imgaug2.augmenters.color.WithBrightnessChannels`.

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
    >>> aug = iaa.MultiplyBrightness((0.5, 1.5))

    Convert each image to a colorspace with a brightness-related channel,
    extract that channel, multiply it by a factor between ``0.5`` and ``1.5``,
    and convert back to the original colorspace.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        mul: ParamInput = (0.7, 1.3),
        to_colorspace: ToColorspaceParamInput = None,  # type: ignore
        from_colorspace: ColorSpace = "RGB",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        if to_colorspace is None:
            to_colorspace = [
                CSPACE_YCrCb,
                CSPACE_HSV,
                CSPACE_HLS,
                CSPACE_Lab,
                CSPACE_Luv,
                CSPACE_YUV,
            ]
        super().__init__(
            mul=mul,
            add=0,
            to_colorspace=to_colorspace,
            from_colorspace=from_colorspace,
            random_order=False,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
class AddToBrightness(MultiplyAndAddToBrightness):
    """Add to the brightness channels of input images.

    This is a wrapper around :class:`WithBrightnessChannels` and hence
    performs internally the same projection to random colorspaces.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.color.MultiplyAndAddToBrightness`.

    Parameters
    ----------
    add : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.airthmetic.Add`.

    to_colorspace : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.color.WithBrightnessChannels`.

    from_colorspace : str, optional
        See :class:`~imgaug2.augmenters.color.WithBrightnessChannels`.

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
    >>> aug = iaa.AddToBrightness((-30, 30))

    Convert each image to a colorspace with a brightness-related channel,
    extract that channel, add between ``-30`` and ``30`` and convert back
    to the original colorspace.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        add: ParamInput = (-30, 30),
        to_colorspace: ToColorspaceParamInput = None,  # type: ignore
        from_colorspace: ColorSpace = "RGB",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        if to_colorspace is None:
            to_colorspace = [
                CSPACE_YCrCb,
                CSPACE_HSV,
                CSPACE_HLS,
                CSPACE_Lab,
                CSPACE_Luv,
                CSPACE_YUV,
            ]
        super().__init__(
            mul=1.0,
            add=add,
            to_colorspace=to_colorspace,
            from_colorspace=from_colorspace,
            random_order=False,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


# TODO Merge this into WithColorspace? A bit problematic due to int16
#      conversion that would make WithColorspace less flexible.
# TODO add option to choose overflow behaviour for hue and saturation channels,
#      e.g. clip, modulo or wrap
