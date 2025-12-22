"""Aspect-ratio size augmenters."""

from __future__ import annotations

from typing import Literal

import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters._typing import ParamInput, RNGInput
from imgaug2.compat.markers import legacy

from ._utils import (
    CropToFixedSizeSamplingResult,
    PadToFixedSizeSamplingResult,
    compute_croppings_to_reach_aspect_ratio,
    compute_paddings_to_reach_aspect_ratio,
)
from .fixed_size import CropToFixedSize, PadToFixedSize
@legacy(version="0.4.0")
class CropToAspectRatio(CropToFixedSize):
    """Crop images until their width/height matches an aspect ratio.

    This augmenter removes either rows or columns until the image reaches
    the desired aspect ratio given in ``width / height``. The cropping
    operation is stopped once the desired aspect ratio is reached or the image
    side to crop reaches a size of ``1``. If any side of the image starts
    with a size of ``0``, the image will not be changed.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    aspect_ratio : number
        The desired aspect ratio, given as ``width/height``. E.g. a ratio
        of ``2.0`` denotes an image that is twice as wide as it is high.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`CropToFixedSize.__init__`.

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
    >>> aug = iaa.CropToAspectRatio(2.0)

    Create an augmenter that crops each image until its aspect ratio is as
    close as possible to ``2.0`` (i.e. two times as many pixels along the
    x-axis than the y-axis).
    The rows to be cropped will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        aspect_ratio: float | int,
        position: str
        | tuple[float | int | iap.StochasticParameter, float | int | iap.StochasticParameter]
        | iap.StochasticParameter = "uniform",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            width=None,
            height=None,
            position=position,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )
        self.aspect_ratio = aspect_ratio

    @legacy(version="0.4.0")
    def _draw_samples(
        self, batch: _BatchInAugmentation, random_state: iarandom.RNG
    ) -> CropToFixedSizeSamplingResult:
        _sizes, offset_xs, offset_ys = super()._draw_samples(batch, random_state)

        shapes = batch.get_rowwise_shapes()
        sizes = []
        for shape in shapes:
            height, width = shape[0:2]

            if height == 0 or width == 0:
                croppings = (0, 0, 0, 0)
            else:
                croppings = compute_croppings_to_reach_aspect_ratio(
                    shape, aspect_ratio=self.aspect_ratio
                )

            # TODO change that
            # note that these are not in the same order as shape tuples
            # in CropToFixedSize
            new_size = (width - croppings[1] - croppings[3], height - croppings[0] - croppings[2])
            sizes.append(new_size)

        return sizes, offset_xs, offset_ys

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.aspect_ratio, self.position]


@legacy(version="0.4.0")
class CenterCropToAspectRatio(CropToAspectRatio):
    """Crop images equally on all sides until they reach an aspect ratio.

    This is the same as :class:`~imgaug2.augmenters.size.CropToAspectRatio`, but
    uses ``position="center"`` by default, which spreads the crop amounts
    equally over all image sides, while
    :class:`~imgaug2.augmenters.size.CropToAspectRatio` by default spreads
    them randomly.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    aspect_ratio : number
        See :func:`CropToAspectRatio.__init__`.

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
    >>> aug = iaa.CenterCropToAspectRatio(2.0)

    Create an augmenter that crops each image until its aspect ratio is as
    close as possible to ``2.0`` (i.e. two times as many pixels along the
    x-axis than the y-axis).
    The rows to be cropped will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        aspect_ratio: float | int,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            aspect_ratio=aspect_ratio,
            position="center",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
class PadToAspectRatio(PadToFixedSize):
    """Pad images until their width/height matches an aspect ratio.

    This augmenter adds either rows or columns until the image reaches
    the desired aspect ratio given in ``width / height``.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    aspect_ratio : number
        The desired aspect ratio, given as ``width/height``. E.g. a ratio
        of ``2.0`` denotes an image that is twice as wide as it is high.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

    pad_mode : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        See :func:`~imgaug2.augmenters.size.PadToFixedSize.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :func:`~imgaug2.augmenters.size.PadToFixedSize.__init__`.

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
    >>> aug = iaa.PadToAspectRatio(2.0)

    Create an augmenter that pads each image until its aspect ratio is as
    close as possible to ``2.0`` (i.e. two times as many pixels along the
    x-axis than the y-axis).
    The rows to be padded will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        aspect_ratio: float | int,
        pad_mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        pad_cval: ParamInput = 0,
        position: str
        | tuple[float | int | iap.StochasticParameter, float | int | iap.StochasticParameter]
        | iap.StochasticParameter = "uniform",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            width=None,
            height=None,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            position=position,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )
        self.aspect_ratio = aspect_ratio

    @legacy(version="0.4.0")
    def _draw_samples(
        self, batch: _BatchInAugmentation, random_state: iarandom.RNG
    ) -> PadToFixedSizeSamplingResult:
        _sizes, pad_xs, pad_ys, pad_modes, pad_cvals = super()._draw_samples(batch, random_state)

        shapes = batch.get_rowwise_shapes()
        sizes = []
        for shape in shapes:
            height, width = shape[0:2]

            paddings = compute_paddings_to_reach_aspect_ratio(shape, aspect_ratio=self.aspect_ratio)

            # TODO change that
            # note that these are not in the same order as shape tuples
            # in PadToFixedSize
            new_size = (width + paddings[1] + paddings[3], height + paddings[0] + paddings[2])
            sizes.append(new_size)

        return sizes, pad_xs, pad_ys, pad_modes, pad_cvals

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.aspect_ratio, self.pad_mode, self.pad_cval, self.position]


@legacy(version="0.4.0")
class CenterPadToAspectRatio(PadToAspectRatio):
    """Pad images equally on all sides until H/W matches an aspect ratio.

    This is the same as :class:`~imgaug2.augmenters.size.PadToAspectRatio`, but
    uses ``position="center"`` by default, which spreads the pad amounts
    equally over all image sides, while
    :class:`~imgaug2.augmenters.size.PadToAspectRatio` by default spreads them
    randomly.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    aspect_ratio : number
        See :func:`PadToAspectRatio.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    pad_mode : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        See :func:`~imgaug2.augmenters.size.PadToAspectRatio.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :func:`~imgaug2.augmenters.size.PadToAspectRatio.__init__`.

    deterministic : bool, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.PadToAspectRatio(2.0)

    Create am augmenter that pads each image until its aspect ratio is as
    close as possible to ``2.0`` (i.e. two times as many pixels along the
    x-axis than the y-axis).
    The rows to be padded will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        aspect_ratio: float | int,
        pad_mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        pad_cval: ParamInput = 0,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            aspect_ratio=aspect_ratio,
            position="center",
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


