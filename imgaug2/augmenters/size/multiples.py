"""Multiples-of size augmenters."""

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
    compute_croppings_to_reach_multiples_of,
    compute_paddings_to_reach_multiples_of,
)
from .fixed_size import CropToFixedSize, PadToFixedSize
@legacy(version="0.4.0")
class CropToMultiplesOf(CropToFixedSize):
    """Crop images down until their height/width is a multiple of a value.

    .. note::

        For a given axis size ``A`` and multiple ``M``, if ``A`` is in the
        interval ``[0 .. M]``, the axis will not be changed.
        As a result, this augmenter can still produce axis sizes that are
        not multiples of the given values.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width_multiple : int or None
        Multiple for the width. Images will be cropped down until their
        width is a multiple of this value.
        If ``None``, image widths will not be altered.

    height_multiple : int or None
        Multiple for the height. Images will be cropped down until their
        height is a multiple of this value.
        If ``None``, image heights will not be altered.

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
    >>> aug = iaa.CropToMultiplesOf(height_multiple=10, width_multiple=6)

    Create an augmenter that crops images to multiples of ``10`` along
    the y-axis (i.e. 10, 20, 30, ...) and to multiples of ``6`` along the
    x-axis (i.e. 6, 12, 18, ...).
    The rows to be cropped will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        width_multiple: int | None,
        height_multiple: int | None,
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
        self.width_multiple = width_multiple
        self.height_multiple = height_multiple

    @legacy(version="0.4.0")
    def _draw_samples(
        self, batch: _BatchInAugmentation, random_state: iarandom.RNG
    ) -> CropToFixedSizeSamplingResult:
        _sizes, offset_xs, offset_ys = super()._draw_samples(batch, random_state)

        shapes = batch.get_rowwise_shapes()
        sizes = []
        for shape in shapes:
            height, width = shape[0:2]
            croppings = compute_croppings_to_reach_multiples_of(
                shape, height_multiple=self.height_multiple, width_multiple=self.width_multiple
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
        return [self.width_multiple, self.height_multiple, self.position]


@legacy(version="0.4.0")
class CenterCropToMultiplesOf(CropToMultiplesOf):
    """Crop images equally on all sides until H/W are multiples of given values.

    This is the same as :class:`~imgaug2.augmenters.size.CropToMultiplesOf`,
    but uses ``position="center"`` by default, which spreads the crop amounts
    equally over all image sides, while
    :class:`~imgaug2.augmenters.size.CropToMultiplesOf` by default spreads
    them randomly.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width_multiple : int or None
        See :func:`CropToMultiplesOf.__init__`.

    height_multiple : int or None
        See :func:`CropToMultiplesOf.__init__`.

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
    >>> aug = iaa.CenterCropToMultiplesOf(height_multiple=10, width_multiple=6)

    Create an augmenter that crops images to multiples of ``10`` along
    the y-axis (i.e. 10, 20, 30, ...) and to multiples of ``6`` along the
    x-axis (i.e. 6, 12, 18, ...).
    The rows to be cropped will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        width_multiple: int | None,
        height_multiple: int | None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            width_multiple=width_multiple,
            height_multiple=height_multiple,
            position="center",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
class PadToMultiplesOf(PadToFixedSize):
    """Pad images until their height/width is a multiple of a value.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width_multiple : int or None
        Multiple for the width. Images will be padded until their
        width is a multiple of this value.
        If ``None``, image widths will not be altered.

    height_multiple : int or None
        Multiple for the height. Images will be padded until their
        height is a multiple of this value.
        If ``None``, image heights will not be altered.

    pad_mode : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        See :func:`~imgaug2.augmenters.size.PadToFixedSize.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :func:`~imgaug2.augmenters.size.PadToFixedSize.__init__`.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

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
    >>> aug = iaa.PadToMultiplesOf(height_multiple=10, width_multiple=6)

    Create an augmenter that pads images to multiples of ``10`` along
    the y-axis (i.e. 10, 20, 30, ...) and to multiples of ``6`` along the
    x-axis (i.e. 6, 12, 18, ...).
    The rows to be padded will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        width_multiple: int | None,
        height_multiple: int | None,
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
        self.width_multiple = width_multiple
        self.height_multiple = height_multiple

    @legacy(version="0.4.0")
    def _draw_samples(
        self, batch: _BatchInAugmentation, random_state: iarandom.RNG
    ) -> PadToFixedSizeSamplingResult:
        _sizes, pad_xs, pad_ys, pad_modes, pad_cvals = super()._draw_samples(batch, random_state)

        shapes = batch.get_rowwise_shapes()
        sizes = []
        for shape in shapes:
            height, width = shape[0:2]
            paddings = compute_paddings_to_reach_multiples_of(
                shape, height_multiple=self.height_multiple, width_multiple=self.width_multiple
            )

            # TODO change that
            # note that these are not in the same order as shape tuples
            # in PadToFixedSize
            new_size = (width + paddings[1] + paddings[3], height + paddings[0] + paddings[2])
            sizes.append(new_size)

        return sizes, pad_xs, pad_ys, pad_modes, pad_cvals

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [
            self.width_multiple,
            self.height_multiple,
            self.pad_mode,
            self.pad_cval,
            self.position,
        ]


@legacy(version="0.4.0")
class CenterPadToMultiplesOf(PadToMultiplesOf):
    """Pad images equally on all sides until H/W are multiples of given values.

    This is the same as :class:`~imgaug2.augmenters.size.PadToMultiplesOf`, but
    uses ``position="center"`` by default, which spreads the pad amounts
    equally over all image sides, while
    :class:`~imgaug2.augmenters.size.PadToMultiplesOf` by default spreads them
    randomly.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width_multiple : int or None
        See :func:`PadToMultiplesOf.__init__`.

    height_multiple : int or None
        See :func:`PadToMultiplesOf.__init__`.

    pad_mode : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        See :func:`~imgaug2.augmenters.size.PadToMultiplesOf.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :func:`~imgaug2.augmenters.size.PadToMultiplesOf.__init__`.

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
    >>> aug = iaa.CenterPadToMultiplesOf(height_multiple=10, width_multiple=6)

    Create an augmenter that pads images to multiples of ``10`` along
    the y-axis (i.e. 10, 20, 30, ...) and to multiples of ``6`` along the
    x-axis (i.e. 6, 12, 18, ...).
    The rows to be padded will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        width_multiple: int | None,
        height_multiple: int | None,
        pad_mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        pad_cval: ParamInput = 0,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            width_multiple=width_multiple,
            height_multiple=height_multiple,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            position="center",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


