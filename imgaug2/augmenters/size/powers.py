"""Power-of-base size augmenters."""

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
    compute_croppings_to_reach_powers_of,
    compute_paddings_to_reach_powers_of,
)
from .fixed_size import CropToFixedSize, PadToFixedSize
@legacy(version="0.4.0")
class CropToPowersOf(CropToFixedSize):
    """Crop images until their height/width is a power of a base.

    This augmenter removes pixels from an axis with size ``S`` leading to the
    new size ``S'`` until ``S' = B^E`` is fulfilled, where ``B`` is a
    provided base (e.g. ``2``) and ``E`` is an exponent from the discrete
    interval ``[1 .. inf)``.

    .. note::

        This augmenter does nothing for axes with size less than ``B^1 = B``.
        If you have images with ``S < B^1``, it is recommended
        to combine this augmenter with a padding augmenter that pads each
        axis up to ``B``.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width_base : int or None
        Base for the width. Images will be cropped down until their
        width fulfills ``width' = width_base ^ E`` with ``E`` being any
        natural number.
        If ``None``, image widths will not be altered.

    height_base : int or None
        Base for the height. Images will be cropped down until their
        height fulfills ``height' = height_base ^ E`` with ``E`` being any
        natural number.
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
    >>> aug = iaa.CropToPowersOf(height_base=3, width_base=2)

    Create an augmenter that crops each image down to powers of ``3`` along
    the y-axis (i.e. 3, 9, 27, ...) and powers of ``2`` along the x-axis (i.e.
    2, 4, 8, 16, ...).
    The rows to be cropped will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        width_base: int | None,
        height_base: int | None,
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
        self.width_base = width_base
        self.height_base = height_base

    @legacy(version="0.4.0")
    def _draw_samples(
        self, batch: _BatchInAugmentation, random_state: iarandom.RNG
    ) -> CropToFixedSizeSamplingResult:
        _sizes, offset_xs, offset_ys = super()._draw_samples(batch, random_state)

        shapes = batch.get_rowwise_shapes()
        sizes = []
        for shape in shapes:
            height, width = shape[0:2]
            croppings = compute_croppings_to_reach_powers_of(
                shape, height_base=self.height_base, width_base=self.width_base
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
        return [self.width_base, self.height_base, self.position]


@legacy(version="0.4.0")
class CenterCropToPowersOf(CropToPowersOf):
    """Crop images equally on all sides until H/W is a power of a base.

    This is the same as :class:`~imgaug2.augmenters.size.CropToPowersOf`, but
    uses ``position="center"`` by default, which spreads the crop amounts
    equally over all image sides, while
    :class:`~imgaug2.augmenters.size.CropToPowersOf` by default spreads them
    randomly.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width_base : int or None
        See :func:`CropToPowersOf.__init__`.

    height_base : int or None
        See :func:`CropToPowersOf.__init__`.

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
    >>> aug = iaa.CropToPowersOf(height_base=3, width_base=2)

    Create an augmenter that crops each image down to powers of ``3`` along
    the y-axis (i.e. 3, 9, 27, ...) and powers of ``2`` along the x-axis (i.e.
    2, 4, 8, 16, ...).
    The rows to be cropped will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        width_base: int | None,
        height_base: int | None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            width_base=width_base,
            height_base=height_base,
            position="center",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
class PadToPowersOf(PadToFixedSize):
    """Pad images until their height/width is a power of a base.

    This augmenter adds pixels to an axis with size ``S`` leading to the
    new size ``S'`` until ``S' = B^E`` is fulfilled, where ``B`` is a
    provided base (e.g. ``2``) and ``E`` is an exponent from the discrete
    interval ``[1 .. inf)``.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width_base : int or None
        Base for the width. Images will be padded down until their
        width fulfills ``width' = width_base ^ E`` with ``E`` being any
        natural number.
        If ``None``, image widths will not be altered.

    height_base : int or None
        Base for the height. Images will be padded until their
        height fulfills ``height' = height_base ^ E`` with ``E`` being any
        natural number.
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
    >>> aug = iaa.PadToPowersOf(height_base=3, width_base=2)

    Create an augmenter that pads each image to powers of ``3`` along the
    y-axis (i.e. 3, 9, 27, ...) and powers of ``2`` along the x-axis (i.e. 2,
    4, 8, 16, ...).
    The rows to be padded will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        width_base: int | None,
        height_base: int | None,
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
        self.width_base = width_base
        self.height_base = height_base

    @legacy(version="0.4.0")
    def _draw_samples(
        self, batch: _BatchInAugmentation, random_state: iarandom.RNG
    ) -> PadToFixedSizeSamplingResult:
        _sizes, pad_xs, pad_ys, pad_modes, pad_cvals = super()._draw_samples(batch, random_state)

        shapes = batch.get_rowwise_shapes()
        sizes = []
        for shape in shapes:
            height, width = shape[0:2]
            paddings = compute_paddings_to_reach_powers_of(
                shape, height_base=self.height_base, width_base=self.width_base
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
        return [self.width_base, self.height_base, self.pad_mode, self.pad_cval, self.position]


@legacy(version="0.4.0")
class CenterPadToPowersOf(PadToPowersOf):
    """Pad images equally on all sides until H/W is a power of a base.

    This is the same as :class:`~imgaug2.augmenters.size.PadToPowersOf`, but uses
    ``position="center"`` by default, which spreads the pad amounts equally
    over all image sides, while :class:`~imgaug2.augmenters.size.PadToPowersOf`
    by default spreads them randomly.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width_base : int or None
        See :func:`PadToPowersOf.__init__`.

    height_base : int or None
        See :func:`PadToPowersOf.__init__`.

    pad_mode : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        See :func:`~imgaug2.augmenters.size.PadToPowersOf.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :func:`~imgaug2.augmenters.size.PadToPowersOf.__init__`.

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
    >>> aug = iaa.CenterPadToPowersOf(height_base=5, width_base=2)

    Create an augmenter that pads each image to powers of ``3`` along the
    y-axis (i.e. 3, 9, 27, ...) and powers of ``2`` along the x-axis (i.e. 2,
    4, 8, 16, ...).
    The rows to be padded will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        width_base: int | None,
        height_base: int | None,
        pad_mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        pad_cval: ParamInput = 0,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            width_base=width_base,
            height_base=height_base,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            position="center",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


