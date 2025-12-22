"""Square-size augmenters."""

from __future__ import annotations

from typing import Literal

import imgaug2.parameters as iap
from imgaug2.augmenters._typing import ParamInput, RNGInput
from imgaug2.compat.markers import legacy

from .aspect_ratio import CropToAspectRatio, PadToAspectRatio
@legacy(version="0.4.0")
class CropToSquare(CropToAspectRatio):
    """Crop images until their width and height are identical.

    This is identical to :class:`~imgaug2.augmenters.size.CropToAspectRatio`
    with ``aspect_ratio=1.0``.

    Images with axis sizes of ``0`` will not be altered.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
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
    >>> aug = iaa.CropToSquare()

    Create an augmenter that crops each image until its square, i.e. height
    and width match.
    The rows to be cropped will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        position: str
        | tuple[float | int | iap.StochasticParameter, float | int | iap.StochasticParameter]
        | iap.StochasticParameter = "uniform",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            aspect_ratio=1.0,
            position=position,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
class CenterCropToSquare(CropToSquare):
    """Crop images equally on all sides until their height/width are identical.

    In contrast to :class:`~imgaug2.augmenters.size.CropToSquare`, this
    augmenter always tries to spread the columns/rows to remove equally over
    both sides of the respective axis to be cropped.
    :class:`~imgaug2.augmenters.size.CropToAspectRatio` by default spreads the
    croppings randomly.

    This augmenter is identical to :class:`~imgaug2.augmenters.size.CropToSquare`
    with ``position="center"``, and thereby the same as
    :class:`~imgaug2.augmenters.size.CropToAspectRatio` with
    ``aspect_ratio=1.0, position="center"``.

    Images with axis sizes of ``0`` will not be altered.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
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
    >>> aug = iaa.CenterCropToSquare()

    Create an augmenter that crops each image until its square, i.e. height
    and width match.
    The rows to be cropped will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            position="center",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
class PadToSquare(PadToAspectRatio):
    """Pad images until their height and width are identical.

    This augmenter is identical to
    :class:`~imgaug2.augmenters.size.PadToAspectRatio` with ``aspect_ratio=1.0``.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
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
    >>> aug = iaa.PadToSquare()

    Create an augmenter that pads each image until its square, i.e. height
    and width match.
    The rows to be padded will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
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
            aspect_ratio=1.0,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            position=position,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
class CenterPadToSquare(PadToSquare):
    """Pad images equally on all sides until their height & width are identical.

    This is the same as :class:`~imgaug2.augmenters.size.PadToSquare`, but uses
    ``position="center"`` by default, which spreads the pad amounts equally
    over all image sides, while :class:`~imgaug2.augmenters.size.PadToSquare`
    by default spreads them randomly. This augmenter is thus also identical to
    :class:`~imgaug2.augmenters.size.PadToAspectRatio` with
    ``aspect_ratio=1.0, position="center"``.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
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
    >>> aug = iaa.CenterPadToSquare()

    Create an augmenter that pads each image until its square, i.e. height
    and width match.
    The rows to be padded will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        pad_mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        pad_cval: ParamInput = 0,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            position="center",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


