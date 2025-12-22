from __future__ import annotations

from typing import Literal

from imgaug2.compat.markers import legacy
from imgaug2.augmenters._typing import ParamInput, RNGInput

from .base import ChildrenInput
from .mask_generators import HorizontalLinearGradientMaskGen, VerticalLinearGradientMaskGen
from .masks import BlendAlphaMask

@legacy(version="0.4.0")
class BlendAlphaHorizontalLinearGradient(BlendAlphaMask):
    """Blend images from two branches along a horizontal linear gradient.

    This class generates a horizontal linear gradient mask (i.e. usually a
    mask with low values on the left and high values on the right) and
    alphas-blends between foreground and background branch using that
    mask.

    This class is a thin wrapper around
    :class:`~imgaug2.augmenters.blend.BlendAlphaMask` together with
    :class:`~imgaug2.augmenters.blend.HorizontalLinearGradientMaskGen`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug2.augmenters.blend.BlendAlphaMask` for details.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.blend.BlendAlphaMask`.

    Parameters
    ----------
    foreground : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    min_value : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.blend.HorizontalLinearGradientMaskGen`.

    max_value : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.blend.HorizontalLinearGradientMaskGen`.

    start_at : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.blend.HorizontalLinearGradientMaskGen`.

    end_at : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.blend.HorizontalLinearGradientMaskGen`.

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
    >>> aug = iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToHue((-100, 100)))

    Create an augmenter that randomizes the hue towards the right of the
    image.

    >>> aug = iaa.BlendAlphaHorizontalLinearGradient(
    >>>     iaa.TotalDropout(1.0),
    >>>     min_value=0.2, max_value=0.8)

    Create an augmenter that replaces pixels towards the right with darker
    and darker values. However it always keeps at least
    20% (``1.0 - max_value``) of the original pixel value on the far right
    and always replaces at least 20% on the far left (``min_value=0.2``).

    >>> aug = iaa.BlendAlphaHorizontalLinearGradient(
    >>>     iaa.AveragePooling(11),
    >>>     start_at=(0.0, 1.0), end_at=(0.0, 1.0))

    Create an augmenter that blends with an average-pooled image according
    to a horizontal gradient that starts at a random x-coordinate and reaches
    its maximum at another random x-coordinate. Due to that randomness,
    the gradient may increase towards the left or right.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        foreground: ChildrenInput = None,
        background: ChildrenInput = None,
        min_value: ParamInput = (0.0, 0.2),
        max_value: ParamInput = (0.8, 1.0),
        start_at: ParamInput = (0.0, 0.2),
        end_at: ParamInput = (0.8, 1.0),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            HorizontalLinearGradientMaskGen(
                min_value=min_value, max_value=max_value, start_at=start_at, end_at=end_at
            ),
            foreground=foreground,
            background=background,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
class BlendAlphaVerticalLinearGradient(BlendAlphaMask):
    """Blend images from two branches along a vertical linear gradient.

    This class generates a vertical linear gradient mask (i.e. usually a
    mask with low values on the left and high values on the right) and
    alphas-blends between foreground and background branch using that
    mask.

    This class is a thin wrapper around
    :class:`~imgaug2.augmenters.blend.BlendAlphaMask` together with
    :class:`~imgaug2.augmenters.blend.VerticalLinearGradientMaskGen`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug2.augmenters.blend.BlendAlphaMask` for details.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.blend.BlendAlphaMask`.

    Parameters
    ----------
    foreground : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    min_value : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.blend.VerticalLinearGradientMaskGen`.

    max_value : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.blend.VerticalLinearGradientMaskGen`.

    start_at : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.blend.VerticalLinearGradientMaskGen`.

    end_at : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.blend.VerticalLinearGradientMaskGen`.

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
    >>> aug = iaa.BlendAlphaVerticalLinearGradient(iaa.AddToHue((-100, 100)))

    Create an augmenter that randomizes the hue towards the bottom of the
    image.

    >>> aug = iaa.BlendAlphaVerticalLinearGradient(
    >>>     iaa.TotalDropout(1.0),
    >>>     min_value=0.2, max_value=0.8)

    Create an augmenter that replaces pixels towards the bottom with darker
    and darker values. However it always keeps at least
    20% (``1.0 - max_value``) of the original pixel value on the far bottom
    and always replaces at least 20% on the far top (``min_value=0.2``).

    >>> aug = iaa.BlendAlphaVerticalLinearGradient(
    >>>     iaa.AveragePooling(11),
    >>>     start_at=(0.0, 1.0), end_at=(0.0, 1.0))

    Create an augmenter that blends with an average-pooled image according
    to a vertical gradient that starts at a random y-coordinate and reaches
    its maximum at another random y-coordinate. Due to that randomness,
    the gradient may increase towards the bottom or top.

    >>> aug = iaa.BlendAlphaVerticalLinearGradient(
    >>>     iaa.Clouds(),
    >>>     start_at=(0.15, 0.35), end_at=0.0)

    Create an augmenter that draws clouds in roughly the top quarter of the
    image.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        foreground: ChildrenInput = None,
        background: ChildrenInput = None,
        min_value: ParamInput = (0.0, 0.2),
        max_value: ParamInput = (0.8, 1.0),
        start_at: ParamInput = (0.0, 0.2),
        end_at: ParamInput = (0.8, 1.0),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            VerticalLinearGradientMaskGen(
                min_value=min_value, max_value=max_value, start_at=start_at, end_at=end_at
            ),
            foreground=foreground,
            background=background,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


