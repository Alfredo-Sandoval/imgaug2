from __future__ import annotations

from typing import Literal

from imgaug2.compat.markers import legacy
from imgaug2.augmenters._typing import ParamInput, RNGInput

from .base import ChildrenInput
from .mask_generators import CheckerboardMaskGen, RegularGridMaskGen
from .masks import BlendAlphaMask

@legacy(version="0.4.0")
class BlendAlphaRegularGrid(BlendAlphaMask):
    """Blend images from two branches according to a regular grid.

    This class generates for each image a mask that splits the image into a
    grid-like pattern of ``H`` rows and ``W`` columns. Each cell is then
    filled with an alpha value, sampled randomly per cell.

    The difference to :class:`AlphaBlendCheckerboard` is that this class
    samples random alpha values per grid cell, while in the checkerboard the
    alpha values follow a fixed pattern.

    This class is a thin wrapper around
    :class:`~imgaug2.augmenters.blend.BlendAlphaMask` together with
    :class:`~imgaug2.augmenters.blend.RegularGridMaskGen`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug2.augmenters.blend.BlendAlphaMask` for details.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.blend.BlendAlphaMask`.

    Parameters
    ----------
    nb_rows : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter
        Number of rows of the checkerboard.
        See :class:`~imgaug2.augmenters.blend.CheckerboardMaskGen` for details.

    nb_cols : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter
        Number of columns of the checkerboard. Analogous to `nb_rows`.
        See :class:`~imgaug2.augmenters.blend.CheckerboardMaskGen` for details.

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

    alpha : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Alpha value of each cell.

        * If ``number``: Exactly that value will be used for all images.
        * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
          per image from the interval ``[a, b]``.
        * If ``list``: A random value will be picked per image from that list.
        * If ``StochasticParameter``: That parameter will be queried once
          per batch for ``(N,)`` values -- one per image.

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
    >>> aug = iaa.BlendAlphaRegularGrid(nb_rows=(4, 6), nb_cols=(1, 4),
    >>>                                 foreground=iaa.Multiply(0.0))

    Create an augmenter that places a ``HxW`` grid on each image, where
    ``H`` (rows) is randomly and uniformly sampled from the interval ``[4, 6]``
    and ``W`` is analogously sampled from the interval ``[1, 4]``. Roughly
    half of the cells in the grid are filled with ``0.0``, the remaining ones
    are unaltered. Which cells exactly are "dropped" is randomly decided
    per image. The resulting effect is similar to
    :class:`~imgaug2.augmenters.arithmetic.CoarseDropout`.

    >>> aug = iaa.BlendAlphaRegularGrid(nb_rows=2, nb_cols=2,
    >>>                                 foreground=iaa.Multiply(0.0),
    >>>                                 background=iaa.AveragePooling(8),
    >>>                                 alpha=[0.0, 0.0, 1.0])

    Create an augmenter that always placed ``2x2`` cells on each image
    and sets about ``1/3`` of them to zero (foreground branch) and
    the remaining ``2/3`` to a pixelated version (background branch).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        nb_rows: ParamInput,
        nb_cols: ParamInput,
        foreground: ChildrenInput = None,
        background: ChildrenInput = None,
        alpha: ParamInput | None = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            RegularGridMaskGen(
                nb_rows=nb_rows, nb_cols=nb_cols, alpha=alpha if alpha is not None else [0.0, 1.0]
            ),
            foreground=foreground,
            background=background,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
class BlendAlphaCheckerboard(BlendAlphaMask):
    """Blend images from two branches according to a checkerboard pattern.

    This class generates for each image a mask following a checkboard layout of
    ``H`` rows and ``W`` columns. Each cell is then filled with either
    ``1.0`` or ``0.0``. The cell at the top-left is always ``1.0``. Its right
    and bottom neighbour cells are ``0.0``. The 4-neighbours of any cell always
    have a value opposite to the cell's value (``0.0`` vs. ``1.0``).

    This class is a thin wrapper around
    :class:`~imgaug2.augmenters.blend.BlendAlphaMask` together with
    :class:`~imgaug2.augmenters.blend.CheckerboardMaskGen`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug2.augmenters.blend.BlendAlphaMask` for details.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.blend.BlendAlphaMask`.

    Parameters
    ----------
    nb_rows : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter
        Number of rows of the checkerboard.
        See :class:`~imgaug2.augmenters.blend.CheckerboardMaskGen` for details.

    nb_cols : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter
        Number of columns of the checkerboard. Analogous to `nb_rows`.
        See :class:`~imgaug2.augmenters.blend.CheckerboardMaskGen` for details.

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
    >>> aug = iaa.BlendAlphaCheckerboard(nb_rows=2, nb_cols=(1, 4),
    >>>                                  foreground=iaa.AddToHue((-100, 100)))

    Create an augmenter that places a ``HxW`` grid on each image, where
    ``H`` (rows) is always ``2`` and ``W`` is randomly and uniformly sampled
    from the interval ``[1, 4]``. For half of the cells in the grid the hue
    is randomly modified, the other half of the cells is unaltered.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        nb_rows: ParamInput,
        nb_cols: ParamInput,
        foreground: ChildrenInput = None,
        background: ChildrenInput = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            CheckerboardMaskGen(nb_rows=nb_rows, nb_cols=nb_cols),
            foreground=foreground,
            background=background,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


