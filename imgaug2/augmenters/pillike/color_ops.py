from __future__ import annotations

from typing import Literal

import imgaug2.augmenters.arithmetic as arithmetic
import imgaug2.augmenters.color as colorlib
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy


@legacy(version="0.4.0")
def solarize_(image: Array, threshold: int = 128) -> Array:
    """Invert all array components above a threshold in-place.

    This function has identical outputs to ``PIL.ImageOps.solarize``.
    It does however work in-place.


    **Supported dtypes**:

    See ``~imgaug2.augmenters.arithmetic.invert_(min_value=None and max_value=None)``.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.
        The array *might* be modified in-place.

    threshold : int, optional
        A threshold to use in order to invert only numbers above or below
        the threshold.

    Returns
    -------
    ndarray
        Inverted image.
        This *can* be the same array as input in `image`, modified in-place.

    """
    return arithmetic.invert_(image, threshold=threshold)




@legacy(version="0.4.0")
def solarize(image: Array, threshold: int = 128) -> Array:
    """Invert all array components above a threshold.

    This function has identical outputs to ``PIL.ImageOps.solarize``.


    **Supported dtypes**:

    See ``~imgaug2.augmenters.arithmetic.invert_(min_value=None and max_value=None)``.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.

    threshold : int, optional
        A threshold to use in order to invert only numbers above or below
        the threshold.

    Returns
    -------
    ndarray
        Inverted image.

    """
    return arithmetic.invert(image, threshold=threshold)




@legacy(version="0.4.0")
def posterize_(image: Array, bits: int) -> Array:
    """Reduce the number of bits for each color channel in-place.

    This function has identical outputs to ``PIL.ImageOps.posterize``.
    It does however work in-place.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.color.quantize_uniform_to_n_bits_`.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.

    bits : int
        The number of bits to keep per component.
        Values in the interval ``[1, 8]`` are valid.

    Returns
    -------
    ndarray
        Posterized image.
        This *can* be the same array as input in `image`, modified in-place.

    """
    return colorlib.posterize(image, bits)




@legacy(version="0.4.0")
def posterize(image: Array, bits: int) -> Array:
    """Reduce the number of bits for each color channel.

    This function has identical outputs to ``PIL.ImageOps.posterize``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.color.quantize_uniform_to_n_bits`.

    Parameters
    ----------
    image : ndarray
        Image array of shape ``(H,W,[C])``.

    bits : int
        The number of bits to keep per component.
        Values in the interval ``[1, 8]`` are valid.

    Returns
    -------
    ndarray
        Posterized image.

    """
    return colorlib.posterize(image, bits)




@legacy(version="0.4.0")
class Solarize(arithmetic.Invert):
    """Augmenter with identical outputs to PIL's ``solarize()`` function.

    This augmenter inverts all pixel values above a threshold.

    The outputs are identical to PIL's ``solarize()``.


    **Supported dtypes**:

    See ``~imgaug2.augmenters.arithmetic.invert_(min_value=None and max_value=None)``.

    Parameters
    ----------
    p : float or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.arithmetic.Invert`.

    threshold : None or number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.arithmetic.Invert`.

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
    >>> aug = iaa.Solarize(0.5, threshold=(32, 128))

    Invert the colors in ``50`` percent of all images for pixels with a
    value between ``32`` and ``128`` or more. The threshold is sampled once
    per image. The thresholding operation happens per channel.

    """

    def __init__(
        self,
        p: ParamInput = 1.0,
        threshold: ParamInput = 128,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            p=p,
            per_channel=False,
            min_value=None,
            max_value=None,
            threshold=threshold,
            invert_above_threshold=True,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class Posterize(colorlib.Posterize):
    """Augmenter with identical outputs to PIL's ``posterize()`` function.

    This augmenter quantizes each array component to ``N`` bits.

    This class is currently an alias for
    :class:`~imgaug2.augmenters.color.Posterize`, which again is an alias
    for :class:`~imgaug2.augmenters.color.UniformColorQuantizationToNBits`,
    i.e. all three classes are right now guarantueed to have the same
    outputs as PIL's function.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.color.Posterize`.

    """


