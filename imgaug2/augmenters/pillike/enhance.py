from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
import PIL.Image
import PIL.ImageEnhance

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
import imgaug2.augmenters.meta as meta
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy

from ._utils import _ensure_valid_shape, _maybe_mlx


@legacy(version="0.4.0")
class _EnhanceCtor(Protocol):
    def __call__(self, image: PIL.Image.Image) -> PIL.ImageEnhance._Enhance: ...




def _apply_enhance_func(image: Array, cls: _EnhanceCtor, factor: float) -> Array:
    iadt.allow_only_uint8({image.dtype})

    if 0 in image.shape:
        return np.copy(image)

    image, is_hw1 = _ensure_valid_shape(image, "imgaug2.augmenters.pillike.enhance_*()")

    # don't return np.asarray(...) as its results are read-only
    result = np.array(cls(PIL.Image.fromarray(image)).enhance(factor))
    if is_hw1:
        result = result[:, :, np.newaxis]
    return result




@legacy(version="0.4.0")
def enhance_color(image: Array, factor: float) -> Array:
    """Change the strength of colors in an image.

    This function has identical outputs to
    ``PIL.ImageEnhance.Color``.


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    factor : number
        Colorfulness of the output image. Values close to ``0.0`` lead
        to grayscale images, values above ``1.0`` increase the strength of
        colors. Sane values are roughly in ``[0.0, 3.0]``.

    Returns
    -------
    ndarray
        Color-modified image.

    """
    maybe = _maybe_mlx(image, "enhance_color", factor=factor)
    if maybe is not None:
        return maybe

    # PIL.ImageEnhance.Color supports images with shape (H,W), (H,W,1), (H,W,3)
    # and (H,W,4). For other channel counts, we provide a best-effort fallback
    # to keep the function usable (and at least preserve shape/dtype).
    if image.ndim == 3 and image.shape[-1] not in (1, 3, 4):
        iadt.allow_only_uint8({image.dtype})
        if 0 in image.shape:
            return np.copy(image)

        nb_channels = image.shape[-1]
        # Can't interpret two-channel images as RGB(A) for color enhancement.
        # Return a copy to preserve dtype/shape.
        if nb_channels < 3:
            return np.copy(image)

        # Enhance first three channels as RGB, keep remaining channels unchanged.
        rgb = _apply_enhance_func(image[..., :3], PIL.ImageEnhance.Color, factor)
        if nb_channels == 3:
            return rgb
        rest = np.copy(image[..., 3:])
        return np.concatenate([rgb, rest], axis=2)

    return _apply_enhance_func(image, PIL.ImageEnhance.Color, factor)




@legacy(version="0.4.0")
def enhance_contrast(image: Array, factor: float) -> Array:
    """Change the contrast of an image.

    This function has identical outputs to
    ``PIL.ImageEnhance.Contrast``.


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    factor : number
        Strength of contrast in the image. Values below ``1.0`` decrease the
        contrast, leading to a gray image around ``0.0``. Values
        above ``1.0`` increase the contrast. Sane values are roughly in
        ``[0.5, 1.5]``.

    Returns
    -------
    ndarray
        Contrast-modified image.

    """
    maybe = _maybe_mlx(image, "enhance_contrast", factor=factor)
    if maybe is not None:
        return maybe

    return _apply_enhance_func(image, PIL.ImageEnhance.Contrast, factor)




@legacy(version="0.4.0")
def enhance_brightness(image: Array, factor: float) -> Array:
    """Change the brightness of images.

    This function has identical outputs to
    ``PIL.ImageEnhance.Brightness``.


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    factor : number
        Brightness of the image. Values below ``1.0`` decrease the brightness,
        leading to a black image around ``0.0``. Values above ``1.0`` increase
        the brightness. Sane values are roughly in ``[0.5, 1.5]``.

    Returns
    -------
    ndarray
        Brightness-modified image.

    """
    maybe = _maybe_mlx(image, "enhance_brightness", factor=factor)
    if maybe is not None:
        return maybe

    return _apply_enhance_func(image, PIL.ImageEnhance.Brightness, factor)




@legacy(version="0.4.0")
def enhance_sharpness(image: Array, factor: float) -> Array:
    """Change the sharpness of an image.

    This function has identical outputs to
    ``PIL.ImageEnhance.Sharpness``.


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to modify.

    factor : number
        Sharpness of the image. Values below ``1.0`` decrease the sharpness,
        values above ``1.0`` increase it. Sane values are roughly in
        ``[0.0, 2.0]``.

    Returns
    -------
    ndarray
        Sharpness-modified image.

    """
    maybe = _maybe_mlx(image, "enhance_sharpness", factor=factor)
    if maybe is not None:
        return maybe

    return _apply_enhance_func(image, PIL.ImageEnhance.Sharpness, factor)




@legacy(version="0.4.0")
class _EnhanceBase(meta.Augmenter):
    @legacy(version="0.4.0")
    def __init__(
        self,
        func: object,
        factor: ParamInput,
        factor_value_range: tuple[float | None, float | None],
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.func = func
        self.factor = iap.handle_continuous_param(
            factor,
            "factor",
            value_range=factor_value_range,
            tuple_to_uniform=True,
            list_to_choice=True,
        )

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

        factors = self._draw_samples(len(batch.images), random_state)
        for image, factor in zip(batch.images, factors, strict=True):
            image[...] = self.func(image, factor)
        return batch

    @legacy(version="0.4.0")
    def _draw_samples(self, nb_rows: int, random_state: iarandom.RNG) -> Array:
        return self.factor.draw_samples((nb_rows,), random_state=random_state)

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.factor]




@legacy(version="0.4.0")
class EnhanceColor(_EnhanceBase):
    """Convert images to grayscale.

    This augmenter has identical outputs to ``PIL.ImageEnhance.Color``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.enhance_color`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Colorfulness of the output image. Values close to ``0.0`` lead
        to grayscale images, values above ``1.0`` increase the strength of
        colors. Sane values are roughly in ``[0.0, 3.0]``.

            * If ``number``: The value will be used for all images.
            * If ``tuple`` ``(a, b)``: A value will be uniformly sampled per
              image from the interval ``[a, b)``.
            * If ``list``: A random value will be picked from the list per
              image.
            * If ``StochasticParameter``: Per batch of size ``N``, the
              parameter will be queried once to return ``(N,)`` samples.

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
    >>> aug = iaa.pillike.EnhanceColor()

    Create an augmenter to remove a random fraction of color from
    input images.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        factor: ParamInput = (0.0, 3.0),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        from imgaug2.augmenters import pillike as pillike_lib

        super().__init__(
            func=pillike_lib.enhance_color,
            factor=factor,
            factor_value_range=(0.0, None),
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class EnhanceContrast(_EnhanceBase):
    """Change the contrast of images.

    This augmenter has identical outputs to ``PIL.ImageEnhance.Contrast``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.enhance_contrast`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Strength of contrast in the image. Values below ``1.0`` decrease the
        contrast, leading to a gray image around ``0.0``. Values
        above ``1.0`` increase the contrast. Sane values are roughly in
        ``[0.5, 1.5]``.

            * If ``number``: The value will be used for all images.
            * If ``tuple`` ``(a, b)``: A value will be uniformly sampled per
              image from the interval ``[a, b)``.
            * If ``list``: A random value will be picked from the list per
              image.
            * If ``StochasticParameter``: Per batch of size ``N``, the
              parameter will be queried once to return ``(N,)`` samples.

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
    >>> aug = iaa.pillike.EnhanceContrast()

    Create an augmenter that worsens the contrast of an image by a random
    factor.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        factor: ParamInput = (0.5, 1.5),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        from imgaug2.augmenters import pillike as pillike_lib

        super().__init__(
            func=pillike_lib.enhance_contrast,
            factor=factor,
            factor_value_range=(0.0, None),
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class EnhanceBrightness(_EnhanceBase):
    """Change the brightness of images.

    This augmenter has identical outputs to
    ``PIL.ImageEnhance.Brightness``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.enhance_brightness`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Brightness of the image. Values below ``1.0`` decrease the brightness,
        leading to a black image around ``0.0``. Values above ``1.0`` increase
        the brightness. Sane values are roughly in ``[0.5, 1.5]``.

            * If ``number``: The value will be used for all images.
            * If ``tuple`` ``(a, b)``: A value will be uniformly sampled per
              image from the interval ``[a, b)``.
            * If ``list``: A random value will be picked from the list per
              image.
            * If ``StochasticParameter``: Per batch of size ``N``, the
              parameter will be queried once to return ``(N,)`` samples.

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
    >>> aug = iaa.pillike.EnhanceBrightness()

    Create an augmenter that worsens the brightness of an image by a random
    factor.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        factor: ParamInput = (0.5, 1.5),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        from imgaug2.augmenters import pillike as pillike_lib

        super().__init__(
            func=pillike_lib.enhance_brightness,
            factor=factor,
            factor_value_range=(0.0, None),
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class EnhanceSharpness(_EnhanceBase):
    """Change the sharpness of images.

    This augmenter has identical outputs to
    ``PIL.ImageEnhance.Sharpness``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.enhance_sharpness`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Sharpness of the image. Values below ``1.0`` decrease the sharpness,
        values above ``1.0`` increase it. Sane values are roughly in
        ``[0.0, 2.0]``.

            * If ``number``: The value will be used for all images.
            * If ``tuple`` ``(a, b)``: A value will be uniformly sampled per
              image from the interval ``[a, b)``.
            * If ``list``: A random value will be picked from the list per
              image.
            * If ``StochasticParameter``: Per batch of size ``N``, the
              parameter will be queried once to return ``(N,)`` samples.

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
    >>> aug = iaa.pillike.EnhanceSharpness()

    Create an augmenter that randomly decreases or increases the sharpness
    of an image.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        factor: ParamInput = (0.0, 2.0),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        from imgaug2.augmenters import pillike as pillike_lib

        super().__init__(
            func=pillike_lib.enhance_sharpness,
            factor=factor,
            factor_value_range=(0.0, None),
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

