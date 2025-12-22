from __future__ import annotations

from typing import Literal

from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy

from .base import _ImgcorruptAugmenterBase
from .core import _call_imgcorrupt_func


@legacy(version="0.4.0")
def apply_fog(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``fog`` from ``imagecorruptions``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    allow_cpu_fallback : bool, optional
        If True, allow CPU fallback for MLX inputs. This performs a
        host<->device roundtrip.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func(
        "fog",
        seed,
        False,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
def apply_frost(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``frost`` from ``imagecorruptions``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    allow_cpu_fallback : bool, optional
        If True, allow CPU fallback for MLX inputs. This performs a
        host<->device roundtrip.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func(
        "frost",
        seed,
        False,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
def apply_snow(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``snow`` from ``imagecorruptions``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    allow_cpu_fallback : bool, optional
        If True, allow CPU fallback for MLX inputs. This performs a
        host<->device roundtrip.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func(
        "snow",
        seed,
        False,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
def apply_spatter(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``spatter`` from ``imagecorruptions``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    allow_cpu_fallback : bool, optional
        If True, allow CPU fallback for MLX inputs. This performs a
        host<->device roundtrip.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func(
        "spatter",
        seed,
        True,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
class Fog(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.fog``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_fog`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

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
    >>> # doctest: +SKIP
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.imgcorruptlike.Fog(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.fog``.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        severity: ParamInput = (1, 5),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            apply_fog,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class Frost(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.frost``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_frost`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

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
    >>> # doctest: +SKIP
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.imgcorruptlike.Frost(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.frost``.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        severity: ParamInput = (1, 5),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            apply_frost,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class Snow(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.snow``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_snow`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

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
    >>> # doctest: +SKIP
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.imgcorruptlike.Snow(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.snow``.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        severity: ParamInput = (1, 5),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            apply_snow,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class Spatter(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.spatter``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_spatter`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

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
    >>> # doctest: +SKIP
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.imgcorruptlike.Spatter(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.spatter``.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        severity: ParamInput = (1, 5),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            apply_spatter,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

