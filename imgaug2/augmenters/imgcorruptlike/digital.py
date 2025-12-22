from __future__ import annotations

from typing import Literal

from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy

from .base import _ImgcorruptAugmenterBase
from .core import _call_imgcorrupt_func


@legacy(version="0.4.0")
def apply_contrast(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``contrast`` from ``imagecorruptions``.


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
        "contrast",
        seed,
        False,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
def apply_brightness(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``brightness`` from ``imagecorruptions``.


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
        "brightness",
        seed,
        False,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
def apply_saturate(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``saturate`` from ``imagecorruptions``.


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
        "saturate",
        seed,
        False,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
def apply_jpeg_compression(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``jpeg_compression`` from ``imagecorruptions``.


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
        "jpeg_compression",
        seed,
        True,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
def apply_pixelate(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``pixelate`` from ``imagecorruptions``.


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
        "pixelate",
        seed,
        True,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
class Contrast(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.contrast``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_contrast`.

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
    >>> aug = iaa.imgcorruptlike.Contrast(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.contrast``.
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
            apply_contrast,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class Brightness(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.brightness``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_brightness`.

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
    >>> aug = iaa.imgcorruptlike.Brightness(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.brightness``.
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
            apply_brightness,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class Saturate(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.saturate``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_saturate`.

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
    >>> aug = iaa.imgcorruptlike.Saturate(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.saturate``.
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
            apply_saturate,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class JpegCompression(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.jpeg_compression``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_jpeg_compression`.

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
    >>> aug = iaa.imgcorruptlike.JpegCompression(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.jpeg_compression``.
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
            apply_jpeg_compression,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class Pixelate(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.pixelate``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_pixelate`.

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
    >>> aug = iaa.imgcorruptlike.Pixelate(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.pixelate``.
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
            apply_pixelate,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

