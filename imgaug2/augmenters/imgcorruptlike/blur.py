from __future__ import annotations

from typing import Literal

import numpy as np

from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _numbajit

from ._types import IntArray, UIntArray
from .base import _ImgcorruptAugmenterBase
from .core import _call_imgcorrupt_func, _gaussian_skimage_compat


@legacy(version="0.4.0")
def apply_gaussian_blur(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``gaussian_blur`` from ``imagecorruptions``.


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
        "gaussian_blur",
        seed,
        False,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
def apply_glass_blur(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``glass_blur`` from ``imagecorruptions``.


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
        _apply_glass_blur_imgaug,
        seed,
        False,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
def _apply_glass_blur_imgaug(x: Array, severity: int = 1) -> Array:
    # false positive on x_shape[0]
    # invalid name for dx, dy

    # original function implementation from
    # https://github.com/bethgelab/imagecorruptions/blob/master/imagecorruptions/corruptions.py
    # this is an improved (i.e. faster) version
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    sigma, max_delta, iterations = c

    x = (
        _gaussian_skimage_compat(np.array(x) / 255.0, sigma=sigma, multichannel=True) * 255
    ).astype(np.uint)
    x_shape = x.shape

    dxxdyy = np.random.randint(
        -max_delta,
        max_delta,
        size=(iterations, x_shape[0] - 2 * max_delta, x_shape[1] - 2 * max_delta, 2),
    )

    x = _apply_glass_blur_imgaug_loop(x, iterations, max_delta, dxxdyy)

    return np.clip(_gaussian_skimage_compat(x / 255.0, sigma=sigma, multichannel=True), 0, 1) * 255




@legacy(version="0.5.0")
@_numbajit(nopython=True, nogil=True, cache=True)
def _apply_glass_blur_imgaug_loop(
    x: UIntArray, iterations: int, max_delta: int, dxxdyy: IntArray
) -> UIntArray:
    x_shape = x.shape
    nb_height = x_shape[0] - 2 * max_delta
    nb_width = x_shape[1] - 2 * max_delta

    # locally shuffle pixels
    for i in range(iterations):
        for j in range(nb_height):
            for k in range(nb_width):
                h = x_shape[0] - max_delta - j
                w = x_shape[1] - max_delta - k
                dx, dy = dxxdyy[i, j, k]
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return x




@legacy(version="0.4.0")
def apply_defocus_blur(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``defocus_blur`` from ``imagecorruptions``.


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
        "defocus_blur",
        seed,
        False,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
def apply_motion_blur(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``motion_blur`` from ``imagecorruptions``.


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
        "motion_blur",
        seed,
        False,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
def apply_zoom_blur(
    x: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``zoom_blur`` from ``imagecorruptions``.


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
        "zoom_blur",
        seed,
        False,
        x,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )




@legacy(version="0.4.0")
class GaussianBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.gaussian_blur``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_gaussian_blur`.

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
    >>> aug = iaa.imgcorruptlike.GaussianBlur(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.gaussian_blur``.
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
            apply_gaussian_blur,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class GlassBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.glass_blur``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_glass_blur`.

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
    >>> aug = iaa.imgcorruptlike.GlassBlur(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.glass_blur``.
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
            apply_glass_blur,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class DefocusBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.defocus_blur``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_defocus_blur`.

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
    >>> aug = iaa.imgcorruptlike.DefocusBlur(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.defocus_blur``.
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
            apply_defocus_blur,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class MotionBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.motion_blur``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_motion_blur`.

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
    >>> aug = iaa.imgcorruptlike.MotionBlur(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.motion_blur``.
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
            apply_motion_blur,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class ZoomBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.zoom_blur``.

    .. note::

        This augmenter only affects images. Other data is not changed.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_zoom_blur`.

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
    >>> aug = iaa.imgcorruptlike.ZoomBlur(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.zoom_blur``.
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
            apply_zoom_blur,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

