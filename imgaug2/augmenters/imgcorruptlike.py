"""
Augmenters that wrap methods from ``imagecorruptions`` package.

See `https://github.com/bethgelab/imagecorruptions`_ for the package.

The package is derived from `https://github.com/hendrycks/robustness`_.
The corresponding `paper <https://arxiv.org/abs/1807.01697>`_ is::

    Hendrycks, Dan and Dietterich, Thomas G.
    Benchmarking Neural Network Robustness to Common Corruptions and
    Surface Variations

with the `newer version <https://arxiv.org/abs/1903.12261>`_ being::

    Hendrycks, Dan and Dietterich, Thomas G.
    Benchmarking Neural Network Robustness to Common Corruptions and
    Perturbations

List of augmenters:

    * :class:`GaussianNoise`
    * :class:`ShotNoise`
    * :class:`ImpulseNoise`
    * :class:`SpeckleNoise`
    * :class:`GaussianBlur`
    * :class:`GlassBlur`
    * :class:`DefocusBlur`
    * :class:`MotionBlur`
    * :class:`ZoomBlur`
    * :class:`Fog`
    * :class:`Frost`
    * :class:`Snow`
    * :class:`Spatter`
    * :class:`Contrast`
    * :class:`Brightness`
    * :class:`Saturate`
    * :class:`JpegCompression`
    * :class:`Pixelate`
    * :class:`ElasticTransform`

.. note::

    The functions provided here have identical outputs to the ones in
    ``imagecorruptions`` when called using the ``corrupt()`` function of
    that package. E.g. the outputs are always ``uint8`` and not
    ``float32`` or ``float64``.

Example usage::

    >>> # Skip the doctests in this file as the imagecorruptions package is
    >>> # not available in all python versions that are otherwise supported
    >>> # by imgaug2.
    >>> # doctest: +SKIP
    >>> import imgaug2 as ia
    >>> import imgaug2.augmenters as iaa
    >>> import numpy as np
    >>> image = np.zeros((64, 64, 3), dtype=np.uint8)
    >>> names, funcs = iaa.imgcorruptlike.get_corruption_names("validation")
    >>> for name, func in zip(names, funcs):
    >>>     image_aug = func(image, severity=5, seed=1)
    >>>     image_aug = ia.draw_text(image_aug, x=20, y=20, text=name)
    >>>     ia.imshow(image_aug)

    Use e.g. ``iaa.imgcorruptlike.GaussianNoise(severity=2)(images=...)`` to
    create and apply a specific augmenter.

Added in 0.4.0.

"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, Literal, Protocol, TypeAlias, cast

import numpy as np
import skimage.filters
from numpy.typing import NDArray

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.imgaug import _numbajit

# TODO add optional dependency

_MISSING_PACKAGE_ERROR_MSG = (
    "Could not import package `imagecorruptions`. This is an optional "
    "dependency of imgaug and must be installed manually in order "
    "to use augmenters from `imgaug2.augmenters.imgcorrupt`. "
    "Use e.g. `pip install imagecorruptions` to install it. See also "
    "https://github.com/bethgelab/imagecorruptions for the repository "
    "of the package."
)

IntArray: TypeAlias = NDArray[np.integer]
UIntArray: TypeAlias = NDArray[np.unsignedinteger]


class CorruptionFunc(Protocol):
    def __call__(self, x: Array, severity: int = 1, seed: int | None = None) -> Array: ...


# Added in 0.6.0.
def _patch_imagecorruptions_modules_() -> tuple[ModuleType, ModuleType]:
    """Patch `imagecorruptions` for compatibility with newer numpy/skimage.

    The upstream `imagecorruptions` package is an optional dependency and may be
    installed in versions that are not fully compatible with newer libraries.
    The goal here is not to change the corruption algorithms, but to keep them
    usable and deterministic in modern environments (e.g. NumPy>=2, skimage with
    `channel_axis`).
    """
    try:
        with warnings.catch_warnings():
            import imagecorruptions
            import imagecorruptions.corruptions as corruptions
    except ImportError:
        raise ImportError(_MISSING_PACKAGE_ERROR_MSG) from None

    # Patch only once per process.
    if getattr(imagecorruptions, "__imgaug2_patched__", False):
        return imagecorruptions, corruptions

    # NumPy removed legacy scalar aliases like `np.float` / `np.int` / `np.bool`
    # (deprecated since 1.20; removed in 1.24+). Some optional dependencies
    # (including older `imagecorruptions` releases) still reference them.
    #
    # Avoid `hasattr(np, ...)` here as it can trigger numpy's deprecation
    # machinery (and warnings) via `__getattr__`. Checking `np.__dict__` keeps
    # this warning-free.
    legacy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
        "long": int,
    }
    for name, value in legacy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    # skimage.filters.gaussian dropped `multichannel` in favor of `channel_axis`.
    gaussian_sig = inspect.signature(skimage.filters.gaussian)
    supports_channel_axis = "channel_axis" in gaussian_sig.parameters
    if supports_channel_axis:

        def gaussian_compat(image: Array, *args: object, **kwargs: object) -> Array:
            if "multichannel" in kwargs:
                multichannel = kwargs.pop("multichannel")
                kwargs["channel_axis"] = -1 if multichannel else None
            return skimage.filters.gaussian(image, *args, **kwargs)

    else:

        def gaussian_compat(image: Array, *args: object, **kwargs: object) -> Array:
            return skimage.filters.gaussian(image, *args, **kwargs)

    corruptions.gaussian = gaussian_compat

    # skimage.util.random_noise now uses an explicit RNG. Older `imagecorruptions`
    # calls it without an RNG, which makes it non-deterministic even if numpy's
    # global RNG is seeded. Patch impulse_noise to pass a deterministic RNG that
    # is derived from numpy's global RNG (and hence affected by `np.random.seed`).
    def impulse_noise_compat(x: Array, severity: int = 1) -> Array:
        c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
        # Draw a seed from the current global RNG state to remain tied to the
        # existing `temporary_numpy_seed()` behavior.
        seed_local = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
        rng = np.random.default_rng(seed_local)
        x = corruptions.sk.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=c, rng=rng)
        return np.clip(x, 0, 1) * 255

    corruptions.impulse_noise = impulse_noise_compat
    if hasattr(imagecorruptions, "corruption_dict"):
        imagecorruptions.corruption_dict["impulse_noise"] = impulse_noise_compat

    # Keep behavior for clip_zoom(), but avoid scipy warning spam.
    corruptions.clipped_zoom = _clipped_zoom_no_scipy_warning

    setattr(imagecorruptions, "__imgaug2_patched__", True)
    return imagecorruptions, corruptions


# Added in 0.6.0.
def _gaussian_skimage_compat(image: Array, sigma: float, multichannel: bool) -> Array:
    """Call `skimage.filters.gaussian` across skimage versions."""
    try:
        return skimage.filters.gaussian(
            image, sigma=sigma, channel_axis=(-1 if multichannel else None)
        )
    except TypeError:
        # Old skimage versions used the `multichannel` kwarg.
        return skimage.filters.gaussian(image, sigma=sigma, multichannel=multichannel)


# Added in 0.4.0.
def _clipped_zoom_no_scipy_warning(img: Array, zoom_factor: float) -> Array:
    from scipy.ndimage import zoom as scizoom

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*output shape of zoom.*")

        # clipping along the width dimension:
        ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
        top0 = (img.shape[0] - ch0) // 2

        # clipping along the height dimension:
        ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
        top1 = (img.shape[1] - ch1) // 2

        img = scizoom(
            img[top0 : top0 + ch0, top1 : top1 + ch1], (zoom_factor, zoom_factor, 1), order=1
        )

        return img


def _call_imgcorrupt_func(
    fname: str | Callable[[Array, int], Array],
    seed: int | None,
    convert_to_pil: bool,
    image: Array,
    severity: int = 1,
) -> Array:
    """Apply an ``imagecorruptions`` function.

    The dtype support below is basically a placeholder to which the
    augmentation functions can point to decrease the amount of documentation.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; indirectly tested (1)
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

        - (1) Tested by comparison with function in ``imagecorruptions``
              package.

    """
    _imagecorruptions, corruptions = _patch_imagecorruptions_modules_()

    iadt.allow_only_uint8({image.dtype})

    input_shape = image.shape

    height, width = input_shape[0:2]
    assert height >= 32 and width >= 32, (
        "Expected the provided image to have a width and height of at least "
        "32 pixels, as that is the lower limit that the wrapped "
        f"imagecorruptions functions use. Got shape {image.shape}."
    )

    ndim = image.ndim
    assert ndim == 2 or (ndim == 3 and (image.shape[2] in [1, 3])), (
        "Expected input image to have shape (height, width) or "
        f"(height, width, 1) or (height, width, 3). Got shape {image.shape}."
    )

    if ndim == 2:
        image = image[..., np.newaxis]
    if image.shape[-1] == 1:
        image = np.tile(image, (1, 1, 3))

    image_in: object
    if convert_to_pil:
        import PIL.Image

        image_in = PIL.Image.fromarray(image)
    else:
        image_in = image

    with iarandom.temporary_numpy_seed(seed):
        if ia.is_callable(fname):
            image_aug = fname(image, severity)
        else:
            image_aug = getattr(corruptions, fname)(image_in, severity)

    if convert_to_pil:
        image_aug = np.asarray(image_aug)

    if ndim == 2:
        image_aug = image_aug[:, :, 0]
    elif input_shape[-1] == 1:
        image_aug = image_aug[:, :, 0:1]

    # this cast is done at the end of imagecorruptions.__init__.corrupt()
    image_aug = np.uint8(image_aug)

    return cast(Array, image_aug)


def get_corruption_names(
    subset: Literal["common", "validation", "all"] = "common",
) -> tuple[list[str], list[Callable[..., Array]]]:
    """Get a named subset of image corruption functions.

    .. note::

        This function returns the augmentation names (as strings) *and* the
        corresponding augmentation functions, while ``get_corruption_names()``
        in ``imagecorruptions`` only returns the augmentation names.

    Added in 0.4.0.

    Parameters
    ----------
    subset : {'common', 'validation', 'all'}, optional.
        Name of the subset of image corruption functions.

    Returns
    -------
    list of str
        Names of the corruption methods, e.g. "gaussian_noise".

    list of callable
        Function corresponding to the name. Is one of the
        ``apply_*()`` functions in this module. Apply e.g.
        via ``func(image, severity=2, seed=123)``.

    """
    # import imagecorruptions, note that it is an optional dependency
    try:
        # imagecorruptions sets its own warnings filter rule via
        # warnings.simplefilter(). That rule is the in effect for the whole
        # program and not just the module. So to prevent that here
        # we use catch_warnings(), which uintuitively does not by default
        # catch warnings but saves and restores the warnings filter settings.
        with warnings.catch_warnings():
            import imagecorruptions
    except ImportError:
        raise ImportError(_MISSING_PACKAGE_ERROR_MSG) from None

    cnames = imagecorruptions.get_corruption_names(subset)
    funcs = [globals()[f"apply_{cname}"] for cname in cnames]

    return cnames, funcs


# ----------------------------------------------------------------------------
# Corruption functions
# ----------------------------------------------------------------------------
# These functions could easily be created dynamically, especially templating
# the docstrings would save many lines of code. It is intentionally not done
# here for the same reasons as in case of the augmenters. See the comment
# further below at the start of the augmenter section for details.


def apply_gaussian_noise(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``gaussian_noise`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("gaussian_noise", seed, False, x, severity)


def apply_shot_noise(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``shot_noise`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("shot_noise", seed, False, x, severity)


def apply_impulse_noise(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``impulse_noise`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("impulse_noise", seed, False, x, severity)


def apply_speckle_noise(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``speckle_noise`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("speckle_noise", seed, False, x, severity)


def apply_gaussian_blur(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``gaussian_blur`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("gaussian_blur", seed, False, x, severity)


def apply_glass_blur(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``glass_blur`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func(_apply_glass_blur_imgaug, seed, False, x, severity)


# Added in 0.4.0.
def _apply_glass_blur_imgaug(x: Array, severity: int = 1) -> Array:
    # false positive on x_shape[0]
    # invalid name for dx, dy
    # pylint: disable=unsubscriptable-object, invalid-name

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


# Added in 0.5.0.
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


def apply_defocus_blur(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``defocus_blur`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("defocus_blur", seed, False, x, severity)


def apply_motion_blur(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``motion_blur`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("motion_blur", seed, False, x, severity)


def apply_zoom_blur(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``zoom_blur`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("zoom_blur", seed, False, x, severity)


def apply_fog(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``fog`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("fog", seed, False, x, severity)


def apply_frost(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``frost`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("frost", seed, False, x, severity)


def apply_snow(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``snow`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("snow", seed, False, x, severity)


def apply_spatter(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``spatter`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("spatter", seed, True, x, severity)


def apply_contrast(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``contrast`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("contrast", seed, False, x, severity)


def apply_brightness(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``brightness`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("brightness", seed, False, x, severity)


def apply_saturate(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``saturate`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("saturate", seed, False, x, severity)


def apply_jpeg_compression(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``jpeg_compression`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("jpeg_compression", seed, True, x, severity)


def apply_pixelate(x: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``pixelate`` from ``imagecorruptions``.

    Added in 0.4.0.

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

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("pixelate", seed, True, x, severity)


def apply_elastic_transform(image: Array, severity: int = 1, seed: int | None = None) -> Array:
    """Apply ``elastic_transform`` from ``imagecorruptions``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike._call_imgcorrupt_func`.

    Parameters
    ----------
    image : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("elastic_transform", seed, False, image, severity)


# ----------------------------------------------------------------------------
# Augmenters
# ----------------------------------------------------------------------------
# The augmenter definitions below are almost identical and mainly differ in
# the names and functions used. It would be fairly trivial to write a
# function that would create these augmenters dynamically (and one is listed
# below as a comment). The downside is that in these cases the documentation
# would also be generated dynamically, which leads to numerous problems:
# (1) users couldn't easily read the documentation while scrolling through
# the code file, (2) IDEs might not be able to use it for code suggestions,
# (3) tools like pylint can't detect and validate it, (4) the imgaug-doc
# tools to parse dtype support don't work with dynamically generated
# documentation (and neither with dynamically generated classes).
# Even though it's by far more code, it seems like the better choice overall
# to just write it out.

# Example function to dynamically generate augmenters, kept for possible
# future uses:
# def _create_augmenter(class_name, func_name):
#     func = globals()["apply_%s" % (func_name,)]
#
#     def __init__(self, severity=1, name=None, deterministic=False,
#                  random_state=None):
#         super(self.__class__, self).__init__(
#             func, severity, name=name, deterministic=deterministic,
#             random_state=random_state)
#
#     augmenter_class = type(class_name,
#                            (_ImgcorruptAugmenterBase,),
#                            {"__init__": __init__})
#
#     augmenter_class.__doc__ = """
#     Wrapper around ``imagecorruptions.corruptions.%s``.
#
#     **Supported dtypes**:
#
#     See :func:`~imgaug2.augmenters.imgcorruptlike.apply_%s`.
#
#     Parameters
#     ----------
#     severity : int, optional
#         Strength of the corruption, with valid values being
#         ``1 <= severity <= 5``.
#
#     name : None or str, optional
#         See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.
#
#     deterministic : bool, optional
#         See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.
#
#     random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
#         See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.
#
#     Examples
#     --------
#     >>> import imgaug2.augmenters as iaa
#     >>> aug = iaa.%s(severity=2)
#
#     Create an augmenter around ``imagecorruptions.corruptions.%s``. Apply it to
#     images using e.g. ``aug(images=[image1, image2, ...])``.
#
#     """ % (func_name, func_name, class_name, func_name)
#
#     return augmenter_class


# Added in 0.4.0.
class _ImgcorruptAugmenterBase(meta.Augmenter):
    def __init__(
        self,
        func: CorruptionFunc,
        severity: ParamInput = 1,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.func = func
        self.severity = iap.handle_discrete_param(
            severity,
            "severity",
            value_range=(1, 5),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )

    # Added in 0.4.0.
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        if batch.images is None:
            return batch

        severities, seeds = self._draw_samples(len(batch.images), random_state=random_state)

        for image, severity, seed in zip(batch.images, severities, seeds, strict=True):
            image[...] = self.func(image, severity=int(severity), seed=int(seed))

        return batch

    # Added in 0.4.0.
    def _draw_samples(self, nb_rows: int, random_state: iarandom.RNG) -> tuple[IntArray, IntArray]:
        severities = self.severity.draw_samples((nb_rows,), random_state=random_state)
        seeds = random_state.generate_seeds_(nb_rows)

        return severities, seeds

    # Added in 0.4.0.
    def get_parameters(self) -> list[iap.StochasticParameter]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.severity]


class GaussianNoise(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.gaussian_noise``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_gaussian_noise`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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
    >>> aug = iaa.imgcorruptlike.GaussianNoise(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.gaussian_noise``.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    # Added in 0.4.0.
    def __init__(
        self,
        severity: ParamInput = (1, 5),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            apply_gaussian_noise,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class ShotNoise(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.shot_noise``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_shot_noise`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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
    >>> aug = iaa.imgcorruptlike.ShotNoise(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.shot_noise``.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    # Added in 0.4.0.
    def __init__(
        self,
        severity: ParamInput = (1, 5),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            apply_shot_noise,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class ImpulseNoise(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.impulse_noise``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_impulse_noise`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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
    >>> aug = iaa.imgcorruptlike.ImpulseNoise(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.impulse_noise``.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    # Added in 0.4.0.
    def __init__(
        self,
        severity: ParamInput = (1, 5),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            apply_impulse_noise,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class SpeckleNoise(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.speckle_noise``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_speckle_noise`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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
    >>> aug = iaa.imgcorruptlike.SpeckleNoise(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.speckle_noise``.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    # Added in 0.4.0.
    def __init__(
        self,
        severity: ParamInput = (1, 5),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            apply_speckle_noise,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class GaussianBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.gaussian_blur``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_gaussian_blur`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class GlassBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.glass_blur``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_glass_blur`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class DefocusBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.defocus_blur``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_defocus_blur`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class MotionBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.motion_blur``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_motion_blur`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class ZoomBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.zoom_blur``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_zoom_blur`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class Fog(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.fog``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_fog`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class Frost(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.frost``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_frost`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class Snow(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.snow``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_snow`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class Spatter(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.spatter``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_spatter`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class Contrast(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.contrast``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_contrast`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class Brightness(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.brightness``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_brightness`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class Saturate(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.saturate``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_saturate`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class JpegCompression(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.jpeg_compression``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_jpeg_compression`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class Pixelate(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.pixelate``.

    .. note::

        This augmenter only affects images. Other data is not changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_pixelate`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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

    # Added in 0.4.0.
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


class ElasticTransform(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.elastic_transform``.

    .. warning::

        This augmenter can currently only transform image-data.
        Batches containing heatmaps, segmentation maps and
        coordinate-based augmentables will be rejected with an error.
        Use :class:`~imgaug2.augmenters.geometric.ElasticTransformation` if
        you have to transform such inputs.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_elastic_transform`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
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
    >>> aug = iaa.imgcorruptlike.ElasticTransform(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.elastic_transform``.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    # Added in 0.4.0.
    def __init__(
        self,
        severity: ParamInput = (1, 5),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            apply_elastic_transform,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
    )

    # Added in 0.4.0.
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        cols = batch.get_column_names()
        assert len(cols) == 0 or (len(cols) == 1 and "images" in cols), (
            "imgcorruptlike.ElasticTransform can currently only process image "
            "data. Got a batch containing: {}. Use "
            "imgaug2.augmenters.geometric.ElasticTransformation for "
            "batches containing non-image data.".format(", ".join(cols))
        )
        return super()._augment_batch_(batch, random_state, parents, hooks)
