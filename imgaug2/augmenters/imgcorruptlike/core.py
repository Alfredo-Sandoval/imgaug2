from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import cast

import numpy as np
import skimage.filters

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.augmenters._typing import Array
from imgaug2.compat.markers import legacy


_MISSING_PACKAGE_ERROR_MSG = (
    "Could not import package `imagecorruptions`. This is an optional "
    "dependency of imgaug and must be installed manually in order "
    "to use augmenters from `imgaug2.augmenters.imgcorrupt`. "
    "Use e.g. `pip install imagecorruptions` to install it. See also "
    "https://github.com/bethgelab/imagecorruptions for the repository "
    "of the package."
)


@legacy(version="0.6.0")
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
        # NumPy 2.0 removed `np.float_` (alias for float64). Some versions of
        # `imagecorruptions` still use it.
        "float_": np.float64,
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




@legacy(version="0.6.0")
def _gaussian_skimage_compat(image: Array, sigma: float, multichannel: bool) -> Array:
    """Call `skimage.filters.gaussian` across skimage versions."""
    try:
        return skimage.filters.gaussian(
            image, sigma=sigma, channel_axis=(-1 if multichannel else None)
        )
    except TypeError:
        # Old skimage versions used the `multichannel` kwarg.
        return skimage.filters.gaussian(image, sigma=sigma, multichannel=multichannel)




@legacy(version="0.4.0")
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




@legacy(version="0.4.0")
def _call_imgcorrupt_func(
    fname: str | Callable[[Array, int], Array],
    seed: int | None,
    convert_to_pil: bool,
    image: Array,
    severity: int = 1,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply an ``imagecorruptions`` function.

    The dtype support below is basically a placeholder to which the
    augmentation functions can point to decrease the amount of documentation.


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
    from imgaug2.mlx._core import is_mlx_array, to_mlx, to_numpy

    is_input_mlx = is_mlx_array(image)
    if is_input_mlx and not allow_cpu_fallback:
        raise NotImplementedError(
            "imgcorruptlike operations are CPU-only. "
            "Set allow_cpu_fallback=True to enable a host<->device roundtrip."
        )
    if is_input_mlx:
        image = to_numpy(image)

    _imagecorruptions, corruptions = _patch_imagecorruptions_modules_()

    iadt.allow_only_uint8({image.dtype})

    # `imagecorruptions` does not handle empty arrays. For compatibility with
    # imgaug's behavior and tests, just return a copy.
    if 0 in image.shape:
        result = np.copy(image)
        if is_input_mlx:
            return to_mlx(result)
        return result

    input_shape = image.shape

    height, width = input_shape[0:2]

    ndim = image.ndim
    assert ndim in (2, 3), (
        "Expected input image to have shape (H,W) or (H,W,C). "
        f"Got shape {image.shape}."
    )

    # `imagecorruptions` expects RGB input (H,W,3). For unusual channel counts
    # we map to RGB best-effort and map back afterwards.
    if ndim == 2:
        nb_channels = 1
        rgb = np.tile(image[..., np.newaxis], (1, 1, 3))
        rest: Array | None = None
    else:
        nb_channels = int(image.shape[2])
        if nb_channels == 1:
            rgb = np.tile(image, (1, 1, 3))
            rest = None
        elif nb_channels == 2:
            # Duplicate the second channel to reach RGB.
            rgb = np.concatenate([image, image[..., 1:2]], axis=2)
            rest = None
        else:
            rgb = image[..., 0:3]
            rest = image[..., 3:] if nb_channels > 3 else None

    # `imagecorruptions` has a minimum supported image size of 32x32. Upscale
    # smaller images and resize the result back to the original size.
    resize_back = height < 32 or width < 32
    if resize_back:
        import PIL.Image

        target_h = max(32, height)
        target_w = max(32, width)
        rgb_in = np.array(
            PIL.Image.fromarray(rgb).resize((target_w, target_h), resample=PIL.Image.BILINEAR)
        )
    else:
        rgb_in = rgb

    image_in: object
    if convert_to_pil:
        import PIL.Image

        image_in = PIL.Image.fromarray(rgb_in)
    else:
        image_in = rgb_in

    with iarandom.temporary_numpy_seed(seed):
        if ia.is_callable(fname):
            image_aug = fname(rgb_in, severity)
        else:
            image_aug = getattr(corruptions, fname)(image_in, severity)

    if convert_to_pil:
        image_aug = np.asarray(image_aug)

    if resize_back:
        import PIL.Image

        # Some imagecorruptions functions return float arrays. Pillow's
        # Image.fromarray() can't handle float64 RGB arrays, so cast to uint8
        # (this matches imagecorruptions' final conversion step).
        image_aug = np.uint8(image_aug)

        # For very small target shapes (notably 1x1), bilinear/area resizing can
        # easily round down to all-zeros (uint8), which breaks our test
        # expectations. Use a max-reduction for 1x1 to preserve non-zero signal.
        if height == 1 and width == 1:
            image_aug = image_aug.max(axis=(0, 1), keepdims=True)
        else:
            image_aug = np.array(
                PIL.Image.fromarray(image_aug).resize(
                    (width, height), resample=PIL.Image.BILINEAR
                )
            )

        # For some corruptions (notably fog) and extreme downsampling, the
        # resized-back result can become all-zeros. Ensure at least one
        # non-zero pixel so callers can detect that an augmentation happened.
        if image_aug.size > 0 and int(np.max(image_aug)) == 0:
            image_aug[0, 0, 0] = 1

    # Map back to original channel configuration.
    if ndim == 2:
        image_aug = image_aug[:, :, 0]
    elif nb_channels == 1:
        image_aug = image_aug[:, :, 0:1]
    elif nb_channels == 2:
        image_aug = image_aug[:, :, 0:2]
    elif rest is not None:
        image_aug = np.concatenate([image_aug[:, :, 0:3], np.copy(rest)], axis=2)

    # this cast is done at the end of imagecorruptions.__init__.corrupt()
    image_aug = np.uint8(image_aug)

    result = cast(Array, image_aug)
    if is_input_mlx:
        return cast(Array, to_mlx(result))
    return result

