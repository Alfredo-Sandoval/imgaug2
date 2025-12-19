"""Core alpha blending utilities shared across augmenters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import cv2
import numpy as np

import imgaug2.dtypes as iadt
from imgaug2.augmenters._typing import Array
from imgaug2.imgaug import _normalize_cv2_input_arr_
from imgaug2.compat.markers import legacy

AlphaInput: TypeAlias = float | int | Sequence[float | int] | Array


def blend_alpha(image_fg: Array, image_bg: Array, alpha: AlphaInput, eps: float = 1e-2) -> Array:
    """
    Blend two images using an alpha blending.

    In alpha blending, the two images are naively mixed using a multiplier.
    Let ``A`` be the foreground image and ``B`` the background image and
    ``a`` is the alpha value. Each pixel intensity is then computed as
    ``a * A_ij + (1-a) * B_ij``.

    **Supported dtypes**:

    See :func:`imgaug2.augmenters.blend.blend_alpha_`.

    Parameters
    ----------
    image_fg : (H,W,[C]) ndarray
        Foreground image. Shape and dtype kind must match the one of the
        background image.

    image_bg : (H,W,[C]) ndarray
        Background image. Shape and dtype kind must match the one of the
        foreground image.

    alpha : number or iterable of number or ndarray
        The blending factor, between ``0.0`` and ``1.0``. Can be interpreted
        as the opacity of the foreground image. Values around ``1.0`` result
        in only the foreground image being visible. Values around ``0.0``
        result in only the background image being visible. Multiple alphas
        may be provided. In these cases, there must be exactly one alpha per
        channel in the foreground/background image. Alternatively, for
        ``(H,W,C)`` images, either one ``(H,W)`` array or an ``(H,W,C)``
        array of alphas may be provided, denoting the elementwise alpha value.

    eps : number, optional
        Controls when an alpha is to be interpreted as exactly ``1.0`` or
        exactly ``0.0``, resulting in only the foreground/background being
        visible and skipping the actual computation.

    Returns
    -------
    image_blend : (H,W,C) ndarray
        Blend of foreground and background image.

    """
    return blend_alpha_(np.copy(image_fg), image_bg, alpha, eps=eps)


@legacy(version="0.5.0")
def blend_alpha_(image_fg: Array, image_bg: Array, alpha: AlphaInput, eps: float = 1e-2) -> Array:
    """
    Blend two images in-place using an alpha blending.

    In alpha blending, the two images are naively mixed using a multiplier.
    Let ``A`` be the foreground image and ``B`` the background image and
    ``a`` is the alpha value. Each pixel intensity is then computed as
    ``a * A_ij + (1-a) * B_ij``.

    Extracted from :func:`blend_alpha`.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; fully tested
        * ``uint32``: yes; fully tested
        * ``uint64``: limited; fully tested (1)
        * ``int8``: yes; fully tested
        * ``int16``: yes; fully tested
        * ``int32``: yes; fully tested
        * ``int64``: limited; fully tested (1)
        * ``float16``: yes; fully tested
        * ``float32``: yes; fully tested
        * ``float64``: limited; fully tested (1)
        * ``float128``: no (2)
        * ``bool``: yes; fully tested (2)

        - (1) Tests show that these dtypes work, but a conversion to
              ``float128`` happens, which only has 96 bits of size instead of
              true 128 bits and hence not twice as much resolution. It is
              possible that these dtypes result in inaccuracies, though the
              tests did not indicate that.
              Note that ``float128`` support is required for these dtypes
              and thus they are not expected to work on Windows machines.
        - (2) Not available due to the input dtype having to be increased to
              an equivalent float dtype with two times the input resolution.
        - (3) Mapped internally to ``float16``.

    Parameters
    ----------
    image_fg : (H,W,[C]) ndarray
        Foreground image. Shape and dtype kind must match the one of the
        background image.
        This image might be modified in-place.

    image_bg : (H,W,[C]) ndarray
        Background image. Shape and dtype kind must match the one of the
        foreground image.

    alpha : number or iterable of number or ndarray
        The blending factor, between ``0.0`` and ``1.0``. Can be interpreted
        as the opacity of the foreground image. Values around ``1.0`` result
        in only the foreground image being visible. Values around ``0.0``
        result in only the background image being visible. Multiple alphas
        may be provided. In these cases, there must be exactly one alpha per
        channel in the foreground/background image. Alternatively, for
        ``(H,W,C)`` images, either one ``(H,W)`` array or an ``(H,W,C)``
        array of alphas may be provided, denoting the elementwise alpha value.

    eps : number, optional
        Controls when an alpha is to be interpreted as exactly ``1.0`` or
        exactly ``0.0``, resulting in only the foreground/background being
        visible and skipping the actual computation.

    Returns
    -------
    image_blend : (H,W,C) ndarray
        Blend of foreground and background image.
        This might be an in-place modified version of `image_fg`.

    """
    assert image_fg.shape == image_bg.shape, (
        "Expected foreground and background images to have the same shape. "
        f"Got {image_fg.shape} and {image_bg.shape}."
    )
    assert image_fg.dtype.kind == image_bg.dtype.kind, (
        "Expected foreground and background images to have the same dtype "
        f"kind. Got {image_fg.dtype.kind} and {image_bg.dtype.kind}."
    )

    # Note: If float128 is not available on the system, _FLOAT128_DTYPE is
    # None, but 'np.dtype("float64") == None' actually equates to True
    # for whatever reason, so we check first if the constant is not None
    # (i.e. if float128 exists).
    if iadt._FLOAT128_DTYPE is not None:
        assert image_fg.dtype != iadt._FLOAT128_DTYPE, (
            "Foreground image was float128, but blend_alpha_() cannot handle that dtype."
        )
        assert image_bg.dtype != iadt._FLOAT128_DTYPE, (
            "Background image was float128, but blend_alpha_() cannot handle that dtype."
        )

    if image_fg.size == 0:
        return image_fg

    input_was_2d = image_fg.ndim == 2
    if input_was_2d:
        image_fg = image_fg[..., np.newaxis]
        image_bg = image_bg[..., np.newaxis]

    input_was_bool = False
    if image_fg.dtype.kind == "b":
        input_was_bool = True
        # use float32 instead of float16 here because it seems to be faster
        image_fg = image_fg.astype(np.float32)
        image_bg = image_bg.astype(np.float32)

    alpha = np.array(alpha, dtype=np.float64)
    if alpha.size == 1:
        pass
    else:
        if alpha.ndim == 2:
            assert alpha.shape == image_fg.shape[0:2], (
                "'alpha' given as an array must match the height and width "
                f"of the foreground and background image. Got shape {alpha.shape} vs "
                f"foreground/background shape {image_fg.shape}."
            )
        elif alpha.ndim == 3:
            assert alpha.shape == image_fg.shape or alpha.shape == image_fg.shape[0:2] + (1,), (
                "'alpha' given as an array must match the height and "
                "width of the foreground and background image. Got "
                f"shape {alpha.shape} vs foreground/background shape {image_fg.shape}."
            )
        else:
            alpha = alpha.reshape((1, 1, -1))

    if not input_was_bool:
        if np.all(alpha >= 1.0 - eps):
            if input_was_2d:
                image_fg = image_fg[..., 0]
            return image_fg
        if np.all(alpha <= eps):
            if input_was_2d:
                image_bg = image_bg[..., 0]
            # use copy() here so that only image_fg has to be copied in
            # blend_alpha()
            return np.copy(image_bg)

    # for efficiency reasons, only test one value of alpha here, even if alpha
    # is much larger
    if alpha.size > 0:
        assert 0 <= alpha.item(0) <= 1.0, (
            "Expected 'alpha' value(s) to be in the interval [0.0, 1.0]. "
            f"Got min {np.min(alpha):.4f} and max {np.max(alpha):.4f}."
        )

    uint8 = iadt._UINT8_DTYPE
    both_uint8 = (image_fg.dtype, image_bg.dtype) == (uint8, uint8)
    if both_uint8:
        if alpha.size == 1:
            image_blend = _blend_alpha_uint8_single_alpha_(
                image_fg, image_bg, float(alpha), inplace=True
            )
        elif alpha.shape == (1, 1, image_fg.shape[2]):
            image_blend = _blend_alpha_uint8_channelwise_alphas_(image_fg, image_bg, alpha[0, 0, :])
        else:
            image_blend = _blend_alpha_uint8_elementwise_(image_fg, image_bg, alpha)
    else:
        image_blend = _blend_alpha_non_uint8(image_fg, image_bg, alpha)

    if input_was_bool:
        image_blend = image_blend > 0.5

    if input_was_2d:
        return image_blend[:, :, 0]
    return image_blend


def _blend_alpha_uint8_single_alpha_(
    image_fg: Array, image_bg: Array, alpha: float, inplace: bool
) -> Array:
    # here we are not guarantueed that inputs have ndim=3, can be ndim=2
    result = cv2.addWeighted(
        _normalize_cv2_input_arr_(image_fg),
        alpha,
        _normalize_cv2_input_arr_(image_bg),
        beta=(1 - alpha),
        gamma=0.0,
        dst=image_fg if inplace else None,
    )
    if result.ndim == 2 and image_fg.ndim == 3:
        return result[:, :, np.newaxis]
    return result


def _blend_alpha_uint8_channelwise_alphas_(image_fg: Array, image_bg: Array, alphas: Array) -> Array:
    # we are guarantueed here that image_fg and image_bg have ndim=3
    result = []
    for i, alpha in enumerate(alphas):
        result.append(
            _blend_alpha_uint8_single_alpha_(
                image_fg[:, :, i], image_bg[:, :, i], float(alpha), inplace=False
            )
        )

    image_blend = _merge_channels(result, image_fg.ndim == 3)
    return image_blend


def _blend_alpha_uint8_elementwise_(image_fg: Array, image_bg: Array, alphas: Array) -> Array:
    betas = 1.0 - alphas

    is_2d = alphas.ndim == 2 or alphas.shape[2] == 1
    area = image_fg.shape[0] * image_fg.shape[1]
    if is_2d and area >= 64 * 64:
        if alphas.ndim == 3:
            alphas = alphas[:, :, 0]
            betas = betas[:, :, 0]

        result = []
        for c in range(image_fg.shape[2]):
            image_fg_mul = image_fg[:, :, c]
            image_bg_mul = image_bg[:, :, c]
            image_fg_mul = cv2.multiply(image_fg_mul, alphas, dtype=cv2.CV_8U)
            image_bg_mul = cv2.multiply(image_bg_mul, betas, dtype=cv2.CV_8U)
            image_fg_mul = cv2.add(image_fg_mul, image_bg_mul, dst=image_fg_mul)
            result.append(image_fg_mul)

        image_blend = _merge_channels(result, image_fg.ndim == 3)
        return image_blend
    else:
        if alphas.ndim == 2:
            alphas = alphas[..., np.newaxis]
            betas = betas[..., np.newaxis]
        if alphas.shape[2] != image_fg.shape[2]:
            alphas = np.tile(alphas, (1, 1, image_fg.shape[2]))
            betas = np.tile(betas, (1, 1, image_fg.shape[2]))

        alphas = alphas.ravel()
        betas = betas.ravel()
        input_shape = image_fg.shape

        image_fg_mul = image_fg.ravel()
        image_bg_mul = image_bg.ravel()
        image_fg_mul = cv2.multiply(image_fg_mul, alphas, dtype=cv2.CV_8U, dst=image_fg_mul)
        image_bg_mul = cv2.multiply(image_bg_mul, betas, dtype=cv2.CV_8U, dst=image_bg_mul)

        image_fg_mul = cv2.add(image_fg_mul, image_bg_mul, dst=image_fg_mul)

        return image_fg_mul.reshape(input_shape)


def _blend_alpha_non_uint8(image_fg: Array, image_bg: Array, alpha: Array) -> Array:
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]

    dt_images = iadt.get_minimal_dtype([image_fg, image_bg])

    # doing the below itemsize increase only for non-float images led to
    # inaccuracies for large float values
    # we also use a minimum of 4 bytes (=float32), as float32 tends to be
    # faster than float16
    isize = dt_images.itemsize * 2
    isize = max(isize, 4)
    dt_name = f"f{isize}"

    # check if float128 (16*8=128) is supported
    assert dt_name != "f16" or hasattr(np, "float128"), (
        f"The input images use dtype '{image_fg.dtype.name}', for which alpha-blending "
        "requires float128 support to compute accurately its output, "
        "but float128 seems to not be available on the current "
        "system."
    )

    dt_blend = np.dtype(dt_name)

    if alpha.dtype != dt_blend:
        alpha = alpha.astype(dt_blend)
    if image_fg.dtype != dt_blend:
        image_fg = image_fg.astype(dt_blend)
    if image_bg.dtype != dt_blend:
        image_bg = image_bg.astype(dt_blend)

    # the following is
    #     image_blend = image_bg + alpha * (image_fg - image_bg)
    # which is equivalent to
    #     image_blend = alpha * image_fg + (1 - alpha) * image_bg
    # but supposedly faster
    image_blend = image_fg - image_bg
    image_blend *= alpha
    image_blend += image_bg

    # Skip clip, because alpha is expected to be in range [0.0, 1.0] and
    # both images must have same dtype.
    # Dont skip round, because otherwise it is very unlikely to hit the
    # image's max possible value
    image_blend = iadt.restore_dtypes_(image_blend, dt_images, clip=False, round=True)

    return image_blend


def _merge_channels(channels: Sequence[Array], input_was_3d: bool) -> Array:
    if len(channels) <= 512:
        image_blend = cv2.merge(channels)
    else:
        image_blend = np.stack(channels, axis=-1)
    if image_blend.ndim == 2 and input_was_3d:
        image_blend = image_blend[:, :, np.newaxis]
    return image_blend
