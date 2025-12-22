from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import PIL.ImageOps

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.augmenters.contrast as contrastlib
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _normalize_cv2_input_arr_

from ._types import IgnoreValues
from ._utils import _maybe_mlx


@legacy(version="0.4.0")
def autocontrast(image: Array, cutoff: int = 0, ignore: IgnoreValues = None) -> Array:
    """Maximize (normalize) image contrast.

    This function calculates a histogram of the input image, removes
    **cutoff** percent of the lightest and darkest pixels from the histogram,
    and remaps the image so that the darkest pixel becomes black (``0``), and
    the lightest becomes white (``255``).

    This function has identical outputs to ``PIL.ImageOps.autocontrast``.
    The speed is almost identical.


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
        The image for which to enhance the contrast.

    cutoff : number
        How many percent to cut off at the low and high end of the
        histogram. E.g. ``20`` will cut off the lowest and highest ``20%``
        of values. Expected value range is ``[0, 100]``.

    ignore : None or int or iterable of int
        Intensity values to ignore, i.e. to treat as background. If ``None``,
        no pixels will be ignored. Otherwise exactly the given intensity
        value(s) will be ignored.

    Returns
    -------
    ndarray
        Contrast-enhanced image.

    """
    maybe = _maybe_mlx(image, "autocontrast", cutoff=cutoff, ignore=ignore)
    if maybe is not None:
        return maybe

    iadt.allow_only_uint8({image.dtype})

    if 0 in image.shape:
        return np.copy(image)

    standard_channels = image.ndim == 2 or image.shape[2] == 3

    if cutoff and standard_channels:
        return _autocontrast_pil(image, cutoff, ignore)
    return _autocontrast_no_pil(image, cutoff, ignore)




@legacy(version="0.4.0")
def _autocontrast_pil(image: Array, cutoff: int, ignore: IgnoreValues) -> Array:
    # don't return np.asarray(...) as its results are read-only
    return np.array(
        PIL.ImageOps.autocontrast(PIL.Image.fromarray(image), cutoff=cutoff, ignore=ignore)
    )


# This function is only faster than the corresponding PIL function if no
# cutoff is used.
# C901 is "<functionname> is too complex"


@legacy(version="0.4.0")
def _autocontrast_no_pil(image: Array, cutoff: int, ignore: IgnoreValues) -> Array:  # noqa: C901
    if ignore is not None and not ia.is_iterable(ignore):
        ignore = [ignore]

    result = np.empty_like(image)
    if result.ndim == 2:
        result = result[..., np.newaxis]
    nb_channels = image.shape[2] if image.ndim >= 3 else 1
    for c_idx in range(nb_channels):
        # using [0] instead of [int(c_idx)] allows this to work with >4
        # channels
        if image.ndim == 2:
            image_c = image[:, :, np.newaxis]
        else:
            image_c = image[:, :, c_idx : c_idx + 1]
        h = cv2.calcHist([_normalize_cv2_input_arr_(image_c)], [0], None, [256], [0, 256])
        if ignore is not None:
            h[ignore] = 0

        if cutoff:
            cs = np.cumsum(h)
            n = cs[-1]
            cut = n * cutoff // 100

            # remove cutoff% pixels from the low end
            lo_cut = cut - cs
            lo_cut_nz = np.nonzero(lo_cut <= 0.0)[0]
            if len(lo_cut_nz) == 0:
                lo = 255
            else:
                lo = lo_cut_nz[0]
            if lo > 0:
                h[:lo] = 0
            h[lo] = lo_cut[lo]

            # remove cutoff% samples from the hi end
            cs_rev = np.cumsum(h[::-1])
            hi_cut = cs_rev - cut
            hi_cut_nz = np.nonzero(hi_cut > 0.0)[0]
            if len(hi_cut_nz) == 0:
                hi = -1
            else:
                hi = 255 - hi_cut_nz[0]
            h[hi + 1 :] = 0
            if hi > -1:
                h[hi] = hi_cut[255 - hi]

        # find lowest/highest samples after preprocessing
        lo = 255
        for idx, lo_val in enumerate(h):
            if lo_val:
                lo = idx
                break
        for hi in range(255, -1, -1):
            if h[hi]:
                break
        if hi <= lo:
            # don't bother
            lut = np.arange(256)
        else:
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            ix = np.arange(256).astype(np.float64) * scale + offset
            ix = np.clip(ix, 0, 255).astype(np.uint8)
            lut = ix
        lut = np.array(lut, dtype=np.uint8)

        # Vectorized implementation of above block.
        # This is overall slower.
        # h_nz = np.nonzero(h)[0]
        # if len(h_nz) <= 1:
        #     lut = np.arange(256).astype(np.uint8)
        # else:
        #     lo = h_nz[0]
        #     hi = h_nz[-1]
        #
        #     scale = 255.0 / (hi - lo)
        #     offset = -lo * scale
        #     ix = np.arange(256).astype(np.float64) * scale + offset
        #     ix = np.clip(ix, 0, 255).astype(np.uint8)
        #     lut = ix

        # TODO change to a single call instead of one per channel
        image_c_aug = ia.apply_lut(image_c, lut)
        result[:, :, c_idx : c_idx + 1] = image_c_aug
    if image.ndim == 2:
        return result[..., 0]
    return result




@legacy(version="0.4.0")
class Autocontrast(contrastlib._ContrastFuncWrapper):
    """Adjust contrast by cutting off ``p%`` of lowest/highest histogram values.

    This augmenter has identical outputs to ``PIL.ImageOps.autocontrast``.

    See :func:`~imgaug2.augmenters.pillike.autocontrast` for more details.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.autocontrast`.

    Parameters
    ----------
    cutoff : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Percentage of values to cut off from the low and high end of each
        image's histogram, before stretching it to ``[0, 255]``.

            * If ``int``: The value will be used for all images.
            * If ``tuple`` ``(a, b)``: A value will be uniformly sampled from
              the discrete interval ``[a..b]`` per image.
            * If ``list``: A random value will be sampled from the list
              per image.
            * If ``StochasticParameter``: A value will be sampled from that
              parameter per image.

    per_channel :  bool or float, optional
        Whether to use the same value for all channels (``False``) or to
        sample a new value for each channel (``True``). If this value is a
        float ``p``, then for ``p`` percent of all images `per_channel` will
        be treated as ``True``, otherwise as ``False``.

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
    >>> aug = iaa.pillike.Autocontrast()

    Modify the contrast of images by cutting off the ``0`` to ``20%`` lowest
    and highest values from the histogram, then stretching it to full length.

    >>> aug = iaa.pillike.Autocontrast((10, 20), per_channel=True)

    Modify the contrast of images by cutting off the ``10`` to ``20%`` lowest
    and highest values from the histogram, then stretching it to full length.
    The cutoff value is sampled per *channel* instead of per *image*.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        cutoff: ParamInput = (0, 20),
        per_channel: ParamInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        from imgaug2.augmenters import pillike as pillike_lib

        params1d = [
            iap.handle_discrete_param(
                cutoff, "cutoff", value_range=(0, 49), tuple_to_uniform=True, list_to_choice=True
            )
        ]
        func = pillike_lib.autocontrast

        super().__init__(
            func,
            params1d,
            per_channel,
            dtypes_allowed="uint8",
            dtypes_disallowed="uint16 uint32 uint64 int8 int16 int32 int64 "
            "float16 float32 float64 float128 "
            "bool",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )
