from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import PIL.Image
import PIL.ImageOps

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.random as iarandom
import imgaug2.augmenters.meta as meta
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters._typing import Array, RNGInput
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _normalize_cv2_input_arr_

from ._utils import _maybe_mlx


_EQUALIZE_USE_PIL_BELOW = 64 * 64  # H*W


@legacy(version="0.4.0")
def equalize(image: Array, mask: Array | None = None) -> Array:
    """Equalize the image histogram.

    See :func:`~imgaug2.augmenters.pillike.equalize_` for details.

    This function is identical in inputs and outputs to
    ``PIL.ImageOps.equalize``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.equalize_`.

    Parameters
    ----------
    image : ndarray
        ``uint8`` ``(H,W,[C])`` image to equalize.

    mask : None or ndarray, optional
        An optional mask. If given, only the pixels selected by the mask are
        included in the analysis.

    Returns
    -------
    ndarray
        Equalized image.

    """
    maybe = _maybe_mlx(image, "equalize", mask=mask)
    if maybe is not None:
        return maybe

    # internally used method works in-place by default and hence needs a copy
    size = image.size
    if size == 0:
        return np.copy(image)
    if size >= _EQUALIZE_USE_PIL_BELOW:
        image = np.copy(image)
    from imgaug2.augmenters import pillike as pillike_lib

    return pillike_lib.equalize_(image, mask)




@legacy(version="0.4.0")
def equalize_(image: Array, mask: Array | None = None) -> Array:
    """Equalize the image histogram in-place.

    This function applies a non-linear mapping to the input image, in order
    to create a uniform distribution of grayscale values in the output image.

    This function has identical outputs to ``PIL.ImageOps.equalize``.
    It does however work in-place.


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
        ``uint8`` ``(H,W,[C])`` image to equalize.

    mask : None or ndarray, optional
        An optional mask. If given, only the pixels selected by the mask are
        included in the analysis.

    Returns
    -------
    ndarray
        Equalized image. *Might* have been modified in-place.

    """
    maybe = _maybe_mlx(image, "equalize", mask=mask)
    if maybe is not None:
        return maybe

    nb_channels = 1 if image.ndim == 2 else image.shape[-1]
    if nb_channels not in [1, 3]:
        result = [equalize_(image[:, :, c]) for c in np.arange(nb_channels)]
        return np.stack(result, axis=-1)

    iadt.allow_only_uint8({image.dtype})

    if mask is not None:
        assert mask.ndim == 2, f"Expected 2-dimensional mask, got shape {mask.shape}."
        assert mask.dtype == iadt._UINT8_DTYPE, (
            f"Expected mask of dtype uint8, got dtype {mask.dtype.name}."
        )

    size = image.size
    if size == 0:
        return image
    if nb_channels == 3 and size < _EQUALIZE_USE_PIL_BELOW:
        return _equalize_pil_(image, mask)
    return _equalize_no_pil_(image, mask)


# note that this is supposed to be a non-PIL reimplementation of PIL's
# equalize, which produces slightly different results from cv2.equalizeHist()


@legacy(version="0.4.0")
def _equalize_no_pil_(image: Array, mask: Array | None = None) -> Array:
    nb_channels = 1 if image.ndim == 2 else image.shape[-1]
    lut = np.empty((256, nb_channels), dtype=np.int32)

    for c_idx in range(nb_channels):
        if image.ndim == 2:
            image_c = image[:, :, np.newaxis]
        else:
            image_c = image[:, :, c_idx : c_idx + 1]
        histo = cv2.calcHist([_normalize_cv2_input_arr_(image_c)], [0], mask, [256], [0, 256])
        if len(histo.nonzero()[0]) <= 1:
            lut[:, c_idx] = np.arange(256).astype(np.int32)
            continue

        step = np.sum(histo[:-1]) // 255
        if not step:
            lut[:, c_idx] = np.arange(256).astype(np.int32)
            continue

        n = step // 2
        cumsum = np.cumsum(histo)
        lut[0, c_idx] = n
        lut[1:, c_idx] = n + cumsum[0:-1]
        lut[:, c_idx] //= int(step)
    lut = np.clip(lut, None, 255, out=lut).astype(np.uint8)
    image = ia.apply_lut_(image, lut)
    return image




@legacy(version="0.4.0")
def _equalize_pil_(image: Array, mask: Array | None = None) -> Array:
    if mask is not None:
        mask = PIL.Image.fromarray(mask).convert("L")

    # don't return np.asarray(...) directly as its results are read-only
    image[...] = np.asarray(PIL.ImageOps.equalize(PIL.Image.fromarray(image), mask=mask))
    return image




@legacy(version="0.4.0")
class Equalize(meta.Augmenter):
    """Equalize the image histogram.

    This augmenter has identical outputs to ``PIL.ImageOps.equalize``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.equalize_`.

    Parameters
    ----------
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
    >>> aug = iaa.pillike.Equalize()

    Equalize the histograms of all input images.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        from imgaug2.augmenters import pillike as pillike_lib

        if batch.images is not None:
            for image in batch.images:
                image[...] = pillike_lib.equalize_(image)
        return batch

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return []
