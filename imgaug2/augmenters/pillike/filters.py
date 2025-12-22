from __future__ import annotations

from typing import Literal

import numpy as np
import PIL.Image
import PIL.ImageFilter

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.random as iarandom
import imgaug2.augmenters.meta as meta
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters._typing import Array, RNGInput
from imgaug2.compat.markers import legacy

from ._utils import _ensure_valid_shape, _maybe_mlx


@legacy(version="0.4.0")
def _filter_by_kernel(image: Array, kernel: PIL.ImageFilter.Filter) -> Array:
    iadt.allow_only_uint8({image.dtype})

    if 0 in image.shape:
        return np.copy(image)

    image, is_hw1 = _ensure_valid_shape(image, "imgaug2.augmenters.pillike.filter_*()")

    image_pil = PIL.Image.fromarray(image)

    image_filtered = image_pil.filter(kernel)

    # don't return np.asarray(...) as its results are read-only
    result = np.array(image_filtered)
    if is_hw1:
        result = result[:, :, np.newaxis]
    return result




@legacy(version="0.4.0")
def filter_blur(image: Array) -> Array:
    """Apply a blur filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.BLUR`` kernel.


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

    Returns
    -------
    ndarray
        Blurred image.

    """
    maybe = _maybe_mlx(image, "filter_blur")
    if maybe is not None:
        return maybe

    return _filter_by_kernel(image, PIL.ImageFilter.BLUR)




@legacy(version="0.4.0")
def filter_smooth(image: Array) -> Array:
    """Apply a smoothness filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.SMOOTH`` kernel.


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

    Returns
    -------
    ndarray
        Smoothened image.

    """
    maybe = _maybe_mlx(image, "filter_smooth")
    if maybe is not None:
        return maybe

    return _filter_by_kernel(image, PIL.ImageFilter.SMOOTH)




@legacy(version="0.4.0")
def filter_smooth_more(image: Array) -> Array:
    """Apply a strong smoothness filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.SMOOTH_MORE`` kernel.


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

    Returns
    -------
    ndarray
        Smoothened image.

    """
    maybe = _maybe_mlx(image, "filter_smooth_more")
    if maybe is not None:
        return maybe

    return _filter_by_kernel(image, PIL.ImageFilter.SMOOTH_MORE)




@legacy(version="0.4.0")
def filter_edge_enhance(image: Array) -> Array:
    """Apply an edge enhancement filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.EDGE_ENHANCE`` kernel.


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

    Returns
    -------
    ndarray
        Image with enhanced edges.

    """
    maybe = _maybe_mlx(image, "filter_edge_enhance")
    if maybe is not None:
        return maybe

    return _filter_by_kernel(image, PIL.ImageFilter.EDGE_ENHANCE)




@legacy(version="0.4.0")
def filter_edge_enhance_more(image: Array) -> Array:
    """Apply a stronger edge enhancement filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.EDGE_ENHANCE_MORE``
    kernel.


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

    Returns
    -------
    ndarray
        Smoothened image.

    """
    maybe = _maybe_mlx(image, "filter_edge_enhance_more")
    if maybe is not None:
        return maybe

    return _filter_by_kernel(image, PIL.ImageFilter.EDGE_ENHANCE_MORE)




@legacy(version="0.4.0")
def filter_find_edges(image: Array) -> Array:
    """Apply an edge detection filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.FIND_EDGES`` kernel.


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

    Returns
    -------
    ndarray
        Image with detected edges.

    """
    maybe = _maybe_mlx(image, "filter_find_edges")
    if maybe is not None:
        return maybe

    return _filter_by_kernel(image, PIL.ImageFilter.FIND_EDGES)




@legacy(version="0.4.0")
def filter_contour(image: Array) -> Array:
    """Apply a contour filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.CONTOUR`` kernel.


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

    Returns
    -------
    ndarray
        Image with pronounced contours.

    """
    maybe = _maybe_mlx(image, "filter_contour")
    if maybe is not None:
        return maybe

    return _filter_by_kernel(image, PIL.ImageFilter.CONTOUR)




@legacy(version="0.4.0")
def filter_emboss(image: Array) -> Array:
    """Apply an emboss filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.EMBOSS`` kernel.


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

    Returns
    -------
    ndarray
        Embossed image.

    """
    maybe = _maybe_mlx(image, "filter_emboss")
    if maybe is not None:
        return maybe

    return _filter_by_kernel(image, PIL.ImageFilter.EMBOSS)




@legacy(version="0.4.0")
def filter_sharpen(image: Array) -> Array:
    """Apply a sharpening filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.SHARPEN`` kernel.


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

    Returns
    -------
    ndarray
        Sharpened image.

    """
    maybe = _maybe_mlx(image, "filter_sharpen")
    if maybe is not None:
        return maybe

    return _filter_by_kernel(image, PIL.ImageFilter.SHARPEN)




@legacy(version="0.4.0")
def filter_detail(image: Array) -> Array:
    """Apply a detail enhancement filter kernel to the image.

    This is the same as using PIL's ``PIL.ImageFilter.DETAIL`` kernel.


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

    Returns
    -------
    ndarray
        Image with enhanced details.

    """
    maybe = _maybe_mlx(image, "filter_detail")
    if maybe is not None:
        return maybe

    return _filter_by_kernel(image, PIL.ImageFilter.DETAIL)


# TODO unify this with the matrix generation for Affine,
#      there is probably no need to keep these separate


@legacy(version="0.4.0")
class _FilterBase(meta.Augmenter):
    @legacy(version="0.4.0")
    def __init__(
        self,
        func: object,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.func = func

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        if batch.images is not None:
            for image in batch.images:
                image[...] = self.func(image)
        return batch

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return []




@legacy(version="0.4.0")
class FilterBlur(_FilterBase):
    """Apply a blur filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.BLUR``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.filter_blur`.

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
    >>> aug = iaa.pillike.FilterBlur()

    Create an augmenter that applies a blur filter kernel to images.

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
            func=filter_blur,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class FilterSmooth(_FilterBase):
    """Apply a smoothening filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.SMOOTH``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.filter_smooth`.

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
    >>> aug = iaa.pillike.FilterSmooth()

    Create an augmenter that applies a smoothening filter kernel to images.

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
            func=filter_smooth,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class FilterSmoothMore(_FilterBase):
    """Apply a strong smoothening filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.BLUR``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.filter_smooth_more`.

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
    >>> aug = iaa.pillike.FilterSmoothMore()

    Create an augmenter that applies a strong smoothening filter kernel to
    images.

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
            func=filter_smooth_more,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class FilterEdgeEnhance(_FilterBase):
    """Apply an edge enhance filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel
    ``PIL.ImageFilter.EDGE_ENHANCE``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.filter_edge_enhance`.

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
    >>> aug = iaa.pillike.FilterEdgeEnhance()

    Create an augmenter that applies a edge enhancement filter kernel to
    images.

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
            func=filter_edge_enhance,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class FilterEdgeEnhanceMore(_FilterBase):
    """Apply a strong edge enhancement filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel
    ``PIL.ImageFilter.EDGE_ENHANCE_MORE``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.filter_edge_enhance_more`.

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
    >>> aug = iaa.pillike.FilterEdgeEnhanceMore()

    Create an augmenter that applies a strong edge enhancement filter kernel
    to images.

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
            func=filter_edge_enhance_more,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class FilterFindEdges(_FilterBase):
    """Apply a edge detection kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel
    ``PIL.ImageFilter.FIND_EDGES``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.filter_find_edges`.

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
    >>> aug = iaa.pillike.FilterFindEdges()

    Create an augmenter that applies an edge detection filter kernel to images.

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
            func=filter_find_edges,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class FilterContour(_FilterBase):
    """Apply a contour detection filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.CONTOUR``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.filter_contour`.

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
    >>> aug = iaa.pillike.FilterContour()

    Create an augmenter that applies a contour detection filter kernel to
    images.

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
            func=filter_contour,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class FilterEmboss(_FilterBase):
    """Apply an emboss filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.EMBOSS``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.filter_emboss`.

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
    >>> aug = iaa.pillike.FilterEmboss()

    Create an augmenter that applies an emboss filter kernel to images.

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
            func=filter_emboss,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class FilterSharpen(_FilterBase):
    """Apply a sharpening filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.SHARPEN``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.filter_sharpen`.

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
    >>> aug = iaa.pillike.FilterSharpen()

    Create an augmenter that applies a sharpening filter kernel to images.

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
            func=filter_sharpen,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )




@legacy(version="0.4.0")
class FilterDetail(_FilterBase):
    """Apply a detail enhancement filter kernel to images.

    This augmenter has identical outputs to
    calling ``PIL.Image.filter`` with kernel ``PIL.ImageFilter.DETAIL``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.filter_detail`.

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
    >>> aug = iaa.pillike.FilterDetail()

    Create an augmenter that applies a detail enhancement filter kernel to
    images.

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
            func=filter_detail,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


