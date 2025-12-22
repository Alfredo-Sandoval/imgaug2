from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Literal

import cv2
import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.compat.markers import legacy
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput

from ._utils import CSPACE_Lab, CSPACE_RGB, ColorSpace, ToColorspaceChoiceInput
from .luts import (
    _QuantizeUniformCenterizedLUTTableSingleton,
    _QuantizeUniformNotCenterizedLUTTableSingleton,
)


class _AbstractColorQuantization(meta.Augmenter, metaclass=ABCMeta):
    def __init__(
        self,
        counts: ParamInput = (2, 16),  # number of bits or colors
        counts_value_range: tuple[int, int | None] = (2, None),
        from_colorspace: ColorSpace = CSPACE_RGB,
        to_colorspace: ToColorspaceChoiceInput | None = None,
        max_size: int | None = 128,
        interpolation: str = "linear",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.counts_value_range = counts_value_range
        self.counts = iap.handle_discrete_param(
            counts,
            "counts",
            value_range=counts_value_range,
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self.from_colorspace = from_colorspace
        self.to_colorspace = to_colorspace
        self.max_size = max_size
        self.interpolation = interpolation

    def _draw_samples(self, n_augmentables: int, random_state: iarandom.RNG) -> Array:
        counts = self.counts.draw_samples((n_augmentables,), random_state)
        counts = np.round(counts).astype(np.int32)

        # Note that we can get values outside of the value range for counts
        # here if a StochasticParameter was provided, e.g.
        # Deterministic(1) is currently not verified.
        counts = np.clip(counts, self.counts_value_range[0], self.counts_value_range[1])

        return counts

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

        images = batch.images
        rss = random_state.duplicate(1 + len(images))
        counts = self._draw_samples(len(images), rss[-1])

        for i, image in enumerate(images):
            batch.images[i] = self._augment_single_image(image, counts[i], rss[i])
        return batch

    def _augment_single_image(self, image: Array, counts: int, random_state: iarandom.RNG) -> Array:
        assert image.shape[-1] in [1, 3, 4], (
            f"Expected image with 1, 3 or 4 channels, got {image.shape[-1]} (shape: {image.shape})."
        )

        orig_shape = image.shape
        image = self._ensure_max_size(image, self.max_size, self.interpolation)

        if image.shape[-1] == 1:
            # 2D image
            image_aug = self._quantize(image, counts)
        else:
            # 3D image with 3 or 4 channels
            alpha_channel = None
            if image.shape[-1] == 4:
                alpha_channel = image[:, :, 3:4]
                image = image[:, :, 0:3]

            if self.to_colorspace is None:
                cs = meta.Identity()
                cs_inv = meta.Identity()
            else:
                from . import ChangeColorspace as change_colorspace_cls

                # We use random_state.copy() in this method, but that is not
                # expected to cause unchanged an random_state, because
                # _augment_batch_() uses an un-copied one for _draw_samples()
                cs = change_colorspace_cls(
                    from_colorspace=self.from_colorspace,
                    to_colorspace=self.to_colorspace,
                    random_state=random_state.copy(),
                )
                to_colorspace = cs._draw_to_colorspace(random_state.copy())
                cs_inv = change_colorspace_cls(
                    from_colorspace=to_colorspace,
                    to_colorspace=self.from_colorspace,
                    random_state=random_state.copy(),
                )

            image_tf = cs.augment_image(image)
            image_tf_aug = self._quantize(image_tf, counts)
            image_aug = cs_inv.augment_image(image_tf_aug)

            if alpha_channel is not None:
                image_aug = np.concatenate([image_aug, alpha_channel], axis=2)

        if orig_shape != image_aug.shape:
            image_aug = ia.imresize_single_image(
                image_aug, orig_shape[0:2], interpolation=self.interpolation
            )

        return image_aug

    @abstractmethod
    def _quantize(self, image: Array, counts: int) -> Array:
        """Apply the augmenter-specific quantization function to an image."""

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [
            self.counts,
            self.from_colorspace,
            self.to_colorspace,
            self.max_size,
            self.interpolation,
        ]

    @classmethod
    def _ensure_max_size(cls, image: Array, max_size: int | None, interpolation: str) -> Array:
        if max_size is not None:
            size = max(image.shape[0], image.shape[1])
            if size > max_size:
                resize_factor = max_size / size
                new_height = int(image.shape[0] * resize_factor)
                new_width = int(image.shape[1] * resize_factor)
                image = ia.imresize_single_image(
                    image, (new_height, new_width), interpolation=interpolation
                )
        return image


class KMeansColorQuantization(_AbstractColorQuantization):
    """
    Quantize colors using k-Means clustering.

    This "collects" the colors from the input image, groups them into
    ``k`` clusters using k-Means clustering and replaces the colors in the
    input image using the cluster centroids.

    This is slower than ``UniformColorQuantization``, but adapts dynamically
    to the color range in the input image.

    .. note::

        This augmenter expects input images to be either grayscale
        or to have 3 or 4 channels and use colorspace `from_colorspace`. If
        images have 4 channels, it is assumed that the 4th channel is an alpha
        channel and it will not be quantized.

    **Supported dtypes**:

    if (image size <= max_size):

        minimum of (
            :class:`~imgaug2.augmenters.color.ChangeColorspace`,
            :func:`~imgaug2.augmenters.color.quantize_kmeans`
        )

    if (image size > max_size):

        minimum of (
            :class:`~imgaug2.augmenters.color.ChangeColorspace`,
            :func:`~imgaug2.augmenters.color.quantize_kmeans`,
            :func:`~imgaug2.imgaug2.imresize_single_image`
        )

    Parameters
    ----------
    n_colors : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Target number of colors in the generated output image.
        This corresponds to the number of clusters in k-Means, i.e. ``k``.
        Sampled values below ``2`` will always be clipped to ``2``.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    to_colorspace : None or str or list of str or imgaug2.parameters.StochasticParameter
        The colorspace in which to perform the quantization.
        See :func:`~imgaug2.augmenters.color.change_colorspace_` for valid values.
        This will be ignored for grayscale input images.

            * If ``None`` the colorspace of input images will not be changed.
            * If a string, it must be among the allowed colorspaces.
            * If a list, it is expected to be a list of strings, each one
              being an allowed colorspace. A random element from the list
              will be chosen per image.
            * If a StochasticParameter, it is expected to return string. A new
              sample will be drawn per image.

    from_colorspace : str, optional
        The colorspace of the input images.
        See `to_colorspace`. Only a single string is allowed.

    max_size : int or None, optional
        Maximum image size at which to perform the augmentation.
        If the width or height of an image exceeds this value, it will be
        downscaled before running the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the augmentation. The final output image has
        the same size as the input image. Use ``None`` to apply no downscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug2.imgaug2.imresize_single_image`.

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
    >>> aug = iaa.KMeansColorQuantization()

    Create an augmenter to apply k-Means color quantization to images using a
    random amount of colors, sampled uniformly from the interval ``[2..16]``.
    It assumes the input image colorspace to be ``RGB`` and clusters colors
    randomly in ``RGB`` or ``Lab`` colorspace.

    >>> aug = iaa.KMeansColorQuantization(n_colors=8)

    Create an augmenter that quantizes images to (up to) eight colors.

    >>> aug = iaa.KMeansColorQuantization(n_colors=(4, 16))

    Create an augmenter that quantizes images to (up to) ``n`` colors,
    where ``n`` is randomly and uniformly sampled from the discrete interval
    ``[4..16]``.

    >>> aug = iaa.KMeansColorQuantization(
    >>>     from_colorspace=iaa.CSPACE_BGR)

    Create an augmenter that quantizes input images that are in
    ``BGR`` colorspace. The quantization happens in ``RGB`` or ``Lab``
    colorspace, into which the images are temporarily converted.

    >>> aug = iaa.KMeansColorQuantization(
    >>>     to_colorspace=[iaa.CSPACE_RGB, iaa.CSPACE_HSV])

    Create an augmenter that quantizes images by clustering colors randomly
    in either ``RGB`` or ``HSV`` colorspace. The assumed input colorspace
    of images is ``RGB``.

    """

    def __init__(
        self,
        n_colors: ParamInput = (2, 16),
        from_colorspace: ColorSpace = CSPACE_RGB,
        to_colorspace: ToColorspaceChoiceInput | None = None,
        max_size: int | None = 128,
        interpolation: str = "linear",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        if to_colorspace is None:
            to_colorspace = [CSPACE_RGB, CSPACE_Lab]
        super().__init__(
            counts=n_colors,
            from_colorspace=from_colorspace,
            to_colorspace=to_colorspace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

    @legacy(version="0.4.0")
    @property
    def n_colors(self) -> iap.StochasticParameter:
        """Alias for property ``counts``."""
        return self.counts

    def _quantize(self, image: Array, counts: int) -> Array:
        from . import quantize_kmeans as quantize_kmeans_fn

        return quantize_kmeans_fn(image, counts)


@ia.deprecated("imgaug2.augmenters.colors.quantize_kmeans")
def quantize_colors_kmeans(
    image: Array, n_colors: int, n_max_iter: int = 10, eps: float = 1.0
) -> Array:
    """Outdated name of :func:`quantize_kmeans`.

    Deprecated since 0.4.0.

    """
    from . import quantize_kmeans as quantize_kmeans_fn

    return quantize_kmeans_fn(arr=image, nb_clusters=n_colors, nb_max_iter=n_max_iter, eps=eps)


@legacy(version="0.4.0")
def quantize_kmeans(arr: Array, nb_clusters: int, nb_max_iter: int = 10, eps: float = 1.0) -> Array:
    """Quantize an array into N bins using k-means clustering.

    If the input is an image, this method returns in an image with a maximum
    of ``N`` colors. Similar colors are grouped to their mean. The k-means
    clustering happens across channels and not channelwise.

    Code similar to https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/
    py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html

    .. warning::

        This function currently changes the RNG state of both OpenCV's
        internal RNG and imgaug's global RNG. This is necessary in order
        to ensure that the k-means clustering happens deterministically.

    Previously called ``quantize_colors_kmeans()``.

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
    arr : ndarray
        Array to quantize. Expected to be of shape ``(H,W)`` or ``(H,W,C)``
        with ``C`` usually being ``1`` or ``3``.

    nb_clusters : int
        Number of clusters to quantize into, i.e. ``k`` in k-means clustering.
        This corresponds to the maximum number of colors in an output image.

    nb_max_iter : int, optional
        Maximum number of iterations that the k-means clustering algorithm
        is run.

    eps : float, optional
        Minimum change of all clusters per k-means iteration. If all clusters
        change by less than this amount in an iteration, the clustering is
        stopped.

    Returns
    -------
    ndarray
        Image with quantized colors.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> import numpy as np
    >>> image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
    >>> image_quantized = iaa.quantize_kmeans(image, 6)

    Generates a ``4x4`` image with ``3`` channels, containing consecutive
    values from ``0`` to ``4*4*3``, leading to an equal number of colors.
    These colors are then quantized so that only ``6`` are remaining. Note
    that the six remaining colors do have to appear in the input image.

    """
    iadt.allow_only_uint8({arr.dtype})
    assert arr.ndim in [2, 3], (
        f"Expected two- or three-dimensional array shape, got shape {arr.shape}."
    )
    assert 2 <= nb_clusters <= 256, (
        "Expected nb_clusters to be in the discrete interval [2..256]. "
        f"Got a value of {nb_clusters} instead."
    )

    # without this check, kmeans throws an exception
    n_pixels = np.prod(arr.shape[0:2])
    if nb_clusters >= n_pixels:
        return np.copy(arr)

    nb_channels = 1 if arr.ndim == 2 else arr.shape[-1]
    pixel_vectors = arr.reshape((-1, nb_channels)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, nb_max_iter, eps)
    attempts = 1

    # We want our quantization function to be deterministic (so that the
    # augmenter using it can also be executed deterministically). Hence we
    # set the RGN seed here.
    # This is fairly ugly, but in cv2 there seems to be no other way to
    # achieve determinism. Using cv2.KMEANS_PP_CENTERS does not help, as it
    # is non-deterministic (tested). In C++ the function has an rgn argument,
    # but not in python. In python there also seems to be no way to read out
    # cv2's RNG state, so we can't set it back after executing this function.
    cv2.setRNGSeed(1)
    _compactness, labels, centers = cv2.kmeans(
        pixel_vectors, nb_clusters, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS
    )
    # cv2 seems to be able to handle SEED_MAX_VALUE (tested) but not floats
    cv2.setRNGSeed(iarandom.get_global_rng().generate_seed_())

    # Convert back to uint8 (or whatever the image dtype was) and to input
    # image shape
    centers_uint8 = np.array(centers, dtype=arr.dtype)
    quantized_flat = centers_uint8[labels.flatten()]
    return quantized_flat.reshape(arr.shape)


class UniformColorQuantization(_AbstractColorQuantization):
    """Quantize colors into N bins with regular distance.

    For ``uint8`` images the equation is ``floor(v/q)*q + q/2`` with
    ``q = 256/N``, where ``v`` is a pixel intensity value and ``N`` is
    the target number of colors after quantization.

    This augmenter is faster than ``KMeansColorQuantization``, but the
    set of possible output colors is constant (i.e. independent of the
    input images). It may produce unsatisfying outputs for input images
    that are made up of very similar colors.

    .. note::

        This augmenter expects input images to be either grayscale
        or to have 3 or 4 channels and use colorspace `from_colorspace`. If
        images have 4 channels, it is assumed that the 4th channel is an alpha
        channel and it will not be quantized.

    **Supported dtypes**:

    if (image size <= max_size):

        minimum of (
            :class:`~imgaug2.augmenters.color.ChangeColorspace`,
            :func:`~imgaug2.augmenters.color.quantize_uniform_`
        )

    if (image size > max_size):

        minimum of (
            :class:`~imgaug2.augmenters.color.ChangeColorspace`,
            :func:`~imgaug2.augmenters.color.quantize_uniform_`,
            :func:`~imgaug2.imgaug2.imresize_single_image`
        )

    Parameters
    ----------
    n_colors : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Target number of colors to use in the generated output image.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    to_colorspace : None or str or list of str or imgaug2.parameters.StochasticParameter
        The colorspace in which to perform the quantization.
        See :func:`~imgaug2.augmenters.color.change_colorspace_` for valid values.
        This will be ignored for grayscale input images.

            * If ``None`` the colorspace of input images will not be changed.
            * If a string, it must be among the allowed colorspaces.
            * If a list, it is expected to be a list of strings, each one
              being an allowed colorspace. A random element from the list
              will be chosen per image.
            * If a StochasticParameter, it is expected to return string. A new
              sample will be drawn per image.

    from_colorspace : str, optional
        The colorspace of the input images.
        See `to_colorspace`. Only a single string is allowed.

    max_size : None or int, optional
        Maximum image size at which to perform the augmentation.
        If the width or height of an image exceeds this value, it will be
        downscaled before running the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the augmentation. The final output image has
        the same size as the input image. Use ``None`` to apply no downscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug2.imgaug2.imresize_single_image`.

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
    >>> aug = iaa.UniformColorQuantization()

    Create an augmenter to apply uniform color quantization to images using a
    random amount of colors, sampled uniformly from the discrete interval
    ``[2..16]``.

    >>> aug = iaa.UniformColorQuantization(n_colors=8)

    Create an augmenter that quantizes images to (up to) eight colors.

    >>> aug = iaa.UniformColorQuantization(n_colors=(4, 16))

    Create an augmenter that quantizes images to (up to) ``n`` colors,
    where ``n`` is randomly and uniformly sampled from the discrete interval
    ``[4..16]``.

    >>> aug = iaa.UniformColorQuantization(
    >>>     from_colorspace=iaa.CSPACE_BGR,
    >>>     to_colorspace=[iaa.CSPACE_RGB, iaa.CSPACE_HSV])

    Create an augmenter that uniformly quantizes images in either ``RGB``
    or ``HSV`` colorspace (randomly picked per image). The input colorspace
    of all images has to be ``BGR``.

    """

    def __init__(
        self,
        n_colors: ParamInput = (2, 16),
        from_colorspace: ColorSpace = CSPACE_RGB,
        to_colorspace: ToColorspaceChoiceInput | None = None,
        max_size: int | None = None,
        interpolation: str = "linear",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            counts=n_colors,
            from_colorspace=from_colorspace,
            to_colorspace=to_colorspace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

    @legacy(version="0.4.0")
    @property
    def n_colors(self) -> iap.StochasticParameter:
        """Alias for property ``counts``."""
        return self.counts

    def _quantize(self, image: Array, counts: int) -> Array:
        from . import quantize_uniform_ as quantize_uniform_fn

        return quantize_uniform_fn(image, counts)


@legacy(version="0.4.0")
class UniformColorQuantizationToNBits(_AbstractColorQuantization):
    """Quantize images by setting ``8-B`` bits of each component to zero.

    This augmenter sets the ``8-B`` highest frequency (rightmost) bits of
    each array component to zero. For ``B`` bits this is equivalent to
    changing each component's intensity value ``v`` to
    ``v' = v & (2**(8-B) - 1)``, e.g. for ``B=3`` this results in
    ``v' = c & ~(2**(3-1) - 1) = c & ~3 = c & ~0000 0011 = c & 1111 1100``.

    This augmenter behaves for ``B`` similarly to
    ``UniformColorQuantization(2**B)``, but quantizes each bin with interval
    ``(a, b)`` to ``a`` instead of to ``a + (b-a)/2``.

    This augmenter is comparable to :func:`PIL.ImageOps.posterize`.

    .. note::

        This augmenter expects input images to be either grayscale
        or to have 3 or 4 channels and use colorspace `from_colorspace`. If
        images have 4 channels, it is assumed that the 4th channel is an alpha
        channel and it will not be quantized.


    **Supported dtypes**:

    if (image size <= max_size):

        minimum of (
            :class:`~imgaug2.augmenters.color.ChangeColorspace`,
            :func:`~imgaug2.augmenters.color.quantize_uniform`
        )

    if (image size > max_size):

        minimum of (
            :class:`~imgaug2.augmenters.color.ChangeColorspace`,
            :func:`~imgaug2.augmenters.color.quantize_uniform`,
            :func:`~imgaug2.imgaug2.imresize_single_image`
        )

    Parameters
    ----------
    nb_bits : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Number of bits to keep in each image's array component.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    to_colorspace : None or str or list of str or imgaug2.parameters.StochasticParameter
        The colorspace in which to perform the quantization.
        See :func:`~imgaug2.augmenters.color.change_colorspace_` for valid values.
        This will be ignored for grayscale input images.

            * If ``None`` the colorspace of input images will not be changed.
            * If a string, it must be among the allowed colorspaces.
            * If a list, it is expected to be a list of strings, each one
              being an allowed colorspace. A random element from the list
              will be chosen per image.
            * If a StochasticParameter, it is expected to return string. A new
              sample will be drawn per image.

    from_colorspace : str, optional
        The colorspace of the input images.
        See `to_colorspace`. Only a single string is allowed.

    max_size : None or int, optional
        Maximum image size at which to perform the augmentation.
        If the width or height of an image exceeds this value, it will be
        downscaled before running the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the augmentation. The final output image has
        the same size as the input image. Use ``None`` to apply no downscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug2.imgaug2.imresize_single_image`.

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
    >>> aug = iaa.UniformColorQuantizationToNBits()

    Create an augmenter to apply uniform color quantization to images using a
    random amount of bits to remove, sampled uniformly from the discrete
    interval ``[1..8]``.

    >>> aug = iaa.UniformColorQuantizationToNBits(nb_bits=(2, 8))

    Create an augmenter that quantizes images by removing ``8-B`` rightmost
    bits from each component, where ``B`` is uniformly sampled from the
    discrete interval ``[2..8]``.

    >>> aug = iaa.UniformColorQuantizationToNBits(
    >>>     from_colorspace=iaa.CSPACE_BGR,
    >>>     to_colorspace=[iaa.CSPACE_RGB, iaa.CSPACE_HSV])

    Create an augmenter that uniformly quantizes images in either ``RGB``
    or ``HSV`` colorspace (randomly picked per image). The input colorspace
    of all images has to be ``BGR``.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        nb_bits: ParamInput = (1, 8),
        from_colorspace: ColorSpace = CSPACE_RGB,
        to_colorspace: ToColorspaceChoiceInput | None = None,
        max_size: int | None = None,
        interpolation: str = "linear",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        # wrt value range: for discrete params, (1, 8) results in
        # DiscreteUniform with interval [1, 8]
        super().__init__(
            counts=nb_bits,
            counts_value_range=(1, 8),
            from_colorspace=from_colorspace,
            to_colorspace=to_colorspace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

    @legacy(version="0.4.0")
    def _quantize(self, image: Array, counts: int) -> Array:
        from . import quantize_uniform_to_n_bits_ as quantize_uniform_to_n_bits_fn

        return quantize_uniform_to_n_bits_fn(image, counts)


@legacy(version="0.4.0")
class Posterize(UniformColorQuantizationToNBits):
    """Alias for :class:`UniformColorQuantizationToNBits`.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.color.UniformColorQuantizationToNBits`.

    """


@ia.deprecated("imgaug2.augmenters.colors.quantize_uniform")
def quantize_colors_uniform(image: Array, n_colors: int) -> Array:
    """Outdated name for :func:`quantize_uniform`.

    Deprecated since 0.4.0.

    """
    from . import quantize_uniform as quantize_uniform_fn

    return quantize_uniform_fn(arr=image, nb_bins=n_colors)


@legacy(version="0.4.0")
def quantize_uniform(arr: Array, nb_bins: int, to_bin_centers: bool = True) -> Array:
    """Quantize an array into N equally-sized bins.

    See :func:`quantize_uniform_` for details.

    Previously called ``quantize_colors_uniform()``.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.color.quantize_uniform_`.

    Parameters
    ----------
    arr : ndarray
        See :func:`quantize_uniform_`.

    nb_bins : int
        See :func:`quantize_uniform_`.

    to_bin_centers : bool
        See :func:`quantize_uniform_`.

    Returns
    -------
    ndarray
        Array with quantized components.

    """
    from . import quantize_uniform_ as quantize_uniform_fn

    return quantize_uniform_fn(np.copy(arr), nb_bins=nb_bins, to_bin_centers=to_bin_centers)


@legacy(version="0.4.0")
def quantize_uniform_(arr: Array, nb_bins: int, to_bin_centers: bool = True) -> Array:
    """Quantize an array into N equally-sized bins in-place.

    This can be used to quantize/posterize an image into N colors.

    For ``uint8`` arrays the equation is ``floor(v/q)*q + q/2`` with
    ``q = 256/N``, where ``v`` is a pixel intensity value and ``N`` is
    the target number of bins (roughly matches number of colors) after
    quantization.


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
    arr : ndarray
        Array to quantize, usually an image. Expected to be of shape ``(H,W)``
        or ``(H,W,C)`` with ``C`` usually being ``1`` or ``3``.
        This array *may* be changed in-place.

    nb_bins : int
        Number of equally-sized bins to quantize into. This corresponds to
        the maximum number of colors in an output image.

    to_bin_centers : bool
        Whether to quantize each bin ``(a, b)`` to ``a + (b-a)/2`` (center
        of bin, ``True``) or to ``a`` (lower boundary, ``False``).

    Returns
    -------
    ndarray
        Array with quantized components. This *may* be the input array with
        components changed in-place.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> import numpy as np
    >>> image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
    >>> image_quantized = iaa.quantize_uniform_(np.copy(image), 6)

    Generates a ``4x4`` image with ``3`` channels, containing consecutive
    values from ``0`` to ``4*4*3``, leading to an equal number of colors.
    Each component is then quantized into one of ``6`` bins that regularly
    split up the value range of ``[0..255]``, i.e. the resolution w.r.t. to
    the value range is reduced.

    """
    if nb_bins == 256 or 0 in arr.shape:
        return arr

    # TODO remove dtype check here? apply_lut_() does that already
    iadt.allow_only_uint8({arr.dtype})
    assert 2 <= nb_bins <= 256, (
        "Expected nb_bins to be in the discrete interval [2..256]. "
        f"Got a value of {nb_bins} instead."
    )

    table_class = (
        _QuantizeUniformCenterizedLUTTableSingleton
        if to_bin_centers
        else _QuantizeUniformNotCenterizedLUTTableSingleton
    )
    table = table_class.get_instance().get_for_nb_bins(nb_bins)
    arr = ia.apply_lut_(arr, table)
    return arr


@legacy(version="0.4.0")
def quantize_uniform_to_n_bits(arr: Array, nb_bits: int) -> Array:
    """Reduce each component in an array to a maximum number of bits.

    See :func:`quantize_uniform_to_n_bits` for details.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.color.quantize_uniform_to_n_bits_`.

    Parameters
    ----------
    arr : ndarray
        See :func:`quantize_uniform_to_n_bits`.

    nb_bits : int
        See :func:`quantize_uniform_to_n_bits`.

    Returns
    -------
    ndarray
        Array with quantized components.

    """
    from . import quantize_uniform_to_n_bits_ as quantize_uniform_to_n_bits_fn

    return quantize_uniform_to_n_bits_fn(np.copy(arr), nb_bits=nb_bits)


@legacy(version="0.4.0")
def quantize_uniform_to_n_bits_(arr: Array, nb_bits: int) -> Array:
    """Reduce each component in an array to a maximum number of bits in-place.

    This operation sets the ``8-B`` highest frequency (rightmost) bits to zero.
    For ``B`` bits this is equivalent to changing each component's intensity
    value ``v`` to ``v' = v & (2**(8-B) - 1)``, e.g. for ``B=3`` this results
    in ``v' = c & ~(2**(3-1) - 1) = c & ~3 = c & ~0000 0011 = c & 1111 1100``.

    This is identical to :func:`quantize_uniform` with ``nb_bins=2**nb_bits``
    and ``to_bin_centers=False``.

    This function produces the same outputs as :func:`PIL.ImageOps.posterize`,
    but is significantly faster.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.color.quantize_uniform_`.

    Parameters
    ----------
    arr : ndarray
        Array to quantize, usually an image. Expected to be of shape ``(H,W)``
        or ``(H,W,C)`` with ``C`` usually being ``1`` or ``3``.
        This array *may* be changed in-place.

    nb_bits : int
        Number of bits to keep in each array component.

    Returns
    -------
    ndarray
        Array with quantized components. This *may* be the input array with
        components changed in-place.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> import numpy as np
    >>> image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
    >>> image_quantized = iaa.quantize_uniform_to_n_bits_(np.copy(image), 6)

    Generates a ``4x4`` image with ``3`` channels, containing consecutive
    values from ``0`` to ``4*4*3``, leading to an equal number of colors.
    These colors are then quantized so that each component's ``8-6=2``
    rightmost bits are set to zero.

    """
    assert 1 <= nb_bits <= 8, (
        f"Expected nb_bits to be in the discrete interval [1..8]. Got a value of {nb_bits} instead."
    )
    from . import quantize_uniform_ as quantize_uniform_fn

    return quantize_uniform_fn(arr, nb_bins=2**nb_bits, to_bin_centers=False)


@legacy(version="0.4.0")
def posterize(arr: Array, nb_bits: int) -> Array:
    """Alias for :func:`quantize_uniform_to_n_bits`.

    This function is an alias for :func:`quantize_uniform_to_n_bits` and was
    added for users familiar with the same function in PIL.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.color.quantize_uniform_to_n_bits`.

    Parameters
    ----------
    arr : ndarray
        See :func:`quantize_uniform_to_n_bits`.

    nb_bits : int
        See :func:`quantize_uniform_to_n_bits`.

    Returns
    -------
    ndarray
        Array with quantized components.

    """
    from . import quantize_uniform_to_n_bits as quantize_uniform_to_n_bits_fn

    return quantize_uniform_to_n_bits_fn(arr, nb_bits)
