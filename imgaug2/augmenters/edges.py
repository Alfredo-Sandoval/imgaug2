"""
Augmenters that deal with edge detection.

List of augmenters:

    * :class:`Canny`

:class:`~imgaug2.augmenters.convolutional.EdgeDetect` and
:class:`~imgaug2.augmenters.convolutional.DirectedEdgeDetect` are currently
still in ``convolutional.py``.

"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Literal, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import blend, meta
from imgaug2.augmenters._typing import ParamInput, RNGInput
from imgaug2.imgaug import _normalize_cv2_input_arr_

ImageArray: TypeAlias = NDArray[np.generic]
BinaryMask: TypeAlias = NDArray[np.bool_]
ColorSamples: TypeAlias = NDArray[np.integer]
FloatArray: TypeAlias = NDArray[np.floating]
NumberArray: TypeAlias = NDArray[np.number]
IntArray: TypeAlias = NDArray[np.integer]

DiscreteParamInput: TypeAlias = int | tuple[int, int] | list[int] | iap.StochasticParameter

# Either a single param, or a tuple of two params (min,max).
HysteresisThresholdsInput: TypeAlias = ParamInput | tuple[ParamInput, ParamInput]


# TODO this should be placed in some other file than edges.py as it could be
#      re-used wherever a binary image is the result
class IBinaryImageColorizer(metaclass=ABCMeta):
    """Interface for classes that convert binary masks to color images."""

    @abstractmethod
    def colorize(
        self,
        image_binary: BinaryMask,
        image_original: ImageArray,
        nth_image: int,
        random_state: iarandom.RNG,
    ) -> ImageArray:
        """
        Convert a binary image to a colorized one.

        Parameters
        ----------
        image_binary : ndarray
            Boolean ``(H,W)`` image.

        image_original : ndarray
            Original ``(H,W,C)`` input image.

        nth_image : int
            Index of the image in the batch.

        random_state : imgaug2.random.RNG
            Random state to use.

        Returns
        -------
        ndarray
            Colorized form of `image_binary`.

        """


# TODO see above, this should be moved to another file
class RandomColorsBinaryImageColorizer(IBinaryImageColorizer):
    """
    Colorizer using two randomly sampled foreground/background colors.

    Parameters
    ----------
    color_true : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Color of the foreground, i.e. all pixels in binary images that are
        ``True``. This parameter will be queried once per image to
        generate ``(3,)`` samples denoting the color. (Note that even for
        grayscale images three values will be sampled and converted to
        grayscale according to ``0.299*R + 0.587*G + 0.114*B``. This is the
        same equation that is also used by OpenCV.)

            * If an int, exactly that value will always be used, i.e. every
              color will be ``(v, v, v)`` for value ``v``.
            * If a tuple ``(a, b)``, three random values from the range
              ``a <= x <= b`` will be sampled per image.
            * If a list, then three random values will be sampled from that
              list per image.
            * If a StochasticParameter, three values will be sampled from the
              parameter per image.

    color_false : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Analogous to `color_true`, but denotes the color for all pixels that
        are ``False`` in the binary input image.

    """

    def __init__(
        self, color_true: DiscreteParamInput = (0, 255), color_false: DiscreteParamInput = (0, 255)
    ) -> None:
        self.color_true = iap.handle_discrete_param(
            color_true,
            "color_true",
            value_range=(0, 255),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )

        self.color_false = iap.handle_discrete_param(
            color_false,
            "color_false",
            value_range=(0, 255),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )

    def _draw_samples(self, random_state: iarandom.RNG) -> tuple[ColorSamples, ColorSamples]:
        color_true = self.color_true.draw_samples((3,), random_state=random_state)
        color_false = self.color_false.draw_samples((3,), random_state=random_state)
        return color_true, color_false

    def colorize(
        self,
        image_binary: BinaryMask,
        image_original: ImageArray,
        nth_image: int,
        random_state: iarandom.RNG,
    ) -> ImageArray:
        assert image_binary.ndim == 2, (
            "Expected binary image to colorize to be 2-dimensional, got "
            f"{image_binary.ndim} dimensions."
        )
        assert image_binary.dtype.kind == "b", (
            "Expected binary image to colorize to be boolean, "
            f"got dtype kind {image_binary.dtype.kind}."
        )
        assert image_original.ndim == 3, (
            "Expected original image to be 3-dimensional, got "
            f"{image_original.ndim} dimensions."
        )
        assert image_original.shape[-1] in [1, 3, 4], (
            "Expected original image to have 1, 3 or 4 channels. Got "
            f"{image_original.shape[-1]} channels."
        )
        assert image_original.dtype == iadt._UINT8_DTYPE, (
            f"Expected original image to have dtype uint8, got dtype {image_original.dtype.name}."
        )

        color_true, color_false = self._draw_samples(random_state)

        nb_channels = min(image_original.shape[-1], 3)
        image_colorized = np.zeros(
            (image_original.shape[0], image_original.shape[1], nb_channels),
            dtype=image_original.dtype,
        )

        if nb_channels == 1:
            # single channel input image, convert colors to grayscale
            image_colorized[image_binary] = (
                0.299 * color_true[0] + 0.587 * color_true[1] + 0.114 * color_true[2]
            )
            image_colorized[~image_binary] = (
                0.299 * color_false[0] + 0.587 * color_false[1] + 0.114 * color_false[2]
            )
        else:
            image_colorized[image_binary] = color_true
            image_colorized[~image_binary] = color_false

        # re-attach alpha channel if it was present in input image
        if image_original.shape[-1] == 4:
            image_colorized = np.dstack([image_colorized, image_original[:, :, 3:4]])

        return image_colorized

    def __str__(self) -> str:
        return (
            "RandomColorsBinaryImageColorizer("
            f"color_true={self.color_true}, color_false={self.color_false})"
        )


class Canny(meta.Augmenter):
    """
    Apply a canny edge detector to input images.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no; not tested
        * ``uint32``: no; not tested
        * ``uint64``: no; not tested
        * ``int8``: no; not tested
        * ``int16``: no; not tested
        * ``int32``: no; not tested
        * ``int64``: no; not tested
        * ``float16``: no; not tested
        * ``float32``: no; not tested
        * ``float64``: no; not tested
        * ``float128``: no; not tested
        * ``bool``: no; not tested

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Blending factor to use in alpha blending.
        A value close to 1.0 means that only the edge image is visible.
        A value close to 0.0 means that only the original image is visible.
        A value close to 0.5 means that the images are merged according to
        `0.5*image + 0.5*edge_image`.
        If a sample from this parameter is 0, no action will be performed for
        the corresponding image.

            * If an int or float, exactly that value will be used.
            * If a tuple ``(a, b)``, a random value from the range
              ``a <= x <= b`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    hysteresis_thresholds : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter or tuple of tuple of number or tuple of list of number or tuple of imgaug2.parameters.StochasticParameter, optional
        Min and max values to use in hysteresis thresholding.
        (This parameter seems to have not very much effect on the results.)
        Either a single parameter or a tuple of two parameters.
        If a single parameter is provided, the sampling happens once for all
        images with `(N,2)` samples being requested from the parameter,
        where each first value denotes the hysteresis minimum and each second
        the maximum.
        If a tuple of two parameters is provided, one sampling of `(N,)` values
        is independently performed per parameter (first parameter: hysteresis
        minimum, second: hysteresis maximum).

            * If this is a single number, both min and max value will always be
              exactly that value.
            * If this is a tuple of numbers ``(a, b)``, two random values from
              the range ``a <= x <= b`` will be sampled per image.
            * If this is a list, two random values will be sampled from that
              list per image.
            * If this is a StochasticParameter, two random values will be
              sampled from that parameter per image.
            * If this is a tuple ``(min, max)`` with ``min`` and ``max``
              both *not* being numbers, they will be treated according to the
              rules above (i.e. may be a number, tuple, list or
              StochasticParameter). A single value will be sampled per image
              and parameter.

    sobel_kernel_size : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Kernel size of the sobel operator initially applied to each image.
        This corresponds to ``apertureSize`` in ``cv2.Canny()``.
        If a sample from this parameter is ``<=1``, no action will be performed
        for the corresponding image.
        The maximum for this parameter is ``7`` (inclusive). Higher values are
        not accepted by OpenCV.
        If an even value ``v`` is sampled, it is automatically changed to
        ``v-1``.

            * If this is a single integer, the kernel size always matches that
              value.
            * If this is a tuple of integers ``(a, b)``, a random discrete
              value will be sampled from the range ``a <= x <= b`` per image.
            * If this is a list, a random value will be sampled from that
              list per image.
            * If this is a StochasticParameter, a random value will be sampled
              from that parameter per image.

    colorizer : None or imgaug2.augmenters.edges.IBinaryImageColorizer, optional
        A strategy to convert binary edge images to color images.
        If this is ``None``, an instance of ``RandomColorBinaryImageColorizer``
        is created, which means that each edge image is converted into an
        ``uint8`` image, where edge and non-edge pixels each have a different
        color that was uniformly randomly sampled from the space of all
        ``uint8`` colors.

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
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.Canny()

    Create an augmenter that generates random blends between images and
    their canny edge representations.

    >>> aug = iaa.Canny(alpha=(0.0, 0.5))

    Create a canny edge augmenter that generates edge images with a blending
    factor of max ``50%``, i.e. the original (non-edge) image is always at
    least partially visible.

    >>> aug = iaa.Canny(
    >>>     alpha=(0.0, 0.5),
    >>>     colorizer=iaa.RandomColorsBinaryImageColorizer(
    >>>         color_true=255,
    >>>         color_false=0
    >>>     )
    >>> )

    Same as in the previous example, but the edge image always uses the
    color white for edges and black for the background.

    >>> aug = iaa.Canny(alpha=(0.5, 1.0), sobel_kernel_size=[3, 7])

    Create a canny edge augmenter that initially preprocesses images using
    a sobel filter with kernel size of either ``3x3`` or ``13x13`` and
    alpha-blends with result using a strength of ``50%`` (both images
    equally visible) to ``100%`` (only edge image visible).

    >>> aug = iaa.Alpha(
    >>>     (0.0, 1.0),
    >>>     iaa.Canny(alpha=1),
    >>>     iaa.MedianBlur(13)
    >>> )

    Create an augmenter that blends a canny edge image with a median-blurred
    version of the input image. The median blur uses a fixed kernel size
    of ``13x13`` pixels.

    """

    def __init__(
        self,
        alpha: ParamInput = (0.0, 1.0),
        hysteresis_thresholds: HysteresisThresholdsInput = (
            (100 - 40, 100 + 40),
            (200 - 40, 200 + 40),
        ),
        sobel_kernel_size: DiscreteParamInput = (3, 7),
        colorizer: IBinaryImageColorizer | None = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.alpha = iap.handle_continuous_param(
            alpha, "alpha", value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True
        )

        if (
            isinstance(hysteresis_thresholds, tuple)
            and len(hysteresis_thresholds) == 2
            and not ia.is_single_number(hysteresis_thresholds[0])
            and not ia.is_single_number(hysteresis_thresholds[1])
        ):
            self.hysteresis_thresholds = (
                iap.handle_discrete_param(
                    hysteresis_thresholds[0],
                    "hysteresis_thresholds[0]",
                    value_range=(0, 255),
                    tuple_to_uniform=True,
                    list_to_choice=True,
                    allow_floats=True,
                ),
                iap.handle_discrete_param(
                    hysteresis_thresholds[1],
                    "hysteresis_thresholds[1]",
                    value_range=(0, 255),
                    tuple_to_uniform=True,
                    list_to_choice=True,
                    allow_floats=True,
                ),
            )
        else:
            self.hysteresis_thresholds = iap.handle_discrete_param(
                hysteresis_thresholds,
                "hysteresis_thresholds",
                value_range=(0, 255),
                tuple_to_uniform=True,
                list_to_choice=True,
                allow_floats=True,
            )

        # we don't use handle_discrete_kernel_size_param() here, because
        # cv2.Canny() can't handle independent height/width values, only a
        # single kernel size
        self.sobel_kernel_size = iap.handle_discrete_param(
            sobel_kernel_size,
            "sobel_kernel_size",
            value_range=(0, 7),  # OpenCV only accepts ksize up to 7
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )

        self.colorizer = colorizer if colorizer is not None else RandomColorsBinaryImageColorizer()

    def _draw_samples(
        self, augmentables: Sequence[ImageArray], random_state: iarandom.RNG
    ) -> tuple[FloatArray, NumberArray, IntArray]:
        nb_images = len(augmentables)
        rss = random_state.duplicate(4)

        alpha_samples = self.alpha.draw_samples((nb_images,), rss[0])

        hthresh = self.hysteresis_thresholds
        if isinstance(hthresh, tuple):
            min_values = hthresh[0].draw_samples((nb_images,), rss[1])
            max_values = hthresh[1].draw_samples((nb_images,), rss[2])
            hthresh_samples = np.stack([min_values, max_values], axis=-1)
        else:
            hthresh_samples = hthresh.draw_samples((nb_images, 2), rss[1])

        sobel_samples = self.sobel_kernel_size.draw_samples((nb_images,), rss[3])

        # verify for hysteresis thresholds that min_value < max_value everywhere
        invalid = hthresh_samples[:, 0] > hthresh_samples[:, 1]
        if np.any(invalid):
            hthresh_samples[invalid, :] = hthresh_samples[invalid, :][:, [1, 0]]

        # ensure that sobel kernel sizes are correct
        # note that OpenCV accepts only kernel sizes that are (a) even
        # and (b) <=7
        assert not np.any(sobel_samples < 0), (
            "Sampled a sobel kernel size below 0 in Canny. Allowed value range is 0 to 7."
        )
        assert not np.any(sobel_samples > 7), (
            "Sampled a sobel kernel size above 7 in Canny. Allowed value range is 0 to 7."
        )
        even_idx = np.mod(sobel_samples, 2) == 0
        sobel_samples[even_idx] -= 1

        return alpha_samples, hthresh_samples, sobel_samples

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

        images = batch.images

        iadt.allow_only_uint8(images, augmenter=self)

        rss = random_state.duplicate(len(images))
        samples = self._draw_samples(images, rss[-1])
        alpha_samples = samples[0]
        hthresh_samples = samples[1]
        sobel_samples = samples[2]

        gen = enumerate(zip(images, alpha_samples, hthresh_samples, sobel_samples, strict=True))
        for i, (image, alpha, hthreshs, sobel) in gen:
            assert image.shape[-1] in [1, 3, 4], (
                "Canny edge detector can currently only handle images with "
                "channel numbers that are 1, 3 or 4. Got "
                f"{image.shape[-1]}."
            )

            has_zero_sized_axes = 0 in image.shape[0:2]
            if alpha > 0 and sobel > 1 and not has_zero_sized_axes:
                image_canny = cv2.Canny(
                    _normalize_cv2_input_arr_(image[:, :, 0:3]),
                    threshold1=hthreshs[0],
                    threshold2=hthreshs[1],
                    apertureSize=sobel,
                    L2gradient=True,
                )
                image_canny = image_canny > 0

                # canny returns a boolean (H,W) image, so we change it to
                # (H,W,C) and then uint8
                image_canny_color = self.colorizer.colorize(
                    image_canny, image, nth_image=i, random_state=rss[i]
                )

                batch.images[i] = blend.blend_alpha(image_canny_color, image, alpha)

        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.alpha, self.hysteresis_thresholds, self.sobel_kernel_size, self.colorizer]

    def __str__(self) -> str:
        return (
            "Canny("
            f"alpha={self.alpha}, "
            f"hysteresis_thresholds={self.hysteresis_thresholds}, "
            f"sobel_kernel_size={self.sobel_kernel_size}, "
            f"colorizer={self.colorizer}, "
            f"name={self.name}, "
            f"deterministic={self.deterministic})"
        )
