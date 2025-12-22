"""Histogram equalization contrast augmenters."""

from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import color as color_lib
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, RNGInput
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _normalize_cv2_input_arr_
import imgaug2.mlx.color as mlx_color

from ._intensity import _IntensityChannelBasedApplier
from ._utils import _is_mlx_list

class AllChannelsHistogramEqualization(meta.Augmenter):
    """
    Apply Histogram Eq. to all channels of images in their original colorspaces.

    In contrast to ``imgaug2.augmenters.contrast.HistogramEqualization``, this
    augmenter operates directly on all channels of the input images. It does
    not perform any colorspace transformations and does not focus on specific
    channels (e.g. ``L`` in ``Lab`` colorspace).

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no (1)
        * ``uint32``: no (2)
        * ``uint64``: no (1)
        * ``int8``: no (1)
        * ``int16``: no (1)
        * ``int32``: no (1)
        * ``int64``: no (1)
        * ``float16``: no (2)
        * ``float32``: no (1)
        * ``float64``: no (1)
        * ``float128``: no (2)
        * ``bool``: no (1)

        - (1) causes cv2 error: ``cv2.error:
              OpenCV(3.4.5) (...)/histogram.cpp:3345: error: (-215:Assertion
              failed) src.type() == CV_8UC1 in function 'equalizeHist'``
        - (2) rejected by cv2

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
    >>> aug = iaa.AllChannelsHistogramEqualization()

    Create an augmenter that applies histogram equalization to all channels
    of input images in the original colorspaces.

    >>> aug = iaa.BlendAlpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization())

    Same as in the previous example, but alpha-blends the contrast-enhanced
    augmented images with the original input images using random blend
    strengths. This leads to random strengths of the contrast adjustment.

    """

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
        if batch.images is None:
            return batch

        images = batch.images

        if _is_mlx_list(images):
            for i, image in enumerate(images):
                if image.size == 0:
                    continue
                images[i] = mlx_color.equalize(image)
            batch.images = images
            return batch

        iadt.allow_only_uint8(images, augmenter=self)

        for i, image in enumerate(images):
            if image.size == 0:
                continue

            image_warped = [
                cv2.equalizeHist(_normalize_cv2_input_arr_(image[..., c]))
                for c in range(image.shape[2])
            ]
            image_warped = np.array(image_warped, dtype=image_warped[0].dtype)
            image_warped = image_warped.transpose((1, 2, 0))

            batch.images[i] = image_warped
        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return []


class HistogramEqualization(meta.Augmenter):
    """
    Apply Histogram Eq. to L/V/L channels of images in HLS/HSV/Lab colorspaces.

    This augmenter is similar to ``imgaug2.augmenters.contrast.CLAHE``.

    The augmenter transforms input images to a target colorspace (e.g.
    ``Lab``), extracts an intensity-related channel from the converted images
    (e.g. ``L`` for ``Lab``), applies Histogram Equalization to the channel
    and then converts the resulting image back to the original colorspace.

    Grayscale images (images without channel axis or with only one channel
    axis) are automatically handled, `from_colorspace` does not have to be
    adjusted for them. For images with four channels (e.g. RGBA), the fourth
    channel is ignored in the colorspace conversion (e.g. from an ``RGBA``
    image, only the ``RGB`` part is converted, normalized, converted back and
    concatenated with the input ``A`` channel). Images with unusual channel
    numbers (2, 5 or more than 5) are normalized channel-by-channel (same
    behaviour as ``AllChannelsHistogramEqualization``, though a warning will
    be raised).

    If you want to apply HistogramEqualization to each channel of the original
    input image's colorspace (without any colorspace conversion), use
    ``imgaug2.augmenters.contrast.AllChannelsHistogramEqualization`` instead.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no (1)
        * ``uint32``: no (1)
        * ``uint64``: no (1)
        * ``int8``: no (1)
        * ``int16``: no (1)
        * ``int32``: no (1)
        * ``int64``: no (1)
        * ``float16``: no (1)
        * ``float32``: no (1)
        * ``float64``: no (1)
        * ``float128``: no (1)
        * ``bool``: no (1)

        - (1) This augmenter uses :class:`AllChannelsHistogramEqualization`,
              which only supports ``uint8``.

    Parameters
    ----------
    from_colorspace : {"RGB", "BGR", "HSV", "HLS", "Lab"}, optional
        Colorspace of the input images.
        If any input image has only one or zero channels, this setting will be
        ignored and it will be assumed that the input is grayscale.
        If a fourth channel is present in an input image, it will be removed
        before the colorspace conversion and later re-added.
        See also :func:`~imgaug2.augmenters.color.change_colorspace_` for
        details.

    to_colorspace : {"Lab", "HLS", "HSV"}, optional
        Colorspace in which to perform Histogram Equalization. For ``Lab``,
        the equalization will only be applied to the first channel (``L``),
        for ``HLS`` to the second (``L``) and for ``HSV`` to the third (``V``).
        To apply histogram equalization to all channels of an input image
        (without colorspace conversion), see
        ``imgaug2.augmenters.contrast.AllChannelsHistogramEqualization``.

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
    >>> aug = iaa.HistogramEqualization()

    Create an augmenter that converts images to ``HLS``/``HSV``/``Lab``
    colorspaces, extracts intensity-related channels (i.e. ``L``/``V``/``L``),
    applies histogram equalization to these channels and converts back to the
    input colorspace.

    >>> aug = iaa.BlendAlpha((0.0, 1.0), iaa.HistogramEqualization())

    Same as in the previous example, but alpha blends the result, leading
    to various strengths of contrast normalization.

    >>> aug = iaa.HistogramEqualization(
    >>>     from_colorspace=iaa.CSPACE_BGR,
    >>>     to_colorspace=iaa.CSPACE_HSV)

    Same as in the first example, but the colorspace of input images has
    to be ``BGR`` (instead of default ``RGB``) and the histogram equalization
    is applied to the ``V`` channel in ``HSV`` colorspace.

    """

    RGB = color_lib.CSPACE_RGB
    BGR = color_lib.CSPACE_BGR
    HSV = color_lib.CSPACE_HSV
    HLS = color_lib.CSPACE_HLS
    Lab = color_lib.CSPACE_Lab

    def __init__(
        self,
        from_colorspace: str = color_lib.CSPACE_RGB,
        to_colorspace: str = color_lib.CSPACE_Lab,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.all_channel_histogram_equalization = AllChannelsHistogramEqualization(
            name=f"{name}_AllChannelsHistogramEqualization"
        )

        self.intensity_channel_based_applier = _IntensityChannelBasedApplier(
            from_colorspace, to_colorspace
        )

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

        iadt.allow_only_uint8(images, augmenter=self)

        def _augment_all_channels_histogram_equalization(
            images_normalized: list[Array], random_state_derived: iarandom.RNG
        ) -> list[Array]:
            # TODO would .augment_batch() be sufficient here
            batch_imgs = _BatchInAugmentation(images=images_normalized)
            batch_out = self.all_channel_histogram_equalization._augment_batch_(
                batch_imgs, random_state_derived, parents + [self], hooks
            )
            assert batch_out.images is not None
            return cast(list[Array], batch_out.images)

        batch.images = self.intensity_channel_based_applier.apply(
            images,
            random_state,
            parents + [self],
            hooks,
            _augment_all_channels_histogram_equalization,
        )
        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        icb_applier = self.intensity_channel_based_applier
        return icb_applier.get_parameters()
