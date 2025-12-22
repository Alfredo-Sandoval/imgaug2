"""Superpixels augmentation."""

from __future__ import annotations

from typing import Literal

import numpy as np
import skimage

# use skimage.segmentation instead `from skimage import segmentation` here,
# because otherwise unittest seems to mix up imgaug2.augmenters.segmentation
# with skimage.segmentation for whatever reason
import skimage.segmentation

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import ParamInput, RNGInput
from imgaug2.compat.markers import legacy

from ._utils import _ensure_image_max_size
from .replace import replace_segments_

_SLIC_SUPPORTS_START_LABEL = tuple(map(int, skimage.__version__.split(".")[0:2])) >= (
    0,
    17,
)


# TODO add compactness parameter
class Superpixels(meta.Augmenter):
    """Transform images parially/completely to their superpixel representation.

    This implementation uses skimage's version of the SLIC algorithm.

    .. note::

        This augmenter is fairly slow. See :ref:`performance`.

    **Supported dtypes**:

    if (image size <= max_size):

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: limited (1)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: limited (1)
        * ``float16``: no (2)
        * ``float32``: no (2)
        * ``float64``: no (3)
        * ``float128``: no (2)
        * ``bool``: yes; tested

        - (1) Superpixel mean intensity replacement requires computing
              these means as ``float64`` s. This can cause inaccuracies for
              large integer values.
        - (2) Error in scikit-image.
        - (3) Loss of resolution in scikit-image.

    if (image size > max_size):

        minimum of (
            ``imgaug2.augmenters.segmentation.Superpixels(image size <= max_size)``,
            :func:`~imgaug2.augmenters.segmentation._ensure_image_max_size`
        )

    Parameters
    ----------
    p_replace : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color (otherwise, the pixels
        are not changed).
        Examples:

            * A probability of ``0.0`` would mean, that the pixels in no
              segment are replaced by their average color (image is not
              changed at all).
            * A probability of ``0.5`` would mean, that around half of all
              segments are replaced by their average color.
            * A probability of ``1.0`` would mean, that all segments are
              replaced by their average color (resulting in a voronoi
              image).

        Behaviour based on chosen datatypes for this parameter:

            * If a ``number``, then that ``number`` will always be used.
            * If ``tuple`` ``(a, b)``, then a random probability will be
              sampled from the interval ``[a, b]`` per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, it is expected to return
              values between ``0.0`` and ``1.0`` and will be queried *for each
              individual segment* to determine whether it is supposed to
              be averaged (``>0.5``) or not (``<=0.5``).
              Recommended to be some form of ``Binomial(...)``.

    n_segments : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Rough target number of how many superpixels to generate (the algorithm
        may deviate from this number). Lower value will lead to coarser
        superpixels. Higher values are computationally more intensive and
        will hence lead to a slowdown.

            * If a single ``int``, then that value will always be used as the
              number of segments.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.
        If the width or height of an image exceeds this value, it will be
        downscaled before the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the process. The final output image has the
        same size as the input image. Note that in case `p_replace` is below
        ``1.0``, the down-/upscaling will affect the not-replaced pixels too.
        Use ``None`` to apply no down-/upscaling.

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
    >>> aug = iaa.Superpixels(p_replace=1.0, n_segments=64)

    Generate around ``64`` superpixels per image and replace all of them with
    their average color (standard superpixel image).

    >>> aug = iaa.Superpixels(p_replace=0.5, n_segments=64)

    Generate around ``64`` superpixels per image and replace half of them
    with their average color, while the other half are left unchanged (i.e.
    they still show the input image's content).

    >>> aug = iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128))

    Generate between ``16`` and ``128`` superpixels per image and replace
    ``25`` to ``100`` percent of them with their average color.

    """

    def __init__(
        self,
        p_replace: ParamInput = (0.5, 1.0),
        n_segments: ParamInput = (50, 120),
        max_size: int | None = 128,
        interpolation: str | int = "linear",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.p_replace = iap.handle_probability_param(
            p_replace, "p_replace", tuple_to_uniform=True, list_to_choice=True
        )
        self.n_segments = iap.handle_discrete_param(
            n_segments,
            "n_segments",
            value_range=(1, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self.max_size = max_size
        self.interpolation = interpolation

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

        iadt.gate_dtypes_strs(
            images,
            allowed="bool uint8 uint16 uint32 uint64 int8 int16 int32 int64",
            disallowed="float16 float32 float64 float128",
            augmenter=self,
        )

        nb_images = len(images)
        rss = random_state.duplicate(1 + nb_images)
        n_segments_samples = self.n_segments.draw_samples((nb_images,), random_state=rss[0])

        # We cant reduce images to 0 or less segments, hence we pick the
        # lowest possible value in these cases (i.e. 1). The alternative
        # would be to not perform superpixel detection in these cases
        # (akin to n_segments=#pixels).
        n_segments_samples = np.clip(n_segments_samples, 1, None)

        for i, (image, rs) in enumerate(zip(images, rss[1:], strict=True)):
            if image.size == 0:
                # Image with 0-sized axis, nothing to change.
                # Placing this before the sampling step should be fine.
                continue

            replace_samples = self.p_replace.draw_samples((n_segments_samples[i],), random_state=rs)

            if np.max(replace_samples) == 0:
                # not a single superpixel would be replaced by its average
                # color, i.e. the image would not be changed, so just keep it
                continue

            orig_shape = image.shape
            image = _ensure_image_max_size(image, self.max_size, self.interpolation)

            # skimage 0.17+ introduces the start_label arg and produces a
            # warning if it is not provided. We use start_label=0 here
            # (old skimage style) (not entirely sure if =0 is required or =1
            # could be used here too, but *seems* like both could work),
            # but skimage will change the default start_label to 1 in the
            # future.
            kwargs = {"start_label": 0} if _SLIC_SUPPORTS_START_LABEL else {}

            segments = skimage.segmentation.slic(
                image, n_segments=n_segments_samples[i], compactness=10, **kwargs
            )

            image_aug = replace_segments_(image, segments, replace_samples > 0.5)

            if orig_shape != image_aug.shape:
                image_aug = ia.imresize_single_image(
                    image_aug, orig_shape[0:2], interpolation=self.interpolation
                )

            batch.images[i] = image_aug
        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p_replace, self.n_segments, self.max_size, self.interpolation]
