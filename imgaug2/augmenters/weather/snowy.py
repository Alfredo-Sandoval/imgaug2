"""Snowy landscape augmenters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import meta
from imgaug2.augmenters import color as colorlib
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy

class FastSnowyLandscape(meta.Augmenter):
    """Convert non-snowy landscapes to snowy ones.

    This augmenter expects to get an image that roughly shows a landscape.

    This augmenter is based on the method proposed in
    https://medium.freecodecamp.org/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f?gi=bca4a13e634c

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

        - (1) This augmenter is based on a colorspace conversion to HLS.
              Hence, only RGB ``uint8`` inputs are sensible.

    Parameters
    ----------
    lightness_threshold : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        All pixels with lightness in HLS colorspace that is below this value
        will have their lightness increased by `lightness_multiplier`.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the discrete interval ``[a..b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    lightness_multiplier : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Multiplier for pixel's lightness value in HLS colorspace.
        Affects all pixels selected via `lightness_threshold`.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the discrete interval ``[a..b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    from_colorspace : str, optional
        The source colorspace of the input images.
        See :func:`~imgaug2.augmenters.color.ChangeColorspace.__init__`.

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
    >>> aug = iaa.FastSnowyLandscape(
    >>>     lightness_threshold=140,
    >>>     lightness_multiplier=2.5
    >>> )

    Search for all pixels in the image with a lightness value in HLS
    colorspace of less than ``140`` and increase their lightness by a factor
    of ``2.5``.

    >>> aug = iaa.FastSnowyLandscape(
    >>>     lightness_threshold=[128, 200],
    >>>     lightness_multiplier=(1.5, 3.5)
    >>> )

    Search for all pixels in the image with a lightness value in HLS
    colorspace of less than ``128`` or less than ``200`` (one of these
    values is picked per image) and multiply their lightness by a factor
    of ``x`` with ``x`` being sampled from ``uniform(1.5, 3.5)`` (once per
    image).

    >>> aug = iaa.FastSnowyLandscape(
    >>>     lightness_threshold=(100, 255),
    >>>     lightness_multiplier=(1.0, 4.0)
    >>> )

    Similar to the previous example, but the lightness threshold is sampled
    from ``uniform(100, 255)`` (per image) and the multiplier
    from ``uniform(1.0, 4.0)`` (per image). This seems to produce good and
    varied results.

    """

    def __init__(
        self,
        lightness_threshold: ParamInput = (100, 255),
        lightness_multiplier: ParamInput = (1.0, 4.0),
        from_colorspace: str = colorlib.CSPACE_RGB,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.lightness_threshold = iap.handle_continuous_param(
            lightness_threshold,
            "lightness_threshold",
            value_range=(0, 255),
            tuple_to_uniform=True,
            list_to_choice=True,
        )
        self.lightness_multiplier = iap.handle_continuous_param(
            lightness_multiplier,
            "lightness_multiplier",
            value_range=(0, None),
            tuple_to_uniform=True,
            list_to_choice=True,
        )
        self.from_colorspace = from_colorspace

    def _draw_samples(
        self, augmentables: Sequence[Array] | Array, random_state: iarandom.RNG
    ) -> tuple[Array, Array]:
        nb_augmentables = len(augmentables)
        rss = random_state.duplicate(2)
        thresh_samples = self.lightness_threshold.draw_samples((nb_augmentables,), rss[1])
        lmul_samples = self.lightness_multiplier.draw_samples((nb_augmentables,), rss[0])
        return thresh_samples, lmul_samples

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

        thresh_samples, lmul_samples = self._draw_samples(images, random_state)

        gen = enumerate(zip(images, thresh_samples, lmul_samples, strict=True))
        for i, (image, thresh, lmul) in gen:
            image_hls = colorlib.change_colorspace_(
                image, colorlib.CSPACE_HLS, self.from_colorspace
            )
            cvt_dtype = image_hls.dtype
            image_hls = image_hls.astype(np.float64)
            lightness = image_hls[..., 1]

            lightness[lightness < thresh] *= lmul

            image_hls = iadt.restore_dtypes_(image_hls, cvt_dtype)
            image_rgb = colorlib.change_colorspace_(
                image_hls, self.from_colorspace, colorlib.CSPACE_HLS
            )

            batch.images[i] = image_rgb

        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.lightness_threshold, self.lightness_multiplier]


# TODO add examples and add these to the overview docs
# TODO add perspective transform to each cloud layer to make them look more
#      distant?
# TODO alpha_mean and density overlap - remove one of them
