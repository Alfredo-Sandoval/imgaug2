"""Coordinate-based augmentable (CBA) helper augmenters."""

from __future__ import annotations

from typing import Literal

import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters._typing import RNGInput
from imgaug2.compat.markers import legacy

from .base import Augmenter
from .utils import Number
class RemoveCBAsByOutOfImageFraction(Augmenter):
    """Remove coordinate-based augmentables exceeding an out of image fraction.

    This augmenter inspects all coordinate-based augmentables (e.g.
    bounding boxes, line strings) within a given batch and removes any such
    augmentable which's out of image fraction is exactly a given value or
    greater than that. The out of image fraction denotes the fraction of the
    augmentable's area that is outside of the image, e.g. for a bounding box
    that has half of its area outside of the image it would be ``0.5``.


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; fully tested
        * ``uint32``: yes; fully tested
        * ``uint64``: yes; fully tested
        * ``int8``: yes; fully tested
        * ``int16``: yes; fully tested
        * ``int32``: yes; fully tested
        * ``int64``: yes; fully tested
        * ``float16``: yes; fully tested
        * ``float32``: yes; fully tested
        * ``float64``: yes; fully tested
        * ``float128``: yes; fully tested
        * ``bool``: yes; fully tested

    Parameters
    ----------
    fraction : number
        Remove any augmentable for which ``fraction_{actual} >= fraction``,
        where ``fraction_{actual}`` denotes the estimated out of image
        fraction.

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
    >>> aug = iaa.Sequential([
    >>>     iaa.Affine(translate_px={"x": (-100, 100)}),
    >>>     iaa.RemoveCBAsByOutOfImageFraction(0.5)
    >>> ])

    Translate all inputs by ``-100`` to ``100`` pixels on the x-axis, then
    remove any coordinate-based augmentable (e.g. bounding boxes) which has
    at least ``50%`` of its area outside of the image plane.

    >>> import imgaug2 as ia
    >>> import imgaug2.augmenters as iaa
    >>> image = ia.quokka_square((100, 100))
    >>> bb = ia.BoundingBox(x1=50-25, y1=0, x2=50+25, y2=100)
    >>> bbsoi = ia.BoundingBoxesOnImage([bb], shape=image.shape)
    >>> aug_without = iaa.Affine(translate_px={"x": 51})
    >>> aug_with = iaa.Sequential([
    >>>     iaa.Affine(translate_px={"x": 51}),
    >>>     iaa.RemoveCBAsByOutOfImageFraction(0.5)
    >>> ])
    >>>
    >>> image_without, bbsoi_without = aug_without(
    >>>     image=image, bounding_boxes=bbsoi)
    >>> image_with, bbsoi_with = aug_with(
    >>>     image=image, bounding_boxes=bbsoi)
    >>>
    >>> assert len(bbsoi_without.bounding_boxes) == 1
    >>> assert len(bbsoi_with.bounding_boxes) == 0

    Create a bounding box on an example image, then translate the image so that
    ``50%`` of the bounding box's area is outside of the image and compare
    the effects and using ``RemoveCBAsByOutOfImageFraction`` with not using it.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        fraction: Number,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.fraction = fraction

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        for column in batch.columns:
            if column.name in ["keypoints", "bounding_boxes", "polygons", "line_strings"]:
                for i, cbaoi in enumerate(column.value):
                    column.value[i] = cbaoi.remove_out_of_image_fraction_(self.fraction)

        return batch

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.fraction]


@legacy(version="0.4.0")
class ClipCBAsToImagePlanes(Augmenter):
    """Clip coordinate-based augmentables to areas within the image plane.

    This augmenter inspects all coordinate-based augmentables (e.g.
    bounding boxes, line strings) within a given batch and from each of them
    parts that are outside of the image plane. Parts within the image plane
    will be retained. This may e.g. shrink down bounding boxes. For keypoints,
    it removes any single points outside of the image plane. Any augmentable
    that is completely outside of the image plane will be removed.


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; fully tested
        * ``uint32``: yes; fully tested
        * ``uint64``: yes; fully tested
        * ``int8``: yes; fully tested
        * ``int16``: yes; fully tested
        * ``int32``: yes; fully tested
        * ``int64``: yes; fully tested
        * ``float16``: yes; fully tested
        * ``float32``: yes; fully tested
        * ``float64``: yes; fully tested
        * ``float128``: yes; fully tested
        * ``bool``: yes; fully tested

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
    >>> aug = iaa.Sequential([
    >>>     iaa.Affine(translate_px={"x": (-100, 100)}),
    >>>     iaa.ClipCBAsToImagePlanes()
    >>> ])

    Translate input data on the x-axis by ``-100`` to ``100`` pixels,
    then cut all coordinate-based augmentables (e.g. bounding boxes) down
    to areas that are within the image planes of their corresponding images.

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
        parents: list[Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        for column in batch.columns:
            if column.name in ["keypoints", "bounding_boxes", "polygons", "line_strings"]:
                for i, cbaoi in enumerate(column.value):
                    column.value[i] = cbaoi.clip_out_of_image_()

        return batch

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return []
