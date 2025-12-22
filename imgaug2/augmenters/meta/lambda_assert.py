"""Lambda and assertion-based augmenters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters._typing import Images, RNGInput
from imgaug2.compat.markers import legacy

from .base import Augmenter
class Lambda(Augmenter):
    """Augmenter that calls a lambda function for each input batch.

    This is useful to add missing functions to a list of augmenters.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    func_images : None or callable, optional
        The function to call for each batch of images.
        It must follow the form::

            function(images, random_state, parents, hooks)

        and return the changed images (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_images`.
        If this is ``None`` instead of a function, the images will not be
        altered.

    func_heatmaps : None or callable, optional
        The function to call for each batch of heatmaps.
        It must follow the form::

            function(heatmaps, random_state, parents, hooks)

        and return the changed heatmaps (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_heatmaps`.
        If this is ``None`` instead of a function, the heatmaps will not be
        altered.

    func_segmentation_maps : None or callable, optional
        The function to call for each batch of segmentation maps.
        It must follow the form::

            function(segmaps, random_state, parents, hooks)

        and return the changed segmaps (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_segmentation_maps`.
        If this is ``None`` instead of a function, the segmentatio maps will
        not be altered.

    func_keypoints : None or callable, optional
        The function to call for each batch of keypoints.
        It must follow the form::

            function(keypoints_on_images, random_state, parents, hooks)

        and return the changed keypoints (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_keypoints`.
        If this is ``None`` instead of a function, the keypoints will not be
        altered.

    func_bounding_boxes : "keypoints" or None or callable, optional
        The function to call for each batch of bounding boxes.
        It must follow the form::

            function(bounding_boxes_on_images, random_state, parents, hooks)

        and return the changed bounding boxes (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_bounding_boxes`.
        If this is ``None`` instead of a function, the bounding boxes will not
        be altered.
        If this is the string ``"keypoints"`` instead of a function, the
        bounding boxes will automatically be augmented by transforming their
        corner vertices to keypoints and calling `func_keypoints`.


    func_polygons : "keypoints" or None or callable, optional
        The function to call for each batch of polygons.
        It must follow the form::

            function(polygons_on_images, random_state, parents, hooks)

        and return the changed polygons (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_polygons`.
        If this is ``None`` instead of a function, the polygons will not
        be altered.
        If this is the string ``"keypoints"`` instead of a function, the
        polygons will automatically be augmented by transforming their
        corner vertices to keypoints and calling `func_keypoints`.

    func_line_strings : "keypoints" or None or callable, optional
        The function to call for each batch of line strings.
        It must follow the form::

            function(line_strings_on_images, random_state, parents, hooks)

        and return the changed line strings (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_line_strings`.
        If this is ``None`` instead of a function, the line strings will not
        be altered.
        If this is the string ``"keypoints"`` instead of a function, the
        line strings will automatically be augmented by transforming their
        corner vertices to keypoints and calling `func_keypoints`.


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
    >>>
    >>> def func_images(images, random_state, parents, hooks):
    >>>     images[:, ::2, :, :] = 0
    >>>     return images
    >>>
    >>> aug = iaa.Lambda(
    >>>     func_images=func_images
    >>> )

    Replace every second row in input images with black pixels. Leave
    other data (e.g. heatmaps, keypoints) unchanged.

    >>> def func_images(images, random_state, parents, hooks):
    >>>     images[:, ::2, :, :] = 0
    >>>     return images
    >>>
    >>> def func_heatmaps(heatmaps, random_state, parents, hooks):
    >>>     for heatmaps_i in heatmaps:
    >>>         heatmaps.arr_0to1[::2, :, :] = 0
    >>>     return heatmaps
    >>>
    >>> def func_keypoints(keypoints_on_images, random_state, parents, hooks):
    >>>     return keypoints_on_images
    >>>
    >>> aug = iaa.Lambda(
    >>>     func_images=func_images,
    >>>     func_heatmaps=func_heatmaps,
    >>>     func_keypoints=func_keypoints
    >>> )

    Replace every second row in images with black pixels, set every second
    row in heatmaps to zero and leave other data (e.g. keypoints)
    unchanged.

    """

    def __init__(
        self,
        func_images: Callable[
            [Images, iarandom.RNG, list[Augmenter], ia.HooksImages | None], Images
        ]
        | None = None,
        func_heatmaps: Callable[
            [list[ia.HeatmapsOnImage], iarandom.RNG, list[Augmenter], ia.HooksHeatmaps | None],
            list[ia.HeatmapsOnImage],
        ]
        | None = None,
        func_segmentation_maps: Callable[
            [
                list[ia.SegmentationMapsOnImage],
                iarandom.RNG,
                list[Augmenter],
                ia.HooksHeatmaps | None,
            ],
            list[ia.SegmentationMapsOnImage],
        ]
        | None = None,
        func_keypoints: Callable[
            [
                list[ia.KeypointsOnImage],
                iarandom.RNG,
                list[Augmenter],
                ia.HooksKeypoints | None,
            ],
            list[ia.KeypointsOnImage],
        ]
        | None = None,
        func_bounding_boxes: Literal["keypoints"]
        | Callable[
            [
                list[ia.BoundingBoxesOnImage],
                iarandom.RNG,
                list[Augmenter],
                ia.HooksKeypoints | None,
            ],
            list[ia.BoundingBoxesOnImage],
        ]
        | None = "keypoints",
        func_polygons: Literal["keypoints"]
        | Callable[
            [list[ia.PolygonsOnImage], iarandom.RNG, list[Augmenter], ia.HooksKeypoints | None],
            list[ia.PolygonsOnImage],
        ]
        | None = "keypoints",
        func_line_strings: Literal["keypoints"]
        | Callable[
            [
                list[ia.LineStringsOnImage],
                iarandom.RNG,
                list[Augmenter],
                ia.HooksKeypoints | None,
            ],
            list[ia.LineStringsOnImage],
        ]
        | None = "keypoints",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.func_images = func_images
        self.func_heatmaps = func_heatmaps
        self.func_segmentation_maps = func_segmentation_maps
        self.func_keypoints = func_keypoints
        self.func_bounding_boxes = func_bounding_boxes
        self.func_polygons = func_polygons
        self.func_line_strings = func_line_strings

    def _augment_images(
        self,
        images: Images,
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksImages | None,
    ) -> Images:
        if self.func_images is not None:
            return self.func_images(images, random_state, parents, hooks)
        return images

    def _augment_heatmaps(
        self,
        heatmaps: list[ia.HeatmapsOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksHeatmaps | None,
    ) -> list[ia.HeatmapsOnImage]:
        if self.func_heatmaps is not None:
            result = self.func_heatmaps(heatmaps, random_state, parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for heatmaps to return list of "
                f"imgaug2.HeatmapsOnImage instances, got {type(result)}."
            )
            only_heatmaps = all([isinstance(el, ia.HeatmapsOnImage) for el in result])
            assert only_heatmaps, (
                "Expected callback function for heatmaps to return list of "
                f"imgaug2.HeatmapsOnImage instances, got {[type(el) for el in result]}."
            )
            return result
        return heatmaps

    def _augment_segmentation_maps(
        self,
        segmaps: list[ia.SegmentationMapsOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksSegmentationMaps | None,
    ) -> list[ia.SegmentationMapsOnImage]:
        if self.func_segmentation_maps is not None:
            result = self.func_segmentation_maps(segmaps, random_state, parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for segmentation maps to return "
                "list of imgaug2.SegmentationMapsOnImage() instances, "
                f"got {type(result)}."
            )
            only_segmaps = all([isinstance(el, ia.SegmentationMapsOnImage) for el in result])
            assert only_segmaps, (
                "Expected callback function for segmentation maps to return "
                "list of imgaug2.SegmentationMapsOnImage() instances, "
                f"got {[type(el) for el in result]}."
            )
            return result
        return segmaps

    def _augment_keypoints(
        self,
        keypoints_on_images: list[ia.KeypointsOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksKeypoints | None,
    ) -> list[ia.KeypointsOnImage]:
        if self.func_keypoints is not None:
            result = self.func_keypoints(keypoints_on_images, random_state, parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for keypoints to return list of "
                "imgaug2.augmentables.kps.KeypointsOnImage instances, "
                f"got {type(result)}."
            )
            only_keypoints = all([isinstance(el, ia.KeypointsOnImage) for el in result])
            assert only_keypoints, (
                "Expected callback function for keypoints to return list of "
                "imgaug2.augmentables.kps.KeypointsOnImage instances, "
                f"got {[type(el) for el in result]}."
            )
            return result
        return keypoints_on_images

    def _augment_polygons(
        self,
        polygons_on_images: list[ia.PolygonsOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksPolygons | None,
    ) -> list[ia.PolygonsOnImage]:
        from imgaug2.augmentables.polys import _ConcavePolygonRecoverer

        if self.func_polygons == "keypoints":
            return self._augment_polygons_as_keypoints(
                polygons_on_images,
                random_state,
                parents,
                hooks,
                recoverer=_ConcavePolygonRecoverer(),
            )
        if self.func_polygons is not None:
            result = self.func_polygons(polygons_on_images, random_state, parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for polygons to return list of "
                "imgaug2.augmentables.polys.PolygonsOnImage instances, "
                f"got {type(result)}."
            )
            only_polygons = all([isinstance(el, ia.PolygonsOnImage) for el in result])
            assert only_polygons, (
                "Expected callback function for polygons to return list of "
                "imgaug2.augmentables.polys.PolygonsOnImage instances, "
                f"got {[type(el) for el in result]}."
            )
            return result
        return polygons_on_images

    @legacy(version="0.4.0")
    def _augment_line_strings(
        self,
        line_strings_on_images: list[ia.LineStringsOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksLineStrings | None,
    ) -> list[ia.LineStringsOnImage]:
        if self.func_line_strings == "keypoints":
            return self._augment_line_strings_as_keypoints(
                line_strings_on_images, random_state, parents, hooks
            )
        if self.func_line_strings is not None:
            result = self.func_line_strings(line_strings_on_images, random_state, parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for line strings to return list of "
                "imgaug2.augmentables.lines.LineStringsOnImage instances, "
                f"got {type(result)}."
            )
            only_ls = all([isinstance(el, ia.LineStringsOnImage) for el in result])
            assert only_ls, (
                "Expected callback function for line strings to return list of "
                "imgaug2.augmentables.lines.LineStringsOnImages instances, "
                f"got {[type(el) for el in result]}."
            )
            return result
        return line_strings_on_images

    @legacy(version="0.4.0")
    def _augment_bounding_boxes(
        self,
        bounding_boxes_on_images: list[ia.BoundingBoxesOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksBoundingBoxes | None,
    ) -> list[ia.BoundingBoxesOnImage]:
        if self.func_bounding_boxes == "keypoints":
            return self._augment_bounding_boxes_as_keypoints(
                bounding_boxes_on_images, random_state, parents, hooks
            )
        if self.func_bounding_boxes is not None:
            result = self.func_bounding_boxes(
                bounding_boxes_on_images, random_state, parents, hooks
            )
            assert ia.is_iterable(result), (
                "Expected callback function for bounding boxes to return list "
                "of imgaug2.augmentables.bbs.BoundingBoxesOnImage instances, "
                f"got {type(result)}."
            )
            only_bbs = all([isinstance(el, ia.BoundingBoxesOnImage) for el in result])
            assert only_bbs, (
                "Expected callback function for polygons to return list of "
                "imgaug2.augmentables.polys.PolygonsOnImage instances, "
                f"got {[type(el) for el in result]}."
            )

            for bboi in bounding_boxes_on_images:
                for bb in bboi.bounding_boxes:
                    if bb.x1 > bb.x2:
                        bb.x1, bb.x2 = bb.x2, bb.x1
                    if bb.y1 > bb.y2:
                        bb.y1, bb.y2 = bb.y2, bb.y1

            return result
        return bounding_boxes_on_images

    def get_parameters(self) -> Sequence[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return []


@legacy(version="0.4.0")
class AssertLambda(Lambda):
    """Assert conditions based on lambda-function to be the case for input data.

    This augmenter applies a lambda function to each image or other input.
    The lambda function must return ``True`` or ``False``. If ``False`` is
    returned, an assertion error is produced.

    This is useful to ensure that generic assumption about the input data
    are actually the case and error out early otherwise.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    func_images : None or callable, optional
        The function to call for each batch of images.
        It must follow the form::

            function(images, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_images`.

    func_heatmaps : None or callable, optional
        The function to call for each batch of heatmaps.
        It must follow the form::

            function(heatmaps, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_heatmaps`.

    func_segmentation_maps : None or callable, optional
        The function to call for each batch of segmentation maps.
        It must follow the form::

            function(segmaps, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_segmentation_maps`.

    func_keypoints : None or callable, optional
        The function to call for each batch of keypoints.
        It must follow the form::

            function(keypoints_on_images, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_keypoints`.

    func_bounding_boxes : None or callable, optional
        The function to call for each batch of bounding boxes.
        It must follow the form::

            function(bounding_boxes_on_images, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_bounding_boxes`.


    func_polygons : None or callable, optional
        The function to call for each batch of polygons.
        It must follow the form::

            function(polygons_on_images, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_polygons`.

    func_line_strings : None or callable, optional
        The function to call for each batch of line strings.
        It must follow the form::

            function(line_strings_on_images, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug2.augmenters.meta.Augmenter._augment_line_strings`.


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

    """

    def __init__(
        self,
        func_images: Callable[[Images, iarandom.RNG, list[Augmenter], ia.HooksImages | None], bool]
        | None = None,
        func_heatmaps: Callable[
            [list[ia.HeatmapsOnImage], iarandom.RNG, list[Augmenter], ia.HooksHeatmaps | None], bool
        ]
        | None = None,
        func_segmentation_maps: Callable[
            [
                list[ia.SegmentationMapsOnImage],
                iarandom.RNG,
                list[Augmenter],
                ia.HooksHeatmaps | None,
            ],
            bool,
        ]
        | None = None,
        func_keypoints: Callable[
            [list[ia.KeypointsOnImage], iarandom.RNG, list[Augmenter], ia.HooksKeypoints | None],
            bool,
        ]
        | None = None,
        func_bounding_boxes: Callable[
            [
                list[ia.BoundingBoxesOnImage],
                iarandom.RNG,
                list[Augmenter],
                ia.HooksKeypoints | None,
            ],
            bool,
        ]
        | None = None,
        func_polygons: Callable[
            [list[ia.PolygonsOnImage], iarandom.RNG, list[Augmenter], ia.HooksKeypoints | None],
            bool,
        ]
        | None = None,
        func_line_strings: Callable[
            [
                list[ia.LineStringsOnImage],
                iarandom.RNG,
                list[Augmenter],
                ia.HooksKeypoints | None,
            ],
            bool,
        ]
        | None = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        def _default(
            var: Callable[[object, iarandom.RNG, list[Augmenter], object], bool] | None,
            augmentable_name: str,
        ) -> _AssertLambdaCallback | None:
            return (
                _AssertLambdaCallback(var, augmentable_name=augmentable_name)
                if var is not None
                else None
            )

        super().__init__(
            func_images=_default(func_images, "images"),
            func_heatmaps=_default(func_heatmaps, "heatmaps"),
            func_segmentation_maps=_default(func_segmentation_maps, "segmentation_maps"),
            func_keypoints=_default(func_keypoints, "keypoints"),
            func_bounding_boxes=_default(func_bounding_boxes, "bounding_boxes"),
            func_polygons=_default(func_polygons, "polygons"),
            func_line_strings=_default(func_line_strings, "line_strings"),
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
class _AssertLambdaCallback:
    @legacy(version="0.4.0")
    def __init__(
        self,
        func: Callable[[object, iarandom.RNG, list[Augmenter], object], bool],
        augmentable_name: str,
    ) -> None:
        self.func = func
        self.augmentable_name = augmentable_name

    @legacy(version="0.4.0")
    def __call__(
        self,
        augmentables: object,
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: object,
    ) -> object:
        assert self.func(augmentables, random_state, parents, hooks), (
            f"Input {self.augmentable_name} did not fulfill user-defined assertion in AssertLambda."
        )
        return augmentables


# Note: This evaluates .shape for kps/polys, but the array shape for
# heatmaps/segmaps.
@legacy(version="0.4.0")
class AssertShape(Lambda):
    """Assert that inputs have a specified shape.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    shape : tuple
        The expected shape, given as a ``tuple``. The number of entries in
        the ``tuple`` must match the number of dimensions, i.e. it must
        contain four entries for ``(N, H, W, C)``. If only a single entity
        is augmented, e.g. via
        :func:`~imgaug2.augmenters.meta.Augmenter.augment_image`, then ``N`` is
        ``1`` in the input to this augmenter. Images that don't have
        a channel axis will automatically have one assigned, i.e. ``C`` is
        at least ``1``.
        For each component of the ``tuple`` one of the following datatypes
        may be used:

            * If a component is ``None``, any value for that dimensions is
              accepted.
            * If a component is ``int``, exactly that value (and no other one)
              will be accepted for that dimension.
            * If a component is a ``tuple`` of two ``int`` s with values ``a``
              and ``b``, only a value within the interval ``[a, b)`` will be
              accepted for that dimension.
            * If an entry is a ``list`` of ``int`` s, only a value from that
              ``list`` will be accepted for that dimension.

    check_images : bool, optional
        Whether to validate input images via the given shape.

    check_heatmaps : bool, optional
        Whether to validate input heatmaps via the given shape.
        The number of heatmaps will be verified as ``N``. For each
        :class:`~imgaug2.augmentables.heatmaps.HeatmapsOnImage` instance
        its array's height and width will be verified as ``H`` and ``W``,
        but not the channel count.

    check_segmentation_maps : bool, optional
        Whether to validate input segmentation maps via the given shape.
        The number of segmentation maps will be verified as ``N``. For each
        :class:`~imgaug2.augmentables.segmaps.SegmentationMapOnImage` instance
        its array's height and width will be verified as ``H`` and ``W``,
        but not the channel count.

    check_keypoints : bool, optional
        Whether to validate input keypoints via the given shape.
        This will check (a) the number of keypoints and (b) for each
        :class:`~imgaug2.augmentables.kps.KeypointsOnImage` instance the
        ``.shape`` attribute, i.e. the shape of the corresponding image.

    check_bounding_boxes : bool, optional
        Whether to validate input bounding boxes via the given shape.
        This will check (a) the number of bounding boxes and (b) for each
        :class:`~imgaug2.augmentables.bbs.BoundingBoxesOnImage` instance the
        ``.shape`` attribute, i.e. the shape of the corresponding image.


    check_polygons : bool, optional
        Whether to validate input polygons via the given shape.
        This will check (a) the number of polygons and (b) for each
        :class:`~imgaug2.augmentables.polys.PolygonsOnImage` instance the
        ``.shape`` attribute, i.e. the shape of the corresponding image.

    check_line_strings : bool, optional
        Whether to validate input line strings via the given shape.
        This will check (a) the number of line strings and (b) for each
        :class:`~imgaug2.augmentables.lines.LineStringsOnImage` instance the
        ``.shape`` attribute, i.e. the shape of the corresponding image.


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
    >>> seq = iaa.Sequential([
    >>>     iaa.AssertShape((None, 32, 32, 3)),
    >>>     iaa.Fliplr(0.5)
    >>> ])

    Verify first for each image batch if it contains a variable number of
    ``32x32`` images with ``3`` channels each. Only if that check succeeds, the
    horizontal flip will be executed. Otherwise an assertion error will be
    raised.

    >>> seq = iaa.Sequential([
    >>>     iaa.AssertShape((None, (32, 64), 32, [1, 3])),
    >>>     iaa.Fliplr(0.5)
    >>> ])

    Similar to the above example, but now the height may be in the interval
    ``[32, 64)`` and the number of channels may be either ``1`` or ``3``.

    """

    def __init__(
        self,
        shape: tuple[object, ...],
        check_images: bool = True,
        check_heatmaps: bool = True,
        check_segmentation_maps: bool = True,
        check_keypoints: bool = True,
        check_bounding_boxes: bool = True,
        check_polygons: bool = True,
        check_line_strings: bool = True,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        assert len(shape) == 4, (
            f"Expected shape to have length 4, got {len(shape)} with shape: {shape!s}."
        )
        self.shape = shape

        def _default(func: object, do_use: bool) -> object | None:
            return func if do_use else None

        super().__init__(
            func_images=_default(_AssertShapeImagesCheck(shape), check_images),
            func_heatmaps=_default(_AssertShapeHeatmapsCheck(shape), check_heatmaps),
            func_segmentation_maps=_default(
                _AssertShapeSegmapCheck(shape), check_segmentation_maps
            ),
            func_keypoints=_default(_AssertShapeKeypointsCheck(shape), check_keypoints),
            func_bounding_boxes=_default(
                _AssertShapeBoundingBoxesCheck(shape), check_bounding_boxes
            ),
            func_polygons=_default(_AssertShapePolygonsCheck(shape), check_polygons),
            func_line_strings=_default(_AssertShapeLineStringsCheck(shape), check_line_strings),
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

    @classmethod
    def _compare(
        cls, observed: int, expected: object, dimension: int, image_index: int | str
    ) -> None:
        if expected is not None:
            if ia.is_single_integer(expected):
                assert observed == expected, (
                    f"Expected dim {dimension} (entry index: {image_index}) to have value {expected}, "
                    f"got {observed}."
                )
            elif isinstance(expected, tuple):
                assert len(expected) == 2, (
                    f"Expected tuple argument 'expected' to contain "
                    f"exactly 2 entries, got {len(expected)}."
                )
                assert expected[0] <= observed < expected[1], (
                    f"Expected dim {dimension} (entry index: {image_index}) to have value in "
                    f"interval [{expected[0]}, {expected[1]}), got {observed}."
                )
            elif isinstance(expected, list):
                assert any([observed == val for val in expected]), (
                    f"Expected dim {dimension} (entry index: {image_index}) to have any value "
                    f"of {expected!s}, got {observed}."
                )
            else:
                raise Exception(
                    f"Invalid datatype for shape entry {dimension}, expected each "
                    f"entry to be an integer, a tuple (with two entries) "
                    f"or a list, got {type(expected)}."
                )

    @classmethod
    def _check_shapes(cls, shapes: Sequence[Sequence[int]], shape_target: Sequence[object]) -> None:
        if shape_target[0] is not None:
            cls._compare(len(shapes), shape_target[0], 0, "ALL")

        for augm_idx, shape in enumerate(shapes):
            # note that dim_idx is here per object, dim 0 of shape target
            # denotes "number of all objects" and was checked above
            for dim_idx, expected in enumerate(shape_target[1:]):
                observed = shape[dim_idx]
                cls._compare(observed, expected, dim_idx, augm_idx)


# Keep these checks as top-level callables to avoid pickling issues when they
# are sent to multiprocessing workers.
@legacy(version="0.4.0")
class _AssertShapeImagesCheck:
    def __init__(self, shape: tuple[object, ...]) -> None:
        self.shape = shape

    def __call__(
        self,
        images: Images,
        _random_state: iarandom.RNG,
        _parents: list[Augmenter],
        _hooks: object,
    ) -> Images:
        # set shape_target so that we check all target dimensions,
        # including C, which isn't checked for the other methods
        AssertShape._check_shapes([obj.shape for obj in images], self.shape)
        return images


@legacy(version="0.4.0")
class _AssertShapeHeatmapsCheck:
    def __init__(self, shape: tuple[object, ...]) -> None:
        self.shape = shape

    def __call__(
        self,
        heatmaps: list[ia.HeatmapsOnImage],
        _random_state: iarandom.RNG,
        _parents: list[Augmenter],
        _hooks: object,
    ) -> list[ia.HeatmapsOnImage]:
        AssertShape._check_shapes([obj.arr_0to1.shape for obj in heatmaps], self.shape[0:3])
        return heatmaps


@legacy(version="0.4.0")
class _AssertShapeSegmapCheck:
    def __init__(self, shape: tuple[object, ...]) -> None:
        self.shape = shape

    def __call__(
        self,
        segmaps: list[ia.SegmentationMapsOnImage],
        _random_state: iarandom.RNG,
        _parents: list[Augmenter],
        _hooks: object,
    ) -> list[ia.SegmentationMapsOnImage]:
        AssertShape._check_shapes([obj.arr.shape for obj in segmaps], self.shape[0:3])
        return segmaps


@legacy(version="0.4.0")
class _AssertShapeKeypointsCheck:
    def __init__(self, shape: tuple[object, ...]) -> None:
        self.shape = shape

    def __call__(
        self,
        keypoints_on_images: list[ia.KeypointsOnImage],
        _random_state: iarandom.RNG,
        _parents: list[Augmenter],
        _hooks: object,
    ) -> list[ia.KeypointsOnImage]:
        AssertShape._check_shapes([obj.shape for obj in keypoints_on_images], self.shape[0:3])
        return keypoints_on_images


@legacy(version="0.4.0")
class _AssertShapeBoundingBoxesCheck:
    def __init__(self, shape: tuple[object, ...]) -> None:
        self.shape = shape

    def __call__(
        self,
        bounding_boxes_on_images: list[ia.BoundingBoxesOnImage],
        _random_state: iarandom.RNG,
        _parents: list[Augmenter],
        _hooks: object,
    ) -> list[ia.BoundingBoxesOnImage]:
        AssertShape._check_shapes([obj.shape for obj in bounding_boxes_on_images], self.shape[0:3])
        return bounding_boxes_on_images


@legacy(version="0.4.0")
class _AssertShapePolygonsCheck:
    def __init__(self, shape: tuple[object, ...]) -> None:
        self.shape = shape

    def __call__(
        self,
        polygons_on_images: list[ia.PolygonsOnImage],
        _random_state: iarandom.RNG,
        _parents: list[Augmenter],
        _hooks: object,
    ) -> list[ia.PolygonsOnImage]:
        AssertShape._check_shapes([obj.shape for obj in polygons_on_images], self.shape[0:3])
        return polygons_on_images


@legacy(version="0.4.0")
class _AssertShapeLineStringsCheck:
    def __init__(self, shape: tuple[object, ...]) -> None:
        self.shape = shape

    def __call__(
        self,
        line_strings_on_images: list[ia.LineStringsOnImage],
        _random_state: iarandom.RNG,
        _parents: list[Augmenter],
        _hooks: object,
    ) -> list[ia.LineStringsOnImage]:
        AssertShape._check_shapes([obj.shape for obj in line_strings_on_images], self.shape[0:3])
        return line_strings_on_images


