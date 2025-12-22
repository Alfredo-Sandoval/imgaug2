from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import UnnormalizedBatch
from imgaug2.augmenters import base as iabase
from imgaug2.augmenters._typing import Array, Images
from imgaug2.compat.markers import legacy

if TYPE_CHECKING:
    from .base import Augmenter


class AugmenterAugmentablesMixin:
    def augment_image(self, image: Array, hooks: ia.HooksImages | None = None) -> Array:
        """Augment a single image.

        Parameters
        ----------
        image : (H,W,C) ndarray or (H,W) ndarray
            The image to augment.
            Channel-axis is optional, but expected to be the last axis if
            present. In most cases, this array should be of dtype ``uint8``,
            which is supported by all augmenters. Support for other dtypes
            varies by augmenter -- see the respective augmenter-specific
            documentation for more details.

        hooks : None or imgaug2.HooksImages, optional
            HooksImages object to dynamically interfere with the augmentation
            process.

        Returns
        -------
        ndarray
            The corresponding augmented image.

        """
        # B1 backend policy: allow MLX arrays, but never implicitly convert
        # numpy -> MLX (or MLX -> numpy) in the augmenter pipeline.
        from imgaug2.mlx._core import is_mlx_array

        assert ia.is_np_array(image) or is_mlx_array(image), (
            "Expected to get a single array of shape (H,W) or (H,W,C) "
            f"for `image`. Got instead type {type(image).__name__}. Use `augment_images(images)` "
            "to augment a list/batch of multiple images."
        )
        assert image.ndim in [2, 3], (
            f"Expected image to have shape (height, width, [channels]), got shape {image.shape}."
        )
        iabase._warn_on_suspicious_single_image_shape(image)
        return self.augment_images([image], hooks=hooks)[0]

    def augment_images(
        self,
        images: Images,
        parents: list[Augmenter] | None = None,
        hooks: ia.HooksImages | None = None,
    ) -> Images:
        """Augment a batch of images.

        Parameters
        ----------
        images : (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            Images to augment.
            The input can be a list of numpy arrays or a single array. Each
            array is expected to have shape ``(H, W, C)`` or ``(H, W)``,
            where ``H`` is the height, ``W`` is the width and ``C`` are the
            channels. The number of channels may differ between images.
            If a list is provided, the height, width and channels may differ
            between images within the provided batch.
            In most cases, the image array(s) should be of dtype ``uint8``,
            which is supported by all augmenters. Support for other dtypes
            varies by augmenter -- see the respective augmenter-specific
            documentation for more details.

        parents : None or list of imgaug2.augmenters.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug2.imgaug2.HooksImages, optional
            :class:`~imgaug2.imgaug2.HooksImages` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        ndarray or list
            Corresponding augmented images.
            If the input was an ``ndarray``, the output is also an ``ndarray``,
            unless the used augmentations have led to different output image
            sizes (as can happen in e.g. cropping).

        Examples
        --------
        >>> import imgaug2.augmenters as iaa
        >>> import numpy as np
        >>> aug = iaa.GaussianBlur((0.0, 3.0))
        >>> # create empty example images
        >>> images = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        >>> images_aug = aug.augment_images(images)

        Create ``2`` empty (i.e. black) example numpy images and apply
        gaussian blurring to them.

        """
        iabase._warn_on_suspicious_multi_image_shapes(images)
        batch_aug = self.augment_batch_(
            UnnormalizedBatch(images=images), parents=parents, hooks=hooks
        )
        images_aug = batch_aug.images_aug
        assert images_aug is not None, "Expected `images_aug` to be set after image augmentation."
        if isinstance(images_aug, np.ndarray):
            return cast(Array, images_aug)
        if isinstance(images_aug, Sequence):
            return cast(Sequence[Array], images_aug)
        raise AssertionError(
            "Expected `images_aug` to be a numpy array or a sequence of numpy arrays, "
            f"got type {type(images_aug)}."
        )

    def _augment_images(
        self,
        images: Images,
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksImages | None,
    ) -> Images:
        """Augment a batch of images in-place.

        This is the internal version of :func:`Augmenter.augment_images`.
        It is called from :func:`Augmenter.augment_images` and should usually
        not be called directly.
        It has to be implemented by every augmenter.
        This method may transform the images in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug2.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.

        Parameters
        ----------
        images : (N,H,W,C) ndarray or list of (H,W,C) ndarray
            Images to augment.
            They may be changed in-place.
            Either a list of ``(H, W, C)`` arrays or a single ``(N, H, W, C)``
            array, where ``N`` is the number of images, ``H`` is the height of
            images, ``W`` is the width of images and ``C`` is the number of
            channels of images. In the case of a list as input, ``H``, ``W``
            and ``C`` may change per image.

        random_state : imgaug2.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug2.augmenters.meta.Augmenter
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_images`.

        hooks : imgaug2.imgaug2.HooksImages or None
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_images`.

        Returns
        ----------
        (N,H,W,C) ndarray or list of (H,W,C) ndarray
            The augmented images.

        """
        return images

    def augment_heatmaps(
        self,
        heatmaps: ia.HeatmapsOnImage | list[ia.HeatmapsOnImage],
        parents: list[Augmenter] | None = None,
        hooks: ia.HooksHeatmaps | None = None,
    ) -> ia.HeatmapsOnImage | list[ia.HeatmapsOnImage]:
        """Augment a batch of heatmaps.

        Parameters
        ----------
        heatmaps : imgaug2.augmentables.heatmaps.HeatmapsOnImage or list of imgaug2.augmentables.heatmaps.HeatmapsOnImage
            Heatmap(s) to augment. Either a single heatmap or a list of
            heatmaps.

        parents : None or list of imgaug2.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``.
            It is set automatically for child augmenters.

        hooks : None or imaug.imgaug2.HooksHeatmaps, optional
            :class:`~imgaug2.imgaug2.HooksHeatmaps` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug2.augmentables.heatmaps.HeatmapsOnImage or list of imgaug2.augmentables.heatmaps.HeatmapsOnImage
            Corresponding augmented heatmap(s).

        """
        return self.augment_batch_(
            UnnormalizedBatch(heatmaps=heatmaps), parents=parents, hooks=hooks
        ).heatmaps_aug

    def _augment_heatmaps(
        self,
        heatmaps: ia.HeatmapsOnImage | list[ia.HeatmapsOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksHeatmaps | None,
    ) -> ia.HeatmapsOnImage | list[ia.HeatmapsOnImage]:
        """Augment a batch of heatmaps in-place.

        This is the internal version of :func:`Augmenter.augment_heatmaps`.
        It is called from :func:`Augmenter.augment_heatmaps` and should
        usually not be called directly.
        This method may augment heatmaps in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug2.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.

        Parameters
        ----------
        heatmaps : list of imgaug2.augmentables.heatmaps.HeatmapsOnImage
            Heatmaps to augment. They may be changed in-place.

        parents : list of imgaug2.augmenters.meta.Augmenter
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_heatmaps`.

        hooks : imgaug2.imgaug2.HooksHeatmaps or None
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_heatmaps`.

        Returns
        ----------
        images : list of imgaug2.augmentables.heatmaps.HeatmapsOnImage
            The augmented heatmaps.

        """
        return heatmaps

    def augment_segmentation_maps(
        self,
        segmaps: ia.SegmentationMapsOnImage | list[ia.SegmentationMapsOnImage],
        parents: list[Augmenter] | None = None,
        hooks: ia.HooksSegmentationMaps | None = None,
    ) -> ia.SegmentationMapsOnImage | list[ia.SegmentationMapsOnImage]:
        """Augment a batch of segmentation maps.

        Parameters
        ----------
        segmaps : imgaug2.augmentables.segmaps.SegmentationMapsOnImage or list of imgaug2.augmentables.segmaps.SegmentationMapsOnImage
            Segmentation map(s) to augment. Either a single segmentation map
            or a list of segmentation maps.

        parents : None or list of imgaug2.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug2.HooksHeatmaps, optional
            :class:`~imgaug2.imgaug2.HooksHeatmaps` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug2.augmentables.segmaps.SegmentationMapsOnImage or list of imgaug2.augmentables.segmaps.SegmentationMapsOnImage
            Corresponding augmented segmentation map(s).

        """
        return self.augment_batch_(
            UnnormalizedBatch(segmentation_maps=segmaps), parents=parents, hooks=hooks
        ).segmentation_maps_aug

    def _augment_segmentation_maps(
        self,
        segmaps: ia.SegmentationMapsOnImage | list[ia.SegmentationMapsOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksSegmentationMaps | None,
    ) -> ia.SegmentationMapsOnImage | list[ia.SegmentationMapsOnImage]:
        """Augment a batch of segmentation in-place.

        This is the internal version of
        :func:`Augmenter.augment_segmentation_maps`.
        It is called from :func:`Augmenter.augment_segmentation_maps` and
        should usually not be called directly.
        This method may augment segmentation maps in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug2.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.

        Parameters
        ----------
        segmaps : list of imgaug2.augmentables.segmaps.SegmentationMapsOnImage
            Segmentation maps to augment. They may be changed in-place.

        parents : list of imgaug2.augmenters.meta.Augmenter
            See
            :func:`~imgaug2.augmenters.meta.Augmenter.augment_segmentation_maps`.

        hooks : imgaug2.imgaug2.HooksHeatmaps or None
            See
            :func:`~imgaug2.augmenters.meta.Augmenter.augment_segmentation_maps`.

        Returns
        ----------
        images : list of imgaug2.augmentables.segmaps.SegmentationMapsOnImage
            The augmented segmentation maps.

        """
        return segmaps

    def augment_keypoints(
        self,
        keypoints_on_images: ia.KeypointsOnImage | list[ia.KeypointsOnImage],
        parents: list[Augmenter] | None = None,
        hooks: ia.HooksKeypoints | None = None,
    ) -> ia.KeypointsOnImage | list[ia.KeypointsOnImage]:
        """Augment a batch of keypoints/landmarks.

        This is the corresponding function to :func:`Augmenter.augment_images`,
        just for keypoints/landmarks (i.e. points on images).
        Usually you will want to call :func:`Augmenter.augment_images` with
        a list of images, e.g. ``augment_images([A, B, C])`` and then
        ``augment_keypoints()`` with the corresponding list of keypoints on
        these images, e.g. ``augment_keypoints([Ak, Bk, Ck])``, where ``Ak``
        are the keypoints on image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding keypoints,
        e.g. by

        >>> import imgaug2.augmenters as iaa
        >>> from imgaug2.augmentables.kps import Keypoint
        >>> from imgaug2.augmentables.kps import KeypointsOnImage
        >>> A = B = C = np.zeros((10, 10), dtype=np.uint8)
        >>> Ak = Bk = Ck = KeypointsOnImage([Keypoint(2, 2)], (10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> kps_aug = seq_det.augment_keypoints([Ak, Bk, Ck])

        Otherwise, different random values will be sampled for the image
        and keypoint augmentations, resulting in different augmentations (e.g.
        images might be rotated by ``30deg`` and keypoints by ``-10deg``).
        Also make sure to call :func:`Augmenter.to_deterministic` again for
        each new batch, otherwise you would augment all batches in the same
        way.

        Note that there is also :func:`Augmenter.augment`, which automatically
        handles the random state alignment.

        Parameters
        ----------
        keypoints_on_images : imgaug2.augmentables.kps.KeypointsOnImage or list of imgaug2.augmentables.kps.KeypointsOnImage
            The keypoints/landmarks to augment.
            Either a single instance of
            :class:`~imgaug2.augmentables.kps.KeypointsOnImage` or a list of
            such instances. Each instance must contain the keypoints of a
            single image.

        parents : None or list of imgaug2.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug2.imgaug2.HooksKeypoints, optional
            :class:`~imgaug2.imgaug2.HooksKeypoints` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug2.augmentables.kps.KeypointsOnImage or list of imgaug2.augmentables.kps.KeypointsOnImage
            Augmented keypoints.

        """
        return self.augment_batch_(
            UnnormalizedBatch(keypoints=keypoints_on_images), parents=parents, hooks=hooks
        ).keypoints_aug

    def _augment_keypoints(
        self,
        keypoints_on_images: ia.KeypointsOnImage | list[ia.KeypointsOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksKeypoints | None,
    ) -> ia.KeypointsOnImage | list[ia.KeypointsOnImage]:
        """Augment a batch of keypoints in-place.

        This is the internal version of :func:`Augmenter.augment_keypoints`.
        It is called from :func:`Augmenter.augment_keypoints` and should
        usually not be called directly.
        This method may transform the keypoints in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug2.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.

        Parameters
        ----------
        keypoints_on_images : list of imgaug2.augmentables.kps.KeypointsOnImage
            Keypoints to augment. They may be changed in-place.

        random_state : imgaug2.random.RNG
            The random state to use for all sampling tasks during the augmentation.

        parents : list of imgaug2.augmenters.meta.Augmenter
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_keypoints`.

        hooks : imgaug2.imgaug2.HooksKeypoints or None
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_keypoints`.

        Returns
        ----------
        list of imgaug2.augmentables.kps.KeypointsOnImage
            The augmented keypoints.

        """
        return keypoints_on_images

    def augment_bounding_boxes(
        self,
        bounding_boxes_on_images: ia.BoundingBoxesOnImage | list[ia.BoundingBoxesOnImage],
        parents: list[Augmenter] | None = None,
        hooks: ia.HooksBoundingBoxes | None = None,
    ) -> ia.BoundingBoxesOnImage | list[ia.BoundingBoxesOnImage]:
        """Augment a batch of bounding boxes.

        This is the corresponding function to
        :func:`Augmenter.augment_images`, just for bounding boxes.
        Usually you will want to call :func:`Augmenter.augment_images` with
        a list of images, e.g. ``augment_images([A, B, C])`` and then
        ``augment_bounding_boxes()`` with the corresponding list of bounding
        boxes on these images, e.g.
        ``augment_bounding_boxes([Abb, Bbb, Cbb])``, where ``Abb`` are the
        bounding boxes on image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding bounding boxes,
        e.g. by

        >>> import imgaug2.augmenters as iaa
        >>> from imgaug2.augmentables.bbs import BoundingBox
        >>> from imgaug2.augmentables.bbs import BoundingBoxesOnImage
        >>> A = B = C = np.ones((10, 10), dtype=np.uint8)
        >>> Abb = Bbb = Cbb = BoundingBoxesOnImage([
        >>>     BoundingBox(1, 1, 9, 9)], (10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> bbs_aug = seq_det.augment_bounding_boxes([Abb, Bbb, Cbb])

        Otherwise, different random values will be sampled for the image
        and bounding box augmentations, resulting in different augmentations
        (e.g. images might be rotated by ``30deg`` and bounding boxes by
        ``-10deg``). Also make sure to call :func:`Augmenter.to_deterministic`
        again for each new batch, otherwise you would augment all batches in
        the same way.

        Note that there is also :func:`Augmenter.augment`, which automatically
        handles the random state alignment.

        Parameters
        ----------
        bounding_boxes_on_images : imgaug2.augmentables.bbs.BoundingBoxesOnImage or list of imgaug2.augmentables.bbs.BoundingBoxesOnImage
            The bounding boxes to augment.
            Either a single instance of
            :class:`~imgaug2.augmentables.bbs.BoundingBoxesOnImage` or a list of
            such instances, with each one of them containing the bounding
            boxes of a single image.

        parents : None or list of imgaug2.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug2.imgaug2.HooksKeypoints, optional
            :class:`~imgaug2.imgaug2.HooksKeypoints` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug2.augmentables.bbs.BoundingBoxesOnImage or list of imgaug2.augmentables.bbs.BoundingBoxesOnImage
            Augmented bounding boxes.

        """
        return self.augment_batch_(
            UnnormalizedBatch(bounding_boxes=bounding_boxes_on_images), parents=parents, hooks=hooks
        ).bounding_boxes_aug

    def augment_polygons(
        self,
        polygons_on_images: ia.PolygonsOnImage | list[ia.PolygonsOnImage],
        parents: list[Augmenter] | None = None,
        hooks: ia.HooksPolygons | None = None,
    ) -> ia.PolygonsOnImage | list[ia.PolygonsOnImage]:
        """Augment a batch of polygons.

        This is the corresponding function to :func:`Augmenter.augment_images`,
        just for polygons.
        Usually you will want to call :func:`Augmenter.augment_images`` with
        a list of images, e.g. ``augment_images([A, B, C])`` and then
        ``augment_polygons()`` with the corresponding list of polygons on these
        images, e.g. ``augment_polygons([A_poly, B_poly, C_poly])``, where
        ``A_poly`` are the polygons on image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding polygons,
        e.g. by

        >>> import imgaug2.augmenters as iaa
        >>> from imgaug2.augmentables.polys import Polygon, PolygonsOnImage
        >>> A = B = C = np.ones((10, 10), dtype=np.uint8)
        >>> Apoly = Bpoly = Cpoly = PolygonsOnImage(
        >>>     [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        >>>     shape=(10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> polys_aug = seq_det.augment_polygons([Apoly, Bpoly, Cpoly])

        Otherwise, different random values will be sampled for the image
        and polygon augmentations, resulting in different augmentations
        (e.g. images might be rotated by ``30deg`` and polygons by
        ``-10deg``). Also make sure to call ``to_deterministic()`` again for
        each new batch, otherwise you would augment all batches in the same
        way.

        Note that there is also :func:`Augmenter.augment`, which automatically
        handles the random state alignment.

        Parameters
        ----------
        polygons_on_images : imgaug2.augmentables.polys.PolygonsOnImage or list of imgaug2.augmentables.polys.PolygonsOnImage
            The polygons to augment.
            Either a single instance of
            :class:`~imgaug2.augmentables.polys.PolygonsOnImage` or a list of
            such instances, with each one of them containing the polygons of
            a single image.

        parents : None or list of imgaug2.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug2.imgaug2.HooksKeypoints, optional
            :class:`~imgaug2.imgaug2.HooksKeypoints` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug2.augmentables.polys.PolygonsOnImage or list of imgaug2.augmentables.polys.PolygonsOnImage
            Augmented polygons.

        """
        return self.augment_batch_(
            UnnormalizedBatch(polygons=polygons_on_images), parents=parents, hooks=hooks
        ).polygons_aug

    def augment_line_strings(
        self,
        line_strings_on_images: ia.LineStringsOnImage | list[ia.LineStringsOnImage],
        parents: list[Augmenter] | None = None,
        hooks: ia.HooksLineStrings | None = None,
    ) -> ia.LineStringsOnImage | list[ia.LineStringsOnImage]:
        """Augment a batch of line strings.

        This is the corresponding function to
        :func:`Augmenter.augment_images``, just for line strings.
        Usually you will want to call :func:`Augmenter.augment_images` with
        a list of images, e.g. ``augment_images([A, B, C])`` and then
        ``augment_line_strings()`` with the corresponding list of line
        strings on these images, e.g.
        ``augment_line_strings([A_line, B_line, C_line])``, where ``A_line``
        are the line strings on image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding line strings,
        e.g. by

        >>> import imgaug2.augmenters as iaa
        >>> from imgaug2.augmentables.lines import LineString
        >>> from imgaug2.augmentables.lines import LineStringsOnImage
        >>> A = B = C = np.ones((10, 10), dtype=np.uint8)
        >>> A_line = B_line = C_line = LineStringsOnImage(
        >>>     [LineString([(0, 0), (1, 0), (1, 1), (0, 1)])],
        >>>     shape=(10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> lines_aug = seq_det.augment_line_strings([A_line, B_line, C_line])

        Otherwise, different random values will be sampled for the image
        and line string augmentations, resulting in different augmentations
        (e.g. images might be rotated by ``30deg`` and line strings by
        ``-10deg``). Also make sure to call ``to_deterministic()`` again for
        each new batch, otherwise you would augment all batches in the same
        way.

        Note that there is also :func:`Augmenter.augment`, which automatically
        handles the random state alignment.

        Parameters
        ----------
        line_strings_on_images : imgaug2.augmentables.lines.LineStringsOnImage or list of imgaug2.augmentables.lines.LineStringsOnImage
            The line strings to augment.
            Either a single instance of
            :class:`~imgaug2.augmentables.lines.LineStringsOnImage` or a list of
            such instances, with each one of them containing the line strings
            of a single image.

        parents : None or list of imgaug2.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as None.
            It is set automatically for child augmenters.

        hooks : None or imgaug2.imgaug2.HooksKeypoints, optional
            :class:`~imgaug2.imgaug2.HooksKeypoints` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug2.augmentables.lines.LineStringsOnImage or list of imgaug2.augmentables.lines.LineStringsOnImage
            Augmented line strings.

        """
        return self.augment_batch_(
            UnnormalizedBatch(line_strings=line_strings_on_images), parents=parents, hooks=hooks
        ).line_strings_aug

    @legacy(version="0.4.0")
    def _augment_bounding_boxes(
        self,
        bounding_boxes_on_images: ia.BoundingBoxesOnImage | list[ia.BoundingBoxesOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksBoundingBoxes | None,
    ) -> ia.BoundingBoxesOnImage | list[ia.BoundingBoxesOnImage]:
        """Augment a batch of bounding boxes on images in-place.

        This is the internal version of
        :func:`Augmenter.augment_bounding_boxes`.
        It is called from :func:`Augmenter.augment_bounding_boxes` and should
        usually not be called directly.
        This method may transform the bounding boxes in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug2.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.


        Parameters
        ----------
        bounding_boxes_on_images : list of imgaug2.augmentables.bbs.BoundingBoxesOnImage
            Polygons to augment. They may be changed in-place.

        random_state : imgaug2.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug2.augmenters.meta.Augmenter
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_bounding_boxes`.

        hooks : imgaug2.imgaug2.HooksKeypoints or None
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_bounding_boxes`.

        Returns
        -------
        list of imgaug2.augmentables.bbs.BoundingBoxesOnImage
            The augmented bounding boxes.

        """
        return self._augment_cbaois_as_keypoints(
            bounding_boxes_on_images, random_state=random_state, parents=parents, hooks=hooks
        )

    def _augment_polygons(
        self,
        polygons_on_images: ia.PolygonsOnImage | list[ia.PolygonsOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksPolygons | None,
    ) -> ia.PolygonsOnImage | list[ia.PolygonsOnImage]:
        """Augment a batch of polygons on images in-place.

        This is the internal version of :func:`Augmenter.augment_polygons`.
        It is called from :func:`Augmenter.augment_polygons` and should
        usually not be called directly.
        This method may transform the polygons in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug2.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.

        Parameters
        ----------
        polygons_on_images : list of imgaug2.augmentables.polys.PolygonsOnImage
            Polygons to augment. They may be changed in-place.

        random_state : imgaug2.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug2.augmenters.meta.Augmenter
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_polygons`.

        hooks : imgaug2.imgaug2.HooksKeypoints or None
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_polygons`.

        Returns
        -------
        list of imgaug2.augmentables.polys.PolygonsOnImage
            The augmented polygons.

        """
        return self._augment_cbaois_as_keypoints(
            polygons_on_images, random_state=random_state, parents=parents, hooks=hooks
        )

    def _augment_line_strings(
        self,
        line_strings_on_images: ia.LineStringsOnImage | list[ia.LineStringsOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksLineStrings | None,
    ) -> ia.LineStringsOnImage | list[ia.LineStringsOnImage]:
        """Augment a batch of line strings in-place.

        This is the internal version of
        :func:`Augmenter.augment_line_strings`.
        It is called from :func:`Augmenter.augment_line_strings` and should
        usually not be called directly.
        This method may transform the line strings in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug2.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.

        Parameters
        ----------
        line_strings_on_images : list of imgaug2.augmentables.lines.LineStringsOnImage
            Line strings to augment. They may be changed in-place.

        random_state : imgaug2.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug2.augmenters.meta.Augmenter
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_line_strings`.

        hooks : imgaug2.imgaug2.HooksKeypoints or None
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_line_strings`.

        Returns
        -------
        list of imgaug2.augmentables.lines.LineStringsOnImage
            The augmented line strings.

        """
        return self._augment_cbaois_as_keypoints(
            line_strings_on_images, random_state=random_state, parents=parents, hooks=hooks
        )

    @legacy(version="0.4.0")
    def _augment_bounding_boxes_as_keypoints(
        self,
        bounding_boxes_on_images: ia.BoundingBoxesOnImage | list[ia.BoundingBoxesOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksBoundingBoxes | None,
    ) -> ia.BoundingBoxesOnImage | list[ia.BoundingBoxesOnImage]:
        """
        Augment BBs by applying keypoint augmentation to their corners.


        Parameters
        ----------
        bounding_boxes_on_images : list of imgaug2.augmentables.bbs.BoundingBoxesOnImages or imgaug2.augmentables.bbs.BoundingBoxesOnImages
            Bounding boxes to augment. They may be changed in-place.

        random_state : imgaug2.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug2.augmenters.meta.Augmenter
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_polygons`.

        hooks : imgaug2.imgaug2.HooksKeypoints or None
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_polygons`.

        Returns
        -------
        list of imgaug2.augmentables.bbs.BoundingBoxesOnImage or imgaug2.augmentables.bbs.BoundingBoxesOnImage
            The augmented bounding boxes.

        """
        return self._augment_cbaois_as_keypoints(
            bounding_boxes_on_images, random_state=random_state, parents=parents, hooks=hooks
        )

    def _augment_polygons_as_keypoints(
        self,
        polygons_on_images: ia.PolygonsOnImage | list[ia.PolygonsOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksPolygons | None,
        recoverer: object | None = None,
    ) -> ia.PolygonsOnImage | list[ia.PolygonsOnImage]:
        """
        Augment polygons by applying keypoint augmentation to their vertices.

        .. warning::

            This method calls
            :func:`~imgaug2.augmenters.meta.Augmenter._augment_keypoints` and
            expects it to do keypoint augmentation. The default for that
            method is to do nothing. It must therefore be overwritten,
            otherwise the polygon augmentation will also do nothing.

        Parameters
        ----------
        polygons_on_images : list of imgaug2.augmentables.polys.PolygonsOnImage or imgaug2.augmentables.polys.PolygonsOnImage
            Polygons to augment. They may be changed in-place.

        random_state : imgaug2.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug2.augmenters.meta.Augmenter
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_polygons`.

        hooks : imgaug2.imgaug2.HooksKeypoints or None
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_polygons`.

        recoverer : None or imgaug2.augmentables.polys._ConcavePolygonRecoverer
            An instance used to repair invalid polygons after augmentation.
            Must offer the method
            ``recover_from(new_exterior, old_polygon, random_state=0)``.
            If ``None`` then invalid polygons are not repaired.

        Returns
        -------
        list of imgaug2.augmentables.polys.PolygonsOnImage or imgaug2.augmentables.polys.PolygonsOnImage
            The augmented polygons.

        """
        func = functools.partial(
            self._augment_keypoints, random_state=random_state, parents=parents, hooks=hooks
        )

        return self._apply_to_polygons_as_keypoints(
            polygons_on_images, func, recoverer, random_state
        )

    def _augment_line_strings_as_keypoints(
        self,
        line_strings_on_images: ia.LineStringsOnImage | list[ia.LineStringsOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksLineStrings | None,
    ) -> ia.LineStringsOnImage | list[ia.LineStringsOnImage]:
        """
        Augment BBs by applying keypoint augmentation to their corners.

        Parameters
        ----------
        line_strings_on_images : list of imgaug2.augmentables.lines.LineStringsOnImages or imgaug2.augmentables.lines.LineStringsOnImages
            Line strings to augment. They may be changed in-place.

        random_state : imgaug2.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug2.augmenters.meta.Augmenter
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_polygons`.

        hooks : imgaug2.imgaug2.HooksKeypoints or None
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_polygons`.

        Returns
        -------
        list of imgaug2.augmentables.lines.LineStringsOnImages or imgaug2.augmentables.lines.LineStringsOnImages
            The augmented line strings.

        """
        return self._augment_cbaois_as_keypoints(
            line_strings_on_images, random_state=random_state, parents=parents, hooks=hooks
        )

    @legacy(version="0.4.0")
    def _augment_cbaois_as_keypoints(
        self,
        cbaois: ia.BoundingBoxesOnImage
        | ia.PolygonsOnImage
        | ia.LineStringsOnImage
        | list[ia.BoundingBoxesOnImage | ia.PolygonsOnImage | ia.LineStringsOnImage],
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksKeypoints | None,
    ) -> (
        ia.BoundingBoxesOnImage
        | ia.PolygonsOnImage
        | ia.LineStringsOnImage
        | list[ia.BoundingBoxesOnImage | ia.PolygonsOnImage | ia.LineStringsOnImage]
    ):
        """
        Augment bounding boxes by applying KP augmentation to their corners.


        Parameters
        ----------
        cbaois : list of imgaug2.augmentables.bbs.BoundingBoxesOnImage or list of imgaug2.augmentables.polys.PolygonsOnImage or list of imgaug2.augmentables.lines.LineStringsOnImage or imgaug2.augmentables.bbs.BoundingBoxesOnImage or imgaug2.augmentables.polys.PolygonsOnImage or imgaug2.augmentables.lines.LineStringsOnImage
            Coordinate-based augmentables to augment. They may be changed
            in-place.

        random_state : imgaug2.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug2.augmenters.meta.Augmenter
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_batch`.

        hooks : imgaug2.imgaug2.HooksKeypoints or None
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_batch`.

        Returns
        -------
        list of imgaug2.augmentables.bbs.BoundingBoxesOnImage or list of imgaug2.augmentables.polys.PolygonsOnImage or list of imgaug2.augmentables.lines.LineStringsOnImage or imgaug2.augmentables.bbs.BoundingBoxesOnImage or imgaug2.augmentables.polys.PolygonsOnImage or imgaug2.augmentables.lines.LineStringsOnImage
            The augmented coordinate-based augmentables.

        """
        func = functools.partial(
            self._augment_keypoints, random_state=random_state, parents=parents, hooks=hooks
        )
        return self._apply_to_cbaois_as_keypoints(cbaois, func)

    @legacy(version="0.4.0")
    @classmethod
    def _apply_to_polygons_as_keypoints(
        cls,
        polygons_on_images: ia.PolygonsOnImage | list[ia.PolygonsOnImage],
        func: Callable[
            [ia.KeypointsOnImage | list[ia.KeypointsOnImage]],
            ia.KeypointsOnImage | list[ia.KeypointsOnImage],
        ],
        recoverer: object | None = None,
        random_state: iarandom.RNG | None = None,
    ) -> ia.PolygonsOnImage | list[ia.PolygonsOnImage]:
        """
        Apply a callback to polygons in keypoint-representation.


        Parameters
        ----------
        polygons_on_images : list of imgaug2.augmentables.polys.PolygonsOnImage or imgaug2.augmentables.polys.PolygonsOnImage
            Polygons to augment. They may be changed in-place.

        func : callable
            The function to apply. Receives a list of
            :class:`~imgaug2.augmentables.kps.KeypointsOnImage` instances as its
            only parameter.

        recoverer : None or imgaug2.augmentables.polys._ConcavePolygonRecoverer
            An instance used to repair invalid polygons after augmentation.
            Must offer the method
            ``recover_from(new_exterior, old_polygon, random_state=0)``.
            If ``None`` then invalid polygons are not repaired.

        random_state : None or imgaug2.random.RNG
            The random state to use for the recoverer.

        Returns
        -------
        list of imgaug2.augmentables.polys.PolygonsOnImage or imgaug2.augmentables.polys.PolygonsOnImage
            The augmented polygons.

        """
        from imgaug2.augmentables.polys import recover_psois_

        psois_orig = None
        if recoverer is not None:
            if isinstance(polygons_on_images, list):
                psois_orig = [psoi.deepcopy() for psoi in polygons_on_images]
            else:
                psois_orig = polygons_on_images.deepcopy()

        psois = cls._apply_to_cbaois_as_keypoints(polygons_on_images, func)

        if recoverer is None:
            return psois

        # Its not really necessary to create an RNG copy for the recoverer
        # here, as the augmentation of the polygons is already finished and
        # used the same samples as the image augmentation. The recoverer might
        # advance the RNG state, but the next call to e.g. augment() will then
        # still use the same (advanced) RNG state for images and polygons.
        # We copy here anyways as it seems cleaner.
        random_state_recoverer = random_state.copy() if random_state is not None else None
        psois = recover_psois_(psois, psois_orig, recoverer, random_state_recoverer)

        return psois

    @legacy(version="0.4.0")
    @classmethod
    def _apply_to_cbaois_as_keypoints(
        cls,
        cbaois: ia.BoundingBoxesOnImage
        | ia.PolygonsOnImage
        | ia.LineStringsOnImage
        | list[ia.BoundingBoxesOnImage | ia.PolygonsOnImage | ia.LineStringsOnImage],
        func: Callable[
            [ia.KeypointsOnImage | list[ia.KeypointsOnImage]],
            ia.KeypointsOnImage | list[ia.KeypointsOnImage],
        ],
    ) -> (
        ia.BoundingBoxesOnImage
        | ia.PolygonsOnImage
        | ia.LineStringsOnImage
        | list[ia.BoundingBoxesOnImage | ia.PolygonsOnImage | ia.LineStringsOnImage]
    ):
        """
        Augment bounding boxes by applying KP augmentation to their corners.


        Parameters
        ----------
        cbaois : list of imgaug2.augmentables.bbs.BoundingBoxesOnImage or list of imgaug2.augmentables.polys.PolygonsOnImage or list of imgaug2.augmentables.lines.LineStringsOnImage or imgaug2.augmentables.bbs.BoundingBoxesOnImage or imgaug2.augmentables.polys.PolygonsOnImage or imgaug2.augmentables.lines.LineStringsOnImage
            Coordinate-based augmentables to augment. They may be changed
            in-place.

        func : callable
            The function to apply. Receives a list of
            :class:`~imgaug2.augmentables.kps.KeypointsOnImage` instances as its
            only parameter.

        Returns
        -------
        list of imgaug2.augmentables.bbs.BoundingBoxesOnImage or list of imgaug2.augmentables.polys.PolygonsOnImage or list of imgaug2.augmentables.lines.LineStringsOnImage or imgaug2.augmentables.bbs.BoundingBoxesOnImage or imgaug2.augmentables.polys.PolygonsOnImage or imgaug2.augmentables.lines.LineStringsOnImage
            The augmented coordinate-based augmentables.

        """
        from imgaug2.augmentables.utils import (
            convert_cbaois_to_kpsois,
            invert_convert_cbaois_to_kpsois_,
        )

        kpsois = convert_cbaois_to_kpsois(cbaois)
        kpsois_aug = func(kpsois)
        return invert_convert_cbaois_to_kpsois_(cbaois, kpsois_aug)
