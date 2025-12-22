"""Augmenters that apply affine or similar geometric transformations.

This module contains augmenters for affine transformations (rotation, shearing,
scaling, translation) and other geometric distortions like piecewise affine
warping, perspective transforms, and elastic deformations.

Key Augmenters:
    - `Affine`: Standard affine transformations.
    - `PiecewiseAffine`: Local distortions via grid control points.
    - `PerspectiveTransform`: Four-point perspective warping.
    - `ElasticTransformation`: Moving pixels locally using distortion fields.
    - `Jigsaw`: Jigsaw puzzle-like shuffling of image parts.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters._typing import Array, Images, RNGInput
from imgaug2.compat.markers import legacy

from . import meta
from .geometry.affine import (
    Affine,
    AffineCv2,
    Rotate,
    ScaleX,
    ScaleY,
    ShearX,
    ShearY,
    TranslateX,
    TranslateY,
)
from .geometry.distortions import (
    GridDistortion,
    OpticalDistortion,
    _GridDistortionShiftMapGenerator,
)
from .geometry.elastic import ElasticTransformation
from .geometry.jigsaw import (
    Jigsaw,
    apply_jigsaw,
    apply_jigsaw_to_coords,
    generate_jigsaw_destinations,
    _JigsawSamples,
)
from .geometry.piecewise import PiecewiseAffine
from .geometry.perspective import PerspectiveTransform
from .geometry.polar import WithPolarWarping

# Local type aliases (more specific than _typing versions for this module)
Shape2D: TypeAlias = tuple[int, int]

class Rot90(meta.Augmenter):
    """
    Rotate images clockwise by multiples of 90 degrees.

    This could also be achieved using ``Affine``, but ``Rot90`` is
    significantly more efficient.

    **Supported dtypes**:

    if (keep_size=False):

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

    if (keep_size=True):

        minimum of (
            ``imgaug2.augmenters.geometric.Rot90(keep_size=False)``,
            :func:`~imgaug2.imgaug2.imresize_many_images`
        )

    Parameters
    ----------
    k : int or list of int or tuple of int or imaug.ALL or imgaug2.parameters.StochasticParameter, optional
        How often to rotate clockwise by 90 degrees.

            * If a single ``int``, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the discrete interval ``[a..b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``imgaug2.ALL``, then equivalant to list ``[0, 1, 2, 3]``.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the value to use.

    keep_size : bool, optional
        After rotation by an odd-valued `k` (e.g. 1 or 3), the resulting image
        may have a different height/width than the original image.
        If this parameter is set to ``True``, then the rotated
        image will be resized to the input image's size. Note that this might
        also cause the augmented image to look distorted.

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
    >>> aug = iaa.Rot90(1)

    Rotate all images by 90 degrees.
    Resize these images afterwards to keep the size that they had before
    augmentation.
    This may cause the images to look distorted.

    >>> aug = iaa.Rot90([1, 3])

    Rotate all images by 90 or 270 degrees.
    Resize these images afterwards to keep the size that they had before
    augmentation.
    This may cause the images to look distorted.

    >>> aug = iaa.Rot90((1, 3))

    Rotate all images by 90, 180 or 270 degrees.
    Resize these images afterwards to keep the size that they had before
    augmentation.
    This may cause the images to look distorted.

    >>> aug = iaa.Rot90((1, 3), keep_size=False)

    Rotate all images by 90, 180 or 270 degrees.
    Does not resize to the original image size afterwards, i.e. each image's
    size may change.

    """

    def __init__(
        self,
        k: int | tuple[int, int] | list[int] | iap.StochasticParameter | Literal["ALL"] = 1,
        keep_size: bool = True,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        if k == ia.ALL:
            k = [0, 1, 2, 3]
        self.k = iap.handle_discrete_param(
            k, "k", value_range=None, tuple_to_uniform=True, list_to_choice=True, allow_floats=False
        )

        self.keep_size = keep_size

    def _draw_samples(self, nb_images: int, random_state: iarandom.RNG) -> Array:
        return self.k.draw_samples((nb_images,), random_state=random_state)

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        ks = self._draw_samples(batch.nb_rows, random_state)

        if batch.images is not None:
            batch.images = self._augment_arrays_by_samples(
                batch.images, ks, self.keep_size, ia.imresize_single_image
            )

        if batch.heatmaps is not None:
            batch.heatmaps = self._augment_maps_by_samples(batch.heatmaps, "arr_0to1", ks)

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps, "arr", ks
            )

        for augm_name in ["keypoints", "bounding_boxes", "polygons", "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(self._augment_keypoints_by_samples, ks=ks)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    @classmethod
    def _augment_arrays_by_samples(
        cls,
        arrs: Images,
        ks: Array,
        keep_size: bool,
        resize_func: Callable[[Array, Shape2D], Array] | None,
    ) -> Images:
        from imgaug2.mlx._core import is_mlx_array, mx

        input_was_array = ia.is_np_array(arrs)
        input_dtype = arrs.dtype if input_was_array else None
        arrs_aug = []
        for arr, k_i in zip(arrs, ks, strict=True):
            if is_mlx_array(arr):
                import imgaug2.mlx as mlx

                # Ty does not currently narrow via TypeGuard when the guarded type is
                # defined on an optional dependency module. Use an explicit isinstance
                # assertion to narrow.
                assert isinstance(arr, mx.array)
                arr_aug = mlx.rot90(arr, k=int(k_i))
                assert isinstance(arr_aug, mx.array)
                if keep_size and arr.shape != arr_aug.shape:
                    arr_aug = mlx.geometry.resize(
                        arr_aug, (int(arr.shape[0]), int(arr.shape[1])), order=1
                    )
            else:
                # adding axes here rotates clock-wise instead of ccw
                arr_aug = np.rot90(arr, k_i, axes=(1, 0))

                do_resize = keep_size and arr.shape != arr_aug.shape and resize_func is not None
                if do_resize:
                    arr_aug = resize_func(arr_aug, arr.shape[0:2])
            arrs_aug.append(arr_aug)
        if keep_size and input_was_array:
            n_shapes = len({arr.shape for arr in arrs_aug})
            if n_shapes == 1:
                arrs_aug = np.array(arrs_aug, dtype=input_dtype)
        return arrs_aug

    @legacy(version="0.4.0")
    def _augment_maps_by_samples(
        self,
        augmentables: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage],
        arr_attr_name: str,
        ks: Array,
    ) -> list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage]:
        arrs = [getattr(map_i, arr_attr_name) for map_i in augmentables]
        arrs_aug = self._augment_arrays_by_samples(arrs, ks, self.keep_size, None)

        maps_aug = []
        gen = zip(augmentables, arrs, arrs_aug, ks, strict=True)
        for augmentable_i, arr, arr_aug, k_i in gen:
            shape_orig = arr.shape
            setattr(augmentable_i, arr_attr_name, arr_aug)
            if self.keep_size:
                augmentable_i = augmentable_i.resize(shape_orig[0:2])
            elif k_i % 2 == 1:
                h, w = augmentable_i.shape[0:2]
                augmentable_i.shape = tuple([w, h] + list(augmentable_i.shape[2:]))
            else:
                # keep_size was False, but rotated by a multiple of 2,
                # hence height and width do not change
                pass
            maps_aug.append(augmentable_i)
        return maps_aug

    @legacy(version="0.4.0")
    def _augment_keypoints_by_samples(
        self, keypoints_on_images: list[ia.KeypointsOnImage], ks: Array
    ) -> list[ia.KeypointsOnImage]:
        result = []
        for kpsoi_i, k_i in zip(keypoints_on_images, ks, strict=True):
            shape_orig = kpsoi_i.shape

            if (k_i % 4) == 0:
                result.append(kpsoi_i)
            else:
                k_i = int(k_i) % 4  # this is also correct when k_i is negative
                h, w = kpsoi_i.shape[0:2]
                h_aug, w_aug = (h, w) if (k_i % 2) == 0 else (w, h)

                # Vectorized rotation: compute final position directly based on k_i
                # k_i=1: (x, y) -> (h - y, x)
                # k_i=2: (x, y) -> (w - x, h - y)
                # k_i=3: (x, y) -> (y, w - x)
                for kp in kpsoi_i.keypoints:
                    y, x = kp.y, kp.x
                    if k_i == 1:
                        kp.x, kp.y = h - y, x
                    elif k_i == 2:
                        kp.x, kp.y = w - x, h - y
                    else:  # k_i == 3
                        kp.x, kp.y = y, w - x

                shape_aug = tuple([h_aug, w_aug] + list(kpsoi_i.shape[2:]))
                kpsoi_i.shape = shape_aug

                if self.keep_size and (h, w) != (h_aug, w_aug):
                    kpsoi_i = kpsoi_i.on_(shape_orig)
                    kpsoi_i.shape = shape_orig

                result.append(kpsoi_i)
        return result

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.k, self.keep_size]
