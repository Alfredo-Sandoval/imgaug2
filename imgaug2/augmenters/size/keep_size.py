"""Keep-size wrapper augmenters."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any, Literal, cast, overload

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, Images, RNGInput
from imgaug2.compat.markers import legacy

from ._utils import (
    KeepSizeByResizeInterpolationInput,
    KeepSizeByResizeInterpolationMapsInput,
    _mlx_resize_order_from_interpolation,
)
class KeepSizeByResize(meta.Augmenter):
    """Resize images back to their input sizes after applying child augmenters.

    Combining this with e.g. a cropping augmenter as the child will lead to
    images being resized back to the input size after the crop operation was
    applied. Some augmenters have a ``keep_size`` argument that achieves the
    same goal (if set to ``True``), though this augmenter offers control over
    the interpolation mode and which augmentables to resize (images, heatmaps,
    segmentation maps).

    **Supported dtypes**:

    See :func:`~imgaug2.imgaug2.imresize_many_images`.

    Parameters
    ----------
    children : Augmenter or list of imgaug2.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to images. These augmenters may change
        the image size.

    interpolation : KeepSizeByResize.NO_RESIZE or {'nearest', 'linear', 'area', 'cubic'} or {cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC} or list of str or list of int or StochasticParameter, optional
        The interpolation mode to use when resizing images.
        Can take any value that :func:`~imgaug2.imgaug2.imresize_single_image`
        accepts, e.g. ``cubic``.

            * If this is ``KeepSizeByResize.NO_RESIZE`` then images will not
              be resized.
            * If this is a single ``str``, it is expected to have one of the
              following values: ``nearest``, ``linear``, ``area``, ``cubic``.
            * If this is a single integer, it is expected to have a value
              identical to one of: ``cv2.INTER_NEAREST``,
              ``cv2.INTER_LINEAR``, ``cv2.INTER_AREA``, ``cv2.INTER_CUBIC``.
            * If this is a ``list`` of ``str`` or ``int``, it is expected that
              each ``str``/``int`` is one of the above mentioned valid ones.
              A random one of these values will be sampled per image.
            * If this is a ``StochasticParameter``, it will be queried once per
              call to ``_augment_images()`` and must return ``N`` ``str`` s or
              ``int`` s (matching the above mentioned ones) for ``N`` images.

    interpolation_heatmaps : KeepSizeByResize.SAME_AS_IMAGES or KeepSizeByResize.NO_RESIZE or {'nearest', 'linear', 'area', 'cubic'} or {cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC} or list of str or list of int or StochasticParameter, optional
        The interpolation mode to use when resizing heatmaps.
        Meaning and valid values are similar to `interpolation`. This
        parameter may also take the value ``KeepSizeByResize.SAME_AS_IMAGES``,
        which will lead to copying the interpolation modes used for the
        corresponding images. The value may also be returned on a per-image
        basis if `interpolation_heatmaps` is provided as a
        ``StochasticParameter`` or may be one possible value if it is
        provided as a ``list`` of ``str``.

    interpolation_segmaps : KeepSizeByResize.SAME_AS_IMAGES or KeepSizeByResize.NO_RESIZE or {'nearest', 'linear', 'area', 'cubic'} or {cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC} or list of str or list of int or StochasticParameter, optional
        The interpolation mode to use when resizing segmentation maps.
        Similar to `interpolation_heatmaps`.
        **Note**: For segmentation maps, only ``NO_RESIZE`` or nearest
        neighbour interpolation (i.e. ``nearest``) make sense in the vast
        majority of all cases.

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
    >>> aug = iaa.KeepSizeByResize(
    >>>     iaa.Crop((20, 40), keep_size=False)
    >>> )

    Apply random cropping to input images, then resize them back to their
    original input sizes. The resizing is done using this augmenter instead
    of the corresponding internal resizing operation in ``Crop``.

    >>> aug = iaa.KeepSizeByResize(
    >>>     iaa.Crop((20, 40), keep_size=False),
    >>>     interpolation="nearest"
    >>> )

    Same as in the previous example, but images are now always resized using
    nearest neighbour interpolation.

    >>> aug = iaa.KeepSizeByResize(
    >>>     iaa.Crop((20, 40), keep_size=False),
    >>>     interpolation=["nearest", "cubic"],
    >>>     interpolation_heatmaps=iaa.KeepSizeByResize.SAME_AS_IMAGES,
    >>>     interpolation_segmaps=iaa.KeepSizeByResize.NO_RESIZE
    >>> )

    Similar to the previous example, but images are now sometimes resized
    using linear interpolation and sometimes using nearest neighbour
    interpolation. Heatmaps are resized using the same interpolation as was
    used for the corresponding image. Segmentation maps are not resized and
    will therefore remain at their size after cropping.

    """

    NO_RESIZE = "NO_RESIZE"
    SAME_AS_IMAGES = "SAME_AS_IMAGES"

    def __init__(
        self,
        children: meta.Augmenter | Sequence[meta.Augmenter] | None,
        interpolation: KeepSizeByResizeInterpolationInput = "cubic",
        interpolation_heatmaps: KeepSizeByResizeInterpolationMapsInput = SAME_AS_IMAGES,
        interpolation_segmaps: KeepSizeByResizeInterpolationMapsInput = "nearest",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.children = children

        @overload
        def _validate_param(
            val: KeepSizeByResizeInterpolationInput, allow_same_as_images: Literal[False]
        ) -> iap.StochasticParameter: ...

        @overload
        def _validate_param(
            val: KeepSizeByResizeInterpolationMapsInput, allow_same_as_images: Literal[True]
        ) -> iap.StochasticParameter | Literal["SAME_AS_IMAGES"]: ...

        @iap._prefetchable_str
        def _validate_param(
            val: KeepSizeByResizeInterpolationMapsInput, allow_same_as_images: bool
        ) -> iap.StochasticParameter | Literal["SAME_AS_IMAGES"]:
            valid_ips_and_resize = ia.IMRESIZE_VALID_INTERPOLATIONS + [KeepSizeByResize.NO_RESIZE]
            if allow_same_as_images and val == self.SAME_AS_IMAGES:
                return self.SAME_AS_IMAGES
            if val in valid_ips_and_resize:
                return iap.Deterministic(val)
            if isinstance(val, list):
                assert len(val) > 0, (
                    "Expected a list of at least one interpolation method. Got an empty list."
                )
                valid_ips_here = valid_ips_and_resize
                if allow_same_as_images:
                    valid_ips_here = valid_ips_here + [KeepSizeByResize.SAME_AS_IMAGES]
                only_valid_ips = all([ip in valid_ips_here for ip in val])
                assert only_valid_ips, (
                    f"Expected each interpolations to be one of '{str(valid_ips_here)}', got "
                    f"'{str(val)}'."
                )
                return iap.Choice(val)
            if isinstance(val, iap.StochasticParameter):
                return val
            raise Exception(
                "Expected interpolation to be one of "
                f"'{ia.IMRESIZE_VALID_INTERPOLATIONS}' or a list of these "
                f"values or a StochasticParameter. Got type {type(val)}."
            )

        self.children = meta.handle_children_list(children, self.name, "then")
        self.interpolation = _validate_param(interpolation, False)
        self.interpolation_heatmaps = _validate_param(interpolation_heatmaps, True)
        self.interpolation_segmaps = _validate_param(interpolation_segmaps, True)

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        with batch.propagation_hooks_ctx(self, hooks, parents):
            images_were_array = None
            if batch.images is not None:
                images_were_array = ia.is_np_array(batch.images)
            shapes_orig = self._get_shapes(batch)

            samples = self._draw_samples(batch.nb_rows, random_state)

            batch = self.children.augment_batch_(batch, parents=parents + [self], hooks=hooks)

            if batch.images is not None:
                batch.images = self._keep_size_images(
                    batch.images, shapes_orig["images"], images_were_array, samples
                )

            if batch.heatmaps is not None:
                # dont use shapes_orig["images"] because they might be None
                batch.heatmaps = self._keep_size_maps(
                    batch.heatmaps, shapes_orig["heatmaps"], shapes_orig["heatmaps_arr"], samples[1]
                )

            if batch.segmentation_maps is not None:
                # dont use shapes_orig["images"] because they might be None
                batch.segmentation_maps = self._keep_size_maps(
                    batch.segmentation_maps,
                    shapes_orig["segmentation_maps"],
                    shapes_orig["segmentation_maps_arr"],
                    samples[2],
                )

            for augm_name in ["keypoints", "bounding_boxes", "polygons", "line_strings"]:
                augm_value = getattr(batch, augm_name)
                if augm_value is not None:
                    func = functools.partial(
                        self._keep_size_keypoints,
                        shapes_orig=shapes_orig[augm_name],
                        interpolations=samples[0],
                    )
                    cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                    setattr(batch, augm_name, cbaois)
        return batch

    @legacy(version="0.4.0")
    @classmethod
    def _keep_size_images(
        cls,
        images: Images,
        shapes_orig: list[Shape],
        images_were_array: bool,
        samples: tuple[Array, Array, Array],
    ) -> Images:
        interpolations, _, _ = samples

        gen = zip(images, interpolations, shapes_orig, strict=True)
        result = []
        for image, interpolation, input_shape in gen:
            if interpolation == KeepSizeByResize.NO_RESIZE:
                result.append(image)
            else:
                from imgaug2.mlx._core import is_mlx_array

                if is_mlx_array(image):
                    import imgaug2.mlx as mlx

                    order = _mlx_resize_order_from_interpolation(interpolation)
                    if order is None:
                        image_np = mlx.to_numpy(image)
                        image_np = ia.imresize_single_image(
                            image_np, input_shape[0:2], interpolation
                        )
                        result.append(mlx.to_mlx(image_np))
                    else:
                        result.append(mlx.geometry.resize(image, input_shape[0:2], order=order))
                else:
                    result.append(ia.imresize_single_image(image, input_shape[0:2], interpolation))

        if images_were_array:
            # note here that NO_RESIZE can have led to different shapes
            nb_shapes = len({image.shape for image in result})
            if nb_shapes == 1:
                # images.dtype does not necessarily work anymore, children
                # might have turned 'images' into list
                result = np.array(result, dtype=result[0].dtype)

        return result

    @legacy(version="0.4.0")
    @classmethod
    def _keep_size_maps(
        cls,
        augmentables: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage],
        shapes_orig_images: list[Shape],
        shapes_orig_arrs: list[Shape],
        interpolations: Array,
    ) -> list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage]:
        result = []
        gen = zip(augmentables, interpolations, shapes_orig_arrs, shapes_orig_images, strict=True)
        for augmentable, interpolation, arr_shape_orig, img_shape_orig in gen:
            if interpolation == "NO_RESIZE":
                result.append(augmentable)
            else:
                augmentable = augmentable.resize(arr_shape_orig[0:2], interpolation=interpolation)
                augmentable.shape = img_shape_orig
                result.append(augmentable)

        return result

    @legacy(version="0.4.0")
    @classmethod
    def _keep_size_keypoints(
        cls, kpsois_aug: list[ia.KeypointsOnImage], shapes_orig: list[Shape], interpolations: Array
    ) -> list[ia.KeypointsOnImage]:
        result = []
        gen = zip(kpsois_aug, interpolations, shapes_orig, strict=True)
        for kpsoi_aug, interpolation, input_shape in gen:
            if interpolation == KeepSizeByResize.NO_RESIZE:
                result.append(kpsoi_aug)
            else:
                result.append(kpsoi_aug.on_(input_shape))

        return result

    @legacy(version="0.4.0")
    @classmethod
    def _get_shapes(cls, batch: _BatchInAugmentation) -> dict[str, list[Shape]]:
        result = dict()
        for column in batch.columns:
            result[column.name] = [cast(Any, cell).shape for cell in column.value]

        if batch.heatmaps is not None:
            result["heatmaps_arr"] = [cell.arr_0to1.shape for cell in batch.heatmaps]

        if batch.segmentation_maps is not None:
            result["segmentation_maps_arr"] = [cell.arr.shape for cell in batch.segmentation_maps]

        return result

    def _draw_samples(
        self, nb_images: int, random_state: iarandom.RNG
    ) -> tuple[Array, Array, Array]:
        rngs = random_state.duplicate(3)
        interpolations = self.interpolation.draw_samples((nb_images,), random_state=rngs[0])

        if self.interpolation_heatmaps == KeepSizeByResize.SAME_AS_IMAGES:
            interpolations_heatmaps = np.copy(interpolations)
        else:
            interpolations_heatmaps = self.interpolation_heatmaps.draw_samples(
                (nb_images,), random_state=rngs[1]
            )

            # Note that `interpolations_heatmaps == self.SAME_AS_IMAGES`
            # works here only if the datatype of the array is such that it
            # may contain strings. It does not work properly for e.g.
            # integer arrays and will produce a single bool output, even
            # for arrays with more than one entry.
            same_as_imgs_idx = [ip == self.SAME_AS_IMAGES for ip in interpolations_heatmaps]

            interpolations_heatmaps[same_as_imgs_idx] = interpolations[same_as_imgs_idx]

        if self.interpolation_segmaps == KeepSizeByResize.SAME_AS_IMAGES:
            interpolations_segmaps = np.copy(interpolations)
        else:
            # TODO This used previously the same seed as the heatmaps part
            #      leading to the same sampled values. Was that intentional?
            #      Doesn't look like it should be that way.
            interpolations_segmaps = self.interpolation_segmaps.draw_samples(
                (nb_images,), random_state=rngs[2]
            )

            # Note that `interpolations_heatmaps == self.SAME_AS_IMAGES`
            # works here only if the datatype of the array is such that it
            # may contain strings. It does not work properly for e.g.
            # integer arrays and will produce a single bool output, even
            # for arrays with more than one entry.
            same_as_imgs_idx = [ip == self.SAME_AS_IMAGES for ip in interpolations_segmaps]

            interpolations_segmaps[same_as_imgs_idx] = interpolations[same_as_imgs_idx]

        return interpolations, interpolations_heatmaps, interpolations_segmaps

    def _to_deterministic(self) -> KeepSizeByResize:
        aug = self.copy()
        aug.children = aug.children.to_deterministic()
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.interpolation, self.interpolation_heatmaps]

    def get_children_lists(self) -> list[list[meta.Augmenter]]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_children_lists`."""
        return cast(list[list[meta.Augmenter]], [self.children])

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"interpolation={self.interpolation}, "
            f"interpolation_heatmaps={self.interpolation_heatmaps}, "
            f"name={self.name}, "
            f"children={self.children}, "
            f"deterministic={self.deterministic}"
            ")"
        )
