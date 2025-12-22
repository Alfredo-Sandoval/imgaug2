"""Polar warping augmenter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, cast

import cv2
import numpy as np
from numpy.typing import NDArray

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters._typing import Array, Images, RNGInput
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _normalize_cv2_input_arr_

from .. import meta

# Local type aliases (more specific than _typing versions for this module)
Shape: TypeAlias = tuple[int, ...]
Shape2D: TypeAlias = tuple[int, int]
Coords: TypeAlias = NDArray[np.floating[Any]]
# TODO semipolar
@legacy(version="0.4.0")
class WithPolarWarping(meta.Augmenter):
    """Augmenter that applies other augmenters in a polar-transformed space.

    This augmenter first transforms an image into a polar representation,
    then applies its child augmenter, then transforms back to cartesian
    space. The polar representation is still in the image's input dtype
    (i.e. ``uint8`` stays ``uint8``) and can be visualized. It can be thought
    of as an "unrolled" version of the image, where previously circular lines
    appear straight. Hence, applying child augmenters in that space can lead
    to circular effects. E.g. replacing rectangular pixel areas in the polar
    representation with black pixels will lead to curved black areas in
    the cartesian result.

    This augmenter can create new pixels in the image. It will fill these
    with black pixels. For segmentation maps it will fill with class
    id ``0``. For heatmaps it will fill with ``0.0``.

    This augmenter is limited to arrays with a height and/or width of
    ``32767`` or less.

    .. warning::

        When augmenting coordinates in polar representation, it is possible
        that these are shifted outside of the polar image, but are inside the
        image plane after transforming back to cartesian representation,
        usually on newly created pixels (i.e. black backgrounds).
        These coordinates are currently not removed. It is recommended to
        not use very strong child transformations when also augmenting
        coordinate-based augmentables.

    .. warning::

        For bounding boxes, this augmenter suffers from the same problem as
        affine rotations applied to bounding boxes, i.e. the resulting
        bounding boxes can have unintuitive (seemingly wrong) appearance.
        This is due to coordinates being "rotated" that are inside the
        bounding box, but do not fall on the object and actually are
        background.
        It is recommended to use this augmenter with caution when augmenting
        bounding boxes.

    .. warning::

        For polygons, this augmenter should not be combined with
        augmenters that perform automatic polygon recovery for invalid
        polygons, as the polygons will frequently appear broken in polar
        representation and their "fixed" version will be very broken in
        cartesian representation. Augmenters that perform such polygon
        recovery are currently ``PerspectiveTransform``, ``PiecewiseAffine``
        and ``ElasticTransformation``.


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: no (2)
        * ``float16``: yes; tested (3)
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no (1)
        * ``bool``: yes; tested (4)

        - (1) OpenCV produces error
          ``TypeError: Expected cv::UMat for argument 'src'``
        - (2) OpenCV produces array of nothing but zeros.
        - (3) Mapepd to ``float32``.
        - (4) Mapped to ``uint8``.

    Parameters
    ----------
    children : imgaug2.augmenters.meta.Augmenter or list of imgaug2.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to images after they were transformed
        to polar representation.

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
    >>> aug = iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.1, 0.1)))

    Apply cropping and padding in polar representation, then warp back to
    cartesian representation.

    >>> aug = iaa.WithPolarWarping(
    >>>     iaa.Affine(
    >>>         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    >>>         rotate=(-35, 35),
    >>>         scale=(0.8, 1.2),
    >>>         shear={"x": (-15, 15), "y": (-15, 15)}
    >>>     )
    >>> )

    Apply affine transformations in polar representation.

    >>> aug = iaa.WithPolarWarping(iaa.AveragePooling((2, 8)))

    Apply average pooling in polar representation. This leads to circular
    bins.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        children: meta.Augmenter | Sequence[meta.Augmenter] | None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.children = meta.handle_children_list(children, self.name, "then")

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        if batch.images is not None:
            iadt.gate_dtypes_strs(
                batch.images,
                allowed="bool uint8 uint16 int8 int16 int32 float16 float32 float64",
                disallowed="uint32 uint64 int64 float128",
                augmenter=self,
            )

        with batch.propagation_hooks_ctx(self, hooks, parents):
            batch, inv_data_bbs = self._convert_bbs_to_polygons_(batch)

            inv_data = {}
            for column in batch.columns:
                func = getattr(self, f"_warp_{column.name}_")
                col_aug, inv_data_col = func(column.value)
                setattr(batch, column.attr_name, col_aug)
                inv_data[column.name] = inv_data_col

            batch = self.children.augment_batch_(batch, parents=parents + [self], hooks=hooks)
            for column in batch.columns:
                func = getattr(self, f"_invert_warp_{column.name}_")
                col_unaug = func(column.value, inv_data[column.name])
                setattr(batch, column.attr_name, col_unaug)

            batch = self._invert_convert_bbs_to_polygons_(batch, inv_data_bbs)

        return batch

    @legacy(version="0.4.0")
    @classmethod
    def _convert_bbs_to_polygons_(
        cls, batch: _BatchInAugmentation
    ) -> tuple[_BatchInAugmentation, tuple[bool, bool]]:
        batch_contained_polygons = batch.polygons is not None
        if batch.bounding_boxes is None:
            return batch, (False, batch_contained_polygons)

        psois = [bbsoi.to_polygons_on_image() for bbsoi in batch.bounding_boxes]
        psois = [psoi.subdivide_(2) for psoi in psois]

        # Mark Polygons that are really Bounding Boxes
        for psoi in psois:
            for polygon in psoi.polygons:
                if polygon.label is None:
                    polygon.label = "$$IMGAUG_BB_AS_POLYGON"
                else:
                    polygon.label = polygon.label + ";$$IMGAUG_BB_AS_POLYGON"

        # Merge Fake-Polygons into existing Polygons
        if batch.polygons is None:
            batch.polygons = psois
        else:
            for psoi, bbs_as_psoi in zip(batch.polygons, psois, strict=True):
                assert psoi.shape == bbs_as_psoi.shape, (
                    "Expected polygons and bounding boxes to have the same "
                    f".shape value, got {psoi.shape} and {bbs_as_psoi.shape}."
                )

                psoi.polygons.extend(bbs_as_psoi.polygons)

        batch.bounding_boxes = None

        return batch, (True, batch_contained_polygons)

    @legacy(version="0.4.0")
    @classmethod
    def _invert_convert_bbs_to_polygons_(
        cls, batch: _BatchInAugmentation, inv_data: tuple[bool, bool]
    ) -> _BatchInAugmentation:
        batch_contained_bbs, batch_contained_polygons = inv_data

        if not batch_contained_bbs:
            return batch

        bbsois = []
        for psoi in batch.polygons:
            polygons = []
            bbs = []
            for polygon in psoi.polygons:
                is_bb = False
                if polygon.label is None:
                    is_bb = False
                elif polygon.label == "$$IMGAUG_BB_AS_POLYGON":
                    polygon.label = None
                    is_bb = True
                elif polygon.label.endswith(";$$IMGAUG_BB_AS_POLYGON"):
                    polygon.label = polygon.label[: -len(";$$IMGAUG_BB_AS_POLYGON")]
                    is_bb = True

                if is_bb:
                    bbs.append(polygon.to_bounding_box())
                else:
                    polygons.append(polygon)

            psoi.polygons = polygons
            bbsoi = ia.BoundingBoxesOnImage(bbs, shape=psoi.shape)
            bbsois.append(bbsoi)

        batch.bounding_boxes = bbsois

        if not batch_contained_polygons:
            batch.polygons = None

        return batch

    @legacy(version="0.4.0")
    @classmethod
    def _warp_images_(cls, images: Images | None) -> tuple[list[Array] | None, list[Shape] | None]:
        return cls._warp_arrays(images, False)

    @legacy(version="0.4.0")
    @classmethod
    def _invert_warp_images_(
        cls, images_warped: list[Array] | None, inv_data: object
    ) -> list[Array] | None:
        return cls._invert_warp_arrays(images_warped, False, inv_data)

    @legacy(version="0.4.0")
    @classmethod
    def _warp_heatmaps_(
        cls, heatmaps: list[ia.HeatmapsOnImage] | None
    ) -> tuple[list[ia.HeatmapsOnImage] | None, object]:
        return cls._warp_maps_(heatmaps, "arr_0to1", False)

    @legacy(version="0.4.0")
    @classmethod
    def _invert_warp_heatmaps_(
        cls, heatmaps_warped: list[ia.HeatmapsOnImage] | None, inv_data: object
    ) -> list[ia.HeatmapsOnImage] | None:
        return cls._invert_warp_maps_(heatmaps_warped, "arr_0to1", False, inv_data)

    @legacy(version="0.4.0")
    @classmethod
    def _warp_segmentation_maps_(
        cls, segmentation_maps: list[ia.SegmentationMapsOnImage] | None
    ) -> tuple[list[ia.SegmentationMapsOnImage] | None, object]:
        return cls._warp_maps_(segmentation_maps, "arr", True)

    @legacy(version="0.4.0")
    @classmethod
    def _invert_warp_segmentation_maps_(
        cls,
        segmentation_maps_warped: list[ia.SegmentationMapsOnImage] | None,
        inv_data: object,
    ) -> list[ia.SegmentationMapsOnImage] | None:
        return cls._invert_warp_maps_(segmentation_maps_warped, "arr", True, inv_data)

    @legacy(version="0.4.0")
    @classmethod
    def _warp_keypoints_(
        cls, kpsois: list[ia.KeypointsOnImage] | None
    ) -> tuple[list[ia.KeypointsOnImage] | None, object]:
        return cls._warp_cbaois_(kpsois)

    @legacy(version="0.4.0")
    @classmethod
    def _invert_warp_keypoints_(
        cls, kpsois_warped: list[ia.KeypointsOnImage] | None, image_shapes_orig: object
    ) -> list[ia.KeypointsOnImage] | None:
        return cls._invert_warp_cbaois_(kpsois_warped, image_shapes_orig)

    @legacy(version="0.4.0")
    @classmethod
    def _warp_bounding_boxes_(cls, bbsois: None) -> None:
        assert bbsois is None, "Expected BBs to have been converted to polygons."
        return None

    @legacy(version="0.4.0")
    @classmethod
    def _invert_warp_bounding_boxes_(cls, bbsois_warped: None, _image_shapes_orig: object) -> None:
        assert bbsois_warped is None, "Expected BBs to have been converted to polygons."
        return None

    @legacy(version="0.4.0")
    @classmethod
    def _warp_polygons_(
        cls, psois: list[ia.PolygonsOnImage] | None
    ) -> tuple[list[ia.PolygonsOnImage] | None, object]:
        return cls._warp_cbaois_(psois)

    @legacy(version="0.4.0")
    @classmethod
    def _invert_warp_polygons_(
        cls, psois_warped: list[ia.PolygonsOnImage] | None, image_shapes_orig: object
    ) -> list[ia.PolygonsOnImage] | None:
        return cls._invert_warp_cbaois_(psois_warped, image_shapes_orig)

    @legacy(version="0.4.0")
    @classmethod
    def _warp_line_strings_(
        cls, lsois: list[ia.LineStringsOnImage] | None
    ) -> tuple[list[ia.LineStringsOnImage] | None, object]:
        return cls._warp_cbaois_(lsois)

    @legacy(version="0.4.0")
    @classmethod
    def _invert_warp_line_strings_(
        cls, lsois_warped: list[ia.LineStringsOnImage] | None, image_shapes_orig: object
    ) -> list[ia.LineStringsOnImage] | None:
        return cls._invert_warp_cbaois_(lsois_warped, image_shapes_orig)

    @legacy(version="0.4.0")
    @classmethod
    def _warp_arrays(
        cls, arrays: Images | None, interpolation_nearest: bool
    ) -> tuple[list[Array] | None, list[Shape] | None]:
        if arrays is None:
            return None, None

        flags = cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
        if interpolation_nearest:
            flags += cv2.INTER_NEAREST

        arrays_warped = []
        shapes_orig = []
        for arr in arrays:
            if 0 in arr.shape:
                arrays_warped.append(arr)
                shapes_orig.append(arr.shape)
                continue

            input_dtype = arr.dtype
            if input_dtype.kind == "b":
                arr = arr.astype(np.uint8) * 255
            elif input_dtype == iadt._FLOAT16_DTYPE:
                arr = arr.astype(np.float32)

            height, width = arr.shape[0:2]

            # remap limitation, see docs for warpPolar()
            assert height <= 32767 and width <= 32767, (
                "WithPolarWarping._warp_arrays() can currently only handle "
                f"arrays with axis sizes below 32767, but got shape {arr.shape}. This "
                "is an OpenCV limitation."
            )

            dest_size = (0, 0)
            center_xy = (width / 2, height / 2)
            max_radius = np.sqrt((height / 2.0) ** 2.0 + (width / 2.0) ** 2.0)

            if arr.ndim == 3 and arr.shape[-1] > 512:
                arr_warped = np.stack(
                    [
                        cv2.warpPolar(
                            _normalize_cv2_input_arr_(arr[..., c_idx]),
                            dest_size,
                            center_xy,
                            max_radius,
                            flags,
                        )
                        for c_idx in np.arange(arr.shape[-1])
                    ],
                    axis=-1,
                )
            else:
                arr_warped = cv2.warpPolar(
                    _normalize_cv2_input_arr_(arr), dest_size, center_xy, max_radius, flags
                )
                if arr_warped.ndim == 2 and arr.ndim == 3:
                    arr_warped = arr_warped[:, :, np.newaxis]

            if input_dtype.kind == "b":
                arr_warped = arr_warped > 128
            elif input_dtype == iadt._FLOAT16_DTYPE:
                arr_warped = arr_warped.astype(np.float16)

            arrays_warped.append(arr_warped)
            shapes_orig.append(arr.shape)
        return arrays_warped, shapes_orig

    @legacy(version="0.4.0")
    @classmethod
    def _invert_warp_arrays(
        cls, arrays_warped: list[Array] | None, interpolation_nearest: bool, inv_data: object
    ) -> list[Array] | None:
        shapes_orig = inv_data
        if arrays_warped is None:
            return None

        flags = cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
        if interpolation_nearest:
            flags += cv2.INTER_NEAREST

        # TODO this does per iteration almost the same as _warp_arrays()
        #      make DRY
        arrays_inv = []
        for arr_warped, shape_orig in zip(arrays_warped, shapes_orig, strict=True):
            if 0 in arr_warped.shape:
                arrays_inv.append(arr_warped)
                continue

            input_dtype = arr_warped.dtype
            if input_dtype.kind == "b":
                arr_warped = arr_warped.astype(np.uint8) * 255
            elif input_dtype == iadt._FLOAT16_DTYPE:
                arr_warped = arr_warped.astype(np.float32)

            height, width = shape_orig[0:2]

            # remap limitation, see docs for warpPolar()
            assert arr_warped.shape[0] <= 32767 and arr_warped.shape[1] <= 32767, (
                "WithPolarWarping._warp_arrays() can currently only "
                "handle arrays with axis sizes below 32767, but got "
                f"shape {arr_warped.shape}. This is an OpenCV limitation."
            )

            dest_size = (width, height)
            center_xy = (width / 2, height / 2)
            max_radius = np.sqrt((height / 2.0) ** 2.0 + (width / 2.0) ** 2.0)

            if arr_warped.ndim == 3 and arr_warped.shape[-1] > 512:
                arr_inv = np.stack(
                    [
                        cv2.warpPolar(
                            _normalize_cv2_input_arr_(arr_warped[..., c_idx]),
                            dest_size,
                            center_xy,
                            max_radius,
                            flags,
                        )
                        for c_idx in np.arange(arr_warped.shape[-1])
                    ],
                    axis=-1,
                )
            else:
                arr_inv = cv2.warpPolar(
                    _normalize_cv2_input_arr_(arr_warped), dest_size, center_xy, max_radius, flags
                )
                if arr_inv.ndim == 2 and arr_warped.ndim == 3:
                    arr_inv = arr_inv[:, :, np.newaxis]

            if input_dtype.kind == "b":
                arr_inv = arr_inv > 128
            elif input_dtype == iadt._FLOAT16_DTYPE:
                arr_inv = arr_inv.astype(np.float16)

            arrays_inv.append(arr_inv)
        return arrays_inv

    @legacy(version="0.4.0")
    @classmethod
    def _warp_maps_(
        cls,
        maps: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage] | None,
        arr_attr_name: str,
        interpolation_nearest: bool,
    ) -> tuple[list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage] | None, object]:
        if maps is None:
            return None, None

        skipped = [False] * len(maps)
        arrays = []
        shapes_imgs_orig = []
        for i, map_i in enumerate(maps):
            if 0 in map_i.shape:
                skipped[i] = True
                arrays.append(np.zeros((0, 0), dtype=np.int32))
                shapes_imgs_orig.append(map_i.shape)
            else:
                arrays.append(getattr(map_i, arr_attr_name))
                shapes_imgs_orig.append(map_i.shape)

        arrays_warped, warparr_inv_data = cls._warp_arrays(arrays, interpolation_nearest)
        shapes_imgs_warped = cls._warp_shape_tuples(shapes_imgs_orig)

        for i, map_i in enumerate(maps):
            if not skipped[i]:
                map_i.shape = shapes_imgs_warped[i]
                setattr(map_i, arr_attr_name, arrays_warped[i])

        return maps, (shapes_imgs_orig, warparr_inv_data, skipped)

    @legacy(version="0.4.0")
    @classmethod
    def _invert_warp_maps_(
        cls,
        maps_warped: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage] | None,
        arr_attr_name: str,
        interpolation_nearest: bool,
        invert_data: object,
    ) -> list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage] | None:
        if maps_warped is None:
            return None

        shapes_imgs_orig, warparr_inv_data, skipped = invert_data

        arrays_warped = []
        for i, map_warped in enumerate(maps_warped):
            if skipped[i]:
                arrays_warped.append(np.zeros((0, 0), dtype=np.int32))
            else:
                arrays_warped.append(getattr(map_warped, arr_attr_name))

        arrays_inv = cls._invert_warp_arrays(arrays_warped, interpolation_nearest, warparr_inv_data)

        for i, map_i in enumerate(maps_warped):
            if not skipped[i]:
                map_i.shape = shapes_imgs_orig[i]
                setattr(map_i, arr_attr_name, arrays_inv[i])

        return maps_warped

    @legacy(version="0.4.0")
    @classmethod
    def _warp_coords(
        cls, coords: list[Coords] | None, image_shapes: list[Shape]
    ) -> tuple[list[Coords] | None, list[Shape] | None]:
        if coords is None:
            return None, None

        image_shapes_warped = cls._warp_shape_tuples(image_shapes)

        flags = cv2.WARP_POLAR_LINEAR

        coords_warped = []
        for coords_i, shape, shape_warped in zip(
            coords, image_shapes, image_shapes_warped, strict=True
        ):
            if 0 in shape:
                coords_warped.append(coords_i)
                continue

            height, width = shape[0:2]
            dest_size = (shape_warped[1], shape_warped[0])
            center_xy = (width / 2, height / 2)
            max_radius = np.sqrt((height / 2.0) ** 2.0 + (width / 2.0) ** 2.0)

            coords_i_warped = cls.warpPolarCoords(coords_i, dest_size, center_xy, max_radius, flags)

            coords_warped.append(coords_i_warped)
        return coords_warped, image_shapes

    @legacy(version="0.4.0")
    @classmethod
    def _invert_warp_coords(
        cls,
        coords_warped: list[Coords] | None,
        image_shapes_after_aug: list[Shape],
        inv_data: object,
    ) -> list[Coords] | None:
        image_shapes_orig = inv_data
        if coords_warped is None:
            return None

        flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
        coords_inv = []
        gen = enumerate(zip(coords_warped, image_shapes_orig, strict=True))
        for i, (coords_i_warped, shape_orig) in gen:
            if 0 in shape_orig:
                coords_inv.append(coords_i_warped)
                continue

            shape_warped = image_shapes_after_aug[i]
            height, width = shape_orig[0:2]
            dest_size = (shape_warped[1], shape_warped[0])
            center_xy = (width / 2, height / 2)
            max_radius = np.sqrt((height / 2.0) ** 2.0 + (width / 2.0) ** 2.0)

            coords_i_inv = cls.warpPolarCoords(
                coords_i_warped, dest_size, center_xy, max_radius, flags
            )

            coords_inv.append(coords_i_inv)
        return coords_inv

    @legacy(version="0.4.0")
    @classmethod
    def _warp_cbaois_(cls, cbaois: list[object] | None) -> tuple[list[object] | None, object]:
        if cbaois is None:
            return None, None

        coords = [cast(Any, cbaoi).to_xy_array() for cbaoi in cbaois]
        image_shapes = [cast(Any, cbaoi).shape for cbaoi in cbaois]
        image_shapes_warped = cls._warp_shape_tuples(image_shapes)

        coords_warped, inv_data = cls._warp_coords(coords, image_shapes)
        for i, (cbaoi, coords_i_warped) in enumerate(zip(cbaois, coords_warped, strict=True)):
            cbaoi = cbaoi.fill_from_xy_array_(coords_i_warped)
            cbaoi.shape = image_shapes_warped[i]
            cbaois[i] = cbaoi

        return cbaois, inv_data

    @legacy(version="0.4.0")
    @classmethod
    def _invert_warp_cbaois_(
        cls, cbaois_warped: list[object] | None, image_shapes_orig: object
    ) -> list[object] | None:
        if cbaois_warped is None:
            return None

        coords = [cast(ia.KeypointsOnImage, cbaoi).to_xy_array() for cbaoi in cbaois_warped]
        image_shapes_after_aug = [cast(ia.KeypointsOnImage, cbaoi).shape for cbaoi in cbaois_warped]

        coords_warped = cls._invert_warp_coords(coords, image_shapes_after_aug, image_shapes_orig)

        cbaois = cbaois_warped
        for i, (cbaoi, coords_i_warped) in enumerate(zip(cbaois, coords_warped, strict=True)):
            cbaoi = cbaoi.fill_from_xy_array_(coords_i_warped)
            cbaoi.shape = image_shapes_orig[i]
            cbaois[i] = cbaoi

        return cbaois

    @legacy(version="0.4.0")
    @classmethod
    def _warp_shape_tuples(cls, shapes: Sequence[Shape]) -> list[Shape]:
        pi = np.pi
        result = []
        for shape in shapes:
            if 0 in shape:
                result.append(shape)
                continue

            height, width = shape[0:2]
            max_radius = np.sqrt((height / 2.0) ** 2.0 + (width / 2.0) ** 2.0)
            # np.round() is here a replacement for cvRound(). It is not fully
            # clear whether the two functions behave exactly identical in all
            # situations.
            # See
            # https://github.com/opencv/opencv/blob/master/
            # modules/core/include/opencv2/core/fast_math.hpp
            # for OpenCV's implementation.
            width = int(np.round(max_radius))
            height = int(np.round(max_radius * pi))
            result.append(tuple([height, width] + list(shape[2:])))
        return result

    @legacy(version="0.4.0")
    @classmethod
    def warpPolarCoords(
        cls,
        src: Coords,
        dsize: Shape2D,
        center: tuple[float, float],
        maxRadius: float,
        flags: int,
    ) -> Coords:
        # See
        # https://docs.opencv.org/3.4.8/da/d54/group__imgproc__transform.html
        # for the equations
        # or also
        # https://github.com/opencv/opencv/blob/master/modules/imgproc/src/
        # imgwarp.cpp
        #
        assert dsize[0] > 0
        assert dsize[1] > 0

        dsize_width = dsize[0]
        dsize_height = dsize[1]

        center_x = center[0]
        center_y = center[1]

        if np.logical_and(flags, cv2.WARP_INVERSE_MAP):
            rho = src[:, 0]
            phi = src[:, 1]
            Kangle = dsize_height / (2 * np.pi)
            angleRad = phi / Kangle
            if np.bitwise_and(flags, cv2.WARP_POLAR_LOG):
                Klog = dsize_width / np.log(maxRadius)
                magnitude = np.exp(rho / Klog)
            else:
                Klin = dsize_width / maxRadius
                magnitude = rho / Klin
            x = center_x + magnitude * np.cos(angleRad)
            y = center_y + magnitude * np.sin(angleRad)

            x = x[:, np.newaxis]
            y = y[:, np.newaxis]

            return np.concatenate([x, y], axis=1)
        else:
            x = src[:, 0]
            y = src[:, 1]

            Kangle = dsize_height / (2 * np.pi)
            Klin = dsize_width / maxRadius

            I_x, I_y = (x - center_x, y - center_y)
            magnitude_I, angle_I = cv2.cartToPolar(I_x, I_y)
            phi = Kangle * angle_I
            # TODO add semilog support here
            rho = Klin * magnitude_I

            return np.concatenate([rho, phi], axis=1)

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return []

    @legacy(version="0.4.0")
    def get_children_lists(self) -> list[list[meta.Augmenter]]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_children_lists`."""
        return cast(list[list[meta.Augmenter]], [self.children])

    @legacy(version="0.4.0")
    def _to_deterministic(self) -> WithPolarWarping:
        aug = self.copy()
        aug.children = aug.children.to_deterministic()
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    @legacy(version="0.4.0")
    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, children={self.children}, "
            f"deterministic={self.deterministic})"
        )


__all__ = ["WithPolarWarping"]
