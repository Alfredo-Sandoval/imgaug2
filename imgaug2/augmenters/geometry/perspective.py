"""Perspective transform augmenter."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage import transform as tf

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmentables.polys import _ConcavePolygonRecoverer
from imgaug2.augmenters._typing import Array, Images, ParamInput, RNGInput
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _normalize_cv2_input_arr_

from .. import meta

# Local type aliases (more specific than _typing versions for this module)
Shape: TypeAlias = tuple[int, ...]
Shape2D: TypeAlias = tuple[int, int]
Matrix: TypeAlias = NDArray[np.floating[Any]]
Coords: TypeAlias = NDArray[np.floating[Any]]
class _PerspectiveTransformSamplingResult:
    def __init__(
        self,
        matrices: list[Matrix],
        max_heights: list[int],
        max_widths: list[int],
        cvals: Array,
        modes: Array,
    ) -> None:
        self.matrices = matrices
        self.max_heights = max_heights
        self.max_widths = max_widths
        self.cvals = cvals
        self.modes = modes


# TODO add arg for image interpolation
class PerspectiveTransform(meta.Augmenter):
    """Apply random four point perspective transformations to images.

    Each of the four points is placed on the image using a random distance from
    its respective corner. The distance is sampled from a normal distribution.

    Supported Dtypes:
        - **Fully Supported**: `uint8`, `float32`, `float64`, `bool` (mapped to float32).
        - **Limited Support**: `uint16` (tested), `int8` (mapped to int16), `int16`, `float16` (mapped to float32).
        - **Not Supported**: `uint32`, `uint64`, `int32`, `int64`, `float128` (OpenCV limitations).

    Parameters:
        scale: Distortion amplitude (normal distribution std dev).
            - number, tuple (a, b), list, or StochasticParameter.
            - Recommended: 0.0 to 0.1.
        keep_size: If True, resize back to original size.
            - If False, output size varies.
        cval: Fill value.
            - number, tuple (min, max), list, or StochasticParameter.
            - Default: 0.
        mode: Boundary mode.
            - "replicate", "constant" (or corresponding cv2 constants).
        fit_output: If True, adjust plane to capture full image.
        polygon_recoverer: Class to repair invalid polygons.
        seed: See `Augmenter`.
        name: See `Augmenter`.

    Example:
        >>> import imgaug2.augmenters as iaa
        >>> # Perspective transform with scale 0.01 to 0.15
        >>> aug = iaa.PerspectiveTransform(scale=(0.01, 0.15))

        >>> # Don't resize back (output images will be smaller/varied)
        >>> aug = iaa.PerspectiveTransform(scale=(0.01, 0.15), keep_size=False)
    """

    _BORDER_MODE_STR_TO_INT = {"replicate": cv2.BORDER_REPLICATE, "constant": cv2.BORDER_CONSTANT}

    def __init__(
        self,
        scale: ParamInput = (0.0, 0.06),
        cval: ParamInput | Literal["ALL"] = 0,
        mode: int | str | list[int | str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        keep_size: bool = True,
        fit_output: bool = False,
        polygon_recoverer: Literal["auto"] | None | _ConcavePolygonRecoverer = "auto",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.scale = iap.handle_continuous_param(
            scale, "scale", value_range=(0, None), tuple_to_uniform=True, list_to_choice=True
        )
        self.jitter = iap.Normal(loc=0, scale=self.scale)

        # setting these to 1x1 caused problems for large scales and polygon
        # augmentation
        # TODO there is now a recoverer for polygons - are these minima still
        #      needed/sensible?
        self.min_width = 2
        self.min_height = 2

        self.cval = iap.handle_cval_arg(cval)
        self.mode = self._handle_mode_arg(mode)
        self.keep_size = keep_size
        self.fit_output = fit_output

        self.polygon_recoverer = polygon_recoverer
        if polygon_recoverer == "auto":
            self.polygon_recoverer = _ConcavePolygonRecoverer()

        # Special order, mode and cval parameters for heatmaps and
        # segmentation maps. These may either be None or a fixed value.
        # Stochastic parameters are currently *not* supported.
        # If set to None, the same values as for images will be used.
        # That is really not recommended for the cval parameter.
        self._order_heatmaps = cv2.INTER_LINEAR
        self._order_segmentation_maps = cv2.INTER_NEAREST
        self._mode_heatmaps = cv2.BORDER_CONSTANT
        self._mode_segmentation_maps = cv2.BORDER_CONSTANT
        self._cval_heatmaps = 0
        self._cval_segmentation_maps = 0

    # TODO unify this somehow with the global _handle_mode_arg() that is
    #      currently used for Affine and PiecewiseAffine
    @classmethod
    @iap._prefetchable_str
    def _handle_mode_arg(
        cls, mode: int | str | list[int | str] | iap.StochasticParameter | Literal["ALL"]
    ) -> iap.StochasticParameter:
        available_modes = [cv2.BORDER_REPLICATE, cv2.BORDER_CONSTANT]
        available_modes_str = ["replicate", "constant"]
        if mode == ia.ALL:
            return iap.Choice(available_modes)
        if ia.is_single_integer(mode):
            assert mode in available_modes, f"Expected mode to be in {available_modes}, got {mode}."
            return iap.Deterministic(mode)
        if ia.is_string(mode):
            assert mode in available_modes_str, (
                f"Expected mode to be in {str(available_modes_str)}, got {mode}."
            )
            return iap.Deterministic(mode)
        if isinstance(mode, list):
            valid_types = all([ia.is_single_integer(val) or ia.is_string(val) for val in mode])
            assert valid_types, (
                "Expected mode list to only contain integers/strings, got "
                f"types {', '.join([str(type(val)) for val in mode])}."
            )
            valid_modes = all([val in available_modes + available_modes_str for val in mode])
            assert valid_modes, (
                f"Expected all mode values to be in {str(available_modes + available_modes_str)}, got {str(mode)}."
            )
            return iap.Choice(mode)
        if isinstance(mode, iap.StochasticParameter):
            return mode
        raise Exception(
            "Expected mode to be imgaug2.ALL, an int, a string, a list "
            f"of int/strings or StochasticParameter, got {type(mode)}."
        )

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        # Advance once, because below we always use random_state.copy() and
        # hence the sampling calls actually don't change random_state's state.
        # Without this, every call of the augmenter would produce the same
        # results.
        random_state.advance_()

        samples_images = self._draw_samples(batch.get_rowwise_shapes(), random_state.copy())

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images, samples_images)

        if batch.heatmaps is not None:
            samples = self._draw_samples(
                [augmentable.arr_0to1.shape for augmentable in batch.heatmaps], random_state.copy()
            )

            batch.heatmaps = self._augment_maps_by_samples(
                batch.heatmaps,
                "arr_0to1",
                samples,
                samples_images,
                self._cval_heatmaps,
                self._mode_heatmaps,
                self._order_heatmaps,
            )

        if batch.segmentation_maps is not None:
            samples = self._draw_samples(
                [augmentable.arr.shape for augmentable in batch.segmentation_maps],
                random_state.copy(),
            )

            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps,
                "arr",
                samples,
                samples_images,
                self._cval_segmentation_maps,
                self._mode_segmentation_maps,
                self._order_segmentation_maps,
            )

        # large scale values cause invalid polygons (unclear why that happens),
        # hence the recoverer
        if batch.polygons is not None:
            func = functools.partial(
                self._augment_keypoints_by_samples, samples_images=samples_images
            )
            batch.polygons = self._apply_to_polygons_as_keypoints(
                batch.polygons, func, recoverer=self.polygon_recoverer
            )

        for augm_name in ["keypoints", "bounding_boxes", "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(
                    self._augment_keypoints_by_samples, samples_images=samples_images
                )
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    @legacy(version="0.4.0")
    def _augment_images_by_samples(
        self, images: Images, samples: _PerspectiveTransformSamplingResult
    ) -> Images:
        from imgaug2.mlx._core import is_mlx_array

        if not any(is_mlx_array(img) for img in images):
            iadt.gate_dtypes_strs(
                images,
                allowed="bool uint8 uint16 int8 int16 float16 float32 float64",
                disallowed="uint32 uint64 int32 int64 float128",
                augmenter=self,
            )

        result = images
        if not self.keep_size:
            result = list(result)

        gen = enumerate(
            zip(
                images,
                samples.matrices,
                samples.max_heights,
                samples.max_widths,
                samples.cvals,
                samples.modes,
                strict=True,
            )
        )

        for i, (image, matrix, max_height, max_width, cval, mode) in gen:
            if is_mlx_array(image):
                import imgaug2.mlx as mlx

                # Map cv2 border modes to the string-based MLX padding modes.
                if int(mode) == cv2.BORDER_CONSTANT:
                    mode_mlx = "constant"
                elif int(mode) == cv2.BORDER_REPLICATE:
                    mode_mlx = "edge"
                elif int(mode) == cv2.BORDER_REFLECT:
                    mode_mlx = "symmetric"
                elif int(mode) == cv2.BORDER_REFLECT_101:
                    mode_mlx = "reflect"
                elif int(mode) == cv2.BORDER_WRAP:
                    mode_mlx = "wrap"
                else:
                    mode_mlx = "constant"

                if image.size == 0:
                    warped = image
                else:
                    # Keep this aligned with the cv2 path: warp to the expanded
                    # output shape and optionally resize back.
                    warped = mlx.perspective_transform(
                        image,
                        matrix,
                        output_shape=(int(max_height), int(max_width)),
                        order=1,
                        cval=cval,
                        mode=mode_mlx,
                    )
                    if self.keep_size:
                        h, w = int(image.shape[0]), int(image.shape[1])
                        warped = mlx.geometry.resize(warped, (h, w), order=1)

                result[i] = warped
                continue

            input_dtype = image.dtype
            if input_dtype == iadt._INT8_DTYPE:
                image = image.astype(np.int16)
            elif input_dtype in {iadt._BOOL_DTYPE, iadt._FLOAT16_DTYPE}:
                image = image.astype(np.float32)

            # cv2.warpPerspective only supports <=4 channels and errors
            # on axes with size zero
            nb_channels = image.shape[2]
            has_zero_sized_axis = image.size == 0
            if has_zero_sized_axis:
                warped = image
            elif nb_channels <= 4:
                warped = cv2.warpPerspective(
                    _normalize_cv2_input_arr_(image),
                    matrix,
                    (max_width, max_height),
                    borderValue=cval,
                    borderMode=mode,
                )
                if warped.ndim == 2 and images[i].ndim == 3:
                    warped = np.expand_dims(warped, 2)
            else:
                # warp each channel on its own
                # note that cv2 removes the channel axis in case of (H,W,1)
                # inputs
                warped = [
                    cv2.warpPerspective(
                        _normalize_cv2_input_arr_(image[..., c]),
                        matrix,
                        (max_width, max_height),
                        borderValue=cval[min(c, len(cval) - 1)],
                        borderMode=mode,
                        flags=cv2.INTER_LINEAR,
                    )
                    for c in range(nb_channels)
                ]
                warped = np.stack(warped, axis=-1)

            if self.keep_size and not has_zero_sized_axis:
                h, w = image.shape[0:2]
                warped = ia.imresize_single_image(warped, (h, w))

            if input_dtype.kind == "b":
                warped = warped > 0.5
            elif warped.dtype != input_dtype:
                warped = iadt.restore_dtypes_(warped, input_dtype)

            result[i] = warped

        return result

    @legacy(version="0.4.0")
    def _augment_maps_by_samples(
        self,
        augmentables: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage],
        arr_attr_name: str,
        samples: _PerspectiveTransformSamplingResult,
        samples_images: _PerspectiveTransformSamplingResult,
        cval: float | int | Array | None,
        mode: int | None,
        flags: int,
    ) -> list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage]:
        result = augmentables

        # estimate max_heights/max_widths for the underlying images
        # this is only necessary if keep_size is False as then the underlying
        # image sizes change and we need to update them here
        # TODO this was re-used from before _augment_batch_() -- reoptimize
        if self.keep_size:
            max_heights_imgs = samples.max_heights
            max_widths_imgs = samples.max_widths
        else:
            max_heights_imgs = samples_images.max_heights
            max_widths_imgs = samples_images.max_widths

        gen = enumerate(
            zip(
                augmentables, samples.matrices, samples.max_heights, samples.max_widths, strict=True
            )
        )

        for i, (augmentable_i, matrix, max_height, max_width) in gen:
            arr = getattr(augmentable_i, arr_attr_name)

            mode_i = mode
            if mode is None:
                mode_i = samples.modes[i]

            cval_i = cval
            if cval is None:
                cval_i = samples.cvals[i]

            nb_channels = arr.shape[2]
            image_has_zero_sized_axis = 0 in augmentable_i.shape
            map_has_zero_sized_axis = arr.size == 0

            if not image_has_zero_sized_axis:
                if not map_has_zero_sized_axis:
                    warped = [
                        cv2.warpPerspective(
                            _normalize_cv2_input_arr_(arr[..., c]),
                            matrix,
                            (max_width, max_height),
                            borderValue=cval_i,
                            borderMode=mode_i,
                            flags=flags,
                        )
                        for c in range(nb_channels)
                    ]
                    warped = np.stack(warped, axis=-1)

                    setattr(augmentable_i, arr_attr_name, warped)

                if self.keep_size:
                    h, w = arr.shape[0:2]
                    augmentable_i = augmentable_i.resize((h, w))
                else:
                    new_shape = (max_heights_imgs[i], max_widths_imgs[i]) + augmentable_i.shape[2:]
                    augmentable_i.shape = new_shape

                result[i] = augmentable_i

        return result

    @legacy(version="0.4.0")
    def _augment_keypoints_by_samples(
        self, kpsois: list[ia.KeypointsOnImage], samples_images: _PerspectiveTransformSamplingResult
    ) -> list[ia.KeypointsOnImage]:
        result = kpsois

        gen = enumerate(
            zip(
                kpsois,
                samples_images.matrices,
                samples_images.max_heights,
                samples_images.max_widths,
                strict=True,
            )
        )

        for i, (kpsoi, matrix, max_height, max_width) in gen:
            image_has_zero_sized_axis = 0 in kpsoi.shape

            if not image_has_zero_sized_axis:
                shape_orig = kpsoi.shape
                shape_new = (max_height, max_width) + kpsoi.shape[2:]
                kpsoi.shape = shape_new
                if not kpsoi.empty:
                    kps_arr = kpsoi.to_xy_array()
                    warped = cv2.perspectiveTransform(np.array([kps_arr], dtype=np.float32), matrix)
                    warped = warped[0]
                    for kp, coords in zip(kpsoi.keypoints, warped, strict=True):
                        kp.x = coords[0]
                        kp.y = coords[1]
                if self.keep_size:
                    kpsoi = kpsoi.on_(shape_orig)
                result[i] = kpsoi

        return result

    @legacy(version="0.4.0")
    def _draw_samples(
        self, shapes: Sequence[Shape], random_state: iarandom.RNG
    ) -> _PerspectiveTransformSamplingResult:
        matrices = []
        max_heights = []
        max_widths = []
        nb_images = len(shapes)
        rngs = random_state.duplicate(3)

        cval_samples = self.cval.draw_samples((nb_images, 3), random_state=rngs[0])
        mode_samples = self.mode.draw_samples((nb_images,), random_state=rngs[1])
        jitter = self.jitter.draw_samples((nb_images, 4, 2), random_state=rngs[2])

        # cv2 perspectiveTransform doesn't accept numpy arrays as cval
        cval_samples_cv2 = cval_samples.tolist()

        # if border modes are represented by strings, convert them to cv2
        # border mode integers
        if mode_samples.dtype.kind not in ["i", "u"]:
            for mode, mapped_mode in self._BORDER_MODE_STR_TO_INT.items():
                mode_samples[mode_samples == mode] = mapped_mode

        # modify jitter to the four corner point coordinates
        # some x/y values have to be modified from `jitter` to `1-jtter`
        # for that
        # TODO remove the abs() here. it currently only allows to "zoom-in",
        #      not to "zoom-out"
        points = np.mod(np.abs(jitter), 1)

        # top left -- no changes needed, just use jitter
        # top right
        points[:, 1, 0] = 1.0 - points[:, 1, 0]  # w = 1.0 - jitter
        # bottom right
        points[:, 2, 0] = 1.0 - points[:, 2, 0]  # w = 1.0 - jitter
        points[:, 2, 1] = 1.0 - points[:, 2, 1]  # h = 1.0 - jitter
        # bottom left
        points[:, 3, 1] = 1.0 - points[:, 3, 1]  # h = 1.0 - jitter

        for shape, points_i in zip(shapes, points, strict=True):
            h, w = shape[0:2]

            points_i[:, 0] *= w
            points_i[:, 1] *= h

            # Obtain a consistent order of the points and unpack them
            # individually.
            # Warning: don't just do (tl, tr, br, bl) = _order_points(...)
            # here, because the reordered points_i is used further below.
            points_i = self._order_points(points_i)
            (tl, tr, br, bl) = points_i

            # compute the width of the new image, which will be the
            # maximum distance between bottom-right and bottom-left
            # x-coordiates or the top-right and top-left x-coordinates
            min_width = None
            max_width = None
            while min_width is None or min_width < self.min_width:
                width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                max_width = int(max(width_top, width_bottom))
                min_width = int(min(width_top, width_bottom))
                if min_width < self.min_width:
                    step_size = (self.min_width - min_width) / 2
                    tl[0] -= step_size
                    tr[0] += step_size
                    bl[0] -= step_size
                    br[0] += step_size

            # compute the height of the new image, which will be the
            # maximum distance between the top-right and bottom-right
            # y-coordinates or the top-left and bottom-left y-coordinates
            min_height = None
            max_height = None
            while min_height is None or min_height < self.min_height:
                height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                max_height = int(max(height_right, height_left))
                min_height = int(min(height_right, height_left))
                if min_height < self.min_height:
                    step_size = (self.min_height - min_height) / 2
                    tl[1] -= step_size
                    tr[1] -= step_size
                    bl[1] += step_size
                    br[1] += step_size

            # now that we have the dimensions of the new image, construct
            # the set of destination points to obtain a "birds eye view",
            # (i.e. top-down view) of the image, again specifying points
            # in the top-left, top-right, bottom-right, and bottom-left
            # order
            # do not use width-1 or height-1 here, as for e.g. width=3, height=2
            # the bottom right coordinate is at (3.0, 2.0) and not (2.0, 1.0)
            dst = np.array(
                [[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]], dtype=np.float32
            )

            # compute the perspective transform matrix and then apply it
            m = cv2.getPerspectiveTransform(points_i, dst)

            if self.fit_output:
                m, max_width, max_height = self._expand_transform(m, (h, w))

            matrices.append(m)
            max_heights.append(max_height)
            max_widths.append(max_width)

        mode_samples = mode_samples.astype(np.int32)
        return _PerspectiveTransformSamplingResult(
            matrices, max_heights, max_widths, cval_samples_cv2, mode_samples
        )

    @classmethod
    def _order_points(cls, pts: Coords) -> Coords:
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        pts_ordered = np.zeros((4, 2), dtype=np.float32)

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        pointwise_sum = pts.sum(axis=1)
        pts_ordered[0] = pts[np.argmin(pointwise_sum)]
        pts_ordered[2] = pts[np.argmax(pointwise_sum)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        pts_ordered[1] = pts[np.argmin(diff)]
        pts_ordered[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return pts_ordered

    @legacy(version="0.4.0")
    @classmethod
    def _expand_transform(cls, matrix: Matrix, shape: Shape2D) -> tuple[Matrix, int, int]:
        height, width = shape
        # do not use width-1 or height-1 here, as for e.g. width=3, height=2
        # the bottom right coordinate is at (3.0, 2.0) and not (2.0, 1.0)
        rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        dst = cv2.perspectiveTransform(np.array([rect]), matrix)[0]

        # get min x, y over transformed 4 points
        # then modify target points by subtracting these minima
        # => shift to (0, 0)
        dst -= dst.min(axis=0, keepdims=True)
        dst = np.around(dst, decimals=0)

        matrix_expanded = cv2.getPerspectiveTransform(rect, dst)
        max_width, max_height = dst.max(axis=0)
        return cast(Matrix, matrix_expanded), int(max_width), int(max_height)

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.jitter, self.keep_size, self.cval, self.mode, self.fit_output]


__all__ = ["PerspectiveTransform"]

