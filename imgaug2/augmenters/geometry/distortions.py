"""Geometric distortion augmenters."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal, TypeAlias

import cv2
import numpy as np
import scipy.ndimage as ndimage

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmentables.kps import compute_geometric_median
from imgaug2.augmenters._typing import Array, Images, ParamInput, RNGInput
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _normalize_cv2_input_arr_

from .. import meta
from .elastic import _ElasticTransformationSamplingResult

# Local type aliases (more specific than _typing versions for this module)
Shape: TypeAlias = tuple[int, ...]
class _GeometricWarpMixin:
    _MAPPING_MODE_SCIPY_CV2 = {
        "constant": cv2.BORDER_CONSTANT,
        "nearest": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "wrap": cv2.BORDER_WRAP,
    }

    _MAPPING_ORDER_SCIPY_CV2 = {
        0: cv2.INTER_NEAREST,
        1: cv2.INTER_LINEAR,
        2: cv2.INTER_CUBIC,
        3: cv2.INTER_CUBIC,
        4: cv2.INTER_CUBIC,
        5: cv2.INTER_CUBIC,
    }

    @classmethod
    @iap._prefetchable
    def _handle_order_arg(
        cls, order: int | tuple[int, int] | list[int] | iap.StochasticParameter | Literal["ALL"]
    ) -> iap.StochasticParameter:
        if order == ia.ALL:
            return iap.Choice([0, 1, 2, 3, 4, 5])
        return iap.handle_discrete_param(
            order,
            "order",
            value_range=(0, 5),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )

    @classmethod
    @iap._prefetchable_str
    def _handle_mode_arg(
        cls, mode: str | Sequence[str] | iap.StochasticParameter | Literal["ALL"]
    ) -> iap.StochasticParameter:
        if mode == ia.ALL:
            return iap.Choice(["constant", "nearest", "reflect", "wrap"])
        if ia.is_string(mode):
            return iap.Deterministic(mode)
        if ia.is_iterable(mode):
            return iap.Choice(mode)
        if isinstance(mode, iap.StochasticParameter):
            return mode
        raise Exception(
            f"Expected mode to be string, list or StochasticParameter, got {type(mode)}."
        )

    def _augment_image_by_samples(
        self,
        image: Array,
        row_idx: int,
        samples: _ElasticTransformationSamplingResult,
        dx: Array,
        dy: Array,
    ) -> Array:
        min_value, _center_value, max_value = iadt.get_value_range_of_dtype(image.dtype)
        cval = max(min(samples.cvals[row_idx], max_value), min_value)

        input_dtype = image.dtype
        if image.dtype == iadt._FLOAT16_DTYPE:
            image = image.astype(np.float32)

        image_aug = self._map_coordinates(
            image, dx, dy, order=samples.orders[row_idx], cval=cval, mode=samples.modes[row_idx]
        )

        if image.dtype != input_dtype:
            image_aug = iadt.restore_dtypes_(image_aug, input_dtype)
        return image_aug

    def _augment_hm_or_sm_by_samples(
        self,
        augmentable: ia.HeatmapsOnImage | ia.SegmentationMapsOnImage,
        row_idx: int,
        samples: _ElasticTransformationSamplingResult,
        dx: Array,
        dy: Array,
        arr_attr_name: str,
        cval: float | int | None,
        mode: str | None,
        order: int | None,
    ) -> ia.HeatmapsOnImage | ia.SegmentationMapsOnImage:
        cval = cval if cval is not None else samples.cvals[row_idx]
        mode = mode if mode is not None else samples.modes[row_idx]
        order = order if order is not None else samples.orders[row_idx]

        arr = getattr(augmentable, arr_attr_name)
        if arr.shape[0:2] == augmentable.shape[0:2]:
            arr_warped = self._map_coordinates(arr, dx, dy, order=order, cval=cval, mode=mode)
            if order >= 3 and isinstance(augmentable, ia.HeatmapsOnImage):
                arr_warped = np.clip(arr_warped, 0.0, 1.0, out=arr_warped)
            setattr(augmentable, arr_attr_name, arr_warped)
        else:
            augmentable = augmentable.resize(augmentable.shape[0:2])
            arr = getattr(augmentable, arr_attr_name)
            arr_warped = self._map_coordinates(arr, dx, dy, order=order, cval=cval, mode=mode)
            if order >= 3 and isinstance(augmentable, ia.HeatmapsOnImage):
                arr_warped = np.clip(arr_warped, 0.0, 1.0, out=arr_warped)
            setattr(augmentable, arr_attr_name, arr_warped)
        return augmentable

    def _map_coordinates(
        self,
        image: Array,
        dx: Array,
        dy: Array,
        order: int = 1,
        cval: float | int = 0,
        mode: str = "constant",
    ) -> Array:
        if image.size == 0:
            return np.copy(image)

        if order == 0 and image.dtype in {iadt._UINT64_DTYPE, iadt._INT64_DTYPE}:
            raise Exception(
                "dtypes uint64 and int64 are only supported in "
                "ElasticTransformation for order=0, got order={order} with "
                f"dtype={image.dtype.name}."
            )

        input_dtype = image.dtype
        if image.dtype.kind == "b":
            image = image.astype(np.float32)
        elif order == 1 and image.dtype in {iadt._INT8_DTYPE, iadt._INT16_DTYPE, iadt._INT32_DTYPE}:
            image = image.astype(np.float64)
        elif order >= 2 and image.dtype == iadt._INT8_DTYPE:
            image = image.astype(np.int16)
        elif order >= 2 and image.dtype == iadt._INT32_DTYPE:
            image = image.astype(np.float64)

        shrt_max = 32767  # maximum of datatype `short`
        backend = "cv2"
        if order == 0:
            bad_dtype_cv2 = image.dtype in iadt._convert_dtype_strs_to_types(
                "uint32 uint64 int64 float128 bool"
            )
        elif order == 1:
            bad_dtype_cv2 = image.dtype in iadt._convert_dtype_strs_to_types(
                "uint32 uint64 int8 int16 int32 int64 float128 bool"
            )
        else:
            bad_dtype_cv2 = image.dtype in iadt._convert_dtype_strs_to_types(
                "uint32 uint64 int8 int32 int64 float128 bool"
            )

        bad_dx_shape_cv2 = dx.shape[0] >= shrt_max or dx.shape[1] >= shrt_max
        bad_dy_shape_cv2 = dy.shape[0] >= shrt_max or dy.shape[1] >= shrt_max
        if bad_dtype_cv2 or bad_dx_shape_cv2 or bad_dy_shape_cv2:
            backend = "scipy"

        assert image.ndim == 3, f"Expected 3-dimensional image, got {image.ndim} dimensions."

        h, w, nb_channels = image.shape
        last = self._last_meshgrid
        if last is not None and last[0].shape == (h, w):
            y, x = self._last_meshgrid
        else:
            y, x = np.meshgrid(
                np.arange(h).astype(np.float32), np.arange(w).astype(np.float32), indexing="ij"
            )
            self._last_meshgrid = (y, x)
        x_shifted = x - dx
        y_shifted = y - dy

        if backend == "cv2":
            x_shifted = np.ascontiguousarray(x_shifted, dtype=np.float32)
            y_shifted = np.ascontiguousarray(y_shifted, dtype=np.float32)

        if backend == "scipy":
            result = np.empty_like(image)

            for c in range(image.shape[2]):
                remapped_flat = ndimage.map_coordinates(
                    image[..., c],
                    (y_shifted.flatten(), x_shifted.flatten()),
                    order=order,
                    cval=cval,
                    mode=mode,
                )
                remapped = remapped_flat.reshape((h, w))
                result[..., c] = remapped
        else:
            if image.dtype.kind == "f":
                cval = float(cval)
            else:
                cval = int(cval)

            border_mode = self._MAPPING_MODE_SCIPY_CV2[mode]
            interpolation = self._MAPPING_ORDER_SCIPY_CV2[order]

            is_nearest_neighbour = interpolation == cv2.INTER_NEAREST
            map1, map2 = cv2.convertMaps(
                x_shifted, y_shifted, cv2.CV_16SC2, nninterpolation=is_nearest_neighbour
            )
            if nb_channels <= 4:
                result = cv2.remap(
                    _normalize_cv2_input_arr_(image),
                    map1,
                    map2,
                    interpolation=interpolation,
                    borderMode=border_mode,
                    borderValue=tuple([cval] * nb_channels),
                )
                if image.ndim == 3 and result.ndim == 2:
                    result = result[..., np.newaxis]
            else:
                current_chan_idx = 0
                result = []
                while current_chan_idx < nb_channels:
                    channels = image[..., current_chan_idx : current_chan_idx + 4]
                    result_c = cv2.remap(
                        _normalize_cv2_input_arr_(channels),
                        map1,
                        map2,
                        interpolation=interpolation,
                        borderMode=border_mode,
                        borderValue=(cval, cval, cval),
                    )
                    if result_c.ndim == 2:
                        result_c = result_c[..., np.newaxis]
                    result.append(result_c)
                    current_chan_idx += 4
                result = np.concatenate(result, axis=2)

        if result.dtype != input_dtype:
            result = iadt.restore_dtypes_(result, input_dtype)

        return result


class _GridDistortionShiftMapGenerator:
    def __init__(self) -> None:
        pass

    def generate(
        self,
        shapes: Sequence[Shape],
        num_steps: Array,
        distort_limits: Array,
        random_state: iarandom.RNG,
    ) -> Iterator[tuple[Array, Array]]:
        # We process chunks to avoid excessive memory usage
        # but here we generate per image mainly because steps/limits might vary

        for i, shape in enumerate(shapes):
            h, w = shape[0:2]
            steps_x = num_steps[i]
            steps_y = num_steps[i]
            limit_x = distort_limits[0][i]
            limit_y = distort_limits[1][i]

            sx = int(w // steps_x)
            sy = int(h // steps_y)

            # Avoid zero division or empty steps
            if sx == 0 or sy == 0:
                yield np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)
                continue

            grid_h = steps_y + 1
            grid_w = steps_x + 1

            # Using random_state for reproducibility
            rss = random_state.duplicate(2)
            map_x = rss[0].uniform(-1.0, 1.0, (grid_h, grid_w)).astype(np.float32)
            map_y = rss[1].uniform(-1.0, 1.0, (grid_h, grid_w)).astype(np.float32)

            step_h = h / steps_y
            step_w = w / steps_x

            # Applying limits
            map_x *= limit_x * step_w * 0.5
            map_y *= limit_y * step_h * 0.5

            # Resize to full image size
            dx = cv2.resize(map_x, (w, h), interpolation=cv2.INTER_CUBIC)
            dy = cv2.resize(map_y, (w, h), interpolation=cv2.INTER_CUBIC)

            yield dx, dy


class GridDistortion(_GeometricWarpMixin, meta.Augmenter):
    """
    Apply grid distortion to images.

    This augmenter divides the image into a grid of cells and randomly moves
    the intersection points of the grid. The movements are interpolated to
    create a smooth distortion field. This is a faster alternative to ``PiecewiseAffine``
    and produces similar non-rigid deformations.

    **Supported dtypes**:

        See :class:`~imgaug2.augmenters.geometric.ElasticTransformation`.

    Parameters
    ----------
    num_steps : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Number of grid cells on each side.

            * If ``int``: Fixed number of steps.
            * If ``tuple`` ``(a, b)``: Randomly picked from ``[a, b]``.
            * If ``list``: Randomly picked from list.
            * If ``StochasticParameter``: Sampled per image.

    distort_limit : float or tuple of float or list of float or imgaug2.parameters.StochasticParameter, optional
        Range of distortion as a fraction of the grid cell size.
        If a single float, the range will be ``(-distort_limit, distort_limit)``.

            * If ``float``: Fixed range.
            * If ``tuple`` ``(a, b)``: Randomly picked from ``[a, b]``.
            * If ``StochasticParameter``: Sampled per image.

    order : int or list of int or imaug.ALL or imgaug2.parameters.StochasticParameter, optional
        Interpolation order to use. See :class:`Affine`.

    cval : number or tuple of number or list of number or imgaug2.ALL or imgaug2.parameters.StochasticParameter, optional
        The constant intensity value used to fill in new pixels. See :class:`Affine`.

    mode : str or list of str or imgaug2.ALL or imgaug2.parameters.StochasticParameter, optional
        Parameter that defines the handling of newly created pixels. See :class:`Affine`.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.
    """

    NB_NEIGHBOURING_KEYPOINTS = 3
    NEIGHBOURING_KEYPOINTS_DISTANCE = 1.0

    def __init__(
        self,
        num_steps: int | tuple[int, int] | list[int] | iap.StochasticParameter = 5,
        distort_limit: ParamInput = (-0.3, 0.3),
        order: int | tuple[int, int] | list[int] | iap.StochasticParameter | Literal["ALL"] = 1,
        cval: ParamInput | Literal["ALL"] = 0,
        mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "reflect",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.num_steps = iap.handle_discrete_param(
            num_steps,
            "num_steps",
            value_range=(1, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self.distort_limit = iap.handle_continuous_param(
            distort_limit,
            "distort_limit",
            value_range=None,
            tuple_to_uniform=True,
            list_to_choice=True,
        )
        self.order = self._handle_order_arg(order)
        self.cval = iap.handle_cval_arg(cval)
        self.mode = self._handle_mode_arg(mode)

        # Reuse heuristics from ElasticTransformation for heatmaps/segmaps
        self._order_heatmaps = 3
        self._order_segmentation_maps = 0
        self._mode_heatmaps = "constant"
        self._mode_segmentation_maps = "constant"
        self._cval_heatmaps = 0.0
        self._cval_segmentation_maps = 0

        self._last_meshgrid = None

        self._last_meshgrid = None

    # Helpers inherited from _GeometricWarpMixin

    def _draw_samples(
        self, nb_images: int, random_state: iarandom.RNG
    ) -> tuple[_ElasticTransformationSamplingResult, Array, Array]:
        rss = random_state.duplicate(nb_images + 4)
        num_steps = self.num_steps.draw_samples((nb_images,), random_state=rss[-4])
        limit_samples = self.distort_limit.draw_samples((2, nb_images), random_state=rss[-3])

        orders = self.order.draw_samples((nb_images,), random_state=rss[-2])
        cvals = self.cval.draw_samples((nb_images,), random_state=rss[-1])
        modes = self.mode.draw_samples((nb_images,), random_state=rss[0])

        return (
            _ElasticTransformationSamplingResult(rss[1], None, None, orders, cvals, modes),
            num_steps,
            limit_samples,
        )

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
                allowed="bool uint8 uint16 uint32 uint64 int8 int16 int32 "
                "int64 float16 float32 float64",
                disallowed="float128",
                augmenter=self,
            )

        shapes = batch.get_rowwise_shapes()
        samples, num_steps_s, limits_s = self._draw_samples(len(shapes), random_state)

        gen = _GridDistortionShiftMapGenerator()
        shift_maps = gen.generate(shapes, num_steps_s, limits_s, samples.random_state)

        for i, (_shape, (dx, dy)) in enumerate(zip(shapes, shift_maps, strict=True)):
            if batch.images is not None:
                batch.images[i] = self._augment_image_by_samples(
                    batch.images[i], i, samples, dx, dy
                )
            if batch.heatmaps is not None:
                batch.heatmaps[i] = self._augment_hm_or_sm_by_samples(
                    batch.heatmaps[i],
                    i,
                    samples,
                    dx,
                    dy,
                    "arr_0to1",
                    self._cval_heatmaps,
                    self._mode_heatmaps,
                    self._order_heatmaps,
                )
            if batch.segmentation_maps is not None:
                batch.segmentation_maps[i] = self._augment_hm_or_sm_by_samples(
                    batch.segmentation_maps[i],
                    i,
                    samples,
                    dx,
                    dy,
                    "arr",
                    self._cval_segmentation_maps,
                    self._mode_segmentation_maps,
                    self._order_segmentation_maps,
                )
            if batch.keypoints is not None:
                batch.keypoints[i] = self._augment_kpsoi_by_shift_map(
                    batch.keypoints[i], dx, dy
                )

        return batch

    def _augment_kpsoi_by_shift_map(
        self, kpsoi: ia.KeypointsOnImage, dx: Array, dy: Array
    ) -> ia.KeypointsOnImage:
        height, width = kpsoi.shape[0:2]

        image_has_zero_sized_axes = 0 in kpsoi.shape
        if kpsoi.empty or image_has_zero_sized_axes:
            return kpsoi

        for kp in kpsoi.keypoints:
            within_image_plane = 0 <= kp.x < width and 0 <= kp.y < height
            if within_image_plane:
                kp_neighborhood = kp.generate_similar_points_manhattan(
                    self.NB_NEIGHBOURING_KEYPOINTS,
                    self.NEIGHBOURING_KEYPOINTS_DISTANCE,
                    return_array=True,
                )

                xx = np.round(kp_neighborhood[:, 0]).astype(np.int32)
                yy = np.round(kp_neighborhood[:, 1]).astype(np.int32)
                inside_image_mask = np.logical_and(
                    np.logical_and(0 <= xx, xx < width), np.logical_and(0 <= yy, yy < height)
                )
                xx = xx[inside_image_mask]
                yy = yy[inside_image_mask]

                xxyy = np.concatenate([xx[:, np.newaxis], yy[:, np.newaxis]], axis=1)

                x_in = xxyy[:, 0].astype(np.float32)
                y_in = xxyy[:, 1].astype(np.float32)
                x_out = x_in
                y_out = y_in

                for _ in range(3):
                    xx_out = np.clip(np.round(x_out).astype(np.int32), 0, width - 1)
                    yy_out = np.clip(np.round(y_out).astype(np.int32), 0, height - 1)
                    x_out = x_in + dx[yy_out, xx_out]
                    y_out = y_in + dy[yy_out, xx_out]

                xxyy_aug = np.stack([x_out, y_out], axis=1)

                med = compute_geometric_median(xxyy_aug)
                kp.x = med[0]
                kp.y = med[1]

        return kpsoi

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.num_steps, self.distort_limit, self.order, self.cval, self.mode]


class OpticalDistortion(_GeometricWarpMixin, meta.Augmenter):
    """
    Apply optical distortion to images (barrel/pincushion).

    **Supported dtypes**:

        See :class:`~imgaug2.augmenters.geometric.ElasticTransformation`.

    Parameters
    ----------
    distort_limit : float or tuple of float or list of float or imgaug2.parameters.StochasticParameter, optional
        Distortion coefficient `k`.
        Positive values lead to pincushion distortion (image shrinks),
        negative values to barrel distortion (image expands).

            * If ``float``: Range ``(-distort_limit, distort_limit)``.
            * If ``tuple``: Range ``(a, b)``.

    shift_limit : float or tuple of float or list of float or imgaug2.parameters.StochasticParameter, optional
        Shift of the optical center.
        Relative to the image size.

            * If ``float``: Range ``(-shift_limit, shift_limit)``.
            * If ``tuple``: Range ``(a, b)``.

    order : int or list of int or imaug.ALL or imgaug2.parameters.StochasticParameter, optional
        Interpolation order.

    cval : number or tuple of number or list of number or imgaug2.ALL or imgaug2.parameters.StochasticParameter, optional
        Constant filling value.

    mode : str or list of str or imgaug2.ALL or imgaug2.parameters.StochasticParameter, optional
        Pixel extension mode.

    seed : None or int, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.
    """

    def __init__(
        self,
        distort_limit: ParamInput = (-0.05, 0.05),
        shift_limit: ParamInput = (-0.05, 0.05),
        order: int | tuple[int, int] | list[int] | iap.StochasticParameter | Literal["ALL"] = 1,
        cval: ParamInput | Literal["ALL"] = 0,
        mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "reflect",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
        k: ParamInput | None = None,
        shift: ParamInput | None = None,
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        if k is not None:
            distort_limit = k
        if shift is not None:
            shift_limit = shift

        self.distort_limit = iap.handle_continuous_param(
            distort_limit,
            "distort_limit",
            value_range=None,
            tuple_to_uniform=True,
            list_to_choice=True,
        )
        self.shift_limit = iap.handle_continuous_param(
            shift_limit, "shift_limit", value_range=None, tuple_to_uniform=True, list_to_choice=True
        )
        self.order = self._handle_order_arg(order)
        self.cval = iap.handle_cval_arg(cval)
        self.mode = self._handle_mode_arg(mode)

        # Reuse heuristics
        self._order_heatmaps = 3
        self._order_segmentation_maps = 0
        self._mode_heatmaps = "constant"
        self._mode_segmentation_maps = "constant"
        self._cval_heatmaps = 0.0
        self._cval_segmentation_maps = 0

        self._last_meshgrid = None

    # Reuse handlers from _GeometricWarpMixin
    def _draw_samples(
        self, nb_images: int, random_state: iarandom.RNG
    ) -> tuple[_ElasticTransformationSamplingResult, Array, Array]:
        rss = random_state.duplicate(nb_images + 4)
        distort_samples = self.distort_limit.draw_samples((nb_images,), random_state=rss[-4])
        shift_samples = self.shift_limit.draw_samples((2, nb_images), random_state=rss[-3])
        orders = self.order.draw_samples((nb_images,), random_state=rss[-2])
        cvals = self.cval.draw_samples((nb_images,), random_state=rss[-1])
        modes = self.mode.draw_samples((nb_images,), random_state=rss[0])

        return (
            _ElasticTransformationSamplingResult(rss[1], None, None, orders, cvals, modes),
            distort_samples,
            shift_samples,
        )

    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        shapes = batch.get_rowwise_shapes()
        samples, dist_s, shift_s = self._draw_samples(len(shapes), random_state)

        for i, shape in enumerate(shapes):
            h, w = shape[0:2]
            k = dist_s[i]
            dx_c = shift_s[0][i] * w
            dy_c = shift_s[1][i] * h

            # Camera matrix
            cx = w * 0.5 + dx_c
            cy = h * 0.5 + dy_c
            fx = w
            fy = h

            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.array([k, 0, 0, 0], dtype=np.float32)

            # Generate maps using initUndistortRectifyMap
            # This computes the undistortion map: map[x_distorted] -> x_corrected
            # If we assume k is distortion of the SOURCE relative to the DEST.
            # Then map[x_dest] points to x_source (distorted).
            # This is exactly what we want if we consider 'k' as the distortion coefficient of the input.

            map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
            )

            # map1 has shape (h, w), contains x coordinates
            # map2 has shape (h, w), contains y coordinates

            # remap requires float maps for linear/cubic.
            # map1, map2 are float32.

            # Convert to dx, dy deviation maps for compatibility with shared methods?
            # Shared methods expect dx, dy where map_x = x + dx.
            # But here we have map_x directly.
            # We can use map1, map2 directly if we adapt the helper, or subtract.
            # Let's subtract to reuse `_augment_image_by_samples` which calls `_map_coordinates`.
            # `_map_coordinates` calls `_remap_cv2`.
            # `_remap_cv2` logic:
            # if dx, dy are provided, it constructs map1 = grid_x + dx.
            # So if we provide dx = map1 - grid_x.

            grid_x, grid_y = np.meshgrid(
                np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
            )
            dx = map1 - grid_x
            dy = map2 - grid_y

            if batch.images is not None:
                batch.images[i] = self._augment_image_by_samples(
                    batch.images[i], i, samples, dx, dy
                )
            if batch.heatmaps is not None:
                batch.heatmaps[i] = self._augment_hm_or_sm_by_samples(
                    batch.heatmaps[i],
                    i,
                    samples,
                    dx,
                    dy,
                    "arr_0to1",
                    self._cval_heatmaps,
                    self._mode_heatmaps,
                    self._order_heatmaps,
                )
            if batch.segmentation_maps is not None:
                batch.segmentation_maps[i] = self._augment_hm_or_sm_by_samples(
                    batch.segmentation_maps[i],
                    i,
                    samples,
                    dx,
                    dy,
                    "arr",
                    self._cval_segmentation_maps,
                    self._mode_segmentation_maps,
                    self._order_segmentation_maps,
                )
            # Keypoints support to be added

        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.distort_limit, self.shift_limit, self.order, self.cval, self.mode]


__all__ = ["GridDistortion", "OpticalDistortion"]

