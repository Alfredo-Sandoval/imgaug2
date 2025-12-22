"""Elastic transformation augmenter."""

from __future__ import annotations

import functools
import itertools
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias, overload

import cv2
import numpy as np
import scipy.ndimage as ndimage

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmentables.kps import compute_geometric_median
from imgaug2.augmentables.polys import _ConcavePolygonRecoverer
from imgaug2.augmenters._typing import Array, Images, ParamInput, RNGInput
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _normalize_cv2_input_arr_

from .. import meta

# Local type aliases (more specific than _typing versions for this module)
Shape: TypeAlias = tuple[int, ...]

if TYPE_CHECKING:
    import mlx.core as _mx

    MlxArray: TypeAlias = _mx.array
else:
    MlxArray: TypeAlias = object

ArrayOrMlx: TypeAlias = Array | MlxArray
# TODO add independent sigmas for x/y
# TODO add independent alphas for x/y
# TODO add backend arg
class ElasticTransformation(meta.Augmenter):
    """Transform images using local pixel displacement fields (elastic distortion).

    Pixels are moved around based on displacement fields controlled by `alpha`
    (strength) and `sigma` (smoothness).

    - **Water-like effect**: High `alpha` (e.g., 50), moderate `sigma` (e.g., 5).
    - **Noisy/Pixelated**: Low `sigma` (e.g., 1.0).

    Inspired by Simard, Steinkraus and Platt (2003).

    Supported Dtypes:
        - **Fully Supported**: `uint8`, `float32`, `float64`, `bool`.
        - **Limited Support**: `uint16`, `int8`, `int16`, `int32`, `float16` (handling varies by order/backend).
        - **Not Supported**: `float128`.
        - **Known Issues**: `uint64`/`int64` with order=0.

    Parameters:
        alpha: Distortion strength (displacement magnitude).
            - number, tuple (min, max), list, or StochasticParameter.
            - Recommended: ~10 * sigma.
        sigma: Smoothness of distortion (gaussian kernel std dev).
            - number, tuple (min, max), list, or StochasticParameter.
            - Low values (<1.5) = noisy. High values = smooth/wavy.
        order: Interpolation order (0-5). 0=fastest/nearest.
            - number, tuple, list, or StochasticParameter.
        cval: Fill value for "constant" mode.
        mode: Boundary mode ("constant", "nearest", "reflect", "wrap").
        polygon_recoverer: Class to repair invalid polygons.
        seed: See `Augmenter`.
        name: See `Augmenter`.

    Example:
        >>> import imgaug2.augmenters as iaa
        >>> # Water-like distortion
        >>> aug = iaa.ElasticTransformation(alpha=50, sigma=5)

        >>> # Random strength
        >>> aug = iaa.ElasticTransformation(alpha=(0.0, 70.0), sigma=5.0)
    """

    NB_NEIGHBOURING_KEYPOINTS = 3
    NEIGHBOURING_KEYPOINTS_DISTANCE = 1.0
    KEYPOINT_AUG_ALPHA_THRESH = 0.05
    # even at high alphas we don't augment keypoints if the sigma is too low,
    # because then the pixel movements are mostly gaussian noise anyways
    KEYPOINT_AUG_SIGMA_THRESH = 1.0

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

    def __init__(
        self,
        alpha: ParamInput = (1.0, 40.0),
        sigma: ParamInput = (4.0, 8.0),
        order: int | tuple[int, int] | list[int] | iap.StochasticParameter | Literal["ALL"] = 0,
        cval: ParamInput | Literal["ALL"] = 0,
        mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        polygon_recoverer: Literal["auto"] | None | _ConcavePolygonRecoverer = "auto",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.alpha = iap.handle_continuous_param(
            alpha, "alpha", value_range=(0, None), tuple_to_uniform=True, list_to_choice=True
        )
        self.sigma = iap.handle_continuous_param(
            sigma, "sigma", value_range=(0, None), tuple_to_uniform=True, list_to_choice=True
        )

        self.order = self._handle_order_arg(order)
        self.cval = iap.handle_cval_arg(cval)
        self.mode = self._handle_mode_arg(mode)

        self.polygon_recoverer = polygon_recoverer
        if polygon_recoverer == "auto":
            self.polygon_recoverer = _ConcavePolygonRecoverer()

        # Special order, mode and cval parameters for heatmaps and
        # segmentation maps. These may either be None or a fixed value.
        # Stochastic parameters are currently *not* supported.
        # If set to None, the same values as for images will be used.
        # That is really not recommended for the cval parameter.
        #
        self._order_heatmaps = 3
        self._order_segmentation_maps = 0
        self._mode_heatmaps = "constant"
        self._mode_segmentation_maps = "constant"
        self._cval_heatmaps = 0.0
        self._cval_segmentation_maps = 0

        self._last_meshgrid = None

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
        cls, mode: str | list[str] | iap.StochasticParameter | Literal["ALL"]
    ) -> iap.StochasticParameter:
        if mode == ia.ALL:
            return iap.Choice(["constant", "nearest", "reflect", "wrap"])
        if ia.is_string(mode):
            return iap.Deterministic(mode)
        if ia.is_iterable(mode):
            assert all([ia.is_string(val) for val in mode]), (
                "Expected mode list to only contain strings, got "
                f"types {', '.join([str(type(val)) for val in mode])}."
            )
            return iap.Choice(mode)
        if isinstance(mode, iap.StochasticParameter):
            return mode
        raise Exception(
            "Expected mode to be imgaug2.ALL, a string, a list of strings "
            f"or StochasticParameter, got {type(mode)}."
        )

    def _draw_samples(
        self, nb_images: int, random_state: iarandom.RNG
    ) -> _ElasticTransformationSamplingResult:
        rss = random_state.duplicate(nb_images + 5)
        alphas = self.alpha.draw_samples((nb_images,), random_state=rss[-5])
        sigmas = self.sigma.draw_samples((nb_images,), random_state=rss[-4])
        orders = self.order.draw_samples((nb_images,), random_state=rss[-3])
        cvals = self.cval.draw_samples((nb_images,), random_state=rss[-2])
        modes = self.mode.draw_samples((nb_images,), random_state=rss[-1])
        return _ElasticTransformationSamplingResult(rss[0], alphas, sigmas, orders, cvals, modes)

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        if batch.images is not None:
            from imgaug2.mlx._core import is_mlx_array

            if not any(is_mlx_array(img) for img in batch.images):
                iadt.gate_dtypes_strs(
                    batch.images,
                    allowed="bool uint8 uint16 uint32 uint64 int8 int16 int32 "
                    "int64 float16 float32 float64",
                    disallowed="float128",
                    augmenter=self,
                )

        shapes = batch.get_rowwise_shapes()
        samples = self._draw_samples(len(shapes), random_state)
        smgen = _ElasticTfShiftMapGenerator()
        shift_maps = smgen.generate(shapes, samples.alphas, samples.sigmas, samples.random_state)

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
                batch.keypoints[i] = self._augment_kpsoi_by_samples(
                    batch.keypoints[i], i, samples, dx, dy
                )
            if batch.bounding_boxes is not None:
                batch.bounding_boxes[i] = self._augment_bbsoi_by_samples(
                    batch.bounding_boxes[i], i, samples, dx, dy
                )
            if batch.polygons is not None:
                batch.polygons[i] = self._augment_psoi_by_samples(
                    batch.polygons[i], i, samples, dx, dy
                )
            if batch.line_strings is not None:
                batch.line_strings[i] = self._augment_lsoi_by_samples(
                    batch.line_strings[i], i, samples, dx, dy
                )

        return batch

    @legacy(version="0.4.0")
    def _augment_image_by_samples(
        self,
        image: Array,
        row_idx: int,
        samples: _ElasticTransformationSamplingResult,
        dx: Array,
        dy: Array,
    ) -> Array:
        from imgaug2.mlx._core import is_mlx_array

        if is_mlx_array(image):
            return self._map_coordinates(
                image,
                dx,
                dy,
                order=int(samples.orders[row_idx]),
                cval=float(samples.cvals[row_idx]),
                mode=str(samples.modes[row_idx]),
            )

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

    @legacy(version="0.4.0")
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

        # note that we do not have to check for zero-sized axes here,
        # because _generate_shift_maps(), _map_coordinates(), .resize()
        # and np.clip() are all known to handle arrays with zero-sized axes

        arr = getattr(augmentable, arr_attr_name)

        if arr.shape[0:2] == augmentable.shape[0:2]:
            arr_warped = self._map_coordinates(arr, dx, dy, order=order, cval=cval, mode=mode)

            # interpolation in map_coordinates() can cause some values to
            # be below/above 1.0, so we clip here
            if order >= 3 and isinstance(augmentable, ia.HeatmapsOnImage):
                arr_warped = np.clip(arr_warped, 0.0, 1.0, out=arr_warped)

            setattr(augmentable, arr_attr_name, arr_warped)
        else:
            # Heatmaps/Segmaps do not have the same size as augmented
            # images. This may result in indices of moved pixels being
            # different. To prevent this, we use the same image size as
            # for the base images, but that requires resizing the heatmaps
            # temporarily to the image sizes.
            height_orig, width_orig = arr.shape[0:2]
            augmentable = augmentable.resize(augmentable.shape[0:2])
            arr = getattr(augmentable, arr_attr_name)

            # TODO will it produce similar results to first downscale the
            #      shift maps and then remap? That would make the remap
            #      step take less operations and would also mean that the
            #      heatmaps wouldnt have to be scaled up anymore. It would
            #      also simplify the code as this branch could be merged
            #      with the one above.
            arr_warped = self._map_coordinates(arr, dx, dy, order=order, cval=cval, mode=mode)

            # interpolation in map_coordinates() can cause some values to
            # be below/above 1.0, so we clip here
            if order >= 3 and isinstance(augmentable, ia.HeatmapsOnImage):
                arr_warped = np.clip(arr_warped, 0.0, 1.0, out=arr_warped)

            setattr(augmentable, arr_attr_name, arr_warped)

            augmentable = augmentable.resize((height_orig, width_orig))

        return augmentable

    @legacy(version="0.4.0")
    def _augment_kpsoi_by_samples(
        self,
        kpsoi: ia.KeypointsOnImage,
        row_idx: int,
        samples: _ElasticTransformationSamplingResult,
        dx: Array,
        dy: Array,
    ) -> ia.KeypointsOnImage:
        height, width = kpsoi.shape[0:2]
        alpha = samples.alphas[row_idx]
        sigma = samples.sigmas[row_idx]

        # Note: this block must be placed after _generate_shift_maps() to
        # keep samples aligned
        # Note: we should stop for zero-sized axes early here, event though
        # there is a height/width check for each keypoint, because the
        # channel number can also be zero
        image_has_zero_sized_axes = 0 in kpsoi.shape
        params_below_thresh = (
            alpha <= self.KEYPOINT_AUG_ALPHA_THRESH or sigma <= self.KEYPOINT_AUG_SIGMA_THRESH
        )

        if kpsoi.empty or image_has_zero_sized_axes or params_below_thresh:
            # ElasticTransformation does not change the shape, hence we can
            # skip the below steps
            return kpsoi

        for kp in kpsoi.keypoints:
            within_image_plane = 0 <= kp.x < width and 0 <= kp.y < height
            if within_image_plane:
                kp_neighborhood = kp.generate_similar_points_manhattan(
                    self.NB_NEIGHBOURING_KEYPOINTS,
                    self.NEIGHBOURING_KEYPOINTS_DISTANCE,
                    return_array=True,
                )

                # We can clip here, because we made sure above that the
                # keypoint is inside the image plane. Keypoints at the
                # bottom row or right columns might be rounded outside
                # the image plane, which we prevent here. We reduce
                # neighbours to only those within the image plane as only
                # for such points we know where to move them.
                xx = np.round(kp_neighborhood[:, 0]).astype(np.int32)
                yy = np.round(kp_neighborhood[:, 1]).astype(np.int32)
                inside_image_mask = np.logical_and(
                    np.logical_and(0 <= xx, xx < width), np.logical_and(0 <= yy, yy < height)
                )
                xx = xx[inside_image_mask]
                yy = yy[inside_image_mask]

                xxyy = np.concatenate([xx[:, np.newaxis], yy[:, np.newaxis]], axis=1)

                # dx/dy define a reverse mapping (output -> input) via
                # x_shifted = x - dx and y_shifted = y - dy. We therefore need
                # to estimate the forward mapping for keypoints (input -> output),
                # i.e. solve p_out - d(p_out) = p_in. A few fixed-point iterations
                # is usually sufficient and improves alignment vs. a single lookup
                # at the input location.
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
                # uncomment to use average instead of median
                # med = np.average(xxyy_aug, 0)
                kp.x = med[0]
                kp.y = med[1]

        return kpsoi

    @legacy(version="0.4.0")
    def _augment_psoi_by_samples(
        self,
        psoi: ia.PolygonsOnImage,
        row_idx: int,
        samples: _ElasticTransformationSamplingResult,
        dx: Array,
        dy: Array,
    ) -> ia.PolygonsOnImage:
        func = functools.partial(
            self._augment_kpsoi_by_samples, row_idx=row_idx, samples=samples, dx=dx, dy=dy
        )
        return self._apply_to_polygons_as_keypoints(psoi, func, recoverer=self.polygon_recoverer)

    @legacy(version="0.4.0")
    def _augment_lsoi_by_samples(
        self,
        lsoi: ia.LineStringsOnImage,
        row_idx: int,
        samples: _ElasticTransformationSamplingResult,
        dx: Array,
        dy: Array,
    ) -> ia.LineStringsOnImage:
        func = functools.partial(
            self._augment_kpsoi_by_samples, row_idx=row_idx, samples=samples, dx=dx, dy=dy
        )
        return self._apply_to_cbaois_as_keypoints(lsoi, func)

    @legacy(version="0.4.0")
    def _augment_bbsoi_by_samples(
        self,
        bbsoi: ia.BoundingBoxesOnImage,
        row_idx: int,
        samples: _ElasticTransformationSamplingResult,
        dx: Array,
        dy: Array,
    ) -> ia.BoundingBoxesOnImage:
        func = functools.partial(
            self._augment_kpsoi_by_samples, row_idx=row_idx, samples=samples, dx=dx, dy=dy
        )
        return self._apply_to_cbaois_as_keypoints(bbsoi, func)

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.alpha, self.sigma, self.order, self.cval, self.mode]

    @overload
    def _map_coordinates(
        self,
        image: Array,
        dx: Array,
        dy: Array,
        order: int = 1,
        cval: float | int = 0,
        mode: str = "constant",
    ) -> Array: ...

    @overload
    def _map_coordinates(
        self,
        image: MlxArray,
        dx: Array,
        dy: Array,
        order: int = 1,
        cval: float | int = 0,
        mode: str = "constant",
    ) -> MlxArray: ...

    def _map_coordinates(
        self,
        image: ArrayOrMlx,
        dx: Array,
        dy: Array,
        order: int = 1,
        cval: float | int = 0,
        mode: str = "constant",
    ) -> ArrayOrMlx:
        """Remap pixels in an image according to x/y shift maps.

        **Supported dtypes**:

        if (backend="scipy" and order=0):

            * ``uint8``: yes
            * ``uint16``: yes
            * ``uint32``: yes
            * ``uint64``: no (1)
            * ``int8``: yes
            * ``int16``: yes
            * ``int32``: yes
            * ``int64``: no (2)
            * ``float16``: yes
            * ``float32``: yes
            * ``float64``: yes
            * ``float128``: no (3)
            * ``bool``: yes

            - (1) produces array filled with only 0
            - (2) produces array filled with <min_value> when testing
                  with <max_value>
            - (3) causes: 'data type no supported'

        if (backend="scipy" and order>0):

            * ``uint8``: yes (1)
            * ``uint16``: yes (1)
            * ``uint32``: yes (1)
            * ``uint64``: yes (1)
            * ``int8``: yes (1)
            * ``int16``: yes (1)
            * ``int32``: yes (1)
            * ``int64``: yes (1)
            * ``float16``: yes (1)
            * ``float32``: yes (1)
            * ``float64``: yes (1)
            * ``float128``: no (2)
            * ``bool``: yes

            - (1) rather loose test, to avoid having to re-compute the
                  interpolation
            - (2) causes: 'data type no supported'

        if (backend="cv2" and order=0):

            * ``uint8``: yes
            * ``uint16``: yes
            * ``uint32``: no (1)
            * ``uint64``: no (2)
            * ``int8``: yes
            * ``int16``: yes
            * ``int32``: yes
            * ``int64``: no (2)
            * ``float16``: yes
            * ``float32``: yes
            * ``float64``: yes
            * ``float128``: no (3)
            * ``bool``: no (4)

            - (1) causes: src data type = 6 is not supported
            - (2) silently converts to int32
            - (3) causes: src data type = 13 is not supported
            - (4) causes: src data type = 0 is not supported

        if (backend="cv2" and order=1):

            * ``uint8``: yes
            * ``uint16``: yes
            * ``uint32``: no (1)
            * ``uint64``: no (2)
            * ``int8``: no (2)
            * ``int16``: no (2)
            * ``int32``: no (2)
            * ``int64``: no (2)
            * ``float16``: yes
            * ``float32``: yes
            * ``float64``: yes
            * ``float128``: no (3)
            * ``bool``: no (4)

            - (1) causes: src data type = 6 is not supported
            - (2) causes: OpenCV(3.4.5) (...)/imgwarp.cpp:1805:
                  error: (-215:Assertion failed) ifunc != 0 in function
                  'remap'
            - (3) causes: src data type = 13 is not supported
            - (4) causes: src data type = 0 is not supported

        if (backend="cv2" and order>=2):

            * ``uint8``: yes
            * ``uint16``: yes
            * ``uint32``: no (1)
            * ``uint64``: no (2)
            * ``int8``: no (2)
            * ``int16``: yes
            * ``int32``: no (2)
            * ``int64``: no (2)
            * ``float16``: yes
            * ``float32``: yes
            * ``float64``: yes
            * ``float128``: no (3)
            * ``bool``: no (4)

            - (1) causes: src data type = 6 is not supported
            - (2) causes: OpenCV(3.4.5) (...)/imgwarp.cpp:1805:
                  error: (-215:Assertion failed) ifunc != 0 in function
                  'remap'
            - (3) causes: src data type = 13 is not supported
            - (4) causes: src data type = 0 is not supported

        """
        from imgaug2.mlx._core import is_mlx_array

        if is_mlx_array(image):
            import imgaug2.mlx as mlx

            order_i = int(order)
            if order_i not in (0, 1):
                raise NotImplementedError(
                    "MLX elastic warp only supports interpolation orders 0 (nearest) and 1 (bilinear). "
                    f"Got order={order_i}."
                )

            img = image
            squeeze_channel_axis = False
            if img.ndim == 2:
                img = img[:, :, None]
                squeeze_channel_axis = True

            h, w = int(img.shape[0]), int(img.shape[1])
            if h == 0 or w == 0:
                return image

            mx = mlx.mx
            dx_mx = mx.array(dx.astype(np.float32, copy=False))
            dy_mx = mx.array(dy.astype(np.float32, copy=False))

            yy, xx = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
            xx = xx.astype(mx.float32)
            yy = yy.astype(mx.float32)

            x_shifted = xx - dx_mx
            y_shifted = yy - dy_mx

            if w > 1:
                gx = (x_shifted / float(w - 1)) * 2.0 - 1.0
            else:
                gx = mx.zeros_like(x_shifted)
            if h > 1:
                gy = (y_shifted / float(h - 1)) * 2.0 - 1.0
            else:
                gy = mx.zeros_like(y_shifted)

            grid = mx.stack([gx, gy], axis=-1)

            mode_str = str(mode).lower()
            padding_mode = {
                "constant": "zeros",
                "nearest": "border",
                "reflect": "reflection",
                "wrap": "wrap",
            }.get(mode_str, "zeros")

            warped = mlx.grid_sample(
                img,
                grid,
                mode="nearest" if order_i == 0 else "bilinear",
                padding_mode=padding_mode,
                cval=float(cval),
            )

            if squeeze_channel_axis:
                warped = warped[:, :, 0]
            return warped

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
            # cv2.convertMaps expects either 32F maps (or 16S variants). When dx/dy
            # are float64, x_shifted/y_shifted become float64 too, which can
            # trigger an OpenCV assertion failure.
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
            # remap only supports up to 4 channels
            if nb_channels <= 4:
                # dst does not seem to improve performance here
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


class _ElasticTransformationSamplingResult:
    def __init__(
        self,
        random_state: iarandom.RNG,
        alphas: Array | None,
        sigmas: Array | None,
        orders: Array,
        cvals: Array,
        modes: Array,
    ) -> None:
        self.random_state = random_state
        self.alphas = alphas
        self.sigmas = sigmas
        self.orders = orders
        self.cvals = cvals
        self.modes = modes


@legacy(version="0.5.0")
class _ElasticTfShiftMapGenerator:
    """Class to generate shift/displacement maps for ElasticTransformation.

    This class re-uses samples for multiple examples. This minimizes the amount
    of sampling that has to be done.


    """

    # Not really necessary to have this as a class, considering it has no
    # attributes. But it makes things easier to read.
    @legacy(version="0.5.0")
    def __init__(self) -> None:
        pass

    @legacy(version="0.5.0")
    def generate(
        self, shapes: Sequence[Shape], alphas: Array, sigmas: Array, random_state: iarandom.RNG
    ) -> Iterator[tuple[Array, Array]]:
        # We will sample shift maps from [0.0, 1.0] and then shift by -0.5 to
        # [-0.5, 0.5]. To bring these maps to [-1.0, 1.0], we have to multiply
        # somewhere by 2. It is fastes to multiply the (fewer) alphas, which
        # we will have to multiply the shift maps with anyways.
        alphas *= 2

        # Configuration for each chunk.
        # switch dx / dy, flip dx lr, flip dx ud, flip dy lr, flip dy ud
        switch = [False, True]
        fliplr_dx = [False, True]
        flipud_dx = [False, True]
        fliplr_dy = [False, True]
        flipud_dy = [False, True]
        configs = list(itertools.product(switch, fliplr_dx, flipud_dx, fliplr_dy, flipud_dy))

        areas = [shape[0] * shape[1] for shape in shapes]
        nb_chunks = len(configs)
        gen = zip(
            self._split_chunks(shapes, nb_chunks),
            self._split_chunks(areas, nb_chunks),
            self._split_chunks(alphas, nb_chunks),
            self._split_chunks(sigmas, nb_chunks),
            strict=True,
        )
        # "_c" denotes a chunk here
        for shapes_c, areas_c, alphas_c, sigmas_c in gen:
            area_max = max(areas_c)

            dxdy = random_state.random((2, area_max))
            dxdy -= 0.5
            dx, dy = dxdy[0, :], dxdy[1, :]

            # dx_lr = flip dx left-right, dx_ud = flip dx up-down
            # dy_lr, dy_ud analogous
            for i, (switch_i, dx_lr, dx_ud, dy_lr, dy_ud) in enumerate(configs):
                if i >= len(shapes_c):
                    break

                dx_i, dy_i = (dx, dy) if not switch_i else (dy, dx)
                shape_i = shapes_c[i][0:2]
                area_i = shape_i[0] * shape_i[1]

                if area_i == 0:
                    yield (np.zeros(shape_i, dtype=np.float32), np.zeros(shape_i, dtype=np.float32))
                else:
                    dx_i = dx_i[0:area_i].reshape(shape_i)
                    dy_i = dy_i[0:area_i].reshape(shape_i)
                    dx_i, dy_i = self._flip(dx_i, dy_i, (dx_lr, dx_ud, dy_lr, dy_ud))
                    dx_i, dy_i = self._mul_alpha(dx_i, dy_i, alphas_c[i])
                    yield self._smoothen_(dx_i, dy_i, sigmas_c[i])

    @legacy(version="0.5.0")
    @classmethod
    def _flip(
        cls, dx: Array, dy: Array, flips: tuple[bool, bool, bool, bool]
    ) -> tuple[Array, Array]:
        # no measureable benefit from using cv2 here
        if flips[0]:
            dx = np.fliplr(dx)
        if flips[1]:
            dx = np.flipud(dx)
        if flips[2]:
            dy = np.fliplr(dy)
        if flips[3]:
            dy = np.flipud(dy)
        return dx, dy

    @legacy(version="0.5.0")
    @classmethod
    def _mul_alpha(cls, dx: Array, dy: Array, alpha: float) -> tuple[Array, Array]:
        # performance drops for cv2.multiply here
        dx = dx * alpha
        dy = dy * alpha
        return dx, dy

    @legacy(version="0.5.0")
    @classmethod
    def _smoothen_(cls, dx: Array, dy: Array, sigma: float) -> tuple[Array, Array]:
        if sigma < 1.5:
            from .. import blur as blur_lib

            dx = blur_lib.blur_gaussian_(dx, sigma)
            dy = blur_lib.blur_gaussian_(dy, sigma)
        else:
            ksize = int(round(2 * sigma))
            dx = cv2.blur(dx, (ksize, ksize), dst=dx)
            dy = cv2.blur(dy, (ksize, ksize), dst=dy)
        return dx, dy

    @legacy(version="0.5.0")
    @classmethod
    def _split_chunks(
        cls, iterable: Sequence[object], chunk_size: int
    ) -> Iterator[Sequence[object]]:
        for i in range(0, len(iterable), chunk_size):
            yield iterable[i : i + chunk_size]


__all__ = ["ElasticTransformation"]
