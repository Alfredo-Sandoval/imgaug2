"""Piecewise affine augmenter."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Literal, TypeAlias, cast

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

from .. import meta
from ._utils import _handle_mode_arg, _handle_order_arg

# Local type aliases (more specific than _typing versions for this module)
Shape: TypeAlias = tuple[int, ...]
class _PiecewiseAffineSamplingResult:
    def __init__(
        self,
        nb_rows: Array,
        nb_cols: Array,
        jitter: list[Array],
        order: Array,
        cval: Array,
        mode: Sequence[str],
    ) -> None:
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.order = order
        self.jitter = jitter
        self.cval = cval
        self.mode = mode

    def get_clipped_cval(self, idx: int, dtype: np.dtype[np.generic]) -> float | int:
        min_value, _, max_value = iadt.get_value_range_of_dtype(dtype)
        cval = self.cval[idx]
        cval = max(min(cval, max_value), min_value)
        return cast(float | int, cval)


class PiecewiseAffine(meta.Augmenter):
    """Apply affine transformations that differ between local neighbourhoods.

    This augmenter places a regular grid of points on an image and randomly
    moves the neighbourhood of these point around via affine transformations.

    Wrapper around scikit-image's `PiecewiseAffine`.
    Note: Very slow. Use `ElasticTransformation` if speed is critical.

    Supported Dtypes:
        - **Fully Supported**: `uint8`, `uint16`, `int8`, `int16`, `float16`, `float32`, `float64`, `bool` (order=0).
        - **Limited Support**: `uint32`, `int32` (possible inaccuracies due to float64 conversion).
        - **Not Supported**: `uint64`, `int64`, `float128`.

    Parameters:
        scale: Distortion amplitude (normal distribution sigma).
            - number, tuple (a, b), list, or StochasticParameter.
            - Recommended: 0.01 to 0.05.
        nb_rows: Number of rows of points in the grid (min 2). default 4.
        nb_cols: Number of columns.
        order: Interpolation order. See `Affine`.
        cval: Fill value. See `Affine`.
        mode: Boundary mode. See `Affine`.
        absolute_scale: If True, `scale` is absolute pixels, else relative.
        polygon_recoverer: Class to repair invalid polygons.
        seed: See `Augmenter`.
        name: See `Augmenter`.

    Example:
        >>> import imgaug2.augmenters as iaa
        >>> # Random moves by 1 to 5 percent magnitude
        >>> aug = iaa.PiecewiseAffine(scale=(0.01, 0.05))

        >>> # Use denser grid (8x8)
        >>> aug = iaa.PiecewiseAffine(scale=(0.01, 0.05), nb_rows=8, nb_cols=8)
    """

    def __init__(
        self,
        scale: ParamInput = (0.0, 0.04),
        nb_rows: int | tuple[int, int] | list[int] | iap.StochasticParameter = (2, 4),
        nb_cols: int | tuple[int, int] | list[int] | iap.StochasticParameter = (2, 4),
        order: int | list[int] | iap.StochasticParameter | Literal["ALL"] = 1,
        cval: ParamInput | Literal["ALL"] = 0,
        mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        absolute_scale: bool = False,
        polygon_recoverer: Literal["auto"] | None | _ConcavePolygonRecoverer = None,
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
        self.nb_rows = iap.handle_discrete_param(
            nb_rows,
            "nb_rows",
            value_range=(2, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self.nb_cols = iap.handle_discrete_param(
            nb_cols,
            "nb_cols",
            value_range=(2, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )

        self.order = _handle_order_arg(order, backend="skimage")
        self.cval = iap.handle_cval_arg(cval)
        self.mode = _handle_mode_arg(mode)

        self.absolute_scale = absolute_scale
        self.polygon_recoverer = polygon_recoverer
        if polygon_recoverer == "auto":
            self.polygon_recoverer = _ConcavePolygonRecoverer()

        # Special order, mode and cval parameters for heatmaps and
        # segmentation maps. These may either be None or a fixed value.
        # Stochastic parameters are currently *not* supported.
        # If set to None, the same values as for images will be used.
        # That is really not recommended for the cval parameter.
        self._order_heatmaps = 3
        self._order_segmentation_maps = 0
        self._mode_heatmaps = "constant"
        self._mode_segmentation_maps = "constant"
        self._cval_heatmaps = 0
        self._cval_segmentation_maps = 0

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        samples = self._draw_samples(batch.nb_rows, random_state)

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images, samples)

        if batch.heatmaps is not None:
            batch.heatmaps = self._augment_maps_by_samples(
                batch.heatmaps,
                "arr_0to1",
                samples,
                self._cval_heatmaps,
                self._mode_heatmaps,
                self._order_heatmaps,
            )

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps,
                "arr",
                samples,
                self._cval_segmentation_maps,
                self._mode_segmentation_maps,
                self._order_segmentation_maps,
            )

        if batch.polygons is not None:
            func = functools.partial(self._augment_keypoints_by_samples, samples=samples)
            batch.polygons = self._apply_to_polygons_as_keypoints(
                batch.polygons, func, recoverer=self.polygon_recoverer
            )

        for augm_name in ["keypoints", "bounding_boxes", "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(self._augment_keypoints_by_samples, samples=samples)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    @legacy(version="0.4.0")
    def _augment_images_by_samples(
        self, images: Images, samples: _PiecewiseAffineSamplingResult
    ) -> Images:
        from imgaug2.mlx._core import is_mlx_array

        if not any(is_mlx_array(img) for img in images):
            iadt.gate_dtypes_strs(
                images,
                allowed="bool uint8 uint16 uint32 int8 int16 int32 float16 float32 float64",
                disallowed="uint64 int64 float128",
                augmenter=self,
            )

        result = images

        for i, image in enumerate(images):
            transformer = self._get_transformer(
                image.shape, image.shape, samples.nb_rows[i], samples.nb_cols[i], samples.jitter[i]
            )

            if transformer is not None:
                if is_mlx_array(image):
                    import imgaug2.mlx as mlx

                    order_i = int(samples.order[i])
                    if order_i not in (0, 1):
                        raise NotImplementedError(
                            "MLX piecewise affine only supports interpolation orders 0 (nearest) and 1 (bilinear). "
                            f"Got order={order_i}."
                        )

                    img = image
                    squeeze_channel_axis = False
                    if img.ndim == 2:
                        img = img[:, :, None]
                        squeeze_channel_axis = True

                    h, w = int(img.shape[0]), int(img.shape[1])
                    if h == 0 or w == 0:
                        result[i] = image
                        continue

                    yy, xx = np.meshgrid(
                        np.arange(h, dtype=np.float32),
                        np.arange(w, dtype=np.float32),
                        indexing="ij",
                    )
                    coords_out = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (H*W, 2) as (x,y)
                    coords_in = transformer.inverse(coords_out).astype(np.float32, copy=False)
                    x_in = coords_in[:, 0].reshape(h, w)
                    y_in = coords_in[:, 1].reshape(h, w)

                    mx = mlx.mx
                    gx = (mx.array(x_in) / float(w - 1)) * 2.0 - 1.0 if w > 1 else mx.zeros((h, w))
                    gy = (mx.array(y_in) / float(h - 1)) * 2.0 - 1.0 if h > 1 else mx.zeros((h, w))
                    grid = mx.stack([gx, gy], axis=-1)

                    mode_str = str(samples.mode[i]).lower()
                    padding_mode = {
                        "constant": "zeros",
                        "edge": "border",
                        "reflect": "reflection",
                        "symmetric": "symmetric",
                        "wrap": "wrap",
                    }.get(mode_str, "zeros")

                    warped = mlx.grid_sample(
                        img,
                        grid,
                        mode="nearest" if order_i == 0 else "bilinear",
                        padding_mode=padding_mode,
                        cval=float(samples.cval[i]),
                    )
                    if squeeze_channel_axis:
                        warped = warped[:, :, 0]

                    result[i] = warped
                    continue

                input_dtype = image.dtype
                if image.dtype.kind == "b":
                    image = image.astype(np.float64)
                # scipy.ndimage (used internally by skimage) does not support float16.
                # Convert to float32 for the warp and convert back afterwards.
                elif input_dtype == iadt._FLOAT16_DTYPE:
                    image = image.astype(np.float32)

                image_warped = tf.warp(
                    image,
                    transformer,
                    order=samples.order[i],
                    mode=samples.mode[i],
                    cval=samples.get_clipped_cval(i, input_dtype),
                    preserve_range=True,
                    output_shape=images[i].shape,
                )

                if input_dtype.kind == "b":
                    image_warped = image_warped > 0.5
                else:
                    # warp seems to change everything to float64, including
                    # uint8, making this necessary
                    image_warped = iadt.restore_dtypes_(image_warped, input_dtype)

                result[i] = image_warped

        return result

    @legacy(version="0.4.0")
    def _augment_maps_by_samples(
        self,
        augmentables: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage],
        arr_attr_name: str,
        samples: _PiecewiseAffineSamplingResult,
        cval: float | int | None,
        mode: str | None,
        order: int,
    ) -> list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage]:
        result = augmentables

        for i, augmentable in enumerate(augmentables):
            arr = getattr(augmentable, arr_attr_name)
            input_dtype = arr.dtype
            if input_dtype == iadt._FLOAT16_DTYPE:
                arr = arr.astype(np.float32)

            transformer = self._get_transformer(
                arr.shape,
                augmentable.shape,
                samples.nb_rows[i],
                samples.nb_cols[i],
                samples.jitter[i],
            )

            if transformer is not None:
                arr_warped = tf.warp(
                    arr,
                    transformer,
                    order=order if order is not None else samples.order[i],
                    mode=mode if mode is not None else samples.mode[i],
                    cval=cval if cval is not None else samples.cval[i],
                    preserve_range=True,
                    output_shape=arr.shape,
                )

                # skimage converts to float64
                arr_warped = arr_warped.astype(input_dtype)

                # order=3 matches cubic interpolation and can cause values
                # to go outside of the range [0.0, 1.0] not clear whether
                # 4+ also do that
                # We don't modify segmaps here, because they don't have a
                # clear value range of [0, 1]
                if order >= 3 and isinstance(augmentable, ia.HeatmapsOnImage):
                    arr_warped = np.clip(arr_warped, 0.0, 1.0, out=arr_warped)

                setattr(augmentable, arr_attr_name, arr_warped)

        return result

    @legacy(version="0.4.0")
    def _augment_keypoints_by_samples(
        self, kpsois: list[ia.KeypointsOnImage], samples: _PiecewiseAffineSamplingResult
    ) -> list[ia.KeypointsOnImage]:
        result = []

        for i, kpsoi in enumerate(kpsois):
            h, w = kpsoi.shape[0:2]
            transformer = self._get_transformer(
                kpsoi.shape, kpsoi.shape, samples.nb_rows[i], samples.nb_cols[i], samples.jitter[i]
            )

            if transformer is None or len(kpsoi.keypoints) == 0:
                result.append(kpsoi)
            else:
                # Augmentation routine that only modifies keypoint coordinates
                # This is efficient (coordinates of all other locations in the
                # image are ignored). The code below should usually work, but
                # for some reason augmented coordinates are often wildly off
                # for large scale parameters (lots of jitter/distortion).
                # The reason for that is unknown.
                """
                coords = keypoints_on_images[i].get_coords_array()
                coords_aug = transformer.inverse(coords)
                result.append(
                    ia.KeypointsOnImage.from_coords_array(
                        coords_aug,
                        shape=keypoints_on_images[i].shape
                    )
                )
                """

                # TODO this could be done a little bit more efficient by
                #      removing first all KPs that are outside of the image
                #      plane so that no corresponding distance map has to
                #      be augmented
                # Image based augmentation routine. Draws the keypoints on
                # the image plane using distance maps (more accurate than
                # just marking the points),  then augments these images, then
                # searches for the new (visual) location of the keypoints.
                # Much slower than directly augmenting the coordinates, but
                # here the only method that reliably works.
                dist_maps = kpsoi.to_distance_maps(inverted=True)
                dist_maps_warped = tf.warp(
                    dist_maps,
                    transformer,
                    order=1,
                    preserve_range=True,
                    output_shape=(kpsoi.shape[0], kpsoi.shape[1], len(kpsoi.keypoints)),
                )

                kps_aug = ia.KeypointsOnImage.from_distance_maps(
                    dist_maps_warped,
                    inverted=True,
                    threshold=0.01,
                    if_not_found_coords={"x": -1, "y": -1},
                    nb_channels=(None if len(kpsoi.shape) < 3 else kpsoi.shape[2]),
                )

                for kp, kp_aug in zip(kpsoi.keypoints, kps_aug.keypoints, strict=True):
                    # Keypoints that were outside of the image plane before the
                    # augmentation were replaced with (-1, -1) by default (as
                    # they can't be drawn on the keypoint images).
                    within_image = 0 <= kp.x < w and 0 <= kp.y < h
                    if within_image:
                        kp.x = kp_aug.x
                        kp.y = kp_aug.y

                result.append(kpsoi)

        return result

    def _draw_samples(
        self, nb_images: int, random_state: iarandom.RNG
    ) -> _PiecewiseAffineSamplingResult:
        rss = random_state.duplicate(6)

        nb_rows_samples = self.nb_rows.draw_samples((nb_images,), random_state=rss[-6])
        nb_cols_samples = self.nb_cols.draw_samples((nb_images,), random_state=rss[-5])
        order_samples = self.order.draw_samples((nb_images,), random_state=rss[-4])
        cval_samples = self.cval.draw_samples((nb_images,), random_state=rss[-3])
        mode_samples = self.mode.draw_samples((nb_images,), random_state=rss[-2])

        nb_rows_samples = np.clip(nb_rows_samples, 2, None)
        nb_cols_samples = np.clip(nb_cols_samples, 2, None)
        nb_cells = nb_rows_samples * nb_cols_samples
        jitter = self.jitter.draw_samples((int(np.sum(nb_cells)), 2), random_state=rss[-1])

        jitter_by_image = []
        counter = 0
        for nb_cells_i in nb_cells:
            jitter_img = jitter[counter : counter + nb_cells_i, :]
            jitter_by_image.append(jitter_img)
            counter += nb_cells_i

        return _PiecewiseAffineSamplingResult(
            nb_rows=nb_rows_samples,
            nb_cols=nb_cols_samples,
            jitter=jitter_by_image,
            order=order_samples,
            cval=cval_samples,
            mode=mode_samples,
        )

    def _get_transformer(
        self,
        augmentable_shape: Shape,
        image_shape: Shape,
        nb_rows: int,
        nb_cols: int,
        jitter_img: Array,
    ) -> tf.PiecewiseAffineTransform | None:
        # get coords on y and x axis of points to move around
        # these coordinates are supposed to be at the centers of each cell
        # (otherwise the first coordinate would be at (0, 0) and could hardly
        # be moved around before leaving the image),
        # so we use here (half cell height/width to H/W minus half
        # height/width) instead of (0, H/W)

        # height/width) instead of (0, H/W)

        y = np.linspace(0, augmentable_shape[0], nb_rows)
        x = np.linspace(0, augmentable_shape[1], nb_cols)

        # (H, W) and (H, W) for H=rows, W=cols
        xx_src, yy_src = np.meshgrid(x, y)

        # (1, HW, 2) => (HW, 2) for H=rows, W=cols
        points_src = np.dstack([yy_src.flat, xx_src.flat])[0]

        any_nonzero = np.any(jitter_img > 0)
        if not any_nonzero:
            return None
        else:
            # Without this, jitter gets changed between different augmentables.
            # TODO if left out, only one test failed -- should be more
            jitter_img = np.copy(jitter_img)

            if self.absolute_scale:
                if image_shape[0] > 0:
                    jitter_img[:, 0] = jitter_img[:, 0] / image_shape[0]
                else:
                    jitter_img[:, 0] = 0.0

                if image_shape[1] > 0:
                    jitter_img[:, 1] = jitter_img[:, 1] / image_shape[1]
                else:
                    jitter_img[:, 1] = 0.0

            jitter_img[:, 0] = jitter_img[:, 0] * augmentable_shape[0]
            jitter_img[:, 1] = jitter_img[:, 1] * augmentable_shape[1]

            points_dest = np.copy(points_src)
            points_dest[:, 0] = points_dest[:, 0] + jitter_img[:, 0]
            points_dest[:, 1] = points_dest[:, 1] + jitter_img[:, 1]

            # tf.warp() results in qhull error if the points are identical,
            # which is mainly the case if any axis is 0
            has_low_axis = any([axis <= 1 for axis in augmentable_shape[0:2]])
            has_zero_channels = (
                augmentable_shape is not None
                and len(augmentable_shape) == 3
                and augmentable_shape[-1] == 0
            ) or (image_shape is not None and len(image_shape) == 3 and image_shape[-1] == 0)

            if has_low_axis or has_zero_channels:
                return None
            else:
                matrix = tf.PiecewiseAffineTransform()
                matrix.estimate(points_src[:, ::-1], points_dest[:, ::-1])
                return matrix

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [
            self.scale,
            self.nb_rows,
            self.nb_cols,
            self.order,
            self.cval,
            self.mode,
            self.absolute_scale,
        ]


__all__ = ["PiecewiseAffine"]


