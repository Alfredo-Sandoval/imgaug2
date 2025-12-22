"""Jigsaw helpers and augmenter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters._typing import Array, RNGInput
from imgaug2.compat.markers import legacy

from .. import meta
from .. import size as size_lib

# Local type aliases (more specific than _typing versions for this module)
Shape: TypeAlias = tuple[int, ...]
Coords: TypeAlias = NDArray[np.floating[Any]]
# TODO allow -1 destinations
def apply_jigsaw(arr: Array, destinations: NDArray[np.integer]) -> Array:
    """Move cells of an image similar to a jigsaw puzzle.

    This function splits the image into `rows x cols` cells and moves each cell
    to the target index specified in `destinations`.

    Supported Dtypes:
        uint8, uint16, uint32, uint64, int8, int16, int32, int64,
        float16, float32, float64, float128, bool

    Parameters:
        arr: Array with at least two dimensions (height, width).
        destinations: 2D array of destination cell IDs in flattened C-order.
            The image height/width must be divisible by the destination grid rows/cols.

    Returns:
        The modified image with shuffled cells.
    """
    nb_rows, nb_cols = destinations.shape[0:2]

    assert arr.ndim >= 2, (
        f"Expected array with at least two dimensions, but got {arr.ndim} with shape {arr.shape}."
    )
    assert (arr.shape[0] % nb_rows) == 0, (
        "Expected image height to by divisible by number of rows, but got "
        f"height {arr.shape[0]} and {nb_rows} rows. Use cropping or padding to modify the image "
        "height or change the number of rows."
    )
    assert (arr.shape[1] % nb_cols) == 0, (
        "Expected image width to by divisible by number of columns, but got "
        f"width {arr.shape[1]} and {nb_cols} columns. Use cropping or padding to modify the image "
        "width or change the number of columns."
    )

    cell_height = arr.shape[0] // nb_rows
    cell_width = arr.shape[1] // nb_cols

    dest_rows, dest_cols = np.unravel_index(destinations.flatten(), (nb_rows, nb_cols))

    # Precompute all source and destination bounds (vectorized)
    source_rows_arr = np.arange(nb_rows).repeat(nb_cols)
    source_cols_arr = np.tile(np.arange(nb_cols), nb_rows)

    source_y1_arr = source_rows_arr * cell_height
    source_x1_arr = source_cols_arr * cell_width
    dest_y1_arr = dest_rows * cell_height
    dest_x1_arr = dest_cols * cell_width

    result = np.zeros_like(arr)
    for i in range(nb_rows * nb_cols):
        sy1, sx1 = source_y1_arr[i], source_x1_arr[i]
        dy1, dx1 = dest_y1_arr[i], dest_x1_arr[i]
        result[dy1 : dy1 + cell_height, dx1 : dx1 + cell_width] = arr[
            sy1 : sy1 + cell_height, sx1 : sx1 + cell_width
        ]

    return result


def apply_jigsaw_to_coords(
    coords: Coords, destinations: NDArray[np.integer], image_shape: Shape
) -> Coords:
    """Move coordinates on an image similar to a jigsaw puzzle.

    Moves coordinates according to the cell shuffling defined by `destinations`.

    Parameters:
        coords: `(N, 2)` array of xy-coordinates.
        destinations: 2D array of destination cell IDs (see `apply_jigsaw`).
        image_shape: The `(height, width, ...)` of the image.

    Returns:
        The moved coordinates.
    """
    nb_rows, nb_cols = destinations.shape[0:2]

    height, width = image_shape[0:2]
    cell_height = height // nb_rows
    cell_width = width // nb_cols

    dest_rows, dest_cols = np.unravel_index(destinations.flatten(), (nb_rows, nb_cols))

    if len(coords) == 0:
        return np.copy(coords)

    x = coords[:, 0]
    y = coords[:, 1]

    # Mask for in-bounds coordinates
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    # Compute source cell indices (vectorized)
    source_row = (y // cell_height).astype(np.intp)
    source_col = (x // cell_width).astype(np.intp)
    source_cell_idx = source_row * nb_cols + source_col

    # Clamp indices for out-of-bounds coords (will be masked out)
    source_cell_idx = np.clip(source_cell_idx, 0, len(dest_rows) - 1)

    # Look up destination cells
    dest_row = dest_rows[source_cell_idx]
    dest_col = dest_cols[source_cell_idx]

    # Compute offsets
    source_y1 = source_row * cell_height
    source_x1 = source_col * cell_width
    dest_y1 = dest_row * cell_height
    dest_x1 = dest_col * cell_width

    # Compute new coordinates
    new_x = dest_x1 + (x - source_x1)
    new_y = dest_y1 + (y - source_y1)

    # Apply only to in-bounds coordinates
    result = np.copy(coords)
    result[in_bounds, 0] = new_x[in_bounds]
    result[in_bounds, 1] = new_y[in_bounds]

    return result


def generate_jigsaw_destinations(
    nb_rows: int, nb_cols: int, max_steps: int, seed: RNGInput, connectivity: int = 4
) -> NDArray[np.integer]:
    """Generate a destination pattern for `apply_jigsaw`.

    Parameters:
        nb_rows: Number of rows to split the image into.
        nb_cols: Number of columns to split the image into.
        max_steps: Maximum distance each cell may move.
        seed: Random seed or RNG instance.
        connectivity: Connectivity for steps (4 or 8). 4 means diagonal moves count as 2 steps.

    Returns:
        2D array of destination cell IDs.
    """
    assert connectivity in (4, 8), f"Expected connectivity of 4 or 8, got {connectivity}."
    random_state = iarandom.RNG.create_if_not_rng_(seed)
    steps = random_state.integers(0, max_steps, size=(nb_rows, nb_cols), endpoint=True)
    directions = random_state.integers(
        0, connectivity, size=(nb_rows, nb_cols, max_steps), endpoint=False
    )
    destinations = np.arange(nb_rows * nb_cols).reshape((nb_rows, nb_cols))

    for step in np.arange(max_steps):
        directions_step = directions[:, :, step]

        for y in np.arange(nb_rows):
            for x in np.arange(nb_cols):
                if steps[y, x] > 0:
                    y_target, x_target = {
                        0: (y - 1, x + 0),
                        1: (y + 0, x + 1),
                        2: (y + 1, x + 0),
                        3: (y + 0, x - 1),
                        4: (y - 1, x - 1),
                        5: (y - 1, x + 1),
                        6: (y + 1, x + 1),
                        7: (y + 1, x - 1),
                    }[directions_step[y, x]]
                    y_target = max(min(y_target, nb_rows - 1), 0)
                    x_target = max(min(x_target, nb_cols - 1), 0)

                    target_steps = steps[y_target, x_target]
                    if (y, x) != (y_target, x_target) and target_steps >= 1:
                        source_dest = destinations[y, x]
                        target_dest = destinations[y_target, x_target]
                        destinations[y, x] = target_dest
                        destinations[y_target, x_target] = source_dest

                        steps[y, x] -= 1
                        steps[y_target, x_target] -= 1

    return destinations



class Jigsaw(meta.Augmenter):
    """Move cells within images similar to jigsaw patterns.

    .. note::

        This augmenter will by default pad images until their height is a
        multiple of `nb_rows`. Analogous for `nb_cols`.

    .. note::

        This augmenter will resize heatmaps and segmentation maps to the
        image size, then apply similar padding as for the corresponding images
        and resize back to the original map size. That also means that images
        may change in shape (due to padding), but heatmaps/segmaps will not
        change. For heatmaps/segmaps, this deviates from pad augmenters that
        will change images and heatmaps/segmaps in corresponding ways and then
        keep the heatmaps/segmaps at the new size.

    .. warning::

        This augmenter currently only supports augmentation of images,
        heatmaps, segmentation maps and keypoints. Other augmentables,
        i.e. bounding boxes, polygons and line strings, will result in errors.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.geometric.apply_jigsaw`.

    Parameters
    ----------
    nb_rows : int or list of int or tuple of int or imgaug2.parameters.StochasticParameter, optional
        How many rows the jigsaw pattern should have.

            * If a single ``int``, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the discrete interval ``[a..b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the value to use.

    nb_cols : int or list of int or tuple of int or imgaug2.parameters.StochasticParameter, optional
        How many cols the jigsaw pattern should have.

            * If a single ``int``, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the discrete interval ``[a..b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the value to use.

    max_steps : int or list of int or tuple of int or imgaug2.parameters.StochasticParameter, optional
        How many steps each jigsaw cell may be moved.

            * If a single ``int``, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the discrete interval ``[a..b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the value to use.

    allow_pad : bool, optional
        Whether to allow automatically padding images until they are evenly
        divisible by ``nb_rows`` and ``nb_cols``.

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
    >>> aug = iaa.Jigsaw(nb_rows=10, nb_cols=10)

    Create a jigsaw augmenter that splits images into ``10x10`` cells
    and shifts them around by ``0`` to ``2`` steps (default setting).

    >>> aug = iaa.Jigsaw(nb_rows=(1, 4), nb_cols=(1, 4))

    Create a jigsaw augmenter that splits each image into ``1`` to ``4``
    cells along each axis.

    >>> aug = iaa.Jigsaw(nb_rows=10, nb_cols=10, max_steps=(1, 5))

    Create a jigsaw augmenter that moves the cells in each image by a random
    amount between ``1`` and ``5`` times (decided per image). Some images will
    be barely changed, some will be fairly distorted.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        nb_rows: int | tuple[int, int] | list[int] | iap.StochasticParameter = (3, 10),
        nb_cols: int | tuple[int, int] | list[int] | iap.StochasticParameter = (3, 10),
        max_steps: int | tuple[int, int] | list[int] | iap.StochasticParameter = 1,
        allow_pad: bool = True,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.nb_rows = iap.handle_discrete_param(
            nb_rows,
            "nb_rows",
            value_range=(1, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self.nb_cols = iap.handle_discrete_param(
            nb_cols,
            "nb_cols",
            value_range=(1, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self.max_steps = iap.handle_discrete_param(
            max_steps,
            "max_steps",
            value_range=(0, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self.allow_pad = allow_pad

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        samples = self._draw_samples(batch, random_state)

        # We resize here heatmaps/segmaps early to the image size in order to
        # avoid problems where the jigsaw cells don't fit perfectly into
        # the heatmap/segmap arrays or there are minor padding-related
        # differences.
        # TODO This step could most likely be avoided.
        # TODO add something like
        #      'with batch.maps_resized_to_image_sizes(): ...'
        batch, maps_shapes_orig = self._resize_maps(batch)

        if self.allow_pad:
            # this is a bit more difficult than one might expect, because we
            # (a) might have different numbers of rows/cols per image
            # (b) might have different shapes per image
            # (c) have non-image data that also requires padding
            # TODO enable support for stochastic parameters in
            #      PadToMultiplesOf, then we can simple use two
            #      DeterministicLists here to generate rowwise values

            for i in np.arange(len(samples.destinations)):
                padder = size_lib.CenterPadToMultiplesOf(
                    width_multiple=samples.nb_cols[i],
                    height_multiple=samples.nb_rows[i],
                    seed=random_state,
                )
                row = batch.subselect_rows_by_indices([i])
                row = padder.augment_batch_(row, parents=parents + [self], hooks=hooks)
                batch = batch.invert_subselect_rows_by_indices_([i], row)

        if batch.images is not None:
            for i, image in enumerate(batch.images):
                image[...] = apply_jigsaw(image, samples.destinations[i])

        if batch.heatmaps is not None:
            for i, heatmap in enumerate(batch.heatmaps):
                heatmap.arr_0to1 = apply_jigsaw(heatmap.arr_0to1, samples.destinations[i])

        if batch.segmentation_maps is not None:
            for i, segmap in enumerate(batch.segmentation_maps):
                segmap.arr = apply_jigsaw(segmap.arr, samples.destinations[i])

        if batch.keypoints is not None:
            for i, kpsoi in enumerate(batch.keypoints):
                xy = kpsoi.to_xy_array()
                xy[...] = apply_jigsaw_to_coords(
                    xy, samples.destinations[i], image_shape=kpsoi.shape
                )
                kpsoi.fill_from_xy_array_(xy)

        has_other_cbaoi = any(
            [
                getattr(batch, attr_name) is not None
                for attr_name in ["bounding_boxes", "polygons", "line_strings"]
            ]
        )
        if has_other_cbaoi:
            raise NotImplementedError(
                "Jigsaw currently only supports augmentation of images, "
                "heatmaps, segmentation maps and keypoints. "
                "Explicitly not supported are: bounding boxes, polygons "
                "and line strings."
            )

        # We don't crop back to the original size, partly because it is
        # rather cumbersome to implement, partly because the padded
        # borders might have been moved into the inner parts of the image

        batch = self._invert_resize_maps(batch, maps_shapes_orig)

        return batch

    @legacy(version="0.4.0")
    def _draw_samples(
        self, batch: _BatchInAugmentation, random_state: iarandom.RNG
    ) -> _JigsawSamples:
        nb_images = batch.nb_rows
        nb_rows = self.nb_rows.draw_samples((nb_images,), random_state=random_state)
        nb_cols = self.nb_cols.draw_samples((nb_images,), random_state=random_state)
        max_steps = self.max_steps.draw_samples((nb_images,), random_state=random_state)
        destinations = []
        for i in np.arange(nb_images):
            destinations.append(
                generate_jigsaw_destinations(
                    nb_rows[i], nb_cols[i], max_steps[i], seed=random_state
                )
            )

        samples = _JigsawSamples(nb_rows, nb_cols, max_steps, destinations)
        return samples

    @legacy(version="0.4.0")
    @classmethod
    def _resize_maps(
        cls, batch: _BatchInAugmentation
    ) -> tuple[_BatchInAugmentation, tuple[list[Shape] | None, list[Shape] | None]]:
        # skip computation of rowwise shapes
        if batch.heatmaps is None and batch.segmentation_maps is None:
            return batch, (None, None)

        image_shapes = batch.get_rowwise_shapes()
        batch.heatmaps, heatmaps_shapes_orig = cls._resize_maps_single_list(
            batch.heatmaps, "arr_0to1", image_shapes
        )
        batch.segmentation_maps, sm_shapes_orig = cls._resize_maps_single_list(
            batch.segmentation_maps, "arr", image_shapes
        )

        return batch, (heatmaps_shapes_orig, sm_shapes_orig)

    @legacy(version="0.4.0")
    @classmethod
    def _resize_maps_single_list(
        cls,
        augmentables: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage] | None,
        arr_attr_name: str,
        image_shapes: Sequence[Shape],
    ) -> tuple[
        list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage] | None,
        list[Shape] | None,
    ]:
        if augmentables is None:
            return None, None

        shapes_orig = []
        augms_resized = []
        for augmentable, image_shape in zip(augmentables, image_shapes, strict=True):
            shape_orig = getattr(augmentable, arr_attr_name).shape
            augm_rs = augmentable.resize(image_shape[0:2])
            augms_resized.append(augm_rs)
            shapes_orig.append(shape_orig)
        return augms_resized, shapes_orig

    @legacy(version="0.4.0")
    @classmethod
    def _invert_resize_maps(
        cls, batch: _BatchInAugmentation, shapes_orig: tuple[list[Shape] | None, list[Shape] | None]
    ) -> _BatchInAugmentation:
        batch.heatmaps = cls._invert_resize_maps_single_list(batch.heatmaps, shapes_orig[0])
        batch.segmentation_maps = cls._invert_resize_maps_single_list(
            batch.segmentation_maps, shapes_orig[1]
        )

        return batch

    @legacy(version="0.4.0")
    @classmethod
    def _invert_resize_maps_single_list(
        cls,
        augmentables: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage] | None,
        shapes_orig: list[Shape] | None,
    ) -> list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage] | None:
        if shapes_orig is None:
            return None

        augms_resized = []
        for augmentable, shape_orig in zip(augmentables, shapes_orig, strict=True):
            augms_resized.append(augmentable.resize(shape_orig[0:2]))
        return augms_resized

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        return [self.nb_rows, self.nb_cols, self.max_steps, self.allow_pad]


@legacy(version="0.4.0")
class _JigsawSamples:
    @legacy(version="0.4.0")
    def __init__(
        self,
        nb_rows: Array,
        nb_cols: Array,
        max_steps: Array,
        destinations: list[NDArray[np.integer]],
    ) -> None:
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.max_steps = max_steps
        self.destinations = destinations


__all__ = [
    "Jigsaw",
    "apply_jigsaw",
    "apply_jigsaw_to_coords",
    "generate_jigsaw_destinations",
]
