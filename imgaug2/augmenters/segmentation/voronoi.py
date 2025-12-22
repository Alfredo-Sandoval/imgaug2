"""Voronoi-based segmentation augmenters."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy

from ._utils import _ensure_image_max_size
from .replace import replace_segments_
from .samplers import (
    IPointsSampler,
    RegularGridPointsSampler,
    RelativeRegularGridPointsSampler,
    DropoutPointsSampler,
    UniformPointsSampler,
    SubsamplingPointsSampler,
)


def segment_voronoi(
    image: Array, cell_coordinates: Array, replace_mask: Array | None = None
) -> Array:
    """Average colors within voronoi cells of an image.

    **Supported dtypes**:

    if (image size <= max_size):

        * ``uint8``: yes; fully tested
        * ``uint16``: no; not tested
        * ``uint32``: no; not tested
        * ``uint64``: no; not tested
        * ``int8``: no; not tested
        * ``int16``: no; not tested
        * ``int32``: no; not tested
        * ``int64``: no; not tested
        * ``float16``: no; not tested
        * ``float32``: no; not tested
        * ``float64``: no; not tested
        * ``float128``: no; not tested
        * ``bool``: no; not tested

    if (image size > max_size):

        minimum of (
            ``imgaug2.augmenters.segmentation.Voronoi(image size <= max_size)``,
            :func:`~imgaug2.augmenters.segmentation._ensure_image_max_size`
        )

    Parameters
    ----------
    image : ndarray
        The image to convert to a voronoi image. May be ``HxW`` or
        ``HxWxC``. Note that for ``RGBA`` images the alpha channel
        will currently also by averaged.

    cell_coordinates : ndarray
        A ``Nx2`` float array containing the center coordinates of voronoi
        cells on the image. Values are expected to be in the interval
        ``[0.0, height-1.0]`` for the y-axis (x-axis analogous).
        If this array contains no coordinate, the image will not be
        changed.

    replace_mask : None or ndarray, optional
        Boolean mask of the same length as `cell_coordinates`, denoting
        for each cell whether its pixels are supposed to be replaced
        by the cell's average color (``True``) or left untouched (``False``).
        If this is set to ``None``, all cells will be replaced.

    Returns
    -------
    ndarray
        Voronoi image.

    """
    from imgaug2.mlx._core import is_mlx_array

    if (
        is_mlx_array(image)
        or is_mlx_array(cell_coordinates)
        or (replace_mask is not None and is_mlx_array(replace_mask))
    ):
        from imgaug2.mlx import segmentation as mlx_segmentation

        return cast(Array, mlx_segmentation.segment_voronoi(image, cell_coordinates, replace_mask))

    input_dims = image.ndim
    if input_dims == 2:
        image = image[..., np.newaxis]

    if len(cell_coordinates) <= 0:
        if input_dims == 2:
            return image[..., 0]
        return image

    height, width = image.shape[0:2]
    ids_of_nearest_cells = _match_pixels_with_voronoi_cells(height, width, cell_coordinates)
    image_aug = replace_segments_(
        image, ids_of_nearest_cells.reshape(image.shape[0:2]), replace_mask
    )

    if input_dims == 2:
        return image_aug[..., 0]
    return image_aug


def _match_pixels_with_voronoi_cells(height: int, width: int, cell_coordinates: Array) -> Array:
    # deferred import so that scipy is an optional dependency
    from scipy.spatial import cKDTree as KDTree  # TODO add scipy for reqs

    tree = KDTree(cell_coordinates)
    pixel_coords = _generate_pixel_coords(height, width)
    pixel_coords_subpixel = pixel_coords.astype(np.float32) + 0.5
    ids_of_nearest_cells = tree.query(pixel_coords_subpixel)[1]
    return ids_of_nearest_cells


def _generate_pixel_coords(height: int, width: int) -> Array:
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    return np.c_[xx.ravel(), yy.ravel()]


# TODO this can be reduced down to a similar problem as Superpixels:
#      generate an integer-based class id map of segments, then replace all
#      segments with the same class id by the average color within that
#      segment
class Voronoi(meta.Augmenter):
    """Average colors of an image within Voronoi cells.

    This augmenter performs the following steps:

        1. Query `points_sampler` to sample random coordinates of cell
           centers. On the image.
        2. Estimate for each pixel to which voronoi cell (i.e. segment)
           it belongs. Each pixel belongs to the cell with the closest center
           coordinate (euclidean distance).
        3. Compute for each cell the average color of the pixels within it.
        4. Replace the pixels of `p_replace` percent of all cells by their
           average color. Do not change the pixels of ``(1 - p_replace)``
           percent of all cells. (The percentages are average values over
           many images. Some images may get more/less cells replaced by
           their average color.)

    This code is very loosely based on
    https://codegolf.stackexchange.com/questions/50299/draw-an-image-as-a-voronoi-map/50345#50345

    **Supported dtypes**:

    See :func:`imgaug2.augmenters.segmentation.segment_voronoi`.

    Parameters
    ----------
    points_sampler : IPointsSampler
        A points sampler which will be queried per image to generate the
        coordinates of the centers of voronoi cells.

    p_replace : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color (otherwise, the pixels
        are not changed).
        Examples:

            * A probability of ``0.0`` would mean, that the pixels in no
              segment are replaced by their average color (image is not
              changed at all).
            * A probability of ``0.5`` would mean, that around half of all
              segments are replaced by their average color.
            * A probability of ``1.0`` would mean, that all segments are
              replaced by their average color (resulting in a voronoi
              image).

        Behaviour based on chosen datatypes for this parameter:

            * If a ``number``, then that ``number`` will always be used.
            * If ``tuple`` ``(a, b)``, then a random probability will be
              sampled from the interval ``[a, b]`` per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, it is expected to return
              values between ``0.0`` and ``1.0`` and will be queried *for each
              individual segment* to determine whether it is supposed to
              be averaged (``>0.5``) or not (``<=0.5``).
              Recommended to be some form of ``Binomial(...)``.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.
        If the width or height of an image exceeds this value, it will be
        downscaled before the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the process. The final output image has the
        same size as the input image. Note that in case `p_replace` is below
        ``1.0``, the down-/upscaling will affect the not-replaced pixels too.
        Use ``None`` to apply no down-/upscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug2.imgaug2.imresize_single_image`.

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
    >>> points_sampler = iaa.RegularGridPointsSampler(n_cols=20, n_rows=40)
    >>> aug = iaa.Voronoi(points_sampler)

    Create an augmenter that places a ``20x40`` (``HxW``) grid of cells on
    the image and replaces all pixels within each cell by the cell's average
    color. The process is performed at an image size not exceeding ``128`` px
    on any side (default). If necessary, the downscaling is performed using
    ``linear`` interpolation (default).

    >>> points_sampler = iaa.DropoutPointsSampler(
    >>>     iaa.RelativeRegularGridPointsSampler(
    >>>         n_cols_frac=(0.05, 0.2),
    >>>         n_rows_frac=0.1),
    >>>     0.2)
    >>> aug = iaa.Voronoi(points_sampler, p_replace=0.9, max_size=None)

    Create a voronoi augmenter that generates a grid of cells dynamically
    adapted to the image size. Larger images get more cells. On the x-axis,
    the distance between two cells is ``w * W`` pixels, where ``W`` is the
    width of the image and ``w`` is always ``0.1``. On the y-axis,
    the distance between two cells is ``h * H`` pixels, where ``H`` is the
    height of the image and ``h`` is sampled uniformly from the interval
    ``[0.05, 0.2]``. To make the voronoi pattern less regular, about ``20``
    percent of the cell coordinates are randomly dropped (i.e. the remaining
    cells grow in size). In contrast to the first example, the image is not
    resized (if it was, the sampling would happen *after* the resizing,
    which would affect ``W`` and ``H``). Not all voronoi cells are replaced
    by their average color, only around ``90`` percent of them. The
    remaining ``10`` percent's pixels remain unchanged.

    """

    def __init__(
        self,
        points_sampler: IPointsSampler,
        p_replace: ParamInput = 1.0,
        max_size: int | None = 128,
        interpolation: str | int = "linear",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        assert isinstance(points_sampler, IPointsSampler), (
            "Expected 'points_sampler' to be an instance of IPointsSampler, "
            f"got {type(points_sampler)}."
        )
        self.points_sampler = points_sampler

        self.p_replace = iap.handle_probability_param(
            p_replace, "p_replace", tuple_to_uniform=True, list_to_choice=True
        )

        self.max_size = max_size
        self.interpolation = interpolation

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        if batch.images is None:
            return batch

        images = batch.images

        iadt.allow_only_uint8(images, augmenter=self)

        rss = random_state.duplicate(len(images))
        for i, (image, rs) in enumerate(zip(images, rss, strict=True)):
            batch.images[i] = self._augment_single_image(image, rs)
        return batch

    def _augment_single_image(self, image: Array, random_state: iarandom.RNG) -> Array:
        rss = random_state.duplicate(2)
        orig_shape = image.shape
        image = _ensure_image_max_size(image, self.max_size, self.interpolation)

        cell_coordinates = self.points_sampler.sample_points([image], rss[0])[0]
        p_replace = self.p_replace.draw_samples((len(cell_coordinates),), rss[1])
        replace_mask = p_replace > 0.5

        from imgaug2.augmenters import segmentation as _segmentation

        image_aug = _segmentation.segment_voronoi(image, cell_coordinates, replace_mask)

        if orig_shape != image_aug.shape:
            image_aug = ia.imresize_single_image(
                image_aug, orig_shape[0:2], interpolation=self.interpolation
            )

        return image_aug

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.points_sampler, self.p_replace, self.max_size, self.interpolation]


class UniformVoronoi(Voronoi):
    """Uniformly sample Voronoi cells on images and average colors within them.

    This augmenter is a shortcut for the combination of
    :class:`~imgaug2.augmenters.segmentation.Voronoi` with
    :class:`~imgaug2.augmenters.segmentation.UniformPointsSampler`. Hence, it
    generates a fixed amount of ``N`` random coordinates of voronoi cells on
    each image. The cell coordinates are sampled uniformly using the image
    height and width as maxima.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.segmentation.Voronoi`.

    Parameters
    ----------
    n_points : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Number of points to sample on each image.

            * If a single ``int``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    p_replace : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color (otherwise, the pixels
        are not changed).
        Examples:

            * A probability of ``0.0`` would mean, that the pixels in no
              segment are replaced by their average color (image is not
              changed at all).
            * A probability of ``0.5`` would mean, that around half of all
              segments are replaced by their average color.
            * A probability of ``1.0`` would mean, that all segments are
              replaced by their average color (resulting in a voronoi
              image).

        Behaviour based on chosen datatypes for this parameter:

            * If a ``number``, then that ``number`` will always be used.
            * If ``tuple`` ``(a, b)``, then a random probability will be
              sampled from the interval ``[a, b]`` per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, it is expected to return
              values between ``0.0`` and ``1.0`` and will be queried *for each
              individual segment* to determine whether it is supposed to
              be averaged (``>0.5``) or not (``<=0.5``).
              Recommended to be some form of ``Binomial(...)``.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.
        If the width or height of an image exceeds this value, it will be
        downscaled before the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the process. The final output image has the
        same size as the input image. Note that in case `p_replace` is below
        ``1.0``, the down-/upscaling will affect the not-replaced pixels too.
        Use ``None`` to apply no down-/upscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug2.imgaug2.imresize_single_image`.

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
    >>> aug = iaa.UniformVoronoi((100, 500))

    Sample for each image uniformly the number of voronoi cells ``N`` from the
    interval ``[100, 500]``. Then generate ``N`` coordinates by sampling
    uniformly the x-coordinates from ``[0, W]`` and the y-coordinates from
    ``[0, H]``, where ``H`` is the image height and ``W`` the image width.
    Then use these coordinates to group the image pixels into voronoi
    cells and average the colors within them. The process is performed at an
    image size not exceeding ``128`` px on any side (default). If necessary,
    the downscaling is performed using ``linear`` interpolation (default).

    >>> aug = iaa.UniformVoronoi(250, p_replace=0.9, max_size=None)

    Same as above, but always samples ``N=250`` cells, replaces only
    ``90`` percent of them with their average color (the pixels of the
    remaining ``10`` percent are not changed) and performs the transformation
    at the original image size (``max_size=None``).

    """

    def __init__(
        self,
        n_points: ParamInput = (50, 500),
        p_replace: ParamInput = (0.5, 1.0),
        max_size: int | None = 128,
        interpolation: str | int = "linear",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            points_sampler=UniformPointsSampler(n_points),
            p_replace=p_replace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class RegularGridVoronoi(Voronoi):
    """Sample Voronoi cells from regular grids and color-average them.

    This augmenter is a shortcut for the combination of
    :class:`~imgaug2.augmenters.segmentation.Voronoi`,
    :class:`~imgaug2.augmenters.segmentation.RegularGridPointsSampler` and
    :class:`~imgaug2.augmenters.segmentation.DropoutPointsSampler`. Hence, it
    generates a regular grid with ``R`` rows and ``C`` columns of coordinates
    on each image. Then, it drops ``p`` percent of the ``R*C`` coordinates
    to randomize the grid. Each image pixel then belongs to the voronoi
    cell with the closest coordinate.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.segmentation.Voronoi`.

    Parameters
    ----------
    n_rows : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Number of rows of coordinates to place on each image, i.e. the number
        of coordinates on the y-axis. Note that for each image, the sampled
        value is clipped to the interval ``[1..H]``, where ``H`` is the image
        height.

            * If a single ``int``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    n_cols : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Number of columns of coordinates to place on each image, i.e. the
        number of coordinates on the x-axis. Note that for each image, the
        sampled value is clipped to the interval ``[1..W]``, where ``W`` is
        the image width.

            * If a single ``int``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    p_drop_points : number or tuple of number or imgaug2.parameters.StochasticParameter, optional
        The probability that a coordinate will be removed from the list
        of all sampled coordinates. A value of ``1.0`` would mean that (on
        average) ``100`` percent of all coordinates will be dropped,
        while ``0.0`` denotes ``0`` percent. Note that this sampler will
        always ensure that at least one coordinate is left after the dropout
        operation, i.e. even ``1.0`` will only drop all *except one*
        coordinate.

            * If a ``float``, then that value will be used for all images.
            * If a ``tuple`` ``(a, b)``, then a value ``p`` will be sampled
              from the interval ``[a, b]`` per image.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per coordinate whether it should be *kept* (sampled
              value of ``>0.5``) or shouldn't be kept (sampled value of
              ``<=0.5``). If you instead want to provide the probability as
              a stochastic parameter, you can usually do
              ``imgaug2.parameters.Binomial(1-p)`` to convert parameter `p` to
              a 0/1 representation.

    p_replace : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color (otherwise, the pixels
        are not changed).
        Examples:

            * A probability of ``0.0`` would mean, that the pixels in no
              segment are replaced by their average color (image is not
              changed at all).
            * A probability of ``0.5`` would mean, that around half of all
              segments are replaced by their average color.
            * A probability of ``1.0`` would mean, that all segments are
              replaced by their average color (resulting in a voronoi
              image).

        Behaviour based on chosen datatypes for this parameter:

            * If a ``number``, then that number will always be used.
            * If ``tuple`` ``(a, b)``, then a random probability will be
              sampled from the interval ``[a, b]`` per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, it is expected to return
              values between ``0.0`` and ``1.0`` and will be queried *for each
              individual segment* to determine whether it is supposed to
              be averaged (``>0.5``) or not (``<=0.5``).
              Recommended to be some form of ``Binomial(...)``.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.
        If the width or height of an image exceeds this value, it will be
        downscaled before the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the process. The final output image has the
        same size as the input image. Note that in case `p_replace` is below
        ``1.0``, the down-/upscaling will affect the not-replaced pixels too.
        Use ``None`` to apply no down-/upscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug2.imgaug2.imresize_single_image`.

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
    >>> aug = iaa.RegularGridVoronoi(10, 20)

    Place a regular grid of ``10x20`` (``height x width``) coordinates on
    each image. Randomly drop on average ``20`` percent of these points
    to create a less regular pattern. Then use the remaining coordinates
    to group the image pixels into voronoi cells and average the colors
    within them. The process is performed at an image size not exceeding
    ``128`` px on any side (default). If necessary, the downscaling is
    performed using ``linear`` interpolation (default).

    >>> aug = iaa.RegularGridVoronoi(
    >>>     (10, 30), 20, p_drop_points=0.0, p_replace=0.9, max_size=None)

    Same as above, generates a grid with randomly ``10`` to ``30`` rows,
    drops none of the generates points, replaces only ``90`` percent of
    the voronoi cells with their average color (the pixels of the remaining
    ``10`` percent are not changed) and performs the transformation
    at the original image size (``max_size=None``).

    """

    def __init__(
        self,
        n_rows: ParamInput = (10, 30),
        n_cols: ParamInput = (10, 30),
        p_drop_points: ParamInput = (0.0, 0.5),
        p_replace: ParamInput = (0.5, 1.0),
        max_size: int | None = 128,
        interpolation: str | int = "linear",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            points_sampler=DropoutPointsSampler(
                RegularGridPointsSampler(n_rows, n_cols), p_drop_points
            ),
            p_replace=p_replace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class RelativeRegularGridVoronoi(Voronoi):
    """Sample Voronoi cells from image-dependent grids and color-average them.

    This augmenter is a shortcut for the combination of
    :class:`~imgaug2.augmenters.segmentation.Voronoi`,
    :class:`~imgaug2.augmenters.segmentation.RegularGridPointsSampler` and
    :class:`~imgaug2.augmenters.segmentation.DropoutPointsSampler`. Hence, it
    generates a regular grid with ``R`` rows and ``C`` columns of coordinates
    on each image. Then, it drops ``p`` percent of the ``R*C`` coordinates
    to randomize the grid. Each image pixel then belongs to the voronoi
    cell with the closest coordinate.

    .. note::

        In contrast to the other voronoi augmenters, this one uses
        ``None`` as the default value for `max_size`, i.e. the color averaging
        is always performed at full resolution. This enables the augmenter to
        make use of the additional points on larger images. It does
        however slow down the augmentation process.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.segmentation.Voronoi`.

    Parameters
    ----------
    n_rows_frac : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Relative number of coordinates to place on the y-axis. For a value
        ``y`` and image height ``H`` the number of actually placed coordinates
        (i.e. computed rows) is given by ``int(round(y*H))``.
        Note that for each image, the number of coordinates is clipped to the
        interval ``[1,H]``, where ``H`` is the image height.

            * If a single ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the interval
              ``[a, b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    n_cols_frac : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Relative number of coordinates to place on the x-axis. For a value
        ``x`` and image height ``W`` the number of actually placed coordinates
        (i.e. computed columns) is given by ``int(round(x*W))``.
        Note that for each image, the number of coordinates is clipped to the
        interval ``[1,W]``, where ``W`` is the image width.

            * If a single ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the interval
              ``[a, b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    p_drop_points : number or tuple of number or imgaug2.parameters.StochasticParameter, optional
        The probability that a coordinate will be removed from the list
        of all sampled coordinates. A value of ``1.0`` would mean that (on
        average) ``100`` percent of all coordinates will be dropped,
        while ``0.0`` denotes ``0`` percent. Note that this sampler will
        always ensure that at least one coordinate is left after the dropout
        operation, i.e. even ``1.0`` will only drop all *except one*
        coordinate.

            * If a ``float``, then that value will be used for all images.
            * If a ``tuple`` ``(a, b)``, then a value ``p`` will be sampled
              from the interval ``[a, b]`` per image.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per coordinate whether it should be *kept* (sampled
              value of ``>0.5``) or shouldn't be kept (sampled value of
              ``<=0.5``). If you instead want to provide the probability as
              a stochastic parameter, you can usually do
              ``imgaug2.parameters.Binomial(1-p)`` to convert parameter `p` to
              a 0/1 representation.

    p_replace : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color (otherwise, the pixels
        are not changed).
        Examples:

            * A probability of ``0.0`` would mean, that the pixels in no
              segment are replaced by their average color (image is not
              changed at all).
            * A probability of ``0.5`` would mean, that around half of all
              segments are replaced by their average color.
            * A probability of ``1.0`` would mean, that all segments are
              replaced by their average color (resulting in a voronoi
              image).

        Behaviour based on chosen datatypes for this parameter:

            * If a ``number``, then that ``number`` will always be used.
            * If ``tuple`` ``(a, b)``, then a random probability will be
              sampled from the interval ``[a, b]`` per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, it is expected to return
              values between ``0.0`` and ``1.0`` and will be queried *for each
              individual segment* to determine whether it is supposed to
              be averaged (``>0.5``) or not (``<=0.5``).
              Recommended to be some form of ``Binomial(...)``.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.
        If the width or height of an image exceeds this value, it will be
        downscaled before the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the process. The final output image has the
        same size as the input image. Note that in case `p_replace` is below
        ``1.0``, the down-/upscaling will affect the not-replaced pixels too.
        Use ``None`` to apply no down-/upscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug2.imgaug2.imresize_single_image`.

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
    >>> aug = iaa.RelativeRegularGridVoronoi(0.1, 0.25)

    Place a regular grid of ``R x C`` coordinates on each image, where
    ``R`` is the number of rows and computed as ``R=0.1*H`` with ``H`` being
    the height of the input image. ``C`` is the number of columns and
    analogously estimated from the image width ``W`` as ``C=0.25*W``.
    Larger images will lead to larger ``R`` and ``C`` values.
    On average, ``20`` percent of these grid coordinates are randomly
    dropped to create a less regular pattern. Then, the remaining coordinates
    are used to group the image pixels into voronoi cells and the colors
    within them are averaged.

    >>> aug = iaa.RelativeRegularGridVoronoi(
    >>>     (0.03, 0.1), 0.1, p_drop_points=0.0, p_replace=0.9, max_size=512)

    Same as above, generates a grid with randomly ``R=r*H`` rows, where
    ``r`` is sampled uniformly from the interval ``[0.03, 0.1]`` and
    ``C=0.1*W`` rows. No points are dropped. The augmenter replaces only
    ``90`` percent of the voronoi cells with their average color (the pixels
    of the remaining ``10`` percent are not changed). Images larger than
    ``512`` px are temporarily downscaled (*before* sampling the grid points)
    so that no side exceeds ``512`` px. This improves performance, but
    degrades the quality of the resulting image.

    """

    def __init__(
        self,
        n_rows_frac: ParamInput = (0.05, 0.15),
        n_cols_frac: ParamInput = (0.05, 0.15),
        p_drop_points: ParamInput = (0.0, 0.5),
        p_replace: ParamInput = (0.5, 1.0),
        max_size: int | None = None,
        interpolation: str | int = "linear",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            points_sampler=DropoutPointsSampler(
                RelativeRegularGridPointsSampler(n_rows_frac, n_cols_frac), p_drop_points
            ),
            p_replace=p_replace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )
