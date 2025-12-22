"""Point samplers for segmentation augmenters."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Sequence

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput


class IPointsSampler(metaclass=ABCMeta):
    """Interface for all point samplers.

    Point samplers return coordinate arrays of shape ``Nx2``.
    These coordinates can be used in other augmenters, see e.g.
    :class:`~imgaug2.augmenters.segmentation.Voronoi`.

    """

    @abstractmethod
    def sample_points(self, images: Sequence[Array] | Array, random_state: RNGInput) -> list[Array]:
        """Generate coordinates of points on images.

        Parameters
        ----------
        images : ndarray or list of ndarray
            One or more images for which to generate points.
            If this is a ``list`` of arrays, each one of them is expected to
            have three dimensions.
            If this is an array, it must be four-dimensional and the first
            axis is expected to denote the image index. For ``RGB`` images
            the array would hence have to be of shape ``(N, H, W, 3)``.

        random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence
            A random state to use for any probabilistic function required
            during the point sampling.
            See :func:`~imgaug2.random.RNG` for details.

        Returns
        -------
        ndarray
            An ``(N,2)`` ``float32`` array containing ``(x,y)`` subpixel
            coordinates, all of which being within the intervals
            ``[0.0, width]`` and ``[0.0, height]``.

        """


def _verify_sample_points_images(images: Sequence[Array] | Array) -> None:
    assert len(images) > 0, "Expected at least one image, got zero."
    if isinstance(images, list):
        assert all([ia.is_np_array(image) for image in images]), (
            "Expected list of numpy arrays, got list of types {}.".format(
                ", ".join([str(type(image)) for image in images]),
            )
        )
        assert all([image.ndim == 3 for image in images]), (
            "Expected each image to have three dimensions, got dimensions {}.".format(
                ", ".join([str(image.ndim) for image in images]),
            )
        )
    else:
        assert ia.is_np_array(images), (
            "Expected either a list of numpy arrays or a single numpy "
            f"array of shape NxHxWxC. Got type {type(images)}."
        )
        assert images.ndim == 4, (
            "Expected a four-dimensional array of shape NxHxWxC. "
            f"Got shape {images.ndim} dimensions (shape: {images.shape})."
        )


class RegularGridPointsSampler(IPointsSampler):
    """Sampler that generates a regular grid of coordinates on an image.

    'Regular grid' here means that on each axis all coordinates have the
    same distance from each other. Note that the distance may change between
    axis.

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

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> sampler = iaa.RegularGridPointsSampler(
    >>>     n_rows=(5, 20),
    >>>     n_cols=50)

    Create a point sampler that generates regular grids of points. These grids
    contain ``r`` points on the y-axis, where ``r`` is sampled
    uniformly from the discrete interval ``[5..20]`` per image.
    On the x-axis, the grids always contain ``50`` points.

    """

    def __init__(self, n_rows: ParamInput, n_cols: ParamInput) -> None:
        self.n_rows = iap.handle_discrete_param(
            n_rows,
            "n_rows",
            value_range=(1, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self.n_cols = iap.handle_discrete_param(
            n_cols,
            "n_cols",
            value_range=(1, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )

    def sample_points(self, images: Sequence[Array] | Array, random_state: RNGInput) -> list[Array]:
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        n_rows_lst, n_cols_lst = self._draw_samples(images, random_state)
        return self._generate_point_grids(images, n_rows_lst, n_cols_lst)

    def _draw_samples(
        self, images: Sequence[Array] | Array, random_state: iarandom.RNG
    ) -> tuple[Array, Array]:
        rss = random_state.duplicate(2)
        n_rows_lst = self.n_rows.draw_samples(len(images), random_state=rss[0])
        n_cols_lst = self.n_cols.draw_samples(len(images), random_state=rss[1])
        return self._clip_rows_and_cols(n_rows_lst, n_cols_lst, images)

    @classmethod
    def _clip_rows_and_cols(
        cls, n_rows_lst: Array, n_cols_lst: Array, images: Sequence[Array] | Array
    ) -> tuple[Array, Array]:
        heights = np.int32([image.shape[0] for image in images])
        widths = np.int32([image.shape[1] for image in images])
        # We clip intentionally not to H-1 or W-1 here. If e.g. an image has
        # a width of 1, we want to get a maximum of 1 column of coordinates.
        # Note that we use two clips here instead of e.g. clip(., 1, height),
        # because images can have height/width zero and in these cases numpy
        # prefers the smaller value in clip(). But currently we want to get
        # at least 1 point for such images.
        n_rows_lst = np.clip(n_rows_lst, None, heights)
        n_cols_lst = np.clip(n_cols_lst, None, widths)
        n_rows_lst = np.clip(n_rows_lst, 1, None)
        n_cols_lst = np.clip(n_cols_lst, 1, None)
        return n_rows_lst, n_cols_lst

    @classmethod
    def _generate_point_grids(
        cls, images: Sequence[Array] | Array, n_rows_lst: Array, n_cols_lst: Array
    ) -> list[Array]:
        grids = []
        for image, n_rows_i, n_cols_i in zip(images, n_rows_lst, n_cols_lst, strict=True):
            grids.append(cls._generate_point_grid(image, n_rows_i, n_cols_i))
        return grids

    @classmethod
    def _generate_point_grid(cls, image: Array, n_rows: int, n_cols: int) -> Array:
        height, width = image.shape[0:2]

        # We do not have to subtract 1 here from height/width as these are
        # subpixel coordinates. Technically, we could also place the cell
        # centers outside of the image plane.
        y_spacing = height / n_rows
        y_start = 0.0 + y_spacing / 2
        y_end = height - y_spacing / 2
        if y_start - 1e-4 <= y_end <= y_start + 1e-4:
            yy = np.float32([y_start])
        else:
            yy = np.linspace(y_start, y_end, num=n_rows)

        x_spacing = width / n_cols
        x_start = 0.0 + x_spacing / 2
        x_end = width - x_spacing / 2
        if x_start - 1e-4 <= x_end <= x_start + 1e-4:
            xx = np.float32([x_start])
        else:
            xx = np.linspace(x_start, x_end, num=n_cols)

        xx, yy = np.meshgrid(xx, yy)
        grid = np.vstack([xx.ravel(), yy.ravel()]).T
        return grid

    def __repr__(self) -> str:
        return f"RegularGridPointsSampler({self.n_rows}, {self.n_cols})"

    def __str__(self) -> str:
        return self.__repr__()


class RelativeRegularGridPointsSampler(IPointsSampler):
    """Regular grid coordinate sampler; places more points on larger images.

    This is similar to ``RegularGridPointsSampler``, but the number of rows
    and columns is given as fractions of each image's height and width.
    Hence, more coordinates are generated for larger images.

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

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> sampler = iaa.RelativeRegularGridPointsSampler(
    >>>     n_rows_frac=(0.01, 0.1),
    >>>     n_cols_frac=0.2)

    Create a point sampler that generates regular grids of points. These grids
    contain ``round(y*H)`` points on the y-axis, where ``y`` is sampled
    uniformly from the interval ``[0.01, 0.1]`` per image and ``H`` is the
    image height. On the x-axis, the grids always contain ``0.2*W`` points,
    where ``W`` is the image width.

    """

    def __init__(self, n_rows_frac: ParamInput, n_cols_frac: ParamInput) -> None:
        eps = 1e-4
        self.n_rows_frac = iap.handle_continuous_param(
            n_rows_frac,
            "n_rows_frac",
            value_range=(0.0 + eps, 1.0),
            tuple_to_uniform=True,
            list_to_choice=True,
        )
        self.n_cols_frac = iap.handle_continuous_param(
            n_cols_frac,
            "n_cols_frac",
            value_range=(0.0 + eps, 1.0),
            tuple_to_uniform=True,
            list_to_choice=True,
        )

    def sample_points(self, images: Sequence[Array] | Array, random_state: RNGInput) -> list[Array]:
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        n_rows, n_cols = self._draw_samples(images, random_state)
        return RegularGridPointsSampler._generate_point_grids(images, n_rows, n_cols)

    def _draw_samples(
        self, images: Sequence[Array] | Array, random_state: iarandom.RNG
    ) -> tuple[Array, Array]:
        n_augmentables = len(images)
        rss = random_state.duplicate(2)
        n_rows_frac = self.n_rows_frac.draw_samples(n_augmentables, random_state=rss[0])
        n_cols_frac = self.n_cols_frac.draw_samples(n_augmentables, random_state=rss[1])
        heights = np.int32([image.shape[0] for image in images])
        widths = np.int32([image.shape[1] for image in images])

        n_rows = np.round(n_rows_frac * heights)
        n_cols = np.round(n_cols_frac * widths)
        n_rows, n_cols = RegularGridPointsSampler._clip_rows_and_cols(n_rows, n_cols, images)

        return n_rows.astype(np.int32), n_cols.astype(np.int32)

    def __repr__(self) -> str:
        return f"RelativeRegularGridPointsSampler({self.n_rows_frac}, {self.n_cols_frac})"

    def __str__(self) -> str:
        return self.__repr__()


class DropoutPointsSampler(IPointsSampler):
    """Remove a defined fraction of sampled points.

    Parameters
    ----------
    other_points_sampler : IPointsSampler
        Another point sampler that is queried to generate a list of points.
        The dropout operation will be applied to that list.

    p_drop : number or tuple of number or imgaug2.parameters.StochasticParameter
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

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> sampler = iaa.DropoutPointsSampler(
    >>>     iaa.RegularGridPointsSampler(10, 20),
    >>>     0.2)

    Create a point sampler that first generates points following a regular
    grid of ``10`` rows and ``20`` columns, then randomly drops ``20`` percent
    of these points.

    """

    def __init__(self, other_points_sampler: IPointsSampler, p_drop: ParamInput) -> None:
        assert isinstance(other_points_sampler, IPointsSampler), (
            "Expected to get an instance of IPointsSampler as argument "
            f"'other_points_sampler', got type {type(other_points_sampler)}."
        )
        self.other_points_sampler = other_points_sampler
        self.p_drop = self._convert_p_drop_to_inverted_mask_param(p_drop)

    @classmethod
    def _convert_p_drop_to_inverted_mask_param(cls, p_drop: ParamInput) -> iap.StochasticParameter:
        # TODO this is the same as in Dropout, make DRY
        # TODO add list as an option
        if ia.is_single_number(p_drop):
            p_drop = iap.Binomial(1 - p_drop)
        elif ia.is_iterable(p_drop):
            assert len(p_drop) == 2, (
                "Expected 'p_drop' given as an iterable to contain exactly "
                f"2 values, got {len(p_drop)}."
            )
            assert p_drop[0] < p_drop[1], (
                "Expected 'p_drop' given as iterable to contain exactly 2 "
                f"values (a, b) with a < b. Got {p_drop[0]:.4f} and {p_drop[1]:.4f}."
            )
            assert 0 <= p_drop[0] <= 1.0 and 0 <= p_drop[1] <= 1.0, (
                "Expected 'p_drop' given as iterable to only contain values "
                f"in the interval [0.0, 1.0], got {p_drop[0]:.4f} and {p_drop[1]:.4f}."
            )
            p_drop = iap.Binomial(iap.Uniform(1 - p_drop[1], 1 - p_drop[0]))
        elif isinstance(p_drop, iap.StochasticParameter):
            pass
        else:
            raise Exception(
                f"Expected p_drop to be float or int or StochasticParameter, got {type(p_drop)}."
            )
        return p_drop

    def sample_points(self, images: Sequence[Array] | Array, random_state: RNGInput) -> list[Array]:
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        rss = random_state.duplicate(2)
        points_on_images = self.other_points_sampler.sample_points(images, rss[0])
        drop_masks = self._draw_samples(points_on_images, rss[1])
        return self._apply_dropout_masks(points_on_images, drop_masks)

    def _draw_samples(
        self, points_on_images: list[Array], random_state: iarandom.RNG
    ) -> list[Array]:
        rss = random_state.duplicate(len(points_on_images))
        drop_masks = [
            self._draw_samples_for_image(points_on_image, rs)
            for points_on_image, rs in zip(points_on_images, rss, strict=True)
        ]
        return drop_masks

    def _draw_samples_for_image(self, points_on_image: Array, random_state: iarandom.RNG) -> Array:
        drop_samples = self.p_drop.draw_samples((len(points_on_image),), random_state)
        keep_mask = drop_samples > 0.5
        return keep_mask

    @classmethod
    def _apply_dropout_masks(
        cls, points_on_images: list[Array], keep_masks: list[Array]
    ) -> list[Array]:
        points_on_images_dropped = []
        for points_on_image, keep_mask in zip(points_on_images, keep_masks, strict=True):
            if len(points_on_image) == 0:
                # other sampler didn't provide any points
                poi_dropped = points_on_image
            else:
                if not np.any(keep_mask):
                    # keep at least one point if all were supposed to be
                    # dropped
                    # TODO this could also be moved into its own point sampler,
                    #      like AtLeastOnePoint(...)
                    idx = (len(points_on_image) - 1) // 2
                    keep_mask = np.copy(keep_mask)
                    keep_mask[idx] = True
                poi_dropped = points_on_image[keep_mask, :]
            points_on_images_dropped.append(poi_dropped)
        return points_on_images_dropped

    def __repr__(self) -> str:
        return f"DropoutPointsSampler({self.other_points_sampler}, {self.p_drop})"

    def __str__(self) -> str:
        return self.__repr__()


class UniformPointsSampler(IPointsSampler):
    """Sample points uniformly on images.

    This point sampler generates `n_points` points per image. The x- and
    y-coordinates are both sampled from uniform distributions matching the
    respective image width and height.

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

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> sampler = iaa.UniformPointsSampler(500)

    Create a point sampler that generates an array of ``500`` random points for
    each input image. The x- and y-coordinates of each point are sampled
    from uniform distributions.

    """

    def __init__(self, n_points: ParamInput) -> None:
        self.n_points = iap.handle_discrete_param(
            n_points,
            "n_points",
            value_range=(1, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )

    def sample_points(self, images: Sequence[Array] | Array, random_state: RNGInput) -> list[Array]:
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        rss = random_state.duplicate(2)
        n_points_imagewise = self._draw_samples(len(images), rss[0])

        n_points_total = np.sum(n_points_imagewise)
        n_components_total = 2 * n_points_total
        coords_relative = rss[1].uniform(0.0, 1.0, n_components_total)
        coords_relative_xy = coords_relative.reshape(n_points_total, 2)

        return self._convert_relative_coords_to_absolute(
            coords_relative_xy, n_points_imagewise, images
        )

    def _draw_samples(self, n_augmentables: int, random_state: iarandom.RNG) -> Array:
        n_points = self.n_points.draw_samples((n_augmentables,), random_state=random_state)
        n_points_clipped = np.clip(n_points, 1, None)
        return n_points_clipped

    @classmethod
    def _convert_relative_coords_to_absolute(
        cls, coords_rel_xy: Array, n_points_imagewise: Array, images: Sequence[Array] | Array
    ) -> list[Array]:
        coords_absolute = []
        i = 0
        for image, n_points_image in zip(images, n_points_imagewise, strict=True):
            height, width = image.shape[0:2]
            xx = coords_rel_xy[i : i + n_points_image, 0]
            yy = coords_rel_xy[i : i + n_points_image, 1]

            xx_int = np.clip(np.round(xx * width), 0, width)
            yy_int = np.clip(np.round(yy * height), 0, height)

            coords_absolute.append(np.stack([xx_int, yy_int], axis=-1))
            i += n_points_image
        return coords_absolute

    def __repr__(self) -> str:
        return f"UniformPointsSampler({self.n_points})"

    def __str__(self) -> str:
        return self.__repr__()


class SubsamplingPointsSampler(IPointsSampler):
    """Ensure that the number of sampled points is below a maximum.

    This point sampler will sample points from another sampler and
    then -- in case more points were generated than an allowed maximum --
    will randomly pick `n_points_max` of these.

    Parameters
    ----------
    other_points_sampler : IPointsSampler
        Another point sampler that is queried to generate a ``list`` of points.
        The dropout operation will be applied to that ``list``.

    n_points_max : int
        Maximum number of allowed points. If `other_points_sampler` generates
        more points than this maximum, a random subset of size `n_points_max`
        will be selected.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> sampler = iaa.SubsamplingPointsSampler(
    >>>     iaa.RelativeRegularGridPointsSampler(0.1, 0.2),
    >>>     50
    >>> )

    Create a points sampler that places ``y*H`` points on the y-axis (with
    ``y`` being ``0.1`` and ``H`` being an image's height) and ``x*W`` on
    the x-axis (analogous). Then, if that number of placed points exceeds
    ``50`` (can easily happen for larger images), a random subset of ``50``
    points will be picked and returned.

    """

    def __init__(self, other_points_sampler: IPointsSampler, n_points_max: int) -> None:
        assert isinstance(other_points_sampler, IPointsSampler), (
            "Expected to get an instance of IPointsSampler as argument "
            f"'other_points_sampler', got type {type(other_points_sampler)}."
        )
        self.other_points_sampler = other_points_sampler
        self.n_points_max = np.clip(n_points_max, -1, None)
        if self.n_points_max == 0:
            ia.warn(
                "Got n_points_max=0 in SubsamplingPointsSampler. "
                "This will result in no points ever getting "
                "returned."
            )

    def sample_points(self, images: Sequence[Array] | Array, random_state: RNGInput) -> list[Array]:
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        _verify_sample_points_images(images)

        rss = random_state.duplicate(len(images) + 1)
        points_on_images = self.other_points_sampler.sample_points(images, rss[-1])
        return [
            self._subsample(points_on_image, self.n_points_max, rs)
            for points_on_image, rs in zip(points_on_images, rss[:-1], strict=True)
        ]

    @classmethod
    def _subsample(
        cls, points_on_image: Array, n_points_max: int, random_state: iarandom.RNG
    ) -> Array:
        if len(points_on_image) <= n_points_max:
            return points_on_image
        indices = np.arange(len(points_on_image))
        indices_to_keep = random_state.permutation(indices)[0:n_points_max]
        return points_on_image[indices_to_keep]

    def __repr__(self) -> str:
        return f"SubsamplingPointsSampler({self.other_points_sampler}, {self.n_points_max:d})"

    def __str__(self) -> str:
        return self.__repr__()


# TODO Add points subsampler that drops points close to each other first
# TODO Add poisson points sampler
# TODO Add jitter points sampler that moves points around
# for both see https://codegolf.stackexchange.com/questions/50299/draw-an-image-as-a-voronoi-map/50345#50345
