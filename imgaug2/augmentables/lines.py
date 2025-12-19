"""Classes representing lines."""

from __future__ import annotations

import copy as copylib
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Literal, TypeVar, cast, overload

import cv2
import numpy as np
import skimage.draw
import skimage.measure
from numpy.typing import NDArray

import imgaug2.imgaug as ia

from imgaug2.augmentables.utils import (
    Number,
    Point2D,
    Point2DList,
    Shape,
    _handle_on_image_shape,
    _normalize_shift_args,
    _remove_out_of_image_fraction_,
    interpolate_points,
    normalize_imglike_shape,
    project_coords_,
)
from imgaug2.compat.markers import legacy
from imgaug2.augmenters._blend_utils import blend_alpha

if TYPE_CHECKING:
    from imgaug2.augmentables.bbs import BoundingBox
    from imgaug2.augmentables.heatmaps import HeatmapsOnImage
    from imgaug2.augmentables.kps import Keypoint, KeypointsOnImage
    from imgaug2.augmentables.polys import Polygon
    from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

_TDefault = TypeVar("_TDefault")


# TODO Add Line class and make LineString a list of Line elements
# TODO add to_distance_maps(), compute_hausdorff_distance(), intersects(),
#      find_self_intersections(), is_self_intersecting(),
#      remove_self_intersections()
class LineString:
    """Class representing line strings.

    A line string is a collection of connected line segments, each
    having a start and end point. Each point is given as its ``(x, y)``
    absolute (sub-)pixel coordinates. The end point of each segment is
    also the start point of the next segment.

    The line string is not closed, i.e. start and end point are expected to
    differ and will not be connected in drawings.

    Parameters
    ----------
    coords : iterable of tuple of number or ndarray
        The points of the line string.

    label : None or str, optional
        The label of the line string.

    """

    def __init__(
        self,
        coords: Sequence[Sequence[float]] | NDArray[np.number],
        label: str | None = None,
    ) -> None:
        """Create a new LineString instance."""
        coords_arr: NDArray[np.float32]
        if ia.is_np_array(coords):
            # avoid unnecessary copies of ndarray inputs
            coords_arr = coords.astype(np.float32, copy=False)
        elif len(coords) == 0:
            coords_arr = np.zeros((0, 2), dtype=np.float32)
        else:
            assert ia.is_iterable(coords), (
                f"Expected 'coords' to be an iterable, got type {type(coords)}."
            )
            assert all([len(coords_i) == 2 for coords_i in coords]), (
                f"Expected 'coords' to contain (x,y) tuples, got {str(coords)}."
            )
            coords_arr = np.asarray(coords, dtype=np.float32)

        assert coords_arr.ndim == 2 and coords_arr.shape[-1] == 2, (
            f"Expected 'coords' to have shape (N, 2), got shape {coords_arr.shape}."
        )

        self.coords: NDArray[np.float32] = coords_arr
        self.label = label

    @property
    def length(self) -> float:
        """Compute the total euclidean length of the line string.

        Returns
        -------
        float
            The length based on euclidean distance, i.e. the sum of the
            lengths of each line segment.

        """
        if len(self.coords) == 0:
            return 0.0
        return np.sum(self.compute_neighbour_distances())

    @property
    def xx(self) -> NDArray[np.float32]:
        """Get an array of x-coordinates of all points of the line string.

        Returns
        -------
        ndarray
            ``float32`` x-coordinates of the line string points.

        """
        return self.coords[:, 0]

    @property
    def yy(self) -> NDArray[np.float32]:
        """Get an array of y-coordinates of all points of the line string.

        Returns
        -------
        ndarray
            ``float32`` y-coordinates of the line string points.

        """
        return self.coords[:, 1]

    @property
    def xx_int(self) -> NDArray[np.int32]:
        """Get an array of discrete x-coordinates of all points.

        The conversion from ``float32`` coordinates to ``int32`` is done
        by first rounding the coordinates to the closest integer and then
        removing everything after the decimal point.

        Returns
        -------
        ndarray
            ``int32`` x-coordinates of the line string points.

        """
        return np.round(self.xx).astype(np.int32)

    @property
    def yy_int(self) -> NDArray[np.int32]:
        """Get an array of discrete y-coordinates of all points.

        The conversion from ``float32`` coordinates to ``int32`` is done
        by first rounding the coordinates to the closest integer and then
        removing everything after the decimal point.

        Returns
        -------
        ndarray
            ``int32`` y-coordinates of the line string points.

        """
        return np.round(self.yy).astype(np.int32)

    @property
    def height(self) -> float:
        """Compute the height of a bounding box encapsulating the line.

        The height is computed based on the two points with lowest and
        largest y-coordinates.

        Returns
        -------
        float
            The height of the line string.

        """
        if len(self.coords) <= 1:
            return 0.0
        return np.max(self.yy) - np.min(self.yy)

    @property
    def width(self) -> float:
        """Compute the width of a bounding box encapsulating the line.

        The width is computed based on the two points with lowest and
        largest x-coordinates.

        Returns
        -------
        float
            The width of the line string.

        """
        if len(self.coords) <= 1:
            return 0.0
        return np.max(self.xx) - np.min(self.xx)

    def get_pointwise_inside_image_mask(self, image: object) -> NDArray[np.bool_]:
        """Determine per point whether it is inside of a given image plane.

        Parameters
        ----------
        image : ndarray or tuple of int
            Either an image with shape ``(H,W,[C])`` or a ``tuple`` denoting
            such an image shape.

        Returns
        -------
        ndarray
            ``(N,) ``bool`` array with one value for each of the ``N`` points
            indicating whether it is inside of the provided image
            plane (``True``) or not (``False``).

        """
        if len(self.coords) == 0:
            return np.zeros((0,), dtype=bool)
        shape = normalize_imglike_shape(image)
        height, width = shape[0:2]
        x_within = np.logical_and(0 <= self.xx, self.xx < width)
        y_within = np.logical_and(0 <= self.yy, self.yy < height)
        return np.logical_and(x_within, y_within)

    def compute_neighbour_distances(self, closed: bool = False) -> NDArray[np.float32]:
        """Compute the euclidean distance between each two consecutive points.

        Parameters
        ----------
        closed : bool, optional
            Whether to also include the distance between the last and first
            point, effectively closing the line string.

        Returns
        -------
        ndarray
            ``(N-1,)`` (or ``(N,)`` if `closed=True`) ``float32`` array of
            euclidean distances between point pairs. Same order as in
            `coords`.

        """
        if len(self.coords) <= 1:
            return np.zeros((0,), dtype=np.float32)
        diffs = self.coords[1:, :] - self.coords[:-1, :]
        if closed and len(self.coords) >= 2:
            diffs = np.vstack([diffs, self.coords[0, :] - self.coords[-1, :]])
        return np.sqrt(np.sum(diffs**2, axis=1))

    # TODO change output to array
    def compute_pointwise_distances(
        self, other: object, default: object | None = None
    ) -> list[float] | object:
        """Compute min distances between points of this and another line string.

        Parameters
        ----------
        other : tuple of number or imgaug2.augmentables.kps.Keypoint or imgaug2.augmentables.LineString
            Other object to which to compute the distances.

        default : any
            Value to return if `other` contains no points.

        Returns
        -------
        list of float or any
            For each coordinate of this line string, the distance to any
            closest location on `other`.
            `default` if no distance could be computed.

        """
        import shapely.geometry

        from imgaug2.augmentables.kps import Keypoint

        if isinstance(other, Keypoint):
            other = shapely.geometry.Point((other.x, other.y))
        elif isinstance(other, LineString):
            if len(other.coords) == 0:
                return default
            if len(other.coords) == 1:
                other = shapely.geometry.Point(other.coords[0, :])
            else:
                other = shapely.geometry.LineString(other.coords)
        elif isinstance(other, tuple):
            assert len(other) == 2, (
                f"Expected tuple 'other' to contain exactly two entries, got {len(other)}."
            )
            other = shapely.geometry.Point(other)
        else:
            raise ValueError(
                f"Expected Keypoint or LineString or tuple (x,y), got type {type(other)}."
            )

        return [shapely.geometry.Point(point).distance(other) for point in self.coords]

    def compute_distance(self, other: object, default: object | None = None) -> float | object:
        """Compute the minimal distance between the line string and `other`.

        Parameters
        ----------
        other : tuple of number or imgaug2.augmentables.kps.Keypoint or imgaug2.augmentables.LineString
            Other object to which to compute the distance.

        default : any
            Value to return if this line string or `other` contain no points.

        Returns
        -------
        float or any
            Minimal distance to `other` or `default` if no distance could be
            computed.

        """
        import shapely.geometry

        from imgaug2.augmentables.kps import Keypoint

        if len(self.coords) == 0:
            return default

        if len(self.coords) == 1:
            geom_self = shapely.geometry.Point(self.coords[0, :])
        else:
            geom_self = shapely.geometry.LineString(self.coords)

        if isinstance(other, Keypoint):
            geom_other = shapely.geometry.Point((other.x, other.y))
        elif isinstance(other, LineString):
            if len(other.coords) == 0:
                return default
            if len(other.coords) == 1:
                geom_other = shapely.geometry.Point(other.coords[0, :])
            else:
                geom_other = shapely.geometry.LineString(other.coords)
        elif isinstance(other, tuple):
            assert len(other) == 2, (
                f"Expected tuple 'other' to contain exactly two entries, got {len(other)}."
            )
            geom_other = shapely.geometry.Point(other)
        else:
            raise ValueError(
                f"Expected Keypoint or LineString or tuple (x,y), got type {type(other)}."
            )

        return geom_self.distance(geom_other)

    def contains(self, other: object, max_distance: float = 1e-4) -> bool:
        """Estimate whether a point is on this line string.

        This method uses a maximum distance to estimate whether a point is
        on a line string.

        Parameters
        ----------
        other : tuple of number or imgaug2.augmentables.kps.Keypoint
            Point to check for.

        max_distance : float
            Maximum allowed euclidean distance between the point and the
            closest point on the line. If the threshold is exceeded, the point
            is not considered to fall on the line.

        Returns
        -------
        bool
            ``True`` if the point is on the line string, ``False`` otherwise.

        """
        return self.compute_distance(other, default=np.inf) < max_distance

    @legacy(version="0.4.0")
    def project_(self, from_shape: object, to_shape: object) -> LineString:
        """Project the line string onto a differently shaped image in-place.

        E.g. if a point of the line string is on its original image at
        ``x=(10 of 100 pixels)`` and ``y=(20 of 100 pixels)`` and is projected
        onto a new image with size ``(width=200, height=200)``, its new
        position will be ``(x=20, y=40)``.

        This is intended for cases where the original image is resized.
        It cannot be used for more complex changes (e.g. padding, cropping).


        Parameters
        ----------
        from_shape : tuple of int or ndarray
            Shape of the original image. (Before resize.)

        to_shape : tuple of int or ndarray
            Shape of the new image. (After resize.)

        Returns
        -------
        imgaug2.augmentables.lines.LineString
            Line string with new coordinates.
            The object may have been modified in-place.

        """
        self.coords = project_coords_(self.coords, from_shape, to_shape)
        return self

    def project(self, from_shape: object, to_shape: object) -> LineString:
        """Project the line string onto a differently shaped image.

        E.g. if a point of the line string is on its original image at
        ``x=(10 of 100 pixels)`` and ``y=(20 of 100 pixels)`` and is projected
        onto a new image with size ``(width=200, height=200)``, its new
        position will be ``(x=20, y=40)``.

        This is intended for cases where the original image is resized.
        It cannot be used for more complex changes (e.g. padding, cropping).

        Parameters
        ----------
        from_shape : tuple of int or ndarray
            Shape of the original image. (Before resize.)

        to_shape : tuple of int or ndarray
            Shape of the new image. (After resize.)

        Returns
        -------
        imgaug2.augmentables.lines.LineString
            Line string with new coordinates.

        """
        return self.deepcopy().project_(from_shape, to_shape)

    @legacy(version="0.4.0")
    def compute_out_of_image_fraction(self, image: object) -> float:
        """Compute fraction of polygon area outside of the image plane.

        This estimates ``f = A_ooi / A``, where ``A_ooi`` is the area of the
        polygon that is outside of the image plane, while ``A`` is the
        total area of the bounding box.


        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape
            and must contain at least two integers.

        Returns
        -------
        float
            Fraction of the polygon area that is outside of the image
            plane. Returns ``0.0`` if the polygon is fully inside of
            the image plane. If the polygon has an area of zero, the polygon
            is treated similarly to a :class:`LineString`, i.e. the fraction
            of the line that is inside the image plane is returned.

        """
        length = self.length
        if length == 0:
            if len(self.coords) == 0:
                return 0.0
            points_ooi = ~self.get_pointwise_inside_image_mask(image)
            return 1.0 if np.all(points_ooi) else 0.0
        lss_clipped = self.clip_out_of_image(image)
        length_after_clip = sum([ls.length for ls in lss_clipped])
        inside_image_factor = length_after_clip / length
        return 1.0 - inside_image_factor

    def is_fully_within_image(self, image: object, default: object = False) -> bool | object:
        """Estimate whether the line string is fully inside an image plane.

        Parameters
        ----------
        image : ndarray or tuple of int
            Either an image with shape ``(H,W,[C])`` or a ``tuple`` denoting
            such an image shape.

        default : any
            Default value to return if the line string contains no points.

        Returns
        -------
        bool or any
            ``True`` if the line string is fully inside the image area.
            ``False`` otherwise.
            Will return `default` if this line string contains no points.

        """
        if len(self.coords) == 0:
            return default
        return np.all(self.get_pointwise_inside_image_mask(image))

    def is_partly_within_image(self, image: object, default: object = False) -> bool | object:
        """
        Estimate whether the line string is at least partially inside the image.

        Parameters
        ----------
        image : ndarray or tuple of int
            Either an image with shape ``(H,W,[C])`` or a ``tuple`` denoting
            such an image shape.

        default : any
            Default value to return if the line string contains no points.

        Returns
        -------
        bool or any
            ``True`` if the line string is at least partially inside the image
            area. ``False`` otherwise.
            Will return `default` if this line string contains no points.

        """
        if len(self.coords) == 0:
            return default
        # check mask first to avoid costly computation of intersection points
        # whenever possible
        mask = self.get_pointwise_inside_image_mask(image)
        if np.any(mask):
            return True
        return len(self.clip_out_of_image(image)) > 0

    def is_out_of_image(
        self,
        image: object,
        fully: bool = True,
        partly: bool = False,
        default: object = True,
    ) -> bool | object:
        """
        Estimate whether the line is partially/fully outside of the image area.

        Parameters
        ----------
        image : ndarray or tuple of int
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.

        fully : bool, optional
            Whether to return ``True`` if the line string is fully outside
            of the image area.

        partly : bool, optional
            Whether to return ``True`` if the line string is at least partially
            outside fo the image area.

        default : any
            Default value to return if the line string contains no points.

        Returns
        -------
        bool or any
            ``True`` if the line string is partially/fully outside of the image
            area, depending on defined parameters.
            ``False`` otherwise.
            Will return `default` if this line string contains no points.

        """
        if len(self.coords) == 0:
            return default

        if self.is_fully_within_image(image):
            return False
        if self.is_partly_within_image(image):
            return partly
        return fully

    def clip_out_of_image(self, image: object) -> list[LineString]:
        """Clip off all parts of the line string that are outside of the image.

        Parameters
        ----------
        image : ndarray or tuple of int
            Either an image with shape ``(H,W,[C])`` or a ``tuple`` denoting
            such an image shape.

        Returns
        -------
        list of imgaug2.augmentables.lines.LineString
            Line strings, clipped to the image shape.
            The result may contain any number of line strins, including zero.

        """
        if len(self.coords) == 0:
            return []

        inside_image_mask = self.get_pointwise_inside_image_mask(image)
        ooi_mask = ~inside_image_mask

        if len(self.coords) == 1:
            if not np.any(inside_image_mask):
                return []
            return [self.copy()]

        if np.all(inside_image_mask):
            return [self.copy()]

        # top, right, bottom, left image edges
        # we subtract eps here, because intersection() works inclusively,
        # i.e. not subtracting eps would be equivalent to 0<=x<=C for C being
        # height or width
        # don't set the eps too low, otherwise points at height/width seem
        # to get rounded to height/width by shapely, which can cause problems
        # when first clipping and then calling is_fully_within_image()
        # returning false
        height, width = normalize_imglike_shape(image)[0:2]
        eps = 1e-3
        edges = [
            LineString([(0.0, 0.0), (width - eps, 0.0)]),
            LineString([(width - eps, 0.0), (width - eps, height - eps)]),
            LineString([(width - eps, height - eps), (0.0, height - eps)]),
            LineString([(0.0, height - eps), (0.0, 0.0)]),
        ]
        intersections = self.find_intersections_with(edges)

        points = []
        gen = enumerate(
            zip(
                self.coords[:-1],
                self.coords[1:],
                ooi_mask[:-1],
                ooi_mask[1:],
                intersections,
                strict=True,
            )
        )
        for i, (line_start, line_end, ooi_start, ooi_end, inter_line) in gen:
            points.append((line_start, False, ooi_start))
            for p_inter in inter_line:
                points.append((p_inter, True, False))

            is_last = i == len(self.coords) - 2
            if is_last and not ooi_end:
                points.append((line_end, False, ooi_end))

        lines = []
        line = []
        for i, (coord, was_added, ooi) in enumerate(points):
            # remove any point that is outside of the image,
            # also start a new line once such a point is detected
            if ooi:
                if len(line) > 0:
                    lines.append(line)
                    line = []
                continue

            if not was_added:
                # add all points that were part of the original line string
                # AND that are inside the image plane
                line.append(coord)
            else:
                is_last_point = i == len(points) - 1
                # ooi is a numpy.bool_, hence the bool(.)
                is_next_ooi = not is_last_point and bool(points[i + 1][2]) is True

                # Add all points that were new (i.e. intersections), so
                # long that they aren't essentially identical to other point.
                # This prevents adding overlapping intersections multiple times.
                # (E.g. when a line intersects with a corner of the image plane
                # and also with one of its edges.)
                p_prev = line[-1] if len(line) > 0 else None
                # ignore next point if end reached or next point is out of image
                p_next = None
                if not is_last_point and not is_next_ooi:
                    p_next = points[i + 1][0]
                dist_prev = None
                dist_next = None
                if p_prev is not None:
                    dist_prev = np.linalg.norm(np.float32(coord) - np.float32(p_prev))
                if p_next is not None:
                    dist_next = np.linalg.norm(np.float32(coord) - np.float32(p_next))

                dist_prev_ok = dist_prev is None or dist_prev > 1e-2
                dist_next_ok = dist_next is None or dist_next > 1e-2
                if dist_prev_ok and dist_next_ok:
                    line.append(coord)

        if len(line) > 0:
            lines.append(line)

        lines = [line for line in lines if len(line) > 0]
        return [self.deepcopy(coords=line) for line in lines]

    # TODO extend this to non line string geometries
    def find_intersections_with(self, other: object) -> list[list[tuple[float, float]]]:
        """Find all intersection points between this line string and `other`.

        Parameters
        ----------
        other : tuple of number or list of tuple of number or list of LineString or LineString
            The other geometry to use during intersection tests.

        Returns
        -------
        list of list of tuple of number
            All intersection points. One list per pair of consecutive start
            and end point, i.e. `N-1` lists of `N` points. Each list may
            be empty or may contain multiple points.

        """
        import shapely.geometry

        geom = _convert_var_to_shapely_geometry(other)

        result = []
        for p_start, p_end in zip(self.coords[:-1], self.coords[1:], strict=True):
            ls = shapely.geometry.LineString([p_start, p_end])
            intersections = ls.intersection(geom)
            intersections = list(_flatten_shapely_collection(intersections))

            intersections_points = []
            for inter in intersections:
                if isinstance(inter, shapely.geometry.linestring.LineString):
                    # Since shapely 1.7a2 (tested in python 3.8),
                    # .intersection() apprently can return LINE STRING EMPTY
                    # (i.e. .coords is an empty list). Before that, the result
                    # of .intersection() was just []. Hence, we first check
                    # the length here.
                    if len(inter.coords) > 0:
                        inter_start = (inter.coords[0][0], inter.coords[0][1])
                        inter_end = (inter.coords[-1][0], inter.coords[-1][1])
                        intersections_points.extend([inter_start, inter_end])
                else:
                    assert isinstance(inter, shapely.geometry.point.Point), (
                        "Expected to find shapely.geometry.point.Point or "
                        "shapely.geometry.linestring.LineString intersection, "
                        f"actually found {type(inter)}."
                    )
                    intersections_points.append((inter.x, inter.y))

            # sort by distance to start point, this makes it later on easier
            # to remove duplicate points
            inter_sorted = sorted(
                intersections_points,
                key=lambda p, ps=p_start: np.linalg.norm(np.float32(p) - ps),
            )

            result.append(inter_sorted)
        return result

    @legacy(version="0.4.0")
    def shift_(self, x: float = 0, y: float = 0) -> LineString:
        """Move this line string along the x/y-axis in-place.

        The origin ``(0, 0)`` is at the top left of the image.


        Parameters
        ----------
        x : number, optional
            Value to be added to all x-coordinates. Positive values shift
            towards the right images.

        y : number, optional
            Value to be added to all y-coordinates. Positive values shift
            towards the bottom images.

        Returns
        -------
        result : imgaug2.augmentables.lines.LineString
            Shifted line string.
            The object may have been modified in-place.

        """
        self.coords[:, 0] += x
        self.coords[:, 1] += y
        return self

    def shift(
        self,
        x: float = 0,
        y: float = 0,
        top: int | None = None,
        right: int | None = None,
        bottom: int | None = None,
        left: int | None = None,
    ) -> LineString:
        """Move this line string along the x/y-axis.

        The origin ``(0, 0)`` is at the top left of the image.

        Parameters
        ----------
        x : number, optional
            Value to be added to all x-coordinates. Positive values shift
            towards the right images.

        y : number, optional
            Value to be added to all y-coordinates. Positive values shift
            towards the bottom images.

        top : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift this object *from* the
            top (towards the bottom).

        right : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift this object *from* the
            right (towards the left).

        bottom : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift this object *from* the
            bottom (towards the top).

        left : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift this object *from* the
            left (towards the right).

        Returns
        -------
        result : imgaug2.augmentables.lines.LineString
            Shifted line string.

        """
        x, y = _normalize_shift_args(x, y, top=top, right=right, bottom=bottom, left=left)
        return self.deepcopy().shift_(x=x, y=y)

    def draw_mask(
        self,
        image_shape: tuple[int, ...],
        size_lines: int = 1,
        size_points: int = 0,
        raise_if_out_of_image: bool = False,
    ) -> NDArray[np.bool_]:
        """Draw this line segment as a binary image mask.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the line mask.

        size_lines : int, optional
            Thickness of the line segments.

        size_points : int, optional
            Size of the points in pixels.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to ``False``, no error will be
            raised and only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            Boolean line mask of shape `image_shape` (no channel axis).

        """
        heatmap = self.draw_heatmap_array(
            image_shape,
            alpha_lines=1.0,
            alpha_points=1.0,
            size_lines=size_lines,
            size_points=size_points,
            antialiased=False,
            raise_if_out_of_image=raise_if_out_of_image,
        )
        return heatmap > 0.5

    def draw_lines_heatmap_array(
        self,
        image_shape: tuple[int, ...],
        alpha: float = 1.0,
        size: int = 1,
        antialiased: bool = True,
        raise_if_out_of_image: bool = False,
    ) -> NDArray[np.float32]:
        """Draw the line segments of this line string as a heatmap array.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the line mask.

        alpha : float, optional
            Opacity of the line string. Higher values denote a more visible
            line string.

        size : int, optional
            Thickness of the line segments.

        antialiased : bool, optional
            Whether to draw the line with anti-aliasing activated.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to ``False``, no error will be
            raised and only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            ``float32`` array of shape `image_shape` (no channel axis) with
            drawn line string. All values are in the interval ``[0.0, 1.0]``.

        """
        assert len(image_shape) == 2 or (len(image_shape) == 3 and image_shape[-1] == 1), (
            f"Expected (H,W) or (H,W,1) as image_shape, got {image_shape}."
        )

        arr = self.draw_lines_on_image(
            np.zeros(image_shape, dtype=np.uint8),
            color=255,
            alpha=alpha,
            size=size,
            antialiased=antialiased,
            raise_if_out_of_image=raise_if_out_of_image,
        )
        return arr.astype(np.float32) / 255.0

    def draw_points_heatmap_array(
        self,
        image_shape: tuple[int, ...],
        alpha: float = 1.0,
        size: int = 1,
        raise_if_out_of_image: bool = False,
    ) -> NDArray[np.float32]:
        """Draw the points of this line string as a heatmap array.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the point mask.

        alpha : float, optional
            Opacity of the line string points. Higher values denote a more
            visible points.

        size : int, optional
            Size of the points in pixels.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to ``False``, no error will be
            raised and only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            ``float32`` array of shape `image_shape` (no channel axis) with
            drawn line string points. All values are in the
            interval ``[0.0, 1.0]``.

        """
        assert len(image_shape) == 2 or (len(image_shape) == 3 and image_shape[-1] == 1), (
            f"Expected (H,W) or (H,W,1) as image_shape, got {image_shape}."
        )

        arr = self.draw_points_on_image(
            np.zeros(image_shape, dtype=np.uint8),
            color=255,
            alpha=alpha,
            size=size,
            raise_if_out_of_image=raise_if_out_of_image,
        )
        return arr.astype(np.float32) / 255.0

    def draw_heatmap_array(
        self,
        image_shape: tuple[int, ...],
        alpha_lines: float = 1.0,
        alpha_points: float = 1.0,
        size_lines: int = 1,
        size_points: int = 0,
        antialiased: bool = True,
        raise_if_out_of_image: bool = False,
    ) -> NDArray[np.float32]:
        """
        Draw the line segments and points of the line string as a heatmap array.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the line mask.

        alpha_lines : float, optional
            Opacity of the line string. Higher values denote a more visible
            line string.

        alpha_points : float, optional
            Opacity of the line string points. Higher values denote a more
            visible points.

        size_lines : int, optional
            Thickness of the line segments.

        size_points : int, optional
            Size of the points in pixels.

        antialiased : bool, optional
            Whether to draw the line with anti-aliasing activated.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to ``False``, no error will be
            raised and only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            ``float32`` array of shape `image_shape` (no channel axis) with
            drawn line segments and points. All values are in the
            interval ``[0.0, 1.0]``.

        """
        heatmap_lines = self.draw_lines_heatmap_array(
            image_shape,
            alpha=alpha_lines,
            size=size_lines,
            antialiased=antialiased,
            raise_if_out_of_image=raise_if_out_of_image,
        )
        if size_points <= 0:
            return heatmap_lines

        heatmap_points = self.draw_points_heatmap_array(
            image_shape,
            alpha=alpha_points,
            size=size_points,
            raise_if_out_of_image=raise_if_out_of_image,
        )

        heatmap = np.dstack([heatmap_lines, heatmap_points])
        return np.max(heatmap, axis=2)

    def draw_lines_on_image(
        self,
        image: NDArray[np.uint8] | tuple[int, ...],
        color: object = (0, 255, 0),
        alpha: float = 1.0,
        size: int = 3,
        antialiased: bool = True,
        raise_if_out_of_image: bool = False,
        copy: bool = True,
    ) -> NDArray[np.uint8]:
        """Draw the line segments of this line string on a given image.

        Parameters
        ----------
        image : ndarray or tuple of int
            The image onto which to draw.
            Expected to be ``uint8`` and of shape ``(H, W, C)`` with ``C``
            usually being ``3`` (other values are not tested).
            If a tuple, expected to be ``(H, W, C)`` and will lead to a new
            ``uint8`` array of zeros being created.

        color : int or iterable of int
            Color to use as RGB, i.e. three values.

        alpha : float, optional
            Opacity of the line string. Higher values denote a more visible
            line string.

        size : int, optional
            Thickness of the line segments.

        antialiased : bool, optional
            Whether to draw the line with anti-aliasing activated.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to ``False``, no error will be
            raised and only the parts inside the image will be drawn.

        copy : bool, optional
            Whether it is allowed to draw directly in the input array
            (``False``) or it has to be copied (``True``).
            The routine may still have to copy, even if ``copy=False`` was
            used. Always use the return value.

        Returns
        -------
        ndarray
            `image` with line drawn on it.

        """
        import imgaug2.dtypes as iadt

        image_was_empty = False
        if isinstance(image, tuple):
            image_was_empty = True
            image = np.zeros(image, dtype=np.uint8)
        assert image.ndim in [2, 3], (
            f"Expected image or shape of form (H,W) or (H,W,C), got shape {image.shape}."
        )

        if len(self.coords) <= 1 or alpha < 0 + 1e-4 or size < 1:
            return np.copy(image) if (copy and not image_was_empty) else image

        if raise_if_out_of_image and self.is_out_of_image(image, partly=False, fully=True):
            raise Exception(
                f"Cannot draw line string '{self.__str__()}' on image with shape {image.shape}, because "
                "it would be out of bounds."
            )

        if image.ndim == 2:
            assert ia.is_single_number(color), (
                "Got a 2D image. Expected then 'color' to be a single number, "
                f"but got {str(color)}."
            )
            color = [color]
        elif image.ndim == 3 and ia.is_single_number(color):
            color = [color] * image.shape[-1]

        height, width = image.shape[0:2]

        # We can't trivially exclude lines outside of the image here, because
        # even if start and end point are outside, there can still be parts of
        # the line inside the image.
        # TODO Do this with edge-wise intersection tests
        lines = []
        for line_start, line_end in zip(self.coords[:-1], self.coords[1:], strict=True):
            # note that line() expects order (y1, x1, y2, x2), hence ([1], [0])
            lines.append((line_start[1], line_start[0], line_end[1], line_end[0]))

        # skimage.draw.line can only handle integers
        lines = np.round(np.float32(lines)).astype(np.int32)
        if lines.size == 0:
            return np.copy(image) if (copy and not image_was_empty) else image

        min_y = int(np.min(lines[:, [0, 2]]))
        max_y = int(np.max(lines[:, [0, 2]]))
        min_x = int(np.min(lines[:, [1, 3]]))
        max_x = int(np.max(lines[:, [1, 3]]))
        pad = max(int(size), 1)

        roi_y1 = max(min_y - pad, 0)
        roi_y2 = min(max_y + pad, height - 1)
        roi_x1 = max(min_x - pad, 0)
        roi_x2 = min(max_x + pad, width - 1)
        if roi_y1 > roi_y2 or roi_x1 > roi_x2:
            return np.copy(image) if (copy and not image_was_empty) else image

        lines = lines.copy()
        lines[:, [0, 2]] -= roi_y1
        lines[:, [1, 3]] -= roi_x1

        image_out = image if (image_was_empty or not copy) else np.copy(image)
        image_roi = image_out[roi_y1 : roi_y2 + 1, roi_x1 : roi_x2 + 1]
        image_roi_f32 = image_roi.astype(np.float32)
        roi_height, roi_width = image_roi_f32.shape[0:2]

        # size == 0 is already covered above
        # Note here that we have to be careful not to draw lines two times
        # at their intersection points, e.g. for (p0, p1), (p1, 2) we could
        # end up drawing at p1 twice, leading to higher values if alpha is
        # used.
        color = np.float32(color)
        heatmap = np.zeros((roi_height, roi_width), dtype=np.float32)
        for line in lines:
            if antialiased:
                rr, cc, val = skimage.draw.line_aa(*line)
            else:
                rr, cc = skimage.draw.line(*line)
                val = 1.0

            # mask check here, because line() can generate coordinates
            # outside of the image plane
            rr_mask = np.logical_and(0 <= rr, rr < roi_height)
            cc_mask = np.logical_and(0 <= cc, cc < roi_width)
            mask = np.logical_and(rr_mask, cc_mask)

            if np.any(mask):
                rr = rr[mask]
                cc = cc[mask]
                val = val[mask] if not ia.is_single_number(val) else val
                heatmap[rr, cc] = val * alpha

        if size > 1:
            kernel = np.ones((size, size), dtype=np.uint8)
            heatmap = cv2.dilate(heatmap, kernel)

        if image_was_empty:
            image_blend = image_roi_f32 + heatmap * color
        else:
            image_color_shape = image_roi_f32.shape[0:2]
            if image_roi_f32.ndim == 3:
                image_color_shape = image_color_shape + (1,)
            image_color = np.tile(color, image_color_shape)
            image_blend = blend_alpha(image_color, image_roi_f32, heatmap)

        image_blend = cast(NDArray[np.uint8], iadt.restore_dtypes_(image_blend, np.uint8))
        image_out[roi_y1 : roi_y2 + 1, roi_x1 : roi_x2 + 1] = image_blend
        return image_out

    def draw_points_on_image(
        self,
        image: NDArray[np.uint8],
        color: object = (0, 128, 0),
        alpha: float = 1.0,
        size: int = 3,
        copy: bool = True,
        raise_if_out_of_image: bool = False,
    ) -> NDArray[np.uint8]:
        """Draw the points of this line string onto a given image.

        Parameters
        ----------
        image : ndarray or tuple of int
            The image onto which to draw.
            Expected to be ``uint8`` and of shape ``(H, W, C)`` with ``C``
            usually being ``3`` (other values are not tested).
            If a tuple, expected to be ``(H, W, C)`` and will lead to a new
            ``uint8`` array of zeros being created.

        color : iterable of int
            Color to use as RGB, i.e. three values.

        alpha : float, optional
            Opacity of the line string points. Higher values denote a more
            visible points.

        size : int, optional
            Size of the points in pixels.

        copy : bool, optional
            Whether it is allowed to draw directly in the input
            array (``False``) or it has to be copied (``True``).
            The routine may still have to copy, even if ``copy=False`` was
            used. Always use the return value.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to ``False``, no error will be
            raised and only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            ``float32`` array of shape `image_shape` (no channel axis) with
            drawn line string points. All values are in the
            interval ``[0.0, 1.0]``.

        """
        from imgaug2.augmentables.kps import KeypointsOnImage

        kpsoi = KeypointsOnImage.from_xy_array(self.coords, shape=image.shape)
        image = kpsoi.draw_on_image(
            image,
            color=color,
            alpha=alpha,
            size=size,
            copy=copy,
            raise_if_out_of_image=raise_if_out_of_image,
        )

        return image

    def draw_on_image(
        self,
        image: NDArray[np.uint8],
        color: object = (0, 255, 0),
        color_lines: object | None = None,
        color_points: object | None = None,
        alpha: float = 1.0,
        alpha_lines: float | None = None,
        alpha_points: float | None = None,
        size: int = 1,
        size_lines: int | None = None,
        size_points: int | None = None,
        antialiased: bool = True,
        raise_if_out_of_image: bool = False,
        copy: bool = True,
    ) -> NDArray[np.uint8]:
        """Draw this line string onto an image.

        Parameters
        ----------
        image : ndarray
            The `(H,W,C)` `uint8` image onto which to draw the line string.

        color : iterable of int, optional
            Color to use as RGB, i.e. three values.
            The color of the line and points are derived from this value,
            unless they are set.

        color_lines : None or iterable of int
            Color to use for the line segments as RGB, i.e. three values.
            If ``None``, this value is derived from `color`.

        color_points : None or iterable of int
            Color to use for the points as RGB, i.e. three values.
            If ``None``, this value is derived from ``0.5 * color``.

        alpha : float, optional
            Opacity of the line string. Higher values denote more visible
            points.
            The alphas of the line and points are derived from this value,
            unless they are set.

        alpha_lines : None or float, optional
            Opacity of the line string. Higher values denote more visible
            line string.
            If ``None``, this value is derived from `alpha`.

        alpha_points : None or float, optional
            Opacity of the line string points. Higher values denote more
            visible points.
            If ``None``, this value is derived from `alpha`.

        size : int, optional
            Size of the line string.
            The sizes of the line and points are derived from this value,
            unless they are set.

        size_lines : None or int, optional
            Thickness of the line segments.
            If ``None``, this value is derived from `size`.

        size_points : None or int, optional
            Size of the points in pixels.
            If ``None``, this value is derived from ``3 * size``.

        antialiased : bool, optional
            Whether to draw the line with anti-aliasing activated.
            This does currently not affect the point drawing.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to ``False``, no error will be
            raised and only the parts inside the image will be drawn.

        copy : bool, optional
            Whether it is allowed to draw directly in the input array
            (``False``) or it has to be copied (``True``).
            The routine may still have to copy, even if ``copy=False`` was
            used. Always use the return value.

        Returns
        -------
        ndarray
            Image with line string drawn on it.

        """

        def _assert_not_none(arg_name: str, arg_value: object) -> None:
            assert arg_value is not None, (
                f"Expected '{arg_name}' to not be None, got type {type(arg_value)}."
            )

        _assert_not_none("color", color)
        _assert_not_none("alpha", alpha)
        _assert_not_none("size", size)

        color_lines = color_lines if color_lines is not None else np.float32(color)
        color_points = color_points if color_points is not None else np.float32(color) * 0.5

        alpha_lines = alpha_lines if alpha_lines is not None else np.float32(alpha)
        alpha_points = alpha_points if alpha_points is not None else np.float32(alpha)

        size_lines = size_lines if size_lines is not None else size
        size_points = size_points if size_points is not None else size * 3

        image = self.draw_lines_on_image(
            image,
            color=np.array(color_lines).astype(np.uint8),
            alpha=alpha_lines,
            size=size_lines,
            antialiased=antialiased,
            raise_if_out_of_image=raise_if_out_of_image,
            copy=copy,
        )

        image = self.draw_points_on_image(
            image,
            color=np.array(color_points).astype(np.uint8),
            alpha=alpha_points,
            size=size_points,
            copy=False,
            raise_if_out_of_image=raise_if_out_of_image,
        )

        return image

    def extract_from_image(
        self,
        image: NDArray[np.generic],
        size: int = 1,
        pad: bool = True,
        pad_max: int | None = None,
        antialiased: bool = True,
        prevent_zero_size: bool = True,
    ) -> NDArray[np.generic]:
        """Extract all image pixels covered by the line string.

        This will only extract pixels overlapping with the line string.
        As a rectangular image array has to be returned, non-overlapping
        pixels will be set to zero.

        This function will by default zero-pad the image if the line string is
        partially/fully outside of the image. This is for consistency with
        the same methods for bounding boxes and polygons.

        Parameters
        ----------
        image : ndarray
            The image of shape `(H,W,[C])` from which to extract the pixels
            within the line string.

        size : int, optional
            Thickness of the line.

        pad : bool, optional
            Whether to zero-pad the image if the object is partially/fully
            outside of it.

        pad_max : None or int, optional
            The maximum number of pixels that may be zero-paded on any side,
            i.e. if this has value ``N`` the total maximum of added pixels
            is ``4*N``.
            This option exists to prevent extremely large images as a result of
            single points being moved very far away during augmentation.

        antialiased : bool, optional
            Whether to apply anti-aliasing to the line string.

        prevent_zero_size : bool, optional
            Whether to prevent height or width of the extracted image from
            becoming zero. If this is set to ``True`` and height or width of
            the line string is below ``1``, the height/width will be increased
            to ``1``. This can be useful to prevent problems, e.g. with image
            saving or plotting. If it is set to ``False``, images will be
            returned as ``(H', W')`` or ``(H', W', 3)`` with ``H`` or ``W``
            potentially being ``0``.

        Returns
        -------
        (H',W') ndarray or (H',W',C) ndarray
            Pixels overlapping with the line string. Zero-padded if the
            line string is partially/fully outside of the image and
            ``pad=True``. If `prevent_zero_size` is activated, it is
            guarantueed that ``H'>0`` and ``W'>0``, otherwise only
            ``H'>=0`` and ``W'>=0``.

        """
        from imgaug2.augmentables.bbs import BoundingBox

        assert image.ndim in [2, 3], f"Expected image of shape (H,W,[C]), got shape {image.shape}."

        if len(self.coords) == 0 or size <= 0:
            if prevent_zero_size:
                return np.zeros((1, 1) + image.shape[2:], dtype=image.dtype)
            return np.zeros((0, 0) + image.shape[2:], dtype=image.dtype)

        xx = self.xx_int
        yy = self.yy_int

        # this would probably work if drawing was subpixel-accurate
        # x1 = np.min(self.coords[:, 0]) - (size / 2)
        # y1 = np.min(self.coords[:, 1]) - (size / 2)
        # x2 = np.max(self.coords[:, 0]) + (size / 2)
        # y2 = np.max(self.coords[:, 1]) + (size / 2)

        # this works currently with non-subpixel-accurate drawing
        sizeh = (size - 1) / 2
        x1 = np.min(xx) - sizeh
        y1 = np.min(yy) - sizeh
        x2 = np.max(xx) + 1 + sizeh
        y2 = np.max(yy) + 1 + sizeh
        bb = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

        if len(self.coords) == 1:
            return bb.extract_from_image(
                image, pad=pad, pad_max=pad_max, prevent_zero_size=prevent_zero_size
            )

        heatmap = self.draw_lines_heatmap_array(
            image.shape[0:2], alpha=1.0, size=size, antialiased=antialiased
        )
        if image.ndim == 3:
            heatmap = np.atleast_3d(heatmap)
        image_masked = image.astype(np.float32) * heatmap
        extract = bb.extract_from_image(
            image_masked, pad=pad, pad_max=pad_max, prevent_zero_size=prevent_zero_size
        )
        return np.clip(np.round(extract), 0, 255).astype(np.uint8)

    def concatenate(
        self,
        other: LineString | tuple[float, float] | Sequence[Sequence[float]] | NDArray[np.number],
    ) -> LineString:
        """Concatenate this line string with another one.

        This will add a line segment between the end point of this line string
        and the start point of `other`.

        Parameters
        ----------
        other : imgaug2.augmentables.lines.LineString or ndarray or iterable of tuple of number
            The points to add to this line string.

        Returns
        -------
        imgaug2.augmentables.lines.LineString
            New line string with concatenated points.
            The `label` of this line string will be kept.

        """
        if not isinstance(other, LineString):
            other = LineString(other)
        return self.deepcopy(coords=np.concatenate([self.coords, other.coords], axis=0))

    def subdivide(self, points_per_edge: int) -> LineString:
        """Derive a new line string with ``N`` interpolated points per edge.

        The interpolated points have (per edge) regular distances to each
        other.

        For each edge between points ``A`` and ``B`` this adds points
        at ``A + (i/(1+N)) * (B - A)``, where ``i`` is the index of the added
        point and ``N`` is the number of points to add per edge.

        Calling this method two times will split each edge at its center
        and then again split each newly created edge at their center.
        It is equivalent to calling `subdivide(3)`.

        Parameters
        ----------
        points_per_edge : int
            Number of points to interpolate on each edge.

        Returns
        -------
        imgaug2.augmentables.lines.LineString
            Line string with subdivided edges.

        """
        if len(self.coords) <= 1 or points_per_edge < 1:
            return self.deepcopy()
        coords = interpolate_points(self.coords, nb_steps=points_per_edge, closed=False)
        return self.deepcopy(coords=coords)

    def to_keypoints(self) -> list[Keypoint]:
        """Convert the line string points to keypoints.

        Returns
        -------
        list of imgaug2.augmentables.kps.Keypoint
            Points of the line string as keypoints.

        """
        from imgaug2.augmentables import kps

        return [kps.Keypoint(x=x, y=y) for (x, y) in self.coords]

    def to_bounding_box(self) -> BoundingBox | None:
        """Generate a bounding box encapsulating the line string.

        Returns
        -------
        None or imgaug2.augmentables.bbs.BoundingBox
            Bounding box encapsulating the line string.
            ``None`` if the line string contained no points.

        """
        from imgaug2.augmentables import bbs

        # we don't have to mind the case of len(.) == 1 here, because
        # zero-sized BBs are considered valid
        if len(self.coords) == 0:
            return None
        return bbs.BoundingBox(
            x1=np.min(self.xx),
            y1=np.min(self.yy),
            x2=np.max(self.xx),
            y2=np.max(self.yy),
            label=self.label,
        )

    def to_polygon(self) -> Polygon:
        """Generate a polygon from the line string points.

        Returns
        -------
        imgaug2.augmentables.polys.Polygon
            Polygon with the same corner points as the line string.
            Note that the polygon might be invalid, e.g. contain less
            than ``3`` points or have self-intersections.

        """
        from imgaug2.augmentables.polys import Polygon

        return Polygon(self.coords, label=self.label)

    def to_heatmap(
        self,
        image_shape: tuple[int, ...],
        size_lines: int = 1,
        size_points: int = 0,
        antialiased: bool = True,
        raise_if_out_of_image: bool = False,
    ) -> HeatmapsOnImage:
        """Generate a heatmap object from the line string.

        This is similar to
        :func:`~imgaug2.augmentables.lines.LineString.draw_lines_heatmap_array`,
        executed with ``alpha=1.0``. The result is wrapped in a
        :class:`~imgaug2.augmentables.heatmaps.HeatmapsOnImage` object instead
        of just an array. No points are drawn.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the line mask.

        size_lines : int, optional
            Thickness of the line.

        size_points : int, optional
            Size of the points in pixels.

        antialiased : bool, optional
            Whether to draw the line with anti-aliasing activated.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to ``False``, no error will be
            raised and only the parts inside the image will be drawn.

        Returns
        -------
        imgaug2.augmentables.heatmaps.HeatmapsOnImage
            Heatmap object containing drawn line string.

        """
        from imgaug2.augmentables.heatmaps import HeatmapsOnImage

        return HeatmapsOnImage(
            self.draw_heatmap_array(
                image_shape,
                size_lines=size_lines,
                size_points=size_points,
                antialiased=antialiased,
                raise_if_out_of_image=raise_if_out_of_image,
            ),
            shape=image_shape,
        )

    def to_segmentation_map(
        self,
        image_shape: tuple[int, ...],
        size_lines: int = 1,
        size_points: int = 0,
        raise_if_out_of_image: bool = False,
    ) -> SegmentationMapsOnImage:
        """Generate a segmentation map object from the line string.

        This is similar to
        :func:`~imgaug2.augmentables.lines.LineString.draw_mask`.
        The result is wrapped in a ``SegmentationMapsOnImage`` object
        instead of just an array.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the line mask.

        size_lines : int, optional
            Thickness of the line.

        size_points : int, optional
            Size of the points in pixels.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to ``False``, no error will be
            raised and only the parts inside the image will be drawn.

        Returns
        -------
        imgaug2.augmentables.segmaps.SegmentationMapsOnImage
            Segmentation map object containing drawn line string.

        """
        from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

        return SegmentationMapsOnImage(
            self.draw_mask(
                image_shape,
                size_lines=size_lines,
                size_points=size_points,
                raise_if_out_of_image=raise_if_out_of_image,
            ),
            shape=image_shape,
        )

    # TODO make this non-approximate
    def coords_almost_equals(
        self, other: object, max_distance: float = 1e-4, points_per_edge: int = 8
    ) -> bool:
        """Compare this and another LineString's coordinates.

        This is an approximate method based on pointwise distances and can
        in rare corner cases produce wrong outputs.

        Parameters
        ----------
        other : imgaug2.augmentables.lines.LineString or tuple of number or ndarray or list of ndarray or list of tuple of number
            The other line string or its coordinates.

        max_distance : float, optional
            Max distance of any point from the other line string before
            the two line strings are evaluated to be unequal.

        points_per_edge : int, optional
            How many points to interpolate on each edge.

        Returns
        -------
        bool
            Whether the two LineString's coordinates are almost identical,
            i.e. the max distance is below the threshold.
            If both have no coordinates, ``True`` is returned.
            If only one has no coordinates, ``False`` is returned.
            Beyond that, the number of points is not evaluated.

        """
        if isinstance(other, LineString):
            pass
        elif isinstance(other, tuple):
            other = LineString([other])
        else:
            other = LineString(other)

        if len(self.coords) == 0 and len(other.coords) == 0:
            return True
        if 0 in [len(self.coords), len(other.coords)]:
            # only one of the two line strings has no coords
            return False

        self_subd = self.subdivide(points_per_edge)
        other_subd = other.subdivide(points_per_edge)

        dist_self2other = self_subd.compute_pointwise_distances(other_subd)
        dist_other2self = other_subd.compute_pointwise_distances(self_subd)
        dist = max(np.max(dist_self2other), np.max(dist_other2self))
        return dist < max_distance

    def almost_equals(
        self, other: LineString, max_distance: float = 1e-4, points_per_edge: int = 8
    ) -> bool:
        """Compare this and another line string.

        Parameters
        ----------
        other: imgaug2.augmentables.lines.LineString
            The other object to compare against. Expected to be a
            ``LineString``.

        max_distance : float, optional
            See :func:`~imgaug2.augmentables.lines.LineString.coords_almost_equals`.

        points_per_edge : int, optional
            See :func:`~imgaug2.augmentables.lines.LineString.coords_almost_equals`.

        Returns
        -------
        bool
            ``True`` if the coordinates are almost equal and additionally
            the labels are equal. Otherwise ``False``.

        """
        if self.label != other.label:
            return False
        return self.coords_almost_equals(
            other, max_distance=max_distance, points_per_edge=points_per_edge
        )

    def copy(
        self,
        coords: Sequence[Sequence[float]] | NDArray[np.number] | None = None,
        label: str | None = None,
    ) -> LineString:
        """Create a shallow copy of this line string.

        Parameters
        ----------
        coords : None or iterable of tuple of number or ndarray
            If not ``None``, then the coords of the copied object will be set
            to this value.

        label : None or str
            If not ``None``, then the label of the copied object will be set to
            this value.

        Returns
        -------
        imgaug2.augmentables.lines.LineString
            Shallow copy.

        """
        return LineString(
            coords=self.coords if coords is None else coords,
            label=self.label if label is None else label,
        )

    def deepcopy(
        self,
        coords: Sequence[Sequence[float]] | NDArray[np.number] | None = None,
        label: str | None = None,
    ) -> LineString:
        """Create a deep copy of this line string.

        Parameters
        ----------
        coords : None or iterable of tuple of number or ndarray
            If not ``None``, then the coords of the copied object will be set
            to this value.

        label : None or str
            If not ``None``, then the label of the copied object will be set to
            this value.

        Returns
        -------
        imgaug2.augmentables.lines.LineString
            Deep copy.

        """
        return LineString(
            coords=np.copy(self.coords) if coords is None else coords,
            label=copylib.deepcopy(self.label) if label is None else label,
        )

    @legacy(version="0.4.0")
    def __getitem__(self, indices: int | slice | Sequence[int]) -> NDArray[np.float32]:
        """Get the coordinate(s) with given indices.


        Returns
        -------
        ndarray
            xy-coordinate(s) as ``ndarray``.

        """
        return self.coords[indices]

    @legacy(version="0.4.0")
    def __iter__(self) -> Iterator[NDArray[np.float32]]:
        """Iterate over the coordinates of this instance.


        Yields
        ------
        ndarray
            An ``(2,)`` ``ndarray`` denoting an xy-coordinate pair.

        """
        return iter(self.coords)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        points_str = ", ".join([f"({x:.2f}, {y:.2f})" for x, y in self.coords])
        return f"LineString([{points_str}], label={self.label})"


class LineStringsOnImage:
    """Object that represents all line strings on a single image.

    Parameters
    ----------
    line_strings : list of imgaug2.augmentables.lines.LineString
        List of line strings on the image.

    shape : tuple of int or ndarray
        The shape of the image on which the objects are placed.
        Either an image with shape ``(H,W,[C])`` or a ``tuple`` denoting
        such an image shape.

    Examples
    --------
    >>> import numpy as np
    >>> from imgaug2.augmentables.lines import LineString, LineStringsOnImage
    >>>
    >>> image = np.zeros((100, 100))
    >>> lss = [
    >>>     LineString([(0, 0), (10, 0)]),
    >>>     LineString([(10, 20), (30, 30), (50, 70)])
    >>> ]
    >>> lsoi = LineStringsOnImage(lss, shape=image.shape)

    """

    def __init__(
        self,
        line_strings: Sequence[LineString],
        shape: Shape | NDArray[np.generic],
    ) -> None:
        assert ia.is_iterable(line_strings), (
            f"Expected 'line_strings' to be an iterable, got type '{type(line_strings)}'."
        )
        line_strings_list = list(line_strings)
        assert all([isinstance(v, LineString) for v in line_strings_list]), (
            "Expected iterable of LineString, got types: {}.".format(
                ", ".join([str(type(v)) for v in line_strings_list])
            )
        )
        self.line_strings = line_strings_list
        self.shape = _handle_on_image_shape(shape, self)

    @legacy(version="0.4.0")
    @property
    def items(self) -> list[LineString]:
        """Get the line strings in this container.


        Returns
        -------
        list of LineString
            Line strings within this container.

        """
        return self.line_strings

    @legacy(version="0.4.0")
    @items.setter
    def items(self, value: list[LineString]) -> None:
        """Set the line strings in this container.


        Parameters
        ----------
        value : list of LineString
            Line strings within this container.

        """
        self.line_strings = value

    @property
    def empty(self) -> bool:
        """Estimate whether this object contains zero line strings.

        Returns
        -------
        bool
            ``True`` if this object contains zero line strings.

        """
        return len(self.line_strings) == 0

    @legacy(version="0.4.0")
    def on_(self, image: Shape | NDArray[np.generic]) -> LineStringsOnImage:
        """Project the line strings from one image shape to a new one in-place.


        Parameters
        ----------
        image : ndarray or tuple of int
            The new image onto which to project.
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.

        Returns
        -------
        imgaug2.augmentables.lines.LineStrings
            Object containing all projected line strings.
            The object and its items may have been modified in-place.

        """
        on_shape = normalize_imglike_shape(image)
        if on_shape[0:2] == self.shape[0:2]:
            self.shape = on_shape  # channels may differ
            return self

        for i, item in enumerate(self.items):
            self.line_strings[i] = item.project_(self.shape, on_shape)
        self.shape = on_shape
        return self

    def on(self, image: Shape | NDArray[np.generic]) -> LineStringsOnImage:
        """Project the line strings from one image shape to a new one.

        Parameters
        ----------
        image : ndarray or tuple of int
            The new image onto which to project.
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.

        Returns
        -------
        imgaug2.augmentables.lines.LineStrings
            Object containing all projected line strings.

        """
        return self.deepcopy().on_(image)

    @classmethod
    def from_xy_arrays(
        cls,
        xy: NDArray[np.number] | Iterable[NDArray[np.number]],
        shape: Shape | NDArray[np.generic],
    ) -> LineStringsOnImage:
        """Convert an ``(N,M,2)`` ndarray to a ``LineStringsOnImage`` object.

        This is the inverse of
        :func:`~imgaug2.augmentables.lines.LineStringsOnImage.to_xy_array`.

        Parameters
        ----------
        xy : (N,M,2) ndarray or iterable of (M,2) ndarray
            Array containing the point coordinates ``N`` line strings
            with each ``M`` points given as ``(x,y)`` coordinates.
            ``M`` may differ if an iterable of arrays is used.
            Each array should usually be of dtype ``float32``.

        shape : tuple of int
            ``(H,W,[C])`` shape of the image on which the line strings are
            placed.

        Returns
        -------
        imgaug2.augmentables.lines.LineStringsOnImage
            Object containing a list of ``LineString`` objects following the
            provided point coordinates.

        """
        lss = []
        for xy_ls in xy:
            lss.append(LineString(xy_ls))
        return cls(lss, shape)

    def to_xy_arrays(self, dtype: object = np.float32) -> list[NDArray[np.generic]]:
        """Convert this object to an iterable of ``(M,2)`` arrays of points.

        This is the inverse of
        :func:`~imgaug2.augmentables.lines.LineStringsOnImage.from_xy_array`.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            Desired output datatype of the ndarray.

        Returns
        -------
        list of ndarray
            The arrays of point coordinates, each given as ``(M,2)``.

        """
        import imgaug2.dtypes as iadt

        return [
            cast(NDArray[np.generic], iadt.restore_dtypes_(np.copy(ls.coords), dtype))
            for ls in self.line_strings
        ]

    def draw_on_image(
        self,
        image: NDArray[np.uint8],
        color: object = (0, 255, 0),
        color_lines: object | None = None,
        color_points: object | None = None,
        alpha: float = 1.0,
        alpha_lines: float | None = None,
        alpha_points: float | None = None,
        size: int = 1,
        size_lines: int | None = None,
        size_points: int | None = None,
        antialiased: bool = True,
        raise_if_out_of_image: bool = False,
    ) -> NDArray[np.uint8]:
        """Draw all line strings onto a given image.

        Parameters
        ----------
        image : ndarray
            The ``(H,W,C)`` ``uint8`` image onto which to draw the line
            strings.

        color : iterable of int, optional
            Color to use as RGB, i.e. three values.
            The color of the lines and points are derived from this value,
            unless they are set.

        color_lines : None or iterable of int
            Color to use for the line segments as RGB, i.e. three values.
            If ``None``, this value is derived from `color`.

        color_points : None or iterable of int
            Color to use for the points as RGB, i.e. three values.
            If ``None``, this value is derived from ``0.5 * color``.

        alpha : float, optional
            Opacity of the line strings. Higher values denote more visible
            points.
            The alphas of the line and points are derived from this value,
            unless they are set.

        alpha_lines : None or float, optional
            Opacity of the line strings. Higher values denote more visible
            line string.
            If ``None``, this value is derived from `alpha`.

        alpha_points : None or float, optional
            Opacity of the line string points. Higher values denote more
            visible points.
            If ``None``, this value is derived from `alpha`.

        size : int, optional
            Size of the line strings.
            The sizes of the line and points are derived from this value,
            unless they are set.

        size_lines : None or int, optional
            Thickness of the line segments.
            If ``None``, this value is derived from `size`.

        size_points : None or int, optional
            Size of the points in pixels.
            If ``None``, this value is derived from ``3 * size``.

        antialiased : bool, optional
            Whether to draw the lines with anti-aliasing activated.
            This does currently not affect the point drawing.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if a line string is fully
            outside of the image. If set to ``False``, no error will be
            raised and only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            Image with line strings drawn on it.

        """
        if len(self.line_strings) == 0:
            return image

        image_out = np.copy(image)
        for ls in self.line_strings:
            image_out = ls.draw_on_image(
                image_out,
                color=color,
                color_lines=color_lines,
                color_points=color_points,
                alpha=alpha,
                alpha_lines=alpha_lines,
                alpha_points=alpha_points,
                size=size,
                size_lines=size_lines,
                size_points=size_points,
                antialiased=antialiased,
                raise_if_out_of_image=raise_if_out_of_image,
                copy=False,
            )

        return image_out

    @legacy(version="0.4.0")
    def remove_out_of_image_(self, fully: bool = True, partly: bool = False) -> LineStringsOnImage:
        """
        Remove all LS that are fully/partially outside of an image in-place.


        Parameters
        ----------
        fully : bool, optional
            Whether to remove line strings that are fully outside of the image.

        partly : bool, optional
            Whether to remove line strings that are partially outside of the
            image.

        Returns
        -------
        imgaug2.augmentables.lines.LineStringsOnImage
            Reduced set of line strings. Those that are fully/partially
            outside of the given image plane are removed.
            The object and its items may have been modified in-place.

        """
        self.line_strings = [
            ls
            for ls in self.line_strings
            if not ls.is_out_of_image(self.shape, fully=fully, partly=partly)
        ]
        return self

    def remove_out_of_image(self, fully: bool = True, partly: bool = False) -> LineStringsOnImage:
        """
        Remove all line strings that are fully/partially outside of an image.

        Parameters
        ----------
        fully : bool, optional
            Whether to remove line strings that are fully outside of the image.

        partly : bool, optional
            Whether to remove line strings that are partially outside of the
            image.

        Returns
        -------
        imgaug2.augmentables.lines.LineStringsOnImage
            Reduced set of line strings. Those that are fully/partially
            outside of the given image plane are removed.

        """
        return self.copy().remove_out_of_image_(fully=fully, partly=partly)

    @legacy(version="0.4.0")
    def remove_out_of_image_fraction_(self, fraction: float) -> LineStringsOnImage:
        """Remove all LS with an OOI fraction of at least `fraction` in-place.

        'OOI' is the abbreviation for 'out of image'.


        Parameters
        ----------
        fraction : number
            Minimum out of image fraction that a line string has to have in
            order to be removed. A fraction of ``1.0`` removes only line
            strings that are ``100%`` outside of the image. A fraction of
            ``0.0`` removes all line strings.

        Returns
        -------
        imgaug2.augmentables.lines.LineStringsOnImage
            Reduced set of line strings, with those that had an out of image
            fraction greater or equal the given one removed.
            The object and its items may have been modified in-place.

        """
        return _remove_out_of_image_fraction_(self, fraction)

    def remove_out_of_image_fraction(self, fraction: float) -> LineStringsOnImage:
        """Remove all LS with an out of image fraction of at least `fraction`.

        Parameters
        ----------
        fraction : number
            Minimum out of image fraction that a line string has to have in
            order to be removed. A fraction of ``1.0`` removes only line
            strings that are ``100%`` outside of the image. A fraction of
            ``0.0`` removes all line strings.

        Returns
        -------
        imgaug2.augmentables.lines.LineStringsOnImage
            Reduced set of line strings, with those that had an out of image
            fraction greater or equal the given one removed.

        """
        return self.copy().remove_out_of_image_fraction_(fraction)

    @legacy(version="0.4.0")
    def clip_out_of_image_(self) -> LineStringsOnImage:
        """
        Clip off all parts of the LSs that are outside of an image in-place.

        .. note::

            The result can contain fewer line strings than the input did. That
            happens when a polygon is fully outside of the image plane.

        .. note::

            The result can also contain *more* line strings than the input
            did. That happens when distinct parts of a line string are only
            connected by line segments that are outside of the image plane and
            hence will be clipped off, resulting in two or more unconnected
            line string parts that are left in the image plane.


        Returns
        -------
        imgaug2.augmentables.lines.LineStringsOnImage
            Line strings, clipped to fall within the image dimensions.
            The count of output line strings may differ from the input count.

        """
        self.line_strings = [
            ls_clipped
            for ls in self.line_strings
            for ls_clipped in ls.clip_out_of_image(self.shape)
        ]
        return self

    def clip_out_of_image(self) -> LineStringsOnImage:
        """
        Clip off all parts of the line strings that are outside of an image.

        .. note::

            The result can contain fewer line strings than the input did. That
            happens when a polygon is fully outside of the image plane.

        .. note::

            The result can also contain *more* line strings than the input
            did. That happens when distinct parts of a line string are only
            connected by line segments that are outside of the image plane and
            hence will be clipped off, resulting in two or more unconnected
            line string parts that are left in the image plane.

        Returns
        -------
        imgaug2.augmentables.lines.LineStringsOnImage
            Line strings, clipped to fall within the image dimensions.
            The count of output line strings may differ from the input count.

        """
        return self.copy().clip_out_of_image_()

    @legacy(version="0.4.0")
    def shift_(self, x: Number = 0, y: Number = 0) -> LineStringsOnImage:
        """Move the line strings along the x/y-axis in-place.

        The origin ``(0, 0)`` is at the top left of the image.


        Parameters
        ----------
        x : number, optional
            Value to be added to all x-coordinates. Positive values shift
            towards the right images.

        y : number, optional
            Value to be added to all y-coordinates. Positive values shift
            towards the bottom images.

        Returns
        -------
        imgaug2.augmentables.lines.LineStringsOnImage
            Shifted line strings.
            The object and its items may have been modified in-place.

        """
        for i, ls in enumerate(self.line_strings):
            self.line_strings[i] = ls.shift_(x=x, y=y)
        return self

    def shift(
        self,
        x: Number = 0,
        y: Number = 0,
        top: int | None = None,
        right: int | None = None,
        bottom: int | None = None,
        left: int | None = None,
    ) -> LineStringsOnImage:
        """Move the line strings along the x/y-axis.

        The origin ``(0, 0)`` is at the top left of the image.

        Parameters
        ----------
        x : number, optional
            Value to be added to all x-coordinates. Positive values shift
            towards the right images.

        y : number, optional
            Value to be added to all y-coordinates. Positive values shift
            towards the bottom images.

        top : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift all objects *from* the
            top (towards the bottom).

        right : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift all objects *from* the
            right (towads the left).

        bottom : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift all objects *from* the
            bottom (towards the top).

        left : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift all objects *from* the
            left (towards the right).

        Returns
        -------
        imgaug2.augmentables.lines.LineStringsOnImage
            Shifted line strings.

        """
        x, y = _normalize_shift_args(x, y, top=top, right=right, bottom=bottom, left=left)
        return self.deepcopy().shift_(x=x, y=y)

    @legacy(version="0.4.0")
    def to_xy_array(self) -> NDArray[np.float32]:
        """Convert all line string coordinates to one array of shape ``(N,2)``.


        Returns
        -------
        (N, 2) ndarray
            Array containing all xy-coordinates of all line strings within this
            instance.

        """
        if self.empty:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate([ls.coords for ls in self.line_strings])

    @legacy(version="0.4.0")
    def fill_from_xy_array_(
        self,
        xy: Sequence[Sequence[Number]] | NDArray[np.number],
    ) -> LineStringsOnImage:
        """Modify the corner coordinates of all line strings in-place.

        .. note::

            This currently expects that `xy` contains exactly as many
            coordinates as the line strings within this instance have corner
            points. Otherwise, an ``AssertionError`` will be raised.


        Parameters
        ----------
        xy : (N, 2) ndarray or iterable of iterable of number
            XY-Coordinates of ``N`` corner points. ``N`` must match the
            number of corner points in all line strings within this instance.

        Returns
        -------
        LineStringsOnImage
            This instance itself, with updated coordinates.
            Note that the instance was modified in-place.

        """
        xy = np.array(xy, dtype=np.float32)

        # note that np.array([]) is (0,), not (0, 2)
        assert xy.shape[0] == 0 or (xy.ndim == 2 and xy.shape[-1] == 2), (
            f"Expected input array to have shape (N,2), got shape {xy.shape}."
        )

        counter = 0
        for ls in self.line_strings:
            nb_points = len(ls.coords)
            nb_points_exp = sum([len(ls_.coords) for ls_ in self.line_strings])
            assert counter + nb_points <= len(xy), (
                "Received fewer points than there are corner points in all line strings. "
                f"Got {len(xy)} points, expected {nb_points_exp}."
            )

            ls.coords[:, ...] = xy[counter : counter + nb_points]
            counter += nb_points

        assert counter == len(xy), (
            "Expected to get exactly as many xy-coordinates as there are "
            "points in all line strings polygons within this instance. "
            f"Got {len(xy)} points, could only assign {counter} points."
        )

        return self

    @legacy(version="0.4.0")
    def to_keypoints_on_image(self) -> KeypointsOnImage:
        """Convert the line strings to one ``KeypointsOnImage`` instance.


        Returns
        -------
        imgaug2.augmentables.kps.KeypointsOnImage
            A keypoints instance containing ``N`` coordinates for a total
            of ``N`` points in the ``coords`` attributes of all line strings.
            Order matches the order in ``line_strings`` and ``coords``
            attributes.

        """
        from imgaug2.augmentables.kps import KeypointsOnImage

        if self.empty:
            return KeypointsOnImage([], shape=self.shape)
        coords = np.concatenate([ls.coords for ls in self.line_strings], axis=0)
        return KeypointsOnImage.from_xy_array(coords, shape=self.shape)

    @legacy(version="0.4.0")
    def invert_to_keypoints_on_image_(self, kpsoi: KeypointsOnImage) -> LineStringsOnImage:
        """Invert the output of ``to_keypoints_on_image()`` in-place.

        This function writes in-place into this ``LineStringsOnImage``
        instance.


        Parameters
        ----------
        kpsoi : imgaug2.augmentables.kps.KeypointsOnImages
            Keypoints to convert back to line strings, i.e. the outputs
            of ``to_keypoints_on_image()``.

        Returns
        -------
        LineStringsOnImage
            Line strings container with updated coordinates.
            Note that the instance is also updated in-place.

        """
        lss = self.line_strings
        coordss = [ls.coords for ls in lss]
        nb_points_exp = sum([len(coords) for coords in coordss])
        assert len(kpsoi.keypoints) == nb_points_exp, (
            f"Expected {nb_points_exp} coordinates, got {len(kpsoi.keypoints)}."
        )

        xy_arr = kpsoi.to_xy_array()

        counter = 0
        for ls in lss:
            coords = ls.coords
            coords[:, :] = xy_arr[counter : counter + len(coords), :]
            counter += len(coords)
        self.shape = kpsoi.shape
        return self

    def copy(
        self,
        line_strings: list[LineString] | None = None,
        shape: Shape | NDArray[np.generic] | None = None,
    ) -> LineStringsOnImage:
        """Create a shallow copy of this object.

        Parameters
        ----------
        line_strings : None or list of imgaug2.augmentables.lines.LineString, optional
            List of line strings on the image.
            If not ``None``, then the ``line_strings`` attribute of the copied
            object will be set to this value.

        shape : None or tuple of int or ndarray, optional
            The shape of the image on which the objects are placed.
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.
            If not ``None``, then the ``shape`` attribute of the copied object
            will be set to this value.

        Returns
        -------
        imgaug2.augmentables.lines.LineStringsOnImage
            Shallow copy.

        """
        if line_strings is None:
            line_strings = self.line_strings[:]
        if shape is None:
            # use tuple() here in case the shape was provided as a list
            shape = tuple(self.shape)

        return LineStringsOnImage(line_strings, shape)

    def deepcopy(
        self,
        line_strings: list[LineString] | None = None,
        shape: Shape | NDArray[np.generic] | None = None,
    ) -> LineStringsOnImage:
        """Create a deep copy of the object.

        Parameters
        ----------
        line_strings : None or list of imgaug2.augmentables.lines.LineString, optional
            List of line strings on the image.
            If not ``None``, then the ``line_strings`` attribute of the copied
            object will be set to this value.

        shape : None or tuple of int or ndarray, optional
            The shape of the image on which the objects are placed.
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.
            If not ``None``, then the ``shape`` attribute of the copied object
            will be set to this value.

        Returns
        -------
        imgaug2.augmentables.lines.LineStringsOnImage
            Deep copy.

        """
        # Manual copy is far faster than deepcopy, so use manual copy here.
        if line_strings is None:
            line_strings = [ls.deepcopy() for ls in self.line_strings]
        if shape is None:
            # use tuple() here in case the shape was provided as a list
            shape = tuple(self.shape)

        return LineStringsOnImage(line_strings, shape)

    @legacy(version="0.4.0")
    def __getitem__(
        self,
        indices: int | slice | Sequence[int],
    ) -> LineString | list[LineString]:
        """Get the line string(s) with given indices.


        Returns
        -------
        list of imgaug2.augmentables.lines.LineString
            Line string(s) with given indices.

        """
        return self.line_strings[indices]

    @legacy(version="0.4.0")
    def __iter__(self) -> Iterator[LineString]:
        """Iterate over the line strings in this container.


        Yields
        ------
        LineString
            A line string in this container.
            The order is identical to the order in the line string list
            provided upon class initialization.

        """
        return iter(self.line_strings)

    @legacy(version="0.4.0")
    def __len__(self) -> int:
        """Get the number of items in this instance.


        Returns
        -------
        int
            Number of items in this instance.

        """
        return len(self.items)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"LineStringsOnImage({str(self.line_strings)}, shape={self.shape})"


def _is_point_on_line(
    line_start: Point2D | Sequence[Number] | NDArray[np.number],
    line_end: Point2D | Sequence[Number] | NDArray[np.number],
    point: Point2D | Sequence[Number] | NDArray[np.number],
    eps: float = 1e-4,
) -> bool:
    dist_s2e = np.linalg.norm(np.float32(line_start) - np.float32(line_end))
    dist_s2p2e = np.linalg.norm(np.float32(line_start) - np.float32(point)) + np.linalg.norm(
        np.float32(point) - np.float32(line_end)
    )
    return bool(-eps < (dist_s2p2e - dist_s2e) < eps)


def _flatten_shapely_collection(collection: object) -> Iterator[object]:
    import shapely.geometry

    if not isinstance(collection, list):
        collection = [collection]
    for item in collection:
        if hasattr(item, "geoms"):
            for subitem in _flatten_shapely_collection(item.geoms):
                # MultiPoint.geoms actually returns a GeometrySequence
                if isinstance(subitem, shapely.geometry.base.GeometrySequence):
                    yield from subitem
                else:
                    yield _flatten_shapely_collection(subitem)
        else:
            yield item


def _convert_var_to_shapely_geometry(
    var: Point2D | Point2DList | list[LineString] | LineString,
) -> object:
    import shapely.geometry

    if isinstance(var, tuple):
        geom = shapely.geometry.Point(var[0], var[1])
    elif isinstance(var, list):
        assert len(var) > 0, (
            f"Expected list to contain at least one coordinate, got {len(var)} coordinates."
        )
        if isinstance(var[0], tuple):
            geom = shapely.geometry.LineString(var)
        elif all([isinstance(v, LineString) for v in var]):
            geom = shapely.geometry.MultiLineString(
                [shapely.geometry.LineString(ls.coords) for ls in var]
            )
        else:
            raise ValueError(
                "Could not convert list-input to shapely geometry. Invalid "
                "datatype. List elements had datatypes: {}.".format(
                    ", ".join([str(type(v)) for v in var])
                )
            )
    elif isinstance(var, LineString):
        geom = shapely.geometry.LineString(var.coords)
    else:
        raise ValueError(
            f"Could not convert input to shapely geometry. Invalid datatype. Got: {type(var)}"
        )
    return geom
