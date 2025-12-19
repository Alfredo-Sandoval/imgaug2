"""Collection of basic functions used throughout imgaug2."""

from __future__ import annotations

import functools
import importlib
import math
import numbers
import sys
import types
import typing
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeAlias, TypeVar

import cv2
import numpy as np
import skimage.draw
import skimage.measure

import imgaug2.dtypes as iadt
from imgaug2._deprecations import deprecated, warn, warn_deprecated
from imgaug2._deprecations import DeprecationWarning  # noqa: F401
from imgaug2.compat.markers import legacy

try:
    import numba
except ImportError:
    numba = None

if TYPE_CHECKING:
    # `imgaug2.imgaug` dynamically creates these names (see `MOVED`) to preserve
    # backwards compatibility. Statically declare them so type checkers (ty) can
    # resolve imports like `from imgaug2.imgaug import pad`.
    from imgaug2.augmenters.meta import Augmenter
    from imgaug2.random import RNG, NumpyGenerator, RNGInput

    BackgroundAugmenter: Any
    BatchLoader: Any
    compute_geometric_median: Any
    pad: Any
    pad_to_aspect_ratio: Any
    pad_to_multiples_of: Any
    compute_paddings_for_aspect_ratio: Any
    compute_paddings_to_reach_multiples_of: Any
    compute_paddings_to_reach_exponents_of: Any
    quokka: Any
    quokka_square: Any
    quokka_heatmap: Any
    quokka_segmentation_map: Any
    quokka_keypoints: Any
    quokka_bounding_boxes: Any
    quokka_polygons: Any

    Keypoint: Any
    KeypointsOnImage: Any
    BoundingBox: Any
    BoundingBoxesOnImage: Any
    LineString: Any
    LineStringsOnImage: Any
    Polygon: Any
    PolygonsOnImage: Any
    MultiPolygon: Any
    _ConcavePolygonRecoverer: Any
    HeatmapsOnImage: Any
    SegmentationMapsOnImage: Any
    Batch: Any

    _convert_points_to_shapely_line_string: Any
    _interpolate_point_pair: Any
    _interpolate_points: Any
    _interpolate_points_by_max_distance: Any


ALL = "ALL"

DEFAULT_FONT_FP = str(Path(__file__).parent / "DejaVuSans.ttf")

_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")

Number: TypeAlias = float | int
SizeParam: TypeAlias = Number | tuple[Number, Number] | list[Number]
ImagesInput: TypeAlias = np.ndarray | Sequence[np.ndarray]
BlockSize: TypeAlias = int | tuple[int, int] | tuple[int, int, int] | list[int]
Interpolation: TypeAlias = str | int | None
NestedIterable: TypeAlias = _T | list["NestedIterable[_T]"] | tuple["NestedIterable[_T]", ...]
Parents: TypeAlias = Sequence["Augmenter"]
HookActivator: TypeAlias = Callable[[ImagesInput, "Augmenter", Parents, bool], bool]
HookProcessor: TypeAlias = Callable[[ImagesInput, "Augmenter", Parents], ImagesInput]


# To check if a dtype instance is among these dtypes, use e.g.
# `dtype.type in NP_FLOAT_TYPES` (do not use `dtype in NP_FLOAT_TYPES` as that
# would fail).
#
# NumPy 2.0 removed `np.sctypes`. We derive the scalar type sets from the
# stable `np.typecodes` mapping instead.
def _get_np_scalar_types(typecodes_key: str) -> set[type[np.generic]]:
    types = set()
    for code in np.typecodes.get(typecodes_key, ""):
        try:
            types.add(np.dtype(code).type)
        except (TypeError, ValueError):
            # Some platforms don't support all codes (e.g. long double/float128).
            continue
    return types


NP_FLOAT_TYPES = _get_np_scalar_types("Float")
NP_INT_TYPES = _get_np_scalar_types("Integer")
NP_UINT_TYPES = _get_np_scalar_types("UnsignedInteger")

IMSHOW_BACKEND_DEFAULT = "matplotlib"

IMRESIZE_VALID_INTERPOLATIONS = [
    "nearest",
    "linear",
    "area",
    "cubic",
    cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,
    cv2.INTER_AREA,
    cv2.INTER_CUBIC,
]

# Cache dict to save kernels used for pooling.
_POOLING_KERNELS_CACHE = {}

_NUMBA_INSTALLED = numba is not None

_UINT8_DTYPE = np.dtype("uint8")


###############################################################################


@legacy
def is_np_array(val: object) -> typing.TypeGuard[np.ndarray]:
    """Check whether a variable is a numpy array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a numpy array. Otherwise ``False``.

    """
    # using np.generic here via isinstance(val, (np.ndarray, np.generic))
    # seems to also fire for scalar numpy values even though those are not
    # arrays
    return isinstance(val, np.ndarray)


@legacy
def is_np_scalar(val: object) -> bool:
    """Check whether a variable is a numpy scalar.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a numpy scalar. Otherwise ``False``.

    """
    # Note that isscalar() alone also fires for thinks like python strings
    # or booleans.
    # The isscalar() was added to make this function not fire for non-scalar
    # numpy types. Not sure if it is necessary.
    return isinstance(val, np.generic) and np.isscalar(val)


@legacy
def is_single_integer(val: object) -> bool:
    """Check whether a variable is an ``int``.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is an ``int``. Otherwise ``False``.

    """
    return isinstance(val, numbers.Integral) and not isinstance(val, bool)


@legacy
def is_single_float(val: object) -> bool:
    """Check whether a variable is a ``float``.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a ``float``. Otherwise ``False``.

    """
    return (
        isinstance(val, numbers.Real) and not is_single_integer(val) and not isinstance(val, bool)
    )


@legacy
def is_single_number(val: object) -> bool:
    """Check whether a variable is a ``number``, i.e. an ``int`` or ``float``.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a ``number``. Otherwise ``False``.

    """
    return is_single_integer(val) or is_single_float(val)


@legacy
def is_iterable(val: object) -> bool:
    """
    Checks whether a variable is iterable.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is an iterable. Otherwise ``False``.

    """
    return isinstance(val, Iterable)


def is_single_string(val: object) -> bool:
    """Check whether a variable is a single string.

    This function checks if the input is a string instance.
    The name follows the naming convention of related functions
    like ``is_single_integer()``, ``is_single_float()``, etc.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a string. Otherwise ``False``.

    """
    return isinstance(val, str)


@legacy
def is_string(val: object) -> bool:
    """Check whether a variable is a string.

    .. deprecated::
        Use :func:`is_single_string` instead for consistency with other
        naming conventions like ``is_single_integer()``.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a string. Otherwise ``False``.

    """
    return is_single_string(val)


@legacy
def is_single_bool(val: object) -> bool:
    """Check whether a variable is a ``bool``.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a ``bool``. Otherwise ``False``.

    """
    return isinstance(val, bool)


@legacy
def is_integer_array(val: object) -> bool:
    """Check whether a variable is a numpy integer array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a numpy integer array. Otherwise ``False``.

    """
    if not is_np_array(val):
        return False
    val = typing.cast(np.ndarray, val)
    return issubclass(val.dtype.type, np.integer)


@legacy
def is_float_array(val: object) -> bool:
    """Check whether a variable is a numpy float array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a numpy float array. Otherwise ``False``.

    """
    if not is_np_array(val):
        return False
    val = typing.cast(np.ndarray, val)
    return issubclass(val.dtype.type, np.floating)


@legacy
def is_callable(val: object) -> bool:
    """Check whether a variable is a callable, e.g. a function.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` if the variable is a callable. Otherwise ``False``.

    """
    return callable(val)


@legacy
def is_generator(val: object) -> bool:
    """Check whether a variable is a generator.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        ``True`` is the variable is a generator. Otherwise ``False``.

    """
    return isinstance(val, types.GeneratorType)


@legacy
def flatten(nested_iterable: NestedIterable[_T]) -> Iterator[_T]:
    """Flatten arbitrarily nested lists/tuples.

    Code partially taken from https://stackoverflow.com/a/10824420.

    Parameters
    ----------
    nested_iterable
        A ``list`` or ``tuple`` of arbitrarily nested values.

    Yields
    ------
    any
        All values in `nested_iterable`, flattened.

    """
    # don't just check if something is iterable here, because then strings
    # and arrays will be split into their characters and components
    if not isinstance(nested_iterable, (list, tuple)):
        yield nested_iterable
    else:
        for i in nested_iterable:
            if isinstance(i, (list, tuple)):
                yield from flatten(i)
            else:
                yield i


@legacy
def caller_name() -> str:
    """Return the name of the caller, e.g. a function.

    Returns
    -------
    str
        The name of the caller as a string

    """
    return sys._getframe(1).f_code.co_name


@legacy
def seed(entropy: int | None = None, seedval: int | None = None) -> None:
    """Set the seed of imgaug's global RNG.

    The global RNG controls most of the "randomness" in imgaug2.

    The global RNG is the default one used by all augmenters. Under special
    circumstances (e.g. when an augmenter is switched to deterministic mode),
    the global RNG is replaced with a local one. The state of that replacement
    may be dependent on the global RNG's state at the time of creating the
    child RNG.

    .. note::

        This function is not yet marked as deprecated, but might be in the
        future. The preferred way to seed `imgaug` is via
        :func:`~imgaug2.random.seed`.

    Parameters
    ----------
    entropy : int
        The seed value to use.

    seedval : None or int, optional
        Deprecated since 0.4.0.

    """
    assert entropy is not None or seedval is not None, (
        "Expected argument 'entropy' or 'seedval' to be not-None, but bothwere None."
    )

    if seedval is not None:
        assert entropy is None, (
            "Argument 'seedval' is the outdated name for 'entropy'. Hence, "
            "if it is provided, 'entropy' must be None. Got 'entropy' value "
            f"of type {type(entropy)}."
        )

        warn_deprecated("Parameter 'seedval' is deprecated. Use 'entropy' instead.")
        entropy = seedval

    import imgaug2.random

    imgaug2.random.seed(entropy)


@legacy(deprecated=True, replacement="imgaug2.random.normalize_generator")
def normalize_random_state(random_state: RNGInput) -> NumpyGenerator:
    # NOTE: Despite the name, this returns a numpy Generator.
    # The legacy name is kept for backwards compatibility.
    """Normalize various inputs to a numpy random generator.

    Parameters
    ----------
    random_state : None or int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.bit_generator.SeedSequence
        See :func:`~imgaug2.random.normalize_generator`.

    Returns
    -------
    numpy.random.Generator
        Generator initialized from the provided input.

    """
    import imgaug2.random

    return imgaug2.random.normalize_generator_(random_state)


@legacy(deprecated=True, replacement="imgaug2.random.get_global_rng")
def current_random_state() -> RNG:
    """Get or create the current global RNG of imgaug2.

    Note that the first call to this function will create a global RNG.

    Returns
    -------
    imgaug2.random.RNG
        The global RNG to use.

    """
    import imgaug2.random

    return imgaug2.random.get_global_rng()


@legacy
@deprecated("imgaug2.random.convert_seed_to_rng")
def new_random_state(seed: int | None = None, fully_random: bool = False) -> RNG:
    """Create a new numpy random number generator.

    Parameters
    ----------
    seed : None or int, optional
        The seed value to use. If ``None`` and `fully_random` is ``False``,
        the seed will be derived from the global RNG. If `fully_random` is
        ``True``, the seed will be provided by the OS.

    fully_random : bool, optional
        Whether the seed will be provided by the OS.

    Returns
    -------
    numpy.random.Generator
        Generator initialized with the provided seed.

    """
    import imgaug2.random

    if seed is None:
        if fully_random:
            return imgaug2.random.RNG.create_fully_random()
        return imgaug2.random.RNG.create_pseudo_random_()
    return imgaug2.random.RNG(seed)


@legacy
@deprecated("imgaug2.random.convert_seed_to_rng")
def dummy_random_state() -> RNG:
    """Create a dummy random state using a seed of ``1``.

    Returns
    -------
    imgaug2.random.RNG
        The new random state.

    """
    import imgaug2.random

    return imgaug2.random.RNG(1)


@legacy
@deprecated("imgaug2.random.copy_generator_unless_global_rng")
def copy_random_state(random_state: NumpyGenerator, force_copy: bool = False) -> NumpyGenerator:
    """Copy an existing numpy (random number) generator.

    Parameters
    ----------
    random_state : numpy.random.Generator
        The generator to copy.

    force_copy : bool, optional
        If ``True``, this function will always create a copy of every random
        state. If ``False``, it will not copy numpy's default random state,
        but all other random states.

    Returns
    -------
    numpy.random.Generator
        The copied generator.

    """
    import imgaug2.random

    if force_copy:
        return imgaug2.random.copy_generator(random_state)
    return imgaug2.random.copy_generator_unless_global_generator(random_state)


@legacy
@deprecated("imgaug2.random.derive_generator_")
def derive_random_state(random_state: NumpyGenerator) -> NumpyGenerator:
    """Derive a child numpy random generator from another one.

    Parameters
    ----------
    random_state : numpy.random.Generator
        The generator from which to derive a new child generator.

    Returns
    -------
    numpy.random.Generator
        A derived child generator.

    """
    import imgaug2.random

    return imgaug2.random.derive_generator_(random_state)


@legacy
@deprecated("imgaug2.random.derive_generators_")
def derive_random_states(random_state: NumpyGenerator, n: int = 1) -> list[NumpyGenerator]:
    """Derive child numpy random generators from another one.

    Parameters
    ----------
    random_state : numpy.random.Generator
        The generator from which to derive new child generators.

    n : int, optional
        Number of child generators to derive.

    Returns
    -------
    list of numpy.random.Generator
        List of derived child generators.

    """
    import imgaug2.random

    return list(imgaug2.random.derive_generators_(random_state, n=n))


@legacy
@deprecated("imgaug2.random.advance_generator_")
def forward_random_state(random_state: RNG) -> None:
    """Advance a numpy random generator's internal state.

    Parameters
    ----------
    random_state : numpy.random.Generator
        Generator of which to advance the internal state.

    """
    import imgaug2.random

    imgaug2.random.advance_generator_(random_state)


# TODO change this to some atan2 stuff?
@legacy
def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculcate the angle in radians between vectors `v1` and `v2`.

    From
    http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python

    Parameters
    ----------
    v1 : (N,) ndarray
        First vector.

    v2 : (N,) ndarray
        Second vector.

    Returns
    -------
    float
        Angle in radians.

    Examples
    --------
    >>> angle_between_vectors(np.float32([1, 0, 0]), np.float32([0, 1, 0]))
    1.570796...

    >>> angle_between_vectors(np.float32([1, 0, 0]), np.float32([1, 0, 0]))
    0.0

    >>> angle_between_vectors(np.float32([1, 0, 0]), np.float32([-1, 0, 0]))
    3.141592...

    """
    length1 = np.linalg.norm(v1)
    length2 = np.linalg.norm(v2)
    v1_unit = (v1 / length1) if length1 > 0 else np.float32(v1) * 0
    v2_unit = (v2 / length2) if length2 > 0 else np.float32(v2) * 0
    # Cast to Python float for stable repr across numpy versions.
    return float(np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)))


@legacy
def compute_line_intersection_point(
    x1: Number,
    y1: Number,
    x2: Number,
    y2: Number,
    x3: Number,
    y3: Number,
    x4: Number,
    y4: Number,
) -> tuple[Number, Number] | bool:
    """Compute the intersection point of two lines.

    Taken from https://stackoverflow.com/a/20679579 .

    Parameters
    ----------
    x1 : number
        x coordinate of the first point on line 1.
        (The lines extends beyond this point.)

    y1 : number
        y coordinate of the first point on line 1.
        (The lines extends beyond this point.)

    x2 : number
        x coordinate of the second point on line 1.
        (The lines extends beyond this point.)

    y2 : number
        y coordinate of the second point on line 1.
        (The lines extends beyond this point.)

    x3 : number
        x coordinate of the first point on line 2.
        (The lines extends beyond this point.)

    y3 : number
        y coordinate of the first point on line 2.
        (The lines extends beyond this point.)

    x4 : number
        x coordinate of the second point on line 2.
        (The lines extends beyond this point.)

    y4 : number
        y coordinate of the second point on line 2.
        (The lines extends beyond this point.)

    Returns
    -------
    tuple of number or bool
        The coordinate of the intersection point as a ``tuple`` ``(x, y)``.
        If the lines are parallel (no intersection point or an infinite number
        of them), the result is ``False``.

    """
    def _make_line(
        point1: tuple[Number, Number],
        point2: tuple[Number, Number],
    ) -> tuple[Number, Number, Number]:
        line_y = point1[1] - point2[1]
        line_x = point2[0] - point1[0]
        slope = point1[0] * point2[1] - point2[0] * point1[1]
        return line_y, line_x, -slope

    line1 = _make_line((x1, y1), (x2, y2))
    line2 = _make_line((x3, y3), (x4, y4))

    D = line1[0] * line2[1] - line1[1] * line2[0]
    Dx = line1[2] * line2[1] - line1[1] * line2[2]
    Dy = line1[0] * line2[2] - line1[2] * line2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    return False


# TODO replace by cv2.putText()?
@legacy
def draw_text(
    img: np.ndarray,
    y: int,
    x: int,
    text: str,
    color: Sequence[int] = (0, 255, 0),
    size: int = 25,
) -> np.ndarray:
    """Draw text on an image.

    This uses by default DejaVuSans as its font, which is included in this
    library.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: yes; not tested
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

        TODO check if other dtypes could be enabled

    Parameters
    ----------
    img : (H,W,3) ndarray
        The image array to draw text on.
        Expected to be of dtype ``uint8`` or ``float32`` (expected value
        range is ``[0.0, 255.0]``).

    y : int
        x-coordinate of the top left corner of the text.

    x : int
        y- coordinate of the top left corner of the text.

    text : str
        The text to draw.

    color : iterable of int, optional
        Color of the text to draw. For RGB-images this is expected to be an
        RGB color.

    size : int, optional
        Font size of the text to draw.

    Returns
    -------
    (H,W,3) ndarray
        Input image with text drawn on it.

    """
    from PIL import Image as PIL_Image
    from PIL import ImageDraw as PIL_ImageDraw
    from PIL import ImageFont as PIL_ImageFont

    assert img.dtype.name in ["uint8", "float32"], (
        "Can currently draw text only on images of dtype 'uint8' or "
        f"'float32'. Got dtype {img.dtype.name}."
    )

    input_dtype = img.dtype
    if img.dtype == np.float32:
        img = img.astype(np.uint8)

    img = PIL_Image.fromarray(img)
    font = PIL_ImageFont.truetype(DEFAULT_FONT_FP, size)
    context = PIL_ImageDraw.Draw(img)
    context.text((x, y), text, fill=tuple(color), font=font)
    img_np = np.asarray(img)

    # PIL/asarray returns read only array
    if not img_np.flags["WRITEABLE"]:
        try:
            # This seems to no longer work with recent NumPy/Pillow combos.
            img_np.setflags(write=True)
        except ValueError as ex:
            if "cannot set WRITEABLE flag to True of this array" in str(ex):
                img_np = np.copy(img_np)

    if img_np.dtype != input_dtype:
        img_np = img_np.astype(input_dtype)

    return img_np


# TODO rename sizes to size?
@legacy
def imresize_many_images(
    images: ImagesInput,
    sizes: SizeParam | None = None,
    interpolation: Interpolation = None,
) -> np.ndarray | list[np.ndarray]:
    """Resize each image in a list or array to a specified size.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested (3)
        * ``int16``: yes; tested
        * ``int32``: limited; tested (4)
        * ``int64``: no (2)
        * ``float16``: yes; tested (5)
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no (1)
        * ``bool``: yes; tested (6)

        - (1) rejected by ``cv2.imresize``
        - (2) results too inaccurate
        - (3) mapped internally to ``int16`` when interpolation!="nearest"
        - (4) only supported for interpolation="nearest", other interpolations
              lead to cv2 error
        - (5) mapped internally to ``float32``
        - (6) mapped internally to ``uint8``

    Parameters
    ----------
    images : (N,H,W,[C]) ndarray or list of (H,W,[C]) ndarray
        Array of the images to resize.
        Usually recommended to be of dtype ``uint8``.

    sizes : float or iterable of int or iterable of float
        The new size of the images, given either as a fraction (a single
        float) or as a ``(height, width)`` ``tuple`` of two integers or as a
        ``(height fraction, width fraction)`` ``tuple`` of two floats.

    interpolation : None or str or int, optional
        The interpolation to use during resize.
        If ``int``, then expected to be one of:

            * ``cv2.INTER_NEAREST`` (nearest neighbour interpolation)
            * ``cv2.INTER_LINEAR`` (linear interpolation)
            * ``cv2.INTER_AREA`` (area interpolation)
            * ``cv2.INTER_CUBIC`` (cubic interpolation)

        If ``str``, then expected to be one of:

            * ``nearest`` (identical to ``cv2.INTER_NEAREST``)
            * ``linear`` (identical to ``cv2.INTER_LINEAR``)
            * ``area`` (identical to ``cv2.INTER_AREA``)
            * ``cubic`` (identical to ``cv2.INTER_CUBIC``)

        If ``None``, the interpolation will be chosen automatically. For size
        increases, ``area`` interpolation will be picked and for size
        decreases, ``linear`` interpolation will be picked.

    Returns
    -------
    (N,H',W',[C]) ndarray
        Array of the resized images.

    Examples
    --------
    >>> import imgaug2 as ia
    >>> images = np.zeros((2, 8, 16, 3), dtype=np.uint8)
    >>> images_resized = ia.imresize_many_images(images, 2.0)
    >>> images_resized.shape
    (2, 16, 32, 3)

    Convert two RGB images of height ``8`` and width ``16`` to images of
    height ``2*8=16`` and width ``2*16=32``.

    >>> images_resized = ia.imresize_many_images(images, (2.0, 4.0))
    >>> images_resized.shape
    (2, 16, 64, 3)

    Convert two RGB images of height ``8`` and width ``16`` to images of
    height ``2*8=16`` and width ``4*16=64``.

    >>> images_resized = ia.imresize_many_images(images, (16, 32))
    >>> images_resized.shape
    (2, 16, 32, 3)

    Converts two RGB images of height ``8`` and width ``16`` to images of
    height ``16`` and width ``32``.

    """

    # we just do nothing if the input contains zero images
    # one could also argue that an exception would be appropriate here
    if len(images) == 0:
        if isinstance(images, np.ndarray):
            return images
        return list(images)

    # verify that sizes contains only values >0
    if is_single_number(sizes) and sizes <= 0:
        raise ValueError(
            f"If 'sizes' is given as a single number, it is expected to be >= 0, got {sizes:.8f}."
        )

    # change after the validation to make the above error messages match the
    # original input
    if is_single_number(sizes):
        sizes = (sizes, sizes)
    else:
        assert len(sizes) == 2, (
            f"If 'sizes' is given as a tuple, it is expected be a tuple of two entries, got {len(sizes)} entries."
        )
        assert all([is_single_number(val) and val >= 0 for val in sizes]), (
            "If 'sizes' is given as a tuple, it is expected be a tuple of two "
            f"ints or two floats, each >= 0, got types {str([type(val) for val in sizes])} with values {str(sizes)}."
        )

    # if input is a list, call this function N times for N images
    # but check beforehand if all images have the same shape, then just
    # convert to a single array and de-convert afterwards
    if isinstance(images, list):
        nb_shapes = len({image.shape for image in images})
        if nb_shapes == 1:
            return list(
                imresize_many_images(np.array(images), sizes=sizes, interpolation=interpolation)
            )

        return [
            imresize_many_images(image[np.newaxis, ...], sizes=sizes, interpolation=interpolation)[
                0, ...
            ]
            for image in images
        ]

    shape = images.shape
    assert images.ndim in [3, 4], f"Expected array of shape (N, H, W, [C]), got shape {str(shape)}"
    nb_images = shape[0]
    height_image, width_image = shape[1], shape[2]
    nb_channels = shape[3] if images.ndim > 3 else None

    height_target, width_target = sizes[0], sizes[1]
    height_target = (
        int(np.round(height_image * height_target))
        if is_single_float(height_target)
        else height_target
    )
    width_target = (
        int(np.round(width_image * width_target)) if is_single_float(width_target) else width_target
    )

    if height_target == height_image and width_target == width_image:
        return np.copy(images)

    # return empty array if input array contains zero-sized axes
    # note that None==0 is not True (for case nb_channels=None)
    if 0 in [height_target, width_target, nb_channels]:
        shape_out = tuple([shape[0], height_target, width_target] + list(shape[3:]))
        return np.zeros(shape_out, dtype=images.dtype)

    # place this after the (h==h' and w==w') check so that images with
    # zero-sized don't result in errors if the aren't actually resized
    # verify that all input images have height/width > 0
    has_zero_size_axes = any([axis == 0 for axis in images.shape[1:]])
    assert not has_zero_size_axes, (
        "Cannot resize images, because at least one image has a height and/or "
        "width and/or number of channels of zero. "
        f"Observed shapes were: {str([image.shape for image in images])}."
    )

    inter = interpolation
    assert inter is None or inter in IMRESIZE_VALID_INTERPOLATIONS, (
        "Expected 'interpolation' to be None or one of {}. Got {}.".format(
            ", ".join([str(valid_ip) for valid_ip in IMRESIZE_VALID_INTERPOLATIONS]), str(inter)
        )
    )
    if inter is None:
        if height_target > height_image or width_target > width_image:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
    elif inter in ["nearest", cv2.INTER_NEAREST]:
        inter = cv2.INTER_NEAREST
    elif inter in ["linear", cv2.INTER_LINEAR]:
        inter = cv2.INTER_LINEAR
    elif inter in ["area", cv2.INTER_AREA]:
        inter = cv2.INTER_AREA
    else:  # if ip in ["cubic", cv2.INTER_CUBIC]:
        inter = cv2.INTER_CUBIC

    if inter == cv2.INTER_NEAREST:
        iadt.gate_dtypes_strs(
            images,
            allowed="bool uint8 uint16 int8 int16 int32 float16 float32 float64",
            disallowed="uint32 uint64 int64 float128",
            augmenter=None,
        )
    else:
        iadt.gate_dtypes_strs(
            images,
            allowed="bool uint8 uint16 int8 int16 float16 float32 float64",
            disallowed="uint32 uint64 int32 int64 float128",
            augmenter=None,
        )

    result_shape = (nb_images, height_target, width_target)
    if nb_channels is not None:
        result_shape = result_shape + (nb_channels,)
    result = np.zeros(result_shape, dtype=images.dtype)
    for i, image in enumerate(images):
        input_dtype = image.dtype
        input_dtype_name = input_dtype.name

        if input_dtype_name == "bool":
            image = image.astype(np.uint8) * 255
        elif input_dtype_name == "int8" and inter != cv2.INTER_NEAREST:
            image = image.astype(np.int16)
        elif input_dtype_name == "float16":
            image = image.astype(np.float32)

        if nb_channels is not None and nb_channels > 512:
            channels = [
                cv2.resize(image[..., c], (width_target, height_target), interpolation=inter)
                for c in range(nb_channels)
            ]
            result_img = np.stack(channels, axis=-1)
        else:
            result_img = cv2.resize(image, (width_target, height_target), interpolation=inter)

        assert result_img.dtype.name == image.dtype.name, (
            f"Expected cv2.resize() to keep the input dtype '{image.dtype.name}', but got "
            f"'{result_img.dtype.name}'. This is an internal error. Please report."
        )

        # cv2 removes the channel axis if input was (H, W, 1)
        # we re-add it (but only if input was not (H, W))
        if len(result_img.shape) == 2 and nb_channels is not None and nb_channels == 1:
            result_img = result_img[:, :, np.newaxis]

        if input_dtype_name == "bool":
            result_img = result_img > 127
        elif input_dtype_name == "int8" and inter != cv2.INTER_NEAREST:
            result_img = iadt.restore_dtypes_(result_img, np.int8)
        elif input_dtype_name == "float16":
            result_img = iadt.restore_dtypes_(result_img, np.float16)
        result[i] = result_img
    return result


def _assert_two_or_three_dims(shape: np.ndarray | Sequence[int]) -> None:
    if hasattr(shape, "shape"):
        shape = shape.shape
    assert len(shape) in [2, 3], (
        f"Expected image with two or three dimensions, but got {len(shape)} dimensions "
        f"and shape {shape}."
    )


@legacy
def imresize_single_image(
    image: np.ndarray,
    sizes: SizeParam,
    interpolation: Interpolation = None,
) -> np.ndarray:
    """Resize a single image.

    **Supported dtypes**:

        See :func:`~imgaug2.imgaug2.imresize_many_images`.

    Parameters
    ----------
    image : (H,W,C) ndarray or (H,W) ndarray
        Array of the image to resize.
        Usually recommended to be of dtype ``uint8``.

    sizes : float or iterable of int or iterable of float
        See :func:`~imgaug2.imgaug2.imresize_many_images`.

    interpolation : None or str or int, optional
        See :func:`~imgaug2.imgaug2.imresize_many_images`.

    Returns
    -------
    (H',W',C) ndarray or (H',W') ndarray
        The resized image.

    """
    _assert_two_or_three_dims(image)

    grayscale = False
    if image.ndim == 2:
        grayscale = True
        image = image[:, :, np.newaxis]

    rs = imresize_many_images(image[np.newaxis, :, :, :], sizes, interpolation=interpolation)
    if grayscale:
        return rs[0, :, :, 0]
    return rs[0, ...]


@legacy
def pool(
    arr: np.ndarray,
    block_size: BlockSize,
    func: Callable[..., np.ndarray | float],
    pad_mode: str = "constant",
    pad_cval: Number = 0,
    preserve_dtype: bool = True,
    cval: Number | None = None,
) -> np.ndarray:
    """Resize an array by pooling values within blocks.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested (2)
        * ``uint64``: no (1)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested (2)
        * ``int64``: no (1)
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested (2)
        * ``bool``: yes; tested

        - (1) results too inaccurate (at least when using np.average as func)
        - (2) Note that scikit-image documentation says that the wrapped
              pooling function converts inputs to ``float64``. Actual tests
              showed no indication of that happening (at least when using
              preserve_dtype=True).

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool. Ideally of datatype ``float64``.

    block_size : int or tuple of int
        Spatial size of each group of values to pool, aka kernel size.

          * If a single ``int``, then a symmetric block of that size along
            height and width will be used.
          * If a ``tuple`` of two values, it is assumed to be the block size
            along height and width of the image-like, with pooling happening
            per channel.
          * If a ``tuple`` of three values, it is assumed to be the block size
            along height, width and channels.

    func : callable
        Function to apply to a given block in order to convert it to a single
        number, e.g. :func:`numpy.average`, :func:`numpy.min`,
        :func:`numpy.max`.

    pad_mode : str, optional
        Padding mode to use if the array cannot be divided by `block_size`
        without remainder. See :func:`~imgaug2.imgaug2.pad` for details.

    pad_cval : number, optional
        Value to use for padding if `mode` is ``constant``.
        See :func:`numpy.pad` for details.

    preserve_dtype : bool, optional
        Whether to convert the array back to the input datatype if it is
        changed away from that in the pooling process.

    cval : None or number, optional
        Deprecated. Old name for `pad_cval`.

    Returns
    -------
    (H',W') ndarray or (H',W',C') ndarray
        Array after pooling.

    """
    if arr.size == 0:
        return np.copy(arr)

    iadt.gate_dtypes_strs(
        {arr.dtype},
        allowed="bool uint8 uint16 uint32 int8 int16 int32 float16 float32 float64 float128",
        disallowed="uint64 int64",
    )

    if cval is not None:
        warn_deprecated("`cval` is a deprecated argument in pool(). Use `pad_cval` instead.")
        pad_cval = cval

    _assert_two_or_three_dims(arr)

    is_valid_int = is_single_integer(block_size) and block_size >= 1
    is_valid_tuple = (
        is_iterable(block_size)
        and len(block_size) in [2, 3]
        and [is_single_integer(val) and val >= 1 for val in block_size]
    )
    assert is_valid_int or is_valid_tuple, (
        "Expected argument 'block_size' to be a single integer >0 or "
        f"a tuple of 2 or 3 values with each one being >0. Got {str(block_size)}."
    )

    if is_single_integer(block_size):
        block_size = [block_size, block_size]
    if len(block_size) < arr.ndim:
        block_size = list(block_size) + [1]

    # We use custom padding here instead of the one from block_reduce(),
    # because (1) it is expected to be faster and (2) it allows us more
    # flexibility wrt to padding modes.
    arr = pad_to_multiples_of(
        arr,
        height_multiple=block_size[0],
        width_multiple=block_size[1],
        mode=pad_mode,
        cval=pad_cval,
    )

    input_dtype = arr.dtype

    arr_reduced = skimage.measure.block_reduce(arr, tuple(block_size), func, cval=cval)
    if preserve_dtype and arr_reduced.dtype.name != input_dtype.name:
        arr_reduced = arr_reduced.astype(input_dtype)
    return arr_reduced


# This automatically calls a special uint8 method if it fulfills standard
# cv2 criteria. Otherwise it falls back to pool().
@legacy
def _pool_dispatcher_(
    arr: np.ndarray,
    block_size: BlockSize,
    func_uint8: Callable[..., np.ndarray],
    blockfunc: Callable[..., np.ndarray | float],
    pad_mode: str = "edge",
    pad_cval: Number = 255,
    preserve_dtype: bool = True,
    cval: Number | None = None,
    copy: bool = False,
) -> np.ndarray:
    if not isinstance(block_size, (tuple, list)):
        block_size = (block_size, block_size)

    if 0 in block_size:
        return arr if not copy else np.copy(arr)

    shape = arr.shape
    nb_channels = 0 if len(shape) <= 2 else shape[-1]

    valid_for_cv2 = (
        arr.dtype.name == "uint8" and len(block_size) == 2 and nb_channels <= 512 and 0 not in shape
    )
    if valid_for_cv2:
        return func_uint8(
            arr, block_size, pad_mode=pad_mode, pad_cval=pad_cval if cval is None else cval
        )
    return pool(
        arr,
        block_size,
        blockfunc,
        pad_mode=pad_mode,
        pad_cval=pad_cval,
        preserve_dtype=preserve_dtype,
        cval=cval,
    )


@legacy
def avg_pool(
    arr: np.ndarray,
    block_size: BlockSize,
    pad_mode: str = "reflect",
    pad_cval: Number = 128,
    preserve_dtype: bool = True,
    cval: Number | None = None,
) -> np.ndarray:
    """Resize an array using average pooling.

    Defaults to ``pad_mode="reflect"`` to ensure that padded values do not
    affect the average.

    .. note::

        This function currently rounds ``0.5`` up for (most) ``uint8``
        images, but rounds it down for other dtypes.

    **Supported dtypes**:

        See :func:`~imgaug2.imgaug2.pool`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool.
        See :func:`~imgaug2.imgaug2.pool` for details.

    block_size : int or tuple of int
        Size of each block of values to pool.
        See :func:`~imgaug2.imgaug2.pool` for details.

    pad_mode : str, optional
        Padding mode to use if the array cannot be divided by `block_size`
        without remainder.
        See :func:`~imgaug2.imgaug2.pad` for details.

    pad_cval : number, optional
        Padding value.
        See :func:`~imgaug2.imgaug2.pool` for details.

    preserve_dtype : bool, optional
        Whether to preserve the input array dtype.
        See  :func:`~imgaug2.imgaug2.pool` for details.

    cval : None or number, optional
        Deprecated. Old name for `pad_cval`.

    Returns
    -------
    (H',W') ndarray or (H',W',C') ndarray
        Array after average pooling.

    """
    return _pool_dispatcher_(
        arr,
        block_size,
        _avg_pool_uint8,
        np.average,
        pad_mode=pad_mode,
        pad_cval=pad_cval,
        preserve_dtype=preserve_dtype,
        cval=cval,
        copy=True,
    )


@legacy
def _avg_pool_uint8(
    arr: np.ndarray,
    block_size: Sequence[int],
    pad_mode: str = "reflect",
    pad_cval: Number = 128,
) -> np.ndarray:
    ndim_in = arr.ndim

    shape = arr.shape
    if shape[0] % block_size[0] != 0 or shape[1] % block_size[1] != 0:
        arr = pad_to_multiples_of(
            arr,
            height_multiple=block_size[0],
            width_multiple=block_size[1],
            mode=pad_mode,
            cval=pad_cval,
        )

    height = arr.shape[0] // block_size[0]
    width = arr.shape[1] // block_size[1]

    arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)

    if arr.ndim < ndim_in:
        arr = arr[:, :, np.newaxis]

    return arr


@legacy
def max_pool(
    arr: np.ndarray,
    block_size: BlockSize,
    pad_mode: str = "edge",
    pad_cval: Number = 0,
    preserve_dtype: bool = True,
    cval: Number | None = None,
) -> np.ndarray:
    """Resize an array using max-pooling.

    Defaults to ``pad_mode="edge"`` to ensure that padded values do not affect
    the maximum, even if the dtype was something else than ``uint8``.

    **Supported dtypes**:

        See :func:`~imgaug2.imgaug2.pool`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool.
        See :func:`~imgaug2.imgaug2.pool` for details.

    block_size : int or tuple of int
        Size of each block of values to pool.
        See :func:`~imgaug2.imgaug2.pool` for details.

    pad_mode : str, optional
        Padding mode to use if the array cannot be divided by `block_size`
        without remainder.
        See :func:`~imgaug2.imgaug2.pad` for details.

    pad_cval : number, optional
        Padding value.
        See :func:`~imgaug2.imgaug2.pool` for details.

    preserve_dtype : bool, optional
        Whether to preserve the input array dtype.
        See  :func:`~imgaug2.imgaug2.pool` for details.

    cval : None or number, optional
        Deprecated. Old name for `pad_cval`.

    Returns
    -------
    (H',W') ndarray or (H',W',C') ndarray
        Array after max-pooling.

    """
    return max_pool_(
        np.copy(arr),
        block_size,
        pad_mode=pad_mode,
        pad_cval=pad_cval,
        preserve_dtype=preserve_dtype,
        cval=cval,
    )


@legacy
def max_pool_(
    arr: np.ndarray,
    block_size: BlockSize,
    pad_mode: str = "edge",
    pad_cval: Number = 0,
    preserve_dtype: bool = True,
    cval: Number | None = None,
) -> np.ndarray:
    """Resize an array in-place using max-pooling.

    Defaults to ``pad_mode="edge"`` to ensure that padded values do not affect
    the maximum, even if the dtype was something else than ``uint8``.


    **Supported dtypes**:

        See :func:`~imgaug2.imgaug2.pool`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool.
        May be altered in-place.
        See :func:`~imgaug2.imgaug2.pool` for details.

    block_size : int or tuple of int
        Size of each block of values to pool.
        See :func:`~imgaug2.imgaug2.pool` for details.

    pad_mode : str, optional
        Padding mode to use if the array cannot be divided by `block_size`
        without remainder.
        See :func:`~imgaug2.imgaug2.pad` for details.

    pad_cval : number, optional
        Padding value.
        See :func:`~imgaug2.imgaug2.pool` for details.

    preserve_dtype : bool, optional
        Whether to preserve the input array dtype.
        See  :func:`~imgaug2.imgaug2.pool` for details.

    cval : None or number, optional
        Deprecated. Old name for `pad_cval`.

    Returns
    -------
    (H',W') ndarray or (H',W',C') ndarray
        Array after max-pooling.
        Might be a view of `arr`.

    """
    return _pool_dispatcher_(
        arr,
        block_size,
        _max_pool_uint8_,
        np.max,
        pad_mode=pad_mode,
        pad_cval=pad_cval,
        preserve_dtype=preserve_dtype,
        cval=cval,
    )


@legacy
def min_pool(
    arr: np.ndarray,
    block_size: BlockSize,
    pad_mode: str = "edge",
    pad_cval: Number = 255,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """Resize an array using min-pooling.

    Defaults to ``pad_mode="edge"`` to ensure that padded values do not affect
    the minimum, even if the dtype was something else than ``uint8``.

    **Supported dtypes**:

        See :func:`~imgaug2.imgaug2.pool`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool.
        See :func:`~imgaug2.imgaug2.pool` for details.

    block_size : int or tuple of int
        Size of each block of values to pool.
        See :func:`~imgaug2.imgaug2.pool` for details.

    pad_mode : str, optional
        Padding mode to use if the array cannot be divided by `block_size`
        without remainder.
        See :func:`~imgaug2.imgaug2.pad` for details.

    pad_cval : number, optional
        Padding value.
        See :func:`~imgaug2.imgaug2.pool` for details.

    preserve_dtype : bool, optional
        Whether to preserve the input array dtype.
        See  :func:`~imgaug2.imgaug2.pool` for details.

    Returns
    -------
    (H',W') ndarray or (H',W',C') ndarray
        Array after min-pooling.

    """
    return min_pool_(
        np.copy(arr),
        block_size,
        pad_mode=pad_mode,
        pad_cval=pad_cval,
        preserve_dtype=preserve_dtype,
    )


@legacy
def min_pool_(
    arr: np.ndarray,
    block_size: BlockSize,
    pad_mode: str = "edge",
    pad_cval: Number = 255,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """Resize an array in-place using min-pooling.

    Defaults to ``pad_mode="edge"`` to ensure that padded values do not affect
    the minimum, even if the dtype was something else than ``uint8``.


    **Supported dtypes**:

        See :func:`~imgaug2.imgaug2.pool`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool.
        May be altered in-place.
        See :func:`~imgaug2.imgaug2.pool` for details.

    block_size : int or tuple of int
        Size of each block of values to pool.
        See :func:`~imgaug2.imgaug2.pool` for details.

    pad_mode : str, optional
        Padding mode to use if the array cannot be divided by `block_size`
        without remainder.
        See :func:`~imgaug2.imgaug2.pad` for details.

    pad_cval : number, optional
        Padding value.
        See :func:`~imgaug2.imgaug2.pool` for details.

    preserve_dtype : bool, optional
        Whether to preserve the input array dtype.
        See  :func:`~imgaug2.imgaug2.pool` for details.

    Returns
    -------
    (H',W') ndarray or (H',W',C') ndarray
        Array after min-pooling.
        Might be a view of `arr`.

    """
    return _pool_dispatcher_(
        arr,
        block_size,
        _min_pool_uint8_,
        np.min,
        pad_mode=pad_mode,
        pad_cval=pad_cval,
        preserve_dtype=preserve_dtype,
    )


@legacy
def _min_pool_uint8_(
    arr: np.ndarray,
    block_size: Sequence[int],
    pad_mode: str = "edge",
    pad_cval: Number = 255,
) -> np.ndarray:
    return _minmax_pool_uint8_(arr, block_size, cv2.erode, pad_mode=pad_mode, pad_cval=pad_cval)


@legacy
def _max_pool_uint8_(
    arr: np.ndarray,
    block_size: Sequence[int],
    pad_mode: str = "edge",
    pad_cval: Number = 0,
) -> np.ndarray:
    return _minmax_pool_uint8_(arr, block_size, cv2.dilate, pad_mode=pad_mode, pad_cval=pad_cval)


@legacy
def _minmax_pool_uint8_(
    arr: np.ndarray,
    block_size: Sequence[int],
    func: Callable[..., np.ndarray],
    pad_mode: str,
    pad_cval: Number,
) -> np.ndarray:
    ndim_in = arr.ndim

    shape = arr.shape
    if shape[0] % block_size[0] != 0 or shape[1] % block_size[1] != 0:
        arr = pad_to_multiples_of(
            arr,
            height_multiple=block_size[0],
            width_multiple=block_size[1],
            mode=pad_mode,
            cval=pad_cval,
        )

    kernel = globals()["_POOLING_KERNELS_CACHE"].get(block_size, None)
    if kernel is None:
        kernel = np.ones(block_size, dtype=np.uint8)
        if block_size[0] <= 30 and block_size[1] <= 30:
            globals()["_POOLING_KERNELS_CACHE"][block_size] = kernel

    arr = cv2.flip(arr, -1)
    arr = func(arr, kernel, iterations=1)
    arr = cv2.flip(arr, -1)

    if arr.ndim < ndim_in:
        arr = arr[:, :, np.newaxis]

    start_height = (block_size[0] - 1) // 2
    start_width = (block_size[1] - 1) // 2
    return arr[start_height :: block_size[0], start_width :: block_size[1]]


@legacy
def median_pool(
    arr: np.ndarray,
    block_size: BlockSize,
    pad_mode: str = "reflect",
    pad_cval: Number = 128,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """Resize an array using median-pooling.

    Defaults to ``pad_mode="reflect"`` to ensure that padded values do not
    affect the average.

    **Supported dtypes**:

        See :func:`~imgaug2.imgaug2.pool`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool.
        See :func:`~imgaug2.imgaug2.pool` for details.

    block_size : int or tuple of int
        Size of each block of values to pool.
        See :func:`~imgaug2.imgaug2.pool` for details.

    pad_mode : str, optional
        Padding mode to use if the array cannot be divided by `block_size`
        without remainder.
        See :func:`~imgaug2.imgaug2.pad` for details.

    pad_cval : number, optional
        Padding value.
        See :func:`~imgaug2.imgaug2.pool` for details.

    preserve_dtype : bool, optional
        Whether to preserve the input array dtype.
        See  :func:`~imgaug2.imgaug2.pool` for details.

    Returns
    -------
    (H',W') ndarray or (H',W',C') ndarray
        Array after min-pooling.

    """
    # This uses a custom dispatcher (compared to avg/min/max pool), because
    # cv2 medianBlur only works with odd kernel sizes > 1, does not support
    # height/width-wise ksizes and uses a different method for ksizes > 5
    # leading to different performance characteristics for ksizes <= 5 and
    # ksizes > 5.

    if not isinstance(block_size, (tuple, list)):
        block_size = (block_size, block_size)

    if 0 in block_size:
        return np.copy(arr)

    shape = arr.shape
    nb_channels = 0 if len(shape) <= 2 else shape[-1]

    valid_for_cv2 = (
        arr.dtype.name == "uint8"
        and len(block_size) == 2
        and block_size[0] == block_size[1]
        and (
            block_size[0] in [3, 5]
            or (block_size[0] in [7, 9, 11, 13] and (shape[0] * shape[1]) <= (32 * 32))
        )
        and nb_channels <= 512
        and 0 not in shape
    )
    if valid_for_cv2:
        return _median_pool_cv2(arr, block_size[0], pad_mode=pad_mode, pad_cval=pad_cval)
    return pool(
        arr,
        block_size,
        np.median,
        pad_mode=pad_mode,
        pad_cval=pad_cval,
        preserve_dtype=preserve_dtype,
    )


# block_size must be a single integer here, in contrast to the other cv2
# pool methods that support (int, int).
@legacy
def _median_pool_cv2(
    arr: np.ndarray,
    block_size: int,
    pad_mode: str,
    pad_cval: Number,
) -> np.ndarray:
    ndim_in = arr.ndim

    shape = arr.shape
    if shape[0] % block_size != 0 or shape[1] % block_size != 0:
        arr = pad_to_multiples_of(
            arr, height_multiple=block_size, width_multiple=block_size, mode=pad_mode, cval=pad_cval
        )

    arr = cv2.medianBlur(arr, block_size)

    if arr.ndim < ndim_in:
        arr = arr[:, :, np.newaxis]

    start_height = (block_size - 1) // 2
    start_width = (block_size - 1) // 2
    return arr[start_height::block_size, start_width::block_size]


@legacy
def draw_grid(images: ImagesInput, rows: int | None = None, cols: int | None = None) -> np.ndarray:
    """Combine multiple images into a single grid-like image.

    Calling this function with four images of the same shape and ``rows=2``,
    ``cols=2`` will combine the four images to a single image array of shape
    ``(2*H, 2*W, C)``, where ``H`` is the height of any of the images
    (analogous ``W``) and ``C`` is the number of channels of any image.

    Calling this function with four images of the same shape and ``rows=4``,
    ``cols=1`` is analogous to calling :func:`numpy.vstack` on the images.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; fully tested
        * ``uint32``: yes; fully tested
        * ``uint64``: yes; fully tested
        * ``int8``: yes; fully tested
        * ``int16``: yes; fully tested
        * ``int32``: yes; fully tested
        * ``int64``: yes; fully tested
        * ``float16``: yes; fully tested
        * ``float32``: yes; fully tested
        * ``float64``: yes; fully tested
        * ``float128``: yes; fully tested
        * ``bool``: yes; fully tested

    Parameters
    ----------
    images : (N,H,W,3) ndarray or iterable of (H,W,3) array
        The input images to convert to a grid.

    rows : None or int, optional
        The number of rows to show in the grid.
        If ``None``, it will be automatically derived.

    cols : None or int, optional
        The number of cols to show in the grid.
        If ``None``, it will be automatically derived.

    Returns
    -------
    (H',W',3) ndarray
        Image of the generated grid.

    """
    nb_images = len(images)
    assert nb_images > 0, "Expected to get at least one image, got none."

    if is_np_array(images):
        assert images.ndim == 4, (
            "Expected to get an array of four dimensions denoting "
            f"(N, H, W, C), got {images.ndim} dimensions and shape {images.shape}."
        )
    else:
        assert is_iterable(images), f"Expected to get an iterable of ndarrays, got {type(images)}."
        assert all([is_np_array(image) for image in images]), (
            "Expected to get an iterable of ndarrays, got types {}.".format(
                ", ".join(
                    [str(type(image)) for image in images],
                )
            )
        )
        assert all([image.ndim == 3 for image in images]), (
            "Expected to get images with three dimensions. Got shapes {}.".format(
                ", ".join([str(image.shape) for image in images])
            )
        )
        assert len({image.dtype.name for image in images}) == 1, (
            "Expected to get images with the same dtypes, got dtypes {}.".format(
                ", ".join([image.dtype.name for image in images])
            )
        )
        assert len({image.shape[-1] for image in images}) == 1, (
            "Expected to get images with the same number of channels, got shapes {}.".format(
                ", ".join([str(image.shape) for image in images])
            )
        )

    cell_height = max([image.shape[0] for image in images])
    cell_width = max([image.shape[1] for image in images])
    nb_channels = images[0].shape[2]

    if rows is None and cols is None:
        rows = cols = int(math.ceil(math.sqrt(nb_images)))
    elif rows is not None:
        cols = int(math.ceil(nb_images / rows))
    elif cols is not None:
        rows = int(math.ceil(nb_images / cols))
    assert rows * cols >= nb_images, (
        "Expected rows*cols to lead to at least as many cells as there were "
        f"images provided, but got {rows} rows, {cols} cols (={rows * cols} cells) for {nb_images} "
        "images. "
    )

    width = cell_width * cols
    height = cell_height * rows
    dtype = images.dtype if is_np_array(images) else images[0].dtype
    grid = np.zeros((height, width, nb_channels), dtype=dtype)
    cell_idx = 0
    for row_idx in range(rows):
        for col_idx in range(cols):
            if cell_idx < nb_images:
                image = images[cell_idx]
                cell_y1 = cell_height * row_idx
                cell_y2 = cell_y1 + image.shape[0]
                cell_x1 = cell_width * col_idx
                cell_x2 = cell_x1 + image.shape[1]
                grid[cell_y1:cell_y2, cell_x1:cell_x2, :] = image
            cell_idx += 1

    return grid


@legacy
def show_grid(images: ImagesInput, rows: int | None = None, cols: int | None = None) -> None:
    """Combine multiple images into a single image and plot the result.

    This will show a window of the results of :func:`~imgaug2.imgaug2.draw_grid`.

    **Supported dtypes**:

        minimum of (
            :func:`~imgaug2.imgaug2.draw_grid`,
            :func:`~imgaug2.imgaug2.imshow`
        )

    Parameters
    ----------
    images : (N,H,W,3) ndarray or iterable of (H,W,3) array
        See :func:`~imgaug2.imgaug2.draw_grid`.

    rows : None or int, optional
        See :func:`~imgaug2.imgaug2.draw_grid`.

    cols : None or int, optional
        See :func:`~imgaug2.imgaug2.draw_grid`.

    """
    grid = draw_grid(images, rows=rows, cols=cols)
    imshow(grid)


@legacy
def imshow(
    image: np.ndarray,
    backend: Literal["matplotlib", "cv2"] = IMSHOW_BACKEND_DEFAULT,
) -> None:
    """Show an image in a window.

    **Supported dtypes**:

        * ``uint8``: yes; not tested
        * ``uint16``: ?
        * ``uint32``: ?
        * ``uint64``: ?
        * ``int8``: ?
        * ``int16``: ?
        * ``int32``: ?
        * ``int64``: ?
        * ``float16``: ?
        * ``float32``: ?
        * ``float64``: ?
        * ``float128``: ?
        * ``bool``: ?

    Parameters
    ----------
    image : (H,W,3) ndarray
        Image to show.

    backend : {'matplotlib', 'cv2'}, optional
        Library to use to show the image. May be either matplotlib or
        OpenCV ('cv2'). OpenCV tends to be faster, but apparently causes more
        technical issues.

    """
    assert backend in ["matplotlib", "cv2"], (
        f"Expected backend 'matplotlib' or 'cv2', got {backend}."
    )

    if backend == "cv2":
        image_bgr = image
        if image.ndim == 3 and image.shape[2] in [3, 4]:
            image_bgr = image[..., 0:3][..., ::-1]

        win_name = "imgaug-default-window"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, image_bgr)
        cv2.waitKey(0)
        cv2.destroyWindow(win_name)
    else:
        # import only when necessary (faster startup; optional dependency;
        # less fragile -- see issue #225)
        import matplotlib.pyplot as plt

        dpi = 96
        h, w = image.shape[0] / dpi, image.shape[1] / dpi
        # if the figure is too narrow, the footer may appear and make the fig
        # suddenly wider (ugly)
        w = max(w, 6)

        fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(f"imgaug2.imshow({image.shape})")
        # cmap=gray is automatically only activate for grayscale images
        ax.imshow(image, cmap="gray")
        plt.show()


@legacy
def do_assert(condition: bool, message: str = "Assertion failed.") -> None:
    """Assert that a ``condition`` holds or raise an ``Exception`` otherwise.

    This was added because `assert` statements are removed in optimized code.
    It replaced `assert` statements throughout the library, but that was
    reverted again for readability and performance reasons.

    Parameters
    ----------
    condition : bool
        If ``False``, an exception is raised.

    message : str, optional
        Error message.

    """
    if not condition:
        raise AssertionError(str(message))


@legacy
def _normalize_cv2_input_arr_(arr: np.ndarray) -> np.ndarray:
    flags = arr.flags
    if not flags["OWNDATA"]:
        arr = np.copy(arr)
        flags = arr.flags
    if not flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


@legacy
def apply_lut(image: np.ndarray, table: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Map an input image to a new one using a lookup table.


    **Supported dtypes**:

        See :func:`~imgaug2.imgaug2.apply_lut_`.

    Parameters
    ----------
    image : ndarray
        See :func:`~imgaug2.imgaug2.apply_lut_`.

    table : ndarray or list of ndarray
        See :func:`~imgaug2.imgaug2.apply_lut_`.

    Returns
    -------
    ndarray
        Image after mapping via lookup table.

    """
    return apply_lut_(np.copy(image), table)


# TODO make this function compatible with short max sized images, probably
#      isn't right now
@legacy
def apply_lut_(image: np.ndarray, table: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Map an input image in-place to a new one using a lookup table.


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        Image of dtype ``uint8`` and shape ``(H,W)`` or ``(H,W,C)``.

    table : ndarray or list of ndarray
        Table of dtype ``uint8`` containing the mapping from old to new
        values. Either a ``list`` of ``C`` ``(256,)`` arrays or a single
        array of shape ``(256,)`` or ``(256, C)`` or ``(1, 256, C)``.
        In case of ``(256,)`` the same table is used for all channels,
        otherwise a channelwise table is used and ``C`` is expected to match
        the number of channels.

    Returns
    -------
    ndarray
        Image after mapping via lookup table.
        This *might* be the same array instance as provided via `image`.

    """

    image_shape_orig = image.shape
    nb_channels = 1 if len(image_shape_orig) == 2 else image_shape_orig[-1]

    if 0 in image_shape_orig:
        return image

    image = _normalize_cv2_input_arr_(image)

    # [(256,), (256,), ...] => (256, C)
    if isinstance(table, list):
        assert len(table) == nb_channels, (
            f"Expected to get {nb_channels} tables (one per channel), got {len(table)} instead."
        )
        table = np.stack(table, axis=-1)

    # (256, C) => (1, 256, C)
    if table.shape == (256, nb_channels):
        table = table[np.newaxis, :, :]

    assert table.shape == (256,) or table.shape == (1, 256, nb_channels), (
        "Expected 'table' to be any of the following: "
        "A list of C (256,) arrays, an array of shape (256,), an array of "
        "shape (256, C), an array of shape (1, 256, C). Transformed 'table' "
        f"up to shape {table.shape} for image with shape {image_shape_orig} (C={nb_channels})."
    )

    if nb_channels > 512:
        if table.shape == (256,):
            table = np.tile(table[np.newaxis, :, np.newaxis], (1, 1, nb_channels))

        subluts = []
        for group_idx in np.arange(int(np.ceil(nb_channels / 512))):
            c_start = group_idx * 512
            c_end = c_start + 512
            subluts.append(apply_lut_(image[:, :, c_start:c_end], table[:, :, c_start:c_end]))

        return np.concatenate(subluts, axis=2)

    assert image.dtype == _UINT8_DTYPE, f"Expected uint8 image, got dtype {image.dtype.name}."

    image = cv2.LUT(image, table, dst=image)
    return image


@legacy
def _identity_decorator(  # noqa: ANN401 - signature mirrors numba.jit and accepts any args.
    *_dec_args: Any,  # noqa: ANN401 - decorator args are intentionally unconstrained.
    **_dec_kwargs: Any,  # noqa: ANN401 - decorator kwargs are intentionally unconstrained.
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def _decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return func(*args, **kwargs)

        return _wrapper

    return _decorator


if numba is not None:
    _numbajit = numba.jit
else:
    _numbajit = _identity_decorator


@legacy
class HooksImages:
    """Class to intervene with image augmentation runs.

    This is e.g. useful to dynamically deactivate some augmenters.

    Parameters
    ----------
    activator : None or callable, optional
        A function that gives permission to execute an augmenter.
        The expected interface is::

            ``f(images, augmenter, parents, default)``

        where ``images`` are the input images to augment, ``augmenter`` is the
        instance of the augmenter to execute, ``parents`` are previously
        executed augmenters and ``default`` is an expected default value to be
        returned if the activator function does not plan to make a decision
        for the given inputs.

    propagator : None or callable, optional
        A function that gives permission to propagate the augmentation further
        to the children of an augmenter. This happens after the activator.
        In theory, an augmenter may augment images itself (if allowed by the
        activator) and then execute child augmenters afterwards (if allowed by
        the propagator). If the activator returned ``False``, the propagation
        step will never be executed.
        The expected interface is::

            ``f(images, augmenter, parents, default)``

        with all arguments having identical meaning to the activator.

    preprocessor : None or callable, optional
        A function to call before an augmenter performed any augmentations.
        The interface is:

            ``f(images, augmenter, parents)``

        with all arguments having identical meaning to the activator.
        It is expected to return the input images, optionally modified.

    postprocessor : None or callable, optional
        A function to call after an augmenter performed augmentations.
        The interface is the same as for the `preprocessor`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug2 as ia
    >>> import imgaug2.augmenters as iaa
    >>> seq = iaa.Sequential([
    >>>     iaa.GaussianBlur(3.0, name="blur"),
    >>>     iaa.Dropout(0.05, name="dropout"),
    >>>     iaa.Affine(translate_px=-5, name="affine")
    >>> ])
    >>> images = [np.zeros((10, 10), dtype=np.uint8)]
    >>>
    >>> def activator(images, augmenter, parents, default):
    >>>     return False if augmenter.name in ["blur", "dropout"] else default
    >>>
    >>> seq_det = seq.to_deterministic()
    >>> images_aug = seq_det.augment_images(images)
    >>> heatmaps = [np.random.rand(*(3, 10, 10))]
    >>> heatmaps_aug = seq_det.augment_images(
    >>>     heatmaps,
    >>>     hooks=ia.HooksImages(activator=activator)
    >>> )

    This augments images and their respective heatmaps in the same way.
    The heatmaps however are only modified by ``Affine``, not by
    ``GaussianBlur`` or ``Dropout``.

    """

    def __init__(
        self,
        activator: HookActivator | None = None,
        propagator: HookActivator | None = None,
        preprocessor: HookProcessor | None = None,
        postprocessor: HookProcessor | None = None,
    ) -> None:
        self.activator = activator
        self.propagator = propagator
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def is_activated(
        self,
        images: ImagesInput,
        augmenter: Augmenter,
        parents: Parents,
        default: bool,
    ) -> bool:
        """Estimate whether an augmenter may be executed.

        This also affects propagation of data to child augmenters.

        Returns
        -------
        bool
            If ``True``, the augmenter may be executed.
            Otherwise ``False``.

        """
        if self.activator is None:
            return default
        return self.activator(images, augmenter, parents, default)

    def is_propagating(
        self,
        images: ImagesInput,
        augmenter: Augmenter,
        parents: Parents,
        default: bool,
    ) -> bool:
        """Estimate whether an augmenter may call its children.

        This function decides whether an augmenter with children is allowed
        to call these in order to further augment the inputs.
        Note that if the augmenter itself performs augmentations (before/after
        calling its children), these may still be executed, even if this
        method returns ``False``.

        Returns
        -------
        bool
            If ``True``, the augmenter may propagate data to its children.
            Otherwise ``False``.

        """
        if self.propagator is None:
            return default
        return self.propagator(images, augmenter, parents, default)

    def preprocess(
        self,
        images: ImagesInput,
        augmenter: Augmenter,
        parents: Parents,
    ) -> ImagesInput:
        """Preprocess input data per augmenter before augmentation.

        Returns
        -------
        (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            The input images, optionally modified.

        """
        if self.preprocessor is None:
            return images
        return self.preprocessor(images, augmenter, parents)

    def postprocess(
        self,
        images: ImagesInput,
        augmenter: Augmenter,
        parents: Parents,
    ) -> ImagesInput:
        """Postprocess input data per augmenter after augmentation.

        Returns
        -------
        (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            The input images, optionally modified.

        """
        if self.postprocessor is None:
            return images
        return self.postprocessor(images, augmenter, parents)


@legacy
class HooksHeatmaps(HooksImages):
    """Class to intervene with heatmap augmentation runs.

    This is e.g. useful to dynamically deactivate some augmenters.

    This class is currently the same as the one for images. This may or may
    not change in the future.

    """


@legacy
class HooksKeypoints(HooksImages):
    """Class to intervene with keypoint augmentation runs.

    This is e.g. useful to dynamically deactivate some augmenters.

    This class is currently the same as the one for images. This may or may
    not change in the future.

    """


@legacy
class HooksSegmentationMaps(HooksImages):
    """Class to intervene with segmentation map augmentation runs."""


@legacy
class HooksPolygons(HooksImages):
    """Class to intervene with polygon augmentation runs."""


@legacy
class HooksLineStrings(HooksImages):
    """Class to intervene with line string augmentation runs."""


@legacy
class HooksBoundingBoxes(HooksImages):
    """Class to intervene with bounding box augmentation runs."""


#####################################################################
# Create classes/functions that were moved to other files.
#####################################################################


@legacy
def _is_moved_class_name(name: str) -> bool:
    name_stripped = name.lstrip("_")
    return bool(name_stripped) and name_stripped[0].isupper()


class _MovedClassProxyMeta(type):
    def _resolve_target(cls) -> type:
        target = getattr(cls, "_moved_target", None)
        if target is None:
            module = importlib.import_module(cls._moved_module_name_new)
            target = getattr(module, cls._moved_attr_name_new)
            cls._moved_target = target
        return target

    def __call__(  # noqa: ANN401 - dynamic proxy forwards to target signature.
        cls,
        *args: Any,  # noqa: ANN401 - proxy forwards target args.
        **kwargs: Any,  # noqa: ANN401 - proxy forwards target kwargs.
    ) -> Any:  # noqa: ANN401 - proxy returns target instance.
        warn_deprecated(
            f"`imgaug2.imgaug.{cls.__name__}` is deprecated. Use `{cls._moved_alt_func}` instead.",
            stacklevel=3,
        )
        return cls._resolve_target()(*args, **kwargs)

    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, cls._resolve_target())

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, cls._resolve_target())

    def __getattr__(cls, item: str) -> Any:  # noqa: ANN401 - dynamic proxy attribute access.
        return getattr(cls._resolve_target(), item)


def _mark_moved_class_or_function(  # noqa: ANN401 - return depends on moved target signature.
    name_old: str,
    module_name_new: str,
    name_new: str | None,
) -> Callable[..., Any] | type:
    name_new = name_new if name_new is not None else name_old

    if _is_moved_class_name(name_old):

        def __mro_entries__(
            cls: _MovedClassProxyMeta, bases: tuple[type, ...]
        ) -> tuple[type, ...]:
            return (cls._resolve_target(),)

        return _MovedClassProxyMeta(
            name_old,
            (),
            {
                "_moved_module_name_new": module_name_new,
                "_moved_attr_name_new": name_new,
                "_moved_alt_func": f"{module_name_new}.{name_new}",
                "_moved_target": None,
                "__mro_entries__": __mro_entries__,
                "__module__": __name__,
            },
        )

    def _func(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - dynamic proxy signature.
        module = importlib.import_module(module_name_new)
        return getattr(module, name_new)(*args, **kwargs)

    # These are legacy access paths that are kept for backwards compatibility,
    # but should emit a deprecation warning to guide users towards the new location.
    _func = deprecated(f"{module_name_new}.{name_new}")(_func)

    return _func


MOVED = [
    ("Keypoint", "imgaug2.augmentables.kps", None),
    ("KeypointsOnImage", "imgaug2.augmentables.kps", None),
    ("BoundingBox", "imgaug2.augmentables.bbs", None),
    ("BoundingBoxesOnImage", "imgaug2.augmentables.bbs", None),
    ("LineString", "imgaug2.augmentables.lines", None),
    ("LineStringsOnImage", "imgaug2.augmentables.lines", None),
    ("Polygon", "imgaug2.augmentables.polys", None),
    ("PolygonsOnImage", "imgaug2.augmentables.polys", None),
    ("MultiPolygon", "imgaug2.augmentables.polys", None),
    ("_ConcavePolygonRecoverer", "imgaug2.augmentables.polys", None),
    ("HeatmapsOnImage", "imgaug2.augmentables.heatmaps", None),
    ("SegmentationMapsOnImage", "imgaug2.augmentables.segmaps", None),
    ("Batch", "imgaug2.augmentables.batches", None),
    ("BatchLoader", "imgaug2.multicore", None),
    ("BackgroundAugmenter", "imgaug2.multicore", None),
    ("compute_geometric_median", "imgaug2.augmentables.kps", None),
    ("_convert_points_to_shapely_line_string", "imgaug2.augmentables.polys", None),
    ("_interpolate_point_pair", "imgaug2.augmentables.polys", None),
    ("_interpolate_points", "imgaug2.augmentables.polys", None),
    ("_interpolate_points_by_max_distance", "imgaug2.augmentables.polys", None),
    ("pad", "imgaug2.augmenters.size", None),
    ("pad_to_aspect_ratio", "imgaug2.augmenters.size", None),
    ("pad_to_multiples_of", "imgaug2.augmenters.size", None),
    (
        "compute_paddings_for_aspect_ratio",
        "imgaug2.augmenters.size",
        "compute_paddings_to_reach_aspect_ratio",
    ),
    ("compute_paddings_to_reach_multiples_of", "imgaug2.augmenters.size", None),
    ("compute_paddings_to_reach_exponents_of", "imgaug2.augmenters.size", None),
    ("quokka", "imgaug2.data", None),
    ("quokka_square", "imgaug2.data", None),
    ("quokka_heatmap", "imgaug2.data", None),
    ("quokka_segmentation_map", "imgaug2.data", None),
    ("quokka_keypoints", "imgaug2.data", None),
    ("quokka_bounding_boxes", "imgaug2.data", None),
    ("quokka_polygons", "imgaug2.data", None),
]

# These loop variables intentionally leak into the module scope in python.
# Define them explicitly so that static type checkers can reliably detect them.
class_name_old: str | None = None
module_name_new: str | None = None
class_name_new: str | None = None

for class_name_old, module_name_new, class_name_new in MOVED:
    locals()[class_name_old] = _mark_moved_class_or_function(
        class_name_old, module_name_new, class_name_new
    )
