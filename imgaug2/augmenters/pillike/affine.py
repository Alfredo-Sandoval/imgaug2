from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import PIL.Image

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
import imgaug2.augmenters.geometric as geometric
import imgaug2.augmenters.meta as meta
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters._typing import Array, Images, ParamInput, RNGInput
from imgaug2.compat.markers import legacy

from ._types import AffineParam, AffineParamOrNone, FillColor
from ._utils import _ensure_valid_shape

if TYPE_CHECKING:
    from imgaug2.augmenters.geometric import _AffineSamplingResult


@legacy(version="0.4.0")
def _create_affine_matrix(
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    translate_x_px: float = 0,
    translate_y_px: float = 0,
    rotate_deg: float = 0,
    shear_x_deg: float = 0,
    shear_y_deg: float = 0,
    center_px: tuple[float, float] = (0, 0),
) -> Array:
    from imgaug2.augmenters.geometry.affine import _RAD_PER_DEGREE, _AffineMatrixGenerator

    scale_x = max(scale_x, 0.0001)
    scale_y = max(scale_y, 0.0001)

    rotate_rad = rotate_deg * _RAD_PER_DEGREE
    shear_x_rad = shear_x_deg * _RAD_PER_DEGREE
    shear_y_rad = shear_y_deg * _RAD_PER_DEGREE

    matrix_gen = _AffineMatrixGenerator()
    matrix_gen.translate(x_px=-center_px[0], y_px=-center_px[1])
    matrix_gen.scale(x_frac=scale_x, y_frac=scale_y)
    matrix_gen.translate(x_px=translate_x_px, y_px=translate_y_px)
    matrix_gen.shear(x_rad=-shear_x_rad, y_rad=shear_y_rad)
    matrix_gen.rotate(rotate_rad)
    matrix_gen.translate(x_px=center_px[0], y_px=center_px[1])

    matrix = matrix_gen.matrix
    matrix = np.linalg.inv(matrix)

    return matrix




@legacy(version="0.4.0")
def warp_affine(
    image: Array,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    translate_x_px: float = 0,
    translate_y_px: float = 0,
    rotate_deg: float = 0,
    shear_x_deg: float = 0,
    shear_y_deg: float = 0,
    fillcolor: FillColor = None,
    center: tuple[float, float] = (0.5, 0.5),
) -> Array:
    """Apply an affine transformation to an image.

    This function has identical outputs to
    ``PIL.Image.transform`` with ``method=PIL.Image.AFFINE``.


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
        The image to modify. Expected to be ``uint8`` with shape ``(H,W)``
        or ``(H,W,C)`` with ``C`` being ``3`` or ``4``.

    scale_x : number, optional
        Affine scale factor along the x-axis, where ``1.0`` denotes an
        identity transform and ``2.0`` is a strong zoom-in effect.

    scale_y : number, optional
        Affine scale factor along the y-axis, where ``1.0`` denotes an
        identity transform and ``2.0`` is a strong zoom-in effect.

    translate_x_px : number, optional
        Affine translation along the x-axis in pixels.
        Positive values translate the image towards the right.

    translate_y_px : number, optional
        Affine translation along the y-axis in pixels.
        Positive values translate the image towards the bottom.

    rotate_deg : number, optional
        Affine rotation in degrees *around the top left* of the image.

    shear_x_deg : number, optional
        Affine shearing in degrees along the x-axis with center point
        being the top-left of the image.

    shear_y_deg : number, optional
        Affine shearing in degrees along the y-axis with center point
        being the top-left of the image.

    fillcolor : None or int or tuple of int, optional
        Color tuple or intensity value to use when filling up newly
        created pixels. ``None`` fills with zeros. ``int`` will only fill
        the ``0`` th channel with that intensity value and all other channels
        with ``0`` (this is the default behaviour of PIL, use a tuple to
        fill all channels).

    center : tuple of number, optional
        Center xy-coordinate of the affine transformation, given as *relative*
        values, i.e. ``(0.0, 0.0)`` sets the transformation center to the
        top-left image corner, ``(1.0, 0.0)`` sets it to the the top-right
        image corner and ``(0.5, 0.5)`` sets it to the image center.
        The transformation center is relevant e.g. for rotations ("rotate
        around this center point"). PIL uses the image top-left corner
        as the transformation center if no centerization is included in the
        affine transformation matrix.

    Returns
    -------
    ndarray
        Image after affine transformation.

    """
    iadt.allow_only_uint8({image.dtype})

    if 0 in image.shape:
        return np.copy(image)

    fillcolor = fillcolor if fillcolor is not None else 0
    if ia.is_iterable(fillcolor):
        # make sure that iterable fillcolor contains only ints
        # otherwise we get a deprecation warning in py3.8
        fillcolor = tuple(map(int, fillcolor))

    image, is_hw1 = _ensure_valid_shape(image, "imgaug2.augmenters.pillike.warp_affine()")

    image_pil = PIL.Image.fromarray(image)

    height, width = image.shape[0:2]
    center_px = (width * center[0], height * center[1])
    matrix = _create_affine_matrix(
        scale_x=scale_x,
        scale_y=scale_y,
        translate_x_px=translate_x_px,
        translate_y_px=translate_y_px,
        rotate_deg=rotate_deg,
        shear_x_deg=shear_x_deg,
        shear_y_deg=shear_y_deg,
        center_px=center_px,
    )
    matrix = matrix[:2, :].flat

    # don't return np.asarray(...) as its results are read-only
    result = np.array(
        image_pil.transform(image_pil.size, PIL.Image.AFFINE, matrix, fillcolor=fillcolor)
    )

    if is_hw1:
        result = result[:, :, np.newaxis]
    return result


# we don't use pil_solarize() here. but instead just subclass Invert,
# which is easier and comes down to the same


@legacy(version="0.4.0")
class Affine(geometric.Affine):
    """Apply PIL-like affine transformations to images.

    This augmenter has identical outputs to
    ``PIL.Image.transform`` with parameter ``method=PIL.Image.AFFINE``.

    .. warning::

        This augmenter can currently only transform image-data.
        Batches containing heatmaps, segmentation maps and
        coordinate-based augmentables will be rejected with an error.
        Use :class:`~imgaug2.augmenters.geometric.Affine` if you have to
        transform such inputs.

    .. note::

        This augmenter uses the image center as the transformation center.
        This has to be explicitly enforced in PIL using corresponding
        translation matrices. Without such translation, PIL uses the image
        top left corner as the transformation center. To mirror that
        behaviour, use ``center=(0.0, 0.0)``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.pillike.warp_affine`.

    Parameters
    ----------
    scale : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        See :class:`~imgaug2.augmenters.geometric.Affine`.

    translate_percent : None or number or tuple of number or list of number or imgaug2.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        See :class:`~imgaug2.augmenters.geometric.Affine`.

    translate_px : None or int or tuple of int or list of int or imgaug2.parameters.StochasticParameter or dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter}, optional
        See :class:`~imgaug2.augmenters.geometric.Affine`.

    rotate : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.geometric.Affine`.

    shear : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter or dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter}, optional
        See :class:`~imgaug2.augmenters.geometric.Affine`.

    fillcolor : number or tuple of number or list of number or imgaug2.ALL or imgaug2.parameters.StochasticParameter, optional
        See parameter ``cval`` in :class:`~imgaug2.augmenters.geometric.Affine`.

    center : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        The center point of the affine transformation, given as relative
        xy-coordinates.
        Set this to ``(0.0, 0.0)`` or ``left-top`` to use the top left image
        corner as the transformation center.
        Set this to ``(0.5, 0.5)`` or ``center-center`` to use the image
        center as the transformation center.
        See also paramerer ``position`` in
        :class:`~imgaug2.augmenters.size.PadToFixedSize` for details
        about valid datatypes of this parameter.

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
    >>> aug = iaa.pillike.Affine(scale={"x": (0.8, 1.2), "y": (0.5, 1.5)})

    Create an augmenter that applies affine scaling (zoom in/out) to images.
    Along the x-axis they are scaled to 80-120% of their size, along
    the y-axis to 50-150% (both values randomly and uniformly chosen per
    image).

    >>> aug = iaa.pillike.Affine(translate_px={"x": 0, "y": [-10, 10]},
    >>>                          fillcolor=128)

    Create an augmenter that translates images along the y-axis by either
    ``-10px`` or ``10px``. Newly created pixels are always filled with
    the value ``128`` (along all channels).

    >>> aug = iaa.pillike.Affine(rotate=(-20, 20), fillcolor=(0, 256))

    Rotate an image by ``-20`` to ``20`` degress and fill up all newly
    created pixels with a random RGB color.

    See the similar augmenter :class:`~imgaug2.augmenters.geometric.Affine`
    for more examples.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        scale: AffineParam = 1.0,
        translate_percent: AffineParamOrNone = None,
        translate_px: AffineParamOrNone = None,
        rotate: ParamInput = 0.0,
        shear: AffineParam = 0.0,
        fillcolor: ParamInput = 0,
        center: object = (0.5, 0.5),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            scale=scale,
            translate_percent=translate_percent,
            translate_px=translate_px,
            rotate=rotate,
            shear=shear,
            order=1,
            cval=fillcolor,
            mode="constant",
            fit_output=False,
            backend="auto",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )
        self.center = iap.handle_position_parameter(center)

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        cols = batch.get_column_names()
        assert len(cols) == 0 or (len(cols) == 1 and "images" in cols), (
            "pillike.Affine can currently only process image data. Got a "
            "batch containing: {}. Use imgaug2.augmenters.geometric.Affine for "
            "batches containing non-image data.".format(", ".join(cols))
        )

        return super()._augment_batch_(batch, random_state, parents, hooks)

    @legacy(version="0.4.0")
    def _augment_images_by_samples(
        self,
        images: Images,
        samples: _AffineSamplingResult,
        image_shapes: Sequence[geometric.Shape] | None = None,
        return_matrices: bool = False,
    ) -> Images | tuple[Images, list[geometric.Matrix]]:
        assert return_matrices is False, (
            "Got unexpectedly return_matrices=True. pillike.Affine does not "
            "yet produce that output."
        )

        from imgaug2.augmenters import pillike as pillike_lib

        for i, image in enumerate(images):
            image_shape = image.shape if image_shapes is None else image_shapes[i]

            params = samples.get_affine_parameters(
                i, arr_shape=image_shape, image_shape=image_shape
            )

            image[...] = pillike_lib.warp_affine(
                image,
                scale_x=params["scale_x"],
                scale_y=params["scale_y"],
                translate_x_px=params["translate_x_px"],
                translate_y_px=params["translate_y_px"],
                rotate_deg=params["rotate_deg"],
                shear_x_deg=params["shear_x_deg"],
                shear_y_deg=params["shear_y_deg"],
                fillcolor=tuple(samples.cval[i]),
                center=(samples.center_x[i], samples.center_y[i]),
            )

        return images

    @legacy(version="0.4.0")
    def _draw_samples(self, nb_samples: int, random_state: iarandom.RNG) -> _AffineSamplingResult:
        # standard affine samples
        samples = super()._draw_samples(nb_samples, random_state)

        # add samples for 'center' parameter, which is not yet a part of
        # Affine
        if isinstance(self.center, tuple):
            xx = self.center[0].draw_samples(nb_samples, random_state=random_state)
            yy = self.center[1].draw_samples(nb_samples, random_state=random_state)
        else:
            xy = self.center.draw_samples((nb_samples, 2), random_state=random_state)
            xx = xy[:, 0]
            yy = xy[:, 1]

        samples.center_x = xx
        samples.center_y = yy
        return samples

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.scale, self.translate, self.rotate, self.shear, self.cval, self.center]
