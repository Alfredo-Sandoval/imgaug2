from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.compat.markers import legacy
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, Images, ParamInput, RNGInput
from ._utils import (
    CValInput,
    FillModeInput,
    IntParamInput,
    PerChannelInput,
    PositionInput,
    SizePercentInput,
    SizePxInput,
)
from .multiply import MultiplyElementwise

# fill modes for apply_cutout_() and Cutout augmenter
# contains roughly:
#     'str fill_mode_name => (str module_name, str function_name)'
# We could also assign the function to each fill mode name instead of its
# name, but that has the disadvantage that these aren't defined yet (they
# are defined further below) and that during unittesting they would be harder
# to mock. (mock.patch() seems to not automatically replace functions
# assigned in that way.)
_CUTOUT_FILL_MODES = {
    "constant": ("imgaug2.augmenters.arithmetic", "_fill_rectangle_constant_"),
    "gaussian": ("imgaug2.augmenters.arithmetic", "_fill_rectangle_gaussian_"),
}


@legacy(version="0.4.0")
def cutout(
    image: Array,
    x1: float | int,
    y1: float | int,
    x2: float | int,
    y2: float | int,
    fill_mode: FillModeInput = "constant",
    cval: CValInput = 0,
    fill_per_channel: bool | float = False,
    seed: RNGInput = None,
) -> Array:
    """Fill a single area within an image using a fill mode.

    This cutout method uses the top-left and bottom-right corner coordinates
    of the cutout region given as absolute pixel values.

    .. note::

        Gaussian fill mode will assume that float input images contain values
        in the interval ``[0.0, 1.0]`` and hence sample values from a
        gaussian within that interval, i.e. from ``N(0.5, std=0.5/3)``.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.arithmetic.cutout_`.


    Parameters
    ----------
    image : ndarray
        Image to modify.

    x1 : number
        See :func:`~imgaug2.augmenters.arithmetic.cutout_`.

    y1 : number
        See :func:`~imgaug2.augmenters.arithmetic.cutout_`.

    x2 : number
        See :func:`~imgaug2.augmenters.arithmetic.cutout_`.

    y2 : number
        See :func:`~imgaug2.augmenters.arithmetic.cutout_`.

    fill_mode : {'constant', 'gaussian'}, optional
        See :func:`~imgaug2.augmenters.arithmetic.cutout_`.

    cval : number or tuple of number, optional
        See :func:`~imgaug2.augmenters.arithmetic.cutout_`.

    fill_per_channel : number or bool, optional
        See :func:`~imgaug2.augmenters.arithmetic.cutout_`.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See :func:`~imgaug2.augmenters.arithmetic.cutout_`.

    Returns
    -------
    ndarray
        Image with area filled in.

    """
    from . import cutout_ as cutout_fn

    return cutout_fn(np.copy(image), x1, y1, x2, y2, fill_mode, cval, fill_per_channel, seed)


@legacy(version="0.4.0")
def cutout_(
    image: Array,
    x1: float | int,
    y1: float | int,
    x2: float | int,
    y2: float | int,
    fill_mode: FillModeInput = "constant",
    cval: CValInput = 0,
    fill_per_channel: bool | float = False,
    seed: RNGInput = None,
) -> Array:
    """Fill a single area within an image using a fill mode (in-place).

    This cutout method uses the top-left and bottom-right corner coordinates
    of the cutout region given as absolute pixel values.

    .. note::

        Gaussian fill mode will assume that float input images contain values
        in the interval ``[0.0, 1.0]`` and hence sample values from a
        gaussian within that interval, i.e. from ``N(0.5, std=0.5/3)``.


    **Supported dtypes**:

    minimum of (
        :func:`~imgaug2.augmenters.arithmetic._fill_rectangle_gaussian_`,
        :func:`~imgaug2.augmenters.arithmetic._fill_rectangle_constant_`
    )

    Parameters
    ----------
    image : ndarray
        Image to modify. Might be modified in-place.

    x1 : number
        X-coordinate of the top-left corner of the cutout region.

    y1 : number
        Y-coordinate of the top-left corner of the cutout region.

    x2 : number
        X-coordinate of the bottom-right corner of the cutout region.

    y2 : number
        Y-coordinate of the bottom-right corner of the cutout region.

    fill_mode : {'constant', 'gaussian'}, optional
        Fill mode to use.

    cval : number or tuple of number, optional
        The constant value to use when filling with mode ``constant``.
        May be an intensity value or color tuple.

    fill_per_channel : number or bool, optional
        Whether to fill in a channelwise fashion.
        If number then a value ``>=0.5`` will be interpreted as ``True``.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        A random number generator to sample random values from.
        Usually an integer seed value or an ``RNG`` instance.
        See :class:`imgaug2.random.RNG` for details.

    Returns
    -------
    ndarray
        Image with area filled in.
        The input image might have been modified in-place.

    """
    import importlib

    height, width = image.shape[0:2]
    x1 = min(max(int(x1), 0), width)
    y1 = min(max(int(y1), 0), height)
    x2 = min(max(int(x2), 0), width)
    y2 = min(max(int(y2), 0), height)

    if x2 > x1 and y2 > y1:
        assert fill_mode in _CUTOUT_FILL_MODES, (
            f"Expected one of the following fill modes: {str(list(_CUTOUT_FILL_MODES.keys()))}. "
            f"Got: {fill_mode}."
        )

        module_name, fname = _CUTOUT_FILL_MODES[fill_mode]
        module = importlib.import_module(module_name)
        func = getattr(module, fname)
        image = func(
            image,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            cval=cval,
            per_channel=(fill_per_channel >= 0.5),
            random_state=(
                iarandom.RNG(seed) if not isinstance(seed, iarandom.RNG) else seed
            ),  # only RNG(.) without "if" is ~8x slower
        )
    return image


@legacy(version="0.4.0")
def _fill_rectangle_gaussian_(
    image: Array,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    cval: CValInput,
    per_channel: bool,
    random_state: iarandom.RNG,
) -> Array:
    """Fill a rectangular image area with samples from a gaussian.


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: limited; tested (1)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: limited; tested (1)
        * ``float16``: yes; tested (2)
        * ``float32``: yes; tested (2)
        * ``float64``: yes; tested (2)
        * ``float128``: limited; tested (1) (2)
        * ``bool``: yes; tested

        - (1) Possible loss of resolution due to gaussian values being sampled
              as ``float64`` s.
        - (2) Float input arrays are assumed to be in interval ``[0.0, 1.0]``
              and all gaussian samples are within that interval too.

    """
    # for float we assume value range [0.0, 1.0]
    # that matches the common use case and also makes the tests way easier
    # we also set bool here manually as the center value returned by
    # get_value_range_for_dtype() is None
    kind = image.dtype.kind
    if kind in ["f", "b"]:
        min_value = 0.0
        center_value = 0.5
        max_value = 1.0
    else:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(image.dtype)

    # set standard deviation to 1/3 of value range to get 99.7% of values
    # within [min v.r., max v.r.]
    # we also divide by 2 because we want to spread towards the
    # "left"/"right" of the center value by half of the value range
    stddev = (float(max_value) - float(min_value)) / 2.0 / 3.0

    height = y2 - y1
    width = x2 - x1
    shape = (height, width)
    if per_channel and image.ndim == 3:
        shape = shape + (image.shape[2],)
    rect = random_state.normal(center_value, stddev, size=shape)
    if image.dtype.kind == "b":
        rect_vr = rect > 0.5
    else:
        rect_vr = np.clip(rect, min_value, max_value).astype(image.dtype)

    if image.ndim == 3:
        image[y1:y2, x1:x2, :] = np.atleast_3d(rect_vr)
    else:
        image[y1:y2, x1:x2] = rect_vr

    return image


@legacy(version="0.4.0")
def _fill_rectangle_constant_(
    image: Array,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    cval: CValInput,
    per_channel: bool,
    random_state: iarandom.RNG,
) -> Array:
    """Fill a rectangular area within an image with constant value(s).

    `cval` may be a single value or one per channel. If the number of items
    in `cval` does not match the number of channels in `image`, it may
    be tiled up to the number of channels.


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    """
    if ia.is_iterable(cval):
        if per_channel:
            nb_channels = None if image.ndim == 2 else image.shape[-1]
            if nb_channels is None:
                cval = cval[0]
            elif len(cval) < nb_channels:
                mul = int(np.ceil(nb_channels / len(cval)))
                cval = np.tile(cval, (mul,))[0:nb_channels]
            elif len(cval) > nb_channels:
                cval = cval[0:nb_channels]
        else:
            cval = cval[0]

    # without the array(), uint64 max value is assigned as 0
    image[y1:y2, x1:x2, ...] = np.array(cval, dtype=image.dtype)

    return image


@legacy(version="0.4.0")
class _CutoutSamples:
    @legacy(version="0.4.0")
    def __init__(
        self,
        nb_iterations: Array,
        pos_x: Array,
        pos_y: Array,
        size_h: Array,
        size_w: Array,
        squared: Array,
        fill_mode: Array,
        cval: Array,
        fill_per_channel: Array,
    ) -> None:
        self.nb_iterations = nb_iterations
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.size_h = size_h
        self.size_w = size_w
        self.squared = squared
        self.fill_mode = fill_mode
        self.cval = cval
        self.fill_per_channel = fill_per_channel


@legacy(version="0.4.0")
class Cutout(meta.Augmenter):
    """Fill one or more rectangular areas in an image using a fill mode.

    See paper "Improved Regularization of Convolutional Neural Networks with
    Cutout" by DeVries and Taylor.

    In contrast to the paper, this implementation also supports replacing
    image sub-areas with gaussian noise, random intensities or random RGB
    colors. It also supports non-squared areas. While the paper uses
    absolute pixel values for the size and position, this implementation
    uses relative values, which seems more appropriate for mixed-size
    datasets. The position parameter furthermore allows more flexibility, e.g.
    gaussian distributions around the center.

    .. note::

        This augmenter affects only image data. Other datatypes (e.g.
        segmentation map pixels or keypoints within the filled areas)
        are not affected.

    .. note::

        Gaussian fill mode will assume that float input images contain values
        in the interval ``[0.0, 1.0]`` and hence sample values from a
        gaussian within that interval, i.e. from ``N(0.5, std=0.5/3)``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.arithmetic.cutout_`.

    Parameters
    ----------
    nb_iterations : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        How many rectangular areas to fill.

            * If ``int``: Exactly that many areas will be filled on all images.
            * If ``tuple`` ``(a, b)``: A value from the interval ``[a, b]``
              will be sampled per image.
            * If ``list``: A random value will be sampled from that ``list``
              per image.
            * If ``StochasticParameter``: That parameter will be used to
              sample ``(B,)`` values per batch of ``B`` images.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        Defines the position of each area to fill.
        Analogous to the definition in e.g.
        :class:`~imgaug2.augmenters.size.CropToFixedSize`.
        Usually, ``uniform`` (anywhere in the image) or ``normal`` (anywhere
        in the image with preference around the center) are sane values.

    size : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        The size of the rectangle to fill as a fraction of the corresponding
        image size, i.e. with value range ``[0.0, 1.0]``. The size is sampled
        independently per image axis.

            * If ``number``: Exactly that size is always used.
            * If ``tuple`` ``(a, b)``: A value from the interval ``[a, b]``
              will be sampled per area and axis.
            * If ``list``: A random value will be sampled from that ``list``
              per area and axis.
            * If ``StochasticParameter``: That parameter will be used to
              sample ``(N, 2)`` values per batch, where ``N`` is the total
              number of areas to fill within the whole batch.

    squared : bool or float or imgaug2.parameters.StochasticParameter, optional
        Whether to generate only squared areas cutout areas or allow
        rectangular ones. If this evaluates to a true-like value, the
        first value from `size` will be converted to absolute pixels and used
        for both axes.

        If this value is a float ``p``, then for ``p`` percent of all areas
        to be filled `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    fill_mode : str or list of str or imgaug2.parameters.StochasticParameter, optional
        Mode to use in order to fill areas. Corresponds to ``mode`` parameter
        in some other augmenters. Valid strings for the mode are:

            * ``contant``: Fill each area with a single value.
            * ``gaussian``: Fill each area with gaussian noise.

        Valid datatypes are:

            * If ``str``: Exactly that mode will alaways be used.
            * If ``list``: A random value will be sampled from that ``list``
              per area.
            * If ``StochasticParameter``: That parameter will be used to
              sample ``(N,)`` values per batch, where ``N`` is the total number
              of areas to fill within the whole batch.

    cval : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        The value to use (i.e. the color) to fill areas if `fill_mode` is
        ```constant``.

            * If ``number``: Exactly that value is used for all areas
              and channels.
            * If ``tuple`` ``(a, b)``: A value from the interval ``[a, b]``
              will be sampled per area (and channel if ``per_channel=True``).
            * If ``list``: A random value will be sampled from that ``list``
              per area (and channel if ``per_channel=True``).
            * If ``StochasticParameter``: That parameter will be used to
              sample ``(N, Cmax)`` values per batch, where ``N`` is the total
              number of areas to fill within the whole batch and ``Cmax``
              is the maximum number of channels in any image (usually ``3``).
              If ``per_channel=False``, only the first value of the second
              axis is used.

    fill_per_channel : bool or float or imgaug2.parameters.StochasticParameter, optional
        Whether to fill each area in a channelwise fashion (``True``) or
        not (``False``).
        The behaviour per fill mode is:

            * ``constant``: Whether to fill all channels with the same value
              (i.e, grayscale) or different values (i.e. usually RGB color).
            * ``gaussian``: Whether to sample once from a gaussian and use the
              values for all channels (i.e. grayscale) or to sample
              channelwise (i.e. RGB colors)

        If this value is a float ``p``, then for ``p`` percent of all areas
        to be filled `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.Cutout(nb_iterations=2)

    Fill per image two random areas, by default with grayish pixels.

    >>> aug = iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)

    Fill per image between one and five areas, each having ``20%``
    of the corresponding size of the height and width (for non-square
    images this results in non-square areas to be filled).

    >>> aug = iaa.Cutout(fill_mode="constant", cval=255)

    Fill all areas with white pixels.

    >>> aug = iaa.Cutout(fill_mode="constant", cval=(0, 255),
    >>>                  fill_per_channel=0.5)

    Fill ``50%`` of all areas with a random intensity value between
    ``0`` and ``256``. Fill the other ``50%`` of all areas with
    random colors.

    >>> aug = iaa.Cutout(fill_mode="gaussian", fill_per_channel=True)

    Fill areas with gaussian channelwise noise (i.e. usually RGB).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        nb_iterations: IntParamInput = 1,
        position: PositionInput = "uniform",
        size: ParamInput = 0.2,
        squared: PerChannelInput = True,
        fill_mode: FillModeInput | Sequence[FillModeInput] | iap.StochasticParameter = "constant",
        cval: CValInput = 128,
        fill_per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.nb_iterations = iap.handle_discrete_param(
            nb_iterations,
            "nb_iterations",
            value_range=(0, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self.position = iap.handle_position_parameter(position)
        self.size = iap.handle_continuous_param(
            size, "size", value_range=(0.0, 1.0 + 1e-4), tuple_to_uniform=True, list_to_choice=True
        )
        self.squared = iap.handle_probability_param(squared, "squared")
        self.fill_mode = self._handle_fill_mode_param(fill_mode)
        self.cval = iap.handle_cval_arg(cval)
        self.fill_per_channel = iap.handle_probability_param(fill_per_channel, "fill_per_channel")

    @classmethod
    @legacy(version="0.4.0")
    def _handle_fill_mode_param(
        cls, fill_mode: FillModeInput | Sequence[FillModeInput] | iap.StochasticParameter
    ) -> iap.StochasticParameter:
        if ia.is_string(fill_mode):
            assert fill_mode in _CUTOUT_FILL_MODES, (
                f"Expected 'fill_mode' to be one of: {str(list(_CUTOUT_FILL_MODES.keys()))}. Got {fill_mode}."
            )
            return iap.Deterministic(fill_mode)
        if isinstance(fill_mode, iap.StochasticParameter):
            return fill_mode
        assert ia.is_iterable(fill_mode), (
            "Expected 'fill_mode' to be a string, "
            f"StochasticParameter or list of strings. Got type {type(fill_mode).__name__}."
        )
        return iap.Choice(fill_mode)

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

        samples = self._draw_samples(batch.images, random_state)

        # map from xyhw to xyxy (both relative coords)
        cutout_height_half = samples.size_h / 2
        cutout_width_half = samples.size_w / 2
        x1_rel = samples.pos_x - cutout_width_half
        y1_rel = samples.pos_y - cutout_height_half
        x2_rel = samples.pos_x + cutout_width_half
        y2_rel = samples.pos_y + cutout_height_half

        nb_iterations_sum = 0
        gen = enumerate(zip(batch.images, samples.nb_iterations, strict=True))
        for i, (image, nb_iterations) in gen:
            start = nb_iterations_sum
            end = start + nb_iterations

            height, width = image.shape[0:2]

            # map from relative xyxy to absolute xyxy coords
            batch.images[i] = self._augment_image_by_samples(
                image,
                x1_rel[start:end] * width,
                y1_rel[start:end] * height,
                x2_rel[start:end] * width,
                y2_rel[start:end] * height,
                samples.squared[start:end],
                samples.fill_mode[start:end],
                samples.cval[start:end],
                samples.fill_per_channel[start:end],
                random_state,
            )

            nb_iterations_sum += nb_iterations

        return batch

    @legacy(version="0.4.0")
    def _draw_samples(self, images: Images, random_state: iarandom.RNG) -> _CutoutSamples:
        rngs = random_state.duplicate(8)
        nb_rows = len(images)
        nb_channels_max = meta.estimate_max_number_of_channels(images)

        nb_iterations = self.nb_iterations.draw_samples((nb_rows,), random_state=rngs[0])
        nb_dropped_areas = int(np.sum(nb_iterations))

        if isinstance(self.position, tuple):
            pos_x = self.position[0].draw_samples((nb_dropped_areas,), random_state=rngs[1])
            pos_y = self.position[1].draw_samples((nb_dropped_areas,), random_state=rngs[2])
        else:
            pos = self.position.draw_samples((nb_dropped_areas, 2), random_state=rngs[1])
            pos_x = pos[:, 0]
            pos_y = pos[:, 1]

        size = self.size.draw_samples((nb_dropped_areas, 2), random_state=rngs[3])
        squared = self.squared.draw_samples((nb_dropped_areas,), random_state=rngs[4])
        fill_mode = self.fill_mode.draw_samples((nb_dropped_areas,), random_state=rngs[5])

        cval = self.cval.draw_samples((nb_dropped_areas, nb_channels_max), random_state=rngs[6])

        fill_per_channel = self.fill_per_channel.draw_samples(
            (nb_dropped_areas,), random_state=rngs[7]
        )

        return _CutoutSamples(
            nb_iterations=nb_iterations,
            pos_x=pos_x,
            pos_y=pos_y,
            size_h=size[:, 0],
            size_w=size[:, 1],
            squared=squared,
            fill_mode=fill_mode,
            cval=cval,
            fill_per_channel=fill_per_channel,
        )

    @classmethod
    @legacy(version="0.4.0")
    def _augment_image_by_samples(
        cls,
        image: Array,
        x1: Array,
        y1: Array,
        x2: Array,
        y2: Array,
        squared: Array,
        fill_mode: Array,
        cval: Array,
        fill_per_channel: Array,
        random_state: iarandom.RNG,
    ) -> Array:
        from . import cutout_ as cutout_fn

        for i, x1_i in enumerate(x1):
            x2_i = x2[i]
            if squared[i] >= 0.5:
                height_h = (y2[i] - y1[i]) / 2
                x_center = x1_i + (x2_i - x1_i) / 2
                x1_i = x_center - height_h
                x2_i = x_center + height_h

            image = cutout_fn(
                image,
                x1=x1_i,
                y1=y1[i],
                x2=x2_i,
                y2=y2[i],
                fill_mode=fill_mode[i],
                cval=cval[i],
                fill_per_channel=fill_per_channel[i],
                seed=random_state,
            )
        return image

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [
            self.nb_iterations,
            self.position,
            self.size,
            self.squared,
            self.fill_mode,
            self.cval,
            self.fill_per_channel,
        ]


# TODO verify that (a, b) still leads to a p being sampled per image and not
#      per batch
class Dropout(MultiplyElementwise):
    """
    Set a fraction of pixels in images to zero.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.arithmetic.MultiplyElementwise`.

    Parameters
    ----------
    p : float or tuple of float or imgaug2.parameters.StochasticParameter, optional
        The probability of any pixel being dropped (i.e. to set it to zero).

            * If a float, then that value will be used for all images. A value
              of ``1.0`` would mean that all pixels will be dropped
              and ``0.0`` that no pixels will be dropped. A value of ``0.05``
              corresponds to ``5`` percent of all pixels being dropped.
            * If a tuple ``(a, b)``, then a value ``p`` will be sampled from
              the interval ``[a, b]`` per image and be used as the pixel's
              dropout probability.
            * If a list, then a value will be sampled from that list per
              batch and used as the probability.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per pixel whether it should be *kept* (sampled value
              of ``>0.5``) or shouldn't be kept (sampled value of ``<=0.5``).
              If you instead want to provide the probability as a stochastic
              parameter, you can usually do ``imgaug2.parameters.Binomial(1-p)``
              to convert parameter `p` to a 0/1 representation.

    per_channel : bool or float or imgaug2.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

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
    >>> aug = iaa.Dropout(0.02)

    Drops ``2`` percent of all pixels.

    >>> aug = iaa.Dropout((0.0, 0.05))

    Drops in each image a random fraction of all pixels, where the fraction
    is uniformly sampled from the interval ``[0.0, 0.05]``.

    >>> aug = iaa.Dropout(0.02, per_channel=True)

    Drops ``2`` percent of all pixels in a channelwise fashion, i.e. it is
    unlikely for any pixel to have all channels set to zero (black pixels).

    >>> aug = iaa.Dropout(0.02, per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for ``50`` percent of all images.

    """

    def __init__(
        self,
        p: ParamInput = (0.0, 0.05),
        per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        p_param = _handle_dropout_probability_param(p, "p")

        super().__init__(
            p_param,
            per_channel=per_channel,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
def _handle_dropout_probability_param(p: ParamInput, name: str) -> iap.StochasticParameter:
    if ia.is_single_number(p):
        p_param = iap.Binomial(1 - p)
    elif isinstance(p, tuple):
        assert len(p) == 2, (
            f"Expected `{name}` to be given as a tuple containing exactly 2 values, "
            f"got {len(p)} values."
        )
        assert p[0] < p[1], (
            f"Expected `{name}` to be given as a tuple containing exactly 2 values "
            f"(a, b) with a < b. Got {p[0]:.4f} and {p[1]:.4f}."
        )
        assert 0 <= p[0] <= 1.0 and 0 <= p[1] <= 1.0, (
            f"Expected `{name}` given as tuple to only contain values in the "
            f"interval [0.0, 1.0], got {p[0]:.4f} and {p[1]:.4f}."
        )

        p_param = iap.Binomial(iap.Uniform(1 - p[1], 1 - p[0]))
    elif ia.is_iterable(p):
        assert all([ia.is_single_number(v) for v in p]), (
            f"Expected iterable parameter '{name}' to only contain numbers, "
            f"got {[type(v) for v in p]}."
        )
        assert all([0 <= p_i <= 1.0 for p_i in p]), (
            "Expected iterable parameter '{}' to only contain probabilities "
            "in the interval [0.0, 1.0], got values {}.".format(
                name, ", ".join([f"{p_i:.4f}" for p_i in p])
            )
        )
        p_param = iap.Binomial(1 - iap.Choice(p))
    elif isinstance(p, iap.StochasticParameter):
        p_param = p
    else:
        raise Exception(
            f"Expected `{name}` to be float or int or tuple (<number>, <number>) "
            f"or StochasticParameter, got type '{type(p).__name__}'."
        )

    return p_param


# TODO invert size_px and size_percent so that larger values denote larger
#      areas being dropped instead of the opposite way around
class CoarseDropout(MultiplyElementwise):
    """
    Set rectangular areas within images to zero.

    In contrast to ``Dropout``, these areas can have larger sizes.
    (E.g. you might end up with three large black rectangles in an image.)
    Note that the current implementation leads to correlated sizes,
    so if e.g. there is any thin and high rectangle that is dropped, there is
    a high likelihood that all other dropped areas are also thin and high.

    This method is implemented by generating the dropout mask at a
    lower resolution (than the image has) and then upsampling the mask
    before dropping the pixels.

    This augmenter is similar to Cutout. Usually, cutout is defined as an
    operation that drops exactly one rectangle from an image, while here
    ``CoarseDropout`` can drop multiple rectangles (with some correlation
    between the sizes of these rectangles).

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.arithmetic.MultiplyElementwise`.

    Parameters
    ----------
    p : float or tuple of float or imgaug2.parameters.StochasticParameter, optional
        The probability of any pixel being dropped (i.e. set to zero) in
        the lower-resolution dropout mask.

            * If a float, then that value will be used for all pixels. A value
              of ``1.0`` would mean, that all pixels will be dropped. A value
              of ``0.0`` would lead to no pixels being dropped.
            * If a tuple ``(a, b)``, then a value ``p`` will be sampled from
              the interval ``[a, b]`` per image and be used as the dropout
              probability.
            * If a list, then a value will be sampled from that list per
              batch and used as the probability.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per pixel whether it should be *kept* (sampled value
              of ``>0.5``) or shouldn't be kept (sampled value of ``<=0.5``).
              If you instead want to provide the probability as a stochastic
              parameter, you can usually do ``imgaug2.parameters.Binomial(1-p)``
              to convert parameter `p` to a 0/1 representation.

    size_px : None or int or tuple of int or imgaug2.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the dropout
        mask in absolute pixel dimensions.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being dropped (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_percent` must be set.
            * If an integer, then that size will always be used for both height
              and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
              which is then upsampled to ``HxW``, where ``H`` is the image size
              and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
              sampled from the discrete interval ``[a..b]``. The dropout mask
              will then be generated at size ``MxN`` and upsampled to ``HxW``.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : None or float or tuple of float or imgaug2.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the dropout
        mask *in percent* of the input image.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being dropped (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_px` must be set.
            * If a float, then that value will always be used as the percentage
              of the height and width (relative to the original size). E.g. for
              value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
              later upsampled to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
              sampled from the interval ``(a, b)`` and used as the size
              fractions, i.e the mask size will be ``(m*H)x(n*W)``.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float or imgaug2.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    min_size : int, optional
        Minimum height and width of the low resolution mask. If
        `size_percent` or `size_px` leads to a lower value than this,
        `min_size` will be used instead. This should never have a value of
        less than ``2``, otherwise one may end up with a ``1x1`` low resolution
        mask, leading easily to the whole image being dropped.

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
    >>> aug = iaa.CoarseDropout(0.02, size_percent=0.5)

    Drops ``2`` percent of all pixels on a lower-resolution image that has
    ``50`` percent of the original image's size, leading to dropped areas that
    have roughly ``2x2`` pixels size.

    >>> aug = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.05, 0.5))

    Generates a dropout mask at ``5`` to ``50`` percent of each input image's
    size. In that mask, ``0`` to ``5`` percent of all pixels are marked as
    being dropped. The mask is afterwards projected to the input image's
    size to apply the actual dropout operation.

    >>> aug = iaa.CoarseDropout((0.0, 0.05), size_px=(2, 16))

    Same as the previous example, but the lower resolution image has ``2`` to
    ``16`` pixels size. On images of e.g. ``224x224` pixels in size this would
    lead to fairly large areas being dropped (height/width of ``224/2`` to
    ``224/16``).

    >>> aug = iaa.CoarseDropout(0.02, size_percent=0.5, per_channel=True)

    Drops ``2`` percent of all pixels at ``50`` percent resolution (``2x2``
    sizes) in a channel-wise fashion, i.e. it is unlikely for any pixel to
    have all channels set to zero (black pixels).

    >>> aug = iaa.CoarseDropout(0.02, size_percent=0.5, per_channel=0.5)

    Same as the previous example, but the `per_channel` feature is only active
    for ``50`` percent of all images.

    """

    def __init__(
        self,
        p: ParamInput = (0.02, 0.1),
        size_px: SizePxInput = None,
        size_percent: SizePercentInput = None,
        per_channel: PerChannelInput = False,
        min_size: int = 3,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        p_param = _handle_dropout_probability_param(p, "p")

        if size_px is not None:
            p_param = iap.FromLowerResolution(
                other_param=p_param, size_px=size_px, min_size=min_size
            )
        elif size_percent is not None:
            p_param = iap.FromLowerResolution(
                other_param=p_param, size_percent=size_percent, min_size=min_size
            )
        else:
            # default if neither size_px nor size_percent is provided
            # is size_px=(3, 8)
            p_param = iap.FromLowerResolution(
                other_param=p_param, size_px=(3, 8), min_size=min_size
            )

        super().__init__(
            p_param,
            per_channel=per_channel,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
class Dropout2d(meta.Augmenter):
    """Drop random channels from images.

    For image data, dropped channels will be filled with zeros.

    .. note::

        This augmenter may also set the arrays of heatmaps and segmentation
        maps to zero and remove all coordinate-based data (e.g. it removes
        all bounding boxes on images that were filled with zeros).
        It does so if and only if *all* channels of an image are dropped.
        If ``nb_keep_channels >= 1`` then that never happens.


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    p : float or tuple of float or imgaug2.parameters.StochasticParameter, optional
        The probability of any channel to be dropped (i.e. set to zero).

            * If a ``float``, then that value will be used for all channels.
              A value of ``1.0`` would mean, that all channels will be dropped.
              A value of ``0.0`` would lead to no channels being dropped.
            * If a tuple ``(a, b)``, then a value ``p`` will be sampled from
              the interval ``[a, b)`` per batch and be used as the dropout
              probability.
            * If a list, then a value will be sampled from that list per
              batch and used as the probability.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per channel whether it should be *kept* (sampled value
              of ``>=0.5``) or shouldn't be kept (sampled value of ``<0.5``).
              If you instead want to provide the probability as a stochastic
              parameter, you can usually do ``imgaug2.parameters.Binomial(1-p)``
              to convert parameter `p` to a 0/1 representation.

    nb_keep_channels : int
        Minimum number of channels to keep unaltered in all images.
        E.g. a value of ``1`` means that at least one channel in every image
        will not be dropped, even if ``p=1.0``. Set to ``0`` to allow dropping
        all channels.

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
    >>> aug = iaa.Dropout2d(p=0.5)

    Create a dropout augmenter that drops on average half of all image
    channels. Dropped channels will be filled with zeros. At least one
    channel is kept unaltered in each image (default setting).

    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.Dropout2d(p=0.5, nb_keep_channels=0)

    Create a dropout augmenter that drops on average half of all image
    channels *and* may drop *all* channels in an image (i.e. images may
    contain nothing but zeros).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        p: ParamInput = 0.1,
        nb_keep_channels: int = 1,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.p = _handle_dropout_probability_param(p, "p")
        self.nb_keep_channels = max(nb_keep_channels, 0)

        self._drop_images = True
        self._drop_heatmaps = True
        self._drop_segmentation_maps = True
        self._drop_keypoints = True
        self._drop_bounding_boxes = True
        self._drop_polygons = True
        self._drop_line_strings = True

        self._heatmaps_cval = 0.0
        self._segmentation_maps_cval = 0

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        imagewise_drop_channel_ids, all_dropped_ids = self._draw_samples(batch, random_state)

        if batch.images is not None:
            for image, drop_ids in zip(batch.images, imagewise_drop_channel_ids, strict=True):
                image[:, :, drop_ids] = 0

        # Skip the non-image data steps below if we won't modify non-image
        # anyways. Minor performance improvement.
        if len(all_dropped_ids) == 0:
            return batch

        if batch.heatmaps is not None and self._drop_heatmaps:
            cval = self._heatmaps_cval
            for drop_idx in all_dropped_ids:
                batch.heatmaps[drop_idx].arr_0to1[...] = cval

        if batch.segmentation_maps is not None and self._drop_segmentation_maps:
            cval = self._segmentation_maps_cval
            for drop_idx in all_dropped_ids:
                batch.segmentation_maps[drop_idx].arr[...] = cval

        for attr_name in ["keypoints", "bounding_boxes", "polygons", "line_strings"]:
            do_drop = getattr(self, f"_drop_{attr_name}")
            attr_value = getattr(batch, attr_name)
            if attr_value is not None and do_drop:
                for drop_idx in all_dropped_ids:
                    # same as e.g.:
                    #     batch.bounding_boxes[drop_idx].bounding_boxes = []
                    setattr(attr_value[drop_idx], attr_name, [])

        return batch

    @legacy(version="0.4.0")
    def _draw_samples(
        self, batch: _BatchInAugmentation, random_state: iarandom.RNG
    ) -> tuple[list[Array], list[int]]:
        # maybe noteworthy here that the channel axis can have size 0,
        # e.g. (5, 5, 0)
        shapes = batch.get_rowwise_shapes()
        shapes = [shape if len(shape) >= 2 else tuple(list(shape) + [1]) for shape in shapes]
        imagewise_channels = np.array([shape[2] for shape in shapes], dtype=np.int32)

        # channelwise drop value over all images (float <0.5 = drop channel)
        p_samples = self.p.draw_samples(
            (int(np.sum(imagewise_channels)),), random_state=random_state
        )

        # We map the flat p_samples array to an imagewise one,
        # convert the mask to channel-ids to drop and remove channel ids if
        # there are more to be dropped than are allowed to be dropped (see
        # nb_keep_channels).
        # We also track all_dropped_ids, which contains the ids of examples
        # (not channel ids!) where all channels were dropped.
        imagewise_channels_to_drop = []
        all_dropped_ids = []
        channel_idx = 0
        for i, nb_channels in enumerate(imagewise_channels):
            p_samples_i = p_samples[channel_idx : channel_idx + nb_channels]

            drop_ids = np.nonzero(p_samples_i < 0.5)[0]
            nb_dropable = max(nb_channels - self.nb_keep_channels, 0)
            if len(drop_ids) > nb_dropable:
                random_state.shuffle(drop_ids)
                drop_ids = drop_ids[:nb_dropable]
            imagewise_channels_to_drop.append(drop_ids)

            all_dropped = len(drop_ids) == nb_channels
            if all_dropped:
                all_dropped_ids.append(i)

            channel_idx += nb_channels

        return imagewise_channels_to_drop, all_dropped_ids

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p, self.nb_keep_channels]


@legacy(version="0.4.0")
class TotalDropout(meta.Augmenter):
    """Drop all channels of a defined fraction of all images.

    For image data, all components of dropped images will be filled with zeros.

    .. note::

        This augmenter also sets the arrays of heatmaps and segmentation
        maps to zero and removes all coordinate-based data (e.g. it removes
        all bounding boxes on images that were filled with zeros).


    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    p : float or tuple of float or imgaug2.parameters.StochasticParameter, optional
        The probability of an image to be filled with zeros.

            * If ``float``: The value will be used for all images.
              A value of ``1.0`` would mean that all images will be set to zero.
              A value of ``0.0`` would lead to no images being set to zero.
            * If ``tuple`` ``(a, b)``: A value ``p`` will be sampled from
              the interval ``[a, b)`` per batch and be used as the dropout
              probability.
            * If a list, then a value will be sampled from that list per
              batch and used as the probability.
            * If ``StochasticParameter``: The parameter will be used to
              determine per image whether it should be *kept* (sampled value
              of ``>=0.5``) or shouldn't be kept (sampled value of ``<0.5``).
              If you instead want to provide the probability as a stochastic
              parameter, you can usually do ``imgaug2.parameters.Binomial(1-p)``
              to convert parameter `p` to a 0/1 representation.

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
    >>> aug = iaa.TotalDropout(1.0)

    Create an augmenter that sets *all* components of all images to zero.

    >>> aug = iaa.TotalDropout(0.5)

    Create an augmenter that sets *all* components of ``50%`` of all images to
    zero.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        p: ParamInput = 1,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.p = _handle_dropout_probability_param(p, "p")

        self._drop_images = True
        self._drop_heatmaps = True
        self._drop_segmentation_maps = True
        self._drop_keypoints = True
        self._drop_bounding_boxes = True
        self._drop_polygons = True
        self._drop_line_strings = True

        self._heatmaps_cval = 0.0
        self._segmentation_maps_cval = 0

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        drop_mask = self._draw_samples(batch, random_state)
        drop_ids = None

        if batch.images is not None and self._drop_images:
            if ia.is_np_array(batch.images):
                batch.images[drop_mask, ...] = 0
            else:
                drop_ids = self._generate_drop_ids_once(drop_mask, drop_ids)
                for drop_idx in drop_ids:
                    batch.images[drop_idx][...] = 0

        if batch.heatmaps is not None and self._drop_heatmaps:
            drop_ids = self._generate_drop_ids_once(drop_mask, drop_ids)
            cval = self._heatmaps_cval
            for drop_idx in drop_ids:
                batch.heatmaps[drop_idx].arr_0to1[...] = cval

        if batch.segmentation_maps is not None and self._drop_segmentation_maps:
            drop_ids = self._generate_drop_ids_once(drop_mask, drop_ids)
            cval = self._segmentation_maps_cval
            for drop_idx in drop_ids:
                batch.segmentation_maps[drop_idx].arr[...] = cval

        for attr_name in ["keypoints", "bounding_boxes", "polygons", "line_strings"]:
            do_drop = getattr(self, f"_drop_{attr_name}")
            attr_value = getattr(batch, attr_name)
            if attr_value is not None and do_drop:
                drop_ids = self._generate_drop_ids_once(drop_mask, drop_ids)
                for drop_idx in drop_ids:
                    # same as e.g.:
                    #     batch.bounding_boxes[drop_idx].bounding_boxes = []
                    setattr(attr_value[drop_idx], attr_name, [])

        return batch

    @legacy(version="0.4.0")
    def _draw_samples(self, batch: _BatchInAugmentation, random_state: iarandom.RNG) -> Array:
        p = self.p.draw_samples((batch.nb_rows,), random_state=random_state)
        drop_mask = p < 0.5
        return drop_mask

    @classmethod
    @legacy(version="0.4.0")
    def _generate_drop_ids_once(cls, drop_mask: Array, drop_ids: Array | None) -> Array:
        if drop_ids is None:
            drop_ids = np.nonzero(drop_mask)[0]
        return drop_ids

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p]
