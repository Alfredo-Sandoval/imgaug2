"""Fixed-size crop/pad augmenters."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Literal

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, Images, ParamInput, RNGInput
from imgaug2.compat.markers import legacy

from ._utils import CropToFixedSizeSamplingResult, PadToFixedSizeSamplingResult, TRBL, _handle_pad_mode_param
from .crop_pad import _crop_and_pad_arr, _crop_and_pad_hms_or_segmaps_, _crop_and_pad_kpsoi_
class PadToFixedSize(meta.Augmenter):
    """Pad images to a predefined minimum width and/or height.

    If images are already at the minimum width/height or are larger, they will
    not be padded. Note that this also means that images will not be cropped if
    they exceed the required width/height.

    The augmenter randomly decides per image how to distribute the required
    padding amounts over the image axis. E.g. if 2px have to be padded on the
    left or right to reach the required width, the augmenter will sometimes
    add 2px to the left and 0px to the right, sometimes add 2px to the right
    and 0px to the left and sometimes add 1px to both sides. Set `position`
    to ``center`` to prevent that.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.size.pad`.

    Parameters
    ----------
    width : int or None
        Pad images up to this minimum width.
        If ``None``, image widths will not be altered.

    height : int or None
        Pad images up to this minimum height.
        If ``None``, image heights will not be altered.

    pad_mode : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        See :func:`~imgaug2.augmenters.size.CropAndPad.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :func:`~imgaug2.augmenters.size.CropAndPad.__init__`.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        Sets the center point of the padding, which determines how the
        required padding amounts are distributed to each side. For a ``tuple``
        ``(a, b)``, both ``a`` and ``b`` are expected to be in range
        ``[0.0, 1.0]`` and describe the fraction of padding applied to the
        left/right (low/high values for ``a``) and the fraction of padding
        applied to the top/bottom (low/high values for ``b``). A padding
        position at ``(0.5, 0.5)`` would be the center of the image and
        distribute the padding equally to all sides. A padding position at
        ``(0.0, 1.0)`` would be the left-bottom and would apply 100% of the
        required padding to the bottom and left sides of the image so that
        the bottom left corner becomes more and more the new image
        center (depending on how much is padded).

            * If string ``uniform`` then the share of padding is randomly and
              uniformly distributed over each side.
              Equivalent to ``(Uniform(0.0, 1.0), Uniform(0.0, 1.0))``.
            * If string ``normal`` then the share of padding is distributed
              based on a normal distribution, leading to a focus on the
              center of the images.
              Equivalent to
              ``(Clip(Normal(0.5, 0.45/2), 0, 1),
              Clip(Normal(0.5, 0.45/2), 0, 1))``.
            * If string ``center`` then center point of the padding is
              identical to the image center.
              Equivalent to ``(0.5, 0.5)``.
            * If a string matching regex
              ``^(left|center|right)-(top|center|bottom)$``, e.g. ``left-top``
              or ``center-bottom`` then sets the center point of the padding
              to the X-Y position matching that description.
            * If a tuple of float, then expected to have exactly two entries
              between ``0.0`` and ``1.0``, which will always be used as the
              combination the position matching (x, y) form.
            * If a ``StochasticParameter``, then that parameter will be queried
              once per call to ``augment_*()`` to get ``Nx2`` center positions
              in ``(x, y)`` form (with ``N`` the number of images).
            * If a ``tuple`` of ``StochasticParameter``, then expected to have
              exactly two entries that will both be queried per call to
              ``augment_*()``, each for ``(N,)`` values, to get the center
              positions. First parameter is used for ``x`` coordinates,
              second for ``y`` coordinates.

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
    >>> aug = iaa.PadToFixedSize(width=100, height=100)

    For image sides smaller than ``100`` pixels, pad to ``100`` pixels. Do
    nothing for the other edges. The padding is randomly (uniformly)
    distributed over the sides, so that e.g. sometimes most of the required
    padding is applied to the left, sometimes to the right (analogous
    top/bottom).

    >>> aug = iaa.PadToFixedSize(width=100, height=100, position="center")

    For image sides smaller than ``100`` pixels, pad to ``100`` pixels. Do
    nothing for the other image sides. The padding is always equally
    distributed over the left/right and top/bottom sides.

    >>> aug = iaa.PadToFixedSize(width=100, height=100, pad_mode=ia.ALL)

    For image sides smaller than ``100`` pixels, pad to ``100`` pixels and
    use any possible padding mode for that. Do nothing for the other image
    sides. The padding is always equally distributed over the left/right and
    top/bottom sides.

    >>> aug = iaa.Sequential([
    >>>     iaa.PadToFixedSize(width=100, height=100),
    >>>     iaa.CropToFixedSize(width=100, height=100)
    >>> ])

    Pad images smaller than ``100x100`` until they reach ``100x100``.
    Analogously, crop images larger than ``100x100`` until they reach
    ``100x100``. The output images therefore have a fixed size of ``100x100``.

    """

    def __init__(
        self,
        width: int | None,
        height: int | None,
        pad_mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        pad_cval: ParamInput = 0,
        position: str
        | tuple[float | int | iap.StochasticParameter, float | int | iap.StochasticParameter]
        | iap.StochasticParameter = "uniform",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.size = (width, height)

        # Position of where to pad. The further to the top left this is, the
        # larger the share of pixels that will be added to the top and left
        # sides. I.e. set to (Deterministic(0.0), Deterministic(0.0)) to only
        # add at the top and left, (Deterministic(1.0), Deterministic(1.0))
        # to only add at the bottom right. Analogously (0.5, 0.5) pads equally
        # on both axis, (0.0, 1.0) pads left and bottom, (1.0, 0.0) pads right
        # and top.
        self.position = iap.handle_position_parameter(position)

        self.pad_mode = _handle_pad_mode_param(pad_mode)
        # TODO enable ALL here like in eg Affine
        self.pad_cval = iap.handle_discrete_param(
            pad_cval,
            "pad_cval",
            value_range=None,
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=True,
        )

        # set these to None to use the same values as sampled for the
        # images (not tested)
        self._pad_mode_heatmaps = "constant"
        self._pad_mode_segmentation_maps = "constant"
        self._pad_cval_heatmaps = 0.0
        self._pad_cval_segmentation_maps = 0

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        # Providing the whole batch to _draw_samples() would not be necessary
        # for this augmenter. The number of rows would be sufficient. This
        # formulation however enables derived augmenters to use rowwise shapes
        # without having to compute them here for this augmenter.
        samples = self._draw_samples(batch, random_state)

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images, samples)

        if batch.heatmaps is not None:
            batch.heatmaps = self._augment_maps_by_samples(
                batch.heatmaps, samples, self._pad_mode_heatmaps, self._pad_cval_heatmaps
            )

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps, samples, self._pad_mode_heatmaps, self._pad_cval_heatmaps
            )

        for augm_name in ["keypoints", "bounding_boxes", "polygons", "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(self._augment_keypoints_by_samples, samples=samples)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    @legacy(version="0.4.0")
    def _augment_images_by_samples(
        self, images: Images, samples: PadToFixedSizeSamplingResult
    ) -> list[Array]:
        result = []
        sizes, pad_xs, pad_ys, pad_modes, pad_cvals = samples
        for i, (image, size) in enumerate(zip(images, sizes, strict=True)):
            width_min, height_min = size
            height_image, width_image = image.shape[:2]
            paddings = self._calculate_paddings(
                height_image, width_image, height_min, width_min, pad_xs[i], pad_ys[i]
            )

            image = _crop_and_pad_arr(
                image, (0, 0, 0, 0), paddings, pad_modes[i], pad_cvals[i], keep_size=False
            )

            result.append(image)

        # TODO result is always a list. Should this be converted to an array
        #      if possible (not guaranteed that all images have same size,
        #      some might have been larger than desired height/width)
        return result

    @legacy(version="0.4.0")
    def _augment_keypoints_by_samples(
        self, keypoints_on_images: list[ia.KeypointsOnImage], samples: PadToFixedSizeSamplingResult
    ) -> list[ia.KeypointsOnImage]:
        result = []
        sizes, pad_xs, pad_ys, _, _ = samples
        for i, (kpsoi, size) in enumerate(zip(keypoints_on_images, sizes, strict=True)):
            width_min, height_min = size
            height_image, width_image = kpsoi.shape[:2]
            paddings_img = self._calculate_paddings(
                height_image, width_image, height_min, width_min, pad_xs[i], pad_ys[i]
            )

            keypoints_padded = _crop_and_pad_kpsoi_(
                kpsoi, (0, 0, 0, 0), paddings_img, keep_size=False
            )

            result.append(keypoints_padded)

        return result

    @legacy(version="0.4.0")
    def _augment_maps_by_samples(
        self,
        augmentables: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage],
        samples: PadToFixedSizeSamplingResult,
        pad_mode: str | None,
        pad_cval: float | int | np.generic | None,
    ) -> list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage]:
        sizes, pad_xs, pad_ys, pad_modes, pad_cvals = samples

        for i, (augmentable, size) in enumerate(zip(augmentables, sizes, strict=True)):
            width_min, height_min = size
            height_img, width_img = augmentable.shape[:2]
            paddings_img = self._calculate_paddings(
                height_img, width_img, height_min, width_min, pad_xs[i], pad_ys[i]
            )

            # TODO for the previous method (and likely the new/current one
            #      too):
            #      for 30x30 padded to 32x32 with 15x15 heatmaps this results
            #      in paddings of 1 on each side (assuming
            #      position=(0.5, 0.5)) giving 17x17 heatmaps when they should
            #      be 16x16. Error is due to each side getting projected 0.5
            #      padding which is rounded to 1. This doesn't seem right.
            augmentables[i] = _crop_and_pad_hms_or_segmaps_(
                augmentables[i],
                (0, 0, 0, 0),
                paddings_img,
                pad_mode=pad_mode if pad_mode is not None else pad_modes[i],
                pad_cval=pad_cval if pad_cval is not None else pad_cvals[i],
                keep_size=False,
            )

        return augmentables

    def _draw_samples(
        self, batch: _BatchInAugmentation, random_state: iarandom.RNG
    ) -> PadToFixedSizeSamplingResult:
        nb_images = batch.nb_rows
        rngs = random_state.duplicate(4)

        if isinstance(self.position, tuple):
            pad_xs = self.position[0].draw_samples(nb_images, random_state=rngs[0])
            pad_ys = self.position[1].draw_samples(nb_images, random_state=rngs[1])
        else:
            pads = self.position.draw_samples((nb_images, 2), random_state=rngs[0])
            pad_xs = pads[:, 0]
            pad_ys = pads[:, 1]

        pad_modes = self.pad_mode.draw_samples(nb_images, random_state=rngs[2])
        pad_cvals = self.pad_cval.draw_samples(nb_images, random_state=rngs[3])

        # We return here the sizes even though they are static as it allows
        # derived augmenters to define image-specific heights/widths.
        return [self.size] * nb_images, pad_xs, pad_ys, pad_modes, pad_cvals

    @classmethod
    def _calculate_paddings(
        cls,
        height_image: int,
        width_image: int,
        height_min: int | None,
        width_min: int | None,
        pad_xs_i: float | np.floating,
        pad_ys_i: float | np.floating,
    ) -> TRBL:
        pad_top = 0
        pad_right = 0
        pad_bottom = 0
        pad_left = 0

        if width_min is not None and width_image < width_min:
            pad_total_x = width_min - width_image
            pad_left = int((1 - pad_xs_i) * pad_total_x)
            pad_right = pad_total_x - pad_left

        if height_min is not None and height_image < height_min:
            pad_total_y = height_min - height_image
            pad_top = int((1 - pad_ys_i) * pad_total_y)
            pad_bottom = pad_total_y - pad_top

        return pad_top, pad_right, pad_bottom, pad_left

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.size[0], self.size[1], self.pad_mode, self.pad_cval, self.position]


@legacy(version="0.4.0")
class CenterPadToFixedSize(PadToFixedSize):
    """Pad images equally on all sides up to given minimum heights/widths.

    This is an alias for :class:`~imgaug2.augmenters.size.PadToFixedSize`
    with ``position="center"``. It spreads the pad amounts equally over
    all image sides, while :class:`~imgaug2.augmenters.size.PadToFixedSize`
    by defaults spreads them randomly.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width : int or None
        See :func:`PadToFixedSize.__init__`.

    height : int or None
        See :func:`PadToFixedSize.__init__`.

    pad_mode : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

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
    >>> aug = iaa.CenterPadToFixedSize(height=20, width=30)

    Create an augmenter that pads images up to ``20x30``, with the padded
    rows added *equally* on the top and bottom (analogous for the padded
    columns).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        width: int | None,
        height: int | None,
        pad_mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        pad_cval: ParamInput = 0,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            position="center",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


# TODO maybe rename this to CropToMaximumSize ?
# TODO this is very similar to CropAndPad, maybe add a way to generate crop
#      values imagewise via a callback in in CropAndPad?
# TODO add crop() function in imgaug, similar to pad
class CropToFixedSize(meta.Augmenter):
    """Crop images down to a predefined maximum width and/or height.

    If images are already at the maximum width/height or are smaller, they
    will not be cropped. Note that this also means that images will not be
    padded if they are below the required width/height.

    The augmenter randomly decides per image how to distribute the required
    cropping amounts over the image axis. E.g. if 2px have to be cropped on
    the left or right to reach the required width, the augmenter will
    sometimes remove 2px from the left and 0px from the right, sometimes
    remove 2px from the right and 0px from the left and sometimes remove 1px
    from both sides. Set `position` to ``center`` to prevent that.

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
    width : int or None
        Crop images down to this maximum width.
        If ``None``, image widths will not be altered.

    height : int or None
        Crop images down to this maximum height.
        If ``None``, image heights will not be altered.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
         Sets the center point of the cropping, which determines how the
         required cropping amounts are distributed to each side. For a
         ``tuple`` ``(a, b)``, both ``a`` and ``b`` are expected to be in
         range ``[0.0, 1.0]`` and describe the fraction of cropping applied
         to the left/right (low/high values for ``a``) and the fraction
         of cropping applied to the top/bottom (low/high values for ``b``).
         A cropping position at ``(0.5, 0.5)`` would be the center of the
         image and distribute the cropping equally over all sides. A cropping
         position at ``(1.0, 0.0)`` would be the right-top and would apply
         100% of the required cropping to the right and top sides of the image.

            * If string ``uniform`` then the share of cropping is randomly
              and uniformly distributed over each side.
              Equivalent to ``(Uniform(0.0, 1.0), Uniform(0.0, 1.0))``.
            * If string ``normal`` then the share of cropping is distributed
              based on a normal distribution, leading to a focus on the center
              of the images.
              Equivalent to
              ``(Clip(Normal(0.5, 0.45/2), 0, 1),
              Clip(Normal(0.5, 0.45/2), 0, 1))``.
            * If string ``center`` then center point of the cropping is
              identical to the image center.
              Equivalent to ``(0.5, 0.5)``.
            * If a string matching regex
              ``^(left|center|right)-(top|center|bottom)$``, e.g.
              ``left-top`` or ``center-bottom`` then sets the center point of
              the cropping to the X-Y position matching that description.
            * If a tuple of float, then expected to have exactly two entries
              between ``0.0`` and ``1.0``, which will always be used as the
              combination the position matching (x, y) form.
            * If a ``StochasticParameter``, then that parameter will be queried
              once per call to ``augment_*()`` to get ``Nx2`` center positions
              in ``(x, y)`` form (with ``N`` the number of images).
            * If a ``tuple`` of ``StochasticParameter``, then expected to have
              exactly two entries that will both be queried per call to
              ``augment_*()``, each for ``(N,)`` values, to get the center
              positions. First parameter is used for ``x`` coordinates,
              second for ``y`` coordinates.

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
    >>> aug = iaa.CropToFixedSize(width=100, height=100)

    For image sides larger than ``100`` pixels, crop to ``100`` pixels. Do
    nothing for the other sides. The cropping amounts are randomly (and
    uniformly) distributed over the sides of the image.

    >>> aug = iaa.CropToFixedSize(width=100, height=100, position="center")

    For sides larger than ``100`` pixels, crop to ``100`` pixels. Do nothing
    for the other sides. The cropping amounts are always equally distributed
    over the left/right sides of the image (and analogously for top/bottom).

    >>> aug = iaa.Sequential([
    >>>     iaa.PadToFixedSize(width=100, height=100),
    >>>     iaa.CropToFixedSize(width=100, height=100)
    >>> ])

    Pad images smaller than ``100x100`` until they reach ``100x100``.
    Analogously, crop images larger than ``100x100`` until they reach
    ``100x100``. The output images therefore have a fixed size of ``100x100``.

    """

    def __init__(
        self,
        width: int | None,
        height: int | None,
        position: str
        | tuple[float | int | iap.StochasticParameter, float | int | iap.StochasticParameter]
        | iap.StochasticParameter = "uniform",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.size = (width, height)

        # Position of where to crop. The further to the top left this is,
        # the larger the share of pixels that will be cropped from the top
        # and left sides. I.e. set to (Deterministic(0.0), Deterministic(0.0))
        # to only crop at the top and left,
        # (Deterministic(1.0), Deterministic(1.0)) to only crop at the bottom
        # right. Analogously (0.5, 0.5) crops equally on both axis,
        # (0.0, 1.0) crops left and bottom, (1.0, 0.0) crops right and top.
        self.position = iap.handle_position_parameter(position)

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        # Providing the whole batch to _draw_samples() would not be necessary
        # for this augmenter. The number of rows would be sufficient. This
        # formulation however enables derived augmenters to use rowwise shapes
        # without having to compute them here for this augmenter.
        samples = self._draw_samples(batch, random_state)

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images, samples)

        if batch.heatmaps is not None:
            batch.heatmaps = self._augment_maps_by_samples(batch.heatmaps, samples)

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps, samples
            )

        for augm_name in ["keypoints", "bounding_boxes", "polygons", "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(self._augment_keypoints_by_samples, samples=samples)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    @legacy(version="0.4.0")
    def _augment_images_by_samples(
        self, images: Images, samples: CropToFixedSizeSamplingResult
    ) -> list[Array]:
        result = []
        sizes, offset_xs, offset_ys = samples
        for i, (image, size) in enumerate(zip(images, sizes, strict=True)):
            w, h = size
            height_image, width_image = image.shape[0:2]

            croppings = self._calculate_crop_amounts(
                height_image, width_image, h, w, offset_ys[i], offset_xs[i]
            )

            image_cropped = _crop_and_pad_arr(image, croppings, (0, 0, 0, 0), keep_size=False)

            result.append(image_cropped)

        return result

    @legacy(version="0.4.0")
    def _augment_keypoints_by_samples(
        self, kpsois: list[ia.KeypointsOnImage], samples: CropToFixedSizeSamplingResult
    ) -> list[ia.KeypointsOnImage]:
        result = []
        sizes, offset_xs, offset_ys = samples
        for i, (kpsoi, size) in enumerate(zip(kpsois, sizes, strict=True)):
            w, h = size
            height_image, width_image = kpsoi.shape[0:2]

            croppings_img = self._calculate_crop_amounts(
                height_image, width_image, h, w, offset_ys[i], offset_xs[i]
            )

            kpsoi_cropped = _crop_and_pad_kpsoi_(
                kpsoi, croppings_img, (0, 0, 0, 0), keep_size=False
            )

            result.append(kpsoi_cropped)

        return result

    @legacy(version="0.4.0")
    def _augment_maps_by_samples(
        self,
        augmentables: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage],
        samples: CropToFixedSizeSamplingResult,
    ) -> list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage]:
        sizes, offset_xs, offset_ys = samples
        for i, (augmentable, size) in enumerate(zip(augmentables, sizes, strict=True)):
            w, h = size
            height_image, width_image = augmentable.shape[0:2]

            croppings_img = self._calculate_crop_amounts(
                height_image, width_image, h, w, offset_ys[i], offset_xs[i]
            )

            augmentables[i] = _crop_and_pad_hms_or_segmaps_(
                augmentable, croppings_img, (0, 0, 0, 0), keep_size=False
            )

        return augmentables

    @classmethod
    def _calculate_crop_amounts(
        cls,
        height_image: int,
        width_image: int,
        height_max: int | None,
        width_max: int | None,
        offset_y: float | np.floating,
        offset_x: float | np.floating,
    ) -> TRBL:
        crop_top = 0
        crop_right = 0
        crop_bottom = 0
        crop_left = 0

        if height_max is not None and height_image > height_max:
            crop_top = int(offset_y * (height_image - height_max))
            crop_bottom = height_image - height_max - crop_top

        if width_max is not None and width_image > width_max:
            crop_left = int(offset_x * (width_image - width_max))
            crop_right = width_image - width_max - crop_left

        return crop_top, crop_right, crop_bottom, crop_left

    def _draw_samples(
        self, batch: _BatchInAugmentation, random_state: iarandom.RNG
    ) -> CropToFixedSizeSamplingResult:
        nb_images = batch.nb_rows
        rngs = random_state.duplicate(2)

        if isinstance(self.position, tuple):
            offset_xs = self.position[0].draw_samples(nb_images, random_state=rngs[0])
            offset_ys = self.position[1].draw_samples(nb_images, random_state=rngs[1])
        else:
            offsets = self.position.draw_samples((nb_images, 2), random_state=rngs[0])
            offset_xs = offsets[:, 0]
            offset_ys = offsets[:, 1]

        offset_xs = 1.0 - offset_xs
        offset_ys = 1.0 - offset_ys

        # We return here the sizes even though they are static as it allows
        # derived augmenters to define image-specific heights/widths.
        return [self.size] * nb_images, offset_xs, offset_ys

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.size[0], self.size[1], self.position]


@legacy(version="0.4.0")
class CenterCropToFixedSize(CropToFixedSize):
    """Take a crop from the center of each image.

    This is an alias for :class:`~imgaug2.augmenters.size.CropToFixedSize` with
    ``position="center"``.

    .. note::

        If images already have a width and/or height below the provided
        width and/or height then this augmenter will do nothing for the
        respective axis. Hence, resulting images can be smaller than the
        provided axis sizes.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width : int or None
        See :func:`CropToFixedSize.__init__`.

    height : int or None
        See :func:`CropToFixedSize.__init__`.

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
    >>> crop = iaa.CenterCropToFixedSize(height=20, width=10)

    Create an augmenter that takes ``20x10`` sized crops from the center of
    images.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        width: int | None,
        height: int | None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            position="center",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


