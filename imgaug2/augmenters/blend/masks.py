from __future__ import annotations

from typing import Literal, cast

import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables import utils as augm_utils
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.compat.markers import legacy
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput

from .base import (
    ChildrenInput,
    CoordinateAugmentable,
    PerChannelInput,
    _generate_branch_outputs,
    _to_deterministic,
    blend_alpha_,
)
from .mask_generators import IBatchwiseMaskGenerator, SomeColorsMaskGen, StochasticParameterMaskGen

# tested indirectly via BlendAlphaElementwise for historic reasons
@legacy(version="0.4.0")
class BlendAlphaMask(meta.Augmenter):
    """
    Alpha-blend two image sources using non-binary masks generated per image.

    This augmenter queries for each image a mask generator to generate
    a ``(H,W)`` or ``(H,W,C)`` channelwise mask ``[0.0, 1.0]``, where
    ``H`` is the image height and ``W`` the width.
    The mask will then be used to alpha-blend pixel- and possibly channel-wise
    between a foreground branch of augmenters and a background branch.
    (Both branches default to the identity operation if not provided.)

    See also `BlendAlpha`.

    .. note::

        It is not recommended to use ``BlendAlphaMask`` with augmenters
        that change the geometry of images (e.g. horizontal flips, affine
        transformations) if you *also* want to augment coordinates (e.g.
        keypoints, polygons, ...), as it is unclear which of the two
        coordinate results (foreground or background branch) should be used
        as the final output coordinates after augmentation.

        Currently, for keypoints the results of the
        foreground and background branch will be mixed. That means that for
        each coordinate the augmented result will be picked from the
        foreground or background branch based on the average alpha mask value
        at the corresponding spatial location.

        For bounding boxes, line strings and polygons, either all objects
        (on an image) of the foreground or all of the background branch will
        be used, based on the average over the whole alpha mask.

    Parameters
    ----------
    mask_generator : IBatchwiseMaskGenerator
        A generator that will be queried per image to generate a mask.

    foreground : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``Augmenter``, then that augmenter will be used as the branch.

    background : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``Augmenter``, then that augmenter will be used as the branch.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See `__init__()`.

    name : None or str, optional
        See `__init__()`.

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

    >>> aug = iaa.BlendAlphaMask(
    >>>     iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()),
    >>>     iaa.Sequential([
    >>>         iaa.Clouds(),
    >>>         iaa.WithChannels([1, 2], iaa.Multiply(0.5))
    >>>     ])
    >>> )

    Create an augmenter that sometimes adds clouds at the bottom and sometimes
    at the top of the image.

    """

    # Currently the mode is only used for keypoint augmentation.
    # either or: use all keypoints from fg or all from bg branch (based
    #   on average of the whole mask).
    # pointwise: decide for each point whether to use the fg or bg
    #   branch's keypoint (based on the average mask value at the point's
    #   xy-location).
    _MODE_EITHER_OR = "either-or"
    _MODE_POINTWISE = "pointwise"
    _MODES = [_MODE_POINTWISE, _MODE_EITHER_OR]

    @legacy(version="0.4.0")
    def __init__(
        self,
        mask_generator: IBatchwiseMaskGenerator,
        foreground: ChildrenInput = None,
        background: ChildrenInput = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.mask_generator = mask_generator

        assert foreground is not None or background is not None, (
            "Expected 'foreground' and/or 'background' to not be None (i.e. "
            "at least one Augmenter), but got two None values."
        )
        self.foreground = meta.handle_children_list(
            foreground, self.name, "foreground", default=None
        )
        self.background = meta.handle_children_list(
            background, self.name, "background", default=None
        )

        # this controls how keypoints and polygons are augmented
        # Non-keypoints currently uses an either-or approach.
        # Using pointwise augmentation is problematic for polygons and line
        # strings, because the order of the points may have changed (e.g.
        # from clockwise to counter-clockwise). For polygons, it is also
        # overall more likely that some child-augmenter added/deleted points
        # and we would need a polygon recoverer.
        # Overall it seems to be the better approach to use all polygons
        # from one branch or the other, which guarantuees their validity.
        # TODO decide the either-or not based on the whole average mask
        #      value but on the average mask value within the polygon's area?
        self._coord_modes = {
            "keypoints": self._MODE_POINTWISE,
            "polygons": self._MODE_EITHER_OR,
            "line_strings": self._MODE_EITHER_OR,
            "bounding_boxes": self._MODE_EITHER_OR,
        }

        self.epsilon = 1e-2

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        batch_fg, batch_bg = _generate_branch_outputs(self, batch, hooks, parents)

        masks = self.mask_generator.draw_masks(batch, random_state)

        for i, mask in enumerate(masks):
            if batch.images is not None:
                batch.images[i] = blend_alpha_(
                    batch_fg.images[i], batch_bg.images[i], mask, eps=self.epsilon
                )

            if batch.heatmaps is not None:
                arr = batch.heatmaps[i].arr_0to1
                arr_height, arr_width = arr.shape[0:2]
                mask_binarized = self._binarize_mask(mask, arr_height, arr_width)
                batch.heatmaps[i].arr_0to1 = blend_alpha_(
                    batch_fg.heatmaps[i].arr_0to1,
                    batch_bg.heatmaps[i].arr_0to1,
                    mask_binarized,
                    eps=self.epsilon,
                )

            if batch.segmentation_maps is not None:
                arr = batch.segmentation_maps[i].arr
                arr_height, arr_width = arr.shape[0:2]
                mask_binarized = self._binarize_mask(mask, arr_height, arr_width)
                batch.segmentation_maps[i].arr = blend_alpha_(
                    batch_fg.segmentation_maps[i].arr,
                    batch_bg.segmentation_maps[i].arr,
                    mask_binarized,
                    eps=self.epsilon,
                )

            for augm_attr_name in ["keypoints", "bounding_boxes", "polygons", "line_strings"]:
                augm_value = getattr(batch, augm_attr_name)
                if augm_value is not None:
                    augm_value[i] = self._blend_coordinates(
                        augm_value[i],
                        getattr(batch_fg, augm_attr_name)[i],
                        getattr(batch_bg, augm_attr_name)[i],
                        mask,
                        self._coord_modes[augm_attr_name],
                    )

        return batch

    @legacy(version="0.4.0")
    @classmethod
    def _binarize_mask(cls, mask: Array, arr_height: int, arr_width: int) -> Array:
        # Average over channels, resize to heatmap/segmap array size
        # (+clip for cubic interpolation). We can use none-NN interpolation
        # for segmaps here as this is just the mask and not the segmap
        # array.
        mask_3d = np.atleast_3d(mask)

        # masks with zero-sized axes crash in np.average() and cannot be
        # upscaled in imresize_single_image()
        if mask.size == 0:
            mask_rs = np.zeros((arr_height, arr_width), dtype=np.float32)
        else:
            mask_avg = np.average(mask_3d, axis=2) if mask_3d.shape[2] > 0 else 1.0
            mask_rs = ia.imresize_single_image(mask_avg, (arr_height, arr_width))
        mask_arr = iadt.clip_(mask_rs, 0, 1.0)
        mask_arr_binarized = mask_arr >= 0.5
        return mask_arr_binarized

    @legacy(version="0.4.0")
    @classmethod
    def _blend_coordinates(
        cls,
        cbaoi: CoordinateAugmentable,
        cbaoi_fg: CoordinateAugmentable,
        cbaoi_bg: CoordinateAugmentable,
        mask_image: Array,
        mode: str,
    ) -> CoordinateAugmentable:
        coords = augm_utils.convert_cbaois_to_kpsois(cbaoi)
        coords_fg = augm_utils.convert_cbaois_to_kpsois(cbaoi_fg)
        coords_bg = augm_utils.convert_cbaois_to_kpsois(cbaoi_bg)

        coords = coords.to_xy_array()
        coords_fg = coords_fg.to_xy_array()
        coords_bg = coords_bg.to_xy_array()

        assert coords.shape == coords_fg.shape == coords_bg.shape, (
            "Expected number of coordinates to not be changed by foreground "
            "or background branch in BlendAlphaMask. But input coordinates "
            f"of shape {coords.shape} were changed to {coords_fg.shape} (foreground) and {coords_bg.shape} "
            "(background). Make sure to not use any augmenters that affect "
            "the existence of coordinates."
        )

        h_img, w_img = mask_image.shape[0:2]

        if mode == cls._MODE_POINTWISE:
            # Augment pointwise, i.e. check for each point and its
            # xy-location the average mask value and pick based on that
            # either the point from the foreground or background branch.
            assert len(coords_fg) == len(coords_bg), (
                "Got different numbers of coordinates before/after "
                "augmentation in BlendAlphaMask. The number of "
                "coordinates is currently not allowed to change for this "
                f"augmenter. Input contained {len(coords)} coordinates, foreground "
                f"branch {len(coords_fg)}, backround branch {len(coords_bg)}."
            )

            coords_aug = []
            subgen = zip(coords, coords_fg, coords_bg, strict=True)
            for coord, coord_fg, coord_bg in subgen:
                x_int = int(np.round(coord[0]))
                y_int = int(np.round(coord[1]))
                if 0 <= y_int < h_img and 0 <= x_int < w_img:
                    alphas_i = mask_image[y_int, x_int, ...]
                    alpha = np.average(alphas_i) if alphas_i.size > 0 else 1.0
                    if alpha > 0.5:
                        coords_aug.append(coord_fg)
                    else:
                        coords_aug.append(coord_bg)
                else:
                    coords_aug.append((x_int, y_int))
        else:
            # Augment with an either-or approach over all points, i.e.
            # based on the average of the whole mask, either all points
            # from the foreground or all points from the background branch
            # are used.
            # Note that we ensured above that _keypoint_mode must be
            # _MODE_EITHER_OR if it wasn't _MODE_POINTWISE.
            mask_image_avg = np.average(mask_image) if mask_image.size > 0 else 1.0
            if mask_image_avg > 0.5:
                coords_aug = coords_fg
            else:
                coords_aug = coords_bg

        kpsoi_aug = ia.KeypointsOnImage.from_xy_array(coords_aug, shape=cbaoi.shape)
        return augm_utils.invert_convert_cbaois_to_kpsois_(cbaoi, kpsoi_aug)

    @legacy(version="0.4.0")
    def _to_deterministic(self) -> meta.Augmenter:
        return cast(meta.Augmenter, _to_deterministic(self))

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See `get_parameters()`."""
        return [self.mask_generator]

    @legacy(version="0.4.0")
    def get_children_lists(self) -> list[list[meta.Augmenter]]:
        """See `get_children_lists()`."""
        return cast(
            list[list[meta.Augmenter]],
            [lst for lst in [self.foreground, self.background] if lst is not None],
        )

    @legacy(version="0.4.0")
    def __str__(self) -> str:
        pattern = "%s(mask_generator=%s, name=%s, foreground=%s, background=%s, deterministic=%s)"
        return pattern % (
            self.__class__.__name__,
            self.mask_generator,
            self.name,
            self.foreground,
            self.background,
            self.deterministic,
        )

@legacy(version="0.4.0")
class BlendAlphaElementwise(BlendAlphaMask):
    """
    Alpha-blend two image sources using alpha/opacity values sampled per pixel.

    This is the same as `BlendAlpha`, except that the opacity factor is
    sampled once per *pixel* instead of once per *image* (or a few times per
    image, if ``BlendAlpha.per_channel`` is set to ``True``).

    See `BlendAlpha` for more details.

    This class is a wrapper around
    `BlendAlphaMask`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        `BlendAlphaMask` for details.

    Before that named `AlphaElementwise`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Opacity of the results of the foreground branch. Values close to
        ``0.0`` mean that the results from the background branch (see
        parameter `background`) make up most of the final image.

    foreground : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``Augmenter``, then that augmenter will be used as the branch.

    background : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``Augmenter``, then that augmenter will be used as the branch.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (``False``)
        or to sample a new value for each channel (``True``).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See `__init__()`.

    name : None or str, optional
        See `__init__()`.

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
    >>> aug = iaa.BlendAlphaElementwise(0.5, iaa.Grayscale(1.0))

    Convert each image to pure grayscale and alpha-blend the result with the
    original image using an alpha of ``50%`` for all pixels, thereby removing
    about ``50%`` of all color. This is equivalent to ``iaa.Grayscale(0.5)``.
    This is also equivalent to ``iaa.BlendAlpha(0.5, iaa.Grayscale(1.0))``, as
    the opacity has a fixed value of ``0.5`` and is hence identical for all
    pixels.

    >>> aug = iaa.BlendAlphaElementwise((0, 1.0), iaa.AddToHue(100))

    Same as in the previous example, but here with hue-shift instead
    of grayscaling and additionally the alpha factor is sampled uniformly
    from the interval ``[0.0, 1.0]`` once per pixel, thereby shifting the
    hue by a random fraction for each pixel.

    >>> aug = iaa.BlendAlphaElementwise(
    >>>     (0.0, 1.0),
    >>>     iaa.Affine(rotate=(-20, 20)),
    >>>     per_channel=0.5)

    First, rotate each image by a random degree sampled uniformly from the
    interval ``[-20, 20]``. Then, alpha-blend that new image with the original
    one using a random factor sampled uniformly from the interval
    ``[0.0, 1.0]`` per pixel. For ``50%`` of all images, the blending happens
    channel-wise and the factor is sampled independently per pixel *and*
    channel (``per_channel=0.5``). Note that the visual effect depends on the
    foreground transformation and the input image; for geometric transforms it
    can be subtle and may show up mostly as color fringes.

    >>> aug = iaa.BlendAlphaElementwise(
    >>>     (0.0, 1.0),
    >>>     foreground=iaa.Add(100),
    >>>     background=iaa.Multiply(0.2))

    Apply two branches of augmenters -- ``A`` and ``B`` -- *independently*
    to input images and alpha-blend the results of these branches using a
    factor ``f``. Branch ``A`` increases image pixel intensities by ``100``
    and ``B`` multiplies the pixel intensities by ``0.2``. ``f`` is sampled
    uniformly from the interval ``[0.0, 1.0]`` per pixel. The resulting images
    contain a bit of ``A`` and a bit of ``B``.

    >>> aug = iaa.BlendAlphaElementwise([0.25, 0.75], iaa.MedianBlur(13))

    Apply median blur to each image and alpha-blend the result with the
    original image using an alpha factor of either exactly ``0.25`` or
    exactly ``0.75`` (sampled once per pixel).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        factor: ParamInput = (0.0, 1.0),
        foreground: ChildrenInput = None,
        background: ChildrenInput = None,
        per_channel: PerChannelInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        factor = iap.handle_continuous_param(
            factor, "factor", value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True
        )
        mask_gen = StochasticParameterMaskGen(factor, per_channel)
        super().__init__(
            mask_gen,
            foreground,
            background,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

    @legacy(version="0.4.0")
    @property
    def factor(self) -> iap.StochasticParameter:
        return self.mask_generator.parameter

@legacy(version="0.4.0")
class BlendAlphaSomeColors(BlendAlphaMask):
    """Blend images from two branches using colorwise masks.

    This class generates masks that "mark" a few colors and replace the
    pixels within these colors with the results of the foreground branch.
    The remaining pixels are replaced with the results of the background
    branch (usually the identity function). That allows to e.g. selectively
    grayscale a few colors, while keeping other colors unchanged.

    This class is a thin wrapper around
    `BlendAlphaMask` together with
    `SomeColorsMaskGen`.

    .. note::

        The underlying mask generator will produce an ``AssertionError`` for
        batches that contain no images.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        `BlendAlphaMask` for details.

    Parameters
    ----------
    foreground : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``Augmenter``, then that augmenter will be used as the branch.

    background : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``Augmenter``, then that augmenter will be used as the branch.

    nb_bins : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        See `SomeColorsMaskGen`.

    smoothness : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See `SomeColorsMaskGen`.

    alpha : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See `SomeColorsMaskGen`.

    rotation_deg : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See `SomeColorsMaskGen`.

    from_colorspace : str, optional
        See `SomeColorsMaskGen`.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See `__init__()`.

    name : None or str, optional
        See `__init__()`.

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
    >>> aug = iaa.BlendAlphaSomeColors(iaa.Grayscale(1.0))

    Create an augmenter that turns randomly removes some colors in images by
    grayscaling them.

    >>> aug = iaa.BlendAlphaSomeColors(iaa.TotalDropout(1.0))

    Create an augmenter that removes some colors in images by replacing them
    with black pixels.

    >>> aug = iaa.BlendAlphaSomeColors(
    >>>     iaa.MultiplySaturation(0.5), iaa.MultiplySaturation(1.5))

    Create an augmenter that desaturates some colors and increases the
    saturation of the remaining ones.

    >>> aug = iaa.BlendAlphaSomeColors(
    >>>     iaa.AveragePooling(7), alpha=[0.0, 1.0], smoothness=0.0)

    Create an augmenter that applies average pooling to some colors.
    Each color tune is either selected (alpha of ``1.0``) or not
    selected (``0.0``). There is no gradual change between similar colors.

    >>> aug = iaa.BlendAlphaSomeColors(
    >>>     iaa.AveragePooling(7), nb_bins=2, smoothness=0.0)

    Create an augmenter that applies average pooling to some colors.
    Choose on average half of all colors in images for the blending operation.

    >>> aug = iaa.BlendAlphaSomeColors(
    >>>     iaa.AveragePooling(7), from_colorspace="BGR")

    Create an augmenter that applies average pooling to some colors with
    input images being in BGR colorspace.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        foreground: ChildrenInput = None,
        background: ChildrenInput = None,
        nb_bins: ParamInput = (5, 15),
        smoothness: ParamInput = (0.1, 0.3),
        alpha: ParamInput | None = None,
        rotation_deg: ParamInput = (0, 360),
        from_colorspace: str = "RGB",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            SomeColorsMaskGen(
                nb_bins=nb_bins,
                smoothness=smoothness,
                alpha=alpha if alpha is not None else [0.0, 1.0],
                rotation_deg=rotation_deg,
                from_colorspace=from_colorspace,
            ),
            foreground=foreground,
            background=background,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )
