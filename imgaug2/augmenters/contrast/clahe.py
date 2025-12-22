"""CLAHE-based contrast augmenters."""

from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import color as color_lib
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _normalize_cv2_input_arr_
import imgaug2.mlx.color as mlx_color

from ._intensity import _IntensityChannelBasedApplier
from ._types import KernelSizeParamInput2D
from ._utils import _is_mlx_list

class AllChannelsCLAHE(meta.Augmenter):
    """Apply CLAHE to all channels of images in their original colorspaces.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) performs
    histogram equilization within image patches, i.e. over local
    neighbourhoods.

    In contrast to ``imgaug2.augmenters.contrast.CLAHE``, this augmenter
    operates directly on all channels of the input images. It does not
    perform any colorspace transformations and does not focus on specific
    channels (e.g. ``L`` in ``Lab`` colorspace).

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: no (2)
        * ``int16``: no (2)
        * ``int32``: no (2)
        * ``int64``: no (2)
        * ``float16``: no (2)
        * ``float32``: no (2)
        * ``float64``: no (2)
        * ``float128``: no (1)
        * ``bool``: no (1)

        - (1) rejected by cv2
        - (2) results in error in cv2: ``cv2.error:
              OpenCV(3.4.2) (...)/clahe.cpp:351: error: (-215:Assertion
              failed) src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
              || _src.type() == (((2) & ((1 << 3) - 1)) + (((1)-1) << 3)) in
              function 'apply'``

    Parameters
    ----------
    clip_limit : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        See ``imgaug2.augmenters.contrast.CLAHE``.

    tile_grid_size_px : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug2.parameters.StochasticParameter, optional
        See ``imgaug2.augmenters.contrast.CLAHE``.

    tile_grid_size_px_min : int, optional
        See ``imgaug2.augmenters.contrast.CLAHE``.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (``False``) or to
        sample a new value for each channel (``True``). If this value is a
        float ``p``, then for ``p`` percent of all images `per_channel` will
        be treated as ``True``, otherwise as ``False``.

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
    >>> aug = iaa.AllChannelsCLAHE()

    Create an augmenter that applies CLAHE to all channels of input images.

    >>> aug = iaa.AllChannelsCLAHE(clip_limit=(1, 10))

    Same as in the previous example, but the `clip_limit` used by CLAHE is
    uniformly sampled per image from the interval ``[1, 10]``. Some images
    will therefore have stronger contrast than others (i.e. higher clip limit
    values).

    >>> aug = iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True)

    Same as in the previous example, but the `clip_limit` is sampled per
    image *and* channel, leading to different levels of contrast for each
    channel.

    """

    def __init__(
        self,
        clip_limit: ParamInput = (0.1, 8),
        tile_grid_size_px: KernelSizeParamInput2D = (3, 12),
        tile_grid_size_px_min: int = 3,
        per_channel: ParamInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.clip_limit = iap.handle_continuous_param(
            clip_limit,
            "clip_limit",
            value_range=(0 + 1e-4, None),
            tuple_to_uniform=True,
            list_to_choice=True,
        )
        self.tile_grid_size_px = iap.handle_discrete_kernel_size_param(
            tile_grid_size_px, "tile_grid_size_px", value_range=(0, None), allow_floats=False
        )
        self.tile_grid_size_px_min = tile_grid_size_px_min
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

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

        if _is_mlx_list(images):
            nb_images = len(images)
            nb_channels = meta.estimate_max_number_of_channels(images)

            mode = "single" if self.tile_grid_size_px[1] is None else "two"
            rss = random_state.duplicate(3 if mode == "single" else 4)
            per_channel = self.per_channel.draw_samples((nb_images,), random_state=rss[0])
            clip_limit = self.clip_limit.draw_samples((nb_images, nb_channels), random_state=rss[1])
            tile_grid_size_px_h = self.tile_grid_size_px[0].draw_samples(
                (nb_images, nb_channels), random_state=rss[2]
            )
            if mode == "single":
                tile_grid_size_px_w = tile_grid_size_px_h
            else:
                tile_grid_size_px_w = self.tile_grid_size_px[1].draw_samples(
                    (nb_images, nb_channels), random_state=rss[3]
                )

            tile_grid_size_px_w = np.maximum(tile_grid_size_px_w, self.tile_grid_size_px_min)
            tile_grid_size_px_h = np.maximum(tile_grid_size_px_h, self.tile_grid_size_px_min)

            from imgaug2.mlx._core import mx

            gen = enumerate(
                zip(
                    images,
                    clip_limit,
                    tile_grid_size_px_h,
                    tile_grid_size_px_w,
                    per_channel,
                    strict=True,
                )
            )
            for i, (image, clip_limit_i, tgs_px_h_i, tgs_px_w_i, pchannel_i) in gen:
                if image.size == 0:
                    continue

                nb_channels = image.shape[2]
                c_param = 0
                image_warped: list[object] = []
                for c in range(nb_channels):
                    if tgs_px_w_i[c_param] > 1 or tgs_px_h_i[c_param] > 1:
                        channel = image[..., c : c + 1]
                        channel_warped = mlx_color.clahe(
                            channel,
                            clip_limit=float(clip_limit_i[c_param]),
                            tile_grid_size=(
                                int(tgs_px_h_i[c_param]),
                                int(tgs_px_w_i[c_param]),
                            ),
                        )
                        image_warped.append(channel_warped)
                    else:
                        image_warped.append(image[..., c : c + 1])
                    if pchannel_i > 0.5:
                        c_param += 1

                images[i] = mx.concatenate(image_warped, axis=-1)

            batch.images = images
            return batch

        iadt.gate_dtypes_strs(
            images,
            allowed="uint8 uint16",
            disallowed="bool uint32 uint64 int8 int16 int32 int64 float16 float32 float64 float128",
            augmenter=self,
        )

        nb_images = len(images)
        nb_channels = meta.estimate_max_number_of_channels(images)

        mode = "single" if self.tile_grid_size_px[1] is None else "two"
        rss = random_state.duplicate(3 if mode == "single" else 4)
        per_channel = self.per_channel.draw_samples((nb_images,), random_state=rss[0])
        clip_limit = self.clip_limit.draw_samples((nb_images, nb_channels), random_state=rss[1])
        tile_grid_size_px_h = self.tile_grid_size_px[0].draw_samples(
            (nb_images, nb_channels), random_state=rss[2]
        )
        if mode == "single":
            tile_grid_size_px_w = tile_grid_size_px_h
        else:
            tile_grid_size_px_w = self.tile_grid_size_px[1].draw_samples(
                (nb_images, nb_channels), random_state=rss[3]
            )

        tile_grid_size_px_w = np.maximum(tile_grid_size_px_w, self.tile_grid_size_px_min)
        tile_grid_size_px_h = np.maximum(tile_grid_size_px_h, self.tile_grid_size_px_min)

        gen = enumerate(
            zip(
                images,
                clip_limit,
                tile_grid_size_px_h,
                tile_grid_size_px_w,
                per_channel,
                strict=True,
            )
        )
        for i, (image, clip_limit_i, tgs_px_h_i, tgs_px_w_i, pchannel_i) in gen:
            if image.size == 0:
                continue

            nb_channels = image.shape[2]
            c_param = 0
            image_warped = []
            for c in range(nb_channels):
                if tgs_px_w_i[c_param] > 1 or tgs_px_h_i[c_param] > 1:
                    clahe = cv2.createCLAHE(
                        clipLimit=clip_limit_i[c_param],
                        tileGridSize=(tgs_px_w_i[c_param], tgs_px_h_i[c_param]),
                    )
                    channel_warped = clahe.apply(_normalize_cv2_input_arr_(image[..., c]))
                    image_warped.append(channel_warped)
                else:
                    image_warped.append(image[..., c])
                if pchannel_i > 0.5:
                    c_param += 1

            # combine channels to one image
            image_warped = np.stack(image_warped, axis=-1)

            batch.images[i] = image_warped
        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [
            self.clip_limit,
            self.tile_grid_size_px,
            self.tile_grid_size_px_min,
            self.per_channel,
        ]


class CLAHE(meta.Augmenter):
    """Apply CLAHE to L/V/L channels in HLS/HSV/Lab colorspaces.

    This augmenter applies CLAHE (Contrast Limited Adaptive Histogram
    Equalization) to images, a form of histogram equalization that normalizes
    within local image patches.
    The augmenter transforms input images to a target colorspace (e.g.
    ``Lab``), extracts an intensity-related channel from the converted
    images (e.g. ``L`` for ``Lab``), applies CLAHE to the channel and then
    converts the resulting image back to the original colorspace.

    Grayscale images (images without channel axis or with only one channel
    axis) are automatically handled, `from_colorspace` does not have to be
    adjusted for them. For images with four channels (e.g. ``RGBA``), the
    fourth channel is ignored in the colorspace conversion (e.g. from an
    ``RGBA`` image, only the ``RGB`` part is converted, normalized, converted
    back and concatenated with the input ``A`` channel). Images with unusual
    channel numbers (2, 5 or more than 5) are normalized channel-by-channel
    (same behaviour as ``AllChannelsCLAHE``, though a warning will be raised).

    If you want to apply CLAHE to each channel of the original input image's
    colorspace (without any colorspace conversion), use
    ``imgaug2.augmenters.contrast.AllChannelsCLAHE`` instead.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no (1)
        * ``uint32``: no (1)
        * ``uint64``: no (1)
        * ``int8``: no (1)
        * ``int16``: no (1)
        * ``int32``: no (1)
        * ``int64``: no (1)
        * ``float16``: no (1)
        * ``float32``: no (1)
        * ``float64``: no (1)
        * ``float128``: no (1)
        * ``bool``: no (1)

        - (1) This augmenter uses
              :class:`~imgaug2.augmenters.color.ChangeColorspace`, which is
              currently limited to ``uint8``.

    Parameters
    ----------
    clip_limit : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Clipping limit. Higher values result in stronger contrast. OpenCV
        uses a default of ``40``, though values around ``5`` seem to already
        produce decent contrast.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the range ``[a, b]``
              will be used per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    tile_grid_size_px : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug2.parameters.StochasticParameter, optional
        Kernel size, i.e. size of each local neighbourhood in pixels.

            * If an ``int``, then that value will be used for all images for
              both kernel height and width.
            * If a tuple ``(a, b)``, then a value from the discrete interval
              ``[a..b]`` will be uniformly sampled per image.
            * If a list, then a random value will be sampled from that list
              per image and used for both kernel height and width.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter per image and used for both kernel
              height and width.
            * If a tuple of tuple of ``int`` given as ``((a, b), (c, d))``,
              then two values will be sampled independently from the discrete
              ranges ``[a..b]`` and ``[c..d]`` per image and used as the
              kernel height and width.
            * If a tuple of lists of ``int``, then two values will be sampled
              independently per image, one from the first list and one from
              the second, and used as the kernel height and width.
            * If a tuple of ``StochasticParameter``, then two values will be
              sampled indepdently per image, one from the first parameter and
              one from the second, and used as the kernel height and width.

    tile_grid_size_px_min : int, optional
        Minimum kernel size in px, per axis. If the sampling results in a
        value lower than this minimum, it will be clipped to this value.

    from_colorspace : {"RGB", "BGR", "HSV", "HLS", "Lab"}, optional
        Colorspace of the input images.
        If any input image has only one or zero channels, this setting will
        be ignored and it will be assumed that the input is grayscale.
        If a fourth channel is present in an input image, it will be removed
        before the colorspace conversion and later re-added.
        See also :func:`~imgaug2.augmenters.color.change_colorspace_` for
        details.

    to_colorspace : {"Lab", "HLS", "HSV"}, optional
        Colorspace in which to perform CLAHE. For ``Lab``, CLAHE will only be
        applied to the first channel (``L``), for ``HLS`` to the
        second (``L``) and for ``HSV`` to the third (``V``). To apply CLAHE
        to all channels of an input image (without colorspace conversion),
        see ``imgaug2.augmenters.contrast.AllChannelsCLAHE``.

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
    >>> aug = iaa.CLAHE()

    Create a standard CLAHE augmenter.

    >>> aug = iaa.CLAHE(clip_limit=(1, 10))

    Create a CLAHE augmenter with a clip limit uniformly sampled from
    ``[1..10]``, where ``1`` is rather low contrast and ``10`` is rather
    high contrast.

    >>> aug = iaa.CLAHE(tile_grid_size_px=(3, 21))

    Create a CLAHE augmenter with kernel sizes of ``SxS``, where ``S`` is
    uniformly sampled from ``[3..21]``. Sampling happens once per image.

    >>> aug = iaa.CLAHE(
    >>>     tile_grid_size_px=iap.Discretize(iap.Normal(loc=7, scale=2)),
    >>>     tile_grid_size_px_min=3)

    Create a CLAHE augmenter with kernel sizes of ``SxS``, where ``S`` is
    sampled from ``N(7, 2)``, but does not go below ``3``.

    >>> aug = iaa.CLAHE(tile_grid_size_px=((3, 21), [3, 5, 7]))

    Create a CLAHE augmenter with kernel sizes of ``HxW``, where ``H`` is
    uniformly sampled from ``[3..21]`` and ``W`` is randomly picked from the
    list ``[3, 5, 7]``.

    >>> aug = iaa.CLAHE(
    >>>     from_colorspace=iaa.CSPACE_BGR,
    >>>     to_colorspace=iaa.CSPACE_HSV)

    Create a CLAHE augmenter that converts images from BGR colorspace to
    HSV colorspace and then applies the local histogram equalization to the
    ``V`` channel of the images (before converting back to ``BGR``).
    Alternatively, ``Lab`` (default) or ``HLS`` can be used as the target
    colorspace. Grayscale images (no channels / one channel) are never
    converted and are instead directly normalized (i.e. `from_colorspace`
    does not have to be changed for them).

    """

    RGB = color_lib.CSPACE_RGB
    BGR = color_lib.CSPACE_BGR
    HSV = color_lib.CSPACE_HSV
    HLS = color_lib.CSPACE_HLS
    Lab = color_lib.CSPACE_Lab

    def __init__(
        self,
        clip_limit: ParamInput = (0.1, 8),
        tile_grid_size_px: KernelSizeParamInput2D = (3, 12),
        tile_grid_size_px_min: int = 3,
        from_colorspace: str = color_lib.CSPACE_RGB,
        to_colorspace: str = color_lib.CSPACE_Lab,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.all_channel_clahe = AllChannelsCLAHE(
            clip_limit=clip_limit,
            tile_grid_size_px=tile_grid_size_px,
            tile_grid_size_px_min=tile_grid_size_px_min,
            name=f"{name}_AllChannelsCLAHE",
        )

        self.intensity_channel_based_applier = _IntensityChannelBasedApplier(
            from_colorspace, to_colorspace
        )

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

        def _augment_all_channels_clahe(
            images_normalized: list[Array], random_state_derived: iarandom.RNG
        ) -> list[Array]:
            # TODO would .augment_batch() be sufficient here?
            batch_imgs = _BatchInAugmentation(images=images_normalized)
            batch_out = self.all_channel_clahe._augment_batch_(
                batch_imgs, random_state_derived, parents + [self], hooks
            )
            assert batch_out.images is not None
            return cast(list[Array], batch_out.images)

        batch.images = self.intensity_channel_based_applier.apply(
            images, random_state, parents + [self], hooks, _augment_all_channels_clahe
        )
        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        ac_clahe = self.all_channel_clahe
        intb_applier = self.intensity_channel_based_applier
        return [
            ac_clahe.clip_limit,
            ac_clahe.tile_grid_size_px,
            ac_clahe.tile_grid_size_px_min,
        ] + intb_applier.get_parameters()


