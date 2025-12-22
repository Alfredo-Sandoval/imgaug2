"""Intensity-channel helpers for contrast augmenters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.augmenters import color as color_lib
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array

from ._types import IntensityChannelFunc

class _IntensityChannelBasedApplier:
    RGB = color_lib.CSPACE_RGB
    BGR = color_lib.CSPACE_BGR
    HSV = color_lib.CSPACE_HSV
    HLS = color_lib.CSPACE_HLS
    Lab = color_lib.CSPACE_Lab
    _CHANNEL_MAPPING = {HSV: 2, HLS: 1, Lab: 0}

    def __init__(self, from_colorspace: str, to_colorspace: str) -> None:
        super().__init__()

        # TODO maybe add CIE, Luv?
        valid_from_colorspaces = [self.RGB, self.BGR, self.Lab, self.HLS, self.HSV]
        assert from_colorspace in valid_from_colorspaces, (
            f"Expected 'from_colorspace' to be one of {valid_from_colorspaces}, got {from_colorspace}."
        )

        valid_to_colorspaces = [self.Lab, self.HLS, self.HSV]
        assert to_colorspace in valid_to_colorspaces, (
            f"Expected 'to_colorspace' to be one of {valid_to_colorspaces}, got {to_colorspace}."
        )

        self.from_colorspace = from_colorspace
        self.to_colorspace = to_colorspace

    def apply(
        self,
        images: Array | Sequence[Array],
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
        func: IntensityChannelFunc,
    ) -> Array | list[Array]:
        input_was_array = ia.is_np_array(images)
        rss = random_state.duplicate(3)

        # normalize images
        # (H, W, 1)      will be used directly in AllChannelsCLAHE
        # (H, W, 3)      will be converted to target colorspace in the next
        #                block
        # (H, W, 4)      will be reduced to (H, W, 3) (remove 4th channel) and
        #                converted to target colorspace in next block
        # (H, W, <else>) will raise a warning and be treated channelwise by
        #                AllChannelsCLAHE
        images_normalized = []
        images_change_cs = []
        images_change_cs_indices = []
        for i, image in enumerate(images):
            nb_channels = image.shape[2]
            if nb_channels == 1:
                images_normalized.append(image)
            elif nb_channels == 3:
                images_normalized.append(None)
                images_change_cs.append(image)
                images_change_cs_indices.append(i)
            elif nb_channels == 4:
                # assume that 4th channel is an alpha channel, e.g. in RGBA
                images_normalized.append(None)
                images_change_cs.append(image[..., 0:3])
                images_change_cs_indices.append(i)
            else:
                parents_str = ", ".join(parent.name for parent in parents)
                ia.warn(
                    f"Got image with {nb_channels} channels in _IntensityChannelBasedApplier "
                    f"(parents: {parents_str}), expected 0, 1, 3 or 4 channels."
                )
                images_normalized.append(image)

        # convert colorspaces of normalized 3-channel images
        images_after_color_conversion = [None] * len(images_normalized)
        if len(images_change_cs) > 0:
            images_new_cs = color_lib.change_colorspaces_(
                images_change_cs,
                to_colorspaces=self.to_colorspace,
                from_colorspaces=self.from_colorspace,
            )

            for image_new_cs, target_idx in zip(
                images_new_cs, images_change_cs_indices, strict=True
            ):
                chan_idx = self._CHANNEL_MAPPING[self.to_colorspace]
                images_normalized[target_idx] = image_new_cs[..., chan_idx : chan_idx + 1]
                images_after_color_conversion[target_idx] = image_new_cs

        # apply function channelwise
        images_aug = func(images_normalized, rss[1])

        # denormalize
        result = []
        images_change_cs = []
        images_change_cs_indices = []
        gen = enumerate(zip(images, images_after_color_conversion, images_aug, strict=True))
        for i, (image, image_conv, image_aug) in gen:
            nb_channels = image.shape[2]
            if nb_channels in [3, 4]:
                chan_idx = self._CHANNEL_MAPPING[self.to_colorspace]
                image_tmp = image_conv
                image_tmp[..., chan_idx : chan_idx + 1] = image_aug

                result.append(None if nb_channels == 3 else image[..., 3:4])
                images_change_cs.append(image_tmp)
                images_change_cs_indices.append(i)
            else:
                result.append(image_aug)

        # invert colorspace conversion
        if len(images_change_cs) > 0:
            images_new_cs = color_lib.change_colorspaces_(
                images_change_cs,
                to_colorspaces=self.from_colorspace,
                from_colorspaces=self.to_colorspace,
            )
            for image_new_cs, target_idx in zip(
                images_new_cs, images_change_cs_indices, strict=True
            ):
                if result[target_idx] is None:
                    result[target_idx] = image_new_cs
                else:
                    # input image had four channels, 4th channel is already
                    # in result
                    result[target_idx] = np.dstack((image_new_cs, result[target_idx]))

        # convert to array if necessary
        if input_was_array:
            result = np.array(result, dtype=result[0].dtype)

        return result

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.from_colorspace, self.to_colorspace]


# TODO add parameter `tile_grid_size_percent`
