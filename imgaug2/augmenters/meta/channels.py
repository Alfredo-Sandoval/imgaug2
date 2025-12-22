"""Channel-specific augmenters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters._typing import Array, Images, RNGInput
from imgaug2.compat.markers import legacy

from .base import Augmenter
from .containers import handle_children_list
from .utils import _HasShape
class WithChannels(Augmenter):
    """Apply child augmenters to specific channels.

    Let ``C`` be one or more child augmenters given to this augmenter.
    Let ``H`` be a list of channels.
    Let ``I`` be the input images.
    Then this augmenter will pick the channels ``H`` from each image
    in ``I`` (resulting in new images) and apply ``C`` to them.
    The result of the augmentation will be merged back into the original
    images.

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
    channels : None or int or list of int, optional
        Sets the channels to be extracted from each image.
        If ``None``, all channels will be used. Note that this is not
        stochastic - the extracted channels are always the same ones.

    children : imgaug2.augmenters.meta.Augmenter or list of imgaug2.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to images, after the channels
        are extracted.

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
    >>> aug = iaa.WithChannels([0], iaa.Add(10))

    Assuming input images are RGB, then this augmenter will add ``10`` only to
    the first channel, i.e. it will make images appear more red.

    """

    def __init__(
        self,
        channels: int | Sequence[int] | None = None,
        children: Augmenter | Sequence[Augmenter] | None = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        # TODO change this to a stochastic parameter
        if channels is None:
            self.channels = None
        elif ia.is_single_integer(channels):
            self.channels = [channels]
        elif ia.is_iterable(channels):
            only_ints = all([ia.is_single_integer(channel) for channel in channels])
            assert only_ints, (
                f"Expected integers as channels, got {[type(channel) for channel in channels]}."
            )
            self.channels = channels
        else:
            raise Exception(
                f"Expected None, int or list of ints as channels, got {type(channels)}."
            )

        self.children = handle_children_list(children, self.name, "then")

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        if self.channels is not None and len(self.channels) == 0:
            return batch

        with batch.propagation_hooks_ctx(self, hooks, parents):
            batch_cp = batch.deepcopy()

            if batch.images is not None:
                batch.images = self._reduce_images_to_channels(batch.images)

            # Note that we augment here all data, including non-image data
            # for which less than 50% of the corresponding image channels
            # were augmented. This is because (a) the system does not yet
            # understand None as cell values and (b) decreasing the length
            # of columns leads to potential RNG misalignments.
            # We replace non-image data that was not supposed to be augmented
            # further below.
            batch = self.children.augment_batch_(batch, parents=parents + [self], hooks=hooks)

            # If the shapes changed we cannot insert the augmented channels
            # into the existing ones as the shapes of the non-augmented
            # channels are still the same.
            if batch.images is not None:
                self._assert_lengths_not_changed(batch.images, batch_cp.images)
                self._assert_shapes_not_changed(batch.images, batch_cp.images)
                self._assert_dtypes_not_changed(batch.images, batch_cp.images)

                batch.images = self._recover_images_array(batch.images, batch_cp.images)

            for column in batch.columns:
                if column.name != "images":
                    value_old = getattr(batch_cp, column.attr_name)
                    value = self._replace_unaugmented_cells(column.value, value_old)
                    setattr(batch, column.attr_name, value)

            if batch.images is not None:
                batch.images = self._invert_reduce_images_to_channels(batch.images, batch_cp.images)

        return batch

    @legacy(version="0.4.0")
    @classmethod
    def _assert_lengths_not_changed(cls, images_aug: Images, images: Images) -> None:
        assert len(images_aug) == len(images), (
            f"Expected that number of images does not change during "
            f"augmentation, but got {len(images_aug)} vs. originally {len(images)} images."
        )

    @legacy(version="0.4.0")
    @classmethod
    def _assert_shapes_not_changed(cls, images_aug: Images, images: Images) -> None:
        if ia.is_np_array(images_aug) and ia.is_np_array(images):
            shapes_same = images_aug.shape[1:3] == images.shape[1:3]
        else:
            shapes_same = all(
                [
                    image_aug.shape[0:2] == image.shape[0:2]
                    for image_aug, image in zip(images_aug, images, strict=True)
                ]
            )
        assert shapes_same, (
            "Heights/widths of images changed in WithChannels from "
            f"{str([image.shape[0:2] for image in images])} to {str([image_aug.shape[0:2] for image_aug in images_aug])}, but expected to be the same."
        )

    @legacy(version="0.4.0")
    @classmethod
    def _assert_dtypes_not_changed(cls, images_aug: Images, images: Images) -> None:
        if ia.is_np_array(images_aug) and ia.is_np_array(images):
            dtypes_same = images_aug.dtype == images.dtype
        else:
            dtypes_same = all(
                [
                    image_aug.dtype == image.dtype
                    for image_aug, image in zip(images_aug, images, strict=True)
                ]
            )

        assert dtypes_same, (
            "dtypes of images changed in WithChannels from "
            f"{str([image.dtype.name for image in images])} to {str([image_aug.dtype.name for image_aug in images_aug])}, but expected to be the same."
        )

    @legacy(version="0.4.0")
    @classmethod
    def _recover_images_array(cls, images_aug: Images, images: Images) -> Images:
        if ia.is_np_array(images):
            return np.array(images_aug)
        return images_aug

    @legacy(version="0.4.0")
    def _reduce_images_to_channels(self, images: Images) -> Images:
        if self.channels is None:
            return images
        if ia.is_np_array(images):
            if images.ndim >= 3 and images.shape[-1] == 0:
                return images
            return images[..., self.channels]
        result = []
        for image in images:
            if image.ndim >= 3 and image.shape[-1] == 0:
                result.append(image)
            else:
                result.append(image[..., self.channels])
        return result

    @legacy(version="0.4.0")
    def _invert_reduce_images_to_channels(self, images_aug: Images, images: Images) -> Images:
        if self.channels is None:
            return images_aug

        if ia.is_np_array(images_aug):
            if images_aug.ndim >= 3 and images_aug.shape[-1] == 0:
                return images_aug
            images[..., self.channels] = images_aug
            return images

        for image, image_aug in zip(images, images_aug, strict=True):
            if image.ndim >= 3 and image.shape[-1] == 0:
                continue
            image[..., self.channels] = image_aug
        return images

    @legacy(version="0.4.0")
    def _replace_unaugmented_cells(
        self, augmentables_aug: Sequence[object], augmentables: Sequence[object]
    ) -> list[object]:
        if self.channels is None:
            return list(augmentables_aug)

        nb_channels_to_aug = len(self.channels)
        nb_channels_lst: list[int] = []
        for augm in augmentables:
            assert hasattr(augm, "shape"), (
                f"Expected augmentable to have attribute `shape`, got {type(augm)}."
            )
            augm_shape = cast(_HasShape, augm).shape
            nb_channels_lst.append(augm_shape[2] if len(augm_shape) > 2 else 1)

        # We use the augmented form of a non-image if at least 50% of the
        # corresponding image's channels were augmented. Otherwise we use
        # the unaugmented form.
        fraction_augmented_lst = [
            nb_channels_to_aug / nb_channels for nb_channels in nb_channels_lst
        ]
        result = [
            (augmentable_aug if fraction_augmented >= 0.5 else augmentable)
            for augmentable_aug, augmentable, fraction_augmented in zip(
                augmentables_aug, augmentables, fraction_augmented_lst, strict=True
            )
        ]
        return result

    def _to_deterministic(self) -> WithChannels:
        aug = self.copy()
        aug.children = aug.children.to_deterministic()
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    def get_parameters(self) -> Sequence[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.channels]

    def get_children_lists(self) -> list[list[Augmenter]]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_children_lists`."""
        return [cast(list[Augmenter], self.children)]

    def __str__(self) -> str:
        pattern = "%s(channels=%s, name=%s, children=%s, deterministic=%s)"
        return pattern % (
            self.__class__.__name__,
            self.channels,
            self.name,
            self.children,
            self.deterministic,
        )

class ChannelShuffle(Augmenter):
    """Randomize the order of channels in input images.

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
    p : float or imgaug2.parameters.StochasticParameter, optional
        Probability of shuffling channels in any given image.
        May be a fixed probability as a ``float``, or a
        :class:`~imgaug2.parameters.StochasticParameter` that returns ``0`` s
        and ``1`` s.

    channels : None or imgaug2.ALL or list of int, optional
        Which channels are allowed to be shuffled with each other.
        If this is ``None`` or ``imgaug2.ALL``, then all channels may be
        shuffled. If it is a ``list`` of ``int`` s,
        then only the channels with indices in that list may be shuffled.
        (Values start at ``0``. All channel indices in the list must exist in
        each image.)

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
    >>> aug = iaa.ChannelShuffle(0.35)

    Shuffle all channels of ``35%`` of all images.

    >>> aug = iaa.ChannelShuffle(0.35, channels=[0, 1])

    Shuffle only channels ``0`` and ``1`` of ``35%`` of all images. As the new
    channel orders ``0, 1`` and ``1, 0`` are both valid outcomes of the
    shuffling, it means that for ``0.35 * 0.5 = 0.175`` or ``17.5%`` of all
    images the order of channels ``0`` and ``1`` is inverted.

    """

    def __init__(
        self,
        p: float | iap.StochasticParameter = 1.0,
        channels: list[int] | Literal["ALL"] | None = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.p = iap.handle_probability_param(p, "p")
        valid_channels = (
            channels is None
            or channels == ia.ALL
            or (isinstance(channels, list) and all([ia.is_single_integer(v) for v in channels]))
        )
        assert valid_channels, f"Expected None or imgaug2.ALL or list of int, got {type(channels)}."
        self.channels = channels

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        if batch.images is None:
            return batch

        images = batch.images

        nb_images = len(images)
        p_samples = self.p.draw_samples((nb_images,), random_state=random_state)
        rss = random_state.duplicate(nb_images)
        for i, (image, p_i, rs) in enumerate(zip(images, p_samples, rss, strict=True)):
            if p_i >= 1 - 1e-4:
                batch.images[i] = shuffle_channels(image, rs, self.channels)
        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p, self.channels]


def shuffle_channels(
    image: Array, random_state: iarandom.RNG, channels: list[int] | Literal["ALL"] | None = None
) -> Array:
    """Randomize the order of (color) channels in an image.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; indirectly tested (1)
        * ``uint32``: yes; indirectly tested (1)
        * ``uint64``: yes; indirectly tested (1)
        * ``int8``: yes; indirectly tested (1)
        * ``int16``: yes; indirectly tested (1)
        * ``int32``: yes; indirectly tested (1)
        * ``int64``: yes; indirectly tested (1)
        * ``float16``: yes; indirectly tested (1)
        * ``float32``: yes; indirectly tested (1)
        * ``float64``: yes; indirectly tested (1)
        * ``float128``: yes; indirectly tested (1)
        * ``bool``: yes; indirectly tested (1)

        - (1) Indirectly tested via :class:`ChannelShuffle`.

    Parameters
    ----------
    image : (H,W,[C]) ndarray
        Image of any dtype for which to shuffle the channels.

    random_state : imgaug2.random.RNG
        The random state to use for this shuffling operation.

    channels : None or imgaug2.ALL or list of int, optional
        Which channels are allowed to be shuffled with each other.
        If this is ``None`` or ``imgaug2.ALL``, then all channels may be
        shuffled. If it is a ``list`` of ``int`` s,
        then only the channels with indices in that list may be shuffled.
        (Values start at ``0``. All channel indices in the list must exist in
        the image.)

    Returns
    -------
    ndarray
        The input image with shuffled channels.

    """
    if image.ndim < 3 or image.shape[2] == 1:
        return image
    nb_channels = image.shape[2]
    all_channels = np.arange(nb_channels)
    is_all_channels = (
        channels is None
        or channels == ia.ALL
        or len(set(all_channels).difference(set(channels))) == 0
    )
    if is_all_channels:
        # note that if this is the case, then 'channels' may be None or
        # imgaug2.ALL, so don't simply move the assignment outside of the
        # if/else
        channels_perm = random_state.permutation(all_channels)
        return image[..., channels_perm]

    channels_perm = random_state.permutation(channels)
    channels_perm_full = all_channels
    for channel_source, channel_target in zip(channels, channels_perm, strict=True):
        channels_perm_full[channel_source] = channel_target
    return image[..., channels_perm_full]
