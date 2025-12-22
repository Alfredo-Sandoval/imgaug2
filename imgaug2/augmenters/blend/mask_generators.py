from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmentables.bbs import BoundingBoxesOnImage
from imgaug2.augmenters import color as colorlib
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _normalize_cv2_input_arr_

from .base import LabelInput, PerChannelInput, _split_1d_array_to_list

if TYPE_CHECKING:
    from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

@legacy(version="0.4.0")
class IBatchwiseMaskGenerator(metaclass=ABCMeta):
    """Interface for classes generating masks for batches.

    Child classes are supposed to receive a batch and generate an iterable
    of masks, one per row (i.e. image), matching the row shape (i.e. image
    shape). This is used in :class:`~imgaug2.augmenters.blend.BlendAlphaMask`.


    """

    @legacy(version="0.4.0")
    @abstractmethod
    def draw_masks(self, batch: _BatchInAugmentation, random_state: RNGInput = None) -> list[Array]:
        """Generate a mask with given shape.

        Parameters
        ----------
        batch : imgaug2.augmentables.batches._BatchInAugmentation
            Shape of the mask to sample.

        random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
            A seed or random number generator to use during the sampling
            process. If ``None``, the global RNG will be used.
            See also :func:`~imgaug2.augmenters.meta.Augmenter.__init__`
            for a similar parameter with more details.

        Returns
        -------
        iterable of ndarray
            Masks, one per row in the batch.
            Each mask must be a ``float32`` array in interval ``[0.0, 1.0]``.
            It must either have the same shape as the row (i.e. the image)
            or shape ``(H, W)`` if all channels are supposed to have the
            same mask.

        """
        raise NotImplementedError


@legacy(version="0.4.0")
class StochasticParameterMaskGen(IBatchwiseMaskGenerator):
    """Mask generator that queries stochastic parameters for mask values.

    This class receives batches for which to generate masks, iterates over
    the batch rows (i.e. images) and generates one mask per row.
    For a row with shape ``(H, W, C)`` (= image shape), it generates
    either a ``(H, W)`` mask (if ``per_channel`` is false-like) or a
    ``(H, W, C)`` mask (if ``per_channel`` is true-like).
    The ``per_channel`` is sampled per batch for each row/image.


    Parameters
    ----------
    parameter : imgaug2.parameters.StochasticParameter
        Stochastic parameter to draw mask samples from.
        Expected to return values in interval ``[0.0, 1.0]`` (not all
        stochastic parameters do that) and must be able to handle sampling
        shapes ``(H, W)`` and ``(H, W, C)`` (all stochastic parameters should
        do that).

    per_channel : bool or float or imgaug2.parameters.StochasticParameter, optional
        Whether to use the same mask for all channels (``False``)
        or to sample a new mask for each channel (``True``).
        If this value is a float ``p``, then for ``p`` percent of all rows
        (i.e. images) `per_channel` will be treated as ``True``, otherwise
        as ``False``.

    """

    @legacy(version="0.4.0")
    def __init__(self, parameter: iap.StochasticParameter, per_channel: PerChannelInput) -> None:
        super().__init__()
        self.parameter = parameter
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

    @legacy(version="0.4.0")
    def draw_masks(self, batch: _BatchInAugmentation, random_state: RNGInput = None) -> list[Array]:
        """
        See :func:`~imgaug2.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.

        """
        shapes = batch.get_rowwise_shapes()
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        per_channel = self.per_channel.draw_samples((len(shapes),), random_state=random_state)

        return [
            self._draw_mask(shape, random_state, per_channel_i)
            for shape, per_channel_i in zip(shapes, per_channel, strict=True)
        ]

    @legacy(version="0.4.0")
    def _draw_mask(
        self, shape: tuple[int, ...], random_state: iarandom.RNG, per_channel: float
    ) -> Array:
        if len(shape) == 2 or per_channel >= 0.5:
            mask = self.parameter.draw_samples(shape, random_state=random_state)
        else:
            # TODO When this was wrongly sampled directly as (H,W,C) no
            #      test for AlphaElementwise ended up failing. That should not
            #      happen.

            # We are guarantueed here to have (H, W, C) as shape (H, W) is
            # handled by the above block.
            # As the mask is not channelwise, we will just return (H, W)
            # instead of (H, W, C).
            mask = self.parameter.draw_samples(shape[0:2], random_state=random_state)

        # mask has no elements if height or width in shape is 0
        if mask.size > 0:
            assert 0 <= mask.item(0) <= 1.0, (
                "Expected 'parameter' samples to be in the interval "
                f"[0.0, 1.0]. Got min {np.min(mask):.4f} and max {np.max(mask):.4f}."
            )

        return mask


@legacy(version="0.4.0")
class SomeColorsMaskGen(IBatchwiseMaskGenerator):
    """Generator that produces masks based on some similar colors in images.

    This class receives batches for which to generate masks, iterates over
    the batch rows (i.e. images) and generates one mask per row.
    The mask contains high alpha values for some colors, while other colors
    get low mask values. Which colors are chosen is random. How wide or
    narrow the selection is (e.g. very specific blue tone or all blue-ish
    colors) is determined by the hyperparameters.

    The color selection method performs roughly the following steps:

      1. Split the full color range of the hue in ``HSV`` into ``nb_bins``
         bins (i.e. ``256/nb_bins`` different possible hue tones).
      2. Shift the bins by ``rotation_deg`` degrees. (This way, the ``0th``
         bin does not always start at exactly ``0deg`` of hue.)
      3. Sample ``alpha`` values for each bin.
      4. Repeat the ``nb_bins`` bins until there are ``256`` bins.
      5. Smoothen the alpha values of neighbouring bins using a gaussian
         kernel. The kernel's ``sigma`` is derived from ``smoothness``.
      6. Associate all hue values in the image with the corresponding bin's
         alpha value. This results in the alpha mask.

    .. note::

        This mask generator will produce an ``AssertionError`` for batches
        that contain no images.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.color.change_colorspaces_`.

    Parameters
    ----------
    nb_bins : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Number of bins. For ``B`` bins, each bin denotes roughly ``360/B``
        degrees of colors in the hue channel. Lower values lead to a coarser
        selection of colors. Expected value range is ``[2, 256]``.

            * If ``int``: Exactly that value will be used for all images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
              per image from the discrete interval ``[a..b]``.
            * If ``list``: A random value will be picked per image from that
              list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(N,)`` values -- one per image.

    smoothness : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Strength of the 1D gaussian kernel applied to the sampled binwise
        alpha values. Larger values will lead to more similar grayscaling of
        neighbouring colors. Expected value range is ``[0.0, 1.0]``.

            * If ``number``: Exactly that value will be used for all images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If ``list``: A random value will be picked per image from that
              list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(N,)`` values -- one per image.

    alpha : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Parameter to sample binwise alpha blending factors from. Expected
        value range is ``[0.0, 1.0]``.  Note that the alpha values will be
        smoothed between neighbouring bins. Hence, it is usually a good idea
        to set this so that the probability distribution peaks are around
        ``0.0`` and ``1.0``, e.g. via a list ``[0.0, 1.0]`` or a ``Beta``
        distribution.
        It is not recommended to set this to a deterministic value, otherwise
        all bins and hence all pixels in the generated mask will have the
        same value.

            * If ``number``: Exactly that value will be used for all bins.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
              per bin from the interval ``[a, b]``.
            * If ``list``: A random value will be picked per bin from that list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(N*B,)`` values -- one per image and bin.

    rotation_deg : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Rotiational shift of each bin as a fraction of ``360`` degrees.
        E.g. ``0.0`` will not shift any bins, while a value of ``0.5`` will
        shift by around ``180`` degrees. This shift is mainly used so that
        the ``0th`` bin does not always start at ``0deg``. Expected value
        range is ``[-360, 360]``. This parameter can usually be kept at the
        default value.

            * If ``number``: Exactly that value will be used for all images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If ``list``: A random value will be picked per image from that
              list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(N,)`` values -- one per image.

    from_colorspace : str, optional
        The source colorspace (of the input images).
        See :func:`~imgaug2.augmenters.color.change_colorspace_`.

    """

    # TODO colorlib.CSPACE_RGB produces 'has no attribute' error?
    @legacy(version="0.4.0")
    def __init__(
        self,
        nb_bins: ParamInput = (5, 15),
        smoothness: ParamInput = (0.1, 0.3),
        alpha: ParamInput | None = None,
        rotation_deg: ParamInput = (0, 360),
        from_colorspace: str = "RGB",
    ) -> None:
        super().__init__()

        self.nb_bins = iap.handle_discrete_param(
            nb_bins, "nb_bins", value_range=(1, 256), tuple_to_uniform=True, list_to_choice=True
        )
        self.smoothness = iap.handle_continuous_param(
            smoothness,
            "smoothness",
            value_range=(0.0, 1.0),
            tuple_to_uniform=True,
            list_to_choice=True,
        )
        self.alpha = iap.handle_continuous_param(
            alpha if alpha is not None else [0.0, 1.0],
            "alpha",
            value_range=(0.0, 1.0),
            tuple_to_uniform=True,
            list_to_choice=True,
        )
        self.rotation_deg = iap.handle_continuous_param(
            rotation_deg,
            "rotation_deg",
            value_range=(-360, 360),
            tuple_to_uniform=True,
            list_to_choice=True,
        )
        self.from_colorspace = from_colorspace

        self.sigma_max = 10.0

    @legacy(version="0.4.0")
    def draw_masks(self, batch: _BatchInAugmentation, random_state: RNGInput = None) -> list[Array]:
        """
        See :func:`~imgaug2.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.

        """
        assert batch.images is not None, (
            "Can only generate masks for batches that contain images, but "
            "got a batch without images."
        )
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        samples = self._draw_samples(batch, random_state=random_state)

        return [self._draw_mask(image, i, samples) for i, image in enumerate(batch.images)]

    @legacy(version="0.4.0")
    def _draw_mask(
        self, image: Array, image_idx: int, samples: tuple[list[Array], Array, Array]
    ) -> Array:
        return self.generate_mask(
            image,
            samples[0][image_idx],
            samples[1][image_idx] * self.sigma_max,
            samples[2][image_idx],
            self.from_colorspace,
        )

    @legacy(version="0.4.0")
    def _draw_samples(
        self, batch: _BatchInAugmentation, random_state: iarandom.RNG
    ) -> tuple[list[Array], Array, Array]:
        nb_rows = batch.nb_rows
        nb_bins = self.nb_bins.draw_samples((nb_rows,), random_state=random_state)
        smoothness = self.smoothness.draw_samples((nb_rows,), random_state=random_state)
        alpha = self.alpha.draw_samples((np.sum(nb_bins),), random_state=random_state)
        rotation_deg = self.rotation_deg.draw_samples((nb_rows,), random_state=random_state)

        nb_bins = np.clip(nb_bins, 1, 256)
        smoothness = np.clip(smoothness, 0.0, 1.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        rotation_bins = np.mod(np.round(rotation_deg * (256 / 360)).astype(np.int32), 256)

        binwise_alphas = _split_1d_array_to_list(alpha, nb_bins)

        return binwise_alphas, smoothness, rotation_bins

    @legacy(version="0.4.0")
    @classmethod
    def generate_mask(
        cls,
        image: Array,
        binwise_alphas: Array,
        sigma: float,
        rotation_bins: int,
        from_colorspace: str,
    ) -> Array:
        """Generate a colorwise alpha mask for a single image.


        Parameters
        ----------
        image : ndarray
            Image for which to generate the mask. Must have shape ``(H,W,3)``
            in colorspace `from_colorspace`.

        binwise_alphas : ndarray
            Alpha values of shape ``(B,)`` with ``B`` in ``[1, 256]``
            and values in interval ``[0.0, 1.0]``. Will be upscaled to
            256 bins by simple repetition. Each bin represents ``1/256`` th
            of the hue.

        sigma : float
            Sigma of the 1D gaussian kernel applied to the upscaled binwise
            alpha value array.

        rotation_bins : int
            By how much to rotate the 256 bin alpha array. The rotation is
            given in number of bins.

        from_colorspace : str
            Colorspace of the input image. One of
            ``imgaug2.augmenters.color.CSPACE_*``.

        Returns
        -------
        ndarray
            ``float32`` mask array of shape ``(H, W)`` with values in
            ``[0.0, 1.0]``

        """
        image_hsv = colorlib.change_colorspace_(
            np.copy(image), to_colorspace=colorlib.CSPACE_HSV, from_colorspace=from_colorspace
        )

        if 0 in image_hsv.shape[0:2]:
            return np.zeros(image_hsv.shape[0:2], dtype=np.float32)

        binwise_alphas = cls._upscale_to_256_alpha_bins(binwise_alphas)
        binwise_alphas = cls._rotate_alpha_bins(binwise_alphas, rotation_bins)
        binwise_alphas_smooth = cls._smoothen_alphas(binwise_alphas, sigma)

        mask = cls._generate_pixelwise_alpha_mask(image_hsv, binwise_alphas_smooth)

        return mask

    @legacy(version="0.4.0")
    @classmethod
    def _upscale_to_256_alpha_bins(cls, alphas: Array) -> Array:
        # repeat alphas bins so that B sampled bins become 256 bins
        nb_bins = len(alphas)
        nb_repeats_per_bin = int(np.ceil(256 / nb_bins))
        alphas = np.repeat(alphas, (nb_repeats_per_bin,))
        alphas = alphas[0:256]
        return alphas

    @legacy(version="0.4.0")
    @classmethod
    def _rotate_alpha_bins(cls, alphas: Array, rotation_bins: int) -> Array:
        # e.g. for offset 2: abcdef -> cdefab
        # note: offset here is expected to be in [0, 256]
        if rotation_bins > 0:
            alphas = np.roll(alphas, -rotation_bins)
        return alphas

    @legacy(version="0.4.0")
    @classmethod
    def _smoothen_alphas(cls, alphas: Array, sigma: float) -> Array:
        if sigma <= 0.0 + 1e-2:
            return alphas

        ksize = max(int(sigma * 2.5), 3)
        ksize_y, ksize_x = (1, ksize)
        if ksize_x % 2 == 0:
            ksize_x += 1

        # we fake here cv2.BORDER_WRAP, because GaussianBlur does not
        # support that mode, i.e. we want:
        #   cdefgh|abcdefgh|abcdefg
        alphas = np.concatenate(
            [
                alphas[-ksize_x:],
                alphas,
                alphas[:ksize_x],
            ]
        )

        alphas = cv2.GaussianBlur(
            _normalize_cv2_input_arr_(alphas[np.newaxis, :]),
            ksize=(ksize_x, ksize_y),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )[0, :]

        # revert fake BORDER_WRAP
        alphas = alphas[ksize_x:-ksize_x]

        return alphas

    @legacy(version="0.4.0")
    @classmethod
    def _generate_pixelwise_alpha_mask(cls, image_hsv: Array, hue_to_alpha: Array) -> Array:
        hue = image_hsv[:, :, 0]
        table = hue_to_alpha * 255
        table = np.clip(np.round(table), 0, 255).astype(np.uint8)
        mask = ia.apply_lut(hue, table)
        return mask.astype(np.float32) / 255.0


@legacy(version="0.4.0")
class _LinearGradientMaskGen(IBatchwiseMaskGenerator):
    @legacy(version="0.4.0")
    def __init__(
        self,
        axis: int,
        min_value: ParamInput = 0.0,
        max_value: ParamInput = 1.0,
        start_at: ParamInput = 0.0,
        end_at: ParamInput = 1.0,
    ) -> None:
        self.axis = axis
        self.min_value = iap.handle_continuous_param(
            min_value,
            "min_value",
            value_range=(0.0, 1.0),
            tuple_to_uniform=True,
            list_to_choice=True,
        )
        self.max_value = iap.handle_continuous_param(
            max_value,
            "max_value",
            value_range=(0.0, 1.0),
            tuple_to_uniform=True,
            list_to_choice=True,
        )
        self.start_at = iap.handle_continuous_param(
            start_at, "start_at", value_range=(0.0, 1.0), tuple_to_uniform=True, list_to_choice=True
        )
        self.end_at = iap.handle_continuous_param(
            end_at, "end_at", value_range=(0.0, 1.0), tuple_to_uniform=True, list_to_choice=True
        )

    @legacy(version="0.4.0")
    def draw_masks(self, batch: _BatchInAugmentation, random_state: RNGInput = None) -> list[Array]:
        """
        See :func:`~imgaug2.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.


        """
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        shapes = batch.get_rowwise_shapes()
        samples = self._draw_samples(len(shapes), random_state=random_state)

        return [self._draw_mask(shape, i, samples) for i, shape in enumerate(shapes)]

    @legacy(version="0.4.0")
    def _draw_mask(
        self, shape: tuple[int, ...], image_idx: int, samples: tuple[Array, Array, Array, Array]
    ) -> Array:
        return self.generate_mask(
            shape,
            samples[0][image_idx],
            samples[1][image_idx],
            samples[2][image_idx],
            samples[3][image_idx],
        )

    @legacy(version="0.4.0")
    def _draw_samples(
        self, nb_rows: int, random_state: iarandom.RNG
    ) -> tuple[Array, Array, Array, Array]:
        min_value = self.min_value.draw_samples((nb_rows,), random_state=random_state)
        max_value = self.max_value.draw_samples((nb_rows,), random_state=random_state)
        start_at = self.start_at.draw_samples((nb_rows,), random_state=random_state)
        end_at = self.end_at.draw_samples((nb_rows,), random_state=random_state)

        return min_value, max_value, start_at, end_at

    @legacy(version="0.4.0")
    @classmethod
    @abstractmethod
    def generate_mask(
        cls,
        shape: tuple[int, ...],
        min_value: float,
        max_value: float,
        start_at: float,
        end_at: float,
    ) -> Array:
        """Generate a horizontal gradient mask.


        Parameters
        ----------
        shape : tuple of int
            Shape of the image. The mask will have the same height and
            width.

        min_value : number
            Minimum value of the gradient in interval ``[0.0, 1.0]``.

        max_value : number
            Maximum value of the gradient in interval ``[0.0, 1.0]``.

        start_at : number
            Position on the x-axis where the linear gradient starts, given as
            a fraction of the axis size. Interval is ``[0.0, 1.0]``.

        end_at : number
            Position on the x-axis where the linear gradient ends, given as
            a fraction of the axis size. Interval is ``[0.0, 1.0]``.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as the image.
            Values are in ``[0.0, 1.0]``.

        """
        raise NotImplementedError

    @legacy(version="0.4.0")
    @classmethod
    def _generate_mask(
        cls,
        shape: tuple[int, ...],
        axis: int,
        min_value: float,
        max_value: float,
        start_at: float,
        end_at: float,
    ) -> Array:
        height, width = shape[0:2]

        axis_size = shape[axis]
        min_value = min(max(min_value, 0.0), 1.0)
        max_value = min(max(max_value, 0.0), 1.0)

        start_at_px = min(max(int(start_at * axis_size), 0), axis_size)
        end_at_px = min(max(int(end_at * axis_size), 0), axis_size)

        inverted = False
        if end_at_px < start_at_px:
            inverted = True
            start_at_px, end_at_px = end_at_px, start_at_px

        before_grad = np.full((start_at_px,), min_value, dtype=np.float32)
        grad = np.linspace(
            start=min_value, stop=max_value, num=end_at_px - start_at_px, dtype=np.float32
        )
        after_grad = np.full((axis_size - end_at_px,), max_value, dtype=np.float32)

        mask = np.concatenate((before_grad, grad, after_grad), axis=0)

        if inverted:
            mask = 1.0 - mask

        if axis == 0:
            mask = mask[:, np.newaxis]
            mask = np.tile(mask, (1, width))
        else:
            mask = mask[np.newaxis, :]
            mask = np.tile(mask, (height, 1))

        return mask


@legacy(version="0.4.0")
class HorizontalLinearGradientMaskGen(_LinearGradientMaskGen):
    """Generator that produces horizontal linear gradient masks.

    This class receives batches and produces for each row (i.e. image)
    a horizontal linear gradient that matches the row's shape (i.e. image
    shape). The gradient increases linearly from a minimum value to a
    maximum value along the x-axis. The start and end points (i.e. where the
    minimum value starts to increase and where it reaches the maximum)
    may be defines as fractions of the width. E.g. for width ``100`` and
    ``start=0.25``, ``end=0.75``, the gradient would have its minimum
    in interval ``[0px, 25px]`` and its maximum in interval ``[75px, 100px]``.

    Note that this has nothing to do with a *derivative* along the x-axis.


    Parameters
    ----------
    min_value : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Minimum value that the mask will have up to the start point of the
        linear gradient.
        Note that `min_value` is allowed to be larger than `max_value`,
        in which case the gradient will start at the (higher) `min_value`
        and decrease towards the (lower) `max_value`.

        * If ``number``: Exactly that value will be used for all images.
        * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
          per image from the interval ``[a, b]``.
        * If ``list``: A random value will be picked per image from that list.
        * If ``StochasticParameter``: That parameter will be queried once
          per batch for ``(N,)`` values -- one per image.

    max_value : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Maximum value that the mask will have at the end of the
        linear gradient.

        Datatypes are analogous to `min_value`.

    start_at : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Position on the x-axis where the linear gradient starts, given as a
        fraction of the axis size. Interval is ``[0.0, 1.0]``, where ``0.0``
        is at the left of the image.
        If ``end_at < start_at`` the gradient will be inverted.

        Datatypes are analogous to `min_value`.

    end_at : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Position on the x-axis where the linear gradient ends, given as a
        fraction of the axis size. Interval is ``[0.0, 1.0]``, where ``0.0``
        is at the right of the image.

        Datatypes are analogous to `min_value`.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        min_value: ParamInput = (0.0, 0.2),
        max_value: ParamInput = (0.8, 1.0),
        start_at: ParamInput = (0.0, 0.2),
        end_at: ParamInput = (0.8, 1.0),
    ) -> None:
        super().__init__(
            axis=1, min_value=min_value, max_value=max_value, start_at=start_at, end_at=end_at
        )

    @legacy(version="0.4.0")
    @classmethod
    def generate_mask(
        cls,
        shape: tuple[int, ...],
        min_value: float,
        max_value: float,
        start_at: float,
        end_at: float,
    ) -> Array:
        """Generate a linear horizontal gradient mask.


        Parameters
        ----------
        shape : tuple of int
            Shape of the image. The mask will have the same height and
            width.

        min_value : number
            Minimum value of the gradient in interval ``[0.0, 1.0]``.

        max_value : number
            Maximum value of the gradient in interval ``[0.0, 1.0]``.

        start_at : number
            Position on the x-axis where the linear gradient starts, given as
            a fraction of the axis size. Interval is ``[0.0, 1.0]``.

        end_at : number
            Position on the x-axis where the linear gradient ends, given as
            a fraction of the axis size. Interval is ``[0.0, 1.0]``.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as the image.
            Values are in ``[0.0, 1.0]``.

        """
        return cls._generate_mask(
            axis=1,
            shape=shape,
            min_value=min_value,
            max_value=max_value,
            start_at=start_at,
            end_at=end_at,
        )


@legacy(version="0.4.0")
class VerticalLinearGradientMaskGen(_LinearGradientMaskGen):
    """Generator that produces vertical linear gradient masks.

    See :class:`~imgaug2.augmenters.blend.HorizontalLinearGradientMaskGen`
    for details.


    Parameters
    ----------
    min_value : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Minimum value that the mask will have up to the start point of the
        linear gradient.
        Note that `min_value` is allowed to be larger than `max_value`,
        in which case the gradient will start at the (higher) `min_value`
        and decrease towards the (lower) `max_value`.

        * If ``number``: Exactly that value will be used for all images.
        * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
          per image from the interval ``[a, b]``.
        * If ``list``: A random value will be picked per image from that list.
        * If ``StochasticParameter``: That parameter will be queried once
          per batch for ``(N,)`` values -- one per image.

    max_value : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Maximum value that the mask will have at the end of the
        linear gradient.

        Datatypes are analogous to `min_value`.

    start_at : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Position on the y-axis where the linear gradient starts, given as a
        fraction of the axis size. Interval is ``[0.0, 1.0]``, where ``0.0``
        is at the top of the image.
        If ``end_at < start_at`` the gradient will be inverted.

        Datatypes are analogous to `min_value`.

    end_at : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Position on the x-axis where the linear gradient ends, given as a
        fraction of the axis size. Interval is ``[0.0, 1.0]``, where ``1.0``
        is at the bottom of the image.

        Datatypes are analogous to `min_value`.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        min_value: ParamInput = (0.0, 0.2),
        max_value: ParamInput = (0.8, 1.0),
        start_at: ParamInput = (0.0, 0.2),
        end_at: ParamInput = (0.8, 1.0),
    ) -> None:
        super().__init__(
            axis=0, min_value=min_value, max_value=max_value, start_at=start_at, end_at=end_at
        )

    @legacy(version="0.4.0")
    @classmethod
    def generate_mask(
        cls,
        shape: tuple[int, ...],
        min_value: float,
        max_value: float,
        start_at: float,
        end_at: float,
    ) -> Array:
        """Generate a linear horizontal gradient mask.


        Parameters
        ----------
        shape : tuple of int
            Shape of the image. The mask will have the same height and
            width.

        min_value : number
            Minimum value of the gradient in interval ``[0.0, 1.0]``.

        max_value : number
            Maximum value of the gradient in interval ``[0.0, 1.0]``.

        start_at : number
            Position on the x-axis where the linear gradient starts, given as
            a fraction of the axis size. Interval is ``[0.0, 1.0]``.

        end_at : number
            Position on the x-axis where the linear gradient ends, given as
            a fraction of the axis size. Interval is ``[0.0, 1.0]``.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as the image.
            Values are in ``[0.0, 1.0]``.

        """
        return cls._generate_mask(
            axis=0,
            shape=shape,
            min_value=min_value,
            max_value=max_value,
            start_at=start_at,
            end_at=end_at,
        )


@legacy(version="0.4.0")
class RegularGridMaskGen(IBatchwiseMaskGenerator):
    """Generate masks following a regular grid pattern.

    This mask generator splits each image into a grid-like pattern of
    ``H`` rows and ``W`` columns. Each cell is then filled with an alpha
    value, sampled randomly per cell.

    The difference to :class:`CheckerboardMaskGen` is that this mask generator
    samples random alpha values per cell, while in the checkerboard the
    alpha values follow a fixed pattern.


    Parameters
    ----------
    nb_rows : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter
        Number of rows of the regular grid.

            * If ``int``: Exactly that value will be used for all images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
              per image from the discrete interval ``[a..b]``.
            * If ``list``: A random value will be picked per image from that
              list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(N,)`` values -- one per image.

    nb_cols : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter
        Number of columns of the checkerboard. Analogous to `nb_rows`.

    alpha : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Alpha value of each cell.

        * If ``number``: Exactly that value will be used for all images.
        * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
          per image from the interval ``[a, b]``.
        * If ``list``: A random value will be picked per image from that list.
        * If ``StochasticParameter``: That parameter will be queried once
          per batch for ``(N,)`` values -- one per image.

    """

    @legacy(version="0.4.0")
    def __init__(
        self, nb_rows: ParamInput, nb_cols: ParamInput, alpha: ParamInput | None = None
    ) -> None:
        self.nb_rows = iap.handle_discrete_param(
            nb_rows,
            "nb_rows",
            value_range=(1, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self.nb_cols = iap.handle_discrete_param(
            nb_cols,
            "nb_cols",
            value_range=(1, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self.alpha = iap.handle_continuous_param(
            alpha if alpha is not None else [0.0, 1.0],
            "alpha",
            value_range=(0.0, 1.0),
            tuple_to_uniform=True,
            list_to_choice=True,
        )

    @legacy(version="0.4.0")
    def draw_masks(self, batch: _BatchInAugmentation, random_state: RNGInput = None) -> list[Array]:
        """
        See :func:`~imgaug2.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.


        """
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        shapes = batch.get_rowwise_shapes()
        nb_rows, nb_cols, alpha = self._draw_samples(len(shapes), random_state=random_state)

        return [
            self.generate_mask(shape, nb_rows_i, nb_cols_i, alpha_i)
            for shape, nb_rows_i, nb_cols_i, alpha_i in zip(
                shapes, nb_rows, nb_cols, alpha, strict=True
            )
        ]

    @legacy(version="0.4.0")
    def _draw_samples(
        self, nb_images: int, random_state: iarandom.RNG
    ) -> tuple[Array, Array, list[Array]]:
        nb_rows = self.nb_rows.draw_samples((nb_images,), random_state=random_state)
        nb_cols = self.nb_cols.draw_samples((nb_images,), random_state=random_state)
        nb_alphas_per_img = nb_rows * nb_cols
        alpha_raw = self.alpha.draw_samples((np.sum(nb_alphas_per_img),), random_state=random_state)

        alpha = _split_1d_array_to_list(alpha_raw, nb_alphas_per_img)

        return nb_rows, nb_cols, alpha

    @legacy(version="0.4.0")
    @classmethod
    def generate_mask(
        cls, shape: tuple[int, ...], nb_rows: int, nb_cols: int, alphas: Array
    ) -> Array:
        """Generate a mask following a checkerboard pattern.


        Parameters
        ----------
        shape : tuple of int
            Height and width of the output mask.

        nb_rows : int
            Number of rows of the checkerboard pattern.

        nb_cols : int
            Number of columns of the checkerboard pattern.

        alphas : ndarray
            1D or 2D array containing for each cell the alpha value, i.e.
            ``nb_rows*nb_cols`` values.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as
            ``segmap.shape``. Values are in ``[0.0, 1.0]``.

        """
        from imgaug2.augmenters import size as sizelib

        height, width = shape[0:2]
        if 0 in (height, width):
            return np.zeros((height, width), dtype=np.float32)

        nb_rows = min(max(nb_rows, 1), height)
        nb_cols = min(max(nb_cols, 1), width)

        cell_height = int(height / nb_rows)
        cell_width = int(width / nb_cols)

        # If there are more alpha values than nb_rows*nb_cols we reduce the
        # number of alpha values.
        alphas = alphas.flat[0 : nb_rows * nb_cols]
        assert alphas.size == nb_rows * nb_cols, (
            "Expected `alphas` to not contain less values than "
            "`nb_rows * nb_cols` (both clipped to [1, height] and "
            f"[1, width] respectively). Got {alphas.size} alpha values vs {nb_rows * nb_cols} expected "
            f"values (nb_rows={nb_rows}, nb_cols={nb_cols}) for requested mask shape {(height, width)}."
        )
        mask = alphas.astype(np.float32).reshape((nb_rows, nb_cols))
        mask = np.repeat(mask, cell_height, axis=0)
        mask = np.repeat(mask, cell_width, axis=1)

        # if mask is too small, reflection pad it on all sides
        missing_height = height - mask.shape[0]
        missing_width = width - mask.shape[1]
        top = int(np.floor(missing_height / 2))
        bottom = int(np.ceil(missing_height / 2))
        left = int(np.floor(missing_width / 2))
        right = int(np.ceil(missing_width / 2))
        mask = sizelib.pad(mask, top=top, right=right, bottom=bottom, left=left, mode="reflect")

        return mask


@legacy(version="0.4.0")
class CheckerboardMaskGen(IBatchwiseMaskGenerator):
    """Generate masks following a checkerboard-like pattern.

    This mask generator splits each image into a regular grid of
    ``H`` rows and ``W`` columns. Each cell is then filled with either
    ``1.0`` or ``0.0``. The cell at the top-left is always ``1.0``. Its right
    and bottom neighbour cells are ``0.0``. The 4-neighbours of any cell always
    have a value opposite to the cell's value (``0.0`` vs. ``1.0``).


    Parameters
    ----------
    nb_rows : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Number of rows of the checkerboard.

            * If ``int``: Exactly that value will be used for all images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
              per image from the discrete interval ``[a..b]``.
            * If ``list``: A random value will be picked per image from that
              list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(N,)`` values -- one per image.

    nb_cols : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Number of columns of the checkerboard. Analogous to `nb_rows`.

    """

    def __init__(self, nb_rows: ParamInput, nb_cols: ParamInput) -> None:
        self.grid = RegularGridMaskGen(nb_rows=nb_rows, nb_cols=nb_cols, alpha=1)

    @legacy(version="0.4.0")
    @property
    def nb_rows(self) -> iap.StochasticParameter:
        """Get the number of rows of the checkerboard grid.


        Returns
        -------
        int
            The number of rows.

        """
        return self.grid.nb_rows

    @legacy(version="0.4.0")
    @property
    def nb_cols(self) -> iap.StochasticParameter:
        """Get the number of columns of the checkerboard grid.


        Returns
        -------
        int
            The number of columns.

        """
        return self.grid.nb_cols

    @legacy(version="0.4.0")
    def draw_masks(self, batch: _BatchInAugmentation, random_state: RNGInput = None) -> list[Array]:
        """
        See :func:`~imgaug2.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.


        """
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        shapes = batch.get_rowwise_shapes()
        nb_rows, nb_cols, _alpha = self.grid._draw_samples(len(shapes), random_state=random_state)

        return [
            self.generate_mask(shape, nb_rows_i, nb_cols_i)
            for shape, nb_rows_i, nb_cols_i in zip(shapes, nb_rows, nb_cols, strict=True)
        ]

    @legacy(version="0.4.0")
    @classmethod
    def generate_mask(cls, shape: tuple[int, ...], nb_rows: int, nb_cols: int) -> Array:
        """Generate a mask following a checkerboard pattern.


        Parameters
        ----------
        shape : tuple of int
            Height and width of the output mask.

        nb_rows : int
            Number of rows of the checkerboard pattern.

        nb_cols : int
            Number of columns of the checkerboard pattern.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as
            ``segmap.shape``. Values are in ``[0.0, 1.0]``.

        """
        height, width = shape[0:2]
        if 0 in (height, width):
            return np.zeros((height, width), dtype=np.float32)
        nb_rows = min(max(nb_rows, 1), height)
        nb_cols = min(max(nb_cols, 1), width)

        alphas = np.full((nb_cols,), 1.0, dtype=np.float32)
        alphas[::2] = 0.0
        alphas = np.tile(alphas[np.newaxis, :], (nb_rows, 1))
        alphas[::2, :] = 1.0 - alphas[::2, :]

        return RegularGridMaskGen.generate_mask(shape, nb_rows, nb_cols, alphas)


@legacy(version="0.4.0")
class SegMapClassIdsMaskGen(IBatchwiseMaskGenerator):
    """Generator that produces masks highlighting segmentation map classes.

    This class produces for each segmentation map in a batch a mask in which
    the locations of a set of provided classes are highlighted (i.e. ``1.0``).
    The classes may be provided as a fixed list of class ids or a stochastic
    parameter from which class ids will be sampled.

    The produced masks are initially of the same height and width as the
    segmentation map arrays and later upscaled to the image height and width.

    .. note::

        Segmentation maps can have multiple channels. If that is the case
        then for each position ``(x, y)`` it is sufficient that any class id
        in any channel matches one of the desired class ids.

    .. note::

        This class will produce an ``AssertionError`` if there are no
        segmentation maps in a batch.


    Parameters
    ----------
    class_ids : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter
        Segmentation map classes to mark in the produced mask.

        If `nb_sample_classes` is ``None`` then this is expected to be either
        a single ``int`` (always mark this one class id) or a ``list`` of
        ``int`` s (always mark these class ids).

        If `nb_sample_classes` is set, then this parameter will be treated
        as a stochastic parameter with the following valid types:

            * If ``int``: Exactly that class id will be used for all
              segmentation maps.
            * If ``tuple`` ``(a, b)``: ``N`` random values will be uniformly
              sampled per segmentation map from the discrete interval
              ``[a..b]`` and used as the class ids.
            * If ``list``: ``N`` random values will be picked per segmentation
              map from that list and used as the class ids.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(sum(N),)`` values.

        ``N`` denotes the number of classes to sample per segmentation
        map (derived from `nb_sample_classes`) and ``sum(N)`` denotes the
        sum of ``N`` s over all segmentation maps.

    nb_sample_classes : None or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Number of class ids to sample (with replacement) per segmentation map.
        As sampling happens with replacement, fewer *unique* class ids may be
        sampled.

            * If ``None``: `class_ids` is expected to be a fixed value of
              class ids to be used for all segmentation maps.
            * If ``int``: Exactly that many class ids will be sampled for all
              segmentation maps.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly
              sampled per segmentation map from the discrete interval
              ``[a..b]``.
            * If ``list`` or ``int``: A random value will be picked per
              segmentation map from that list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(B,)`` values, where ``B`` is the number of
              segmentation maps.

    """

    @legacy(version="0.4.0")
    def __init__(self, class_ids: ParamInput, nb_sample_classes: ParamInput | None = None) -> None:
        if nb_sample_classes is None:
            if ia.is_single_integer(class_ids):
                class_ids = [class_ids]
            assert isinstance(class_ids, list), (
                "Expected `class_ids` to be a single integer or a list of "
                f"integers if `nb_sample_classes` is None. Got type `{type(class_ids).__name__}`. "
                "Set `nb_sample_classes` to e.g. an integer to enable "
                "stochastic parameters for `class_ids`."
            )
            self.class_ids = class_ids
            self.nb_sample_classes = None
        else:
            self.class_ids = iap.handle_discrete_param(
                class_ids,
                "class_ids",
                value_range=(0, None),
                tuple_to_uniform=True,
                list_to_choice=True,
                allow_floats=False,
            )
            self.nb_sample_classes = iap.handle_discrete_param(
                nb_sample_classes,
                "nb_sample_classes",
                value_range=(0, None),
                tuple_to_uniform=True,
                list_to_choice=True,
                allow_floats=False,
            )

    @legacy(version="0.4.0")
    def draw_masks(self, batch: _BatchInAugmentation, random_state: RNGInput = None) -> list[Array]:
        """
        See :func:`~imgaug2.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.


        """
        assert batch.segmentation_maps is not None, (
            "Can only generate masks for batches that contain segmentation "
            "maps, but got a batch without them."
        )
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        class_ids = self._draw_samples(batch.nb_rows, random_state=random_state)

        return [
            self.generate_mask(segmap, class_ids_i)
            for segmap, class_ids_i in zip(batch.segmentation_maps, class_ids, strict=True)
        ]

    @legacy(version="0.4.0")
    def _draw_samples(self, nb_rows: int, random_state: iarandom.RNG) -> list[Sequence[int]]:
        nb_sample_classes = self.nb_sample_classes
        if nb_sample_classes is None:
            assert isinstance(self.class_ids, list), (
                f"Expected list got {type(self.class_ids).__name__}."
            )
            return cast(list[Sequence[int]], [self.class_ids] * nb_rows)

        nb_sample_classes = nb_sample_classes.draw_samples((nb_rows,), random_state=random_state)
        nb_sample_classes = np.clip(nb_sample_classes, 0, None)
        class_ids_raw = self.class_ids.draw_samples(
            (np.sum(nb_sample_classes),), random_state=random_state
        )

        class_ids = _split_1d_array_to_list(class_ids_raw, nb_sample_classes)

        return cast(list[Sequence[int]], class_ids)

    # TODO this could be simplified to something like:
    #      segmap.keep_only_classes(class_ids).draw_mask()
    @legacy(version="0.4.0")
    @classmethod
    def generate_mask(cls, segmap: SegmentationMapsOnImage, class_ids: Sequence[int]) -> Array:
        """Generate a mask of where the segmentation map has the given classes.


        Parameters
        ----------
        segmap : imgaug2.augmentables.segmap.SegmentationMapsOnImage
            The segmentation map for which to generate the mask.

        class_ids : iterable of int
            IDs of the classes to set to ``1.0``.
            For an ``(x, y)`` position, it is enough that *any* channel
            at the given location to have one of these class ids to be marked
            as ``1.0``.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as
            ``segmap.shape``. Values are in ``[0.0, 1.0]``.

        """
        mask = np.zeros(segmap.arr.shape[0:2], dtype=bool)

        for class_id in class_ids:
            # note that segmap has shape (H,W,C), so we max() along C
            mask_i = np.any(segmap.arr == class_id, axis=2)
            mask = np.logical_or(mask, mask_i)

        mask = mask.astype(np.float32)
        mask = ia.imresize_single_image(mask, segmap.shape[0:2])

        return mask


@legacy(version="0.4.0")
class BoundingBoxesMaskGen(IBatchwiseMaskGenerator):
    """Generator that produces masks highlighting bounding boxes.

    This class produces for each row (i.e. image + bounding boxes) in a batch
    a mask in which the inner areas of bounding box rectangles with given
    labels are marked (i.e. set to ``1.0``). The labels may be provided as a
    fixed list of strings or a stochastic parameter from which labels will be
    sampled. If no labels are provided, all bounding boxes will be marked.

    A pixel will be set to ``1.0`` if *at least* one bounding box at that
    location has one of the requested labels, even if there is *also* one
    bounding box at that location with a not requested label.

    .. note::

        This class will produce an ``AssertionError`` if there are no
        bounding boxes in a batch.


    Parameters
    ----------
    labels : None or str or list of str or imgaug2.parameters.StochasticParameter
        Labels of bounding boxes to select for.

        If `nb_sample_labels` is ``None`` then this is expected to be either
        also ``None`` (select all BBs) or a single ``str`` (select BBs with
        this one label) or a ``list`` of ``str`` s (always select BBs with
        these labels).

        If `nb_sample_labels` is set, then this parameter will be treated
        as a stochastic parameter with the following valid types:

            * If ``None``: Ignore the sampling count  and always use all
              bounding boxes.
            * If ``str``: Exactly that label will be used for all
              images.
            * If ``list`` of ``str``: ``N`` random values will be picked per
              image from that list and used as the labels.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(sum(N),)`` values.

        ``N`` denotes the number of labels to sample per segmentation
        map (derived from `nb_sample_labels`) and ``sum(N)`` denotes the
        sum of ``N`` s over all images.

    nb_sample_labels : None or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Number of labels to sample (with replacement) per image.
        As sampling happens with replacement, fewer *unique* labels may be
        sampled.

            * If ``None``: `labels` is expected to also be ``None`` or a fixed
              value of labels to be used for all images.
            * If ``int``: Exactly that many labels will be sampled for all
              images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly
              sampled per image from the discrete interval ``[a..b]``.
            * If ``list``: A random value will be picked per image from
              that list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(B,)`` values, where ``B`` is the number of
              images.

    """

    @legacy(version="0.4.0")
    def __init__(
        self, labels: LabelInput = None, nb_sample_labels: ParamInput | None = None
    ) -> None:
        if labels is None:
            self.labels = None
            self.nb_sample_labels = None
        elif nb_sample_labels is None:
            if ia.is_string(labels):
                labels = [labels]
            assert isinstance(labels, list), (
                "Expected `labels` a single string or a list of "
                f"strings if `nb_sample_labels` is None. Got type `{type(labels).__name__}`. "
                "Set `nb_sample_labels` to e.g. an integer to enable "
                "stochastic parameters for `labels`."
            )
            self.labels = labels
            self.nb_sample_labels = None
        else:
            self.labels = iap.handle_categorical_string_param(labels, "labels")
            self.nb_sample_labels = iap.handle_discrete_param(
                nb_sample_labels,
                "nb_sample_labels",
                value_range=(0, None),
                tuple_to_uniform=True,
                list_to_choice=True,
                allow_floats=False,
            )

    @legacy(version="0.4.0")
    def draw_masks(self, batch: _BatchInAugmentation, random_state: RNGInput = None) -> list[Array]:
        """
        See :func:`~imgaug2.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.


        """
        assert batch.bounding_boxes is not None, (
            "Can only generate masks for batches that contain bounding boxes, "
            "but got a batch without them."
        )
        random_state = iarandom.RNG.create_if_not_rng_(random_state)

        if self.labels is None:
            return [self.generate_mask(bbsoi, None) for bbsoi in batch.bounding_boxes]

        labels = self._draw_samples(batch.nb_rows, random_state=random_state)

        return [
            self.generate_mask(bbsoi, labels_i)
            for bbsoi, labels_i in zip(batch.bounding_boxes, labels, strict=True)
        ]

    @legacy(version="0.4.0")
    def _draw_samples(self, nb_rows: int, random_state: iarandom.RNG) -> list[Sequence[str]]:
        nb_sample_labels = self.nb_sample_labels
        if nb_sample_labels is None:
            assert isinstance(self.labels, list), f"Expected list got {type(self.labels).__name__}."
            return cast(list[Sequence[str]], [self.labels] * nb_rows)

        nb_sample_labels = nb_sample_labels.draw_samples((nb_rows,), random_state=random_state)
        nb_sample_labels = np.clip(nb_sample_labels, 0, None)
        labels_raw = self.labels.draw_samples(
            (np.sum(nb_sample_labels),), random_state=random_state
        )

        labels = _split_1d_array_to_list(labels_raw, nb_sample_labels)

        return cast(list[Sequence[str]], labels)

    # TODO this could be simplified to something like
    #      bbsoi.only_labels(labels).draw_mask()
    @legacy(version="0.4.0")
    @classmethod
    def generate_mask(cls, bbsoi: BoundingBoxesOnImage, labels: Sequence[str] | None) -> Array:
        """Generate a mask of the areas of bounding boxes with given labels.


        Parameters
        ----------
        bbsoi : imgaug2.augmentables.bbs.BoundingBoxesOnImage
            The bounding boxes for which to generate the mask.

        labels : None or iterable of str
            Labels of the bounding boxes to set to ``1.0``.
            For an ``(x, y)`` position, it is enough that *any* bounding box
            at the given location has one of the labels.
            If this is ``None``, all bounding boxes will be marked.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as
            ``segmap.shape``. Values are in ``[0.0, 1.0]``.

        """
        labels = set(labels) if labels is not None else None
        height, width = bbsoi.shape[0:2]
        mask = np.zeros((height, width), dtype=np.float32)

        for bb in bbsoi:
            if labels is None or bb.label in labels:
                x1 = min(max(int(bb.x1), 0), width)
                y1 = min(max(int(bb.y1), 0), height)
                x2 = min(max(int(bb.x2), 0), width)
                y2 = min(max(int(bb.y2), 0), height)
                if x1 < x2 and y1 < y2:
                    mask[y1:y2, x1:x2] = 1.0

        return mask


@legacy(version="0.4.0")
class InvertMaskGen(IBatchwiseMaskGenerator):
    """Generator that inverts the outputs of other mask generators.

    This class receives batches and calls for each row (i.e. image)
    a child mask generator to produce a mask. That mask is then inverted
    for ``p%`` of all rows, i.e. converted to ``1.0 - mask``.


    Parameters
    ----------
    p : bool or float or imgaug2.parameters.StochasticParameter, optional
        Probability of inverting each mask produced by the other mask
        generator.

    child : IBatchwiseMaskGenerator
        The other mask generator to invert.

    """

    @legacy(version="0.4.0")
    def __init__(self, p: PerChannelInput, child: IBatchwiseMaskGenerator) -> None:
        self.p = iap.handle_probability_param(p, "p")
        self.child = child

    @legacy(version="0.4.0")
    def draw_masks(self, batch: _BatchInAugmentation, random_state: RNGInput = None) -> list[Array]:
        """
        See :func:`~imgaug2.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.


        """
        random_state = iarandom.RNG.create_if_not_rng_(random_state)
        masks = self.child.draw_masks(batch, random_state=random_state)
        p = self.p.draw_samples(len(masks), random_state=random_state)
        for mask, p_i in zip(masks, p, strict=True):
            if p_i >= 0.5:
                mask[...] = 1.0 - mask
        return masks
