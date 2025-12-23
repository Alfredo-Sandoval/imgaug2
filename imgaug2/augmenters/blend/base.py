from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, TypeAlias, cast

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmentables.bbs import BoundingBoxesOnImage
from imgaug2.augmentables.kps import KeypointsOnImage
from imgaug2.augmentables.lines import LineStringsOnImage
from imgaug2.augmentables.polys import PolygonsOnImage
from imgaug2.augmenters import meta
import imgaug2.augmenters._blend_utils as blend_utils
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy

AlphaInput: TypeAlias = blend_utils.AlphaInput
_blend_alpha_uint8_single_alpha_ = blend_utils._blend_alpha_uint8_single_alpha_
_blend_alpha_uint8_channelwise_alphas_ = blend_utils._blend_alpha_uint8_channelwise_alphas_
_blend_alpha_uint8_elementwise_ = blend_utils._blend_alpha_uint8_elementwise_
_blend_alpha_non_uint8 = blend_utils._blend_alpha_non_uint8
_merge_channels = blend_utils._merge_channels
PerChannelInput: TypeAlias = bool | float | iap.StochasticParameter
ChildrenInput: TypeAlias = meta.Augmenter | Sequence[meta.Augmenter] | None
CoordinateAugmentable: TypeAlias = (
    KeypointsOnImage | BoundingBoxesOnImage | PolygonsOnImage | LineStringsOnImage
)
UpscaleMethodInput: TypeAlias = None | Literal["ALL"] | str | list[str] | iap.StochasticParameter
AggregationMethodInput: TypeAlias = Literal["ALL"] | str | list[str] | iap.StochasticParameter
SigmoidInput: TypeAlias = bool | float
LabelInput: TypeAlias = None | str | list[str] | iap.StochasticParameter

class _BranchAugmenter(Protocol):
    foreground: meta.Augmenter | None
    background: meta.Augmenter | None
    deterministic: bool
    random_state: iarandom.RNG

    def copy(self) -> _BranchAugmenter: ...

def blend_alpha(image_fg: Array, image_bg: Array, alpha: AlphaInput, eps: float = 1e-2) -> Array:
    """Blend two images using an alpha blending.

    See `blend_alpha()` for details.
    """
    from imgaug2.mlx._core import is_mlx_array

    if is_mlx_array(image_fg) or is_mlx_array(image_bg) or is_mlx_array(alpha):
        from imgaug2.mlx import blend as mlx_blend

        return cast(Array, mlx_blend.blend_alpha(image_fg, image_bg, alpha, eps=eps))

    return blend_utils.blend_alpha(image_fg, image_bg, alpha, eps=eps)

def blend_alpha_(image_fg: Array, image_bg: Array, alpha: AlphaInput, eps: float = 1e-2) -> Array:
    """Blend two images in-place using an alpha blending.

    See `blend_alpha_()` for details.
    """
    from imgaug2.mlx._core import is_mlx_array

    if is_mlx_array(image_fg) or is_mlx_array(image_bg) or is_mlx_array(alpha):
        from imgaug2.mlx import blend as mlx_blend

        return cast(Array, mlx_blend.blend_alpha(image_fg, image_bg, alpha, eps=eps))

    return blend_utils.blend_alpha_(image_fg, image_bg, alpha, eps=eps)

def _split_1d_array_to_list(arr: Array, sizes: Sequence[int]) -> list[Array]:
    result = []
    i = 0
    for size in sizes:
        result.append(arr[i : i + size])
        i += size
    return result

@legacy(version="0.4.0")
def _generate_branch_outputs(
    augmenter: _BranchAugmenter,
    batch: _BatchInAugmentation,
    hooks: ia.HooksImages | None,
    parents: list[meta.Augmenter],
) -> tuple[_BatchInAugmentation, _BatchInAugmentation]:
    parents_extended = parents + [augmenter]

    # Note here that the propagation hook removes columns in the batch
    # and re-adds them afterwards. So the batch should not be copied
    # after the `with` statement.
    outputs_fg = batch
    if augmenter.foreground is not None:
        outputs_fg = outputs_fg.deepcopy()
        with outputs_fg.propagation_hooks_ctx(augmenter, hooks, parents):
            if augmenter.foreground is not None:
                outputs_fg = augmenter.foreground.augment_batch_(
                    outputs_fg, parents=parents_extended, hooks=hooks
                )

    outputs_bg = batch
    if augmenter.background is not None:
        outputs_bg = outputs_bg.deepcopy()
        with outputs_bg.propagation_hooks_ctx(augmenter, hooks, parents):
            outputs_bg = augmenter.background.augment_batch_(
                outputs_bg, parents=parents_extended, hooks=hooks
            )

    return cast(_BatchInAugmentation, outputs_fg), cast(_BatchInAugmentation, outputs_bg)

@legacy(version="0.4.0")
def _to_deterministic(augmenter: _BranchAugmenter) -> _BranchAugmenter:
    aug = augmenter.copy()
    aug.foreground = aug.foreground.to_deterministic() if aug.foreground is not None else None
    aug.background = aug.background.to_deterministic() if aug.background is not None else None
    aug.deterministic = True
    aug.random_state = augmenter.random_state.derive_rng_()
    return aug

@legacy(version="0.4.0")
class BlendAlpha(meta.Augmenter):
    """
    Alpha-blend two image sources using an alpha/opacity value.

    The two image sources can be imagined as branches.
    If a source is not given, it is automatically the same as the input.
    Let ``FG`` be the foreground branch and ``BG`` be the background branch.
    Then the result images are defined as ``factor * FG + (1-factor) * BG``,
    where ``factor`` is an overlay factor.

    .. note::

        It is not recommended to use ``BlendAlpha`` with augmenters
        that change the geometry of images (e.g. horizontal flips, affine
        transformations) if you *also* want to augment coordinates (e.g.
        keypoints, polygons, ...), as it is unclear which of the two
        coordinate results (foreground or background branch) should be used
        as the coordinates after augmentation.

        Currently, if ``factor >= 0.5`` (per image), the results of the
        foreground branch are used as the new coordinates, otherwise the
        results of the background branch.

    Before that named `Alpha`.

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

    per_channel : bool or float or imgaug2.parameters.StochasticParameter, optional
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
    >>> aug = iaa.BlendAlpha(0.5, iaa.Grayscale(1.0))

    Convert each image to pure grayscale and alpha-blend the result with the
    original image using an alpha of ``50%``, thereby removing about ``50%`` of
    all color. This is equivalent to ``iaa.Grayscale(0.5)``.

    >>> aug = iaa.BlendAlpha((0.0, 1.0), iaa.Grayscale(1.0))

    Same as in the previous example, but the alpha factor is sampled uniformly
    from the interval ``[0.0, 1.0]`` once per image, thereby removing a random
    fraction of all colors. This is equivalent to
    ``iaa.Grayscale((0.0, 1.0))``.

    >>> aug = iaa.BlendAlpha(
    >>>     (0.0, 1.0),
    >>>     iaa.Affine(rotate=(-20, 20)),
    >>>     per_channel=0.5)

    First, rotate each image by a random degree sampled uniformly from the
    interval ``[-20, 20]``. Then, alpha-blend that new image with the original
    one using a random factor sampled uniformly from the interval
    ``[0.0, 1.0]``. For ``50%`` of all images, the blending happens
    channel-wise and the factor is sampled independently per channel
    (``per_channel=0.5``). As a result, e.g. the red channel may look visibly
    rotated (factor near ``1.0``), while the green and blue channels may not
    look rotated (factors near ``0.0``).

    >>> aug = iaa.BlendAlpha(
    >>>     (0.0, 1.0),
    >>>     foreground=iaa.Add(100),
    >>>     background=iaa.Multiply(0.2))

    Apply two branches of augmenters -- ``A`` and ``B`` -- *independently*
    to input images and alpha-blend the results of these branches using a
    factor ``f``. Branch ``A`` increases image pixel intensities by ``100``
    and ``B`` multiplies the pixel intensities by ``0.2``. ``f`` is sampled
    uniformly from the interval ``[0.0, 1.0]`` per image. The resulting images
    contain a bit of ``A`` and a bit of ``B``.

    >>> aug = iaa.BlendAlpha([0.25, 0.75], iaa.MedianBlur(13))

    Apply median blur to each image and alpha-blend the result with the
    original image using an alpha factor of either exactly ``0.25`` or
    exactly ``0.75`` (sampled once per image).

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
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.factor = iap.handle_continuous_param(
            factor, "factor", value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True
        )

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

        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

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

        columns = batch.columns
        shapes = batch.get_rowwise_shapes()
        nb_images = len(shapes)
        nb_channels_max = max([shape[2] if len(shape) > 2 else 1 for shape in shapes])
        rngs = random_state.duplicate(2)
        per_channel = self.per_channel.draw_samples(nb_images, random_state=rngs[0])
        alphas = self.factor.draw_samples((nb_images, nb_channels_max), random_state=rngs[1])

        use_mlx_batch = False
        alpha_batch = None
        if batch.images is not None and batch_fg.images is not None and batch_bg.images is not None:
            from imgaug2.mlx._core import is_mlx_array

            if (
                is_mlx_array(batch.images)
                and is_mlx_array(batch_fg.images)
                and is_mlx_array(batch_bg.images)
            ):
                use_mlx_batch = True
                alpha_batch = np.zeros((nb_images, 1, 1, nb_channels_max), dtype=np.float32)

        for i, shape in enumerate(shapes):
            nb_channels = shape[2] if len(shape) > 2 else 1
            if per_channel[i] > 0.5:
                alphas_i = alphas[i, 0:nb_channels]
            else:
                # We catch here the case of alphas[i] being empty, which can
                # happen if all images have 0 channels.
                # In that case the alpha value doesn't matter as the image
                # contains zero values anyways.
                alphas_i = alphas[i, 0] if alphas[i].size > 0 else 0

            # compute alpha for non-image data -- average() also works with
            # scalars
            alphas_i_avg = np.average(alphas_i)
            use_fg_branch = alphas_i_avg >= 0.5

            # blend images
            if use_mlx_batch:
                if alpha_batch is None:
                    raise AssertionError("Internal error: alpha_batch expected for MLX batch path.")
                if np.ndim(alphas_i) == 0:
                    alpha_batch[i, 0, 0, :nb_channels] = float(alphas_i)
                else:
                    alpha_batch[i, 0, 0, :nb_channels] = alphas_i
            elif batch.images is not None:
                assert batch_fg.images is not None
                assert batch_bg.images is not None
                batch.images[i] = blend_alpha_(
                    batch_fg.images[i], batch_bg.images[i], alphas_i, eps=self.epsilon
                )

            # blend non-images
            # TODO Use gradual blending for heatmaps here (as for images)?
            #      Heatmaps are probably the only augmentable where this makes
            #      sense.
            for column in columns:
                if column.name != "images":
                    batch_use = batch_fg if use_fg_branch else batch_bg
                    column.value[i] = getattr(batch_use, column.attr_name)[i]

        if use_mlx_batch and batch.images is not None:
            if alpha_batch is None:
                raise AssertionError("Internal error: alpha_batch expected for MLX batch path.")
            from imgaug2.mlx import blend as mlx_blend

            batch.images = mlx_blend.blend_alpha(
                batch_fg.images,
                batch_bg.images,
                alpha_batch,
                eps=self.epsilon,
            )

        return batch

    @legacy(version="0.4.0")
    def _to_deterministic(self) -> meta.Augmenter:
        return cast(meta.Augmenter, _to_deterministic(self))

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See `get_parameters()`."""
        return [self.factor, self.per_channel]

    @legacy(version="0.4.0")
    def get_children_lists(self) -> list[list[meta.Augmenter]]:
        """See `get_children_lists()`."""
        return cast(
            list[list[meta.Augmenter]],
            [lst for lst in [self.foreground, self.background] if lst is not None],
        )

    @legacy(version="0.4.0")
    def __str__(self) -> str:
        pattern = (
            "%s(factor=%s, per_channel=%s, name=%s, foreground=%s, background=%s, deterministic=%s)"
        )
        return pattern % (
            self.__class__.__name__,
            self.factor,
            self.per_channel,
            self.name,
            self.foreground,
            self.background,
            self.deterministic,
        )
