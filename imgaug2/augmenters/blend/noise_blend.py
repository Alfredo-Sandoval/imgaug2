from __future__ import annotations

from typing import Literal

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
from imgaug2.compat.markers import legacy
from imgaug2.augmenters._typing import ParamInput, RNGInput

from .base import (
    AggregationMethodInput,
    ChildrenInput,
    PerChannelInput,
    SigmoidInput,
    UpscaleMethodInput,
)
from .masks import BlendAlphaElementwise

@legacy(version="0.4.0")
class BlendAlphaSimplexNoise(BlendAlphaElementwise):
    """Alpha-blend two image sources using simplex noise alpha masks.

    The alpha masks are sampled using a simplex noise method, roughly creating
    connected blobs of 1s surrounded by 0s. If nearest neighbour
    upsampling is used, these blobs can be rectangular with sharp edges.

    Before that named `SimplexNoiseAlpha`.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.blend.BlendAlphaElementwise`.

    Parameters
    ----------
    foreground : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (``False``)
        or to sample a new value for each channel (``True``).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``, otherwise as ``False``.

    size_px_max : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        The simplex noise is always generated in a low resolution environment.
        This parameter defines the maximum size of that environment (in
        pixels). The environment is initialized at the same size as the input
        image and then downscaled, so that no side exceeds `size_px_max`
        (aspect ratio is kept).

            * If int, then that number will be used as the size for all
              iterations.
            * If tuple of two ``int`` s ``(a, b)``, then a value will be
              sampled per iteration from the discrete interval ``[a..b]``.
            * If a list of ``int`` s, then a value will be picked per iteration
              at random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per iteration.

    upscale_method : None or imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the input image size. This parameter controls
        the upscaling method.

            * If ``None``, then either ``nearest`` or ``linear`` or ``cubic``
              is picked. Most weight is put on ``linear``, followed by
              ``cubic``.
            * If ``imgaug2.ALL``, then either ``nearest`` or ``linear`` or
              ``area`` or ``cubic`` is picked per iteration (all same
              probability).
            * If a string, then that value will be used as the method (must be
              ``nearest`` or ``linear`` or ``area`` or ``cubic``).
            * If list of string, then a random value will be picked from that
              list per iteration.
            * If ``StochasticParameter``, then a random value will be sampled
              from that parameter per iteration.

    iterations : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        How often to repeat the simplex noise generation process per image.

            * If ``int``, then that number will be used as the iterations for
              all images.
            * If tuple of two ``int`` s ``(a, b)``, then a value will be
              sampled per image from the discrete interval ``[a..b]``.
            * If a list of ``int`` s, then a value will be picked per image at
              random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per image.

    aggregation_method : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        The noise maps (from each iteration) are combined to one noise map
        using an aggregation process. This parameter defines the method used
        for that process. Valid methods are ``min``, ``max`` or ``avg``,
        where ``min`` combines the noise maps by taking the (elementwise)
        minimum over all iteration's results, ``max`` the (elementwise)
        maximum and ``avg`` the (elementwise) average.

            * If ``imgaug2.ALL``, then a random value will be picked per image
              from the valid ones.
            * If a string, then that value will always be used as the method.
            * If a list of string, then a random value will be picked from
              that list per image.
            * If a ``StochasticParameter``, then a random value will be
              sampled from that paramter per image.

    sigmoid : bool or number, optional
        Whether to apply a sigmoid function to the final noise maps, resulting
        in maps that have more extreme values (close to 0.0 or 1.0).

            * If ``bool``, then a sigmoid will always (``True``) or never
              (``False``) be applied.
            * If a number ``p`` with ``0<=p<=1``, then a sigmoid will be
              applied to ``p`` percent of all final noise maps.

    sigmoid_thresh : None or number or tuple of number or imgaug2.parameters.StochasticParameter, optional
        Threshold of the sigmoid, when applied. Thresholds above zero
        (e.g. ``5.0``) will move the saddle point towards the right, leading
        to more values close to 0.0.

            * If ``None``, then ``Normal(0, 5.0)`` will be used.
            * If number, then that threshold will be used for all images.
            * If tuple of two numbers ``(a, b)``, then a random value will
              be sampled per image from the interval ``[a, b]``.
            * If ``StochasticParameter``, then a random value will be sampled
              from that parameter per image.

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
    >>> aug = iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1.0))

    Detect per image all edges, mark them in a black and white image and
    then alpha-blend the result with the original image using simplex noise
    masks.

    >>> aug = iaa.BlendAlphaSimplexNoise(
    >>>     iaa.EdgeDetect(1.0),
    >>>     upscale_method="nearest")

    Same as in the previous example, but using only nearest neighbour
    upscaling to scale the simplex noise masks to the final image sizes, i.e.
    no nearest linear upsampling is used. This leads to rectangles with sharp
    edges.

    >>> aug = iaa.BlendAlphaSimplexNoise(
    >>>     iaa.EdgeDetect(1.0),
    >>>     upscale_method="linear")

    Same as in the previous example, but using only linear upscaling to
    scale the simplex noise masks to the final image sizes, i.e. no nearest
    neighbour upsampling is used. This leads to rectangles with smooth edges.

    >>> aug = iaa.BlendAlphaSimplexNoise(
    >>>     iaa.EdgeDetect(1.0),
    >>>     sigmoid_thresh=iap.Normal(10.0, 5.0))

    Same as in the first example, but using a threshold for the sigmoid
    function that is further to the right. This is more conservative, i.e.
    the generated noise masks will be mostly black (values around ``0.0``),
    which means that most of the original images (parameter/branch
    `background`) will be kept, rather than using the results of the
    augmentation (parameter/branch `foreground`).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        foreground: ChildrenInput = None,
        background: ChildrenInput = None,
        per_channel: PerChannelInput = False,
        size_px_max: ParamInput = (2, 16),
        upscale_method: UpscaleMethodInput = None,
        iterations: ParamInput = (1, 3),
        aggregation_method: AggregationMethodInput = "max",
        sigmoid: SigmoidInput = True,
        sigmoid_thresh: ParamInput | None = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        upscale_method_default = iap.Choice(["nearest", "linear", "cubic"], p=[0.05, 0.6, 0.35])
        sigmoid_thresh_default = iap.Normal(0.0, 5.0)

        noise = iap.SimplexNoise(
            size_px_max=size_px_max,
            upscale_method=(
                upscale_method if upscale_method is not None else upscale_method_default
            ),
        )

        if iterations != 1:
            noise = iap.IterativeNoiseAggregator(
                noise, iterations=iterations, aggregation_method=aggregation_method
            )

        use_sigmoid = sigmoid is True or (ia.is_single_number(sigmoid) and sigmoid >= 0.01)
        if use_sigmoid:
            noise = iap.Sigmoid.create_for_noise(
                noise,
                threshold=(
                    sigmoid_thresh if sigmoid_thresh is not None else sigmoid_thresh_default
                ),
                activated=sigmoid,
            )

        super().__init__(
            factor=noise,
            foreground=foreground,
            background=background,
            per_channel=per_channel,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
class BlendAlphaFrequencyNoise(BlendAlphaElementwise):
    """Alpha-blend two image sources using frequency noise masks.

    The alpha masks are sampled using frequency noise of varying scales,
    which can sometimes create large connected blobs of ``1`` s surrounded
    by ``0`` s and other times results in smaller patterns. If nearest
    neighbour upsampling is used, these blobs can be rectangular with sharp
    edges.

    Before that named `FrequencyNoiseAlpha`.

    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.blend.BlendAlphaElementwise`.

    Parameters
    ----------
    exponent : number or tuple of number of list of number or imgaug2.parameters.StochasticParameter, optional
        Exponent to use when scaling in the frequency domain.
        Sane values are in the range ``-4`` (large blobs) to ``4`` (small
        patterns). To generate cloud-like structures, use roughly ``-2``.

            * If number, then that number will be used as the exponent for all
              iterations.
            * If tuple of two numbers ``(a, b)``, then a value will be sampled
              per iteration from the interval ``[a, b]``.
            * If a list of numbers, then a value will be picked per iteration
              at random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per iteration.

    foreground : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (``False``)
        or to sample a new value for each channel (``True``).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``, otherwise as ``False``.

    size_px_max : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        The noise is generated in a low resolution environment.
        This parameter defines the maximum size of that environment (in
        pixels). The environment is initialized at the same size as the input
        image and then downscaled, so that no side exceeds `size_px_max`
        (aspect ratio is kept).

            * If ``int``, then that number will be used as the size for all
              iterations.
            * If tuple of two ``int`` s ``(a, b)``, then a value will be
              sampled per iteration from the discrete interval ``[a..b]``.
            * If a list of ``int`` s, then a value will be picked per
              iteration at random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per iteration.

    upscale_method : None or imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the input image size. This parameter controls
        the upscaling method.

            * If ``None``, then either ``nearest`` or ``linear`` or ``cubic``
              is picked. Most weight is put on ``linear``, followed by
              ``cubic``.
            * If ``imgaug2.ALL``, then either ``nearest`` or ``linear`` or
              ``area`` or ``cubic`` is picked per iteration (all same
              probability).
            * If string, then that value will be used as the method (must be
              ``nearest`` or ``linear`` or ``area`` or ``cubic``).
            * If list of string, then a random value will be picked from that
              list per iteration.
            * If ``StochasticParameter``, then a random value will be sampled
              from that parameter per iteration.

    iterations : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        How often to repeat the simplex noise generation process per
        image.

            * If ``int``, then that number will be used as the iterations for
              all images.
            * If tuple of two ``int`` s ``(a, b)``, then a value will be
              sampled per image from the discrete interval ``[a..b]``.
            * If a list of ``int`` s, then a value will be picked per image at
              random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per image.

    aggregation_method : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        The noise maps (from each iteration) are combined to one noise map
        using an aggregation process. This parameter defines the method used
        for that process. Valid methods are ``min``, ``max`` or ``avg``,
        where 'min' combines the noise maps by taking the (elementwise) minimum
        over all iteration's results, ``max`` the (elementwise) maximum and
        ``avg`` the (elementwise) average.

            * If ``imgaug2.ALL``, then a random value will be picked per image
              from the valid ones.
            * If a string, then that value will always be used as the method.
            * If a list of string, then a random value will be picked from
              that list per image.
            * If a ``StochasticParameter``, then a random value will be sampled
              from that parameter per image.

    sigmoid : bool or number, optional
        Whether to apply a sigmoid function to the final noise maps, resulting
        in maps that have more extreme values (close to ``0.0`` or ``1.0``).

            * If ``bool``, then a sigmoid will always (``True``) or never
              (``False``) be applied.
            * If a number ``p`` with ``0<=p<=1``, then a sigmoid will be applied to
              ``p`` percent of all final noise maps.

    sigmoid_thresh : None or number or tuple of number or imgaug2.parameters.StochasticParameter, optional
        Threshold of the sigmoid, when applied. Thresholds above zero
        (e.g. ``5.0``) will move the saddle point towards the right, leading to
        more values close to ``0.0``.

            * If ``None``, then ``Normal(0, 5.0)`` will be used.
            * If number, then that threshold will be used for all images.
            * If tuple of two numbers ``(a, b)``, then a random value will
              be sampled per image from the range ``[a, b]``.
            * If ``StochasticParameter``, then a random value will be sampled
              from that parameter per image.

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
    >>> aug = iaa.BlendAlphaFrequencyNoise(foreground=iaa.EdgeDetect(1.0))

    Detect per image all edges, mark them in a black and white image and
    then alpha-blend the result with the original image using frequency noise
    masks.

    >>> aug = iaa.BlendAlphaFrequencyNoise(
    >>>     foreground=iaa.EdgeDetect(1.0),
    >>>     upscale_method="nearest")

    Same as the first example, but using only linear upscaling to
    scale the frequency noise masks to the final image sizes, i.e. no nearest
    neighbour upsampling is used. This results in smooth edges.

    >>> aug = iaa.BlendAlphaFrequencyNoise(
    >>>     foreground=iaa.EdgeDetect(1.0),
    >>>     upscale_method="linear")

    Same as the first example, but using only linear upscaling to
    scale the frequency noise masks to the final image sizes, i.e. no nearest
    neighbour upsampling is used. This results in smooth edges.

    >>> aug = iaa.BlendAlphaFrequencyNoise(
    >>>     foreground=iaa.EdgeDetect(1.0),
    >>>     upscale_method="linear",
    >>>     exponent=-2,
    >>>     sigmoid=False)

    Same as in the previous example, but with the exponent set to a constant
    ``-2`` and the sigmoid deactivated, resulting in cloud-like patterns
    without sharp edges.

    >>> aug = iaa.BlendAlphaFrequencyNoise(
    >>>     foreground=iaa.EdgeDetect(1.0),
    >>>     sigmoid_thresh=iap.Normal(10.0, 5.0))

    Same as the first example, but using a threshold for the sigmoid function
    that is further to the right. This is more conservative, i.e. the generated
    noise masks will be mostly black (values around ``0.0``), which means that
    most of the original images (parameter/branch `background`) will be kept,
    rather than using the results of the augmentation (parameter/branch
    `foreground`).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        exponent: ParamInput = (-4, 4),
        foreground: ChildrenInput = None,
        background: ChildrenInput = None,
        per_channel: PerChannelInput = False,
        size_px_max: ParamInput = (4, 16),
        upscale_method: UpscaleMethodInput = None,
        iterations: ParamInput = (1, 3),
        aggregation_method: AggregationMethodInput | None = None,
        sigmoid: SigmoidInput = 0.5,
        sigmoid_thresh: ParamInput | None = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        aggregation_method = (
            aggregation_method if aggregation_method is not None else ["avg", "max"]
        )
        upscale_method_default = iap.Choice(["nearest", "linear", "cubic"], p=[0.05, 0.6, 0.35])
        sigmoid_thresh_default = iap.Normal(0.0, 5.0)

        noise = iap.FrequencyNoise(
            exponent=exponent,
            size_px_max=size_px_max,
            upscale_method=(
                upscale_method if upscale_method is not None else upscale_method_default
            ),
        )

        if iterations != 1:
            noise = iap.IterativeNoiseAggregator(
                noise, iterations=iterations, aggregation_method=aggregation_method
            )

        use_sigmoid = sigmoid is True or (ia.is_single_number(sigmoid) and sigmoid >= 0.01)
        if use_sigmoid:
            noise = iap.Sigmoid.create_for_noise(
                noise,
                threshold=(
                    sigmoid_thresh if sigmoid_thresh is not None else sigmoid_thresh_default
                ),
                activated=sigmoid,
            )

        super().__init__(
            factor=noise,
            foreground=foreground,
            background=background,
            per_channel=per_channel,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


