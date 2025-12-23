from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from opensimplex import OpenSimplex

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.compat.markers import legacy

from .base import ParamInput, StochasticParameter
from .continuous import Uniform
from .discrete import Choice, Deterministic, DiscreteUniform
from .handles import _assert_arg_is_stoch_param, handle_continuous_param, handle_discrete_param

class FromLowerResolution(StochasticParameter):
    """Parameter to sample from other parameters at lower image resolutions.

    This parameter is intended to be used with parameters that would usually
    sample one value per pixel (or one value per pixel and channel). Instead
    of sampling from the other parameter at full resolution, it samples at
    lower resolution, e.g. ``0.5*H x 0.5*W`` with ``H`` being the height and
    ``W`` being the width. After the low-resolution sampling this parameter
    then upscales the result to ``HxW``.

    This parameter is intended to produce coarse samples. E.g. combining
    this with `Binomial` can lead to large rectangular areas of
    ``1`` s and ``0`` s.

    Parameters
    ----------
    other_param : imgaug2.parameters.StochasticParameter
        The other parameter which is to be sampled on a coarser image.

    size_percent : None or number or iterable of number or imgaug2.parameters.StochasticParameter, optional
        Size of the 2d sampling plane in percent of the requested size.
        I.e. this is relative to the size provided in the call to
        ``draw_samples(size)``. Lower values will result in smaller sampling
        planes, which are then upsampled to `size`. This means that lower
        values will result in larger rectangles. The size may be provided as
        a constant value or a tuple ``(a, b)``, which will automatically be
        converted to the continuous uniform range ``[a, b)`` or a
        `StochasticParameter`, which will be queried per call to
        `draw_sample()` and
        `draw_samples()`.

    size_px : None or number or iterable of numbers or imgaug2.parameters.StochasticParameter, optional
        Size of the 2d sampling plane in pixels.
        Lower values will result in smaller sampling planes, which are then
        upsampled to the input `size` of ``draw_samples(size)``.
        This means that lower values will result in larger rectangles.
        The size may be provided as a constant value or a tuple ``(a, b)``,
        which will automatically be converted to the discrete uniform
        range ``[a..b]`` or a `StochasticParameter`, which will be
        queried once per call to `draw_sample()` and
        `draw_samples()`.

    method : str or int or imgaug2.parameters.StochasticParameter, optional
        Upsampling/interpolation method to use. This is used after the sampling
        is finished and the low resolution plane has to be upsampled to the
        requested `size` in ``draw_samples(size, ...)``. The method may be
        the same as in `imresize_many_images()`. Usually
        ``nearest`` or ``linear`` are good choices. ``nearest`` will result
        in rectangles with sharp edges and ``linear`` in rectangles with
        blurry and round edges. The method may be provided as a
        `StochasticParameter`, which will be queried once per call to
        `draw_sample()` and
        `draw_samples()`.

    min_size : int, optional
        Minimum size in pixels of the low resolution sampling plane.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.FromLowerResolution(
    >>>     Binomial(0.05),
    >>>     size_px=(2, 16),
    >>>     method=Choice(["nearest", "linear"]))

    Samples from a binomial distribution with ``p=0.05``. The sampling plane
    will always have a size HxWxC with H and W being independently sampled
    from ``[2..16]`` (i.e. it may range from ``2x2xC`` up to ``16x16xC`` max,
    but may also be e.g. ``4x8xC``). The upsampling method will be ``nearest``
    in ``50%`` of all cases and ``linear`` in the other 50 percent. The result
    will sometimes be rectangular patches of sharp ``1`` s surrounded by
    ``0`` s and sometimes blurry blobs of ``1``s, surrounded by values
    ``<1.0``.

    """

    def __init__(
        self,
        other_param: StochasticParameter,
        size_percent: ParamInput | None = None,
        size_px: ParamInput | None = None,
        method: str | int | StochasticParameter = "nearest",
        min_size: int = 1,
    ) -> None:
        super().__init__()

        assert size_percent is not None or size_px is not None, (
            "Expected either 'size_percent' or 'size_px' to be provided, got neither of them."
        )

        if size_percent is not None:
            self.size_method = "percent"
            self.size_px = None
            if ia.is_single_number(size_percent):
                self.size_percent = Deterministic(size_percent)
            elif ia.is_iterable(size_percent):
                assert len(size_percent) == 2, (
                    f"Expected iterable 'size_percent' to contain exactly 2 "
                    f"values, got {len(size_percent)}."
                )
                self.size_percent = Uniform(size_percent[0], size_percent[1])
            elif isinstance(size_percent, StochasticParameter):
                self.size_percent = size_percent
            else:
                raise Exception(
                    "Expected int, float, tuple of two ints/floats or "
                    "StochasticParameter for size_percent, "
                    f"got {type(size_percent)}."
                )
        else:  # = elif size_px is not None:
            self.size_method = "px"
            self.size_percent = None
            if ia.is_single_integer(size_px):
                self.size_px = Deterministic(size_px)
            elif ia.is_iterable(size_px):
                assert len(size_px) == 2, (
                    f"Expected iterable 'size_px' to contain exactly 2 values, got {len(size_px)}."
                )
                self.size_px = DiscreteUniform(size_px[0], size_px[1])
            elif isinstance(size_px, StochasticParameter):
                self.size_px = size_px
            else:
                raise Exception(
                    "Expected int, float, tuple of two ints/floats or "
                    "StochasticParameter for size_px, "
                    f"got {type(size_px)}."
                )

        self.other_param = other_param

        if ia.is_string(method) or ia.is_single_integer(method):
            self.method = Deterministic(method)
        elif isinstance(method, StochasticParameter):
            self.method = method
        else:
            raise Exception(f"Expected string or StochasticParameter, got {type(method)}.")

        self.min_size = min_size

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        if len(size) == 3:
            n = 1
            h, w, c = size
        elif len(size) == 4:
            n, h, w, c = size
        else:
            raise Exception(
                "FromLowerResolution can only generate samples "
                "of shape (H, W, C) or (N, H, W, C), "
                f"requested was {str(size)}."
            )

        if self.size_method == "percent":
            hw_percents = self.size_percent.draw_samples((n, 2), random_state=random_state)
            hw_pxs = (hw_percents * np.array([h, w])).astype(np.int32)
        else:
            hw_pxs = self.size_px.draw_samples((n, 2), random_state=random_state)

        methods = self.method.draw_samples((n,), random_state=random_state)
        result = None
        for i, (hw_px, method) in enumerate(zip(hw_pxs, methods, strict=False)):
            h_small = max(hw_px[0], self.min_size)
            w_small = max(hw_px[1], self.min_size)
            samples = self.other_param.draw_samples(
                (1, h_small, w_small, c), random_state=random_state
            )

            # This (1) makes sure that samples are of dtypes supported by
            # imresize_many_images, and (2) forces samples to be float-kind
            # if the requested interpolation is something else than nearest
            # neighbour interpolation. (2) is a bit hacky and makes sure that
            # continuous values are produced for e.g. cubic interpolation.
            # This is particularly important for e.g. binomial distributios
            # used in FromLowerResolution and thereby in e.g. CoarseDropout,
            # where integer-kinds would lead to sharp edges despite using
            # cubic interpolation.
            if samples.dtype.kind == "f":
                samples = iadt.restore_dtypes_(samples, np.float32)
            elif samples.dtype.kind == "i":
                if method == "nearest":
                    samples = iadt.restore_dtypes_(samples, np.int32)
                else:
                    samples = iadt.restore_dtypes_(samples, np.float32)
            else:
                assert samples.dtype.kind == "u", (
                    "FromLowerResolution can only process outputs of kind "
                    f"f (float), i (int) or u (uint), got {samples.dtype.kind}."
                )
                if method == "nearest":
                    samples = iadt.restore_dtypes_(samples, np.uint16)
                else:
                    samples = iadt.restore_dtypes_(samples, np.float32)

            samples_upscaled = ia.imresize_many_images(samples, (h, w), interpolation=method)

            if result is None:
                result = np.zeros((n, h, w, c), dtype=samples_upscaled.dtype)
            result[i] = samples_upscaled

        if len(size) == 3:
            return result[0]
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if self.size_method == "percent":
            pattern = "FromLowerResolution(size_percent=%s, method=%s, other_param=%s)"
            return pattern % (self.size_percent, self.method, self.other_param)

        pattern = "FromLowerResolution(size_px=%s, method=%s, other_param=%s)"
        return pattern % (self.size_px, self.method, self.other_param)

@legacy

class IterativeNoiseAggregator(StochasticParameter):
    """Aggregate multiple iterations of samples from another parameter.

    This is supposed to be used in conjunction with `SimplexNoise` or
    `FrequencyNoise`. If a shape ``S`` is requested, it will request
    ``I`` times ``S`` samples from the underlying parameter, where ``I`` is
    the number of iterations. The ``I`` arrays will be combined to a single
    array of shape ``S`` using an aggregation method, e.g. simple averaging.

    Parameters
    ----------
    other_param : StochasticParameter
        The other parameter from which to sample one or more times.

    iterations : int or iterable of int or list of int or imgaug2.parameters.StochasticParameter, optional
        The number of iterations.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of
        `draw_sample()` or
        `draw_samples()`.

    aggregation_method : imgaug2.ALL or {'min', 'avg', 'max'} or list of str or imgaug2.parameters.StochasticParameter, optional
        The method to use to aggregate the samples of multiple iterations
        to a single output array. All methods combine several arrays of
        shape ``S`` each to a single array of shape ``S`` and hence work
        elementwise. Known methods are ``min`` (take the minimum over all
        iterations), ``max`` (take the maximum) and ``avg`` (take the average).

            * If `StochasticParameter`, a value will be sampled from
              that parameter once per call and must be one of the described
              methods..

        "per call" denotes a call of
        `draw_sample()` or
        `draw_samples()`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> noise = iap.IterativeNoiseAggregator(
    >>>     iap.SimplexNoise(),
    >>>     iterations=(2, 5),
    >>>     aggregation_method="max")

    Create a parameter that -- upon each call -- generates ``2`` to ``5``
    arrays of simplex noise with the same shape. Then it combines these
    noise maps to a single map using elementwise maximum.

    """

    def __init__(
        self,
        other_param: StochasticParameter,
        iterations: ParamInput = (1, 3),
        aggregation_method: ParamInput | str | list[str] | None = None,
    ) -> None:
        super().__init__()
        if aggregation_method is None:
            aggregation_method = ["max", "avg"]

        _assert_arg_is_stoch_param("other_param", other_param)
        self.other_param = other_param

        def _assert_within_bounds(_iterations: Iterable[int]) -> None:
            assert all([1 <= val <= 10000 for val in _iterations]), (
                "Expected 'iterations' to only contain values within "
                "the interval [1, 1000], got values {}.".format(
                    ", ".join([str(val) for val in _iterations]),
                )
            )

        if ia.is_single_integer(iterations):
            _assert_within_bounds([iterations])
            self.iterations = Deterministic(iterations)
        elif isinstance(iterations, list):
            assert len(iterations) > 0, (
                f"Expected 'iterations' of type list to contain at least one "
                f"entry, got {len(iterations)}."
            )
            _assert_within_bounds(iterations)
            self.iterations = Choice(iterations)
        elif ia.is_iterable(iterations):
            assert len(iterations) == 2, (
                f"Expected iterable non-list 'iteratons' to contain exactly "
                f"two entries, got {len(iterations)}."
            )
            assert all([ia.is_single_integer(val) for val in iterations]), (
                "Expected iterable non-list 'iterations' to only contain "
                "integers, got types {}.".format(
                    ", ".join([str(type(val)) for val in iterations]),
                )
            )
            _assert_within_bounds(iterations)
            self.iterations = DiscreteUniform(iterations[0], iterations[1])
        elif isinstance(iterations, StochasticParameter):
            self.iterations = iterations
        else:
            raise Exception(
                "Expected iterations to be int or tuple of two ints or "
                f"StochasticParameter, got {type(iterations)}."
            )

        if aggregation_method == ia.ALL:
            self.aggregation_method = Choice(["min", "max", "avg"])
        elif ia.is_string(aggregation_method):
            self.aggregation_method = Deterministic(aggregation_method)
        elif isinstance(aggregation_method, list):
            assert len(aggregation_method) >= 1, (
                f"Expected at least one aggregation method got {len(aggregation_method)}."
            )
            assert all([ia.is_string(val) for val in aggregation_method]), (
                "Expected aggregation methods provided as strings, got types {}.".format(
                    ", ".join([str(type(v)) for v in aggregation_method])
                )
            )
            self.aggregation_method = Choice(aggregation_method)
        elif isinstance(aggregation_method, StochasticParameter):
            self.aggregation_method = aggregation_method
        else:
            raise Exception(
                "Expected aggregation_method to be string or list of strings "
                f"or StochasticParameter, got {type(aggregation_method)}."
            )

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        rngs = random_state.duplicate(2)
        aggregation_method = self.aggregation_method.draw_sample(random_state=rngs[0])
        iterations = self.iterations.draw_sample(random_state=rngs[1])
        assert iterations > 0, (
            f"Expected to sample at least one iteration of aggregation. Got {iterations}."
        )

        rngs_iterations = rngs[1].duplicate(iterations)

        result = np.zeros(size, dtype=np.float32)
        for i in range(iterations):
            noise_iter = self.other_param.draw_samples(size, random_state=rngs_iterations[i])

            if aggregation_method == "avg":
                result += noise_iter
            elif aggregation_method == "min":
                if i == 0:
                    result = noise_iter
                else:
                    result = np.minimum(result, noise_iter)
            else:  # self.aggregation_method == "max"
                if i == 0:
                    result = noise_iter
                else:
                    result = np.maximum(result, noise_iter)

        if aggregation_method == "avg":
            result = result / iterations

        return result

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        opstr = str(self.other_param)
        return f"IterativeNoiseAggregator({opstr}, {str(self.iterations)}, {str(self.aggregation_method)})"

@legacy

class _NoiseParameterMixin:
    """Mixin providing shared _draw_samples logic for noise parameters.

    Subclasses must implement _draw_samples_hw(height, width, random_state).
    """

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        assert len(size) in [2, 3], (
            f"Expected requested noise to have shape (H, W) or (H, W, C), got shape {size}."
        )
        height, width = size[0:2]
        nb_channels = 1 if len(size) == 2 else size[2]

        channels = [
            self._draw_samples_hw(height, width, random_state) for _ in np.arange(nb_channels)
        ]

        if len(size) == 2:
            return channels[0]
        return np.stack(channels, axis=-1)

    def _draw_samples_hw(self, height: int, width: int, random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        raise NotImplementedError("Subclasses must implement _draw_samples_hw")

@legacy
class SimplexNoise(_NoiseParameterMixin, StochasticParameter):
    """Parameter that generates simplex noise of varying resolutions.

    This parameter expects to sample noise for 2d planes, i.e. for
    sizes ``(H, W, [C])`` and will return a value in the range ``[0.0, 1.0]``
    per spatial location in that plane.

    The noise is sampled from low resolution planes and
    upscaled to the requested height and width. The size of the low
    resolution plane may be defined (large values can be slow) and the
    interpolation method for upscaling can be set.

    Parameters
    ----------
    size_px_max : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Maximum height and width in pixels of the low resolution plane.
        Upon any sampling call, the requested shape will be downscaled until
        the height or width (whichever is larger) does not exceed this maximum
        value anymore. Then the noise will be sampled at that shape and later
        upscaled back to the requested shape.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    upscale_method : str or int or list of str or list of int or imgaug2.parameters.StochasticParameter, optional
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the originally requested shape (i.e. usually
        the image size). This parameter controls the interpolation method to
        use. See also `imresize_many_images()` for a
        description of possible values.

            * If `StochasticParameter`, then a random value will be
              sampled from that parameter per call.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.SimplexNoise(upscale_method="linear")

    Create a parameter that produces smooth simplex noise of varying sizes.

    >>> param = iap.SimplexNoise(
    >>>     size_px_max=(8, 16),
    >>>     upscale_method="nearest")

    Create a parameter that produces rectangular simplex noise of rather
    high detail.

    """

    def __init__(
        self,
        size_px_max: ParamInput = (2, 16),
        upscale_method: ParamInput | str | list[str] | None = None,
    ) -> None:
        super().__init__()
        if upscale_method is None:
            upscale_method = ["linear", "nearest"]

        self.size_px_max = handle_discrete_param(size_px_max, "size_px_max", value_range=(1, 10000))

        if upscale_method == ia.ALL:
            self.upscale_method = Choice(["nearest", "linear", "area", "cubic"])
        elif ia.is_string(upscale_method):
            self.upscale_method = Deterministic(upscale_method)
        elif isinstance(upscale_method, list):
            assert len(upscale_method) >= 1, (
                f"Expected at least one upscale method, got {len(upscale_method)}."
            )
            assert all([ia.is_string(val) for val in upscale_method]), (
                "Expected all upscale methods to be strings, got types {}.".format(
                    ", ".join([str(type(v)) for v in upscale_method])
                )
            )
            self.upscale_method = Choice(upscale_method)
        elif isinstance(upscale_method, StochasticParameter):
            self.upscale_method = upscale_method
        else:
            raise Exception(
                "Expected upscale_method to be string or list of strings or "
                f"StochasticParameter, got {type(upscale_method)}."
            )

    def _draw_samples_hw(self, height: int, width: int, random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        iterations = 1
        rngs = random_state.duplicate(1 + iterations)
        aggregation_method = "max"
        upscale_methods = self.upscale_method.draw_samples((iterations,), random_state=rngs[0])
        result = np.zeros((height, width), dtype=np.float32)
        for i in range(iterations):
            noise_iter = self._draw_samples_iteration(
                height, width, rngs[1 + i], upscale_methods[i]
            )
            if aggregation_method == "avg":
                result += noise_iter
            elif aggregation_method == "min":
                if i == 0:
                    result = noise_iter
                else:
                    result = np.minimum(result, noise_iter)
            else:  # self.aggregation_method == "max"
                if i == 0:
                    result = noise_iter
                else:
                    result = np.maximum(result, noise_iter)

        if aggregation_method == "avg":
            result = result / iterations

        return result

    def _draw_samples_iteration(
        self, height: int, width: int, rng: iarandom.RNG, upscale_method: str
    ) -> Any:  # noqa: ANN401
        opensimplex_seed = rng.generate_seed_()

        # we have to use int(.) here, otherwise we can get warnings about
        # value overflows in OpenSimplex L103
        generator = OpenSimplex(seed=int(opensimplex_seed))

        maxlen = max(height, width)
        size_px_max = self.size_px_max.draw_sample(random_state=rng)
        if maxlen > size_px_max:
            downscale_factor = size_px_max / maxlen
            h_small = int(height * downscale_factor)
            w_small = int(width * downscale_factor)
        else:
            h_small = height
            w_small = width

        # don't go below Hx1 or 1xW
        h_small = max(h_small, 1)
        w_small = max(w_small, 1)

        noise = np.zeros((h_small, w_small), dtype=np.float32)
        for y in range(h_small):
            for x in range(w_small):
                noise[y, x] = generator.noise2(y=y, x=x)

        # Normalize from [-1.0, 1.0] to [0.0, 1.0]
        noise_0to1 = (noise + 1.0) / 2
        noise_0to1 = np.clip(noise_0to1, 0.0, 1.0)

        if noise_0to1.shape != (height, width):
            noise_0to1_uint8 = (noise_0to1 * 255).astype(np.uint8)
            noise_0to1_3d = np.tile(noise_0to1_uint8[..., np.newaxis], (1, 1, 3))
            noise_0to1 = ia.imresize_single_image(
                noise_0to1_3d, (height, width), interpolation=upscale_method
            )
            noise_0to1 = (noise_0to1[..., 0] / 255.0).astype(np.float32)

        return noise_0to1

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"SimplexNoise({str(self.size_px_max)}, {str(self.upscale_method)})"

@legacy
class FrequencyNoise(_NoiseParameterMixin, StochasticParameter):
    """Parameter to generate noise of varying frequencies.

    This parameter expects to sample noise for 2d planes, i.e. for
    sizes ``(H, W, [C])`` and will return a value in the range ``[0.0, 1.0]``
    per spatial location in that plane.

    The exponent controls the frequencies and therefore noise patterns.
    Small values (around ``-4.0``) will result in large blobs. Large values
    (around ``4.0``) will result in small, repetitive patterns.

    The noise is sampled from low resolution planes and
    upscaled to the requested height and width. The size of the low
    resolution plane may be defined (high values can be slow) and the
    interpolation method for upscaling can be set.

    Parameters
    ----------
    exponent : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Exponent to use when scaling in the frequency domain.
        Sane values are in the range ``-4`` (large blobs) to ``4`` (small
        patterns). To generate cloud-like structures, use roughly ``-2``.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

    size_px_max : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        Maximum height and width in pixels of the low resolution plane.
        Upon any sampling call, the requested shape will be downscaled until
        the height or width (whichever is larger) does not exceed this maximum
        value anymore. Then the noise will be sampled at that shape and later
        upscaled back to the requested shape.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    upscale_method : imgaug2.ALL or str or list of str or imgaug2.parameters.StochasticParameter, optional
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the originally requested shape (i.e. usually
        the image size). This parameter controls the interpolation method to
        use. See also `imresize_many_images()` for a
        description of possible values.

            * If `StochasticParameter`, then a random value will be
              sampled from that parameter per call.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.FrequencyNoise(
    >>>     exponent=-2,
    >>>     size_px_max=(16, 32),
    >>>     upscale_method="linear")

    Create a parameter that produces noise with cloud-like patterns.

    """

    def __init__(
        self,
        exponent: ParamInput = (-4, 4),
        size_px_max: ParamInput = (4, 32),
        upscale_method: ParamInput | str | list[str] | None = None,
    ) -> None:
        super().__init__()
        if upscale_method is None:
            upscale_method = ["linear", "nearest"]

        self.exponent = handle_continuous_param(exponent, "exponent")
        self.size_px_max = handle_discrete_param(size_px_max, "size_px_max", value_range=(1, 10000))

        if upscale_method == ia.ALL:
            self.upscale_method = Choice(["nearest", "linear", "area", "cubic"])
        elif ia.is_string(upscale_method):
            self.upscale_method = Deterministic(upscale_method)
        elif isinstance(upscale_method, list):
            assert len(upscale_method) >= 1, (
                f"Expected at least one upscale method, got {len(upscale_method)}."
            )
            is_all_strings = all([ia.is_string(val) for val in upscale_method])
            assert is_all_strings, (
                "Expected all upscale methods to be strings, got types {}.".format(
                    ", ".join([str(type(v)) for v in upscale_method])
                )
            )
            self.upscale_method = Choice(upscale_method)
        elif isinstance(upscale_method, StochasticParameter):
            self.upscale_method = upscale_method
        else:
            raise Exception(
                "Expected upscale_method to be string or list of strings or "
                f"StochasticParameter, got {type(upscale_method)}."
            )

        self._distance_matrix_cache = np.zeros((0, 0), dtype=np.float32)

    def _draw_samples_hw(self, height: int, width: int, random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        # code here is similar to:
        #   http://www.redblobgames.com/articles/noise/2d/
        #   http://www.redblobgames.com/articles/noise/2d/2d-noise.js
        maxlen = max(height, width)
        size_px_max = self.size_px_max.draw_sample(random_state=random_state)
        h_small, w_small = height, width
        if maxlen > size_px_max:
            downscale_factor = size_px_max / maxlen
            h_small = int(height * downscale_factor)
            w_small = int(width * downscale_factor)

        # don't go below Hx4 or 4xW
        h_small = max(h_small, 4)
        w_small = max(w_small, 4)

        # exponents to pronounce some frequencies
        exponent = self.exponent.draw_sample(random_state=random_state)

        # base function to invert, derived from a distance matrix (euclidean
        # distance to image center)
        f = self._get_distance_matrix_cached((h_small, w_small))

        # prevent divide by zero warnings at the image corners in
        # f**exponent
        f[0, 0] = 1
        f[-1, 0] = 1
        f[0, -1] = 1
        f[-1, -1] = 1

        scale = f**exponent

        # invert setting corners to 1
        scale[0, 0] = 0
        scale[-1, 0] = 0
        scale[0, -1] = 0
        scale[-1, -1] = 0

        # generate random base matrix
        # first channel: wn_r, second channel: wn_a
        wn = random_state.random(size=(2, h_small, w_small))
        wn[0, ...] *= max(h_small, w_small) ** 2
        wn[1, ...] *= 2 * np.pi
        wn[0, ...] *= np.cos(wn[1, ...])
        wn[1, ...] *= np.sin(wn[1, ...])
        wn *= scale[np.newaxis, :, :]
        wn = wn.transpose((1, 2, 0))
        if wn.dtype != np.float32:
            wn = wn.astype(np.float32)

        # equivalent but slightly faster then:
        #   wn_freqs_mul = np.zeros(treal.shape, dtype=np.complex128)
        #   wn_freqs_mul.real = wn[0]
        #   wn_freqs_mul.imag = wn[1]
        #   wn_inv = np.fft.ifft2(wn_freqs_mul).real
        wn_inv = cv2.idft(wn)[:, :, 0]

        # normalize to 0 to 1
        # equivalent to but slightly faster than:
        #   wn_inv_min = np.min(wn_inv)
        #   wn_inv_max = np.max(wn_inv)
        #   noise_0to1 = (wn_inv - wn_inv_min) / (wn_inv_max - wn_inv_min)
        # does not accept wn_inv as dst directly
        noise_0to1 = cv2.normalize(
            wn_inv, dst=np.zeros_like(wn_inv), alpha=0.01, beta=1.0, norm_type=cv2.NORM_MINMAX
        )

        # upscale from low resolution to image size
        if noise_0to1.shape != (height, width):
            upscale_method = self.upscale_method.draw_sample(random_state=random_state)
            noise_0to1 = ia.imresize_single_image(
                noise_0to1.astype(np.float32), (height, width), interpolation=upscale_method
            )
            if upscale_method == "cubic":
                noise_0to1 = np.clip(noise_0to1, 0.0, 1.0)

        return noise_0to1

    def _get_distance_matrix_cached(self, size: tuple[int, int]) -> NDArray[np.float32]:
        cache = self._distance_matrix_cache
        height, width = cache.shape
        if height < size[0] or width < size[1]:
            self._distance_matrix_cache = self._create_distance_matrix(
                (max(height, size[0]), max(width, size[1]))
            )

        return self._extract_distance_matrix(self._distance_matrix_cache, size)

    @classmethod
    def _extract_distance_matrix(
        cls, matrix: NDArray[np.float32], size: tuple[int, int]
    ) -> NDArray[np.float32]:
        height, width = matrix.shape[0:2]
        leftover_y = (height - size[0]) / 2
        leftover_x = (width - size[1]) / 2
        y1 = int(np.floor(leftover_y))
        y2 = height - int(np.ceil(leftover_y))
        x1 = int(np.floor(leftover_x))
        x2 = width - int(np.ceil(leftover_x))
        return matrix[y1:y2, x1:x2]

    @classmethod
    def _create_distance_matrix(cls, size: tuple[int, int]) -> NDArray[np.float32]:
        def _create_line(line_size: int) -> NDArray[np.int64]:
            start = np.arange(line_size // 2)
            middle = [line_size // 2] if line_size % 2 == 1 else []
            end = start[::-1]
            return np.concatenate([start, middle, end])

        height, width = size
        ydist = _create_line(height) ** 2
        xdist = _create_line(width) ** 2
        ydist_2d = np.broadcast_to(ydist[:, np.newaxis], size)
        xdist_2d = np.broadcast_to(xdist[np.newaxis, :], size)
        dist = np.sqrt(ydist_2d + xdist_2d)
        return dist.astype(np.float32)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"FrequencyNoise({str(self.exponent)}, {str(self.size_px_max)}, {str(self.upscale_method)})"
