"""Snowflake augmenters."""

from __future__ import annotations

from typing import Literal

import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import arithmetic, blur, contrast, meta
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy

class SnowflakesLayer(meta.Augmenter):
    """Add a single layer of falling snowflakes to images.

    **Supported dtypes**:

        * ``uint8``: yes; indirectly tested (1)
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

        - (1) indirectly tested via tests for :class:`Snowflakes`

    Parameters
    ----------
    density : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Density of the snowflake layer, as a probability of each pixel in
        low resolution space to be a snowflake.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be in the interval ``[0.01, 0.075]``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    density_uniformity : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Size uniformity of the snowflakes. Higher values denote more
        similarly sized snowflakes.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be around ``0.5``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    flake_size : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Size of the snowflakes. This parameter controls the resolution at
        which snowflakes are sampled. Higher values mean that the resolution
        is closer to the input image's resolution and hence each sampled
        snowflake will be smaller (because of the smaller pixel size).

        Valid values are in the interval ``(0.0, 1.0]``.
        Recommended values:

            * On 96x128 a value of ``(0.1, 0.4)`` worked well.
            * On 192x256 a value of ``(0.2, 0.7)`` worked well.
            * On 960x1280 a value of ``(0.7, 0.95)`` worked well.

        Datatype behaviour:

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    flake_size_uniformity : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Controls the size uniformity of the snowflakes. Higher values mean
        that the snowflakes are more similarly sized.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be around ``0.5``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    angle : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Angle in degrees of motion blur applied to the snowflakes, where
        ``0.0`` is motion blur that points straight upwards.
        Recommended to be in the interval ``[-30, 30]``.
        See also :func:`~imgaug2.augmenters.blur.MotionBlur.__init__`.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    speed : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Perceived falling speed of the snowflakes. This parameter controls the
        motion blur's kernel size. It follows roughly the form
        ``kernel_size = image_size * speed``. Hence, values around ``1.0``
        denote that the motion blur should "stretch" each snowflake over the
        whole image.

        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended values:

            * On 96x128 a value of ``(0.01, 0.05)`` worked well.
            * On 192x256 a value of ``(0.007, 0.03)`` worked well.
            * On 960x1280 a value of ``(0.001, 0.03)`` worked well.

        Datatype behaviour:

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    blur_sigma_fraction : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Standard deviation (as a fraction of the image size) of gaussian blur
        applied to the snowflakes.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be in the interval ``[0.0001, 0.001]``. May still
        require tinkering based on image size.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    blur_sigma_limits : tuple of float, optional
        Controls allowed min and max values of `blur_sigma_fraction`
        after(!) multiplication with the image size. First value is the
        minimum, second value is the maximum. Values outside of that range
        will be clipped to be within that range. This prevents extreme
        values for very small or large images.

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

    """

    def __init__(
        self,
        density: ParamInput,
        density_uniformity: ParamInput,
        flake_size: ParamInput,
        flake_size_uniformity: ParamInput,
        angle: ParamInput,
        speed: ParamInput,
        blur_sigma_fraction: ParamInput,
        blur_sigma_limits: tuple[float, float] = (0.5, 3.75),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.density = density
        self.density_uniformity = iap.handle_continuous_param(
            density_uniformity, "density_uniformity", value_range=(0.0, 1.0)
        )
        self.flake_size = iap.handle_continuous_param(
            flake_size, "flake_size", value_range=(0.0 + 1e-4, 1.0)
        )
        self.flake_size_uniformity = iap.handle_continuous_param(
            flake_size_uniformity, "flake_size_uniformity", value_range=(0.0, 1.0)
        )
        self.angle = iap.handle_continuous_param(angle, "angle")
        self.speed = iap.handle_continuous_param(speed, "speed", value_range=(0.0, 1.0))
        self.blur_sigma_fraction = iap.handle_continuous_param(
            blur_sigma_fraction, "blur_sigma_fraction", value_range=(0.0, 1.0)
        )

        # (min, max), same for all images
        self.blur_sigma_limits = blur_sigma_limits

        # (height, width), same for all images
        self.gate_noise_size = (8, 8)

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

        rss = random_state.duplicate(len(images))
        for i, (image, rs) in enumerate(zip(images, rss, strict=True)):
            batch.images[i] = self.draw_on_image(image, rs)
        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [
            self.density,
            self.density_uniformity,
            self.flake_size,
            self.flake_size_uniformity,
            self.angle,
            self.speed,
            self.blur_sigma_fraction,
            self.blur_sigma_limits,
            self.gate_noise_size,
        ]

    def draw_on_image(self, image: Array, random_state: iarandom.RNG) -> Array:
        assert image.ndim == 3, (
            f"Expected input image to be three-dimensional, got {image.ndim} dimensions."
        )
        assert image.shape[2] in [1, 3], (
            "Expected to get image with a channel axis of size 1 or 3, "
            f"got {image.shape[2]} (shape: {image.shape})"
        )

        rss = random_state.duplicate(2)

        flake_size_sample = self.flake_size.draw_sample(random_state)
        flake_size_uniformity_sample = self.flake_size_uniformity.draw_sample(random_state)
        angle_sample = self.angle.draw_sample(random_state)
        speed_sample = self.speed.draw_sample(random_state)
        blur_sigma_fraction_sample = self.blur_sigma_fraction.draw_sample(random_state)

        height, width, nb_channels = image.shape
        downscale_factor = np.clip(1.0 - flake_size_sample, 0.001, 1.0)
        height_down = max(1, int(height * downscale_factor))
        width_down = max(1, int(width * downscale_factor))
        noise = self._generate_noise(height_down, width_down, self.density, rss[0])

        # gate the sampled noise via noise in range [0.0, 1.0]
        # this leads to less flakes in some areas of the image and more in
        # other areas
        gate_noise = iap.Beta(1.0, 1.0 - self.density_uniformity)
        noise = self._gate(noise, gate_noise, self.gate_noise_size, rss[1])
        noise = ia.imresize_single_image(noise, (height, width), interpolation="cubic")

        # apply a bit of gaussian blur and then motion blur according to
        # angle and speed
        sigma = max(height, width) * blur_sigma_fraction_sample
        sigma = np.clip(sigma, self.blur_sigma_limits[0], self.blur_sigma_limits[1])
        noise_small_blur = self._blur(noise, sigma)
        noise_small_blur = self._motion_blur(
            noise_small_blur, angle=angle_sample, speed=speed_sample, random_state=random_state
        )

        noise_small_blur_rgb = self._postprocess_noise(
            noise_small_blur, flake_size_uniformity_sample, nb_channels
        )

        return self._blend(image, speed_sample, noise_small_blur_rgb)

    @classmethod
    def _generate_noise(
        cls, height: int, width: int, density: ParamInput, random_state: iarandom.RNG
    ) -> Array:
        noise = arithmetic.Salt(p=density, random_state=random_state)
        return noise.augment_image(np.zeros((height, width), dtype=np.uint8))

    @classmethod
    def _gate(
        cls,
        noise: Array,
        gate_noise: iap.StochasticParameter,
        gate_size: tuple[int, int],
        random_state: iarandom.RNG,
    ) -> Array:
        # the beta distribution here has most of its weight around 1.0 and
        # will only rarely sample values around 0.0 the average of the
        # sampled values seems to be at around 0.6-0.75
        gate_noise = gate_noise.draw_samples(gate_size, random_state)
        gate_noise_up = ia.imresize_single_image(
            gate_noise, noise.shape[0:2], interpolation="cubic"
        )
        gate_noise_up = np.clip(gate_noise_up, 0.0, 1.0)
        return np.clip(noise.astype(np.float32) * gate_noise_up, 0, 255).astype(np.uint8)

    @classmethod
    def _blur(cls, noise: Array, sigma: float) -> Array:
        return blur.blur_gaussian_(noise, sigma=sigma)

    @classmethod
    def _motion_blur(
        cls, noise: Array, angle: float, speed: float, random_state: iarandom.RNG
    ) -> Array:
        size = max(noise.shape[0:2])
        k = int(speed * size)
        if k <= 1:
            return noise

        # we use max(k, 3) here because MotionBlur errors for anything less
        # than 3
        blurer = blur.MotionBlur(k=max(k, 3), angle=angle, direction=1.0, random_state=random_state)
        return blurer.augment_image(noise)

    @legacy(version="0.4.0")
    @classmethod
    def _postprocess_noise(
        cls, noise_small_blur: Array, flake_size_uniformity_sample: float, nb_channels: int
    ) -> Array:
        # use contrast adjustment of noise to make the flake size a bit less
        # uniform then readjust the noise values to make them more visible
        # again
        gain = 1.0 + 2 * (1 - flake_size_uniformity_sample)
        gain_adj = 1.0 + 5 * (1 - flake_size_uniformity_sample)
        noise_small_blur = contrast.GammaContrast(gain).augment_image(noise_small_blur)
        noise_small_blur = noise_small_blur.astype(np.float32) * gain_adj
        noise_small_blur_rgb = np.tile(noise_small_blur[..., np.newaxis], (1, 1, nb_channels))
        return noise_small_blur_rgb

    @legacy(version="0.4.0")
    @classmethod
    def _blend(cls, image: Array, speed_sample: float, noise_small_blur_rgb: Array) -> Array:
        # blend:
        # sum for a bit of glowy, hardly visible flakes
        # max for the main flakes
        image_f32 = image.astype(np.float32)
        image_f32 = cls._blend_by_sum(image_f32, (0.1 + 20 * speed_sample) * noise_small_blur_rgb)
        image_f32 = cls._blend_by_max(image_f32, (1.0 + 20 * speed_sample) * noise_small_blur_rgb)
        return image_f32

    # TODO replace this by a function from module blend.py
    @classmethod
    def _blend_by_sum(cls, image_f32: Array, noise_small_blur_rgb: Array) -> Array:
        image_f32 = image_f32 + noise_small_blur_rgb
        return np.clip(image_f32, 0, 255).astype(np.uint8)

    # TODO replace this by a function from module blend.py
    @classmethod
    def _blend_by_max(cls, image_f32: Array, noise_small_blur_rgb: Array) -> Array:
        image_f32 = np.maximum(image_f32, noise_small_blur_rgb)
        return np.clip(image_f32, 0, 255).astype(np.uint8)


class Snowflakes(meta.SomeOf):
    """Add falling snowflakes to images.

    This is a wrapper around
    :class:`~imgaug2.augmenters.weather.SnowflakesLayer`. It executes 1 to 3
    layers per image.

    **Supported dtypes**:

        * ``uint8``: yes; tested
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

        - (1) Parameters of this augmenter are optimized for the value range
              of ``uint8``. While other dtypes may be accepted, they will lead
              to images augmented in ways inappropriate for the respective
              dtype.

    Parameters
    ----------
    density : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Density of the snowflake layer, as a probability of each pixel in
        low resolution space to be a snowflake.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be in the interval ``[0.01, 0.075]``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    density_uniformity : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Size uniformity of the snowflakes. Higher values denote more
        similarly sized snowflakes.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be around ``0.5``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    flake_size : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Size of the snowflakes. This parameter controls the resolution at
        which snowflakes are sampled. Higher values mean that the resolution
        is closer to the input image's resolution and hence each sampled
        snowflake will be smaller (because of the smaller pixel size).

        Valid values are in the interval ``(0.0, 1.0]``.
        Recommended values:

            * On ``96x128`` a value of ``(0.1, 0.4)`` worked well.
            * On ``192x256`` a value of ``(0.2, 0.7)`` worked well.
            * On ``960x1280`` a value of ``(0.7, 0.95)`` worked well.

        Datatype behaviour:

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    flake_size_uniformity : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Controls the size uniformity of the snowflakes. Higher values mean
        that the snowflakes are more similarly sized.
        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended to be around ``0.5``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    angle : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Angle in degrees of motion blur applied to the snowflakes, where
        ``0.0`` is motion blur that points straight upwards.
        Recommended to be in the interval ``[-30, 30]``.
        See also :func:`~imgaug2.augmenters.blur.MotionBlur.__init__`.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    speed : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Perceived falling speed of the snowflakes. This parameter controls the
        motion blur's kernel size. It follows roughly the form
        ``kernel_size = image_size * speed``. Hence, values around ``1.0``
        denote that the motion blur should "stretch" each snowflake over
        the whole image.

        Valid values are in the interval ``[0.0, 1.0]``.
        Recommended values:

            * On ``96x128`` a value of ``(0.01, 0.05)`` worked well.
            * On ``192x256`` a value of ``(0.007, 0.03)`` worked well.
            * On ``960x1280`` a value of ``(0.001, 0.03)`` worked well.

        Datatype behaviour:

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

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
    >>> aug = iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05))

    Add snowflakes to small images (around ``96x128``).

    >>> aug = iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03))

    Add snowflakes to medium-sized images (around ``192x256``).

    >>> aug = iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))

    Add snowflakes to large images (around ``960x1280``).

    """

    def __init__(
        self,
        density: ParamInput = (0.005, 0.075),
        density_uniformity: ParamInput = (0.3, 0.9),
        flake_size: ParamInput = (0.2, 0.7),
        flake_size_uniformity: ParamInput = (0.4, 0.8),
        angle: ParamInput = (-30, 30),
        speed: ParamInput = (0.007, 0.03),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        layer = SnowflakesLayer(
            density=density,
            density_uniformity=density_uniformity,
            flake_size=flake_size,
            flake_size_uniformity=flake_size_uniformity,
            angle=angle,
            speed=speed,
            blur_sigma_fraction=(0.0001, 0.001),
            seed=seed,
            random_state=random_state,
            deterministic=deterministic,
        )

        super().__init__(
            (1, 3),
            children=[layer.deepcopy() for _ in range(3)],
            random_order=False,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )
