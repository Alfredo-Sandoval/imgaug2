"""Rain augmenters."""

from __future__ import annotations

from typing import Literal

import numpy as np

from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy

from .snowflakes import SnowflakesLayer

@legacy(version="0.4.0")
class RainLayer(SnowflakesLayer):
    """Add a single layer of falling raindrops to images.


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

        - (1) indirectly tested via tests for :class:`Rain`

    Parameters
    ----------
    density : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Same as in :class:`~imgaug2.augmenters.weather.SnowflakesLayer`.

    density_uniformity : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Same as in :class:`~imgaug2.augmenters.weather.SnowflakesLayer`.

    drop_size : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Same as `flake_size` in
        :class:`~imgaug2.augmenters.weather.SnowflakesLayer`.

    drop_size_uniformity : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Same as `flake_size_uniformity` in
        :class:`~imgaug2.augmenters.weather.SnowflakesLayer`.

    angle : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Same as in :class:`~imgaug2.augmenters.weather.SnowflakesLayer`.

    speed : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Same as in :class:`~imgaug2.augmenters.weather.SnowflakesLayer`.

    blur_sigma_fraction : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Same as in :class:`~imgaug2.augmenters.weather.SnowflakesLayer`.

    blur_sigma_limits : tuple of float, optional
        Same as in :class:`~imgaug2.augmenters.weather.SnowflakesLayer`.

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

    @legacy(version="0.4.0")
    def __init__(
        self,
        density: ParamInput,
        density_uniformity: ParamInput,
        drop_size: ParamInput,
        drop_size_uniformity: ParamInput,
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
            density,
            density_uniformity,
            drop_size,
            drop_size_uniformity,
            angle,
            speed,
            blur_sigma_fraction,
            blur_sigma_limits=blur_sigma_limits,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

    @legacy(version="0.4.0")
    @classmethod
    def _blur(cls, noise: Array, sigma: float) -> Array:
        return noise

    @legacy(version="0.4.0")
    @classmethod
    def _postprocess_noise(
        cls, noise_small_blur: Array, flake_size_uniformity_sample: float, nb_channels: int
    ) -> Array:
        noise_small_blur_rgb = np.tile(noise_small_blur[..., np.newaxis], (1, 1, nb_channels))
        return noise_small_blur_rgb

    @legacy(version="0.4.0")
    @classmethod
    def _blend(cls, image: Array, speed_sample: float, noise_small_blur_rgb: Array) -> Array:
        # We set the mean color based on the noise here. That's a pseudo-random
        # approach that saves us from adding the random state as a parameter.
        # Note that the sum of noise_small_blur_rgb can be 0 when at least one
        # image axis size is 0.
        noise_sum = np.sum(noise_small_blur_rgb.flat[0:1000])
        noise_sum = noise_sum if noise_sum > 0 else 1
        drop_mean_color = 110 + (240 - 110) % noise_sum
        noise_small_blur_rgb = noise_small_blur_rgb / 255.0
        # The 1.3 multiplier increases the visibility of drops a bit.
        noise_small_blur_rgb = np.clip(1.3 * noise_small_blur_rgb, 0, 1.0)
        image_f32 = image.astype(np.float32)
        image_f32 = (1 - noise_small_blur_rgb) * image_f32 + noise_small_blur_rgb * drop_mean_color
        return np.clip(image_f32, 0, 255).astype(np.uint8)


@legacy(version="0.4.0")
class Rain(meta.SomeOf):
    """Add falling snowflakes to images.

    This is a wrapper around
    :class:`~imgaug2.augmenters.weather.RainLayer`. It executes 1 to 3
    layers per image.

    .. note::

        This augmenter currently seems to work best for medium-sized images
        around ``192x256``. For smaller images, you may want to increase the
        `speed` value to e.g. ``(0.1, 0.3)``, otherwise the drops tend to
        look like snowflakes. For larger images, you may want to increase
        the `drop_size` to e.g. ``(0.10, 0.20)``.


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
    drop_size : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        See :class:`~imgaug2.augmenters.weather.RainLayer`.

    speed : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        See :class:`~imgaug2.augmenters.weather.RainLayer`.

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
    >>> aug = iaa.Rain(speed=(0.1, 0.3))

    Add rain to small images (around ``96x128``).

    >>> aug = iaa.Rain()

    Add rain to medium sized images (around ``192x256``).

    >>> aug = iaa.Rain(drop_size=(0.10, 0.20))

    Add rain to large images (around ``960x1280``).

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        nb_iterations: ParamInput = (1, 3),
        drop_size: ParamInput = (0.01, 0.02),
        speed: ParamInput = (0.04, 0.20),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        layer = RainLayer(
            density=(0.03, 0.14),
            density_uniformity=(0.8, 1.0),
            drop_size=drop_size,
            drop_size_uniformity=(0.2, 0.5),
            angle=(-15, 15),
            speed=speed,
            blur_sigma_fraction=(0.001, 0.001),
            seed=seed,
            random_state=random_state,
            deterministic=deterministic,
        )

        super().__init__(
            nb_iterations,
            children=[layer.deepcopy() for _ in range(3)],
            random_order=False,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )
