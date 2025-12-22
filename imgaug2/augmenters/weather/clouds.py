"""Cloud and fog augmenters."""

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

class CloudLayer(meta.Augmenter):
    """Add a single layer of clouds to an image.

    **Supported dtypes**:

        * ``uint8``: yes; indirectly tested (1)
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: yes; not tested
        * ``float32``: yes; not tested
        * ``float64``: yes; not tested
        * ``float128``: yes; not tested (2)
        * ``bool``: no

        - (1) Indirectly tested via tests for :class:`Clouds`` and :class:`Fog`
        - (2) Note that random values are usually sampled as ``int64`` or
              ``float64``, which ``float128`` images would exceed. Note also
              that random values might have to upscaled, which is done
              via :func:`~imgaug2.imgaug2.imresize_many_images` and has its own
              limited dtype support (includes however floats up to ``64bit``).

    Parameters
    ----------
    intensity_mean : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Mean intensity of the clouds (i.e. mean color).
        Recommended to be in the interval ``[190, 255]``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    intensity_freq_exponent : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Exponent of the frequency noise used to add fine intensity to the
        mean intensity.
        Recommended to be in the interval ``[-2.5, -1.5]``.
        See :func:`~imgaug2.parameters.FrequencyNoise.__init__` for details.

    intensity_coarse_scale : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Standard deviation of the gaussian distribution used to add more
        localized intensity to the mean intensity. Sampled in low resolution
        space, i.e. affects final intensity on a coarse level.
        Recommended to be in the interval ``(0, 10]``.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    alpha_min : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Minimum alpha when blending cloud noise with the image.
        High values will lead to clouds being "everywhere".
        Recommended to usually be at around ``0.0`` for clouds and ``>0`` for
        fog.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    alpha_multiplier : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Multiplier for the sampled alpha values. High values will lead to
        denser clouds wherever they are visible.
        Recommended to be in the interval ``[0.3, 1.0]``.
        Note that this parameter currently overlaps with `density_multiplier`,
        which is applied a bit later to the alpha mask.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    alpha_size_px_max : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Controls the image size at which the alpha mask is sampled.
        Lower values will lead to coarser alpha masks and hence larger
        clouds (and empty areas).
        See :func:`~imgaug2.parameters.FrequencyNoise.__init__` for details.

    alpha_freq_exponent : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Exponent of the frequency noise used to sample the alpha mask.
        Similarly to `alpha_size_max_px`, lower values will lead to coarser
        alpha patterns.
        Recommended to be in the interval ``[-4.0, -1.5]``.
        See :func:`~imgaug2.parameters.FrequencyNoise.__init__` for details.

    sparsity : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Exponent applied late to the alpha mask. Lower values will lead to
        coarser cloud patterns, higher values to finer patterns.
        Recommended to be somewhere around ``1.0``.
        Do not deviate far from that value, otherwise the alpha mask might
        get weird patterns with sudden fall-offs to zero that look very
        unnatural.

            * If a ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled
              per image from that parameter.

    density_multiplier : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Late multiplier for the alpha mask, similar to `alpha_multiplier`.
        Set this higher to get "denser" clouds wherever they are visible.
        Recommended to be around ``[0.5, 1.5]``.

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

    """

    def __init__(
        self,
        intensity_mean: ParamInput,
        intensity_freq_exponent: ParamInput,
        intensity_coarse_scale: ParamInput,
        alpha_min: ParamInput,
        alpha_multiplier: ParamInput,
        alpha_size_px_max: ParamInput,
        alpha_freq_exponent: ParamInput,
        sparsity: ParamInput,
        density_multiplier: ParamInput,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.intensity_mean = iap.handle_continuous_param(intensity_mean, "intensity_mean")
        self.intensity_freq_exponent = intensity_freq_exponent
        self.intensity_coarse_scale = intensity_coarse_scale
        self.alpha_min = iap.handle_continuous_param(alpha_min, "alpha_min")
        self.alpha_multiplier = iap.handle_continuous_param(alpha_multiplier, "alpha_multiplier")
        self.alpha_size_px_max = alpha_size_px_max
        self.alpha_freq_exponent = alpha_freq_exponent
        self.sparsity = iap.handle_continuous_param(sparsity, "sparsity")
        self.density_multiplier = iap.handle_continuous_param(
            density_multiplier, "density_multiplier"
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

        rss = random_state.duplicate(len(images))
        for i, (image, rs) in enumerate(zip(images, rss, strict=True)):
            batch.images[i] = self.draw_on_image(image, rs)
        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [
            self.intensity_mean,
            self.alpha_min,
            self.alpha_multiplier,
            self.alpha_size_px_max,
            self.alpha_freq_exponent,
            self.intensity_freq_exponent,
            self.sparsity,
            self.density_multiplier,
            self.intensity_coarse_scale,
        ]

    def draw_on_image(self, image: Array, random_state: iarandom.RNG) -> Array:
        iadt.gate_dtypes_strs(
            image,
            allowed="uint8 float16 float32 float64 float128",
            disallowed="bool uint16 uint32 uint64 int8 int16 int32 int64",
            augmenter=self,
        )

        alpha, intensity = self.generate_maps(image, random_state)
        alpha = alpha[..., np.newaxis]
        intensity = intensity[..., np.newaxis]

        if image.dtype.kind == "f":
            alpha = alpha.astype(image.dtype)
            intensity = intensity.astype(image.dtype)
            return (1 - alpha) * image + alpha * intensity

        intensity = np.clip(intensity, 0, 255)
        # TODO use blend_alpha_() here
        return np.clip(
            (1 - alpha) * image.astype(alpha.dtype) + alpha * intensity.astype(alpha.dtype), 0, 255
        ).astype(np.uint8)

    def generate_maps(self, image: Array, random_state: iarandom.RNG) -> tuple[Array, Array]:
        intensity_mean_sample = self.intensity_mean.draw_sample(random_state)
        alpha_min_sample = self.alpha_min.draw_sample(random_state)
        alpha_multiplier_sample = self.alpha_multiplier.draw_sample(random_state)
        alpha_size_px_max = self.alpha_size_px_max
        intensity_freq_exponent = self.intensity_freq_exponent
        alpha_freq_exponent = self.alpha_freq_exponent
        sparsity_sample = self.sparsity.draw_sample(random_state)
        density_multiplier_sample = self.density_multiplier.draw_sample(random_state)

        height, width = image.shape[0:2]
        rss_alpha, rss_intensity = random_state.duplicate(2)

        intensity_coarse = self._generate_intensity_map_coarse(
            height,
            width,
            intensity_mean_sample,
            iap.Normal(0, scale=self.intensity_coarse_scale),
            rss_intensity,
        )
        intensity_fine = self._generate_intensity_map_fine(
            height, width, intensity_mean_sample, intensity_freq_exponent, rss_intensity
        )
        intensity = intensity_coarse + intensity_fine

        alpha = self._generate_alpha_mask(
            height,
            width,
            alpha_min_sample,
            alpha_multiplier_sample,
            alpha_freq_exponent,
            alpha_size_px_max,
            sparsity_sample,
            density_multiplier_sample,
            rss_alpha,
        )

        return alpha, intensity

    @classmethod
    def _generate_intensity_map_coarse(
        cls,
        height: int,
        width: int,
        intensity_mean: float,
        intensity_local_offset: iap.StochasticParameter,
        random_state: iarandom.RNG,
    ) -> Array:
        # TODO (8, 8) might be too simplistic for some image sizes
        height_intensity, width_intensity = (8, 8)
        intensity = intensity_mean + intensity_local_offset.draw_samples(
            (height_intensity, width_intensity), random_state
        )
        intensity = ia.imresize_single_image(intensity, (height, width), interpolation="cubic")

        return intensity

    @classmethod
    def _generate_intensity_map_fine(
        cls,
        height: int,
        width: int,
        intensity_mean: float,
        exponent: ParamInput,
        random_state: iarandom.RNG,
    ) -> Array:
        intensity_details_generator = iap.FrequencyNoise(
            exponent=exponent,
            size_px_max=max(height, width, 1),  # 1 here for case H, W being 0
            upscale_method="cubic",
        )
        intensity_details = intensity_details_generator.draw_samples((height, width), random_state)
        return intensity_mean * ((2 * intensity_details - 1.0) / 5.0)

    @classmethod
    def _generate_alpha_mask(
        cls,
        height: int,
        width: int,
        alpha_min: float,
        alpha_multiplier: float,
        exponent: ParamInput,
        alpha_size_px_max: ParamInput,
        sparsity: float,
        density_multiplier: float,
        random_state: iarandom.RNG,
    ) -> Array:
        alpha_generator = iap.FrequencyNoise(
            exponent=exponent, size_px_max=alpha_size_px_max, upscale_method="cubic"
        )
        alpha_local = alpha_generator.draw_samples((height, width), random_state)
        alpha = alpha_min + (alpha_multiplier * alpha_local)
        alpha = (alpha**sparsity) * density_multiplier
        alpha = np.clip(alpha, 0.0, 1.0)

        return alpha


# TODO add vertical gradient alpha to have clouds only at skylevel/groundlevel
# TODO add configurable parameters
class Clouds(meta.SomeOf):
    """
    Add clouds to images.

    This is a wrapper around :class:`~imgaug2.augmenters.weather.CloudLayer`.
    It executes 1 to 2 layers per image, leading to varying densities and
    frequency patterns of clouds.

    This augmenter seems to be fairly robust w.r.t. the image size. Tested
    with ``96x128``, ``192x256`` and ``960x1280``.

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
    >>> aug = iaa.Clouds()

    Create an augmenter that adds clouds to images.

    """

    def __init__(
        self,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        layers = [
            CloudLayer(
                intensity_mean=(196, 255),
                intensity_freq_exponent=(-2.5, -2.0),
                intensity_coarse_scale=10,
                alpha_min=0,
                alpha_multiplier=(0.25, 0.75),
                alpha_size_px_max=(2, 8),
                alpha_freq_exponent=(-2.5, -2.0),
                sparsity=(0.8, 1.0),
                density_multiplier=(0.5, 1.0),
                seed=seed,
                random_state=random_state,
                deterministic=deterministic,
            ),
            CloudLayer(
                intensity_mean=(196, 255),
                intensity_freq_exponent=(-2.0, -1.0),
                intensity_coarse_scale=10,
                alpha_min=0,
                alpha_multiplier=(0.5, 1.0),
                alpha_size_px_max=(64, 128),
                alpha_freq_exponent=(-2.0, -1.0),
                sparsity=(1.0, 1.4),
                density_multiplier=(0.8, 1.5),
                seed=seed,
                random_state=random_state,
                deterministic=deterministic,
            ),
        ]

        super().__init__(
            (1, 2),
            children=layers,
            random_order=False,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


# TODO add vertical gradient alpha to have fog only at skylevel/groundlevel
# TODO add configurable parameters
class Fog(CloudLayer):
    """Add fog to images.

    This is a wrapper around :class:`~imgaug2.augmenters.weather.CloudLayer`.
    It executes a single layer per image with a configuration leading to
    fairly dense clouds with low-frequency patterns.

    This augmenter seems to be fairly robust w.r.t. the image size. Tested
    with ``96x128``, ``192x256`` and ``960x1280``.

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
    >>> aug = iaa.Fog()

    Create an augmenter that adds fog to images.

    """

    def __init__(
        self,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            intensity_mean=(220, 255),
            intensity_freq_exponent=(-2.0, -1.5),
            intensity_coarse_scale=2,
            alpha_min=(0.7, 0.9),
            alpha_multiplier=0.3,
            alpha_size_px_max=(2, 8),
            alpha_freq_exponent=(-4.0, -2.0),
            sparsity=0.9,
            density_multiplier=(0.4, 0.9),
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


# TODO add examples and add these to the overview docs
# TODO snowflakes are all almost 100% white, add some grayish tones and
#      maybe color to them
