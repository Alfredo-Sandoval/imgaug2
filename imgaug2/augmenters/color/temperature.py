from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.compat.markers import legacy
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, Images, ParamInput, RNGInput

from ._utils import CSPACE_RGB, ColorSpace, ColorSpaceInput, KelvinInput
from .colorspace import change_colorspace_

_KELVIN_TO_RGB_TABLE_FP = Path(__file__).resolve().parents[1] / "kelvin_to_rgb_table.json"

_PLANCKIAN_XYZ_TO_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ],
    dtype=np.float64,
)


@legacy(version="0.4.0")
class _KelvinToRGBTableSingleton:
    _INSTANCE = None

    @legacy(version="0.4.0")
    @classmethod
    def get_instance(cls) -> _KelvinToRGBTable:
        if cls._INSTANCE is None:
            cls._INSTANCE = _KelvinToRGBTable()
        return cls._INSTANCE


@legacy(version="0.4.0")
class _KelvinToRGBTable:
    _TABLE = None

    @legacy(version="0.4.0")
    def __init__(self) -> None:
        self.table = self.create_table()

    @legacy(version="0.4.0")
    def transform_kelvins_to_rgb_multipliers(self, kelvins: Array) -> Array:
        """Transform kelvin values to corresponding multipliers for RGB images.

        A single returned multiplier denotes the channelwise multipliers
        in the range ``[0.0, 1.0]`` to apply to an image to change its kelvin
        value to the desired one.


        Parameters
        ----------
        kelvins : iterable of number
            Imagewise temperatures in kelvin.

        Returns
        -------
        ndarray
            ``float32 (N, 3) ndarrays``, one per kelvin.

        """
        kelvins = np.clip(kelvins, 1000, 40000)

        tbl_indices = kelvins / 100 - (1000 // 100)
        tbl_indices_floored = np.floor(tbl_indices)
        tbl_indices_ceiled = np.ceil(tbl_indices)
        interpolation_factors = tbl_indices - tbl_indices_floored

        tbl_indices_floored_int = tbl_indices_floored.astype(np.int32)
        tbl_indices_ceiled_int = tbl_indices_ceiled.astype(np.int32)

        multipliers_floored = self.table[tbl_indices_floored_int, :]
        multipliers_ceiled = self.table[tbl_indices_ceiled_int, :]
        multipliers = multipliers_floored + interpolation_factors[:, np.newaxis] * (
            multipliers_ceiled - multipliers_floored
        )

        return multipliers

    @legacy(version="0.4.0")
    @classmethod
    def create_table(cls) -> Array:
        with _KELVIN_TO_RGB_TABLE_FP.open("r", encoding="utf-8") as table_file:
            table_values = json.load(table_file)
        table = np.asarray(table_values, dtype=np.float32) / 255.0
        _KelvinToRGBTable._TABLE = table
        return cast(Array, table)


_PLANCKIAN_XYZ_TO_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ],
    dtype=np.float64,
)


def _kelvin_to_rgb_planckian(kelvins: Array) -> Array:
    """Approximate black-body temperature -> RGB multipliers via Planckian locus.

    Uses the Hernandez-Andres et al. approximation for CIE xy chromaticities,
    then converts to sRGB (D65). Inputs are clipped to [1667, 25000] K and
    each output row is normalized to a max channel value of 1.0.
    Currently unused; reserved for a future release.
    """
    kelvins_arr = np.asarray(kelvins, dtype=np.float64).reshape(-1)
    kelvins_arr = np.clip(kelvins_arr, 1667.0, 25000.0)

    x = np.empty_like(kelvins_arr)
    mask_low = kelvins_arr <= 4000.0
    t_low = kelvins_arr[mask_low]
    t_high = kelvins_arr[~mask_low]
    x[mask_low] = (
        -0.2661239e9 / (t_low**3) - 0.2343580e6 / (t_low**2) + 0.8776956e3 / t_low + 0.179910
    )
    x[~mask_low] = (
        -3.0258469e9 / (t_high**3) + 2.1070379e6 / (t_high**2) + 0.2226347e3 / t_high + 0.240390
    )

    y = np.empty_like(kelvins_arr)
    mask_low_y = kelvins_arr <= 2222.0
    mask_mid = (kelvins_arr > 2222.0) & (kelvins_arr <= 4000.0)
    mask_high = kelvins_arr > 4000.0
    y[mask_low_y] = (
        -1.1063814 * (x[mask_low_y] ** 3)
        - 1.34811020 * (x[mask_low_y] ** 2)
        + 2.18555832 * x[mask_low_y]
        - 0.20219683
    )
    y[mask_mid] = (
        -0.9549476 * (x[mask_mid] ** 3)
        - 1.37418593 * (x[mask_mid] ** 2)
        + 2.09137015 * x[mask_mid]
        - 0.16748867
    )
    y[mask_high] = (
        3.0817580 * (x[mask_high] ** 3)
        - 5.87338670 * (x[mask_high] ** 2)
        + 3.75112997 * x[mask_high]
        - 0.37001483
    )

    y_safe = np.clip(y, 1e-12, None)
    x_xyz = x / y_safe
    z_xyz = (1.0 - x - y_safe) / y_safe
    xyz = np.stack([x_xyz, np.ones_like(x_xyz), z_xyz], axis=1)

    rgb_linear = xyz @ _PLANCKIAN_XYZ_TO_SRGB.T
    rgb_linear = np.clip(rgb_linear, 0.0, None)
    rgb = np.where(
        rgb_linear <= 0.0031308,
        12.92 * rgb_linear,
        1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055,
    )
    rgb = np.clip(rgb, 0.0, None)

    max_channel = np.max(rgb, axis=1, keepdims=True)
    rgb_norm = np.where(max_channel > 0.0, rgb / max_channel, 0.0)
    return rgb_norm.astype(np.float32)


@legacy(version="0.4.0")
def change_color_temperatures_(
    images: Images,
    kelvins: KelvinInput,
    from_colorspaces: ColorSpaceInput = CSPACE_RGB,
) -> Images:
    """Change in-place the temperature of images to given values in Kelvin.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.color.change_colorspace_`.

    Parameters
    ----------
    images : ndarray or list of ndarray
        The images which's color temperature is supposed to be changed.
        Either a list of ``(H,W,3)`` arrays or a single ``(N,H,W,3)`` array.

    kelvins : iterable of number
        Temperatures in Kelvin. One per image. Expected value range is in
        the interval ``(1000, 4000)``.

    from_colorspaces : str or list of str, optional
        The source colorspace.
        See :func:`~imgaug2.augmenters.color.change_colorspaces_`.
        Defaults to ``RGB``.

    Returns
    -------
    ndarray or list of ndarray
        Images with target color temperatures.
        The input array(s) might have been changed in-place.

    """
    # we return here early, because we validate below the first kelvin value
    if len(images) == 0:
        return images

    # TODO this is very similar to the validation in change_colorspaces_().
    #      Make DRY.
    def _validate(
        arg: ColorSpaceInput | KelvinInput,
        arg_name: str,
        datatype: Literal["str", "number"],
    ) -> Array | Sequence[ColorSpace] | Sequence[float | int]:
        if ia.is_iterable(arg) and not ia.is_string(arg):
            assert len(arg) == len(images), (
                f"If `{arg_name}` is provided as an iterable it must have the same "
                f"length as `images`. Got length {len(arg)}, expected {len(images)}."
            )
        elif datatype == "str":
            assert ia.is_string(arg), (
                f"Expected `{arg_name}` to be either an iterable of strings or a single "
                f"string. Got type {type(arg).__name__}."
            )
            arg = [arg] * len(images)
        else:
            assert ia.is_single_number(arg), (
                f"Expected `{arg_name}` to be either an iterable of numbers or a single "
                f"number. Got type {type(arg).__name__}."
            )
            arg = np.tile(np.float32([arg]), (len(images),))
        return cast(
            Array | Sequence[ColorSpace] | Sequence[float | int],
            arg,
        )

    kelvins = _validate(kelvins, "kelvins", "number")
    from_colorspaces = _validate(from_colorspaces, "from_colorspaces", "str")

    # list `kelvins` inputs are not yet converted to ndarray by _validate()
    kelvins = np.array(kelvins, dtype=np.float32)

    # Validate only one kelvin value for performance reasons.
    # If values are outside that range, the kelvin table simply clips them.
    # If there are no images (and hence no kelvin values), we already returned
    # above.
    assert 1000 <= kelvins[0] <= 40000, (
        "Expected Kelvin values in the interval [1000, 40000]. "
        f"Got interval [{np.min(kelvins):.8f}, {np.max(kelvins):.8f}]."
    )

    table = _KelvinToRGBTableSingleton.get_instance()
    rgb_multipliers = table.transform_kelvins_to_rgb_multipliers(kelvins)
    rgb_multipliers_nhwc = rgb_multipliers.reshape((-1, 1, 1, 3))

    gen = enumerate(zip(images, rgb_multipliers_nhwc, from_colorspaces, strict=True))
    for i, (image, rgb_multiplier_hwc, from_colorspace) in gen:
        image_rgb = change_colorspace_(
            image, to_colorspace=CSPACE_RGB, from_colorspace=from_colorspace
        )

        # we always have uint8 at this point as only that is accepted by
        # convert_colorspace

        # all multipliers are in the range [0.0, 1.0], hence we can afford to
        # not clip here
        image_temp_adj = np.round(image_rgb.astype(np.float32) * rgb_multiplier_hwc).astype(
            np.uint8
        )

        image_orig_cspace = change_colorspace_(
            image_temp_adj, to_colorspace=from_colorspace, from_colorspace=CSPACE_RGB
        )
        images[i] = image_orig_cspace
    return images


@legacy(version="0.4.0")
def change_color_temperature(
    image: Array, kelvin: float | int, from_colorspace: ColorSpace = CSPACE_RGB
) -> Array:
    """Change the temperature of an image to a given value in Kelvin.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.color.change_color_temperatures_`.

    Parameters
    ----------
    image : ndarray
        The image which's color temperature is supposed to be changed.
        Expected to be of shape ``(H,W,3)`` array.

    kelvin : number
        The temperature in Kelvin. Expected value range is in
        the interval ``(1000, 4000)``.

    from_colorspace : str, optional
        The source colorspace.
        See :func:`~imgaug2.augmenters.color.change_colorspaces_`.
        Defaults to ``RGB``.

    Returns
    -------
    ndarray
        Image with target color temperature.

    """
    from . import change_color_temperatures_ as change_color_temperatures_fn

    return change_color_temperatures_fn(
        image[np.newaxis, ...], [kelvin], from_colorspaces=[from_colorspace]
    )[0]


@legacy(version="0.4.0")
class ChangeColorTemperature(meta.Augmenter):
    """Change the temperature to a provided Kelvin value.

    Low Kelvin values around ``1000`` to ``4000`` will result in red, yellow
    or orange images. Kelvin values around ``10000`` to ``40000`` will result
    in progressively darker blue tones.

    Color temperatures taken from
    `<http://www.vendian.org/mncharity/dir3/blackbody/UnstableURLs/bbr_color.html>`_

    Basic method to change color temperatures taken from
    `<https://stackoverflow.com/a/11888449>`_


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.color.change_color_temperatures_`.

    Parameters
    ----------
    kelvin : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Temperature in Kelvin. The temperatures of images will be modified to
        this value. Must be in the interval ``[1000, 40000]``.

            * If a number, exactly that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the
              interval ``[a, b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
            ``list`` per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.ChangeColorTemperature((1100, 10000))

    Create an augmenter that changes the color temperature of images to
    a random value between ``1100`` and ``10000`` Kelvin.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        kelvin: ParamInput = (1000, 11000),
        from_colorspace: ColorSpace = CSPACE_RGB,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.kelvin = iap.handle_continuous_param(
            kelvin, "kelvin", value_range=(1000, 40000), tuple_to_uniform=True, list_to_choice=True
        )
        self.from_colorspace = from_colorspace

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        if batch.images is not None:
            from . import change_color_temperatures_ as change_color_temperatures_fn

            nb_rows = batch.nb_rows
            kelvins = self.kelvin.draw_samples((nb_rows,), random_state=random_state)

            batch.images = change_color_temperatures_fn(
                batch.images, kelvins, from_colorspaces=self.from_colorspace
            )

        return batch

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.kelvin, self.from_colorspace]
