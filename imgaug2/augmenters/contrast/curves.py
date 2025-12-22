"""Contrast curve functions and augmenters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import skimage.exposure as ski_exposure

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy
from imgaug2.mlx._core import is_mlx_array
import imgaug2.mlx.pointwise as mlx_pointwise

from ._types import ContrastFunc, DTypeStrs
from ._utils import _is_mlx_list

class _ContrastFuncWrapper(meta.Augmenter):
    def __init__(
        self,
        func: ContrastFunc,
        params1d: Sequence[iap.StochasticParameter],
        per_channel: ParamInput,
        dtypes_allowed: DTypeStrs | None = None,
        dtypes_disallowed: DTypeStrs | None = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
        func_mlx: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.func = func
        self.func_mlx = func_mlx
        self.params1d = params1d
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")
        self.dtypes_allowed = dtypes_allowed
        self.dtypes_disallowed = dtypes_disallowed

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

        if self.func_mlx is not None and (is_mlx_array(images) or _is_mlx_list(images)):
            # MLX Fast-path
            nb_images = len(images)
            rss = random_state.duplicate(1 + nb_images)
            per_channel = self.per_channel.draw_samples((nb_images,), random_state=rss[0])

            # Since MLX ops support broadcasting, we try to gather all params
            # and invoke the op once per image (or ideally batched if params allow).
            # Current structure samples params per image/channel.
            # To stay consistent with standard augmentation logic, we loop but keep data on GPU.

            gen = enumerate(zip(images, per_channel, rss[1:], strict=True))
            for i, (image, per_channel_i, rs) in gen:
                nb_channels = 1 if per_channel_i <= 0.5 else image.shape[2]
                samples_i = [
                    param.draw_samples((nb_channels,), random_state=rs) for param in self.params1d
                ]

                # If per_channel is active, we might have different params per channel.
                # MLX pointwise ops usually broadcast. If samples_i has shape (C,),
                # and image has (H, W, C), it will broadcast correctly.

                # samples_i is a list of arrays (one per parameter), each array is (nb_channels,)
                # We simply pass these arrays to the function.

                args = tuple([image] + samples_i)
                batch.images[i] = self.func_mlx(*args)

            return batch

        if self.dtypes_allowed is not None:
            iadt.gate_dtypes_strs(
                images,
                allowed=self.dtypes_allowed,
                disallowed=self.dtypes_disallowed,
                augmenter=self,
            )

        nb_images = len(images)
        rss = random_state.duplicate(1 + nb_images)
        per_channel = self.per_channel.draw_samples((nb_images,), random_state=rss[0])

        gen = enumerate(zip(images, per_channel, rss[1:], strict=True))
        for i, (image, per_channel_i, rs) in gen:
            nb_channels = 1 if per_channel_i <= 0.5 else image.shape[2]
            # TODO improve efficiency by sampling once
            samples_i = [
                param.draw_samples((nb_channels,), random_state=rs) for param in self.params1d
            ]
            if per_channel_i > 0.5:
                input_dtype = image.dtype
                # TODO This was previously a cast of image to float64. Do the
                #      adjust_* functions return float64?
                result = []
                for c in range(nb_channels):
                    samples_i_c = [sample_i[c] for sample_i in samples_i]
                    args = tuple([image[..., c]] + samples_i_c)
                    result.append(self.func(*args))
                image_aug = np.stack(result, axis=-1)
                image_aug = image_aug.astype(input_dtype)
            else:
                # don't use something like samples_i[...][0] here, because
                # that returns python scalars and is slightly less accurate
                # than keeping the numpy values
                args = tuple([image] + samples_i)
                image_aug = self.func(*args)
            batch.images[i] = image_aug
        return batch

    def get_parameters(self) -> list[iap.StochasticParameter]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return list(self.params1d)


def adjust_contrast_gamma(arr: Array, gamma: float) -> Array:
    """
    Adjust image contrast by scaling pixel values to ``255*((v/255)**gamma)``.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested (1) (2) (3)
        * ``uint16``: yes; tested (2) (3)
        * ``uint32``: yes; tested (2) (3)
        * ``uint64``: yes; tested (2) (3) (4)
        * ``int8``: limited; tested (2) (3) (5)
        * ``int16``: limited; tested (2) (3) (5)
        * ``int32``: limited; tested (2) (3) (5)
        * ``int64``: limited; tested (2) (3) (4) (5)
        * ``float16``: limited; tested (5)
        * ``float32``: limited; tested (5)
        * ``float64``: limited; tested (5)
        * ``float128``: no (6)
        * ``bool``: no (7)

        - (1) Handled by ``cv2``. Other dtypes are handled by ``skimage``.
        - (2) Normalization is done as ``I_ij/max``, where ``max`` is the
              maximum value of the dtype, e.g. 255 for ``uint8``. The
              normalization is reversed afterwards, e.g. ``result*255`` for
              ``uint8``.
        - (3) Integer-like values are not rounded after applying the contrast
              adjustment equation (before inverting the normalization to
              ``[0.0, 1.0]`` space), i.e. projection from continuous
              space to discrete happens according to floor function.
        - (4) Note that scikit-image doc says that integers are converted to
              ``float64`` values before applying the contrast normalization
              method. This might lead to inaccuracies for large 64bit integer
              values. Tests showed no indication of that happening though.
        - (5) Must not contain negative values. Values >=0 are fully supported.
        - (6) Leads to error in scikit-image.
        - (7) Does not make sense for contrast adjustments.

    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to adjust the contrast. Dtype ``uint8`` is fastest.

    gamma : number
        Exponent for the contrast adjustment. Higher values darken the image.

    Returns
    -------
    numpy.ndarray
        Array with adjusted contrast.

    """
    if arr.size == 0:
        return np.copy(arr)

    # int8 is also possible according to docs
    # https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#cv2.LUT ,
    # but here it seemed like `d` was 0 for CV_8S, causing that to fail
    if arr.dtype == iadt._UINT8_DTYPE:
        min_value, _center_value, max_value = iadt.get_value_range_of_dtype(arr.dtype)
        dynamic_range = max_value - min_value

        value_range = np.linspace(0, 1.0, num=dynamic_range + 1, dtype=np.float32)

        # 255 * ((I_ij/255)**gamma)
        # using np.float32(.) here still works when the input is a numpy array
        # of size 1
        table = min_value + (value_range ** np.float32(gamma)) * dynamic_range
        table = np.clip(table, min_value, max_value).astype(arr.dtype)
        arr_aug = ia.apply_lut(arr, table)
        return arr_aug
    return ski_exposure.adjust_gamma(arr, gamma)


def adjust_contrast_sigmoid(arr: Array, gain: float, cutoff: float, inv: bool = False) -> Array:
    """
    Adjust image contrast to ``255*1/(1+exp(gain*(cutoff-I_ij/255)))``.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested (1) (2) (3)
        * ``uint16``: yes; tested (2) (3)
        * ``uint32``: yes; tested (2) (3)
        * ``uint64``: yes; tested (2) (3) (4)
        * ``int8``: limited; tested (2) (3) (5)
        * ``int16``: limited; tested (2) (3) (5)
        * ``int32``: limited; tested (2) (3) (5)
        * ``int64``: limited; tested (2) (3) (4) (5)
        * ``float16``: limited; tested (5)
        * ``float32``: limited; tested (5)
        * ``float64``: limited; tested (5)
        * ``float128``: no (6)
        * ``bool``: no (7)

        - (1) Handled by ``cv2``. Other dtypes are handled by ``skimage``.
        - (2) Normalization is done as ``I_ij/max``, where ``max`` is the
              maximum value of the dtype, e.g. 255 for ``uint8``. The
              normalization is reversed afterwards, e.g. ``result*255``
              for ``uint8``.
        - (3) Integer-like values are not rounded after applying the contrast
              adjustment equation before inverting the normalization
              to ``[0.0, 1.0]`` space), i.e. projection from continuous
              space to discrete happens according to floor function.
        - (4) Note that scikit-image doc says that integers are converted to
              ``float64`` values before applying the contrast normalization
              method. This might lead to inaccuracies for large 64bit integer
              values. Tests showed no indication of that happening though.
        - (5) Must not contain negative values. Values >=0 are fully supported.
        - (6) Leads to error in scikit-image.
        - (7) Does not make sense for contrast adjustments.

    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to adjust the contrast. Dtype ``uint8`` is fastest.

    gain : number
        Multiplier for the sigmoid function's output.
        Higher values lead to quicker changes from dark to light pixels.

    cutoff : number
        Cutoff that shifts the sigmoid function in horizontal direction.
        Higher values mean that the switch from dark to light pixels happens
        later, i.e. the pixels will remain darker.

    inv : bool
        Whether to invert the sigmoid correction.

    Returns
    -------
    numpy.ndarray
        Array with adjusted contrast.

    """
    inv = bool(np.asarray(inv).item() > 0.5)
    if arr.size == 0:
        return np.copy(arr)

    # int8 is also possible according to docs
    # https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#cv2.LUT ,
    # but here it seemed like `d` was 0 for CV_8S, causing that to fail
    if arr.dtype == iadt._UINT8_DTYPE:
        min_value, _center_value, max_value = iadt.get_value_range_of_dtype(arr.dtype)
        dynamic_range = max_value - min_value

        value_range = np.linspace(0, 1.0, num=dynamic_range + 1, dtype=np.float32)

        # 255 * 1/(1 + exp(gain*(cutoff - I_ij/255)))
        # using np.float32(.) here still works when the input is a numpy array
        # of size 1
        gain = np.float32(gain)
        cutoff = np.float32(cutoff)
        table = min_value + dynamic_range * 1 / (1 + np.exp(gain * (cutoff - value_range)))
        if inv:
            table = min_value + dynamic_range - table
        table = np.clip(table, min_value, max_value).astype(arr.dtype)
        arr_aug = ia.apply_lut(arr, table)
        return arr_aug
    return ski_exposure.adjust_sigmoid(arr, cutoff=cutoff, gain=gain, inv=inv)


def adjust_contrast_log(arr: Array, gain: float, inv: bool = False) -> Array:
    """
    Adjust image contrast by scaling pixels to ``255*gain*log_2(1+v/255)``.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested (1) (2) (3)
        * ``uint16``: yes; tested (2) (3)
        * ``uint32``: no; tested (2) (3) (8)
        * ``uint64``: no; tested (2) (3) (4) (8)
        * ``int8``: limited; tested (2) (3) (5)
        * ``int16``: limited; tested (2) (3) (5)
        * ``int32``: no; tested (2) (3) (5) (8)
        * ``int64``: no; tested (2) (3) (4) (5) (8)
        * ``float16``: limited; tested (5)
        * ``float32``: limited; tested (5)
        * ``float64``: limited; tested (5)
        * ``float128``: no (6)
        * ``bool``: no (7)

        - (1) Handled by ``cv2``. Other dtypes are handled by ``skimage``.
        - (2) Normalization is done as ``I_ij/max``, where ``max`` is the
              maximum value of the dtype, e.g. 255 for ``uint8``. The
              normalization is reversed afterwards, e.g. ``result*255`` for
              ``uint8``.
        - (3) Integer-like values are not rounded after applying the contrast
              adjustment equation (before inverting the normalization
              to ``[0.0, 1.0]`` space), i.e. projection from continuous
              space to discrete happens according to floor function.
        - (4) Note that scikit-image doc says that integers are converted to
              ``float64`` values before applying the contrast normalization
              method. This might lead to inaccuracies for large 64bit integer
              values. Tests showed no indication of that happening though.
        - (5) Must not contain negative values. Values >=0 are fully supported.
        - (6) Leads to error in scikit-image.
        - (7) Does not make sense for contrast adjustments.
        - (8) No longer supported since numpy 1.17. Previously: 'yes' for
              ``uint32``, ``uint64``; 'limited' for ``int32``, ``int64``.

    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to adjust the contrast. Dtype ``uint8`` is fastest.

    gain : number
        Multiplier for the logarithm result. Values around 1.0 lead to a
        contrast-adjusted images. Values above 1.0 quickly lead to partially
        broken images due to exceeding the datatype's value range.

    inv : bool
        Whether to invert the logarithmic correction.

    Returns
    -------
    numpy.ndarray
        Array with adjusted contrast.

    """
    inv = bool(np.asarray(inv).item() > 0.5)
    if arr.size == 0:
        return np.copy(arr)

    # int8 is also possible according to docs
    # https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#cv2.LUT ,
    # but here it seemed like `d` was 0 for CV_8S, causing that to fail
    if arr.dtype == iadt._UINT8_DTYPE:
        min_value, _center_value, max_value = iadt.get_value_range_of_dtype(arr.dtype)
        dynamic_range = max_value - min_value

        value_range = np.linspace(0, 1.0, num=dynamic_range + 1, dtype=np.float32)

        # 255 * gain * log2(1 + I_ij/255)
        # using np.float32(.) here still works when the input is a numpy array
        # of size 1
        gain = np.float32(gain)
        if inv:
            table = min_value + dynamic_range * gain * (2**value_range - 1)
        else:
            table = min_value + dynamic_range * gain * np.log2(1 + value_range)
        table = np.clip(table, min_value, max_value).astype(arr.dtype)
        arr_aug = ia.apply_lut(arr, table)
        return arr_aug
    return ski_exposure.adjust_log(arr, gain=gain, inv=inv)


def adjust_contrast_linear(arr: Array, alpha: float) -> Array:
    """Adjust contrast by scaling each pixel to ``127 + alpha*(v-127)``.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested (1) (2)
        * ``uint16``: yes; tested (2)
        * ``uint32``: yes; tested (2)
        * ``uint64``: no (3)
        * ``int8``: yes; tested (2)
        * ``int16``: yes; tested (2)
        * ``int32``: yes; tested (2)
        * ``int64``: no (2)
        * ``float16``: yes; tested (2)
        * ``float32``: yes; tested (2)
        * ``float64``: yes; tested (2)
        * ``float128``: no (2)
        * ``bool``: no (4)

        - (1) Handled by ``cv2``. Other dtypes are handled by raw ``numpy``.
        - (2) Only tested for reasonable alphas with up to a value of
              around ``100``.
        - (3) Conversion to ``float64`` is done during augmentation, hence
              ``uint64``, ``int64``, and ``float128`` support cannot be
              guaranteed.
        - (4) Does not make sense for contrast adjustments.

    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to adjust the contrast. Dtype ``uint8`` is fastest.

    alpha : number
        Multiplier to linearly pronounce (``>1.0``), dampen (``0.0`` to
        ``1.0``) or invert (``<0.0``) the difference between each pixel value
        and the dtype's center value, e.g. ``127`` for ``uint8``.

    Returns
    -------
    numpy.ndarray
        Array with adjusted contrast.

    """
    if arr.size == 0:
        return np.copy(arr)

    # int8 is also possible according to docs
    # https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#cv2.LUT ,
    # but here it seemed like `d` was 0 for CV_8S, causing that to fail
    if arr.dtype == iadt._UINT8_DTYPE:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(arr.dtype)
        center_value = int(center_value)

        value_range = np.arange(0, 256, dtype=np.float32)

        # 127 + alpha*(I_ij-127)
        # using np.float32(.) here still works when the input is a numpy array
        # of size 1
        alpha = np.float32(alpha)
        table = center_value + alpha * (value_range - center_value)
        table = np.clip(table, min_value, max_value).astype(arr.dtype)
        arr_aug = ia.apply_lut(arr, table)
        return arr_aug
    else:
        input_dtype = arr.dtype
        _min_value, center_value, _max_value = iadt.get_value_range_of_dtype(input_dtype)
        if input_dtype.kind in ["u", "i"]:
            center_value = int(center_value)
        image_aug = center_value + alpha * (arr.astype(np.float64) - center_value)
        image_aug = iadt.restore_dtypes_(image_aug, input_dtype)
        return image_aug


class GammaContrast(_ContrastFuncWrapper):
    """
    Adjust image contrast by scaling pixel values to ``255*((v/255)**gamma)``.

    Values in the range ``gamma=(0.5, 2.0)`` seem to be sensible.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.contrast.adjust_contrast_gamma`.

    Parameters
    ----------
    gamma : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Exponent for the contrast adjustment. Higher values darken the image.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the range ``[a, b]``
              will be used per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (``False``) or to
        sample a new value for each channel (``True``). If this value is a
        float ``p``, then for ``p`` percent of all images `per_channel` will
        be treated as ``True``, otherwise as ``False``.

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
    >>> aug = iaa.GammaContrast((0.5, 2.0))

    Modify the contrast of images according to ``255*((v/255)**gamma)``,
    where ``v`` is a pixel value and ``gamma`` is sampled uniformly from
    the interval ``[0.5, 2.0]`` (once per image).

    >>> aug = iaa.GammaContrast((0.5, 2.0), per_channel=True)

    Same as in the previous example, but ``gamma`` is sampled once per image
    *and* channel.

    """

    def __init__(
        self,
        gamma: ParamInput = (0.7, 1.7),
        per_channel: ParamInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        params1d = [
            iap.handle_continuous_param(
                gamma, "gamma", value_range=None, tuple_to_uniform=True, list_to_choice=True
            )
        ]
        import imgaug2.augmenters.contrast as contrastlib
        func = contrastlib.adjust_contrast_gamma
        super().__init__(
            func,
            params1d,
            per_channel,
            dtypes_allowed="uint8 uint16 uint32 uint64 int8 int16 int32 int64 "
            "float16 float32 float64",
            dtypes_disallowed="float128 bool",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
            func_mlx=mlx_pointwise.gamma_contrast,
        )


class SigmoidContrast(_ContrastFuncWrapper):
    """
    Adjust image contrast to ``255*1/(1+exp(gain*(cutoff-I_ij/255)))``.

    Values in the range ``gain=(5, 20)`` and ``cutoff=(0.25, 0.75)`` seem to
    be sensible.

    A combination of ``gain=5.5`` and ``cutof=0.45`` is fairly close to
    the identity function.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.contrast.adjust_contrast_sigmoid`.

    Parameters
    ----------
    gain : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Multiplier for the sigmoid function's output.
        Higher values lead to quicker changes from dark to light pixels.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the interval
              ``[a, b]`` will be sampled uniformly per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    cutoff : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Cutoff that shifts the sigmoid function in horizontal direction.
        Higher values mean that the switch from dark to light pixels happens
        later, i.e. the pixels will remain darker.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the range ``[a, b]``
              will be used per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    inv : bool or float or imgaug2.parameters.StochasticParameter, optional
        Whether to invert the sigmoid correction. If this value is a float
        ``p``, then for ``p`` percent of all images `inv` will be treated as
        ``True``, otherwise as ``False``.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (``False``) or to
        sample a new value for each channel (``True``). If this value is a
        float ``p``, then for ``p`` percent of all images `per_channel` will
        be treated as ``True``, otherwise as ``False``.

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
    >>> aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))

    Modify the contrast of images according to
    ``255*1/(1+exp(gain*(cutoff-v/255)))``, where ``v`` is a pixel value,
    ``gain`` is sampled uniformly from the interval ``[3, 10]`` (once per
    image) and ``cutoff`` is sampled uniformly from the interval
    ``[0.4, 0.6]`` (also once per image).

    >>> aug = iaa.SigmoidContrast(
    >>>     gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)

    Same as in the previous example, but ``gain`` and ``cutoff`` are each
    sampled once per image *and* channel.

    """

    def __init__(
        self,
        gain: ParamInput = (5, 6),
        cutoff: ParamInput = (0.3, 0.6),
        inv: ParamInput = False,
        per_channel: ParamInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        params1d = [
            iap.handle_continuous_param(
                gain, "gain", value_range=(0, None), tuple_to_uniform=True, list_to_choice=True
            ),
            iap.handle_continuous_param(
                cutoff, "cutoff", value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True
            ),
            iap.handle_probability_param(inv, "inv"),
        ]
        import imgaug2.augmenters.contrast as contrastlib
        func = contrastlib.adjust_contrast_sigmoid

        super().__init__(
            func,
            params1d,
            per_channel,
            dtypes_allowed="uint8 uint16 uint32 uint64 int8 int16 int32 int64 "
            "float16 float32 float64",
            dtypes_disallowed="float128 bool",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
            func_mlx=mlx_pointwise.sigmoid_contrast,
        )


class LogContrast(_ContrastFuncWrapper):
    """Adjust image contrast by scaling pixels to ``255*gain*log_2(1+v/255)``.

    This augmenter is fairly similar to
    ``imgaug2.augmenters.arithmetic.Multiply``.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.contrast.adjust_contrast_log`.

    Parameters
    ----------
    gain : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Multiplier for the logarithm result. Values around ``1.0`` lead to a
        contrast-adjusted images. Values above ``1.0`` quickly lead to
        partially broken images due to exceeding the datatype's value range.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
              will uniformly sampled be used per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    inv : bool or float or imgaug2.parameters.StochasticParameter, optional
        Whether to invert the logarithmic correction. If this value is a
        float ``p``, then for ``p`` percent of all images `inv` will be
        treated as ``True``, otherwise as ``False``.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (``False``) or to
        sample a new value for each channel (``True``). If this value is a
        float ``p``, then for ``p`` percent of all images `per_channel` will
        be treated as ``True``, otherwise as ``False``.

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
    >>> aug = iaa.LogContrast(gain=(0.6, 1.4))

    Modify the contrast of images according to ``255*gain*log_2(1+v/255)``,
    where ``v`` is a pixel value and ``gain`` is sampled uniformly from the
    interval ``[0.6, 1.4]`` (once per image).

    >>> aug = iaa.LogContrast(gain=(0.6, 1.4), per_channel=True)

    Same as in the previous example, but ``gain`` is sampled once per image
    *and* channel.

    """

    def __init__(
        self,
        gain: ParamInput = (0.4, 1.6),
        inv: ParamInput = False,
        per_channel: ParamInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        params1d = [
            iap.handle_continuous_param(
                gain, "gain", value_range=(0, None), tuple_to_uniform=True, list_to_choice=True
            ),
            iap.handle_probability_param(inv, "inv"),
        ]
        import imgaug2.augmenters.contrast as contrastlib
        func = contrastlib.adjust_contrast_log

        super().__init__(
            func,
            params1d,
            per_channel,
            dtypes_allowed="uint8 uint16 uint32 uint64 int8 int16 int32 int64 "
            "float16 float32 float64",
            dtypes_disallowed="float128 bool",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
            func_mlx=mlx_pointwise.log_contrast,
        )


class LinearContrast(_ContrastFuncWrapper):
    """Adjust contrast by scaling each pixel to ``127 + alpha*(v-127)``.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.contrast.adjust_contrast_linear`.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Multiplier to linearly pronounce (``>1.0``), dampen (``0.0`` to
        ``1.0``) or invert (``<0.0``) the difference between each pixel value
        and the dtype's center value, e.g. ``127`` for ``uint8``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
              will be used per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (``False``) or to
        sample a new value for each channel (``True``). If this value is a
        float ``p``, then for ``p`` percent of all images `per_channel` will
        be treated as ``True``, otherwise as ``False``.

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
    >>> aug = iaa.LinearContrast((0.4, 1.6))

    Modify the contrast of images according to `127 + alpha*(v-127)``,
    where ``v`` is a pixel value and ``alpha`` is sampled uniformly from the
    interval ``[0.4, 1.6]`` (once per image).

    >>> aug = iaa.LinearContrast((0.4, 1.6), per_channel=True)

    Same as in the previous example, but ``alpha`` is sampled once per image
    *and* channel.

    """

    def __init__(
        self,
        alpha: ParamInput = (0.6, 1.4),
        per_channel: ParamInput = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        params1d = [
            iap.handle_continuous_param(
                alpha, "alpha", value_range=None, tuple_to_uniform=True, list_to_choice=True
            )
        ]
        import imgaug2.augmenters.contrast as contrastlib
        func = contrastlib.adjust_contrast_linear

        super().__init__(
            func,
            params1d,
            per_channel,
            dtypes_allowed="uint8 uint16 uint32 int8 int16 int32 float16 float32 float64",
            dtypes_disallowed="uint64 int64 float128 bool",
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
            func_mlx=mlx_pointwise.linear_contrast,
        )


