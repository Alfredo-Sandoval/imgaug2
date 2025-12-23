from __future__ import annotations

from typing import Any

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.compat.markers import legacy

from .base import ParamInput, StochasticParameter
from .discrete import Deterministic
from .handles import _assert_arg_is_stoch_param, handle_continuous_param, handle_probability_param
from .utils import both_np_float_if_one_is_float, force_np_float_dtype

class Clip(StochasticParameter):
    """Clip another parameter to a defined value range.

    Parameters
    ----------
    other_param : imgaug2.parameters.StochasticParameter
        The other parameter, which's values are to be clipped.

    minval : None or number, optional
        The minimum value to use.
        If ``None``, no minimum will be used.

    maxval : None or number, optional
        The maximum value to use.
        If ``None``, no maximum will be used.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Clip(Normal(0, 1.0), minval=-2.0, maxval=2.0)

    Create a standard gaussian distribution, which's values never go below
    ``-2.0`` or above ``2.0``. Note that this will lead to small "bumps" of
    higher probability at ``-2.0`` and ``2.0``, as values below/above these
    will be clipped to them. For smoother limitations on gaussian
    distributions, see `TruncatedNormal`.

    """

    def __init__(
        self,
        other_param: StochasticParameter,
        minval: float | None = None,
        maxval: float | None = None,
    ) -> None:
        super().__init__()

        _assert_arg_is_stoch_param("other_param", other_param)
        assert minval is None or ia.is_single_number(minval), (
            f"Expected 'minval' to be None or a number, got type {type(minval)}."
        )
        assert maxval is None or ia.is_single_number(maxval), (
            f"Expected 'maxval' to be None or a number, got type {type(maxval)}."
        )

        self.other_param = other_param
        self.minval = minval
        self.maxval = maxval

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        samples = self.other_param.draw_samples(size, random_state=random_state)
        if self.minval is not None or self.maxval is not None:
            # Note that this would produce a warning if 'samples' is int64
            # or uint64
            samples = np.clip(samples, self.minval, self.maxval, out=samples)
        return samples

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        opstr = str(self.other_param)
        if self.minval is not None and self.maxval is not None:
            return f"Clip({opstr}, {float(self.minval):.6f}, {float(self.maxval):.6f})"
        if self.minval is not None:
            return f"Clip({opstr}, {float(self.minval):.6f}, None)"
        if self.maxval is not None:
            return f"Clip({opstr}, None, {float(self.maxval):.6f})"
        return f"Clip({opstr}, None, None)"

@legacy
class Discretize(StochasticParameter):
    """Convert a continuous distribution to a discrete one.

    This will round the values and then cast them to integers.
    Values sampled from already discrete distributions are not changed.

    Parameters
    ----------
    other_param : imgaug2.parameters.StochasticParameter
        The other parameter, which's values are to be discretized.

    round : bool, optional
        Whether to round before converting to integer dtype.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Discretize(iap.Normal(0, 1.0))

    Create a discrete standard gaussian distribution.

    """

    def __init__(self, other_param: StochasticParameter, round: bool = True) -> None:
        super().__init__()
        _assert_arg_is_stoch_param("other_param", other_param)
        self.other_param = other_param
        self.round = round

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        samples = self.other_param.draw_samples(size, random_state=random_state)
        assert samples.dtype.kind in ["u", "i", "b", "f"], (
            "Expected to get uint, int, bool or float dtype as samples in "
            f"Discretize(), but got dtype '{samples.dtype.name}' (kind '{samples.dtype.kind}') instead."
        )

        if samples.dtype.kind in ["u", "i", "b"]:
            return samples

        # floats seem to reliably cover ints that have half the number of
        # bits -- probably not the case for float128 though as that is
        # really float96
        bitsize = 8 * samples.dtype.itemsize // 2
        # in case some weird system knows something like float8 we set a
        # lower bound here -- shouldn't happen though
        bitsize = max(bitsize, 8)
        dtype = np.dtype(f"int{bitsize:d}")
        if self.round:
            samples = np.round(samples)
        return samples.astype(dtype)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        opstr = str(self.other_param)
        return f"Discretize({opstr}, round={str(self.round)})"

@legacy
class Multiply(StochasticParameter):
    """Multiply the samples of another stochastic parameter.

    Parameters
    ----------
    other_param : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Other parameter which's sampled values are to be multiplied with `val`.
        Let ``S`` be the requested shape of samples, then the datatype
        behaviour is as follows:

            * If a `StochasticParameter`, that parameter will be
              queried once per call to generate an array of shape ``S``.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    val : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Multiplier to use.
        Datatype behaviour is analogous to `other_param`, though if
        ``elementwise=False`` (the default), only a single sample will be
        generated per call instead of ``S``.

    elementwise : bool, optional
        Controls the sampling behaviour of `val`.
        If set to ``False``, a single samples will be requested from `val` and
        used as the constant multiplier.
        If set to ``True``, samples of shape ``S`` will be requested from
        `val` and multiplied elementwise with the samples of `other_param`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Multiply(iap.Uniform(0.0, 1.0), -1)

    Convert a uniform distribution from ``[0.0, 1.0)`` to ``(-1.0, 0.0]``.

    """

    def __init__(
        self, other_param: ParamInput, val: ParamInput, elementwise: bool = False
    ) -> None:
        super().__init__()

        self.other_param = handle_continuous_param(other_param, "other_param", prefetch=False)
        self.val = handle_continuous_param(val, "val", prefetch=False)
        self.elementwise = elementwise

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        rngs = random_state.duplicate(2)
        samples = self.other_param.draw_samples(size, random_state=rngs[0])

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            val_samples = self.val.draw_samples(size, random_state=rngs[1])
        else:
            val_samples = self.val.draw_sample(random_state=rngs[1])

        if elementwise:
            return np.multiply(samples, val_samples)
        return samples * val_samples

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Multiply({str(self.other_param)}, {str(self.val)}, {self.elementwise})"

@legacy
class Divide(StochasticParameter):
    """Divide the samples of another stochastic parameter.

    This parameter will automatically prevent division by zero (uses 1.0)
    as the denominator in these cases.

    Parameters
    ----------
    other_param : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Other parameter which's sampled values are to be divided by `val`.
        Let ``S`` be the requested shape of samples, then the datatype
        behaviour is as follows:

            * If a `StochasticParameter`, that parameter will be
              queried once per call to generate an array of shape ``S``.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    val : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Denominator to use.
        Datatype behaviour is analogous to `other_param`, though if
        ``elementwise=False`` (the default), only a single sample will be
        generated per call instead of ``S``.

    elementwise : bool, optional
        Controls the sampling behaviour of `val`.
        If set to ``False``, a single samples will be requested from `val` and
        used as the constant denominator.
        If set to ``True``, samples of shape ``S`` will be requested from
        `val` and used to divide the samples of `other_param` elementwise.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Divide(iap.Uniform(0.0, 1.0), 2)

    Convert a uniform distribution ``[0.0, 1.0)`` to ``[0, 0.5)``.

    """

    def __init__(
        self, other_param: ParamInput, val: ParamInput, elementwise: bool = False
    ) -> None:
        super().__init__()

        self.other_param = handle_continuous_param(other_param, "other_param", prefetch=False)
        self.val = handle_continuous_param(val, "val", prefetch=False)
        self.elementwise = elementwise

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        rngs = random_state.duplicate(2)
        samples = self.other_param.draw_samples(size, random_state=rngs[0])

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            val_samples = self.val.draw_samples(size, random_state=rngs[1])

            # prevent division by zero
            val_samples[val_samples == 0] = 1

            return np.divide(force_np_float_dtype(samples), force_np_float_dtype(val_samples))
        else:
            val_sample = self.val.draw_sample(random_state=rngs[1])

            # prevent division by zero
            if val_sample == 0:
                val_sample = 1

            return force_np_float_dtype(samples) / float(val_sample)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Divide({str(self.other_param)}, {str(self.val)}, {self.elementwise})"

# TODO sampling (N,) from something like 10+Uniform(0, 1) will return
#      N times the same value as (N,) values will be sampled from 10, but only
#      one from Uniform() unless elementwise=True is explicitly set. That
#      seems unintuitive. How can this be prevented?
@legacy
class Add(StochasticParameter):
    """Add to the samples of another stochastic parameter.

    Parameters
    ----------
    other_param : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Samples of `val` will be added to samples of this parameter.
        Let ``S`` be the requested shape of samples, then the datatype
        behaviour is as follows:

            * If a `StochasticParameter`, that parameter will be
              queried once per call to generate an array of shape ``S``.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    val : number or tuple of two number or list of number or imgaug2.parameters.StochasticParameter
        Value to add to the samples of `other_param`.
        Datatype behaviour is analogous to `other_param`, though if
        ``elementwise=False`` (the default), only a single sample will be
        generated per call instead of ``S``.

    elementwise : bool, optional
        Controls the sampling behaviour of `val`.
        If set to ``False``, a single samples will be requested from `val` and
        used as the constant multiplier.
        If set to ``True``, samples of shape ``S`` will be requested from
        `val` and added elementwise with the samples of `other_param`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Add(Uniform(0.0, 1.0), 1.0)

    Convert a uniform distribution from ``[0.0, 1.0)`` to ``[1.0, 2.0)``.

    """

    def __init__(
        self, other_param: ParamInput, val: ParamInput, elementwise: bool = False
    ) -> None:
        super().__init__()

        self.other_param = handle_continuous_param(other_param, "other_param", prefetch=False)
        self.val = handle_continuous_param(val, "val", prefetch=False)
        self.elementwise = elementwise

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        rngs = random_state.duplicate(2)
        samples = self.other_param.draw_samples(size, random_state=rngs[0])

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            val_samples = self.val.draw_samples(size, random_state=rngs[1])
        else:
            val_samples = self.val.draw_sample(random_state=rngs[1])

        if elementwise:
            return np.add(samples, val_samples)
        return samples + val_samples

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Add({str(self.other_param)}, {str(self.val)}, {self.elementwise})"

@legacy
class Subtract(StochasticParameter):
    """Subtract from the samples of another stochastic parameter.

    Parameters
    ----------
    other_param : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Samples of `val` will be subtracted from samples of this parameter.
        Let ``S`` be the requested shape of samples, then the datatype
        behaviour is as follows:

            * If a `StochasticParameter`, that parameter will be
              queried once per call to generate an array of shape ``S``.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    val : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Value to subtract from the other parameter.
        Datatype behaviour is analogous to `other_param`, though if
        ``elementwise=False`` (the default), only a single sample will be
        generated per call instead of ``S``.

    elementwise : bool, optional
        Controls the sampling behaviour of `val`.
        If set to ``False``, a single samples will be requested from `val` and
        used as the constant multiplier.
        If set to ``True``, samples of shape ``S`` will be requested from
        `val` and subtracted elementwise from the samples of `other_param`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Subtract(iap.Uniform(0.0, 1.0), 1.0)

    Convert a uniform distribution from ``[0.0, 1.0)`` to ``[-1.0, 0.0)``.

    """

    def __init__(
        self, other_param: ParamInput, val: ParamInput, elementwise: bool = False
    ) -> None:
        super().__init__()

        self.other_param = handle_continuous_param(other_param, "other_param", prefetch=False)
        self.val = handle_continuous_param(val, "val", prefetch=False)
        self.elementwise = elementwise

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        rngs = random_state.duplicate(2)
        samples = self.other_param.draw_samples(size, random_state=rngs[0])

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            val_samples = self.val.draw_samples(size, random_state=rngs[1])
        else:
            val_samples = self.val.draw_sample(random_state=rngs[1])

        if elementwise:
            return np.subtract(samples, val_samples)
        return samples - val_samples

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Subtract({str(self.other_param)}, {str(self.val)}, {self.elementwise})"

@legacy
class Power(StochasticParameter):
    """Exponentiate the samples of another stochastic parameter.

    Parameters
    ----------
    other_param : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Other parameter which's sampled values are to be exponentiated by `val`.
        Let ``S`` be the requested shape of samples, then the datatype
        behaviour is as follows:

            * If a `StochasticParameter`, that parameter will be
              queried once per call to generate an array of shape ``S``.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    val : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Value to use exponentiate the samples of `other_param`.
        Datatype behaviour is analogous to `other_param`, though if
        ``elementwise=False`` (the default), only a single sample will be
        generated per call instead of ``S``.

    elementwise : bool, optional
        Controls the sampling behaviour of `val`.
        If set to ``False``, a single samples will be requested from `val` and
        used as the constant multiplier.
        If set to ``True``, samples of shape ``S`` will be requested from
        `val` and used to exponentiate elementwise the samples of `other_param`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Power(iap.Uniform(0.0, 1.0), 2)

    Converts a uniform range ``[0.0, 1.0)`` to a distribution that is peaked
    towards 1.0.

    """

    def __init__(
        self, other_param: ParamInput, val: ParamInput, elementwise: bool = False
    ) -> None:
        super().__init__()

        self.other_param = handle_continuous_param(other_param, "other_param", prefetch=False)
        self.val = handle_continuous_param(val, "val", prefetch=False)
        self.elementwise = elementwise

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        rngs = random_state.duplicate(2)
        samples = self.other_param.draw_samples(size, random_state=rngs[0])

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            exponents = self.val.draw_samples(size, random_state=rngs[1])
        else:
            exponents = self.val.draw_sample(random_state=rngs[1])

        # without this we get int results in the case of
        # Power(<int>, <stochastic float param>)
        samples, exponents = both_np_float_if_one_is_float(samples, exponents)
        samples_dtype = samples.dtype

        # Using complex128 to handle negative bases with fractional exponents
        result = np.power(samples.astype(np.complex128), exponents).real
        if result.dtype != samples_dtype:
            result = result.astype(samples_dtype)

        return result

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Power({str(self.other_param)}, {str(self.val)}, {self.elementwise})"

@legacy
class Absolute(StochasticParameter):
    """Convert the samples of another parameter to their absolute values.

    Parameters
    ----------
    other_param : imgaug2.parameters.StochasticParameter
        Other parameter which's sampled values are to be modified.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Absolute(iap.Uniform(-1.0, 1.0))

    Convert a uniform distribution from ``[-1.0, 1.0)`` to ``[0.0, 1.0]``.

    """

    def __init__(self, other_param: StochasticParameter) -> None:
        super().__init__()

        _assert_arg_is_stoch_param("other_param", other_param)

        self.other_param = other_param

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        samples = self.other_param.draw_samples(size, random_state=random_state)
        return np.absolute(samples)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        opstr = str(self.other_param)
        return f"Absolute({opstr})"

@legacy
class RandomSign(StochasticParameter):
    """Convert a parameter's samples randomly to positive or negative values.

    Parameters
    ----------
    other_param : imgaug2.parameters.StochasticParameter
        Other parameter which's sampled values are to be modified.

    p_positive : number
        Fraction of values that are supposed to be turned to positive values.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.RandomSign(iap.Poisson(1))

    Create a poisson distribution with ``alpha=1`` that is mirrored/copied (not
    flipped) at the y-axis.

    """

    def __init__(self, other_param: StochasticParameter, p_positive: float = 0.5) -> None:
        super().__init__()

        _assert_arg_is_stoch_param("other_param", other_param)
        assert ia.is_single_number(p_positive), (
            f"Expected 'p_positive' to be a number, got {type(p_positive)}."
        )
        assert 0.0 <= p_positive <= 1.0, (
            f"Expected 'p_positive' to be in the interval [0.0, 1.0], got {p_positive:.4f}."
        )

        self.other_param = other_param
        self.p_positive = p_positive

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        rss = random_state.duplicate(2)
        samples = self.other_param.draw_samples(size, random_state=rss[0])
        # TODO add method to change from uint to int here instead of assert
        assert samples.dtype.kind in ["f", "i"], (
            f"Expected to get samples of kind float or int, but got dtype {samples.dtype.name} "
            f"of kind {samples.dtype.kind}."
        )
        # TODO convert to same kind as samples
        coinflips = rss[1].binomial(1, self.p_positive, size=size).astype(np.int8)
        signs = coinflips * 2 - 1
        # Add absolute here to guarantee that we get p_positive percent of
        # positive values. Otherwise we would merely flip p_positive percent
        # of all signs.
        result = np.absolute(samples) * signs
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        opstr = str(self.other_param)
        return f"RandomSign({opstr}, {self.p_positive:.2f})"

@legacy
class ForceSign(StochasticParameter):
    """Convert a parameter's samples to either positive or negative values.

    Parameters
    ----------
    other_param : imgaug2.parameters.StochasticParameter
        Other parameter which's sampled values are to be modified.

    positive : bool
        Whether to force all signs to be positive (``True``) or
        negative (``False``).

    mode : {'invert', 'reroll'}, optional
        Method to change the signs. Valid values are ``invert`` and ``reroll``.
        ``invert`` means that wrong signs are simply flipped.
        ``reroll`` means that all samples with wrong signs are sampled again,
        optionally many times, until they randomly end up having the correct
        sign.

    reroll_count_max : int, optional
        If `mode` is set to ``reroll``, this determines how often values may
        be rerolled before giving up and simply flipping the sign (as in
        ``mode="invert"``). This shouldn't be set too high, as rerolling is
        expensive.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.ForceSign(iap.Poisson(1), positive=False)

    Create a poisson distribution with ``alpha=1`` that is flipped towards
    negative values.

    """

    def __init__(
        self,
        other_param: StochasticParameter,
        positive: bool,
        mode: str = "invert",
        reroll_count_max: int = 2,
    ) -> None:
        super().__init__()

        _assert_arg_is_stoch_param("other_param", other_param)
        self.other_param = other_param

        assert positive in [True, False], (
            f"Expected 'positive' to be True or False, got type {type(positive)}."
        )
        self.positive = positive

        assert mode in ["invert", "reroll"], (
            f"Expected 'mode' to be \"invert\" or \"reroll\", got {mode}."
        )
        self.mode = mode

        assert ia.is_single_integer(reroll_count_max), (
            f"Expected 'reroll_count_max' to be an integer, got type {type(reroll_count_max)}."
        )
        self.reroll_count_max = reroll_count_max

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        rngs = random_state.duplicate(1 + self.reroll_count_max)
        samples = self.other_param.draw_samples(size, random_state=rngs[0])

        if self.mode == "invert":
            if self.positive:
                samples[samples < 0] *= -1
            else:
                samples[samples > 0] *= -1
        else:
            if self.positive:
                bad_samples = np.where(samples < 0)[0]
            else:
                bad_samples = np.where(samples > 0)[0]

            reroll_count = 0
            while len(bad_samples) > 0 and reroll_count < self.reroll_count_max:
                # This rerolls the full input size, even when only a tiny
                # fraction of the values were wrong. That is done, because not
                # all parameters necessarily support any number of dimensions
                # for `size`, so we cant just resample size=N for N values
                # with wrong signs.
                # There is still quite some room for improvement here.
                samples_reroll = self.other_param.draw_samples(
                    size, random_state=rngs[1 + reroll_count]
                )
                samples[bad_samples] = samples_reroll[bad_samples]

                reroll_count += 1
                if self.positive:
                    bad_samples = np.where(samples < 0)[0]
                else:
                    bad_samples = np.where(samples > 0)[0]

            if len(bad_samples) > 0:
                samples[bad_samples] *= -1

        return samples

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        opstr = str(self.other_param)
        return f"ForceSign({opstr}, {self.positive!s}, {self.mode}, {self.reroll_count_max:d})"

@legacy
def Positive(
    other_param: StochasticParameter, mode: str = "invert", reroll_count_max: int = 2
) -> ForceSign:
    """Convert another parameter's results to positive values.

    Parameters
    ----------
    other_param : imgaug2.parameters.StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    mode : {'invert', 'reroll'}, optional
        How to change the signs. Valid values are ``invert`` and ``reroll``.
        ``invert`` means that wrong signs are simply flipped.
        ``reroll`` means that all samples with wrong signs are sampled again,
        optionally many times, until they randomly end up having the correct
        sign.

    reroll_count_max : int, optional
        If `mode` is set to ``reroll``, this determines how often values may
        be rerolled before giving up and simply flipping the sign (as in
        ``mode="invert"``). This shouldn't be set too high, as rerolling is
        expensive.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Positive(iap.Normal(0, 1), mode="reroll")

    Create a gaussian distribution that has only positive values.
    If any negative value is sampled in the process, that sample is resampled
    up to two times to get a positive one. If it isn't positive after the
    second resampling step, the sign is simply flipped.

    """
    return ForceSign(
        other_param=other_param, positive=True, mode=mode, reroll_count_max=reroll_count_max
    )

@legacy
def Negative(
    other_param: StochasticParameter, mode: str = "invert", reroll_count_max: int = 2
) -> ForceSign:
    """Convert another parameter's results to negative values.

    Parameters
    ----------
    other_param : imgaug2.parameters.StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    mode : {'invert', 'reroll'}, optional
        How to change the signs. Valid values are ``invert`` and ``reroll``.
        ``invert`` means that wrong signs are simply flipped.
        ``reroll`` means that all samples with wrong signs are sampled again,
        optionally many times, until they randomly end up having the correct
        sign.

    reroll_count_max : int, optional
        If `mode` is set to ``reroll``, this determines how often values may
        be rerolled before giving up and simply flipping the sign (as in
        ``mode=\"invert\"``). This shouldn't be set too high, as rerolling is
        expensive.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Negative(iap.Normal(0, 1), mode=\"reroll\")

    Create a gaussian distribution that has only negative values.
    If any positive value is sampled in the process, that sample is resampled
    up to two times to get a negative one. If it isn't negative after the
    second resampling step, the sign is simply flipped.

    """
    return ForceSign(
        other_param=other_param, positive=False, mode=mode, reroll_count_max=reroll_count_max
    )

@legacy
class Sigmoid(StochasticParameter):
    """Apply a sigmoid function to the outputs of another parameter.

    This is intended to be used in combination with `SimplexNoise` or
    `FrequencyNoise`. It pushes the noise values away from ``~0.5`` and
    towards ``0.0`` or ``1.0``, making the noise maps more binary.

    Parameters
    ----------
    other_param : imgaug2.parameters.StochasticParameter
        The other parameter to which the sigmoid will be applied.

    threshold : number or tuple of number or iterable of number or imgaug2.parameters.StochasticParameter, optional
        Sets the value of the sigmoid's saddle point, i.e. where values
        start to quickly shift from ``0.0`` to ``1.0``.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    activated : bool or number, optional
        Defines whether the sigmoid is activated. If this is ``False``, the
        results of `other_param` will not be altered. This may be set to a
        ``float`` ``p`` in value range``[0.0, 1.0]``, which will result in
        `activated` being ``True`` in ``p`` percent of all calls.

    mul : number, optional
        The results of `other_param` will be multiplied with this value before
        applying the sigmoid. For noise values (range ``[0.0, 1.0]``) this
        should be set to about ``20``.

    add : number, optional
        This value will be added to the results of `other_param` before
        applying the sigmoid. For noise values (range ``[0.0, 1.0]``) this
        should be set to about ``-10.0``, provided `mul` was set to ``20``.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Sigmoid(
    >>>     iap.SimplexNoise(),
    >>>     activated=0.5,
    >>>     mul=20,
    >>>     add=-10)

    Applies a sigmoid to simplex noise in ``50%`` of all calls. The noise
    results are modified to match the sigmoid's expected value range. The
    sigmoid's outputs are in the range ``[0.0, 1.0]``.

    """

    def __init__(
        self,
        other_param: StochasticParameter,
        threshold: ParamInput = (-10, 10),
        activated: ParamInput = True,
        mul: float = 1,
        add: float = 0,
    ) -> None:
        super().__init__()
        _assert_arg_is_stoch_param("other_param", other_param)
        self.other_param = other_param

        self.threshold = handle_continuous_param(threshold, "threshold", prefetch=False)
        self.activated = handle_probability_param(activated, "activated", prefetch=False)

        assert ia.is_single_number(mul), f"Expected 'mul' to be a number, got type {type(mul)}."
        assert mul > 0, f"Expected 'mul' to be greater than zero, got {mul:.4f}."
        self.mul = mul

        assert ia.is_single_number(add), f"Expected 'add' to be a number, got type {type(add)}."
        self.add = add

    @staticmethod
    def create_for_noise(
        other_param: StochasticParameter,
        threshold: ParamInput = (-10, 10),
        activated: ParamInput = True,
    ) -> Sigmoid:
        """Create a Sigmoid adjusted for noise parameters.

        "noise" here denotes `SimplexNoise` and `FrequencyNoise`.

        Parameters
        ----------
        other_param : imgaug2.parameters.StochasticParameter
            See `__init__()`.

        threshold : number or tuple of number or iterable of number or imgaug2.parameters.StochasticParameter, optional
            See `__init__()`.

        activated : bool or number, optional
            See `__init__()`.

        Returns
        -------
        Sigmoid
            A sigmoid adjusted to be used with noise.

        """
        return Sigmoid(other_param, threshold, activated, mul=20, add=-10)

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        rngs = random_state.duplicate(3)
        result = self.other_param.draw_samples(size, random_state=rngs[0])
        if result.dtype.kind != "f":
            result = result.astype(np.float32)
        activated = self.activated.draw_sample(random_state=rngs[1])
        threshold = self.threshold.draw_sample(random_state=rngs[2])
        if activated > 0.5:
            # threshold must be subtracted here, not added
            # higher threshold = move threshold of sigmoid towards the right
            #                  = make it harder to pass the threshold
            #                  = more 0.0s / less 1.0s
            # by subtracting a high value, it moves each x towards the left,
            # leading to more values being left of the threshold, leading
            # to more 0.0s
            return 1 / (1 + np.exp(-(result * self.mul + self.add - threshold)))
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        opstr = str(self.other_param)
        return f"Sigmoid({opstr}, {str(self.threshold)}, {str(self.activated)}, {str(self.mul)}, {str(self.add)})"
