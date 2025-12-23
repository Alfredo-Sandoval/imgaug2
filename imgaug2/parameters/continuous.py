from __future__ import annotations

from typing import Any

import numpy as np
import scipy
import scipy.stats

import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.compat.markers import legacy

from .base import ParamInput, StochasticParameter
from .handles import handle_continuous_param, handle_discrete_param

class Normal(StochasticParameter):
    """Parameter that resembles a normal/gaussian distribution.

    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        The mean of the normal distribution.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    scale : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        The standard deviation of the normal distribution.
        If this parameter reaches ``0``, the output array will be filled with
        `loc`.
        Datatype behaviour is the analogous to `loc`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Normal(Choice([-1.0, 1.0]), 1.0)

    Create a gaussian distribution with a mean that differs by call.
    Samples values may sometimes follow ``N(-1.0, 1.0)`` and sometimes
    ``N(1.0, 1.0)``.

    """

    def __init__(self, loc: ParamInput, scale: ParamInput) -> None:
        super().__init__()

        self.loc = handle_continuous_param(loc, "loc")
        self.scale = handle_continuous_param(scale, "scale", value_range=(0, None))

    @property
    @legacy(version="0.5.0")
    def prefetchable(self) -> bool:
        """See `prefetchable()`."""
        return True

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        loc = self.loc.draw_sample(random_state=random_state)
        scale = self.scale.draw_sample(random_state=random_state)
        assert scale >= 0, f"Expected scale to be >=0, got {scale:.4f}."
        if scale == 0:
            return np.full(size, loc, dtype=np.float32)
        return random_state.normal(loc, scale, size=size).astype(np.float32)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Normal(loc={self.loc}, scale={self.scale})"

# TODO docstring for parameters is outdated
@legacy
class TruncatedNormal(StochasticParameter):
    """Parameter that resembles a truncated normal distribution.

    A truncated normal distribution is similar to a normal distribution,
    except the domain is smoothly bounded to a min and max value.

    This is a wrapper around `truncnorm()`.

    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        The mean of the normal distribution.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    scale : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        The standard deviation of the normal distribution.
        If this parameter reaches ``0``, the output array will be filled with
        `loc`.
        Datatype behaviour is the same as for `loc`.

    low : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        The minimum value of the truncated normal distribution.
        Datatype behaviour is the same as for `loc`.

    high : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        The maximum value of the truncated normal distribution.
        Datatype behaviour is the same as for `loc`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.TruncatedNormal(0, 5.0, low=-10, high=10)
    >>> samples = param.draw_samples(100, random_state=0)
    >>> assert np.all(samples >= -10)
    >>> assert np.all(samples <= 10)

    Create a truncated normal distribution with its minimum at ``-10.0``
    and its maximum at ``10.0``.

    """

    def __init__(
        self,
        loc: ParamInput,
        scale: ParamInput,
        low: ParamInput = -np.inf,
        high: ParamInput = np.inf,
    ) -> None:
        super().__init__()

        self.loc = handle_continuous_param(loc, "loc")
        self.scale = handle_continuous_param(scale, "scale", value_range=(0, None))
        self.low = handle_continuous_param(low, "low")
        self.high = handle_continuous_param(high, "high")

    @property
    @legacy(version="0.5.0")
    def prefetchable(self) -> bool:
        """See `prefetchable()`."""
        return True

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        loc = self.loc.draw_sample(random_state=random_state)
        scale = self.scale.draw_sample(random_state=random_state)
        low = self.low.draw_sample(random_state=random_state)
        high = self.high.draw_sample(random_state=random_state)
        seed = random_state.generate_seed_()
        if low > high:
            low, high = high, low
        assert scale >= 0, f"Expected scale to be >=0, got {scale:.4f}."
        if scale == 0:
            return np.full(size, fill_value=loc, dtype=np.float32)
        a = (low - loc) / scale
        b = (high - loc) / scale
        tnorm = scipy.stats.truncnorm(a=a, b=b, loc=loc, scale=scale)

        # Using a seed here works with both np.random interfaces.
        # Last time tried, scipy crashed when providing just
        # random_state.generator on the new np.random interface.
        return tnorm.rvs(size=size, random_state=seed).astype(np.float32)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"TruncatedNormal(loc={self.loc}, scale={self.scale}, low={self.low}, high={self.high})"
        )

@legacy
class Laplace(StochasticParameter):
    """Parameter that resembles a (continuous) laplace distribution.

    This is a wrapper around numpy's `laplace()`.

    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        The position of the distribution peak, similar to the mean in normal
        distributions.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    scale : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        The exponential decay factor, similar to the standard deviation in
        gaussian distributions.
        If this parameter reaches ``0``, the output array will be filled with
        `loc`.
        Datatype behaviour is the analogous to `loc`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Laplace(0, 1.0)

    Create a laplace distribution, which's peak is at ``0`` and decay is
    ``1.0``.

    """

    def __init__(self, loc: ParamInput, scale: ParamInput) -> None:
        super().__init__()

        self.loc = handle_continuous_param(loc, "loc")
        self.scale = handle_continuous_param(scale, "scale", value_range=(0, None))

    @property
    @legacy(version="0.5.0")
    def prefetchable(self) -> bool:
        """See `prefetchable()`."""
        return True

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        loc = self.loc.draw_sample(random_state=random_state)
        scale = self.scale.draw_sample(random_state=random_state)
        assert scale >= 0, f"Expected scale to be >=0, got {scale}."
        if scale == 0:
            return np.full(size, loc, dtype=np.float32)
        return random_state.laplace(loc, scale, size=size).astype(np.float32)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Laplace(loc={self.loc}, scale={self.scale})"

@legacy
class ChiSquare(StochasticParameter):
    """Parameter that resembles a (continuous) chi-square distribution.

    This is a wrapper around numpy's `chisquare()`.

    Parameters
    ----------
    df : int or tuple of two int or list of int or imgaug2.parameters.StochasticParameter
        Degrees of freedom. Expected value range is ``[1, inf)``.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.ChiSquare(df=2)

    Create a chi-square distribution with two degrees of freedom.

    """

    def __init__(self, df: ParamInput) -> None:
        super().__init__()

        self.df = handle_discrete_param(df, "df", value_range=(1, None))

    @property
    @legacy(version="0.5.0")
    def prefetchable(self) -> bool:
        """See `prefetchable()`."""
        return True

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        df = self.df.draw_sample(random_state=random_state)
        assert df >= 1, f"Expected df to be >=1, got {df:d}."
        return random_state.chisquare(df, size=size).astype(np.float32)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"ChiSquare(df={self.df})"

@legacy
class Weibull(StochasticParameter):
    """
    Parameter that resembles a (continuous) weibull distribution.

    This is a wrapper around numpy's `weibull()`.

    Parameters
    ----------
    a : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Shape parameter of the distribution.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Weibull(a=0.5)

    Create a weibull distribution with shape 0.5.

    """

    def __init__(self, a: ParamInput) -> None:
        super().__init__()

        self.a = handle_continuous_param(a, "a", value_range=(0.0001, None))

    @property
    @legacy(version="0.5.0")
    def prefetchable(self) -> bool:
        """See `prefetchable()`."""
        return True

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        a = self.a.draw_sample(random_state=random_state)
        assert a > 0, f"Expected a to be >0, got {a:.4f}."
        return random_state.weibull(a, size=size).astype(np.float32)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Weibull(a={self.a})"

# TODO rename (a, b) to (low, high) as in numpy?
@legacy
class Uniform(StochasticParameter):
    """Parameter that resembles a uniform distribution over ``[a, b)``.

    Parameters
    ----------
    a : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Lower bound of the interval.
        If ``a>b``, `a` and `b` will automatically be flipped.
        If ``a==b``, all generated values will be identical to `a`.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    b : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Upper bound of the interval. Analogous to `a`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Uniform(0, 10.0)
    >>> sample = param.draw_sample()
    >>> assert 0 <= sample < 10.0

    Create and sample from a uniform distribution over ``[0, 10.0)``.

    """

    def __init__(self, a: ParamInput, b: ParamInput) -> None:
        super().__init__()

        self.a = handle_continuous_param(a, "a")
        self.b = handle_continuous_param(b, "b")

    @property
    @legacy(version="0.5.0")
    def prefetchable(self) -> bool:
        """See `prefetchable()`."""
        return True

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        a = self.a.draw_sample(random_state=random_state)
        b = self.b.draw_sample(random_state=random_state)
        if a > b:
            a, b = b, a
        elif a == b:
            return np.full(size, a, dtype=np.float32)
        return random_state.uniform(a, b, size).astype(np.float32)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Uniform({self.a}, {self.b})"

@legacy
class Beta(StochasticParameter):
    """Parameter that resembles a (continuous) beta distribution.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        alpha parameter of the beta distribution.
        Expected value range is ``(0, inf)``. Values below ``0`` are
        automatically clipped to ``0+epsilon``.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    beta : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Beta parameter of the beta distribution. Analogous to `alpha`.

    epsilon : number
        Clipping parameter. If `alpha` or `beta` end up ``<=0``, they are clipped to ``0+epsilon``.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Beta(0.4, 0.6)

    Create a beta distribution with ``alpha=0.4`` and ``beta=0.6``.

    """

    def __init__(
        self, alpha: ParamInput, beta: ParamInput, epsilon: float = 0.0001
    ) -> None:
        super().__init__()

        self.alpha = handle_continuous_param(alpha, "alpha")
        self.beta = handle_continuous_param(beta, "beta")

        assert ia.is_single_number(epsilon), (
            f"Expected epsilon to a number, got type {type(epsilon)}."
        )
        self.epsilon = epsilon

    @property
    @legacy(version="0.5.0")
    def prefetchable(self) -> bool:
        """See `prefetchable()`."""
        return True

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        alpha = self.alpha.draw_sample(random_state=random_state)
        beta = self.beta.draw_sample(random_state=random_state)
        alpha = max(alpha, self.epsilon)
        beta = max(beta, self.epsilon)
        return random_state.beta(alpha, beta, size=size).astype(np.float32)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Beta({self.alpha}, {self.beta})"
