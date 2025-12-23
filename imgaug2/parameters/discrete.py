from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.compat.markers import legacy

from .base import Numberish, ParamInput, StochasticParameter
from .handles import handle_continuous_param, handle_discrete_param

@legacy
class Deterministic(StochasticParameter):
    """Parameter that is a constant value.

    If ``N`` values are sampled from this parameter, it will return ``N`` times
    ``V``, where ``V`` is the constant value.

    Parameters
    ----------
    value : number or str or imgaug2.parameters.StochasticParameter
        A constant value to use.
        A string may be provided to generate arrays of strings.
        If this is a StochasticParameter, a single value will be sampled
        from it exactly once and then used as the constant value.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Deterministic(10)
    >>> param.draw_sample()
    np.int32(10)

    Will always sample the value 10.

    """

    def __init__(self, value: float | int | str | StochasticParameter) -> None:
        super().__init__()

        if isinstance(value, StochasticParameter):
            self.value = value.draw_sample()
        elif ia.is_single_number(value) or ia.is_string(value):
            self.value = value
        else:
            raise Exception(
                f"Expected StochasticParameter object or number or string, got {type(value)}."
            )

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        kwargs = {}
        if ia.is_single_integer(self.value):
            kwargs = {"dtype": np.int32}
        elif ia.is_single_float(self.value):
            kwargs = {"dtype": np.float32}
        return np.full(size, self.value, **kwargs)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if ia.is_single_integer(self.value):
            return f"Deterministic(int {self.value:d})"
        if ia.is_single_float(self.value):
            return f"Deterministic(float {self.value:.8f})"
        return f"Deterministic({str(self.value)})"

# TODO replace two-value parameters used in tests with this
@legacy(version="0.4.0")
class DeterministicList(StochasticParameter):
    """Parameter that repeats elements from a list in the given order.

    E.g. of samples of shape ``(A, B, C)`` are requested, this parameter will
    return the first ``A*B*C`` elements, reshaped to ``(A, B, C)`` from the
    provided list. If the list contains less than ``A*B*C`` elements, it
    will (by default) be tiled until it is long enough (i.e. the sampling
    will start again at the first element, if necessary multiple times).

    Parameters
    ----------
    values : ndarray or iterable of number
        An iterable of values to sample from in the order within the iterable.

    """

    @legacy(version="0.4.0")
    def __init__(self, values: NDArray | Iterable[Numberish]) -> None:
        super().__init__()

        assert ia.is_iterable(values), (
            f"Expected to get an iterable as input, got type {type(values).__name__}."
        )
        assert len(values) > 0, "Expected to get at least one value, got zero."

        if ia.is_np_array(values):
            # this would not be able to handle e.g. [[1, 2], [3]] and output
            # dtype object due to the non-regular shape, hence we have the
            # else block
            values_arr = np.asarray(values)
            self.values = values_arr.flatten()
        else:
            self.values = np.array(list(ia.flatten(values)))
            kind = self.values.dtype.kind

            # limit to 32bit instead of 64bit for efficiency
            if kind == "i":
                self.values = self.values.astype(np.int32)
            elif kind == "f":
                self.values = self.values.astype(np.float32)

    @legacy(version="0.4.0")
    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        nb_requested = int(np.prod(size))
        values = self.values
        if nb_requested > self.values.size:
            # we don't use itertools.cycle() here, as that would require
            # running through a loop potentially many times (as `size` can
            # be very large), which would be slow
            multiplier = int(np.ceil(nb_requested / values.size))
            values = np.tile(values, (multiplier,))
        return values[:nb_requested].reshape(size)

    @legacy(version="0.4.0")
    def __repr__(self) -> str:
        return self.__str__()

    @legacy(version="0.4.0")
    def __str__(self) -> str:
        if self.values.dtype.kind == "f":
            values = [f"{value:.4f}" for value in self.values]
            return "DeterministicList([{}])".format(", ".join(values))
        return f"DeterministicList({str(self.values.tolist())})"

@legacy
class Choice(StochasticParameter):
    """Parameter that samples value from a list of allowed values.

    Parameters
    ----------
    a : iterable
        List of allowed values.
        Usually expected to be ``int`` s, ``float`` s or ``str`` s.
        May also contain ``StochasticParameter`` s. Each
        ``StochasticParameter`` that is randomly picked will automatically be
        replaced by a sample of itself (or by ``N`` samples if the parameter
        was picked ``N`` times).

    replace : bool, optional
        Whether to perform sampling with or without replacing.

    p : None or iterable of number, optional
        Probabilities of each element in `a`.
        Must have the same length as `a` (if provided).

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Choice([5, 17, 25], p=[0.25, 0.5, 0.25])
    >>> sample = param.draw_sample()
    >>> assert sample in [5, 17, 25]

    Create and sample from a parameter, which will produce with ``50%``
    probability the sample ``17`` and in the other ``50%`` of all cases the
    sample ``5`` or ``25``..

    """

    def __init__(
        self,
        a: Iterable[Any],
        replace: bool = True,
        p: Iterable[float] | None = None,
    ) -> None:
        super().__init__()

        assert ia.is_iterable(a), f"Expected a to be an iterable (e.g. list), got {type(a)}."
        self.a = a
        self.replace = replace
        if p is not None:
            assert ia.is_iterable(p), f"Expected p to be None or an iterable, got {type(p)}."
            assert len(p) == len(a), (
                f"Expected lengths of a and p to be identical, got {len(a)} and {len(p)}."
            )
        self.p = p

    @property
    @legacy(version="0.5.0")
    def prefetchable(self) -> bool:
        """See `prefetchable()`."""
        return self.replace

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        if any([isinstance(a_i, StochasticParameter) for a_i in self.a]):
            rngs = random_state.duplicate(1 + len(self.a))
            samples = rngs[0].choice(self.a, np.prod(size), replace=self.replace, p=self.p)

            # collect the sampled parameters and how many samples must be taken
            # from each of them
            params_counter = defaultdict(lambda: 0)
            for sample in samples:
                if isinstance(sample, StochasticParameter):
                    key = str(sample)
                    params_counter[key] += 1

            # collect per parameter once the required number of samples
            # iterate here over self.a to always use the same seed for
            # the same parameter
            # TODO this might fail if the same parameter is added multiple
            #      times to self.a?
            # TODO this will fail if a parameter cant handle size=(N,)
            param_to_samples = dict()
            for i, param in enumerate(self.a):
                key = str(param)
                if key in params_counter:
                    param_to_samples[key] = param.draw_samples(
                        size=(params_counter[key],), random_state=rngs[1 + i]
                    )

            # assign the values sampled from the parameters to the `samples`
            # array by replacing the respective parameter
            param_to_readcount = defaultdict(lambda: 0)
            for i, sample in enumerate(samples):
                if isinstance(sample, StochasticParameter):
                    key = str(sample)
                    readcount = param_to_readcount[key]
                    samples[i] = param_to_samples[key][readcount]
                    param_to_readcount[key] += 1

            samples = samples.reshape(size)
        else:
            samples = random_state.choice(self.a, size, replace=self.replace, p=self.p)

        dtype = samples.dtype
        if dtype.itemsize * 8 > 32:
            # strings have kind "U"
            kind = dtype.kind
            if kind == "i":
                samples = samples.astype(np.int32)
            elif kind == "u":
                samples = samples.astype(np.uint32)
            elif kind == "f":
                samples = samples.astype(np.float32)

        return samples

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Choice(a={str(self.a)}, replace={str(self.replace)}, p={str(self.p)})"

@legacy
class Binomial(StochasticParameter):
    """Binomial distribution.

    Parameters
    ----------
    p : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Probability of the binomial distribution. Expected to be in the
        interval ``[0.0, 1.0]``.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Binomial(Uniform(0.01, 0.2))

    Create a binomial distribution that uses a varying probability between
    ``0.01`` and ``0.2``, randomly and uniformly estimated once per sampling
    call.

    """

    def __init__(self, p: ParamInput) -> None:
        super().__init__()
        self.p = handle_continuous_param(p, "p")

    @property
    @legacy(version="0.5.0")
    def prefetchable(self) -> bool:
        """See `prefetchable()`."""
        return True

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        p = self.p.draw_sample(random_state=random_state)
        assert 0 <= p <= 1.0, (
            f"Expected probability p to be in the interval [0.0, 1.0], got {p:.4f}."
        )
        return random_state.binomial(1, p, size).astype(np.int32)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Binomial({self.p})"

@legacy
class DiscreteUniform(StochasticParameter):
    """Uniform distribution over the discrete interval ``[a..b]``.

    Parameters
    ----------
    a : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter
        Lower bound of the interval.
        If ``a>b``, `a` and `b` will automatically be flipped.
        If ``a==b``, all generated values will be identical to `a`.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    b : int or imgaug2.parameters.StochasticParameter
        Upper bound of the interval. Analogous to `a`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.DiscreteUniform(10, Choice([20, 30, 40]))
    >>> sample = param.draw_sample()
    >>> assert 10 <= sample <= 40

    Create a discrete uniform distribution which's interval differs between
    calls and can be ``[10..20]``, ``[10..30]`` or ``[10..40]``.

    """

    def __init__(self, a: ParamInput, b: ParamInput) -> None:
        super().__init__()

        self.a = handle_discrete_param(a, "a")
        self.b = handle_discrete_param(b, "b")

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
            return np.full(size, a, dtype=np.int32)
        return random_state.integers(a, b + 1, size, dtype=np.int32)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"DiscreteUniform({self.a}, {self.b})"

@legacy
class Poisson(StochasticParameter):
    """Parameter that resembles a poisson distribution.

    A poisson distribution with ``lambda=0`` has its highest probability at
    point ``0`` and decreases quickly from there.
    Poisson distributions are discrete and never negative.

    Parameters
    ----------
    lam : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter
        Lambda parameter of the poisson distribution.

            * If a `StochasticParameter`, that parameter will be
              queried once per call.

        "per call" denotes a call of `draw_sample()` or
        `draw_samples()`.

    Examples
    --------
    >>> import imgaug2.parameters as iap
    >>> param = iap.Poisson(1)
    >>> sample = param.draw_sample()
    >>> assert sample >= 0

    Create a poisson distribution with ``lambda=1`` and sample a value from
    it.

    """

    def __init__(self, lam: ParamInput) -> None:
        super().__init__()

        self.lam = handle_continuous_param(lam, "lam")

    @property
    @legacy(version="0.5.0")
    def prefetchable(self) -> bool:
        """See `prefetchable()`."""
        return True

    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        lam = self.lam.draw_sample(random_state=random_state)
        lam = max(lam, 0)

        return random_state.poisson(lam=lam, size=size).astype(np.int32)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Poisson({self.lam})"
