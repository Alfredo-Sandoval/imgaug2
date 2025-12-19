"""Random helpers and RNG wrapper for imgaug2.

Unifies NumPy's Generator/RandomState APIs and provides helpers for
seeding and state management.
"""

from __future__ import annotations

import builtins
import copy as copylib
from collections.abc import Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from imgaug2.compat.markers import legacy

if TYPE_CHECKING:
    from numpy._typing import _ShapeLike

_T = TypeVar("_T", bound=np.generic)

Size: TypeAlias = int | tuple[int, ...] | None
ShapeLike: TypeAlias = "int | tuple[int, ...] | _ShapeLike"
GeneratorState: TypeAlias = dict[str, object] | tuple[object, ...]
NumpyGenerator: TypeAlias = np.random.Generator | np.random.RandomState

# State tuple type for legacy RandomState
RandomStateState: TypeAlias = tuple[str, NDArray[np.uint32], int, int, float]

FloatArray: TypeAlias = NDArray[np.floating[Any]]
IntArray: TypeAlias = NDArray[np.integer[Any]]
GenericArray: TypeAlias = NDArray[np.generic]

# Detect which NumPy RNG API is available (Generator vs RandomState).
# We only parse major/minor; dev/build suffixes are ignored.
SUPPORTS_NEW_NP_RNG_STYLE: bool = False
BIT_GENERATOR: type[Any] | None = None
_NP_VERSION: list[int] = list(map(int, np.__version__.split(".")[0:2]))
if _NP_VERSION[0] > 1 or _NP_VERSION[1] >= 17:
    SUPPORTS_NEW_NP_RNG_STYLE = True
    BIT_GENERATOR = np.random.SFC64

    # BitGenerator interface location differs between NumPy 1.17 and 1.18+.
    if _NP_VERSION[1] == 17:
        import numpy.random.bit_generator as _bit_gen_module  # type: ignore[import-not-found]

        _BIT_GENERATOR_INTERFACE: type[Any] = _bit_gen_module.BitGenerator
    else:
        _BIT_GENERATOR_INTERFACE = np.random.BitGenerator

# Global RNG instance (lazy-initialized).
GLOBAL_RNG: RNG | None = None

# Use 2**31-1 as max; 2**31 fails on some platforms.
SEED_MIN_VALUE: int = 0
SEED_MAX_VALUE: int = 2**31 - 1

_RNG_IDX: int = 1

# TODO add 'with resetted_rng(...)'
# TODO change random_state to rng or seed


class RNG:
    """Unified RNG wrapper for NumPy Generator/RandomState.

    Accepts a seed, an existing RNG, or a NumPy RNG instance and exposes a
    stable API across NumPy versions.
    """

    # TODO add maybe a __new__ here that feeds-through an RNG input without
    #      wrapping it in RNG(rng_input)?
    def __init__(self, generator: RNGInput) -> None:
        global _RNG_IDX

        if isinstance(generator, RNG):
            self.generator = generator.generator
        else:
            self.generator = normalize_generator_(generator)
        self._is_new_rng_style = not isinstance(self.generator, np.random.RandomState)

        # Unique id used by AutoPrefetcher; state-based ids are too costly.
        self._idx = _RNG_IDX
        _RNG_IDX += 1

    @property
    def state(self) -> GeneratorState:
        """Get the state of this RNG.

        Returns:
            The state of the RNG (tuple or dict).
            Returns a copy; in-place changes to the return value will not affect the RNG.
        """
        return get_generator_state(self.generator)

    @state.setter
    def state(self, value: GeneratorState) -> None:
        """Set the state of the RNG in-place.

        Parameters:
            value: The new state of the RNG (tuple or dict), matching the format from `state`.
        """
        self.set_state_(value)

    def set_state_(self, value: GeneratorState) -> RNG:
        """Set the state of the RNG in-place.

        Parameters:
            value: The new state of the RNG.

        Returns:
            The RNG instance itself.
        """
        set_generator_state_(self.generator, value)
        return self

    def use_state_of_(self, other: RNG) -> RNG:
        """Copy and use (in-place) the state of another RNG.

        Note:
            Ensure neither RNG is the global RNG to avoid unexpected side effects.

        Parameters:
            other: The other RNG whose state will be copied.

        Returns:
            The RNG instance itself.
        """
        return self.set_state_(other.state)

    def is_global_rng(self) -> bool:
        """Check if this RNG wraps the same generator as the global RNG.

        Returns:
            `True` if the underlying generator is identical to the global one; `False` otherwise.
        """
        # We use .generator here, because otherwise RNG(global_rng) would be
        # viewed as not-identical to the global RNG, even though its generator
        # and bit generator are identical.
        return get_global_rng().generator is self.generator

    def equals_global_rng(self) -> bool:
        """Check if this RNG shares the same state as the global RNG.

        Returns:
            `True` if standard sampling produces the same values as the global RNG; `False` otherwise.
        """
        return get_global_rng().equals(self)

    def generate_seed_(self) -> int:
        """Sample a random seed and advance the generator.

        Returns:
            The sampled seed (int).
        """
        return generate_seed_(self.generator)

    def generate_seeds_(self, n: int) -> IntArray:
        """Generate `n` random seed values, advancing the generator state.

        Parameters:
            n: Number of seeds to sample.

        Returns:
            1D array of `int32` seeds.
        """
        return generate_seeds_(self.generator, n)

    def reset_cache_(self) -> RNG:
        """Reset all cache of this RNG.

        Returns
        -------
        RNG
            The RNG itself.

        """
        reset_generator_cache_(self.generator)
        return self

    def derive_rng_(self) -> RNG:
        """Create a child RNG.

        This advances the underlying generator's state.

        Returns
        -------
        RNG
            A child RNG.

        """
        return self.derive_rngs_(1)[0]

    def derive_rngs_(self, n: int) -> list[RNG]:
        """Create `n` child RNGs, advancing the generator state.

        Parameters:
            n: Number of child RNGs to derive.

        Returns:
            List of new `RNG` instances.
        """
        return [RNG(gen) for gen in derive_generators_(self.generator, n)]

    def equals(self, other: RNG) -> bool:
        """Estimate whether this RNG and `other` have the same state.

        Returns
        -------
        bool
            ``True`` if this RNG's generator and the generator of `other`
            have equal internal states. ``False`` otherwise.

        """
        assert isinstance(other, RNG), (
            f"Expected 'other' to be an RNG, got type {type(other)}. "
            "Use imgaug2.random.is_generator_equal_to() to compare "
            "numpy generators or RandomStates."
        )
        return is_generator_equal_to(self.generator, other.generator)

    def advance_(self) -> RNG:
        """Advance the RNG's internal state in-place by one step.

        Note:
            This advances the generator by sampling a value. For drastic state changes,
            consider deriving a new RNG.

        Returns:
            The RNG instance itself.
        """
        advance_generator_(self.generator)
        return self

    def copy(self) -> RNG:
        """Create a copy of this RNG.

        Returns
        -------
        RNG
            Copy of this RNG. The copy will produce the same random samples.

        """
        return RNG(copy_generator(self.generator))

    def copy_unless_global_rng(self) -> RNG:
        """Create a copy of this RNG unless it is the global RNG.

        Returns
        -------
        RNG
            Copy of this RNG unless it is the global RNG. In the latter case
            the RNG instance itself will be returned without any changes.

        """
        if self.is_global_rng():
            return self
        return self.copy()

    def duplicate(self, n: int) -> list[RNG]:
        """Create a list containing `n` references to this RNG.

        Note:
            This returns the *same* RNG instance `n` times, not copies.
            Used primarily as a placeholder for deprecated child derivation calls.

        Parameters:
            n: Length of the output list.

        Returns:
            List containing `n` references to this RNG.
        """
        return [self for _ in range(n)]

    @classmethod
    def create_fully_random(cls) -> RNG:
        """Create a new RNG seeded with OS entropy.

        Returns:
            A new independent `RNG` instance.
        """
        return RNG(create_fully_random_generator())

    @classmethod
    def create_pseudo_random_(cls) -> RNG:
        """Create a new RNG derived from the global RNG.

        Advances the global RNG state.

        Returns:
            A new `RNG` instance derived from the global generator.
        """
        return get_global_rng().derive_rng_()

    @classmethod
    def create_if_not_rng_(cls, generator: RNGInput) -> RNG:
        """Create a new RNG from a generator input, returning existing RNGs unchanged.

        Parameters:
            generator: The valid RNG input (see `__init__`).

        Returns:
            The input if it is already an `RNG`, otherwise a new `RNG` wrapper.
        """
        if isinstance(generator, RNG):
            return generator
        return RNG(generator)

    ###########################################################################
    # Below:
    #   Aliases for methods of numpy.random.Generator functions
    #
    # The methods below could also be handled with less code using some magic
    # methods. Explicitly writing things down here has the advantage that
    # the methods actually appear in the autogenerated API.
    ###########################################################################

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: Size = None,
        dtype: DTypeLike = "int32",
        endpoint: bool = False,
    ) -> int | IntArray:
        """Sample integers from [low, high).

        Wraps `numpy.random.Generator.integers` (or `randint`).

        Note:
            Default `dtype` is `int32` (unlike NumPy's `int64`).
        """
        return polyfill_integers(
            self.generator, low=low, high=high, size=size, dtype=dtype, endpoint=endpoint
        )

    def random(
        self,
        size: Size,
        dtype: DTypeLike = "float32",
        out: FloatArray | None = None,
    ) -> float | FloatArray:
        """Sample random floats in [0.0, 1.0).

        Wraps `numpy.random.Generator.random` (or `random_sample`).

        Note:
            Default `dtype` is `float32` (unlike NumPy's `float64`).
        """
        return polyfill_random(self.generator, size=size, dtype=dtype, out=out)

    # TODO add support for Generator's 'axis' argument
    def choice(
        self,
        a: int | ArrayLike,
        size: Size = None,
        replace: bool = True,
        p: ArrayLike | None = None,
    ) -> Any:  # noqa: ANN401
        """Generate a random sample from a given 1-D array.

        Wraps `numpy.random.Generator.choice`.
        """
        return self.generator.choice(a=a, size=size, replace=replace, p=p)  # type: ignore[call-overload]

    def bytes(self, length: int) -> builtins.bytes:
        """Return random bytes.

        Wraps `numpy.random.Generator.bytes`.
        """
        return self.generator.bytes(length=length)

    # TODO mark in-place
    def shuffle(self, x: ArrayLike) -> None:
        """Modify a sequence in-place by shuffling its contents.

        Wraps `numpy.random.Generator.shuffle`.
        """
        # note that shuffle() does not allow keyword arguments
        # note that shuffle() works in-place
        self.generator.shuffle(x)  # type: ignore[arg-type]

    def permutation(self, x: int | ArrayLike) -> GenericArray:
        """Randomly permute a sequence, or return a permuted range.

        Wraps `numpy.random.Generator.permutation`.
        """
        # note that permutation() does not allow keyword arguments
        return self.generator.permutation(x)  # type: ignore[call-overload, return-value]

    def beta(self, a: float, b: float, size: Size = None) -> float | FloatArray:
        """Draw samples from a Beta distribution."""
        return self.generator.beta(a=a, b=b, size=size)

    def binomial(self, n: int, p: float, size: Size = None) -> int | IntArray:
        """Draw samples from a Binomial distribution."""
        return self.generator.binomial(n=n, p=p, size=size)

    def chisquare(self, df: float, size: Size = None) -> float | FloatArray:
        """Draw samples from a Chi-square distribution."""
        return self.generator.chisquare(df=df, size=size)

    def dirichlet(self, alpha: Sequence[float] | FloatArray, size: Size = None) -> FloatArray:
        """Draw samples from the Dirichlet distribution."""
        return self.generator.dirichlet(alpha=alpha, size=size)

    def exponential(self, scale: float = 1.0, size: Size = None) -> float | FloatArray:
        """Draw samples from an Exponential distribution."""
        return self.generator.exponential(scale=scale, size=size)

    def f(self, dfnum: float, dfden: float, size: Size = None) -> float | FloatArray:
        """Draw samples from an F distribution."""
        return self.generator.f(dfnum=dfnum, dfden=dfden, size=size)

    def gamma(self, shape: float, scale: float = 1.0, size: Size = None) -> float | FloatArray:
        """Draw samples from a Gamma distribution."""
        return self.generator.gamma(shape=shape, scale=scale, size=size)

    def geometric(self, p: float, size: Size = None) -> int | IntArray:
        """Draw samples from the Geometric distribution."""
        return self.generator.geometric(p=p, size=size)

    def gumbel(self, loc: float = 0.0, scale: float = 1.0, size: Size = None) -> float | FloatArray:
        """Draw samples from a Gumbel distribution."""
        return self.generator.gumbel(loc=loc, scale=scale, size=size)

    def hypergeometric(
        self, ngood: int, nbad: int, nsample: int, size: Size = None
    ) -> int | IntArray:
        """Draw samples from a Hypergeometric distribution."""
        return self.generator.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=size)

    def laplace(
        self, loc: float = 0.0, scale: float = 1.0, size: Size = None
    ) -> float | FloatArray:
        """Draw samples from the Laplace or double exponential distribution."""
        return self.generator.laplace(loc=loc, scale=scale, size=size)

    def logistic(
        self, loc: float = 0.0, scale: float = 1.0, size: Size = None
    ) -> float | FloatArray:
        """Draw samples from a Logistic distribution."""
        return self.generator.logistic(loc=loc, scale=scale, size=size)

    def lognormal(
        self, mean: float = 0.0, sigma: float = 1.0, size: Size = None
    ) -> float | FloatArray:
        """Draw samples from a Lognormal distribution."""
        return self.generator.lognormal(mean=mean, sigma=sigma, size=size)

    def logseries(self, p: float, size: Size = None) -> int | IntArray:
        """Draw samples from a Log Series distribution."""
        return self.generator.logseries(p=p, size=size)

    def multinomial(
        self, n: int, pvals: Sequence[float] | FloatArray, size: Size = None
    ) -> IntArray:
        """Draw samples from a Multinomial distribution."""
        return self.generator.multinomial(n=n, pvals=pvals, size=size)

    def multivariate_normal(
        self,
        mean: ArrayLike,
        cov: ArrayLike,
        size: Size = None,
        check_valid: str = "warn",
        tol: float = 1e-8,
    ) -> FloatArray:
        """Draw samples from a Multivariate Normal distribution."""
        return self.generator.multivariate_normal(
            mean=mean,
            cov=cov,
            size=size,
            check_valid=check_valid,
            tol=tol,  # type: ignore[arg-type]
        )

    def negative_binomial(self, n: int, p: float, size: Size = None) -> int | IntArray:
        """Draw samples from a Negative Binomial distribution."""
        return self.generator.negative_binomial(n=n, p=p, size=size)

    def noncentral_chisquare(self, df: float, nonc: float, size: Size = None) -> float | FloatArray:
        """Draw samples from a Noncentral Chi-square distribution."""
        return self.generator.noncentral_chisquare(df=df, nonc=nonc, size=size)

    def noncentral_f(
        self, dfnum: float, dfden: float, nonc: float, size: Size = None
    ) -> float | FloatArray:
        """Draw samples from a Noncentral F distribution."""
        return self.generator.noncentral_f(dfnum=dfnum, dfden=dfden, nonc=nonc, size=size)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Size = None) -> float | FloatArray:
        """Draw random samples from a Normal (Gaussian) distribution."""
        return self.generator.normal(loc=loc, scale=scale, size=size)

    def pareto(self, a: float, size: Size = None) -> float | FloatArray:
        """Draw samples from a Pareto II or Lomax distribution."""
        return self.generator.pareto(a=a, size=size)

    def poisson(self, lam: float = 1.0, size: Size = None) -> int | IntArray:
        """Draw samples from a Poisson distribution."""
        return self.generator.poisson(lam=lam, size=size)

    def power(self, a: float, size: Size = None) -> float | FloatArray:
        """Draw samples in [0, 1] from a Power distribution with positive exponent a - 1."""
        return self.generator.power(a=a, size=size)

    def rayleigh(self, scale: float = 1.0, size: Size = None) -> float | FloatArray:
        """Draw samples from a Rayleigh distribution."""
        return self.generator.rayleigh(scale=scale, size=size)

    def standard_cauchy(self, size: Size = None) -> float | FloatArray:
        """Draw samples from a standard Cauchy distribution with mode=0."""
        return self.generator.standard_cauchy(size=size)

    def standard_exponential(
        self,
        size: Size = None,
        dtype: DTypeLike = "float32",
        method: str = "zig",
        out: FloatArray | None = None,
    ) -> float | FloatArray:
        """Draw samples from the standard exponential distribution.

        Note:
            Default `dtype` is `float32`.
        """
        if self._is_new_rng_style:
            return self.generator.standard_exponential(
                size=size,
                dtype=dtype,
                method=method,
                out=out,  # type: ignore[arg-type]
            )
        raw_result = self.generator.standard_exponential(size=size)
        if isinstance(raw_result, np.ndarray):
            result: float | FloatArray = raw_result.astype(dtype)
        else:
            result = raw_result
        if out is not None and isinstance(result, np.ndarray):
            assert out.dtype.name == result.dtype.name, (
                "Expected out array to have the same dtype as "
                f"standard_exponential()'s result array. Got {out.dtype.name} (out) and "
                f"{result.dtype.name} (result) instead."
            )
            out[...] = result
        return result

    def standard_gamma(
        self,
        shape: float,
        size: Size = None,
        dtype: DTypeLike = "float32",
        out: FloatArray | None = None,
    ) -> float | FloatArray:
        """Draw samples from a standard Gamma distribution.

        Note:
            Default `dtype` is `float32`.
        """
        if self._is_new_rng_style:
            return self.generator.standard_gamma(shape=shape, size=size, dtype=dtype, out=out)  # type: ignore[call-overload, return-value]
        result = self.generator.standard_gamma(shape=shape, size=size).astype(dtype)
        if out is not None:
            assert out.dtype.name == result.dtype.name, (
                "Expected out array to have the same dtype as "
                f"standard_gamma()'s result array. Got {out.dtype.name} (out) and "
                f"{result.dtype.name} (result) instead."
            )
            out[...] = result
        return result

    def standard_normal(
        self,
        size: Size = None,
        dtype: DTypeLike = "float32",
        out: FloatArray | None = None,
    ) -> float | FloatArray:
        """Draw samples from a standard Normal distribution (mean=0, stdev=1).

        Note:
            Default `dtype` is `float32`.
        """
        if self._is_new_rng_style:
            return self.generator.standard_normal(size=size, dtype=dtype, out=out)  # type: ignore[call-overload, return-value]
        raw_result = self.generator.standard_normal(size=size)
        if isinstance(raw_result, np.ndarray):
            result: float | FloatArray = raw_result.astype(dtype)
        else:
            result = raw_result
        if out is not None and isinstance(result, np.ndarray):
            assert out.dtype.name == result.dtype.name, (
                "Expected out array to have the same dtype as "
                f"standard_normal()'s result array. Got {out.dtype.name} (out) and "
                f"{result.dtype.name} (result) instead."
            )
            out[...] = result
        return result

    def standard_t(self, df: float, size: Size = None) -> float | FloatArray:
        """Draw samples from a standard Student's t distribution with `df` degrees of freedom."""
        return self.generator.standard_t(df=df, size=size)

    def triangular(
        self, left: float, mode: float, right: float, size: Size = None
    ) -> float | FloatArray:
        """Draw samples from a Triangular distribution over the interval [left, right]."""
        return self.generator.triangular(left=left, mode=mode, right=right, size=size)

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Size = None) -> float | FloatArray:
        """Draw samples from a Uniform distribution over [low, high)."""
        return self.generator.uniform(low=low, high=high, size=size)

    def vonmises(self, mu: float, kappa: float, size: Size = None) -> float | FloatArray:
        """Draw samples from a von Mises distribution."""
        return self.generator.vonmises(mu=mu, kappa=kappa, size=size)

    def wald(self, mean: float, scale: float, size: Size = None) -> float | FloatArray:
        """Draw samples from a Wald (inverse Gaussian) distribution."""
        return self.generator.wald(mean=mean, scale=scale, size=size)

    def weibull(self, a: float, size: Size = None) -> float | FloatArray:
        """Draw samples from a Weibull distribution."""
        return self.generator.weibull(a=a, size=size)

    def zipf(self, a: float, size: Size = None) -> int | IntArray:
        """Draw samples from a Zipf distribution."""
        return self.generator.zipf(a=a, size=size)

    ##################################################################
    # Outdated methods from RandomState
    # These are added here for backwards compatibility in case of old
    # custom augmenters and Lambda calls that rely on the RandomState
    # API.
    ##################################################################

    @legacy(version="0.4.0")
    def rand(self, *args: int) -> float | FloatArray:
        """Call :func:`numpy.random.RandomState.rand`.

        .. warning::

            This method is outdated in numpy. Use :func:`RNG.random` instead.


        """
        return self.random(size=args)

    @legacy(version="0.4.0")
    def randint(
        self,
        low: int,
        high: int | None = None,
        size: Size = None,
        dtype: DTypeLike = "int32",
    ) -> int | IntArray:
        """Call :func:`numpy.random.RandomState.randint`.

        .. note::

            Changed `dtype` argument default value from numpy's ``I`` to
            ``int32``.

        .. warning::

            This method is outdated in numpy. Use :func:`RNG.integers`
            instead.


        """
        return self.integers(low=low, high=high, size=size, dtype=dtype, endpoint=False)

    @legacy(version="0.4.0")
    def randn(self, *args: int) -> float | FloatArray:
        """Call :func:`numpy.random.RandomState.randn`.

        .. warning::

            This method is outdated in numpy. Use :func:`RNG.standard_normal`
            instead.


        """
        return self.standard_normal(size=args)

    @legacy(version="0.4.0")
    def random_integers(
        self, low: int, high: int | None = None, size: Size = None
    ) -> int | IntArray:
        """Call :func:`numpy.random.RandomState.random_integers`.

        .. warning::

            This method is outdated in numpy. Use :func:`RNG.integers`
            instead.


        """
        if high is None:
            return self.integers(low=1, high=low, size=size, endpoint=True)
        return self.integers(low=low, high=high, size=size, endpoint=True)

    @legacy(version="0.4.0")
    def random_sample(self, size: Size) -> float | FloatArray:
        """Call :func:`numpy.random.RandomState.random_sample`.

        .. warning::

            This method is outdated in numpy. Use :func:`RNG.uniform`
            instead.


        """
        return self.uniform(0.0, 1.0, size=size)

    @legacy(version="0.4.0")
    def tomaxint(self, size: Size = None) -> int | IntArray:
        """Call :func:`numpy.random.RandomState.tomaxint`.

        .. warning::

            This method is outdated in numpy. Use :func:`RNG.integers`
            instead.


        """
        import sys

        maxint = sys.maxsize
        int32max = np.iinfo(np.int32).max
        return self.integers(0, min(maxint, int32max), size=size, endpoint=True)


def supports_new_numpy_rng_style() -> bool:
    """
    Determine whether numpy supports the new ``random`` interface (v1.17+).

    Returns
    -------
    bool
        ``True`` if the new ``random`` interface is supported by numpy, i.e.
        if numpy has version 1.17 or later. Otherwise ``False``, i.e.
        numpy has version 1.16 or older and ``numpy.random.RandomState``
        should be used instead.

    """
    return SUPPORTS_NEW_NP_RNG_STYLE


def get_global_rng() -> RNG:
    """
    Get or create the current global RNG of imgaug2.

    Note that the first call to this function will create a global RNG.

    Returns
    -------
    RNG
        The global RNG to use.

    """
    # TODO change global_rng to singleton
    global GLOBAL_RNG
    if GLOBAL_RNG is None:
        # This uses numpy's random state to sample a seed.
        # Alternatively, `secrets.randbits(n_bits)` (3.6+) and
        # `os.urandom(n_bytes)` could be used.
        # See https://stackoverflow.com/a/27286733/3760780
        # for an explanation how random.seed() picks a random seed value.
        # np.random has randint method like RandomState, so use polyfill directly
        seed_ = polyfill_integers(np.random.mtrand._rand, SEED_MIN_VALUE, SEED_MAX_VALUE, size=(1,))  # type: ignore[arg-type]
        assert isinstance(seed_, np.ndarray)
        seed_val = int(seed_[0])

        GLOBAL_RNG = RNG(convert_seed_to_generator(seed_val))
    return GLOBAL_RNG


# This is an in-place operation, but does not use a trailing slash to indicate
# that in order to match the interface of `random` and `numpy.random`.
def seed(entropy: int) -> None:
    """Set the seed of imgaug's global RNG (in-place).

    The global RNG controls most of the "randomness" in imgaug2.

    The global RNG is the default one used by all augmenters. Under special
    circumstances (e.g. when an augmenter is switched to deterministic mode),
    the global RNG is replaced with a local one. The state of that replacement
    may be dependent on the global RNG's state at the time of creating the
    child RNG.

    Parameters
    ----------
    entropy : int
        The seed value to use.

    """
    if SUPPORTS_NEW_NP_RNG_STYLE:
        _seed_np117_(entropy)
    else:
        _seed_np116_(entropy)


def _seed_np117_(entropy: int) -> None:
    # We can't easily seed a BitGenerator in-place, nor can we easily modify
    # a Generator's bit_generator in-place. So instead we create a new
    # bit generator and set the current global RNG's internal bit generator
    # state to a copy of the new bit generator's state.
    assert BIT_GENERATOR is not None
    get_global_rng().state = BIT_GENERATOR(entropy).state


def _seed_np116_(entropy: int) -> None:
    generator = get_global_rng().generator
    assert isinstance(generator, np.random.RandomState)
    generator.seed(entropy)


def normalize_generator(generator: RNGInput) -> NumpyGenerator:
    """Normalize various inputs to a numpy (random number) generator.

    This function will first copy the provided argument, i.e. it never returns
    a provided instance itself.

    Parameters
    ----------
    generator : None or int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState
        The numpy random number generator to normalize. In case of numpy
        version 1.17 or later, this shouldn't be a ``RandomState`` as that
        class is outdated.
        Behaviour for different datatypes:

          * If ``None``: The global RNG's generator is returned.
          * If ``int``: In numpy 1.17+, the value is used as a seed for a
            ``Generator``, i.e. it will be provided as the entropy to a
            ``SeedSequence``, which will then be used for an ``SFC64`` bit
            generator and wrapped by a ``Generator``, which is then returned.
            In numpy <=1.16, the value is used as a seed for a ``RandomState``,
            which will then be returned.
          * If :class:`numpy.random.Generator`: That generator will be
            returned.
          * If :class:`numpy.random.BitGenerator`: A numpy
            generator will be created and returned that contains the bit
            generator.
          * If :class:`numpy.random.SeedSequence`: A numpy
            generator will be created and returned that contains an ``SFC64``
            bit generator initialized with the given ``SeedSequence``.
          * If :class:`numpy.random.RandomState`: In numpy <=1.16, this
            ``RandomState`` will be returned. In numpy 1.17+, a seed will be
            derived from this ``RandomState`` and a new
            ``numpy.generator.Generator`` based on an ``SFC64`` bit generator
            will be created and returned.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator`` (even if
        the input was a ``RandomState``).

    """
    return normalize_generator_(copylib.deepcopy(generator))


def normalize_generator_(generator: RNGInput) -> NumpyGenerator:
    """Normalize in-place various inputs to a numpy (random number) generator.

    This function will try to return the provided instance itself.

    Parameters
    ----------
    generator : None or int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState
        See :func:`~imgaug2.random.normalize_generator`.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator`` (even if
        the input was a ``RandomState``).

    """
    if not SUPPORTS_NEW_NP_RNG_STYLE:
        return _normalize_generator_np116_(generator)  # type: ignore[arg-type]
    return _normalize_generator_np117_(generator)


def _normalize_generator_np117_(generator: RNGInput) -> np.random.Generator:
    if generator is None:
        global_generator = get_global_rng().generator
        assert isinstance(global_generator, np.random.Generator)
        return global_generator

    if isinstance(generator, np.random.SeedSequence):
        assert BIT_GENERATOR is not None
        return np.random.Generator(BIT_GENERATOR(generator))

    if isinstance(generator, _BIT_GENERATOR_INTERFACE):
        gen = np.random.Generator(generator)  # type: ignore[arg-type]
        # TODO is it necessary/sensible here to reset the cache?
        reset_generator_cache_(gen)
        return gen

    if isinstance(generator, np.random.Generator):
        # TODO is it necessary/sensible here to reset the cache?
        reset_generator_cache_(generator)
        return generator

    if isinstance(generator, np.random.RandomState):
        # TODO warn
        # TODO reset the cache here too?
        return _convert_seed_to_generator_np117(generate_seed_(generator))

    # seed given
    seed_ = generator
    if isinstance(seed_, np.integer):
        seed_ = int(seed_)
    assert isinstance(seed_, int)
    return _convert_seed_to_generator_np117(seed_)


def _normalize_generator_np116_(
    random_state: int | np.random.RandomState | None,
) -> np.random.RandomState:
    if random_state is None:
        global_generator = get_global_rng().generator
        assert isinstance(global_generator, np.random.RandomState)
        return global_generator
    if isinstance(random_state, np.random.RandomState):
        # TODO reset the cache here, like in np117?
        return random_state
    # seed given
    seed_ = random_state
    if isinstance(seed_, np.integer):
        seed_ = int(seed_)
    assert isinstance(seed_, int)
    return _convert_seed_to_generator_np116(seed_)


def convert_seed_to_generator(entropy: int) -> NumpyGenerator:
    """Convert a seed value to a numpy (random number) generator.

    Parameters
    ----------
    entropy : int
        The seed value to use.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        Both are initialized with the provided seed.

    """
    if not SUPPORTS_NEW_NP_RNG_STYLE:
        return _convert_seed_to_generator_np116(entropy)
    return _convert_seed_to_generator_np117(entropy)


def _convert_seed_to_generator_np117(entropy: int) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence(entropy)
    return convert_seed_sequence_to_generator(seed_sequence)


def _convert_seed_to_generator_np116(entropy: int) -> np.random.RandomState:
    return np.random.RandomState(entropy)


def convert_seed_sequence_to_generator(
    seed_sequence: np.random.SeedSequence,
) -> np.random.Generator:
    """Convert a seed sequence to a numpy (random number) generator.

    Parameters
    ----------
    seed_sequence : numpy.random.SeedSequence
        The seed value to use.

    Returns
    -------
    numpy.random.Generator
        Generator initialized with the provided seed sequence.

    """
    assert BIT_GENERATOR is not None
    return np.random.Generator(BIT_GENERATOR(seed_sequence))


def create_pseudo_random_generator_() -> NumpyGenerator:
    """Create a new numpy (random) generator, derived from the global RNG.

    This function advances the global RNG's state.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        Both are initialized with a seed sampled from the global RNG.

    """
    # could also use derive_rng(get_global_rng()) here
    random_seed = generate_seed_(get_global_rng().generator)
    return convert_seed_to_generator(random_seed)


def create_fully_random_generator() -> NumpyGenerator:
    """Create a new numpy (random) generator, derived from OS's entropy.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        Both are initialized with entropy requested from the OS. They are
        hence independent of entered seeds or the library's global RNG.

    """
    if not SUPPORTS_NEW_NP_RNG_STYLE:
        return _create_fully_random_generator_np116()
    return _create_fully_random_generator_np117()


def _create_fully_random_generator_np117() -> np.random.Generator:
    # TODO need entropy here?
    return np.random.Generator(np.random.SFC64())


def _create_fully_random_generator_np116() -> np.random.RandomState:
    return np.random.RandomState()


def generate_seed_(generator: NumpyGenerator) -> int:
    """Sample a seed from the provided generator.

    This function advances the generator's state.

    See ``SEED_MIN_VALUE`` and ``SEED_MAX_VALUE`` for the seed's value
    range.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator from which to sample the seed.

    Returns
    -------
    int
        The sampled seed.

    """
    return generate_seeds_(generator, 1)[0]


def generate_seeds_(generator: NumpyGenerator, n: int) -> IntArray:
    """Sample `n` seeds from the provided generator.

    This function advances the generator's state.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator from which to sample the seed.

    n : int
        Number of seeds to sample.

    Returns
    -------
    ndarray
        1D-array of ``int32`` seeds.

    """
    seeds = polyfill_integers(generator, SEED_MIN_VALUE, SEED_MAX_VALUE, size=(n,))
    assert isinstance(seeds, np.ndarray)
    return cast(IntArray, seeds)


def copy_generator(generator: NumpyGenerator) -> NumpyGenerator:
    """Copy an existing numpy (random number) generator.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator to copy.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        Both are copies of the input argument.

    """
    if isinstance(generator, np.random.RandomState):
        return _copy_generator_np116(generator)
    return _copy_generator_np117(generator)


def _copy_generator_np117(generator: np.random.Generator) -> np.random.Generator:
    # TODO not sure if it is enough to only copy the state
    # TODO initializing a bit gen and then copying the state might be slower
    #      then just deepcopying the whole thing
    old_bit_gen = generator.bit_generator
    new_bit_gen = old_bit_gen.__class__(1)
    new_bit_gen.state = copylib.deepcopy(old_bit_gen.state)
    return np.random.Generator(new_bit_gen)


def _copy_generator_np116(random_state: np.random.RandomState) -> np.random.RandomState:
    rs_copy = np.random.RandomState(1)
    state = random_state.get_state()
    rs_copy.set_state(state)
    return rs_copy


def copy_generator_unless_global_generator(generator: NumpyGenerator) -> NumpyGenerator:
    """Copy a numpy generator unless it is the current global generator.

    "global generator" here denotes the generator contained in the
    global RNG's ``.generator`` attribute.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator to copy.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        Both are copies of the input argument, unless that input is
        identical to the global generator. If it is identical, the
        instance itself will be returned without copying it.

    """
    if generator is get_global_rng().generator:
        return generator
    return copy_generator(generator)


def reset_generator_cache_(generator: NumpyGenerator) -> NumpyGenerator:
    """Reset a numpy (random number) generator's internal cache.

    This function modifies the generator's state in-place.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator of which to reset the cache.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        In both cases the input argument itself.

    """
    if isinstance(generator, np.random.RandomState):
        return _reset_generator_cache_np116_(generator)
    return _reset_generator_cache_np117_(generator)


def _reset_generator_cache_np117_(generator: np.random.Generator) -> np.random.Generator:
    # This deactivates usage of the cache. We could also remove the cached
    # value itself in "uinteger", but setting the RNG to ignore the cached
    # value should be enough.
    state = _get_generator_state_np117(generator)
    state["has_uint32"] = 0
    _set_generator_state_np117_(generator, state)
    return generator


def _reset_generator_cache_np116_(random_state: np.random.RandomState) -> np.random.RandomState:
    # State tuple content:
    #   'MT19937', array of ints, unknown int, cache flag, cached value
    # The cache flag only affects the standard_normal() method.
    state: list[Any] = list(random_state.get_state())  # type: ignore[arg-type]
    state[-2] = 0
    random_state.set_state(tuple(state))  # type: ignore[arg-type]
    return random_state


def derive_generator_(generator: NumpyGenerator) -> NumpyGenerator:
    """Create a child numpy (random number) generator from an existing one.

    This advances the generator's state.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator from which to derive a new child generator.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        In both cases a derived child generator.

    """
    return derive_generators_(generator, n=1)[0]


# TODO does this advance the RNG in 1.17? It should advance it for security
#      reasons
def derive_generators_(generator: NumpyGenerator, n: int) -> Sequence[NumpyGenerator]:
    """Create child numpy (random number) generators from an existing one.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator from which to derive new child generators.

    n : int
        Number of child generators to derive.

    Returns
    -------
    list of numpy.random.Generator or list of numpy.random.RandomState
        In numpy <=1.16 a list of  ``RandomState`` s,
        in 1.17+ a list of ``Generator`` s.
        In both cases lists of derived child generators.

    """
    if isinstance(generator, np.random.RandomState):
        return _derive_generators_np116_(generator, n=n)
    return _derive_generators_np117_(generator, n=n)


def _derive_generators_np117_(generator: np.random.Generator, n: int) -> list[np.random.Generator]:
    # TODO possible to get the SeedSequence from 'rng'?
    """
    advance_rng_(rng)
    rng = copylib.deepcopy(rng)
    reset_rng_cache_(rng)
    state = rng.bit_generator.state
    rngs = []
    for i in range(n):
        state["state"]["state"] += (i * 100003 + 17)
        rng.bit_generator.state = state
        rngs.append(rng)
        rng = copylib.deepcopy(rng)
    return rngs
    """

    # We generate here two integers instead of one, because the internal state
    # of the RNG might have one 32bit integer still cached up, which would
    # then be returned first when calling integers(). This should usually be
    # fine, but there is some risk involved that this will lead to sampling
    # many times the same seed in loop constructions (if the internal state
    # is not properly advanced and the cache is then also not reset). Adding
    # 'size=(2,)' decreases that risk. (It is then enough to e.g. call once
    # random() to advance the internal state. No resetting of caches is
    # needed.)
    seed_ = generator.integers(SEED_MIN_VALUE, SEED_MAX_VALUE, dtype="int32", size=(2,))[-1]

    seed_seq = np.random.SeedSequence(seed_)
    seed_seqs = seed_seq.spawn(n)
    return [convert_seed_sequence_to_generator(seed_seq) for seed_seq in seed_seqs]


def _derive_generators_np116_(
    random_state: np.random.RandomState, n: int
) -> list[np.random.RandomState]:
    seed_ = random_state.randint(SEED_MIN_VALUE, SEED_MAX_VALUE)
    return [_convert_seed_to_generator_np116(seed_ + i) for i in range(n)]


def get_generator_state(generator: NumpyGenerator) -> GeneratorState:
    """Get the state of this provided generator.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator, which's state is supposed to be extracted.

    Returns
    -------
    tuple or dict
        The state of the generator.
        In numpy 1.17+, the bit generator's state will be returned.
        In numpy <=1.16, the ``RandomState`` 's state is returned.
        In both cases the state is a copy. In-place changes will not affect
        the RNG.

    """
    if isinstance(generator, np.random.RandomState):
        return _get_generator_state_np116(generator)
    return _get_generator_state_np117(generator)


def _get_generator_state_np117(generator: np.random.Generator) -> dict[str, object]:
    return dict(generator.bit_generator.state)


def _get_generator_state_np116(random_state: np.random.RandomState) -> tuple[object, ...]:
    state = random_state.get_state()
    assert isinstance(state, tuple)
    return state


def set_generator_state_(generator: NumpyGenerator, state: GeneratorState) -> None:
    """Set the state of a numpy (random number) generator in-place.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator, which's state is supposed to be modified.

    state : tuple or dict
        The new state of the generator.
        Should correspond to the output of
        :func:`~imgaug2.random.get_generator_state`.

    """
    if isinstance(generator, np.random.RandomState):
        assert isinstance(state, tuple)
        _set_generator_state_np116_(generator, state)
    else:
        assert isinstance(state, dict)
        _set_generator_state_np117_(generator, state)


def _set_generator_state_np117_(generator: np.random.Generator, state: dict[str, object]) -> None:
    generator.bit_generator.state = state


def _set_generator_state_np116_(
    random_state: np.random.RandomState, state: tuple[object, ...]
) -> None:
    random_state.set_state(state)  # type: ignore[arg-type]


def is_generator_equal_to(generator: NumpyGenerator, other_generator: NumpyGenerator) -> bool:
    """Estimate whether two generator have the same class and state.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        First generator used in the comparison.

    other_generator : numpy.random.Generator or numpy.random.RandomState
        Second generator used in the comparison.

    Returns
    -------
    bool
        ``True`` if `generator` 's class and state are the same as the
        class and state of `other_generator`. ``False`` otherwise.

    """
    if isinstance(generator, np.random.RandomState):
        assert isinstance(other_generator, np.random.RandomState)
        return _is_generator_equal_to_np116(generator, other_generator)
    assert isinstance(other_generator, np.random.Generator)
    return _is_generator_equal_to_np117(generator, other_generator)


def _is_generator_equal_to_np117(
    generator: np.random.Generator, other_generator: np.random.Generator
) -> bool:
    assert generator.__class__ is other_generator.__class__, (
        "Expected both rngs to have the same class, "
        f"got types '{type(generator)}' and '{type(other_generator)}'."
    )

    state1 = _get_generator_state_np117(generator)
    state2 = _get_generator_state_np117(other_generator)
    assert state1["bit_generator"] == "SFC64", (
        "Can currently only compare the states of numpy.random.SFC64 bit "
        "generators, got {}.".format(state1["bit_generator"])
    )
    assert state2["bit_generator"] == "SFC64", (
        "Can currently only compare the states of numpy.random.SFC64 bit "
        "generators, got {}.".format(state2["bit_generator"])
    )

    if state1["has_uint32"] != state2["has_uint32"]:
        return False

    if state1["has_uint32"] == state2["has_uint32"] == 1:
        if state1["uinteger"] != state2["uinteger"]:
            return False

    # Access nested state - use cast for type narrowing
    inner_state1 = cast(dict[str, Any], state1["state"])
    inner_state2 = cast(dict[str, Any], state2["state"])
    return np.array_equal(inner_state1["state"], inner_state2["state"])


def _is_generator_equal_to_np116(
    random_state: np.random.RandomState, other_random_state: np.random.RandomState
) -> bool:
    state1 = _get_generator_state_np116(random_state)
    state2 = _get_generator_state_np116(other_random_state)
    # Note that state1 and state2 are tuples with the value at index 1 being
    # a numpy array and the values at 2-4 being ints/floats, so we can't just
    # apply array_equal to state1[1:4+1] and state2[1:4+1]. We need a loop
    # here.
    for i in range(1, 4 + 1):
        if not np.array_equal(state1[i], state2[i]):  # type: ignore[arg-type]
            return False
    return True


def advance_generator_(generator: NumpyGenerator) -> None:
    """Advance a numpy random generator's internal state in-place by one step.

    This advances the generator's state.

    .. note::

        This simply samples one or more random values. This means that
        a call of this method will not completely change the outputs of
        the next called sampling method. To achieve more drastic output
        changes, call :func:`~imgaug2.random.derive_generator_`.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        Generator of which to advance the internal state.

    """
    if isinstance(generator, np.random.RandomState):
        _advance_generator_np116_(generator)
    else:
        _advance_generator_np117_(generator)


def _advance_generator_np117_(generator: np.random.Generator) -> None:
    _reset_generator_cache_np117_(generator)
    generator.random()


def _advance_generator_np116_(generator: np.random.RandomState) -> None:
    _reset_generator_cache_np116_(generator)
    generator.uniform()


def polyfill_integers(
    generator: NumpyGenerator,
    low: int | IntArray,
    high: int | IntArray | None = None,
    size: Size = None,
    dtype: DTypeLike = "int32",
    endpoint: bool = False,
) -> int | IntArray:
    """Sample integers from a generator in different numpy versions.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator to sample from. If it is a ``RandomState``,
        :func:`numpy.random.RandomState.randint` will be called,
        otherwise :func:`numpy.random.Generator.integers`.

    low : int or array-like of ints
        See :func:`numpy.random.Generator.integers`.

    high : int or array-like of ints, optional
        See :func:`numpy.random.Generator.integers`.

    size : int or tuple of ints, optional
        See :func:`numpy.random.Generator.integers`.

    dtype : {str, dtype}, optional
        See :func:`numpy.random.Generator.integers`.

    endpoint : bool, optional
        See :func:`numpy.random.Generator.integers`.

    Returns
    -------
    int or ndarray of ints
        See :func:`numpy.random.Generator.integers`.

    """
    if hasattr(generator, "randint"):
        if endpoint:
            if high is None:
                high = low + 1
                low = 0
            else:
                high = high + 1
        return generator.randint(low=low, high=high, size=size, dtype=dtype)  # type: ignore[call-overload, return-value]
    return generator.integers(low=low, high=high, size=size, dtype=dtype, endpoint=endpoint)  # type: ignore[call-overload, return-value]


def polyfill_random(
    generator: NumpyGenerator,
    size: Size,
    dtype: DTypeLike = "float32",
    out: FloatArray | None = None,
) -> float | FloatArray:
    """Sample random floats from a generator in different numpy versions.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator to sample from. Both ``RandomState`` and ``Generator``
        support ``random()``, but with different interfaces.

    size : int or tuple of ints, optional
        See :func:`numpy.random.Generator.random`.

    dtype : {str, dtype}, optional
        See :func:`numpy.random.Generator.random`.

    out : ndarray, optional
        See :func:`numpy.random.Generator.random`.


    Returns
    -------
    float or ndarray of floats
        See :func:`numpy.random.Generator.random`.

    """
    if hasattr(generator, "random_sample"):
        # note that numpy.random in <=1.16 supports random(), but
        # numpy.random.RandomState does not
        result = generator.random_sample(size=size).astype(dtype)  # type: ignore[union-attr]
        if out is not None:
            assert out.dtype.name == result.dtype.name, (
                "Expected out array to have the same dtype as "
                f"random_sample()'s result array. Got {out.dtype.name} (out) and {result.dtype.name} (result) "
                "instead."
            )
            out[...] = result
        return result
    return generator.random(size=size, dtype=dtype, out=out)  # type: ignore[call-overload, return-value]


RNGInput: TypeAlias = (
    int
    | RNG
    | np.random.Generator
    | np.random.BitGenerator
    | np.random.SeedSequence
    | np.random.RandomState
    | None
)


@legacy(version="0.4.0")
class temporary_numpy_seed:
    """Context to temporarily alter the random state of ``numpy.random``.

    The random state's internal state will be set back to the original one
    once the context finishes.


    Parameters
    ----------
    entropy : None or int
        The seed value to use.
        If `None` then the seed will not be altered and the internal state
        of ``numpy.random`` will not be reset back upon context exit (i.e.
        this context will do nothing).

    """

    def __init__(self, entropy: int | None = None) -> None:
        self.old_state: dict[str, Any] | RandomStateState | None = None
        self.entropy: int | None = entropy

    def __enter__(self) -> None:
        if self.entropy is not None:
            self.old_state = np.random.get_state()  # type: ignore[assignment]
            np.random.seed(self.entropy)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.entropy is not None:
            assert self.old_state is not None
            np.random.set_state(self.old_state)  # type: ignore[arg-type]
