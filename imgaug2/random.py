"""Random helpers and RNG wrapper for imgaug2.

Wraps NumPy's Generator API and provides helpers for seeding and
state management.
"""

from __future__ import annotations

import copy as copylib
from collections.abc import Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from imgaug2.compat.markers import legacy

if TYPE_CHECKING:
    from numpy._typing import _ShapeLike

Size: TypeAlias = int | tuple[int, ...] | None
ShapeLike: TypeAlias = "int | tuple[int, ...] | _ShapeLike"
GeneratorState: TypeAlias = dict[str, object]
NumpyGenerator: TypeAlias = np.random.Generator

FloatArray: TypeAlias = NDArray[np.floating[Any]]
IntArray: TypeAlias = NDArray[np.integer[Any]]
GenericArray: TypeAlias = NDArray[np.generic]

BIT_GENERATOR: type[np.random.BitGenerator] = np.random.SFC64

# Global RNG instance (lazy-initialized).
GLOBAL_RNG: RNG | None = None

# Use 2**31-1 as max; 2**31 fails on some platforms.
SEED_MIN_VALUE: int = 0
SEED_MAX_VALUE: int = 2**31 - 1

_RNG_IDX: int = 1

# TODO add 'with resetted_rng(...)'
# TODO change random_state to rng or seed


class RNG:
    """Unified RNG wrapper for NumPy Generator.

    Accepts a seed, an existing RNG, or a NumPy RNG instance.
    """

    # TODO add maybe a __new__ here that feeds-through an RNG input without
    #      wrapping it in RNG(rng_input)?
    def __init__(self, generator: RNGInput) -> None:
        global _RNG_IDX

        if isinstance(generator, RNG):
            self.generator = generator.generator
        else:
            self.generator = normalize_generator_(generator)
        # Unique id used by AutoPrefetcher; state-based ids are too costly.
        self._idx = _RNG_IDX
        _RNG_IDX += 1

    @property
    def state(self) -> GeneratorState:
        """Get the state of this RNG.

        Returns:
            The state of the RNG (dict).
            Returns a copy; in-place changes to the return value will not affect the RNG.
        """
        return get_generator_state(self.generator)

    @state.setter
    def state(self, value: GeneratorState) -> None:
        """Set the state of the RNG in-place.

        Parameters:
            value: The new state of the RNG (dict), matching the format from `state`.
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
            "numpy generators."
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
        return self.generator.integers(
            low=low, high=high, size=size, dtype=dtype, endpoint=endpoint
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
        return self.generator.random(size=size, dtype=dtype, out=out)

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

    def bytes(self, length: int) -> bytes:
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
        return self.generator.standard_exponential(
            size=size,
            dtype=dtype,
            method=method,
            out=out,  # type: ignore[arg-type]
        )

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
        return self.generator.standard_gamma(  # type: ignore[call-overload, return-value]
            shape=shape, size=size, dtype=dtype, out=out
        )

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
        return self.generator.standard_normal(  # type: ignore[call-overload, return-value]
            size=size, dtype=dtype, out=out
        )

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
        GLOBAL_RNG = RNG(create_fully_random_generator())
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
    # We can't easily seed a BitGenerator in-place, nor can we easily modify
    # a Generator's bit_generator in-place. So instead we create a new
    # bit generator and set the current global RNG's internal bit generator
    # state to a copy of the new bit generator's state.
    get_global_rng().state = BIT_GENERATOR(entropy).state


def normalize_generator(generator: RNGInput) -> NumpyGenerator:
    """Normalize various inputs to a numpy (random number) generator.

    This function will first copy the provided argument, i.e. it never returns
    a provided instance itself.

    Parameters
    ----------
    generator : None or int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or imgaug2.random.RNG
        The numpy random number generator to normalize. Behaviour for
        different datatypes:

          * If ``None``: The global RNG's generator is returned.
          * If ``int``: The value is used as entropy for a ``SeedSequence``,
            which seeds an ``SFC64`` bit generator wrapped by a ``Generator``.
          * If :class:`numpy.random.Generator`: That generator will be
            returned.
          * If :class:`numpy.random.BitGenerator`: A numpy generator will be
            created and returned that contains the bit generator.
          * If :class:`numpy.random.SeedSequence`: A numpy generator will be
            created and returned that contains an ``SFC64`` bit generator
            initialized with the given ``SeedSequence``.
          * If :class:`imgaug2.random.RNG`: The wrapped numpy generator will
            be returned.

    Returns
    -------
    numpy.random.Generator
        Generator initialized from the provided input.

    Notes
    -----
    Legacy numpy RNG inputs are no longer supported and will raise
    ``TypeError``.

    """
    return normalize_generator_(copylib.deepcopy(generator))


def normalize_generator_(generator: RNGInput) -> NumpyGenerator:
    """Normalize in-place various inputs to a numpy (random number) generator.

    This function will try to return the provided instance itself.

    Parameters
    ----------
    generator : None or int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or imgaug2.random.RNG
        See :func:`~imgaug2.random.normalize_generator`.

    Returns
    -------
    numpy.random.Generator
        Generator initialized from the provided input.

    """
    if generator is None:
        global_generator = get_global_rng().generator
        return global_generator

    if isinstance(generator, RNG):
        return generator.generator

    if isinstance(generator, np.random.SeedSequence):
        return np.random.Generator(BIT_GENERATOR(generator))

    if isinstance(generator, np.random.BitGenerator):
        gen = np.random.Generator(generator)  # type: ignore[arg-type]
        # TODO is it necessary/sensible here to reset the cache?
        reset_generator_cache_(gen)
        return gen

    if isinstance(generator, np.random.Generator):
        # TODO is it necessary/sensible here to reset the cache?
        reset_generator_cache_(generator)
        return generator

    # seed given
    seed_ = generator
    if isinstance(seed_, np.integer):
        seed_ = int(seed_)
    if isinstance(seed_, int):
        return convert_seed_to_generator(seed_)
    raise TypeError(
        "Expected RNG input: None, int, numpy.random.Generator, "
        "numpy.random.BitGenerator, numpy.random.SeedSequence, or imgaug2.random.RNG."
    )


def convert_seed_to_generator(entropy: int) -> NumpyGenerator:
    """Convert a seed value to a numpy (random number) generator.

    Parameters
    ----------
    entropy : int
        The seed value to use.

    Returns
    -------
    numpy.random.Generator
        Generator initialized with the provided seed.

    """
    seed_sequence = np.random.SeedSequence(entropy)
    return convert_seed_sequence_to_generator(seed_sequence)


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
    return np.random.Generator(BIT_GENERATOR(seed_sequence))


def create_pseudo_random_generator_() -> NumpyGenerator:
    """Create a new numpy (random) generator, derived from the global RNG.

    This function advances the global RNG's state.

    Returns
    -------
    numpy.random.Generator
        Generator initialized with a seed sampled from the global RNG.

    """
    # could also use derive_rng(get_global_rng()) here
    random_seed = generate_seed_(get_global_rng().generator)
    return convert_seed_to_generator(random_seed)


def create_fully_random_generator() -> NumpyGenerator:
    """Create a new numpy (random) generator, derived from OS's entropy.

    Returns
    -------
    numpy.random.Generator
        Generator initialized with entropy requested from the OS. It is
        independent of entered seeds or the library's global RNG.

    """
    # TODO need entropy here?
    return np.random.Generator(BIT_GENERATOR())


def generate_seed_(generator: NumpyGenerator) -> int:
    """Sample a seed from the provided generator.

    This function advances the generator's state.

    See ``SEED_MIN_VALUE`` and ``SEED_MAX_VALUE`` for the seed's value
    range.

    Parameters
    ----------
    generator : numpy.random.Generator
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
    generator : numpy.random.Generator
        The generator from which to sample the seed.

    n : int
        Number of seeds to sample.

    Returns
    -------
    ndarray
        1D-array of ``int32`` seeds.

    """
    seeds = generator.integers(
        SEED_MIN_VALUE, SEED_MAX_VALUE, size=(n,), dtype=np.int32, endpoint=False
    )
    return cast(IntArray, seeds)


def copy_generator(generator: NumpyGenerator) -> NumpyGenerator:
    """Copy an existing numpy (random number) generator.

    Parameters
    ----------
    generator : numpy.random.Generator
        The generator to copy.

    Returns
    -------
    numpy.random.Generator
        Copy of the input generator.

    """
    return _copy_generator_np117(generator)


def _copy_generator_np117(generator: np.random.Generator) -> np.random.Generator:
    # TODO not sure if it is enough to only copy the state
    # TODO initializing a bit gen and then copying the state might be slower
    #      then just deepcopying the whole thing
    old_bit_gen = generator.bit_generator
    new_bit_gen = old_bit_gen.__class__(1)
    new_bit_gen.state = copylib.deepcopy(old_bit_gen.state)
    return np.random.Generator(new_bit_gen)


def copy_generator_unless_global_generator(generator: NumpyGenerator) -> NumpyGenerator:
    """Copy a numpy generator unless it is the current global generator.

    "global generator" here denotes the generator contained in the
    global RNG's ``.generator`` attribute.

    Parameters
    ----------
    generator : numpy.random.Generator
        The generator to copy.

    Returns
    -------
    numpy.random.Generator
        Copy of the input argument, unless it is identical to the global
        generator. If it is identical, the instance itself will be returned.

    """
    if generator is get_global_rng().generator:
        return generator
    return copy_generator(generator)


def reset_generator_cache_(generator: NumpyGenerator) -> NumpyGenerator:
    """Reset a numpy (random number) generator's internal cache.

    This function modifies the generator's state in-place.

    Parameters
    ----------
    generator : numpy.random.Generator
        The generator of which to reset the cache.

    Returns
    -------
    numpy.random.Generator
        The input argument itself.

    """
    return _reset_generator_cache_np117_(generator)


def _reset_generator_cache_np117_(generator: np.random.Generator) -> np.random.Generator:
    # This deactivates usage of the cache. We could also remove the cached
    # value itself in "uinteger", but setting the RNG to ignore the cached
    # value should be enough.
    state = _get_generator_state_np117(generator)
    state["has_uint32"] = 0
    _set_generator_state_np117_(generator, state)
    return generator


def derive_generator_(generator: NumpyGenerator) -> NumpyGenerator:
    """Create a child numpy (random number) generator from an existing one.

    This advances the generator's state.

    Parameters
    ----------
    generator : numpy.random.Generator
        The generator from which to derive a new child generator.

    Returns
    -------
    numpy.random.Generator
        A derived child generator.

    """
    return derive_generators_(generator, n=1)[0]


# TODO ensure this advances the RNG for security reasons
def derive_generators_(generator: NumpyGenerator, n: int) -> Sequence[NumpyGenerator]:
    """Create child numpy (random number) generators from an existing one.

    Parameters
    ----------
    generator : numpy.random.Generator
        The generator from which to derive new child generators.

    n : int
        Number of child generators to derive.

    Returns
    -------
    list of numpy.random.Generator
        List of derived child generators.

    """
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


def get_generator_state(generator: NumpyGenerator) -> GeneratorState:
    """Get the state of this provided generator.

    Parameters
    ----------
    generator : numpy.random.Generator
        The generator, which's state is supposed to be extracted.

    Returns
    -------
    dict
        The state of the generator.
        The bit generator's state will be returned.
        The state is a copy. In-place changes will not affect the RNG.

    """
    return _get_generator_state_np117(generator)


def _get_generator_state_np117(generator: np.random.Generator) -> dict[str, object]:
    return dict(generator.bit_generator.state)


def set_generator_state_(generator: NumpyGenerator, state: GeneratorState) -> None:
    """Set the state of a numpy (random number) generator in-place.

    Parameters
    ----------
    generator : numpy.random.Generator
        The generator, which's state is supposed to be modified.

    state : dict
        The new state of the generator.
        Should correspond to the output of
        :func:`~imgaug2.random.get_generator_state`.

    """
    assert isinstance(state, dict)
    _set_generator_state_np117_(generator, state)


def _set_generator_state_np117_(generator: np.random.Generator, state: dict[str, object]) -> None:
    generator.bit_generator.state = state


def is_generator_equal_to(generator: NumpyGenerator, other_generator: NumpyGenerator) -> bool:
    """Estimate whether two generator have the same class and state.

    Parameters
    ----------
    generator : numpy.random.Generator
        First generator used in the comparison.

    other_generator : numpy.random.Generator
        Second generator used in the comparison.

    Returns
    -------
    bool
        ``True`` if `generator` 's class and state are the same as the
        class and state of `other_generator`. ``False`` otherwise.

    """
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
    generator : numpy.random.Generator
        Generator of which to advance the internal state.

    """
    _advance_generator_np117_(generator)


def _advance_generator_np117_(generator: np.random.Generator) -> None:
    _reset_generator_cache_np117_(generator)
    generator.random()


RNGInput: TypeAlias = (
    int
    | RNG
    | np.random.Generator
    | np.random.BitGenerator
    | np.random.SeedSequence
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
        self.old_state: tuple[object, ...] | None = None
        self.entropy: int | None = entropy

    def __enter__(self) -> None:
        if self.entropy is not None:
            self.old_state = np.random.get_state()
            np.random.seed(self.entropy)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.entropy is not None:
            assert self.old_state is not None
            np.random.set_state(self.old_state)
