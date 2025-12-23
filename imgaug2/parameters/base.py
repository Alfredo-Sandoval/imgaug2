from __future__ import annotations

import copy as copy_module
import tempfile
from abc import ABCMeta, abstractmethod
from functools import reduce
from operator import mul as mul_op
from typing import Any, TypeAlias, Union

import imageio
import numpy as np
from numpy.typing import NDArray

import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.compat.markers import legacy

Numberish: TypeAlias = Union[float, int, "StochasticParameter"]
ParamInput: TypeAlias = Union[float, tuple[float, float], list[float], "StochasticParameter"]

_PREFETCHING_ENABLED = True


class StochasticParameter(metaclass=ABCMeta):
    """Abstract parent class for all stochastic parameters.

    Stochastic parameters are here all parameters from which values are
    supposed to be sampled. Usually the sampled values are to a degree random.
    E.g. a stochastic parameter may be the uniform distribution over the
    interval ``[-10, 10]``. Samples from that distribution (and therefore the
    stochastic parameter) could be ``5.2``, ``-3.7``, ``-9.7``, ``6.4``, etc.

    """

    def __init__(self) -> None:  # noqa: B027
        pass

    @property
    @legacy(version="0.5.0")
    def prefetchable(self) -> bool:
        """Determines whether this parameter may be prefetched.


        Returns
        -------
        bool
            Whether to allow prefetching of this parameter's samples.
            This should usually only be ``True`` for parameters that actually
            perform random sampling, i.e. depend on an RNG.

        """
        return False

    def draw_sample(
        self,
        random_state: iarandom.RNGInput = None,
    ) -> Any:  # noqa: ANN401
        """
        Draws a single sample value from this parameter.

        Parameters
        ----------
        random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
            A seed or random number generator to use during the sampling
            process. If ``None``, the global RNG will be used.
            See also `__init__()`
            for a similar parameter with more details.

        Returns
        -------
        any
            A single sample value.

        """
        return self.draw_samples(1, random_state=random_state)[0]

    def draw_samples(
        self,
        size: int | tuple[int, ...],
        random_state: iarandom.RNGInput = None,
    ) -> Any:  # noqa: ANN401
        """Draw one or more samples from the parameter.

        Parameters
        ----------
        size : tuple of int or int
            Number of samples by dimension.

        random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
            A seed or random number generator to use during the sampling
            process. If ``None``, the global RNG will be used.
            See also `__init__()`
            for a similar parameter with more details.

        Returns
        -------
        ndarray
            Sampled values. Usually a numpy ndarray of basically any dtype,
            though not strictly limited to numpy arrays. Its shape is expected
            to match `size`.

        """
        if not isinstance(random_state, iarandom.RNG):
            random_state = iarandom.RNG(random_state)
        samples = self._draw_samples(
            size if not ia.is_single_integer(size) else tuple([size]), random_state  # type: ignore
        )
        random_state.advance_()
        return samples

    @abstractmethod
    def _draw_samples(
        self, size: tuple[int, ...], random_state: iarandom.RNG
    ) -> Any:  # noqa: ANN401
        raise NotImplementedError()

    def __add__(self, other: float | int | StochasticParameter) -> StochasticParameter:
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Add

            return Add(self, other)
        raise Exception(
            f"Invalid datatypes in: StochasticParameter + {type(other)}. "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def __sub__(self, other: float | int | StochasticParameter) -> StochasticParameter:
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Subtract

            return Subtract(self, other)
        raise Exception(
            f"Invalid datatypes in: StochasticParameter - {type(other)}. "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def __mul__(self, other: float | int | StochasticParameter) -> StochasticParameter:
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Multiply

            return Multiply(self, other)
        raise Exception(
            f"Invalid datatypes in: StochasticParameter * {type(other)}. "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def __pow__(
        self, other: float | int | StochasticParameter, z: Any | None = None  # noqa: ANN401
    ) -> StochasticParameter:
        if z is not None:
            raise NotImplementedError(
                "Modulo power is currently not supported by StochasticParameter."
            )
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Power

            return Power(self, other)
        raise Exception(
            f"Invalid datatypes in: StochasticParameter ** {type(other)}. "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def __div__(self, other: float | int | StochasticParameter) -> StochasticParameter:
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Divide

            return Divide(self, other)
        raise Exception(
            f"Invalid datatypes in: StochasticParameter / {type(other)}. "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def __truediv__(self, other: float | int | StochasticParameter) -> StochasticParameter:
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Divide

            return Divide(self, other)
        raise Exception(
            f"Invalid datatypes in: StochasticParameter / {type(other)} (truediv). "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def __floordiv__(self, other: float | int | StochasticParameter) -> StochasticParameter:
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Discretize, Divide

            return Discretize(Divide(self, other))
        raise Exception(
            f"Invalid datatypes in: StochasticParameter // {type(other)} (floordiv). "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def __radd__(self, other: float | int | StochasticParameter) -> StochasticParameter:
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Add

            return Add(other, self)
        raise Exception(
            f"Invalid datatypes in: {type(other)} + StochasticParameter. "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def __rsub__(self, other: float | int | StochasticParameter) -> StochasticParameter:
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Subtract

            return Subtract(other, self)
        raise Exception(
            f"Invalid datatypes in: {type(other)} - StochasticParameter. "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def __rmul__(self, other: float | int | StochasticParameter) -> StochasticParameter:
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Multiply

            return Multiply(other, self)
        raise Exception(
            f"Invalid datatypes in: {type(other)} * StochasticParameter. "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def __rpow__(
        self, other: float | int | StochasticParameter, z: Any | None = None  # noqa: ANN401
    ) -> StochasticParameter:
        if z is not None:
            raise NotImplementedError(
                "Modulo power is currently not supported by StochasticParameter."
            )
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Power

            return Power(other, self)
        raise Exception(
            f"Invalid datatypes in: {type(other)} ** StochasticParameter. "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def __rdiv__(self, other: float | int | StochasticParameter) -> StochasticParameter:
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Divide

            return Divide(other, self)
        raise Exception(
            f"Invalid datatypes in: {type(other)} / StochasticParameter. "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def __rtruediv__(self, other: float | int | StochasticParameter) -> StochasticParameter:
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Divide

            return Divide(other, self)
        raise Exception(
            f"Invalid datatypes in: {type(other)} / StochasticParameter (rtruediv). "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def __rfloordiv__(self, other: float | int | StochasticParameter) -> StochasticParameter:
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            from .transforms import Discretize, Divide

            return Discretize(Divide(other, self))
        raise Exception(
            f"Invalid datatypes in: StochasticParameter // {type(other)} (rfloordiv). "
            "Expected second argument to be number or "
            "StochasticParameter."
        )

    def copy(self) -> StochasticParameter:
        """Create a shallow copy of this parameter.

        Returns
        -------
        imgaug2.parameters.StochasticParameter
            Shallow copy.

        """
        return copy_module.copy(self)

    def deepcopy(self) -> StochasticParameter:
        """Create a deep copy of this parameter.

        Returns
        -------
        imgaug2.parameters.StochasticParameter
            Deep copy.

        """
        return copy_module.deepcopy(self)

    def draw_distribution_graph(
        self,
        title: str | bool | None = None,
        size: tuple[int, ...] = (1000, 1000),
        bins: int = 100,
    ) -> NDArray:
        """Generate an image visualizing the parameter's sample distribution.

        Parameters
        ----------
        title : None or False or str, optional
            Title of the plot. ``None`` is automatically replaced by a title
            derived from ``str(param)``. If set to ``False``, no title will be
            shown.

        size : tuple of int
            Number of points to sample. This is always expected to have at
            least two values. The first defines the number of sampling runs,
            the second (and further) dimensions define the size assigned
            to each `draw_samples()`
            call. E.g. ``(10, 20, 15)`` will lead to ``10`` calls of
            ``draw_samples(size=(20, 15))``. The results will be merged to a
            single 1d array.

        bins : int
            Number of bins in the plot histograms.

        Returns
        -------
        data : (H,W,3) ndarray
            Image of the plot.

        """
        # import only when necessary (faster startup; optional dependency;
        # less fragile -- see issue #225)
        import matplotlib.pyplot as plt

        points = []
        for _ in range(size[0]):
            points.append(self.draw_samples(size[1:]).flatten())
        points = np.concatenate(points)

        fig = plt.figure()
        fig.add_subplot(111)
        ax = fig.gca()
        heights, bins = np.histogram(points, bins=bins)
        heights = heights / sum(heights)
        ax.bar(
            bins[:-1], heights, width=(max(bins) - min(bins)) / len(bins), color="blue", alpha=0.75
        )

        if title is None:
            title = str(self)
        if title is not False:
            # split long titles - otherwise matplotlib generates errors
            title_fragments = [title[i : i + 50] for i in range(0, len(title), 50)]
            ax.set_title("\n".join(title_fragments))
        fig.tight_layout(pad=0)

        with tempfile.NamedTemporaryFile(mode="wb+", suffix=".png") as f:
            # We don't add bbox_inches='tight' here so that
            # draw_distributions_grid has an easier time combining many plots.
            # Note that we could also use 'f.name' here instead of 'f', but
            # that fails on Windows.
            fig.savefig(f, format="png")

            # Use f.seek() here, because otherwise we get an error that
            # the file was not a png image.
            f.seek(0)
            data = imageio.imread(f, pilmode="RGB", format="png")[..., 0:3]

        plt.close()

        return data


@legacy(version="0.5.0")
class AutoPrefetcher(StochasticParameter):
    """Parameter that prefetches random samples from a child parameter.

    This parameter will fetch ``N`` random samples in one big swoop and then
    return ``M`` of these samples upon each call, with ``M << N``.
    This improves the sampling efficiency by performing as few sampling
    calls as possible.

    This parameter will only start to prefetch after the first call.
    In some cases this prevents inefficiencies when augmenters are only used
    once. (Though this only works if the respective augmenter performs
    a single sampling call per batch and not one call per image.)

    This parameter will throw away its prefetched samples if a new RNG
    is provided (compared to the previous call). It will however ignore the
    state of the RNG.

    This parameter should only wrap leaf nodes. In something like
    ``Add(1, Normal(Uniform(0, 1), Uniform(0, 2)))`` it should only be applied
    to the two ``Uniform`` instaces. Otherwise, only a single sample of
    ``Uniform(0, 1)`` might be taken and influence thousands of samples of
    ``Normal``.

    Note that the samples returned by this parameter are part of a larger
    array. In-place changes to these samples should hence be performed with
    some caution.


    """

    @legacy(version="0.5.0")
    def __init__(self, other_param: StochasticParameter, nb_prefetch: int) -> None:
        super().__init__()
        self.other_param = other_param
        self.nb_prefetch = nb_prefetch

        self.samples: NDArray | None = None
        self.index = 0
        self.last_rng_idx: int | None = None

    @legacy(version="0.5.0")
    def _draw_samples(self, size: tuple[int, ...], random_state: iarandom.RNG) -> Any:  # noqa: ANN401
        if not _PREFETCHING_ENABLED:
            return self.other_param.draw_samples(size, random_state)

        if self.last_rng_idx is None or random_state._idx != self.last_rng_idx:
            self.last_rng_idx = random_state._idx
            self.samples = None
            return self.other_param.draw_samples(size, random_state)

        self.last_rng_idx = random_state._idx

        nb_components = reduce(mul_op, size)

        if nb_components >= self.nb_prefetch:
            return self.other_param.draw_samples(size, random_state)

        if self.samples is None:
            self._prefetch(random_state)

        leftover = len(self.samples) - self.index - nb_components  # type: ignore
        if leftover <= 0:
            self._prefetch(random_state)

        samples = self.samples[self.index : self.index + nb_components]  # type: ignore
        self.index += nb_components

        return samples.reshape(size)

    @legacy(version="0.5.0")
    def _prefetch(self, random_state: iarandom.RNG) -> None:
        samples = self.other_param.draw_samples((self.nb_prefetch,), random_state)
        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.concatenate([self.samples[self.index :], samples], axis=0)
        self.index = 0

    @legacy(version="0.5.0")
    def __getattr__(self, attr: str) -> Any:  # noqa: ANN401
        other_param = super().__getattribute__("other_param")
        return getattr(other_param, attr)

    @legacy(version="0.5.0")
    def __repr__(self) -> str:
        return self.__str__()

    @legacy(version="0.5.0")
    def __str__(self) -> str:
        has_samples = self.samples is not None
        samples_shape = self.samples.shape if has_samples else "None"  # type: ignore
        samples_dtype = self.samples.dtype.name if has_samples else "None"  # type: ignore
        return (
            f"AutoPrefetcher("
            f"nb_prefetch={self.nb_prefetch:d}, "
            f"samples={samples_shape} (dtype {samples_dtype}), "
            f"index={self.index:d}, "
            f"last_rng_idx={self.last_rng_idx}, "
            f"other_param={self.other_param!s}"
            f")"
        )
