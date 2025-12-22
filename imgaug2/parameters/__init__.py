"""Classes and methods to use for parameters of augmenters.

This module provides probability distributions, noise parameters, and
helper functions for normalizing parameter-related user inputs.
"""

from __future__ import annotations

from .base import AutoPrefetcher, Numberish, ParamInput, StochasticParameter
from .continuous import Beta, ChiSquare, Laplace, Normal, TruncatedNormal, Uniform, Weibull
from .discrete import Binomial, Choice, Deterministic, DeterministicList, DiscreteUniform, Poisson
from .handles import (
    _assert_arg_is_stoch_param,
    _check_value_range,
    handle_categorical_string_param,
    handle_continuous_param,
    handle_cval_arg,
    handle_discrete_kernel_size_param,
    handle_discrete_param,
    handle_position_parameter,
    handle_probability_param,
)
from .noise import FrequencyNoise, FromLowerResolution, IterativeNoiseAggregator, SimplexNoise, _NoiseParameterMixin
from .prefetch import (
    _NB_PREFETCH,
    _NB_PREFETCH_STRINGS,
    _prefetchable,
    _prefetchable_str,
    _wrap_leafs_of_param_in_prefetchers,
    _wrap_leafs_of_param_in_prefetchers_recursive,
    _wrap_param_in_prefetchers,
    no_prefetching,
    toggle_prefetching,
    toggled_prefetching,
)
from .transforms import (
    Absolute,
    Add,
    Clip,
    Discretize,
    Divide,
    ForceSign,
    Multiply,
    Negative,
    Positive,
    Power,
    RandomSign,
    Sigmoid,
    Subtract,
)
from .utils import (
    both_np_float_if_one_is_float,
    draw_distributions_grid,
    force_np_float_dtype,
    show_distributions_grid,
)

__all__ = [
    "StochasticParameter",
    "AutoPrefetcher",
    "Deterministic",
    "DeterministicList",
    "Choice",
    "Binomial",
    "DiscreteUniform",
    "Poisson",
    "Normal",
    "TruncatedNormal",
    "Laplace",
    "ChiSquare",
    "Weibull",
    "Uniform",
    "Beta",
    "Clip",
    "Discretize",
    "Multiply",
    "Divide",
    "Add",
    "Subtract",
    "Power",
    "Absolute",
    "RandomSign",
    "ForceSign",
    "Sigmoid",
    "Positive",
    "Negative",
    "FromLowerResolution",
    "IterativeNoiseAggregator",
    "SimplexNoise",
    "FrequencyNoise",
    "handle_continuous_param",
    "handle_discrete_param",
    "handle_categorical_string_param",
    "handle_discrete_kernel_size_param",
    "handle_probability_param",
    "handle_cval_arg",
    "handle_position_parameter",
    "force_np_float_dtype",
    "both_np_float_if_one_is_float",
    "draw_distributions_grid",
    "show_distributions_grid",
    "toggle_prefetching",
    "toggled_prefetching",
    "no_prefetching",
    "ParamInput",
    "Numberish",
]
