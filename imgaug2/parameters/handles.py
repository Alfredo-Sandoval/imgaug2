from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np

import imgaug2.imgaug as ia
from imgaug2.compat.markers import legacy

from .base import ParamInput, StochasticParameter
from .prefetch import _NB_PREFETCH, _NB_PREFETCH_STRINGS, _wrap_leafs_of_param_in_prefetchers


@legacy
def _check_value_range(
    value: float | int,
    name: str,
    value_range: tuple[float | None, float | None] | Callable[[float | int], Any] | None,
) -> bool:
    if value_range is None:
        return True

    if isinstance(value_range, tuple):
        assert len(value_range) == 2, (
            f"If 'value_range' is a tuple, it must contain exactly 2 entries, "
            f"got {len(value_range)}."
        )

        if value_range[0] is None and value_range[1] is None:
            return True

        if value_range[0] is None:
            assert value <= value_range[1], (  # type: ignore
                f"Parameter '{name}' is outside of the expected value "
                f"range (x <= {value_range[1]:.4f})"
            )
            return True

        if value_range[1] is None:
            assert value_range[0] <= value, (  # type: ignore
                f"Parameter '{name}' is outside of the expected value "
                f"range ({value_range[0]:.4f} <= x)"
            )
            return True

        assert value_range[0] <= value <= value_range[1], (  # type: ignore
            f"Parameter '{name}' is outside of the expected value "
            f"range ({value_range[0]:.4f} <= x <= {value_range[1]:.4f})"
        )

        return True

    if ia.is_callable(value_range):
        value_range(value)
        return True

    raise Exception(f"Unexpected input for value_range, got {str(value_range)}.")


# NOTE: `_check_value_range()` validates the *provided* bounds and is inclusive
# on the upper end. When `param` is converted to `Uniform(a, b)`, the produced
# samples are in the half-open interval ``[a, b)``. If you need strict upper
# bound validation (e.g. require ``x < b``), pass a callable via `value_range`.
@legacy
def handle_continuous_param(
    param: float | tuple[float, float] | list[float] | StochasticParameter,
    name: str,
    value_range: tuple[float | None, float | None] | Callable[[float | int], Any] | None = None,
    tuple_to_uniform: bool = True,
    list_to_choice: bool = True,
    prefetch: bool = True,
) -> StochasticParameter:
    from .continuous import Uniform
    from .discrete import Choice, Deterministic

    result = None

    if ia.is_single_number(param):
        _check_value_range(param, name, value_range)  # type: ignore
        result = Deterministic(param)
    elif tuple_to_uniform and isinstance(param, tuple):
        assert len(param) == 2, (
            f"Expected parameter '{name}' with type tuple to have exactly two "
            f"entries, but got {len(param)}."
        )
        assert all([ia.is_single_number(v) for v in param]), (
            f"Expected parameter '{name}' with type tuple to only contain "
            f"numbers, got {[type(v) for v in param]}."
        )
        _check_value_range(param[0], name, value_range)
        _check_value_range(param[1], name, value_range)
        result = Uniform(param[0], param[1])
    elif list_to_choice and ia.is_iterable(param) and not isinstance(param, tuple):
        assert all([ia.is_single_number(v) for v in param]), (  # type: ignore
            f"Expected iterable parameter '{name}' to only contain numbers, "
            f"got {[type(v) for v in param]}."
        )
        for param_i in param:  # type: ignore
            _check_value_range(param_i, name, value_range)
        result = Choice(param)  # type: ignore
    elif isinstance(param, StochasticParameter):
        result = param

    if result is not None:
        if prefetch:
            return _wrap_leafs_of_param_in_prefetchers(result, _NB_PREFETCH)
        return result

    allowed_type = "number"
    list_str = f", list of {allowed_type}" if list_to_choice else ""
    raise Exception(
        f"Expected {allowed_type}, tuple of two {allowed_type}{list_str} or StochasticParameter for {name}, "
        f"got {type(param)}."
    )


@legacy
def handle_discrete_param(
    param: int
    | float
    | tuple[int | float, int | float]
    | list[int | float]
    | StochasticParameter,
    name: str,
    value_range: tuple[float | None, float | None] | Callable[[float | int], Any] | None = None,
    tuple_to_uniform: bool = True,
    list_to_choice: bool = True,
    allow_floats: bool = True,
    prefetch: bool = True,
) -> StochasticParameter:
    from .discrete import Choice, Deterministic, DiscreteUniform

    result = None

    if ia.is_single_integer(param) or (allow_floats and ia.is_single_float(param)):
        _check_value_range(param, name, value_range)  # type: ignore
        result = Deterministic(int(param))  # type: ignore
    elif tuple_to_uniform and isinstance(param, tuple):
        assert len(param) == 2, (
            f"Expected parameter '{name}' with type tuple to have exactly two "
            f"entries, but got {len(param)}."
        )
        is_valid_types = all(
            [ia.is_single_number(v) if allow_floats else ia.is_single_integer(v) for v in param]
        )
        assert is_valid_types, (
            "Expected parameter '{}' of type tuple to only contain {}, got {}.".format(
                name,
                "number" if allow_floats else "integer",
                [type(v) for v in param],
            )
        )

        _check_value_range(param[0], name, value_range)
        _check_value_range(param[1], name, value_range)
        result = DiscreteUniform(int(param[0]), int(param[1]))
    elif list_to_choice and ia.is_iterable(param) and not isinstance(param, tuple):
        is_valid_types = all(
            [ia.is_single_number(v) if allow_floats else ia.is_single_integer(v) for v in param]
        )
        assert is_valid_types, (
            "Expected iterable parameter '{}' to only contain {}, got {}.".format(
                name,
                "number" if allow_floats else "integer",
                [type(v) for v in param],
            )
        )

        for param_i in param:
            _check_value_range(param_i, name, value_range)
        result = Choice([int(param_i) for param_i in param])
    elif isinstance(param, StochasticParameter):
        result = param

    if result is not None:
        if prefetch:
            return _wrap_leafs_of_param_in_prefetchers(result, _NB_PREFETCH)
        return result

    allowed_type = "number" if allow_floats else "int"
    list_str = f", list of {allowed_type}" if list_to_choice else ""
    raise Exception(
        f"Expected {allowed_type}, tuple of two {allowed_type}{list_str} or StochasticParameter for {name}, "
        f"got {type(param)}."
    )


@legacy(version="0.4.0")
def handle_categorical_string_param(
    param: str | list[str] | StochasticParameter | Any,  # noqa: ANN401
    name: str,
    valid_values: Sequence[str] | None = None,
    prefetch: bool = True,
) -> StochasticParameter:
    from .discrete import Choice, Deterministic

    result = None

    if param == ia.ALL and valid_values is not None:
        result = Choice(list(valid_values))
    elif ia.is_string(param):
        if valid_values is not None:
            assert param in valid_values, (
                "Expected parameter '{}' to be one of: {}. Got: {}.".format(
                    name, ", ".join(list(valid_values)), param
                )
            )
        result = Deterministic(param)
    elif isinstance(param, list):
        assert all([ia.is_string(val) for val in param]), (
            "Expected list provided for parameter '{}' to only contain "
            "strings, got types: {}.".format(name, ", ".join([type(v).__name__ for v in param]))
        )
        if valid_values is not None:
            assert all([val in valid_values for val in param]), (
                "Expected list provided for parameter '{}' to only contain "
                "the following allowed strings: {}. Got strings: {}.".format(
                    name, ", ".join(valid_values), ", ".join(param)
                )
            )
        result = Choice(param)
    elif isinstance(param, StochasticParameter):
        result = param

    # we currently prefetch only 1k values here instead of 10k, because
    # strings might be rather long
    if result is not None:
        if prefetch:
            return _wrap_leafs_of_param_in_prefetchers(result, _NB_PREFETCH_STRINGS)
        return result

    raise Exception(
        "Expected parameter '{}' to be{} a string, a list of "
        "strings or StochasticParameter, got {}.".format(
            name,
            " imgaug2.ALL," if valid_values is not None else "",
            type(param).__name__,
        )
    )


@legacy
def handle_discrete_kernel_size_param(
    param: int
    | float
    | tuple[int | float, int | float]
    | list[int | float]
    | StochasticParameter,
    name: str,
    value_range: tuple[float | None, float | None] | Callable[[float | int], Any] | None = (
        1,
        None,
    ),
    allow_floats: bool = True,
    prefetch: bool = True,
) -> tuple[StochasticParameter | None, StochasticParameter | None]:
    from .discrete import Choice, Deterministic, DiscreteUniform


    result: tuple[StochasticParameter | None, StochasticParameter | None] = None, None
    if ia.is_single_integer(param) or (allow_floats and ia.is_single_float(param)):
        _check_value_range(param, name, value_range)  # type: ignore
        result = Deterministic(int(param)), None  # type: ignore
    elif isinstance(param, tuple):
        assert len(param) == 2, (
            f"Expected parameter '{name}' with type tuple to have exactly two "
            f"entries, but got {len(param)}."
        )
        if all([ia.is_single_integer(param_i) for param_i in param]) or (
            allow_floats and all([ia.is_single_float(param_i) for param_i in param])
        ):
            _check_value_range(param[0], name, value_range)  # type: ignore
            _check_value_range(param[1], name, value_range)  # type: ignore
            result = DiscreteUniform(int(param[0]), int(param[1])), None  # type: ignore
        elif all([isinstance(param_i, StochasticParameter) for param_i in param]):
            result = param[0], param[1]  # type: ignore
        else:
            handled = (
                handle_discrete_param(
                    param[0], f"{name}[0]", value_range, allow_floats=allow_floats  # type: ignore
                ),
                handle_discrete_param(
                    param[1], f"{name}[1]", value_range, allow_floats=allow_floats  # type: ignore
                ),
            )

            result = handled
    elif ia.is_iterable(param) and not isinstance(param, tuple):
        is_valid_types = all(
            [ia.is_single_number(v) if allow_floats else ia.is_single_integer(v) for v in param]  # type: ignore
        )
        assert is_valid_types, (
            "Expected iterable parameter '{}' to only contain {}, got {}.".format(
                name,
                "number" if allow_floats else "integer",
                [type(v) for v in param],  # type: ignore
            )
        )

        for param_i in param:  # type: ignore
            _check_value_range(param_i, name, value_range)
        result = Choice([int(param_i) for param_i in param]), None  # type: ignore
    elif isinstance(param, StochasticParameter):
        result = param, None

    result_pf = []
    for v in result:
        if v is not None and prefetch:
            v = _wrap_leafs_of_param_in_prefetchers(v, _NB_PREFETCH)
        result_pf.append(v)

    if result_pf != [None, None]:
        return tuple(result_pf)  # type: ignore

    raise Exception(
        f"Expected int, tuple/list with 2 entries or StochasticParameter. Got {type(param)}."
    )


@legacy
def handle_probability_param(
    param: float | int | bool | tuple[float, float] | list[float] | StochasticParameter,
    name: str,
    tuple_to_uniform: bool = False,
    list_to_choice: bool = False,
    prefetch: bool = True,
) -> StochasticParameter:
    from .continuous import Uniform
    from .discrete import Binomial, Choice, Deterministic

    eps = 1e-6

    result = None

    if param in [True, False, 0, 1]:
        result = Deterministic(int(param))
    elif ia.is_single_number(param):
        assert 0.0 <= param <= 1.0, (  # type: ignore
            f"Expected probability of parameter '{name}' to be in the interval "
            f"[0.0, 1.0], got {param:.4f}."
        )
        if 0.0 - eps < param < 0.0 + eps or 1.0 - eps < param < 1.0 + eps:  # type: ignore
            return Deterministic(int(np.round(param)))  # type: ignore
        result = Binomial(param)  # type: ignore
    elif tuple_to_uniform and isinstance(param, tuple):
        assert all([ia.is_single_number(v) for v in param]), (
            f"Expected parameter '{name}' of type tuple to only contain numbers, "
            f"got {[type(v) for v in param]}."
        )
        assert len(param) == 2, (
            f"Expected parameter '{name}' of type tuple to contain exactly 2 "
            f"entries, got {len(param)}."
        )
        assert 0 <= param[0] <= 1.0 and 0 <= param[1] <= 1.0, (
            f"Expected parameter '{name}' of type tuple to contain two "
            "probabilities in the interval [0.0, 1.0]. "
            f"Got values {param[0]:.4f} and {param[1]:.4f}."
        )
        result = Binomial(Uniform(param[0], param[1]))
    elif list_to_choice and ia.is_iterable(param):
        assert all([ia.is_single_number(v) for v in param]), (  # type: ignore
            f"Expected iterable parameter '{name}' to only contain numbers, "
            f"got {[type(v) for v in param]}."
        )
        assert all([0 <= p_i <= 1.0 for p_i in param]), (  # type: ignore
            "Expected iterable parameter '{}' to only contain probabilities "
            "in the interval [0.0, 1.0], got values {}.".format(
                name, ", ".join([f"{p_i:.4f}" for p_i in param])  # type: ignore
            )
        )
        result = Binomial(Choice(param))  # type: ignore
    elif isinstance(param, StochasticParameter):
        result = param

    if result is not None:
        if prefetch:
            return _wrap_leafs_of_param_in_prefetchers(result, _NB_PREFETCH)
        return result

    raise Exception(
        f"Expected boolean or number or StochasticParameter for {name}, got {type(param)}."
    )

def _assert_arg_is_stoch_param(arg_name: str, arg_value: Any) -> None:  # noqa: ANN401
    assert isinstance(arg_value, StochasticParameter), (
        f"Expected '{arg_name}' to be a StochasticParameter, got type {arg_value}."
    )


def handle_cval_arg(cval: ParamInput | Any) -> StochasticParameter:
    """Handle the `cval` parameter used in geometric transformations.

    This converts user-provided `cval` inputs into StochasticParameter instances.

    Parameters
    ----------
    cval : {float, tuple of float, list of float, StochasticParameter, "ALL"}
        The fill color value(s) to use.
        If ``"ALL"``, returns a uniform distribution over [0, 255].

    Returns
    -------
    StochasticParameter
        The normalized cval parameter.

    """
    from .continuous import Uniform

    if cval == ia.ALL:
        # Note: This is dynamically created per image. Consider making this
        # more efficient by caching or using a per-dtype approach.
        return Uniform(0, 255)  # skimage transform expects float
    return handle_continuous_param(
        cval, "cval", value_range=None, tuple_to_uniform=True, list_to_choice=True
    )


def handle_position_parameter(
    position: str
    | tuple[float | int | StochasticParameter, float | int | StochasticParameter]
    | StochasticParameter,
) -> StochasticParameter | tuple[StochasticParameter, StochasticParameter]:
    from .continuous import Normal, Uniform
    from .discrete import Deterministic
    from .transforms import Clip

    """Handle the `position` parameter used in augmenters.

    This converts various position formats into normalized StochasticParameter instances.

    Parameters
    ----------
    position : str or tuple or StochasticParameter
        Position specification. Can be one of:
        - "uniform": Random position from uniform distribution [0, 1]
        - "normal": Random position from normal distribution, clipped to [0, 1]
        - "center": Fixed position at center (0.5, 0.5)
        - "{h}-{v}": Directional position like "left-top", "center-center", etc.
        - tuple of two numbers/StochasticParameters: Explicit x, y positions
        - StochasticParameter: Custom parameter

    Returns
    -------
    StochasticParameter or tuple of StochasticParameter
        The normalized position parameter(s).

    """
    if position == "uniform":
        return Uniform(0.0, 1.0), Uniform(0.0, 1.0)
    if position == "normal":
        return (
            Clip(Normal(loc=0.5, scale=0.35 / 2), minval=0.0, maxval=1.0),
            Clip(Normal(loc=0.5, scale=0.35 / 2), minval=0.0, maxval=1.0),
        )
    if position == "center":
        return Deterministic(0.5), Deterministic(0.5)
    if ia.is_string(position) and re.match(
        r"^(left|center|right)-(top|center|bottom)$", position
    ):
        mapping = {"top": 0.0, "center": 0.5, "bottom": 1.0, "left": 0.0, "right": 1.0}
        return (
            Deterministic(mapping[position.split("-")[0]]),
            Deterministic(mapping[position.split("-")[1]]),
        )
    if isinstance(position, StochasticParameter):
        return position
    if isinstance(position, tuple):
        assert len(position) == 2, (
            "Expected tuple with two entries as position parameter. "
            f"Got {len(position)} entries with types {str([type(item) for item in position])}."
        )
        for item in position:
            if ia.is_single_number(item) and (item < 0 or item > 1.0):
                raise Exception(
                    "Both position values must be within the value range "
                    f"[0.0, 1.0]. Got type {type(item)} with value {item:.8f}."
                )
        position_list = [
            Deterministic(item) if ia.is_single_number(item) else item
            for item in position
        ]

        only_sparams = all(isinstance(item, StochasticParameter) for item in position_list)
        assert only_sparams, (
            "Expected tuple with two entries that are both either "
            f"StochasticParameter or float/int. Got types {str([type(item) for item in position_list])}."
        )
        return cast(
            tuple[StochasticParameter, StochasticParameter],
            (position_list[0], position_list[1]),
        )
    raise Exception(
        "Expected one of the following as position parameter: string "
        "'uniform', string 'normal', string 'center', a string matching "
        "regex ^(left|center|right)-(top|center|bottom)$, a single "
        "StochasticParameter or a tuple of two entries, both being either "
        "StochasticParameter or floats or int. Got instead type {} with "
        "content '{}'.".format(
            type(position),
            (str(position) if len(str(position)) < 20 else str(position)[0:20] + "..."),
        )
    )
