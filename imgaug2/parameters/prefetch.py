from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from imgaug2.compat.markers import legacy

from . import base as _base

_NB_PREFETCH = 10000
_NB_PREFETCH_STRINGS = 1000

StochasticParameter = _base.StochasticParameter
AutoPrefetcher = _base.AutoPrefetcher


@legacy(version="0.5.0")
def _prefetchable(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def _inner(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        param = func(*args, **kwargs)
        return _wrap_leafs_of_param_in_prefetchers(param, _NB_PREFETCH)

    return _inner


@legacy(version="0.5.0")
def _prefetchable_str(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def _inner(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        param = func(*args, **kwargs)
        return _wrap_leafs_of_param_in_prefetchers(param, _NB_PREFETCH_STRINGS)

    return _inner


@legacy(version="0.5.0")
def _wrap_param_in_prefetchers(param: Any, nb_prefetch: int) -> Any:  # noqa: ANN401
    for key, value in param.__dict__.items():
        if isinstance(value, StochasticParameter):
            param.__dict__[key] = _wrap_param_in_prefetchers(value, nb_prefetch)

    if param.prefetchable:
        return AutoPrefetcher(param, nb_prefetch)
    return param


@legacy(version="0.5.0")
def _wrap_leafs_of_param_in_prefetchers(param: Any, nb_prefetch: int) -> Any:  # noqa: ANN401
    param_wrapped, _did_wrap_any_child = _wrap_leafs_of_param_in_prefetchers_recursive(
        param, nb_prefetch
    )
    return param_wrapped


@legacy(version="0.5.0")
def _wrap_leafs_of_param_in_prefetchers_recursive(
    param: Any, nb_prefetch: int  # noqa: ANN401
) -> tuple[Any, bool]:  # noqa: ANN401
    # Do not descent into AutoPrefetcher, otherwise we risk turning an
    # AutoPrefetcher(X) into AutoPrefetcher(AutoPrefetcher(X)) if X is
    # prefetchable
    if isinstance(param, AutoPrefetcher):
        # report did_wrap_any_child=True here, so that parent parameters
        # are not wrapped in prefetchers, which could lead to ugly scenarios
        # like AutoPrefetcher(Normal(AutoPrefetcher(Uniform(-1.0, 1.0))),
        return param, True

    if isinstance(param, (list, tuple)):
        result = []
        did_wrap_any_child = False
        for param_i in param:
            param_i_wrapped, did_wrap_any_child_i = _wrap_leafs_of_param_in_prefetchers_recursive(
                param_i, nb_prefetch
            )
            result.append(param_i_wrapped)
            did_wrap_any_child = did_wrap_any_child or did_wrap_any_child_i

        if not did_wrap_any_child:
            return param, False
        if isinstance(param, tuple):
            return tuple(result), did_wrap_any_child
        return result, did_wrap_any_child

    if not isinstance(param, StochasticParameter):
        return param, False

    did_wrap_any_child = False
    for key, value in param.__dict__.items():
        param_wrapped, did_wrap_i = _wrap_leafs_of_param_in_prefetchers_recursive(
            value, nb_prefetch
        )

        param.__dict__[key] = param_wrapped
        did_wrap_any_child = did_wrap_any_child or did_wrap_i

    if param.prefetchable and not did_wrap_any_child and _base._PREFETCHING_ENABLED:
        return AutoPrefetcher(param, nb_prefetch), True
    return param, did_wrap_any_child


@legacy(version="0.5.0")
def toggle_prefetching(enabled: bool) -> None:
    """Toggle prefetching on or off.


    Parameters
    ----------
    enabled : bool
        Whether enabled is activated (``True``) or off (``False``).

    """
    _base._PREFETCHING_ENABLED = enabled


@legacy(version="0.5.0")
class toggled_prefetching:
    """Context that toggles prefetching on or off depending on a flag.


    Parameters
    ----------
    enabled : bool
        Whether enabled is activated (``True``) or off (``False``).

    """

    @legacy(version="0.5.0")
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._old_state = None

    @legacy(version="0.5.0")
    def __enter__(self) -> None:
        self._old_state = _base._PREFETCHING_ENABLED
        _base._PREFETCHING_ENABLED = self.enabled

    @legacy(version="0.5.0")
    def __exit__(
        self,
        exception_type: Any,  # noqa: ANN401
        exception_value: Any,  # noqa: ANN401
        exception_traceback: Any,  # noqa: ANN401
    ) -> None:
        _base._PREFETCHING_ENABLED = self._old_state


@legacy(version="0.5.0")
class no_prefetching(toggled_prefetching):
    """Context that deactviates prefetching.


    """

    @legacy(version="0.5.0")
    def __init__(self) -> None:
        super().__init__(False)
