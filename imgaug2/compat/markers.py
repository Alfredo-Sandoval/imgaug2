"""Function origin markers for tracking legacy and new code.

This module provides decorators to mark functions and classes by their origin:
legacy code from the original imgaug library or new additions in imgaug2.
Markers support deprecation warnings, version tracking, and metadata for
documentation generation and code auditing.

Examples
--------
Mark legacy function without deprecation:

    >>> @legacy
    ... def old_function():
    ...     pass

Mark deprecated legacy function with replacement:

    >>> @legacy(version="0.2.0", deprecated=True, replacement="new_function")
    ... def old_function_with_details():
    ...     pass

Mark new function with version:

    >>> @new(version="0.5.0")
    ... def modern_function():
    ...     pass

Check function origin:

    >>> marker = get_marker(old_function)
    >>> marker.origin
    'legacy'
    >>> is_legacy(old_function)
    True
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, ParamSpec, Protocol, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class _NamedCallable(Protocol[P, R]):
    __name__: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class FunctionMarker:
    """Metadata container for function origin markers.

    Attributes
    ----------
    origin : str
        Origin of the function ('legacy' or 'new').
    version : str or None
        Version when function was introduced or deprecated.
    deprecated : bool
        Whether function is deprecated.
    replacement : str or None
        Name of replacement function if deprecated.
    notes : str or None
        Additional notes about the function.
    """

    __slots__ = ("origin", "version", "deprecated", "replacement", "notes")

    def __init__(
        self,
        origin: str,
        version: str | None = None,
        deprecated: bool = False,
        replacement: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Initialize function marker.

        Parameters
        ----------
        origin : str
            Origin of the function ('legacy' or 'new').
        version : str or None, default=None
            Version when function was introduced or deprecated.
        deprecated : bool, default=False
            Whether function is deprecated.
        replacement : str or None, default=None
            Name of replacement function if deprecated.
        notes : str or None, default=None
            Additional notes about the function.
        """
        self.origin = origin
        self.version = version
        self.deprecated = deprecated
        self.replacement = replacement
        self.notes = notes

    def __repr__(self) -> str:
        parts = [f"origin={self.origin!r}"]
        if self.version:
            parts.append(f"version={self.version!r}")
        if self.deprecated:
            parts.append("deprecated=True")
        if self.replacement:
            parts.append(f"replacement={self.replacement!r}")
        return f"FunctionMarker({', '.join(parts)})"


def _apply_marker(
    func: _NamedCallable[P, R] | property,
    origin: str,
    version: str | None,
    deprecated: bool,
    replacement: str | None,
    notes: str | None,
) -> Callable[P, R] | property:
    """Apply marker metadata to a function.

    Parameters
    ----------
    func : callable or property
        Function or property to mark.
    origin : str
        Origin identifier ('legacy' or 'new').
    version : str or None
        Version information.
    deprecated : bool
        Whether function is deprecated.
    replacement : str or None
        Replacement function name.
    notes : str or None
        Additional notes.

    Returns
    -------
    callable or property
        Marked function/property, wrapped with deprecation warning if deprecated.
    """
    marker = FunctionMarker(
        origin=origin,
        version=version,
        deprecated=deprecated,
        replacement=replacement,
        notes=notes,
    )

    if isinstance(func, property):
        if deprecated:
            # We need to wrap the getter
            original_getter = func.fget
            if original_getter is None:
                raise ValueError("Cannot deprecate a property without a getter.")

            @functools.wraps(original_getter)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                msg = f"Property {original_getter.__name__} is deprecated."
                if replacement:
                    msg += f" Use {replacement} instead."
                if notes:
                    msg += f" {notes}"
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                return original_getter(*args, **kwargs)

            setattr(wrapper, "__imgaug2_marker__", marker)
            return property(wrapper, func.fset, func.fdel, func.__doc__)

        if func.fget is not None:
            setattr(func.fget, "__imgaug2_marker__", marker)
        return func

    setattr(func, "__imgaug2_marker__", marker)

    if deprecated:

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            msg = f"{func.__name__} is deprecated."
            if replacement:
                msg += f" Use {replacement} instead."
            if notes:
                msg += f" {notes}"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        setattr(wrapper, "__imgaug2_marker__", marker)
        return wrapper

    return func


# --- Legacy decorator ---


@overload
def legacy(func: type[T]) -> type[T]: ...


@overload
def legacy(func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def legacy(
    *,
    version: str | None = None,
    deprecated: bool = False,
    replacement: str | None = None,
    notes: str | None = None,
) -> Callable[[type[T]], type[T]]: ...


@overload
def legacy(
    *,
    version: str | None = None,
    deprecated: bool = False,
    replacement: str | None = None,
    notes: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def legacy(
    func: Callable[P, R] | None = None,
    *,
    version: str | None = None,
    deprecated: bool = False,
    replacement: str | None = None,
    notes: str | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a function as legacy code from original imgaug.

    This decorator can be used with or without arguments to mark functions
    from the original imgaug codebase. Optionally marks them as deprecated.

    Parameters
    ----------
    func : callable or None, default=None
        Function to mark (when used without arguments).
    version : str or None, default=None
        Version when this function was introduced in original imgaug.
    deprecated : bool, default=False
        If True, emits DeprecationWarning when called.
    replacement : str or None, default=None
        Name of the function that should be used instead.
    notes : str or None, default=None
        Additional notes about the function.

    Returns
    -------
    callable or decorator
        Marked function if used without arguments, decorator otherwise.

    Examples
    --------
    Simple usage without arguments:

        >>> @legacy
        ... def old_function():
        ...     pass

    With deprecation:

        >>> @legacy(version="0.2.0", deprecated=True, replacement="new_func")
        ... def old_func():
        ...     pass
    """
    if func is not None:
        return _apply_marker(func, "legacy", version, deprecated, replacement, notes)

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        return _apply_marker(fn, "legacy", version, deprecated, replacement, notes)

    return decorator


@overload
def new(func: type[T]) -> type[T]: ...


@overload
def new(func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def new(
    *,
    version: str | None = None,
    notes: str | None = None,
) -> Callable[[type[T]], type[T]]: ...


@overload
def new(
    *,
    version: str | None = None,
    notes: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def new(
    func: Callable[P, R] | None = None,
    *,
    version: str | None = None,
    notes: str | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a function as new code added in imgaug2.

    This decorator marks functions that are new additions in imgaug2, not
    present in the original imgaug library.

    Parameters
    ----------
    func : callable or None, default=None
        Function to mark (when used without arguments).
    version : str or None, default=None
        Version when this function was added to imgaug2.
    notes : str or None, default=None
        Additional notes about the function.

    Returns
    -------
    callable or decorator
        Marked function if used without arguments, decorator otherwise.

    Examples
    --------
    Simple usage:

        >>> @new
        ... def modern_function():
        ...     pass

    With version tracking:

        >>> @new(version="0.5.0")
        ... def added_in_05():
        ...     pass
    """
    if func is not None:
        return _apply_marker(func, "new", version, False, None, notes)

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        return _apply_marker(fn, "new", version, False, None, notes)

    return decorator


def get_marker(func: Callable[..., object] | property) -> FunctionMarker | None:
    """Get the marker metadata for a function.

    Parameters
    ----------
    func : callable or property
        Function or property to inspect.

    Returns
    -------
    FunctionMarker or None
        Marker metadata if present, otherwise None.
    """
    if isinstance(func, property):
        return getattr(func.fget, "__imgaug2_marker__", None)
    return getattr(func, "__imgaug2_marker__", None)


def is_legacy(func: Callable[..., object]) -> bool:
    """Check if a function is marked as legacy.

    Parameters
    ----------
    func : callable
        Function to check.

    Returns
    -------
    bool
        True if function has legacy marker, False otherwise.
    """
    marker = get_marker(func)
    return marker is not None and marker.origin == "legacy"


def is_new(func: Callable[..., object]) -> bool:
    """Check if a function is marked as new.

    Parameters
    ----------
    func : callable
        Function to check.

    Returns
    -------
    bool
        True if function has new marker, False otherwise.
    """
    marker = get_marker(func)
    return marker is not None and marker.origin == "new"


__all__ = [
    "FunctionMarker",
    "get_marker",
    "is_legacy",
    "is_new",
    "legacy",
    "new",
]
