"""Decorators for marking function origins (legacy vs new).

These decorators help track which functions are from the original imgaug
codebase and which are new additions in imgaug2. They can be used for:
- Documentation generation
- Deprecation warnings
- Code auditing
- IDE hints (via function metadata)

Usage:
    from imgaug2.compat.markers import legacy, new

    @legacy
    def old_function():
        ...

    @legacy(version="0.2.0", deprecated=True, replacement="new_function")
    def old_function_with_details():
        ...

    @new
    def modern_function():
        ...

    @new(version="0.5.0")
    def modern_function_with_version():
        ...
"""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload

if TYPE_CHECKING:
    from typing import ParamSpec

    P = ParamSpec("P")

F = TypeVar("F", bound=Callable[..., Any])


class FunctionMarker:
    """Metadata container for function origin markers."""

    __slots__ = ("origin", "version", "deprecated", "replacement", "notes")

    def __init__(
        self,
        origin: str,
        version: str | None = None,
        deprecated: bool = False,
        replacement: str | None = None,
        notes: str | None = None,
    ) -> None:
        self.origin = origin  # "legacy" or "new"
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
    func: F,
    origin: str,
    version: str | None,
    deprecated: bool,
    replacement: str | None,
    notes: str | None,
) -> F:
    """Apply marker metadata to a function."""
    marker = FunctionMarker(
        origin=origin,
        version=version,
        deprecated=deprecated,
        replacement=replacement,
        notes=notes,
    )

    # Store marker as function attribute
    func.__imgaug2_marker__ = marker  # type: ignore[attr-defined]

    # Add deprecation warning if needed
    if deprecated:

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            msg = f"{func.__name__} is deprecated."
            if replacement:
                msg += f" Use {replacement} instead."
            if notes:
                msg += f" {notes}"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        wrapper.__imgaug2_marker__ = marker  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return func


# --- Legacy decorator ---


@overload
def legacy(func: F) -> F: ...


@overload
def legacy(
    *,
    version: str | None = None,
    deprecated: bool = False,
    replacement: str | None = None,
    notes: str | None = None,
) -> Callable[[F], F]: ...


def legacy(
    func: F | None = None,
    *,
    version: str | None = None,
    deprecated: bool = False,
    replacement: str | None = None,
    notes: str | None = None,
) -> F | Callable[[F], F]:
    """Mark a function as legacy (from original imgaug).

    Can be used with or without arguments:
        @legacy
        def my_func(): ...

        @legacy(version="0.2.0", deprecated=True)
        def old_func(): ...

    Parameters
    ----------
    version : str, optional
        Version when this function was introduced in original imgaug.
    deprecated : bool, default False
        If True, emits a DeprecationWarning when called.
    replacement : str, optional
        Name of the function that should be used instead.
    notes : str, optional
        Additional notes about the function.

    """
    if func is not None:
        # Called without arguments: @legacy
        return _apply_marker(func, "legacy", version, deprecated, replacement, notes)

    # Called with arguments: @legacy(...)
    def decorator(fn: F) -> F:
        return _apply_marker(fn, "legacy", version, deprecated, replacement, notes)

    return decorator


# --- New decorator ---


@overload
def new(func: F) -> F: ...


@overload
def new(
    *,
    version: str | None = None,
    notes: str | None = None,
) -> Callable[[F], F]: ...


def new(
    func: F | None = None,
    *,
    version: str | None = None,
    notes: str | None = None,
) -> F | Callable[[F], F]:
    """Mark a function as new (added in imgaug2).

    Can be used with or without arguments:
        @new
        def my_func(): ...

        @new(version="0.5.0")
        def added_in_05(): ...

    Parameters
    ----------
    version : str, optional
        Version when this function was added to imgaug2.
    notes : str, optional
        Additional notes about the function.

    """
    if func is not None:
        # Called without arguments: @new
        return _apply_marker(func, "new", version, False, None, notes)

    # Called with arguments: @new(...)
    def decorator(fn: F) -> F:
        return _apply_marker(fn, "new", version, False, None, notes)

    return decorator


# --- Utility functions ---


def get_marker(func: Callable[..., Any]) -> FunctionMarker | None:
    """Get the marker for a function, if any."""
    return getattr(func, "__imgaug2_marker__", None)


def is_legacy(func: Callable[..., Any]) -> bool:
    """Check if a function is marked as legacy."""
    marker = get_marker(func)
    return marker is not None and marker.origin == "legacy"


def is_new(func: Callable[..., Any]) -> bool:
    """Check if a function is marked as new."""
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
