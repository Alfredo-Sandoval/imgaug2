"""Deprecation utilities shared across imgaug2 modules."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Literal, ParamSpec, TypeVar

from imgaug2.compat.markers import legacy

_P = ParamSpec("_P")
_R = TypeVar("_R")


@legacy
class DeprecationWarning(Warning):
    """Warning for deprecated calls.

    We define our own DeprecationWarning subclass so that deprecations in
    imgaug2 are consistently visible and distinguishable.

    """


@legacy
def warn(msg: str, category: type[Warning] = UserWarning, stacklevel: int = 2) -> None:
    """Generate a a warning with stacktrace.

    Parameters
    ----------
    msg : str
        The message of the warning.

    category : class
        The class of the warning to produce.

    stacklevel : int, optional
        How many steps above this function to "jump" in the stacktrace when
        displaying file and line number of the error message.
        Usually ``2``.

    """
    warnings.warn(msg, category=category, stacklevel=stacklevel)


@legacy
def warn_deprecated(msg: str, stacklevel: int = 2) -> None:
    """Generate a non-silent deprecation warning with stacktrace.

    The used warning is ``imgaug2.imgaug2.DeprecationWarning``.

    Parameters
    ----------
    msg : str
        The message of the warning.

    stacklevel : int, optional
        How many steps above this function to "jump" in the stacktrace when
        displaying file and line number of the error message.
        Usually ``2``

    """
    warn(msg, category=DeprecationWarning, stacklevel=stacklevel)


@legacy
class deprecated:
    """Decorator to mark deprecated functions with warning.

    Adapted from
    <https://github.com/scikit-image/scikit-image/blob/master/skimage/_shared/utils.py>.

    Parameters
    ----------
    alt_func : None or str, optional
        If given, tell user what function to use instead.

    behavior : {'warn', 'raise'}, optional
        Behavior during call to deprecated function: ``warn`` means that the
        user is warned that the function is deprecated; ``raise`` means that
        an error is raised.

    removed_version : None or str, optional
        The package version in which the deprecated function will be removed.

    comment : None or str, optional
        An optional comment that will be appended to the warning message.

    """

    def __init__(
        self,
        alt_func: str | None = None,
        behavior: Literal["warn", "raise"] = "warn",
        removed_version: str | None = None,
        comment: str | None = None,
    ) -> None:
        self.alt_func = alt_func
        self.behavior = behavior
        self.removed_version = removed_version
        self.comment = comment

    def __call__(self, func: Callable[_P, _R]) -> Callable[_P, _R]:
        alt_msg = None
        if self.alt_func is not None:
            alt_msg = f"Use ``{self.alt_func}`` instead."

        rmv_msg = None
        if self.removed_version is not None:
            rmv_msg = f"It will be removed in version {self.removed_version}."

        comment_msg = None
        if self.comment is not None and len(self.comment) > 0:
            comment_msg = "{}.".format(self.comment.rstrip(". "))

        addendum = " ".join(
            [submsg for submsg in [alt_msg, rmv_msg, comment_msg] if submsg is not None]
        )

        @functools.wraps(func)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            # getargpec() is deprecated

            # TODO add class name if class method
            import inspect

            arg_names = inspect.getfullargspec(func)[0]

            func_name = getattr(func, "__name__", func.__class__.__name__)

            if "self" in arg_names or "cls" in arg_names:
                main_msg = (
                    f"Method ``{args[0].__class__.__name__}.{func_name}()`` is deprecated."
                )
            else:
                main_msg = f"Function ``{func_name}()`` is deprecated."

            msg = (main_msg + " " + addendum).rstrip(" ").replace("``", "`")

            if self.behavior == "warn":
                # Route through `imgaug2.imgaug.warn_deprecated` for backwards
                # compatibility (tests and downstream code often patch that).
                from imgaug2 import imgaug as ia

                ia.warn_deprecated(msg, stacklevel=3)
            elif self.behavior == "raise":
                raise DeprecationWarning(msg)
            return func(*args, **kwargs)

        # modify doc string to display deprecation warning
        doc = "**Deprecated**. " + addendum
        if wrapped.__doc__ is None:
            wrapped.__doc__ = doc
        else:
            wrapped.__doc__ = doc + "\n\n    " + wrapped.__doc__

        return wrapped
