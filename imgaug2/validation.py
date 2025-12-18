"""Helper functions to validate input data and produce error messages."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import imgaug2.imgaug as ia


def convert_iterable_to_string_of_types(iterable_var: Iterable[Any]) -> str:
    """Convert an iterable of values to a string of their types.

    Parameters
    ----------
    iterable_var : iterable
        An iterable of variables, e.g. a list of integers.

    Returns
    -------
    str
        String representation of the types in `iterable_var`. One per item
        in `iterable_var`. Separated by commas.

    """
    types = [str(type(var_i)) for var_i in iterable_var]
    return ", ".join(types)


def is_iterable_of(iterable_var: Any, classes: type | Iterable[type]) -> bool:
    """Check whether `iterable_var` contains only instances of given classes.

    Parameters
    ----------
    iterable_var : iterable
        An iterable of items that will be matched against `classes`.

    classes : type or iterable of type
        One or more classes that each item in `var` must be an instanceof.
        If this is an iterable, a single match per item is enough.

    Returns
    -------
    bool
        Whether `var` only contains instances of `classes`.
        If `var` was empty, ``True`` will be returned.

    """
    if not ia.is_iterable(iterable_var):
        return False

    classes_tuple = (classes,) if isinstance(classes, type) else tuple(classes)

    for var_i in iterable_var:
        if not isinstance(var_i, classes_tuple):
            return False

    return True


def assert_is_iterable_of(iterable_var: Any, classes: type | Iterable[type]) -> None:
    """Assert that `iterable_var` only contains instances of given classes.

    Parameters
    ----------
    iterable_var : iterable
        See :func:`~imgaug2.validation.is_iterable_of`.

    classes : type or iterable of type
        See :func:`~imgaug2.validation.is_iterable_of`.

    """
    valid = is_iterable_of(iterable_var, classes)
    if not valid:
        if isinstance(classes, type):
            expected_types_str = classes.__name__
        else:
            expected_types_str = ", ".join([class_.__name__ for class_ in classes])
        if not ia.is_iterable(iterable_var):
            raise AssertionError(
                f"Expected an iterable of the following types: {expected_types_str}. "
                f"Got instead a single instance of: {type(iterable_var).__name__}."
            )

        raise AssertionError(
            f"Expected an iterable of the following types: {expected_types_str}. "
            f"Got an iterable of types: {convert_iterable_to_string_of_types(iterable_var)}."
        )
