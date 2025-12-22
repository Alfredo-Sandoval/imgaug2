"""Context helpers for meta augmenters."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

import imgaug2.random as iarandom

if TYPE_CHECKING:
    from .base import Augmenter


class _maybe_deterministic_ctx:
    """Context that resets an RNG to its initial state upon exit.

    This allows to execute some sampling functions and leave the code block
    with the used RNG in the same state as before.

    Parameters
    ----------
    random_state : imgaug2.random.RNG or imgaug2.augmenters.meta.Augmenter
        The RNG to reset. If this is an augmenter, then the augmenter's
        RNG will be used.

    deterministic : None or bool
        Whether to reset the RNG upon exit (``True``) or not (``False``).
        Allowed to be ``None`` iff `random_state` was an augmenter, in which
        case that augmenter's ``deterministic`` attribute will be used.

    """

    def __init__(
        self, random_state: iarandom.RNG | Augmenter, deterministic: bool | None = None
    ) -> None:
        if deterministic is None:
            augmenter = random_state
            self.random_state = augmenter.random_state
            self.deterministic = augmenter.deterministic
        else:
            assert deterministic is not None, "Expected boolean as `deterministic`, got None."
            self.random_state = random_state
            self.deterministic = deterministic
        self.old_state = None

    def __enter__(self) -> None:
        if self.deterministic:
            self.old_state = self.random_state.state

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        exception_traceback: TracebackType | None,
    ) -> None:
        if self.old_state is not None:
            self.random_state.state = self.old_state
