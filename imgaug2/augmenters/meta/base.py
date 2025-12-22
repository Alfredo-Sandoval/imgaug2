"""Augmenter base class and core batch APIs."""

from __future__ import annotations

from abc import ABCMeta
from typing import Literal

import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.augmenters._typing import RNGInput

from ._context import _maybe_deterministic_ctx
from ._augmenter_batches import AugmenterBatchMixin
from ._augmenter_augmentables import AugmenterAugmentablesMixin
from ._augmenter_highlevel import AugmenterHighLevelMixin
from ._augmenter_deterministic import AugmenterDeterministicMixin
from ._augmenter_children import AugmenterChildrenMixin


__all__ = ["Augmenter", "_maybe_deterministic_ctx"]


class Augmenter(
    AugmenterBatchMixin,
    AugmenterAugmentablesMixin,
    AugmenterHighLevelMixin,
    AugmenterDeterministicMixin,
    AugmenterChildrenMixin,
    metaclass=ABCMeta,
):
    """
    Base class for Augmenter objects.
    All augmenters derive from this class.

    Parameters
    ----------
    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        Seed to use for this augmenter's random number generator (RNG) or
        alternatively an RNG itself. Setting this parameter allows to
        control/influence the random number sampling of this specific
        augmenter without affecting other augmenters. Usually, there is no
        need to set this parameter.

            * If ``None``: The global RNG is used (shared by all
              augmenters).
            * If ``int``: The value will be used as a seed for a new
              :class:`~imgaug2.random.RNG` instance.
            * If :class:`~imgaug2.random.RNG`: The ``RNG`` instance will be
              used without changes.
            * If :class:`~imgaug2.random.Generator`: A new
              :class:`~imgaug2.random.RNG` instance will be
              created, containing that generator.
            * If :class:`~imgaug2.random.bit_generator.BitGenerator`: Will
              be wrapped in a :class:`~imgaug2.random.Generator`. Then
              similar behaviour to :class:`~imgaug2.random.Generator`
              parameters.
            * If :class:`~imgaug2.random.SeedSequence`: Will
              be wrapped in a new bit generator and
              :class:`~imgaug2.random.Generator`. Then
              similar behaviour to :class:`~imgaug2.random.Generator`
              parameters.

        If a new bit generator has to be created, it will be an instance
        of :class:`numpy.random.SFC64`.


    name : None or str, optional
        Name given to the Augmenter instance. This name is used when
        converting the instance to a string, e.g. for ``print`` statements.
        It is also used for ``find``, ``remove`` or similar operations
        on augmenters with children.
        If ``None``, ``UnnamedX`` will be used as the name, where ``X``
        is the Augmenter's class name.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    """

    def __init__(
        self,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        """Create a new Augmenter instance."""
        super().__init__()

        assert name is None or ia.is_string(name), (
            f"Expected name to be None or string-like, got {type(name)}."
        )
        if name is None:
            self.name = f"Unnamed{self.__class__.__name__}"
        else:
            self.name = name

        if deterministic != "deprecated":
            ia.warn_deprecated(
                "The parameter `deterministic` is deprecated "
                "in `imgaug2.augmenters.meta.Augmenter`. Use "
                "`.to_deterministic()` to switch into deterministic mode.",
                stacklevel=4,
            )
            assert ia.is_single_bool(deterministic), (
                f"Expected deterministic to be a boolean, got {type(deterministic)}."
            )
        else:
            deterministic = False

        self.deterministic = deterministic

        if random_state != "deprecated":
            assert seed is None, "Cannot set both `seed` and `random_state`."
            seed = random_state

        if deterministic and seed is None:
            # Usually if None is provided, the global RNG will be used.
            # In case of deterministic mode we most likely rather want a local
            # RNG, which is here created.
            self.random_state = iarandom.RNG.create_pseudo_random_()
        else:
            # self.random_state = iarandom.normalize_rng_(random_state)
            self.random_state = iarandom.RNG.create_if_not_rng_(seed)

        self.activated = True
