from __future__ import annotations

from typing import Literal

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import ParamInput, RNGInput
from imgaug2.compat.markers import legacy

from ._types import CorruptionFunc, IntArray


@legacy(version="0.4.0")
class _ImgcorruptAugmenterBase(meta.Augmenter):
    def __init__(
        self,
        func: CorruptionFunc,
        severity: ParamInput = 1,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.func = func
        self.severity = iap.handle_discrete_param(
            severity,
            "severity",
            value_range=(1, 5),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        if batch.images is None:
            return batch

        severities, seeds = self._draw_samples(len(batch.images), random_state=random_state)

        for image, severity, seed in zip(batch.images, severities, seeds, strict=True):
            image[...] = self.func(image, severity=int(severity), seed=int(seed))

        return batch

    @legacy(version="0.4.0")
    def _draw_samples(self, nb_rows: int, random_state: iarandom.RNG) -> tuple[IntArray, IntArray]:
        severities = self.severity.draw_samples((nb_rows,), random_state=random_state)
        seeds = random_state.generate_seeds_(nb_rows)

        return severities, seeds

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[iap.StochasticParameter]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.severity]


