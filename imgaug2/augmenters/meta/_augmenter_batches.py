from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, cast

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import Batch, UnnormalizedBatch, _BatchInAugmentation
from imgaug2.compat.markers import legacy

from ._context import _maybe_deterministic_ctx

if TYPE_CHECKING:
    from .base import Augmenter


class AugmenterBatchMixin:
    def augment_batches(
        self,
        batches: Batch | UnnormalizedBatch | Iterable[object],
        hooks: ia.HooksImages | None = None,
        background: bool = False,
    ) -> Iterator[object]:
        """Augment multiple batches.

        In contrast to other ``augment_*`` method, this one **yields**
        batches instead of returning a full list. This is more suited
        for most training loops.

        This method also also supports augmentation on multiple cpu cores,
        activated via the `background` flag. If the `background` flag
        is activated, an instance of :class:`~imgaug2.multicore.Pool` will
        be spawned using all available logical CPU cores and an
        ``output_buffer_size`` of ``C*10``, where ``C`` is the number of
        logical CPU cores. I.e. a maximum of ``C*10`` batches will be somewhere
        in the augmentation pipeline (or waiting to be retrieved by downstream
        functions) before this method temporarily stops the loading of new
        batches from `batches`.

        Parameters
        ----------
        batches : imgaug2.augmentables.batches.Batch or imgaug2.augmentables.batches.UnnormalizedBatch or iterable of imgaug2.augmentables.batches.Batch or iterable of imgaug2.augmentables.batches.UnnormalizedBatch
            A single batch or a list of batches to augment.

        hooks : None or imgaug2.HooksImages, optional
            HooksImages object to dynamically interfere with the augmentation
            process.

        background : bool, optional
            Whether to augment the batches in background processes.
            If ``True``, hooks can currently not be used as that would require
            pickling functions.
            Note that multicore augmentation distributes the batches onto
            different CPU cores. It does *not* split the data *within* batches.
            It is therefore *not* sensible to use ``background=True`` to
            augment a single batch. Only use it for multiple batches.
            Note also that multicore augmentation needs some time to start. It
            is therefore not recommended to use it for very few batches.

        Yields
        -------
        imgaug2.augmentables.batches.Batch or imgaug2.augmentables.batches.UnnormalizedBatch or iterable of imgaug2.augmentables.batches.Batch or iterable of imgaug2.augmentables.batches.UnnormalizedBatch
            Augmented batches.

        """
        if isinstance(batches, (Batch, UnnormalizedBatch)):
            batches = [batches]

        assert (
            ia.is_iterable(batches) and not ia.is_np_array(batches) and not ia.is_string(batches)
        ) or ia.is_generator(batches), (
            "Expected either (a) an iterable that is not an array or a "
            f"string or (b) a generator. Got: {type(batches)}"
        )

        if background:
            assert hooks is None, "Hooks can not be used when background augmentation is activated."

        def _normalize_batch(idx: int, batch: object) -> tuple[Batch, str]:
            if isinstance(batch, Batch):
                batch_copy = batch.deepcopy()
                batch_copy.data = (idx, batch_copy.data)
                batch_normalized = batch_copy
                batch_orig_dt = "imgaug2.Batch"
            elif isinstance(batch, UnnormalizedBatch):
                batch_copy = batch.to_normalized_batch()
                batch_copy.data = (idx, batch_copy.data)
                batch_normalized = batch_copy
                batch_orig_dt = "imgaug2.UnnormalizedBatch"
            elif isinstance(batch, np.ndarray):
                assert batch.ndim in (3, 4), (
                    "Expected numpy array to have shape (N, H, W) or "
                    f"(N, H, W, C), got {batch.shape}."
                )
                batch_normalized = Batch(images=batch, data=(idx,))
                batch_orig_dt = "numpy_array"
            elif isinstance(batch, list):
                if len(batch) == 0:
                    batch_normalized = Batch(data=(idx,))
                    batch_orig_dt = "empty_list"
                elif ia.is_np_array(batch[0]):
                    batch_normalized = Batch(images=batch, data=(idx,))
                    batch_orig_dt = "list_of_numpy_arrays"
                elif isinstance(batch[0], ia.HeatmapsOnImage):
                    batch_normalized = Batch(heatmaps=batch, data=(idx,))
                    batch_orig_dt = "list_of_imgaug.HeatmapsOnImage"
                elif isinstance(batch[0], ia.SegmentationMapsOnImage):
                    batch_normalized = Batch(segmentation_maps=batch, data=(idx,))
                    batch_orig_dt = "list_of_imgaug.SegmentationMapsOnImage"
                elif isinstance(batch[0], ia.KeypointsOnImage):
                    batch_normalized = Batch(keypoints=batch, data=(idx,))
                    batch_orig_dt = "list_of_imgaug.KeypointsOnImage"
                elif isinstance(batch[0], ia.BoundingBoxesOnImage):
                    batch_normalized = Batch(bounding_boxes=batch, data=(idx,))
                    batch_orig_dt = "list_of_imgaug.BoundingBoxesOnImage"
                elif isinstance(batch[0], ia.PolygonsOnImage):
                    batch_normalized = Batch(polygons=batch, data=(idx,))
                    batch_orig_dt = "list_of_imgaug.PolygonsOnImage"
                else:
                    raise Exception(
                        "Unknown datatype in batch[0]. Expected numpy array "
                        "or imgaug2.HeatmapsOnImage or "
                        "imgaug2.SegmentationMapsOnImage or "
                        "imgaug2.KeypointsOnImage or "
                        "imgaug2.BoundingBoxesOnImage, "
                        "or imgaug2.PolygonsOnImage, "
                        f"got {type(batch[0])}."
                    )
            else:
                raise Exception(
                    "Unknown datatype of batch. Expected imgaug2.Batch or "
                    "imgaug2.UnnormalizedBatch or "
                    "numpy array or list of (numpy array or "
                    "imgaug2.HeatmapsOnImage or "
                    "imgaug2.SegmentationMapsOnImage "
                    "or imgaug2.KeypointsOnImage or "
                    "imgaug2.BoundingBoxesOnImage or "
                    f"imgaug2.PolygonsOnImage). Got {type(batch)}."
                )

            if batch_orig_dt not in ["imgaug2.Batch", "imgaug2.UnnormalizedBatch"]:
                ia.warn_deprecated(
                    "Received an input in augment_batches() that was not an "
                    "instance of imgaug2.augmentables.batches.Batch "
                    "or imgaug2.augmentables.batches.UnnormalizedBatch, but "
                    f"instead {batch_orig_dt}. This is deprecated. Use augment() for such "
                    "data or wrap it in a Batch instance."
                )
            return batch_normalized, batch_orig_dt

        # unnormalization of non-Batch/UnnormalizedBatch is for legacy support
        def _unnormalize_batch(batch_aug: Batch, batch_orig: object, batch_orig_dt: str) -> object:
            if batch_orig_dt == "imgaug2.Batch":
                batch_unnormalized = batch_aug
                # change (i, .data) back to just .data
                batch_unnormalized.data = batch_unnormalized.data[1]
            elif batch_orig_dt == "imgaug2.UnnormalizedBatch":
                # change (i, .data) back to just .data
                batch_aug.data = batch_aug.data[1]

                batch_orig_unnormalized = cast(UnnormalizedBatch, batch_orig)
                batch_unnormalized = batch_orig_unnormalized.fill_from_augmented_normalized_batch(
                    batch_aug
                )
            elif batch_orig_dt == "numpy_array":
                batch_unnormalized = batch_aug.images_aug
            elif batch_orig_dt == "empty_list":
                batch_unnormalized = []
            elif batch_orig_dt == "list_of_numpy_arrays":
                batch_unnormalized = batch_aug.images_aug
            elif batch_orig_dt == "list_of_imgaug.HeatmapsOnImage":
                batch_unnormalized = batch_aug.heatmaps_aug
            elif batch_orig_dt == "list_of_imgaug.SegmentationMapsOnImage":
                batch_unnormalized = batch_aug.segmentation_maps_aug
            elif batch_orig_dt == "list_of_imgaug.KeypointsOnImage":
                batch_unnormalized = batch_aug.keypoints_aug
            elif batch_orig_dt == "list_of_imgaug.BoundingBoxesOnImage":
                batch_unnormalized = batch_aug.bounding_boxes_aug
            else:  # only option left
                assert batch_orig_dt == "list_of_imgaug.PolygonsOnImage", (
                    f"Got an unexpected type {type(batch_orig_dt)}."
                )
                batch_unnormalized = batch_aug.polygons_aug
            return batch_unnormalized

        if not background:
            # singlecore augmentation

            for idx, batch in enumerate(batches):
                batch_normalized, batch_orig_dt = _normalize_batch(idx, batch)
                batch_normalized = self.augment_batch_(batch_normalized, hooks=hooks)
                batch_unnormalized = _unnormalize_batch(batch_normalized, batch, batch_orig_dt)

                yield batch_unnormalized
        else:
            # multicore augmentation
            import imgaug2.multicore as multicore

            id_to_batch_orig = dict()

            def load_batches() -> Iterator[Batch]:
                for idx, batch in enumerate(batches):
                    batch_normalized, batch_orig_dt = _normalize_batch(idx, batch)
                    id_to_batch_orig[idx] = (batch, batch_orig_dt)
                    yield batch_normalized

            with multicore.Pool(self) as pool:
                # note that pool.processes is None here
                output_buffer_size = pool.pool._processes * 10

                for batch_aug in pool.imap_batches(
                    load_batches(), output_buffer_size=output_buffer_size
                ):
                    idx = batch_aug.data[0]
                    assert idx in id_to_batch_orig, (
                        f"Got idx {idx:d} from Pool, which is not known."
                    )
                    batch_orig, batch_orig_dt = id_to_batch_orig[idx]
                    batch_unnormalized = _unnormalize_batch(batch_aug, batch_orig, batch_orig_dt)
                    del id_to_batch_orig[idx]
                    yield batch_unnormalized

    # we deprecate here so that users switch to `augment_batch_()` and in the
    # future we can add a `parents` parameter here without having to consider
    # that a breaking change
    @ia.deprecated(
        "augment_batch_()",
        comment="`augment_batch()` was renamed to "
        "`augment_batch_()` as it changes all `*_unaug` "
        "attributes of batches in-place. Note that "
        "`augment_batch_()` has now a `parents` parameter. "
        "Calls of the style `augment_batch(batch, hooks)` "
        "must be changed to "
        "`augment_batch(batch, hooks=hooks)`.",
    )
    def augment_batch(
        self,
        batch: Batch | UnnormalizedBatch | _BatchInAugmentation,
        hooks: ia.HooksImages | None = None,
    ) -> Batch | UnnormalizedBatch | _BatchInAugmentation:
        """Augment a single batch.

        Deprecated since 0.4.0.

        """
        # We call augment_batch_() directly here without copy, because this
        # method never copies. Would make sense to add a copy here if the
        # method is un-deprecated at some point.
        return self.augment_batch_(batch, hooks=hooks)

    # TODO add more tests
    @legacy(version="0.4.0")
    def augment_batch_(
        self,
        batch: Batch | UnnormalizedBatch | _BatchInAugmentation,
        parents: list[Augmenter] | None = None,
        hooks: ia.HooksImages | None = None,
    ) -> Batch | UnnormalizedBatch | _BatchInAugmentation:
        """
        Augment a single batch in-place.


        Parameters
        ----------
        batch : imgaug2.augmentables.batches.Batch or imgaug2.augmentables.batches.UnnormalizedBatch or imgaug2.augmentables.batch._BatchInAugmentation
            A single batch to augment.

            If :class:`imgaug2.augmentables.batches.UnnormalizedBatch`
            or :class:`imgaug2.augmentables.batches.Batch`, then the ``*_aug``
            attributes may be modified in-place, while the ``*_unaug``
            attributes will not be modified.
            If :class:`imgaug2.augmentables.batches._BatchInAugmentation`,
            then all attributes may be modified in-place.

        parents : None or list of imgaug2.augmenters.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug2.HooksImages, optional
            HooksImages object to dynamically interfere with the augmentation
            process.

        Returns
        -------
        imgaug2.augmentables.batches.Batch or imgaug2.augmentables.batches.UnnormalizedBatch
            Augmented batch.

        """
        # this chain of if/elses would be more beautiful if it was
        # (1st) UnnormalizedBatch, (2nd) Batch, (3rd) BatchInAugmenation.
        # We check for _BatchInAugmentation first as it is expected to be the
        # most common input (due to child calls).
        batch_unnorm = None
        batch_norm = None
        if isinstance(batch, _BatchInAugmentation):
            batch_inaug = batch
        elif isinstance(batch, UnnormalizedBatch):
            batch_unnorm = batch
            batch_norm = batch.to_normalized_batch()
            batch_inaug = batch_norm.to_batch_in_augmentation()
        elif isinstance(batch, Batch):
            batch_norm = batch
            batch_inaug = batch_norm.to_batch_in_augmentation()
        else:
            raise ValueError(
                "Expected UnnormalizedBatch, Batch or _BatchInAugmentation, "
                f"got {type(batch).__name__}."
            )

        columns = batch_inaug.columns

        # hooks preprocess
        if hooks is not None:
            for column in columns:
                value = hooks.preprocess(column.value, augmenter=self, parents=parents)
                setattr(batch_inaug, column.attr_name, value)

            # refresh so that values are updated for later functions
            columns = batch_inaug.columns

        # set augmentables to None if this augmenter is deactivated or hooks
        # demands it
        set_to_none = []
        if not self.activated:
            for column in columns:
                set_to_none.append(column)
                setattr(batch_inaug, column.attr_name, None)
        elif hooks is not None:
            for column in columns:
                activated = hooks.is_activated(
                    column.value, augmenter=self, parents=parents, default=self.activated
                )
                if not activated:
                    set_to_none.append(column)
                    setattr(batch_inaug, column.attr_name, None)

        # If _augment_batch_() follows legacy-style and ends up calling
        # _augment_images() and similar methods, we don't need the
        # deterministic context here. But if there is a custom implementation
        # of _augment_batch_(), then we should have this here. It causes very
        # little overhead.
        with _maybe_deterministic_ctx(self):
            if not batch_inaug.empty:
                pf_enabled = not self.deterministic
                with iap.toggled_prefetching(pf_enabled):
                    batch_inaug = self._augment_batch_(
                        batch_inaug,
                        random_state=self.random_state,
                        parents=parents if parents is not None else [],
                        hooks=hooks,
                    )

        # revert augmentables being set to None for non-activated augmenters
        for column in set_to_none:
            setattr(batch_inaug, column.attr_name, column.value)

        # hooks postprocess
        if hooks is not None:
            # refresh as contents may have been changed in _augment_batch_()
            columns = batch_inaug.columns

            for column in columns:
                augm_value = hooks.postprocess(column.value, augmenter=self, parents=parents)
                setattr(batch_inaug, column.attr_name, augm_value)

        if batch_unnorm is not None:
            batch_norm = batch_norm.fill_from_batch_in_augmentation_(batch_inaug)
            batch_unnorm = batch_unnorm.fill_from_augmented_normalized_batch_(batch_norm)
            return batch_unnorm
        if batch_norm is not None:
            batch_norm = batch_norm.fill_from_batch_in_augmentation_(batch_inaug)
            return batch_norm
        return batch_inaug

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        """Augment a single batch in-place.

        This is the internal version of :func:`Augmenter.augment_batch_`.
        It is called from :func:`Augmenter.augment_batch_` and should usually
        not be called directly.
        This method may transform the batches in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.


        Parameters
        ----------
        batch : imgaug2.augmentables.batches._BatchInAugmentation
            The normalized batch to augment. May be changed in-place.

        random_state : imgaug2.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug2.augmenters.meta.Augmenter
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_batch_`.

        hooks : imgaug2.imgaug2.HooksImages or None
            See :func:`~imgaug2.augmenters.meta.Augmenter.augment_batch_`.

        Returns
        -------
        imgaug2.augmentables.batches._BatchInAugmentation
            The augmented batch.

        """
        # The code below covers the case of older augmenters that still have
        # _augment_images(), _augment_keypoints(), ... methods that augment
        # each input type on its own (including re-sampling from random
        # variables). The code block can be safely overwritten by a method
        # augmenting a whole batch of data in one step.

        columns = batch.columns
        multiple_columns = len(columns) > 1

        # For multi-column data (e.g. images + BBs) we need deterministic mode
        # within this batch, otherwise the datatypes within this batch would
        # get different samples.
        deterministic = self.deterministic or multiple_columns

        # set attribute batch.T_aug with result of self.augment_T() for each
        # batch.T_unaug (that had any content)
        for column in columns:
            with _maybe_deterministic_ctx(random_state, deterministic):
                pf_enabled = not self.deterministic
                with iap.toggled_prefetching(pf_enabled):
                    value = getattr(self, "_augment_" + column.name)(
                        column.value, random_state=random_state, parents=parents, hooks=hooks
                    )
                    setattr(batch, column.attr_name, value)

        # If the augmenter was alread in deterministic mode, we can expect
        # that to_deterministic() was called, which advances the RNG. But
        # if it wasn't and we had to auto-switch for the batch, there was not
        # advancement yet.
        if multiple_columns and not self.deterministic:
            random_state.advance_()

        return batch

