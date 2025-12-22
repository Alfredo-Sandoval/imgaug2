from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmenters._typing import RNGInput
from imgaug2.compat.markers import legacy

if TYPE_CHECKING:
    from .base import Augmenter


class AugmenterDeterministicMixin:
    def to_deterministic(self, n: int | None = None) -> Augmenter | list[Augmenter]:
        """Convert this augmenter from a stochastic to a deterministic one.

        A stochastic augmenter samples pseudo-random values for each parameter,
        image and batch.
        A deterministic augmenter also samples new values for each parameter
        and image, but not batch. Instead, for consecutive batches it will
        sample the same values (provided the number of images and their sizes
        don't change).
        From a technical perspective this means that a deterministic augmenter
        starts each batch's augmentation with a random number generator in
        the same state (i.e. same seed), instead of advancing that state from
        batch to batch.

        Using determinism is useful to (a) get the same augmentations for
        two or more image batches (e.g. for stereo cameras), (b) to augment
        images and corresponding data on them (e.g. segmentation maps or
        bounding boxes) in the same way.

        Parameters
        ----------
        n : None or int, optional
            Number of deterministic augmenters to return.
            If ``None`` then only one :class:`~imgaug2.augmenters.meta.Augmenter`
            instance will be returned.
            If ``1`` or higher, a list containing ``n``
            :class:`~imgaug2.augmenters.meta.Augmenter` instances will be
            returned.

        Returns
        -------
        imgaug2.augmenters.meta.Augmenter or list of imgaug2.augmenters.meta.Augmenter
            A single Augmenter object if `n` was None,
            otherwise a list of Augmenter objects (even if `n` was ``1``).

        """
        assert n is None or n >= 1, f"Expected 'n' to be None or >=1, got {n}."
        if n is None:
            return self.to_deterministic(1)[0]
        return [self._to_deterministic() for _ in range(n)]

    def with_probability(self, p: float | iap.StochasticParameter) -> Augmenter:
        """Wrap this augmenter in :class:`~imgaug2.augmenters.meta.Sometimes`.

        This is a convenience method to get a uniform "probability of applying"
        interface across *all* augmenters, including those that don't have a
        dedicated `p` parameter.

        Example
        -------
        >>> import imgaug2.augmenters as iaa
        >>> aug = iaa.Add((-10, 10)).with_probability(0.2)

        Equivalent to:

        >>> aug = iaa.Sometimes(0.2, iaa.Add((-10, 10)))
        """
        from .containers import Sometimes

        return Sometimes(p, self)

    def _to_deterministic(self) -> Augmenter:
        """Convert this augmenter from a stochastic to a deterministic one.

        Augmenter-specific implementation of
        :func:`~imgaug2.augmenters.meta.to_deterministic`. This function is
        expected to return a single new deterministic
        :class:`~imgaug2.augmenters.meta.Augmenter` instance of this augmenter.

        Returns
        -------
        det : imgaug2.augmenters.meta.Augmenter
            Deterministic variation of this Augmenter object.

        """
        aug = self.copy()

        # This was changed for 0.2.8 from deriving a new random state based on
        # the global random state to deriving it from the augmenter's local
        # random state. This should reduce the risk that re-runs of scripts
        # lead to different results upon small changes somewhere. It also
        # decreases the likelihood of problems when using multiprocessing
        # (the child processes might use the same global random state as the
        # parent process). Note for the latter point that augment_batches()
        # might call to_deterministic() if the batch contains multiply types
        # of augmentables.
        # aug.random_state = iarandom.create_random_rng()
        aug.random_state = self.random_state.derive_rng_()

        aug.deterministic = True
        return aug

    @ia.deprecated("imgaug2.augmenters.meta.Augmenter.seed_")
    def reseed(self, random_state: RNGInput = None, deterministic_too: bool = False) -> None:
        """Old name of :func:`~imgaug2.augmenters.meta.Augmenter.seed_`.

        Deprecated since 0.4.0.

        """
        self.seed_(entropy=random_state, deterministic_too=deterministic_too)

    # TODO mark this as in-place
    @legacy(version="0.4.0")
    def seed_(self, entropy: RNGInput = None, deterministic_too: bool = False) -> None:
        """Seed this augmenter and all of its children.

        This method assigns a new random number generator to the
        augmenter and all of its children (if it has any). The new random
        number generator is *derived* from the provided seed or RNG -- or from
        the global random number generator if ``None`` was provided.
        Note that as child RNGs are *derived*, they do not all use the same
        seed.

        If this augmenter or any child augmenter had a random number generator
        that pointed to the global random state, it will automatically be
        replaced with a local random state. This is similar to what
        :func:`~imgaug2.augmenters.meta.Augmenter.localize_random_state`
        does.

        This method is useful when augmentations are run in the
        background (i.e. on multiple cores).
        It should be called before sending this
        :class:`~imgaug2.augmenters.meta.Augmenter` instance to a
        background worker or once within each worker with different seeds
        (i.e., if ``N`` workers are used, the function should be called
        ``N`` times). Otherwise, all background workers will
        use the same seeds and therefore apply the same augmentations.
        Note that :func:`Augmenter.augment_batches` and :func:`Augmenter.pool`
        already do this automatically.


        Parameters
        ----------
        entropy : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
            A seed or random number generator that is used to derive new
            random number generators for this augmenter and its children.
            If an ``int`` is provided, it will be interpreted as a seed.
            If ``None`` is provided, the global random number generator will
            be used.

        deterministic_too : bool, optional
            Whether to also change the seed of an augmenter ``A``, if ``A``
            is deterministic. This is the case both when this augmenter
            object is ``A`` or one of its children is ``A``.

        Examples
        --------
        >>> import imgaug2.augmenters as iaa
        >>> aug = iaa.Sequential([
        >>>     iaa.Crop(px=(0, 10)),
        >>>     iaa.Crop(px=(0, 10))
        >>> ])
        >>> aug.seed_(1)

        Seed an augmentation sequence containing two crop operations. Even
        though the same seed was used, the two operations will still sample
        different pixel amounts to crop as the child-specific seed is merely
        derived from the provided seed.

        """
        assert isinstance(deterministic_too, bool), (
            f"Expected 'deterministic_too' to be a boolean, got type {deterministic_too}."
        )

        if entropy is None:
            random_state = iarandom.RNG.create_pseudo_random_()
        else:
            random_state = iarandom.RNG.create_if_not_rng_(entropy)

        if not self.deterministic or deterministic_too:
            # note that derive_rng_() (used below) advances the RNG, so
            # child augmenters get a different RNG state
            self.random_state = random_state.copy()

        for lst in self.get_children_lists():
            for aug in lst:
                aug.seed_(entropy=random_state.derive_rng_(), deterministic_too=deterministic_too)

    def localize_random_state(self, recursive: bool = True) -> Augmenter:
        """Assign augmenter-specific RNGs to this augmenter and its children.

        See :func:`Augmenter.localize_random_state_` for more details.

        Parameters
        ----------
        recursive : bool, optional
            See
            :func:`~imgaug2.augmenters.meta.Augmenter.localize_random_state_`.

        Returns
        -------
        imgaug2.augmenters.meta.Augmenter
            Copy of the augmenter and its children, with localized RNGs.

        """
        aug = self.deepcopy()
        aug.localize_random_state_(recursive=recursive)
        return aug

    # TODO rename random_state -> rng
    def localize_random_state_(self, recursive: bool = True) -> Augmenter:
        """Assign augmenter-specific RNGs to this augmenter and its children.

        This method iterates over this augmenter and all of its children and
        replaces any pointer to the global RNG with a new local (i.e.
        augmenter-specific) RNG.

        A random number generator (RNG) is used for the sampling of random
        values.
        The global random number generator exists exactly once throughout
        the library and is shared by many augmenters.
        A local RNG (usually) exists within exactly one augmenter and is
        only used by that augmenter.

        Usually there is no need to change global into local RNGs.
        The only noteworthy exceptions are

            * Whenever you want to use determinism (so that the global RNG is
              not accidentally reverted).
            * Whenever you want to copy RNGs from one augmenter to
              another. (Copying the global RNG would usually not be useful.
              Copying the global RNG from augmenter A to B, then executing A
              and then B would result in B's (global) RNG's state having
              already changed because of A's sampling. So the samples of
              A and B would differ.)

        The case of determinism is handled automatically by
        :func:`~imgaug2.augmenters.meta.Augmenter.to_deterministic`.
        Only when you copy RNGs (via
        :func:`~imgaug2.augmenters.meta.Augmenter.copy_random_state`),
        you need to call this function first.

        Parameters
        ----------
        recursive : bool, optional
            Whether to localize the RNGs of the augmenter's children too.

        Returns
        -------
        imgaug2.augmenters.meta.Augmenter
            Returns itself (with localized RNGs).

        """
        if self.random_state.is_global_rng():
            self.random_state = self.random_state.derive_rng_()
        if recursive:
            for lst in self.get_children_lists():
                for child in lst:
                    child.localize_random_state_(recursive=recursive)
        return self

    # TODO adapt random_state -> rng
    def copy_random_state(
        self,
        source: Augmenter,
        recursive: bool = True,
        matching: Literal["position", "name"] = "position",
        matching_tolerant: bool = True,
        copy_determinism: bool = False,
    ) -> Augmenter:
        """Copy the RNGs from a source augmenter sequence.

        Parameters
        ----------
        source : imgaug2.augmenters.meta.Augmenter
            See :func:`~imgaug2.augmenters.meta.Augmenter.copy_random_state_`.

        recursive : bool, optional
            See :func:`~imgaug2.augmenters.meta.Augmenter.copy_random_state_`.

        matching : {'position', 'name'}, optional
            See :func:`~imgaug2.augmenters.meta.Augmenter.copy_random_state_`.

        matching_tolerant : bool, optional
            See :func:`~imgaug2.augmenters.meta.Augmenter.copy_random_state_`.

        copy_determinism : bool, optional
            See :func:`~imgaug2.augmenters.meta.Augmenter.copy_random_state_`.

        Returns
        -------
        imgaug2.augmenters.meta.Augmenter
            Copy of the augmenter itself (with copied RNGs).

        """
        aug = self.deepcopy()
        aug.copy_random_state_(
            source,
            recursive=recursive,
            matching=matching,
            matching_tolerant=matching_tolerant,
            copy_determinism=copy_determinism,
        )
        return aug

    def copy_random_state_(
        self,
        source: Augmenter,
        recursive: bool = True,
        matching: Literal["position", "name"] = "position",
        matching_tolerant: bool = True,
        copy_determinism: bool = False,
    ) -> Augmenter:
        """Copy the RNGs from a source augmenter sequence (in-place).

        .. note::

            The source augmenters are not allowed to use the global RNG.
            Call
            :func:`~imgaug2.augmenters.meta.Augmenter.localize_random_state_`
            once on the source to localize all random states.

        Parameters
        ----------
        source : imgaug2.augmenters.meta.Augmenter
            The source augmenter(s) from where to copy the RNG(s).
            The source may have children (e.g. the source can be a
            :class:`~imgaug2.augmenters.meta.Sequential`).

        recursive : bool, optional
            Whether to copy the RNGs of the source augmenter *and*
            all of its children (``True``) or just the source
            augmenter (``False``).

        matching : {'position', 'name'}, optional
            Defines the matching mode to use during recursive copy.
            This is used to associate source augmenters with target augmenters.
            If ``position`` then the target and source sequences of augmenters
            are turned into flattened lists and are associated based on
            their list indices. If ``name`` then the target and source
            augmenters are matched based on their names (i.e.
            ``augmenter.name``).

        matching_tolerant : bool, optional
            Whether to use tolerant matching between source and target
            augmenters. If set to ``False``: Name matching will raise an
            exception for any target augmenter which's name does not appear
            among the source augmenters. Position matching will raise an
            exception if source and target augmenter have an unequal number
            of children.

        copy_determinism : bool, optional
            Whether to copy the ``deterministic`` attributes from source to
            target augmenters too.

        Returns
        -------
        imgaug2.augmenters.meta.Augmenter
            The augmenter itself.

        """
        # Note: the target random states are localized, but the source random
        # states don't have to be localized. That means that they can be
        # the global random state. Worse, if copy_random_state() was called,
        # the target random states would have different identities, but
        # same states. If multiple target random states were the global random
        # state, then after deepcopying them, they would all share the same
        # identity that is different to the global random state. I.e., if the
        # state of any random state of them is set in-place, it modifies the
        # state of all other target random states (that were once global),
        # but not the global random state.
        # Summary: Use target = source.copy() here, instead of
        # target.use_state_of_(source).

        source_augs = [source] + source.get_all_children(flat=True) if recursive else [source]
        target_augs = [self] + self.get_all_children(flat=True) if recursive else [self]

        global_rs_exc_msg = (
            "You called copy_random_state_() with a source that uses global "
            "RNGs. Call localize_random_state_() on the source "
            "first or initialize your augmenters with local random states, "
            "e.g. via Dropout(..., random_state=1234)."
        )

        if matching == "name":
            source_augs_dict = {aug.name: aug for aug in source_augs}
            target_augs_dict = {aug.name: aug for aug in target_augs}

            different_lengths = len(source_augs_dict) < len(source_augs) or len(
                target_augs_dict
            ) < len(target_augs)
            if different_lengths:
                ia.warn(
                    "Matching mode 'name' with recursive=True was chosen in "
                    "copy_random_state_, but either the source or target "
                    "augmentation sequence contains multiple augmenters with "
                    "the same name."
                )

            for name in target_augs_dict:
                if name in source_augs_dict:
                    if source_augs_dict[name].random_state.is_global_rng():
                        raise Exception(global_rs_exc_msg)
                    # has to be copy(), see above
                    target_augs_dict[name].random_state = source_augs_dict[name].random_state.copy()
                    if copy_determinism:
                        target_augs_dict[name].deterministic = source_augs_dict[name].deterministic
                elif not matching_tolerant:
                    raise Exception(f"Augmenter name '{name}' not found among source augmenters.")
        elif matching == "position":
            if len(source_augs) != len(target_augs) and not matching_tolerant:
                raise Exception("Source and target augmentation sequences have different lengths.")
            for source_aug, target_aug in zip(
                source_augs, target_augs, strict=not matching_tolerant
            ):
                if source_aug.random_state.is_global_rng():
                    raise Exception(global_rs_exc_msg)
                # has to be copy(), see above
                target_aug.random_state = source_aug.random_state.copy()
                if copy_determinism:
                    target_aug.deterministic = source_aug.deterministic
        else:
            raise Exception(
                f"Unknown matching method '{matching}'. Valid options are 'name' and 'position'."
            )

        return self

