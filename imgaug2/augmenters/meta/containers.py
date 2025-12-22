"""Container augmenters and list helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters._typing import Array, RNGInput
from imgaug2.compat.markers import legacy

from .base import Augmenter


@legacy
def handle_children_list(
    lst: Augmenter | Sequence[Augmenter] | None,
    augmenter_name: str,
    lst_name: str,
    default: Augmenter | Sequence[Augmenter] | None | Literal["sequential"] = "sequential",
) -> Augmenter | Sequence[Augmenter] | None:
    """Normalize an augmenter list provided by a user."""
    if lst is None:
        if default == "sequential":
            return Sequential([], name=f"{augmenter_name}-{lst_name}")
        return default
    if isinstance(lst, Augmenter):
        if ia.is_iterable(lst):
            only_augmenters = all([isinstance(child, Augmenter) for child in lst])
            assert only_augmenters, "Expected all children to be augmenters, got types {}.".format(
                ", ".join([str(type(v)) for v in lst])
            )
            return lst
        return Sequential(lst, name=f"{augmenter_name}-{lst_name}")
    if ia.is_iterable(lst):
        if len(lst) == 0 and default != "sequential":
            return default
        only_augmenters = all([isinstance(child, Augmenter) for child in lst])
        assert only_augmenters, "Expected all children to be augmenters, got types {}.".format(
            ", ".join([str(type(v)) for v in lst])
        )
        return Sequential(lst, name=f"{augmenter_name}-{lst_name}")
    raise Exception(
        f"Expected None, Augmenter or list/tuple as children list {lst_name} "
        f"for augmenter with name {augmenter_name}, got {type(lst)}."
    )
class Sequential(Augmenter, list):
    """List augmenter containing child augmenters to apply to inputs.

    This augmenter is simply a list of other augmenters. To augment an image
    or any other data, it iterates over its children and applies each one
    of them independently to the data. (This also means that the second
    applied augmenter will already receive augmented input data and augment
    it further.)

    This augmenter offers the option to apply its children in random order
    using the `random_order` parameter. This should often be activated as
    it greatly increases the space of possible augmentations.

    .. note::

        You are *not* forced to use :class:`~imgaug2.augmenters.meta.Sequential`
        in order to use other augmenters. Each augmenter can be used on its
        own, e.g the following defines an augmenter for horizontal flips and
        then augments a single image:

        >>> import numpy as np
        >>> import imgaug2.augmenters as iaa
        >>> image = np.zeros((32, 32, 3), dtype=np.uint8)
        >>> aug = iaa.Fliplr(0.5)
        >>> image_aug = aug.augment_image(image)

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    children : imgaug2.augmenters.meta.Augmenter or list of imgaug2.augmenters.meta.Augmenter or None, optional
        The augmenters to apply to images.

    random_order : bool, optional
        Whether to apply the child augmenters in random order.
        If ``True``, the order will be randomly sampled once per batch.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug2.augmenters as iaa
    >>> imgs = [np.random.rand(10, 10)]
    >>> seq = iaa.Sequential([
    >>>     iaa.Fliplr(0.5),
    >>>     iaa.Flipud(0.5)
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    Create a :class:`~imgaug2.augmenters.meta.Sequential` that always first
    applies a horizontal flip augmenter and then a vertical flip augmenter.
    Each of these two augmenters has a ``50%`` probability of actually
    flipping the image.

    >>> seq = iaa.Sequential([
    >>>     iaa.Fliplr(0.5),
    >>>     iaa.Flipud(0.5)
    >>> ], random_order=True)
    >>> imgs_aug = seq.augment_images(imgs)

    Create a :class:`~imgaug2.augmenters.meta.Sequential` that sometimes first
    applies a horizontal flip augmenter (followed by a vertical flip
    augmenter) and sometimes first a vertical flip augmenter (followed by a
    horizontal flip augmenter). Again, each of them has a ``50%`` probability
    of actually flipping the image.

    """

    def __init__(
        self,
        children: Augmenter | Sequence[Augmenter] | None = None,
        random_order: bool = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        Augmenter.__init__(
            self, seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        if children is None:
            list.__init__(self, [])
        elif isinstance(children, Augmenter):
            # this must be separate from `list.__init__(self, children)`,
            # otherwise in `Sequential(OneOf(...))` the OneOf(...) is
            # interpreted as a list and OneOf's children become Sequential's
            # children
            list.__init__(self, [children])
        elif ia.is_iterable(children):
            assert all([isinstance(child, Augmenter) for child in children]), (
                "Expected all children to be augmenters, got types {}.".format(
                    ", ".join([str(type(v)) for v in children])
                )
            )
            list.__init__(self, children)
        else:
            raise Exception(
                f"Expected None or Augmenter or list of Augmenter, got {type(children)}."
            )

        assert ia.is_single_bool(random_order), (
            f"Expected random_order to be boolean, got {type(random_order)}."
        )
        self.random_order = random_order

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        with batch.propagation_hooks_ctx(self, hooks, parents):
            if self.random_order:
                order = random_state.permutation(len(self))
            else:
                order = range(len(self))

            for index in order:
                batch = self[index].augment_batch_(batch, parents=parents + [self], hooks=hooks)
        return batch

    def _to_deterministic(self) -> Sequential:
        augs = [aug.to_deterministic() for aug in self]
        seq = self.copy()
        seq[:] = augs
        seq.random_state = self.random_state.derive_rng_()
        seq.deterministic = True
        return seq

    def get_parameters(self) -> Sequence[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.random_order]

    def add(self, augmenter: Augmenter) -> None:
        """Add an augmenter to the list of child augmenters.

        Parameters
        ----------
        imgaug2.augmenters.meta.Augmenter
            The augmenter to add.

        """
        self.append(augmenter)

    def get_children_lists(self) -> list[list[Augmenter]]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_children_lists`."""
        return [self]

    def __str__(self) -> str:
        augs_str = ", ".join([aug.__str__() for aug in self])
        pattern = "%s(name=%s, random_order=%s, children=[%s], deterministic=%s)"
        return pattern % (
            self.__class__.__name__,
            self.name,
            self.random_order,
            augs_str,
            self.deterministic,
        )


class SomeOf(Augmenter, list):
    """List augmenter that applies only some of its children to inputs.

    This augmenter is similar to :class:`~imgaug2.augmenters.meta.Sequential`,
    but may apply only a fixed or random subset of its child augmenters to
    inputs. E.g. the augmenter could be initialized with a list of 20 child
    augmenters and then apply 5 randomly chosen child augmenters to images.

    The subset of augmenters to apply (and their order) is sampled once
    *per image*. If `random_order` is ``True``, the order will be sampled once
    *per batch* (similar to :class:`~imgaug2.augmenters.meta.Sequential`).

    This augmenter currently does not support replacing (i.e. picking the same
    child multiple times) due to implementation difficulties in connection
    with deterministic augmenters.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    n : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter or None, optional
        Count of augmenters to apply.

            * If ``int``, then exactly `n` of the child augmenters are applied
              to every image.
            * If tuple of two ``int`` s ``(a, b)``, then a random value will
              be uniformly sampled per image from the discrete interval
              ``[a..b]`` and denote the number of child augmenters to pick
              and apply. ``b`` may be set to ``None``, which is then equivalent
              to ``(a..C)`` with ``C`` denoting the number of children that
              the augmenter has.
            * If ``StochasticParameter``, then ``N`` numbers will be sampled
              for ``N`` images. The parameter is expected to be discrete.
            * If ``None``, then the total number of available children will be
              used (i.e. all children will be applied).

    children : imgaug2.augmenters.meta.Augmenter or list of imgaug2.augmenters.meta.Augmenter or None, optional
        The augmenters to apply to images.
        If this is a list of augmenters, it will be converted to a
        :class:`~imgaug2.augmenters.meta.Sequential`.

    random_order : boolean, optional
        Whether to apply the child augmenters in random order.
        If ``True``, the order will be randomly sampled once per batch.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> imgs = [np.random.rand(10, 10)]
    >>> seq = iaa.SomeOf(1, [
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0)
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    Apply either ``Fliplr`` or ``Flipud`` to images.

    >>> seq = iaa.SomeOf((1, 3), [
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0),
    >>>     iaa.GaussianBlur(1.0)
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    Apply one to three of the listed augmenters (``Fliplr``, ``Flipud``,
    ``GaussianBlur``) to images. They are always applied in the
    provided order, i.e. first ``Fliplr``, second ``Flipud``, third
    ``GaussianBlur``.

    >>> seq = iaa.SomeOf((1, None), [
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0),
    >>>     iaa.GaussianBlur(1.0)
    >>> ], random_order=True)
    >>> imgs_aug = seq.augment_images(imgs)

    Apply one to all of the listed augmenters (``Fliplr``, ``Flipud``,
    ``GaussianBlur``) to images. They are applied in random order, i.e.
    sometimes ``GaussianBlur`` first, followed by ``Fliplr``, sometimes
    ``Fliplr`` followed by ``Flipud`` followed by ``Blur`` etc.
    The order is sampled once per batch.

    """

    def __init__(
        self,
        n: int | tuple[int, int | None] | list[int | None] | iap.StochasticParameter | None = None,
        children: Augmenter | Sequence[Augmenter] | None = None,
        random_order: bool = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        Augmenter.__init__(
            self, seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        # TODO use handle_children_list() here?
        if children is None:
            list.__init__(self, [])
        elif isinstance(children, Augmenter):
            # this must be separate from `list.__init__(self, children)`,
            # otherwise in `SomeOf(OneOf(...))` the OneOf(...) is
            # interpreted as a list and OneOf's children become SomeOf's
            # children
            list.__init__(self, [children])
        elif ia.is_iterable(children):
            assert all([isinstance(child, Augmenter) for child in children]), (
                "Expected all children to be augmenters, got types {}.".format(
                    ", ".join([str(type(v)) for v in children])
                )
            )
            list.__init__(self, children)
        else:
            raise Exception(
                f"Expected None or Augmenter or list of Augmenter, got {type(children)}."
            )

        self.n, self.n_mode = self._handle_arg_n(n)

        assert ia.is_single_bool(random_order), (
            f"Expected random_order to be boolean, got {type(random_order)}."
        )
        self.random_order = random_order

    @classmethod
    def _handle_arg_n(
        cls, n: int | Sequence[int | None] | iap.StochasticParameter | None
    ) -> tuple[int | tuple[int, int | None] | iap.StochasticParameter | None, str]:
        if ia.is_single_number(n):
            n = int(n)
            n_mode = "deterministic"
        elif n is None:
            n = None
            n_mode = "None"
        elif ia.is_iterable(n):
            assert len(n) == 2, (
                f"Expected iterable 'n' to contain exactly two values, got {len(n)}."
            )
            if ia.is_single_number(n[0]) and n[1] is None:
                n = (int(n[0]), None)
                n_mode = "(int,None)"
            elif ia.is_single_number(n[0]) and ia.is_single_number(n[1]):
                n = iap.DiscreteUniform(int(n[0]), int(n[1]))
                n_mode = "stochastic"
            else:
                raise Exception(
                    f"Expected tuple of (int, None) or (int, int), got {[type(el) for el in n]}"
                )
        elif isinstance(n, iap.StochasticParameter):
            n_mode = "stochastic"
        else:
            raise Exception(
                f"Expected int, (int, None), (int, int) or StochasticParameter, got {type(n)}"
            )
        return n, n_mode

    def _get_n(self, nb_images: int, random_state: iarandom.RNG) -> Sequence[int]:
        if self.n_mode == "deterministic":
            assert isinstance(self.n, int)
            return [self.n] * nb_images
        if self.n_mode == "None":
            return [len(self)] * nb_images
        if self.n_mode == "(int,None)":
            assert isinstance(self.n, tuple)
            assert isinstance(self.n[0], int)
            param = iap.DiscreteUniform(self.n[0], len(self))
            samples = param.draw_samples((nb_images,), random_state=random_state)
            return [int(v) for v in samples]
        if self.n_mode == "stochastic":
            assert isinstance(self.n, iap.StochasticParameter)
            samples = self.n.draw_samples((nb_images,), random_state=random_state)
            return [int(v) for v in samples]
        raise Exception(f"Invalid n_mode: {self.n_mode}")

    def _get_augmenter_order(self, random_state: iarandom.RNG) -> Array:
        if not self.random_order:
            augmenter_order = np.arange(len(self))
        else:
            augmenter_order = random_state.permutation(len(self))
        return augmenter_order

    def _get_augmenter_active(self, nb_rows: int, random_state: iarandom.RNG) -> Array:
        nn = self._get_n(nb_rows, random_state)
        nn = [min(n, len(self)) for n in nn]
        augmenter_active = np.zeros((nb_rows, len(self)), dtype=bool)
        for row_idx, n_true in enumerate(nn):
            if n_true > 0:
                augmenter_active[row_idx, 0:n_true] = 1
        for row in augmenter_active:
            random_state.shuffle(row)
        return augmenter_active

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        with batch.propagation_hooks_ctx(self, hooks, parents):
            # This must happen before creating the augmenter_active array,
            # otherwise in case of determinism the number of augmented images
            # would change the random_state's state, resulting in the order
            # being dependent on the number of augmented images (and not be
            # constant). By doing this first, the random state is always the
            # same (when determinism is active), so the order is always the
            # same.
            augmenter_order = self._get_augmenter_order(random_state)

            # create an array of active augmenters per image
            # e.g.
            #  [[0, 0, 1],
            #   [1, 0, 1],
            #   [1, 0, 0]]
            # would signal, that augmenter 3 is active for the first image,
            # augmenter 1 and 3 for the 2nd image and augmenter 1 for the 3rd.
            augmenter_active = self._get_augmenter_active(batch.nb_rows, random_state)

            for augmenter_index in augmenter_order:
                active = augmenter_active[:, augmenter_index].nonzero()[0]

                if len(active) > 0:
                    batch_sub = batch.subselect_rows_by_indices(active)
                    batch_sub = self[augmenter_index].augment_batch_(
                        batch_sub, parents=parents + [self], hooks=hooks
                    )
                    batch = batch.invert_subselect_rows_by_indices_(active, batch_sub)

            return batch

    def _to_deterministic(self) -> SomeOf:
        augs = [aug.to_deterministic() for aug in self]
        seq = self.copy()
        seq[:] = augs
        seq.random_state = self.random_state.derive_rng_()
        seq.deterministic = True
        return seq

    def get_parameters(self) -> Sequence[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.n]

    def add(self, augmenter: Augmenter) -> None:
        """Add an augmenter to the list of child augmenters.

        Parameters
        ----------
        augmenter : imgaug2.augmenters.meta.Augmenter
            The augmenter to add.

        """
        self.append(augmenter)

    def get_children_lists(self) -> list[list[Augmenter]]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_children_lists`."""
        return [self]

    def __str__(self) -> str:
        augs_str = ", ".join([aug.__str__() for aug in self])
        pattern = "%s(name=%s, n=%s, random_order=%s, augmenters=[%s], deterministic=%s)"
        return pattern % (
            self.__class__.__name__,
            self.name,
            str(self.n),
            str(self.random_order),
            augs_str,
            self.deterministic,
        )


class OneOf(SomeOf):
    """Augmenter that always executes exactly one of its children.

    **Supported dtypes**:

    See :class:`imgaug2.augmenters.meta.SomeOf`.

    Parameters
    ----------
    children : imgaug2.augmenters.meta.Augmenter or list of imgaug2.augmenters.meta.Augmenter
        The choices of augmenters to apply.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> images = [np.ones((10, 10), dtype=np.uint8)]  # dummy example images
    >>> seq = iaa.OneOf([
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0)
    >>> ])
    >>> images_aug = seq.augment_images(images)

    Flip each image either horizontally or vertically.

    >>> images = [np.ones((10, 10), dtype=np.uint8)]  # dummy example images
    >>> seq = iaa.OneOf([
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Sequential([
    >>>         iaa.GaussianBlur(1.0),
    >>>         iaa.Dropout(0.05),
    >>>         iaa.AdditiveGaussianNoise(0.1*255)
    >>>     ]),
    >>>     iaa.Noop()
    >>> ])
    >>> images_aug = seq.augment_images(images)

    Either flip each image horizontally, or add blur+dropout+noise or do
    nothing.

    """

    def __init__(
        self,
        children: Augmenter | Sequence[Augmenter],
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            n=1,
            children=children,
            random_order=False,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class Sometimes(Augmenter):
    """Apply child augmenter(s) with a probability of `p`.

    Let ``C`` be one or more child augmenters given to
    :class:`~imgaug2.augmenters.meta.Sometimes`.
    Let ``p`` be the fraction of images (or other data) to augment.
    Let ``I`` be the input images (or other data).
    Let ``N`` be the number of input images (or other entities).
    Then (on average) ``p*N`` images of ``I`` will be augmented using ``C``.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    p : float or imgaug2.parameters.StochasticParameter, optional
        Sets the probability with which the given augmenters will be applied to
        input images/data. E.g. a value of ``0.5`` will result in ``50%`` of
        all input images (or other augmentables) being augmented.

    then_list : None or imgaug2.augmenters.meta.Augmenter or list of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) to apply to `p%` percent of all images.
        If this is a list of augmenters, it will be converted to a
        :class:`~imgaug2.augmenters.meta.Sequential`.

    else_list : None or imgaug2.augmenters.meta.Augmenter or list of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) to apply to ``(1-p)`` percent of all images.
        These augmenters will be applied only when the ones in `then_list`
        are *not* applied (either-or-relationship).
        If this is a list of augmenters, it will be converted to a
        :class:`~imgaug2.augmenters.meta.Sequential`.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3))

    Apply ``GaussianBlur`` to ``50%`` of all input images.

    >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3), iaa.Fliplr(1.0))

    Apply ``GaussianBlur`` to ``50%`` of all input images. Apply ``Fliplr``
    to the other ``50%`` of all input images.

    """

    def __init__(
        self,
        p: float | iap.StochasticParameter = 0.5,
        then_list: Augmenter | Sequence[Augmenter] | None = None,
        else_list: Augmenter | Sequence[Augmenter] | None = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.p = iap.handle_probability_param(p, "p")

        self.then_list = handle_children_list(then_list, self.name, "then", default=None)
        self.else_list = handle_children_list(else_list, self.name, "else", default=None)

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        with batch.propagation_hooks_ctx(self, hooks, parents):
            samples = self.p.draw_samples((batch.nb_rows,), random_state=random_state)

            # create lists/arrays of images for if and else lists (one for each)
            # note that np.where returns tuple(array([0, 5, 9, ...])) or
            # tuple(array([]))
            indices_then_list = np.where(samples == 1)[0]
            indices_else_list = np.where(samples == 0)[0]

            indice_lists = [indices_then_list, indices_else_list]
            augmenter_lists = [self.then_list, self.else_list]

            # For then_list: collect augmentables to be processed by then_list
            # augmenters, apply them to the list, then map back to the output
            # list. Analogous for else_list.
            # TODO maybe this would be easier if augment_*() accepted a list
            #      that can contain Nones
            for indices, augmenters in zip(indice_lists, augmenter_lists, strict=True):
                if augmenters is not None and len(augmenters) > 0:
                    batch_sub = batch.subselect_rows_by_indices(indices)
                    batch_sub = augmenters.augment_batch_(
                        batch_sub, parents=parents + [self], hooks=hooks
                    )
                    batch = batch.invert_subselect_rows_by_indices_(indices, batch_sub)

            return batch

    def _to_deterministic(self) -> Sometimes:
        aug = self.copy()
        aug.then_list = (
            aug.then_list.to_deterministic() if aug.then_list is not None else aug.then_list
        )
        aug.else_list = (
            aug.else_list.to_deterministic() if aug.else_list is not None else aug.else_list
        )
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    def get_parameters(self) -> Sequence[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p]

    def get_children_lists(self) -> list[list[Augmenter]]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_children_lists`."""
        result = []
        if self.then_list is not None:
            result.append(self.then_list)
        if self.else_list is not None:
            result.append(self.else_list)
        return result

    def __str__(self) -> str:
        pattern = "%s(p=%s, name=%s, then_list=%s, else_list=%s, deterministic=%s)"
        return pattern % (
            self.__class__.__name__,
            self.p,
            self.name,
            self.then_list,
            self.else_list,
            self.deterministic,
        )


