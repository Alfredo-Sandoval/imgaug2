from __future__ import annotations

import copy as copy_module
import re
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeVar

import imgaug2.imgaug as ia
from imgaug2.compat.markers import legacy

if TYPE_CHECKING:
    from .base import Augmenter

_AugmenterT = TypeVar("_AugmenterT", bound="Augmenter")


class AugmenterChildrenMixin:
    @abstractmethod
    def get_parameters(self) -> Sequence[object]:
        """Get the parameters of this augmenter.

        Returns
        -------
        list
            List of parameters of arbitrary types (usually child class
            of :class:`~imgaug2.parameters.StochasticParameter`, but not
            guaranteed to be).

        """
        raise NotImplementedError()

    def get_children_lists(self) -> list[list[Augmenter]]:
        """Get a list of lists of children of this augmenter.

        For most augmenters, the result will be a single empty list.
        For augmenters with children it will often be a list with one
        sublist containing all children. In some cases the augmenter will
        contain multiple distinct lists of children, e.g. an if-list and an
        else-list. This will lead to a result consisting of a single list
        with multiple sublists, each representing the respective sublist of
        children.

        E.g. for an if/else-augmenter that executes the children ``A1``,
        ``A2`` if a condition is met and otherwise executes the children
        ``B1``, ``B2``, ``B3`` the result will be
        ``[[A1, A2], [B1, B2, B3]]``.

        IMPORTANT: While the topmost list may be newly created, each of the
        sublist must be editable inplace resulting in a changed children list
        of the augmenter. E.g. if an Augmenter
        ``IfElse(condition, [A1, A2], [B1, B2, B3])`` returns
        ``[[A1, A2], [B1, B2, B3]]``
        for a call to
        :func:`~imgaug2.augmenters.meta.Augmenter.get_children_lists` and
        ``A2`` is removed inplace from ``[A1, A2]``, then the children lists
        of ``IfElse(...)`` must also change to ``[A1], [B1, B2, B3]``. This
        is used in
        :func:`~imgaug2.augmeneters.meta.Augmenter.remove_augmenters_`.

        Returns
        -------
        list of list of imgaug2.augmenters.meta.Augmenter
            One or more lists of child augmenter.
            Can also be a single empty list.

        """
        return []

    def get_all_children(self, flat: bool = False) -> list[Augmenter | list[Augmenter]]:
        """Get all children of this augmenter as a list.

        If the augmenter has no children, the returned list is empty.

        Parameters
        ----------
        flat : bool
            If set to ``True``, the returned list will be flat.

        Returns
        -------
        list of imgaug2.augmenters.meta.Augmenter
            The children as a nested or flat list.

        """
        result = []
        for lst in self.get_children_lists():
            for aug in lst:
                result.append(aug)
                children = aug.get_all_children(flat=flat)
                if len(children) > 0:
                    if flat:
                        result.extend(children)
                    else:
                        result.append(children)
        return result

    def find_augmenters(
        self,
        func: Callable[[Augmenter, list[Augmenter]], bool],
        parents: list[Augmenter] | None = None,
        flat: bool = True,
    ) -> list[Augmenter | list[Augmenter]]:
        """Find augmenters that match a condition.

        This function will compare this augmenter and all of its children
        with a condition. The condition is a lambda function.

        Parameters
        ----------
        func : callable
            A function that receives a
            :class:`~imgaug2.augmenters.meta.Augmenter` instance and a list of
            parent :class:`~imgaug2.augmenters.meta.Augmenter` instances and
            must return ``True``, if that augmenter is valid match or
            ``False`` otherwise.

        parents : None or list of imgaug2.augmenters.meta.Augmenter, optional
            List of parent augmenters.
            Intended for nested calls and can usually be left as ``None``.

        flat : bool, optional
            Whether to return the result as a flat list (``True``)
            or a nested list (``False``). In the latter case, the nesting
            matches each augmenters position among the children.

        Returns
        ----------
        list of imgaug2.augmenters.meta.Augmenter
            Nested list if `flat` was set to ``False``.
            Flat list if `flat` was set to ``True``.

        Examples
        --------
        >>> import imgaug2.augmenters as iaa
        >>> aug = iaa.Sequential([
        >>>     iaa.Fliplr(0.5, name="fliplr"),
        >>>     iaa.Flipud(0.5, name="flipud")
        >>> ])
        >>> print(aug.find_augmenters(lambda a, parents: a.name == "fliplr"))

        Return the first child augmenter (``Fliplr`` instance).

        """
        if parents is None:
            parents = []

        result = []
        if func(self, parents):
            result.append(self)

        subparents = parents + [self]
        for lst in self.get_children_lists():
            for aug in lst:
                found = aug.find_augmenters(func, parents=subparents, flat=flat)
                if len(found) > 0:
                    if flat:
                        result.extend(found)
                    else:
                        result.append(found)
        return result

    def find_augmenters_by_name(
        self, name: str, regex: bool = False, flat: bool = True
    ) -> list[Augmenter | list[Augmenter]]:
        """Find augmenter(s) by name.

        Parameters
        ----------
        name : str
            Name of the augmenter(s) to search for.

        regex : bool, optional
            Whether `name` parameter is a regular expression.

        flat : bool, optional
            See :func:`~imgaug2.augmenters.meta.Augmenter.find_augmenters`.

        Returns
        -------
        augmenters : list of imgaug2.augmenters.meta.Augmenter
            Nested list if `flat` was set to ``False``.
            Flat list if `flat` was set to ``True``.

        """
        return self.find_augmenters_by_names([name], regex=regex, flat=flat)

    def find_augmenters_by_names(
        self, names: Sequence[str], regex: bool = False, flat: bool = True
    ) -> list[Augmenter | list[Augmenter]]:
        """Find augmenter(s) by names.

        Parameters
        ----------
        names : list of str
            Names of the augmenter(s) to search for.

        regex : bool, optional
            Whether `names` is a list of regular expressions.
            If it is, an augmenter is considered a match if *at least* one
            of these expressions is a match.

        flat : boolean, optional
            See :func:`~imgaug2.augmenters.meta.Augmenter.find_augmenters`.

        Returns
        -------
        augmenters : list of imgaug2.augmenters.meta.Augmenter
            Nested list if `flat` was set to ``False``.
            Flat list if `flat` was set to ``True``.

        """
        if regex:

            def comparer(aug: Augmenter, _parents: list[Augmenter]) -> bool:
                for pattern in names:
                    if re.match(pattern, aug.name):
                        return True
                return False

            return self.find_augmenters(comparer, flat=flat)
        return self.find_augmenters(lambda aug, parents: aug.name in names, flat=flat)

    # TODO remove copy arg
    # TODO allow first arg to be string name, class type or func
    def remove_augmenters(
        self,
        func: Callable[[Augmenter, list[Augmenter]], bool],
        copy: bool = True,
        identity_if_topmost: bool = True,
        noop_if_topmost: bool | None = None,
    ) -> Augmenter | None:
        """Remove this augmenter or children that match a condition.

        Parameters
        ----------
        func : callable
            Condition to match per augmenter.
            The function must expect the augmenter itself and a list of parent
            augmenters and returns ``True`` if that augmenter is supposed to
            be removed, or ``False`` otherwise.
            E.g. ``lambda a, parents: a.name == "fliplr" and len(parents) == 1``
            removes an augmenter with name ``fliplr`` if it is the direct child
            of the augmenter upon which ``remove_augmenters()`` was initially
            called.

        copy : bool, optional
            Whether to copy this augmenter and all if its children before
            removing. If ``False``, removal is performed in-place.

        identity_if_topmost : bool, optional
            If ``True`` and the condition (lambda function) leads to the
            removal of the topmost augmenter (the one this function is called
            on initially), then that topmost augmenter will be replaced by an
            instance of :class:`~imgaug2.augmenters.meta.Noop` (i.e. an
            augmenter that doesn't change its inputs). If ``False``, ``None``
            will be returned in these cases.
            This can only be ``False`` if copy is set to ``True``.

        noop_if_topmost : bool, optional
            Deprecated since 0.4.0.

        Returns
        -------
        imgaug2.augmenters.meta.Augmenter or None
            This augmenter after the removal was performed.
            ``None`` is returned if the condition was matched for the
            topmost augmenter, `copy` was set to ``True`` and `noop_if_topmost`
            was set to ``False``.

        Examples
        --------
        >>> import imgaug2.augmenters as iaa
        >>> seq = iaa.Sequential([
        >>>     iaa.Fliplr(0.5, name="fliplr"),
        >>>     iaa.Flipud(0.5, name="flipud"),
        >>> ])
        >>> seq = seq.remove_augmenters(lambda a, parents: a.name == "fliplr")

        This removes the augmenter ``Fliplr`` from the ``Sequential``
        object's children.

        """
        if noop_if_topmost is not None:
            ia.warn_deprecated(
                "Parameter 'noop_if_topmost' is deprecated. Use 'identity_if_topmost' instead."
            )
            identity_if_topmost = noop_if_topmost

        if func(self, []):
            if not copy:
                raise Exception(
                    "Inplace removal of topmost augmenter requested, "
                    "which is currently not possible. Set 'copy' to True."
                )

            if identity_if_topmost:
                from .identity import Identity

                return Identity()
            return None

        aug = self if not copy else self.deepcopy()
        aug.remove_augmenters_(func, parents=[])
        return aug

    @ia.deprecated("remove_augmenters_")
    def remove_augmenters_inplace(
        self,
        func: Callable[[Augmenter, list[Augmenter]], bool],
        parents: list[Augmenter] | None = None,
    ) -> None:
        """Old name for :func:`~imgaug2.meta.Augmenter.remove_augmenters_`.

        Deprecated since 0.4.0.

        """
        self.remove_augmenters_(func=func, parents=parents)

    # TODO allow first arg to be string name, class type or func
    # TODO remove parents arg + add _remove_augmenters_() with parents arg
    @legacy(version="0.4.0")
    def remove_augmenters_(
        self,
        func: Callable[[Augmenter, list[Augmenter]], bool],
        parents: list[Augmenter] | None = None,
    ) -> None:
        """Remove in-place children of this augmenter that match a condition.

        This is functionally identical to
        :func:`~imgaug2.augmenters.meta.remove_augmenters` with
        ``copy=False``, except that it does not affect the topmost augmenter
        (the one on which this function is initially called on).


        Parameters
        ----------
        func : callable
            See :func:`~imgaug2.augmenters.meta.Augmenter.remove_augmenters`.

        parents : None or list of imgaug2.augmenters.meta.Augmenter, optional
            List of parent :class:`~imgaug2.augmenters.meta.Augmenter` instances
            that lead to this augmenter. If ``None``, an empty list will be
            used. This parameter can usually be left empty and will be set
            automatically for children.

        Examples
        --------
        >>> import imgaug2.augmenters as iaa
        >>> seq = iaa.Sequential([
        >>>     iaa.Fliplr(0.5, name="fliplr"),
        >>>    iaa.Flipud(0.5, name="flipud"),
        >>> ])
        >>> seq.remove_augmenters_(lambda a, parents: a.name == "fliplr")

        This removes the augmenter ``Fliplr`` from the ``Sequential``
        object's children.

        """
        parents = [] if parents is None else parents
        subparents = parents + [self]
        for lst in self.get_children_lists():
            to_remove = []
            for i, aug in enumerate(lst):
                if func(aug, subparents):
                    to_remove.append((i, aug))

            for count_removed, (i, _aug) in enumerate(to_remove):
                del lst[i - count_removed]

            for aug in lst:
                aug.remove_augmenters_(func, subparents)

    def copy(self: _AugmenterT) -> _AugmenterT:
        """Create a shallow copy of this Augmenter instance.

        Returns
        -------
        imgaug2.augmenters.meta.Augmenter
            Shallow copy of this Augmenter instance.

        """
        return copy_module.copy(self)

    def deepcopy(self: _AugmenterT) -> _AugmenterT:
        """Create a deep copy of this Augmenter instance.

        Returns
        -------
        imgaug2.augmenters.meta.Augmenter
            Deep copy of this Augmenter instance.

        """
        # TODO if this augmenter has child augmenters and multiple of them
        #      use the global random state, then after copying, these
        #      augmenters share a single new random state that is a copy of
        #      the global random state (i.e. all use the same *instance*,
        #      not just state). This can lead to confusing bugs.
        # TODO write a custom copying routine?
        return copy_module.deepcopy(self)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        params = self.get_parameters()
        params_str = ", ".join([param.__str__() for param in params])
        return f"{self.__class__.__name__}(name={self.name}, parameters=[{params_str}], deterministic={self.deterministic})"
