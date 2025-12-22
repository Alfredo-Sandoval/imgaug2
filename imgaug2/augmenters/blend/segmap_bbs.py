from __future__ import annotations

from typing import Literal

from imgaug2.compat.markers import legacy
from imgaug2.augmenters._typing import ParamInput, RNGInput

from .base import ChildrenInput, LabelInput
from .mask_generators import BoundingBoxesMaskGen, SegMapClassIdsMaskGen
from .masks import BlendAlphaMask

@legacy(version="0.4.0")
class BlendAlphaSegMapClassIds(BlendAlphaMask):
    """Blend images from two branches based on segmentation map ids.

    This class generates masks that are ``1.0`` at pixel locations covered
    by specific classes in segmentation maps.

    This class is a thin wrapper around
    :class:`~imgaug2.augmenters.blend.BlendAlphaMask` together with
    :class:`~imgaug2.augmenters.blend.SegMapClassIdsMaskGen`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug2.augmenters.blend.BlendAlphaMask` for details.

    .. note::

        Segmentation maps can have multiple channels. If that is the case
        then for each position ``(x, y)`` it is sufficient that any class id
        in any channel matches one of the desired class ids.

    .. note::

        This class will produce an ``AssertionError`` if there are no
        segmentation maps in a batch.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.blend.BlendAlphaMask`.

    Parameters
    ----------
    class_ids : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter
        See :class:`~imgaug2.augmenters.blend.SegMapClassIdsMaskGen`.

    foreground : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    nb_sample_classes : None or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.blend.SegMapClassIdsMaskGen`.

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
    >>> aug = iaa.BlendAlphaSegMapClassIds(
    >>>     [1, 3],
    >>>     foreground=iaa.AddToHue((-100, 100)))

    Create an augmenter that randomizes the hue wherever the segmentation maps
    contain the classes ``1`` or ``3``.

    >>> aug = iaa.BlendAlphaSegMapClassIds(
    >>>     [1, 2, 3, 4],
    >>>     nb_sample_classes=2,
    >>>     foreground=iaa.GaussianBlur(3.0))

    Create an augmenter that randomly picks ``2`` classes from the
    list ``[1, 2, 3, 4]`` and blurs the image content wherever these classes
    appear in the segmentation map. Note that as the sampling of class ids
    happens *with replacement*, it is not guaranteed to sample two *unique*
    class ids.

    >>> aug = iaa.Sometimes(0.2,
    >>>     iaa.BlendAlphaSegMapClassIds(
    >>>         2,
    >>>         background=iaa.TotalDropout(1.0)))

    Create an augmenter that zeros for roughly every fifth image all
    image pixels that do *not* belong to class id ``2`` (note that the
    `background` branch was used, not the `foreground` branch).
    Example use case: Human body landmark detection where both the
    landmarks/keypoints and the body segmentation map are known. Train the
    model to detect landmarks and sometimes remove all non-body information
    to force the model to become more independent of the background.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        class_ids: ParamInput,
        foreground: ChildrenInput = None,
        background: ChildrenInput = None,
        nb_sample_classes: ParamInput | None = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            SegMapClassIdsMaskGen(class_ids=class_ids, nb_sample_classes=nb_sample_classes),
            foreground=foreground,
            background=background,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


@legacy(version="0.4.0")
class BlendAlphaBoundingBoxes(BlendAlphaMask):
    """Blend images from two branches based on areas enclosed in bounding boxes.

    This class generates masks that are ``1.0`` within bounding boxes of given
    labels. A mask pixel will be set to ``1.0`` if *at least* one bounding box
    covers the area and has one of the requested labels.

    This class is a thin wrapper around
    :class:`~imgaug2.augmenters.blend.BlendAlphaMask` together with
    :class:`~imgaug2.augmenters.blend.BoundingBoxesMaskGen`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug2.augmenters.blend.BlendAlphaMask` for details.

    .. note::

        This class will produce an ``AssertionError`` if there are no
        bounding boxes in a batch.


    **Supported dtypes**:

    See :class:`~imgaug2.augmenters.blend.BlendAlphaMask`.

    Parameters
    ----------
    labels : None or str or list of str or imgaug2.parameters.StochasticParameter
        See :class:`~imgaug2.augmenters.blend.BoundingBoxesMaskGen`.

    foreground : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug2.augmenters.meta.Augmenter or iterable of imgaug2.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    nb_sample_labels : None or tuple of int or list of int or imgaug2.parameters.StochasticParameter, optional
        See :class:`~imgaug2.augmenters.blend.BoundingBoxesMaskGen`.

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
    >>> aug = iaa.BlendAlphaBoundingBoxes("person",
    >>>                                   foreground=iaa.Grayscale(1.0))

    Create an augmenter that removes color within bounding boxes having the
    label ``person``.

    >>> aug = iaa.BlendAlphaBoundingBoxes(["person", "car"],
    >>>                                   foreground=iaa.AddToHue((-255, 255)))

    Create an augmenter that randomizes the hue within bounding boxes that
    have the label ``person`` or ``car``.

    >>> aug = iaa.BlendAlphaBoundingBoxes(["person", "car"],
    >>>                                   foreground=iaa.AddToHue((-255, 255)),
    >>>                                   nb_sample_labels=1)

    Create an augmenter that randomizes the hue within bounding boxes that
    have either the label ``person`` or ``car``. Only one label is picked per
    image. Note that the sampling happens with replacement, so if
    ``nb_sample_classes`` would be ``>1``, it could still lead to only one
    *unique* label being sampled.

    >>> aug = iaa.BlendAlphaBoundingBoxes(None,
    >>>                                   background=iaa.Multiply(0.0))

    Create an augmenter that zeros all pixels (``Multiply(0.0)``)
    that are *not* (``background`` branch) within bounding boxes of
    *any* (``None``) label. In other words, all pixels outside of bounding
    boxes become black.
    Note that we don't use ``TotalDropout`` here, because by default it will
    also remove all coordinate-based augmentables, which will break the
    blending of such inputs.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        labels: LabelInput,
        foreground: ChildrenInput = None,
        background: ChildrenInput = None,
        nb_sample_labels: ParamInput | None = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            BoundingBoxesMaskGen(labels=labels, nb_sample_labels=nb_sample_labels),
            foreground=foreground,
            background=background,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


