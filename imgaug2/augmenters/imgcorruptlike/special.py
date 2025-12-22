from __future__ import annotations

from typing import Literal

from imgaug2.augmenters._typing import Array, ParamInput, RNGInput
from imgaug2.compat.markers import legacy

from .base import _ImgcorruptAugmenterBase
from .core import _call_imgcorrupt_func


@legacy(version="0.4.0")
def apply_elastic_transform(
    image: Array,
    severity: int = 1,
    seed: int | None = None,
    allow_cpu_fallback: bool = False,
) -> Array:
    """Apply ``elastic_transform`` from ``imagecorruptions``.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike._call_imgcorrupt_func`.

    Parameters
    ----------
    image : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    allow_cpu_fallback : bool, optional
        If True, allow CPU fallback for MLX inputs. This performs a
        host<->device roundtrip.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func(
        "elastic_transform",
        seed,
        False,
        image,
        severity,
        allow_cpu_fallback=allow_cpu_fallback,
    )


# ----------------------------------------------------------------------------
# Augmenters
# ----------------------------------------------------------------------------
# The augmenter definitions below are almost identical and mainly differ in
# the names and functions used. It would be fairly trivial to write a
# function that would create these augmenters dynamically (and one is listed
# below as a comment). The downside is that in these cases the documentation
# would also be generated dynamically, which leads to numerous problems:
# (1) users couldn't easily read the documentation while scrolling through
# the code file, (2) IDEs might not be able to use it for code suggestions,
# (3) linting tools can't detect and validate it, (4) the imgaug-doc
# tools to parse dtype support don't work with dynamically generated
# documentation (and neither with dynamically generated classes).
# Even though it's by far more code, it seems like the better choice overall
# to just write it out.

# Example function to dynamically generate augmenters, kept for possible
# future uses:
# def _create_augmenter(class_name, func_name):
#     func = globals()["apply_%s" % (func_name,)]
#
#     def __init__(self, severity=1, name=None, deterministic=False,
#                  random_state=None):
#         super(self.__class__, self).__init__(
#             func, severity, name=name, deterministic=deterministic,
#             random_state=random_state)
#
#     augmenter_class = type(class_name,
#                            (_ImgcorruptAugmenterBase,),
#                            {"__init__": __init__})
#
#     augmenter_class.__doc__ = """
#     Wrapper around ``imagecorruptions.corruptions.%s``.
#
#     **Supported dtypes**:
#
#     See :func:`~imgaug2.augmenters.imgcorruptlike.apply_%s`.
#
#     Parameters
#     ----------
#     severity : int, optional
#         Strength of the corruption, with valid values being
#         ``1 <= severity <= 5``.
#
#     name : None or str, optional
#         See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.
#
#     deterministic : bool, optional
#         See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.
#
#     random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
#         See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.
#
#     Examples
#     --------
#     >>> import imgaug2.augmenters as iaa
#     >>> aug = iaa.%s(severity=2)
#
#     Create an augmenter around ``imagecorruptions.corruptions.%s``. Apply it to
#     images using e.g. ``aug(images=[image1, image2, ...])``.
#
#     """ % (func_name, func_name, class_name, func_name)
#
#     return augmenter_class




@legacy(version="0.4.0")
class ElasticTransform(_ImgcorruptAugmenterBase):
    """
    Wrapper around ``imagecorruptions.corruptions.elastic_transform``.

    .. warning::

        This augmenter can currently only transform image-data.
        Batches containing heatmaps, segmentation maps and
        coordinate-based augmentables will be rejected with an error.
        Use :class:`~imgaug2.augmenters.geometric.ElasticTransformation` if
        you have to transform such inputs.


    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.imgcorruptlike.apply_elastic_transform`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

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
    >>> # doctest: +SKIP
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.imgcorruptlike.ElasticTransform(severity=2)

    Create an augmenter around
    ``imagecorruptions.corruptions.elastic_transform``.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        severity: ParamInput = (1, 5),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            apply_elastic_transform,
            severity,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        cols = batch.get_column_names()
        assert len(cols) == 0 or (len(cols) == 1 and "images" in cols), (
            "imgcorruptlike.ElasticTransform can currently only process image "
            "data. Got a batch containing: {}. Use "
            "imgaug2.augmenters.geometric.ElasticTransformation for "
            "batches containing non-image data.".format(", ".join(cols))
        )
        return super()._augment_batch_(batch, random_state, parents, hooks)
