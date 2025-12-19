"""Augmenters that apply mirroring/flipping operations to images.

This module provides augmenters for horizontally and vertically flipping images
and their associated augmentables (keypoints, bounding boxes, etc.).

Key Augmenters:
    - `Fliplr`: Flip images horizontally (left-right mirror).
    - `Flipud`: Flip images vertically (up-down mirror).
"""

from __future__ import annotations

from typing import Literal, TypeAlias, cast, overload

import cv2
import numpy as np
from numpy.typing import NDArray

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import meta
from imgaug2.imgaug import _normalize_cv2_input_arr_
from imgaug2.compat.markers import legacy

NumpyArray: TypeAlias = NDArray[np.generic]

_FLIPLR_DTYPES_CV2 = iadt._convert_dtype_strs_to_types("uint8 uint16 int8 int16")


@overload
def fliplr(arr: NumpyArray) -> NumpyArray: ...


@overload
def fliplr(arr: object) -> object: ...


def fliplr(arr: object) -> object:
    """Flip an image-like array horizontally.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; fully tested
        * ``uint32``: yes; fully tested
        * ``uint64``: yes; fully tested
        * ``int8``: yes; fully tested
        * ``int16``: yes; fully tested
        * ``int32``: yes; fully tested
        * ``int64``: yes; fully tested
        * ``float16``: yes; fully tested
        * ``float32``: yes; fully tested
        * ``float64``: yes; fully tested
        * ``float128``: yes; fully tested
        * ``bool``: yes; fully tested

    Parameters
    ----------
    arr : ndarray
        A 2D/3D `(H, W, [C])` image array.

    Returns
    -------
    ndarray
        Horizontally flipped array.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug2.augmenters.flip as flip
    >>> arr = np.arange(16).reshape((4, 4))
    >>> arr_flipped = flip.fliplr(arr)

    Create a ``4x4`` array and flip it horizontally.

    """
    # MLX fast-path (B1): only when input is already on device.
    from imgaug2.mlx._core import is_mlx_array

    if is_mlx_array(arr):
        import imgaug2.mlx as mlx

        return mlx.fliplr(arr)

    assert ia.is_np_array(arr), f"Expected numpy array, got type {type(arr)}."
    arr_np = cast(NumpyArray, arr)

    # we don't check here if #channels > 512, because the cv2 function also
    # kinda works with that, it is very rare to happen and would induce an
    # additional check (with significant relative impact on runtime considering
    # flipping is already ultra fast)
    if arr_np.dtype in _FLIPLR_DTYPES_CV2:
        return _fliplr_cv2(arr_np)
    return _fliplr_sliced(arr_np)


def _fliplr_sliced(arr: NumpyArray) -> NumpyArray:
    return arr[:, ::-1, ...]


def _fliplr_cv2(arr: NumpyArray) -> NumpyArray:
    # cv2.flip() returns None for arrays with zero height or width
    # and turns channels=0 to channels=512
    if arr.size == 0:
        return np.copy(arr)

    # cv2.flip() fails for more than 512 channels
    if arr.ndim == 3 and arr.shape[-1] > 512:
        # TODO this is quite inefficient right now
        channels = [
            cv2.flip(_normalize_cv2_input_arr_(arr[..., c]), 1) for c in range(arr.shape[-1])
        ]
        result = np.stack(channels, axis=-1)
    else:
        # Normalization from imgaug2.imgaug2._normalize_cv2_input_arr_().
        # Moved here for performance reasons. Keep this aligned.
        # TODO recalculate timings, they were computed without this.
        flags = arr.flags
        if not flags["OWNDATA"]:
            arr = np.copy(arr)
            flags = arr.flags
        if not flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        result = cv2.flip(_normalize_cv2_input_arr_(arr), 1)

    if result.ndim == 2 and arr.ndim == 3:
        return result[..., np.newaxis]
    return result


@overload
def flipud(arr: NumpyArray) -> NumpyArray: ...


@overload
def flipud(arr: object) -> object: ...


def flipud(arr: object) -> object:
    """Flip an image-like array vertically.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; fully tested
        * ``uint32``: yes; fully tested
        * ``uint64``: yes; fully tested
        * ``int8``: yes; fully tested
        * ``int16``: yes; fully tested
        * ``int32``: yes; fully tested
        * ``int64``: yes; fully tested
        * ``float16``: yes; fully tested
        * ``float32``: yes; fully tested
        * ``float64``: yes; fully tested
        * ``float128``: yes; fully tested
        * ``bool``: yes; fully tested

    Parameters
    ----------
    arr : ndarray
        A 2D/3D `(H, W, [C])` image array.

    Returns
    -------
    ndarray
        Vertically flipped array.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug2.augmenters.flip as flip
    >>> arr = np.arange(16).reshape((4, 4))
    >>> arr_flipped = flip.flipud(arr)

    Create a ``4x4`` array and flip it vertically.

    """
    # MLX fast-path (B1): only when input is already on device.
    from imgaug2.mlx._core import is_mlx_array

    if is_mlx_array(arr):
        import imgaug2.mlx as mlx

        return mlx.flipud(arr)

    # Note that this function is currently not called by Flipud for performance
    # reasons. Changing this will therefore not affect Flipud.
    return arr[::-1, ...]


def HorizontalFlip(*args: object, **kwargs: object) -> Fliplr:
    """Alias for Fliplr."""
    return Fliplr(*args, **kwargs)


def VerticalFlip(*args: object, **kwargs: object) -> Flipud:
    """Alias for Flipud."""
    return Flipud(*args, **kwargs)


class Fliplr(meta.Augmenter):
    """Flip/mirror input images horizontally.

    .. note::

        The default value for the probability is ``0.0``.
        So, to flip *all* input images use ``Fliplr(1.0)`` and *not* just
        ``Fliplr()``.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.flip.fliplr`.

    Parameters
    ----------
    p : number or imgaug2.parameters.StochasticParameter, optional
        Probability of each image to get flipped.

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
    >>> aug = iaa.Fliplr(0.5)

    Flip ``50`` percent of all images horizontally.


    >>> aug = iaa.Fliplr(1.0)

    Flip all images horizontally.

    """

    def __init__(
        self,
        p: float | int | bool | iap.StochasticParameter = 1,
        seed: iarandom.RNGInput = None,
        name: str | None = None,
        random_state: iarandom.RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.p = iap.handle_probability_param(p, "p")

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        samples = self.p.draw_samples((batch.nb_rows,), random_state=random_state)
        for i, sample in enumerate(samples):
            if sample >= 0.5:
                if batch.images is not None:
                    batch.images[i] = fliplr(batch.images[i])

                if batch.heatmaps is not None:
                    batch.heatmaps[i].arr_0to1 = fliplr(batch.heatmaps[i].arr_0to1)

                if batch.segmentation_maps is not None:
                    batch.segmentation_maps[i].arr = fliplr(batch.segmentation_maps[i].arr)

                if batch.keypoints is not None:
                    kpsoi = batch.keypoints[i]
                    width = kpsoi.shape[1]
                    for kp in kpsoi.keypoints:
                        kp.x = width - float(kp.x)

                if batch.bounding_boxes is not None:
                    bbsoi = batch.bounding_boxes[i]
                    width = bbsoi.shape[1]
                    for bb in bbsoi.bounding_boxes:
                        # after flip, x1 ends up right of x2
                        x1, x2 = bb.x1, bb.x2
                        bb.x1 = width - x2
                        bb.x2 = width - x1

                if batch.polygons is not None:
                    psoi = batch.polygons[i]
                    width = psoi.shape[1]
                    for poly in psoi.polygons:
                        # TODO maybe reverse the order of points afterwards?
                        #      the flip probably inverts them
                        poly.exterior[:, 0] = width - poly.exterior[:, 0]

                if batch.line_strings is not None:
                    lsoi = batch.line_strings[i]
                    width = lsoi.shape[1]
                    for ls in lsoi.line_strings:
                        # TODO maybe reverse the order of points afterwards?
                        #      the flip probably inverts them
                        ls.coords[:, 0] = width - ls.coords[:, 0]

        return batch

    def get_parameters(self) -> list[iap.StochasticParameter]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p]


# TODO merge with Fliplr
class Flipud(meta.Augmenter):
    """Flip/mirror input images vertically.

    .. note::

        The default value for the probability is ``0.0``.
        So, to flip *all* input images use ``Flipud(1.0)`` and *not* just
        ``Flipud()``.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.flip.flipud`.

    Parameters
    ----------
    p : number or imgaug2.parameters.StochasticParameter, optional
        Probability of each image to get flipped.

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
    >>> aug = iaa.Flipud(0.5)

    Flip ``50`` percent of all images vertically.

    >>> aug = iaa.Flipud(1.0)

    Flip all images vertically.

    """

    def __init__(
        self,
        p: float | int | bool | iap.StochasticParameter = 1,
        seed: iarandom.RNGInput = None,
        name: str | None = None,
        random_state: iarandom.RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.p = iap.handle_probability_param(p, "p")

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        samples = self.p.draw_samples((batch.nb_rows,), random_state=random_state)
        for i, sample in enumerate(samples):
            if sample >= 0.5:
                if batch.images is not None:
                    # We currently do not use flip.flipud() here, because that
                    # saves a function call.
                    batch.images[i] = batch.images[i][::-1, ...]

                if batch.heatmaps is not None:
                    batch.heatmaps[i].arr_0to1 = batch.heatmaps[i].arr_0to1[::-1, ...]

                if batch.segmentation_maps is not None:
                    batch.segmentation_maps[i].arr = batch.segmentation_maps[i].arr[::-1, ...]

                if batch.keypoints is not None:
                    kpsoi = batch.keypoints[i]
                    height = kpsoi.shape[0]
                    for kp in kpsoi.keypoints:
                        kp.y = height - float(kp.y)

                if batch.bounding_boxes is not None:
                    bbsoi = batch.bounding_boxes[i]
                    height = bbsoi.shape[0]
                    for bb in bbsoi.bounding_boxes:
                        # after flip, y1 ends up right of y2
                        y1, y2 = bb.y1, bb.y2
                        bb.y1 = height - y2
                        bb.y2 = height - y1

                if batch.polygons is not None:
                    psoi = batch.polygons[i]
                    height = psoi.shape[0]
                    for poly in psoi.polygons:
                        # TODO maybe reverse the order of points afterwards?
                        #      the flip probably inverts them
                        poly.exterior[:, 1] = height - poly.exterior[:, 1]

                if batch.line_strings is not None:
                    lsoi = batch.line_strings[i]
                    height = lsoi.shape[0]
                    for ls in lsoi.line_strings:
                        # TODO maybe reverse the order of points afterwards?
                        #      the flip probably inverts them
                        ls.coords[:, 1] = height - ls.coords[:, 1]

        return batch

    def get_parameters(self) -> list[iap.StochasticParameter]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p]
