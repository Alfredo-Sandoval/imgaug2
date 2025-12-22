from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import cv2
import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import meta
from imgaug2.augmenters._blend_utils import blend_alpha
from imgaug2.augmenters._typing import Array, Images, ParamInput, RNGInput
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _normalize_cv2_input_arr_
from imgaug2.mlx._core import is_mlx_array
import imgaug2.mlx.color as mlx_color

from ._utils import (
    CSPACE_ALL,
    CSPACE_BGR,
    CSPACE_CIE,
    CSPACE_GRAY,
    CSPACE_HLS,
    CSPACE_HSV,
    CSPACE_Lab,
    CSPACE_Luv,
    CSPACE_RGB,
    CSPACE_YCrCb,
    CSPACE_YUV,
    ChildrenInput,
    ColorSpace,
    ColorSpaceInput,
    ToColorspaceChoiceInput,
    ToColorspaceParamInput,
    _is_mlx_list,
)


def _get_opencv_attr(attr_names: Sequence[str]) -> int | None:
    for attr_name in attr_names:
        if hasattr(cv2, attr_name):
            return getattr(cv2, attr_name)
    ia.warn(
        f"Could not find any of the following attributes in cv2: {attr_names}. "
        "This can cause issues with colorspace transformations."
    )
    return None


_CSPACE_OPENCV_CONV_VARS = {
    # RGB
    (CSPACE_RGB, CSPACE_BGR): cv2.COLOR_RGB2BGR,
    (CSPACE_RGB, CSPACE_GRAY): cv2.COLOR_RGB2GRAY,
    (CSPACE_RGB, CSPACE_YCrCb): _get_opencv_attr(["COLOR_RGB2YCR_CB"]),
    (CSPACE_RGB, CSPACE_HSV): cv2.COLOR_RGB2HSV,
    (CSPACE_RGB, CSPACE_HLS): cv2.COLOR_RGB2HLS,
    (CSPACE_RGB, CSPACE_Lab): _get_opencv_attr(["COLOR_RGB2LAB", "COLOR_RGB2Lab"]),
    (CSPACE_RGB, CSPACE_Luv): cv2.COLOR_RGB2LUV,
    (CSPACE_RGB, CSPACE_YUV): cv2.COLOR_RGB2YUV,
    (CSPACE_RGB, CSPACE_CIE): cv2.COLOR_RGB2XYZ,
    # BGR
    (CSPACE_BGR, CSPACE_RGB): cv2.COLOR_BGR2RGB,
    (CSPACE_BGR, CSPACE_GRAY): cv2.COLOR_BGR2GRAY,
    (CSPACE_BGR, CSPACE_YCrCb): _get_opencv_attr(["COLOR_BGR2YCR_CB"]),
    (CSPACE_BGR, CSPACE_HSV): cv2.COLOR_BGR2HSV,
    (CSPACE_BGR, CSPACE_HLS): cv2.COLOR_BGR2HLS,
    (CSPACE_BGR, CSPACE_Lab): _get_opencv_attr(["COLOR_BGR2LAB", "COLOR_BGR2Lab"]),
    (CSPACE_BGR, CSPACE_Luv): cv2.COLOR_BGR2LUV,
    (CSPACE_BGR, CSPACE_YUV): cv2.COLOR_BGR2YUV,
    (CSPACE_BGR, CSPACE_CIE): cv2.COLOR_BGR2XYZ,
    # GRAY
    # YCrCb
    (CSPACE_YCrCb, CSPACE_RGB): _get_opencv_attr(["COLOR_YCrCb2RGB", "COLOR_YCR_CB2RGB"]),
    (CSPACE_YCrCb, CSPACE_BGR): _get_opencv_attr(["COLOR_YCrCb2BGR", "COLOR_YCR_CB2BGR"]),
    # HSV
    (CSPACE_HSV, CSPACE_RGB): cv2.COLOR_HSV2RGB,
    (CSPACE_HSV, CSPACE_BGR): cv2.COLOR_HSV2BGR,
    # HLS
    (CSPACE_HLS, CSPACE_RGB): cv2.COLOR_HLS2RGB,
    (CSPACE_HLS, CSPACE_BGR): cv2.COLOR_HLS2BGR,
    # Lab
    (CSPACE_Lab, CSPACE_RGB): _get_opencv_attr(["COLOR_Lab2RGB", "COLOR_LAB2RGB"]),
    (CSPACE_Lab, CSPACE_BGR): _get_opencv_attr(["COLOR_Lab2BGR", "COLOR_LAB2BGR"]),
    # Luv
    (CSPACE_Luv, CSPACE_RGB): _get_opencv_attr(["COLOR_Luv2RGB", "COLOR_LUV2RGB"]),
    (CSPACE_Luv, CSPACE_BGR): _get_opencv_attr(["COLOR_Luv2BGR", "COLOR_LUV2BGR"]),
    # YUV
    (CSPACE_YUV, CSPACE_RGB): cv2.COLOR_YUV2RGB,
    (CSPACE_YUV, CSPACE_BGR): cv2.COLOR_YUV2BGR,
    # CIE
    (CSPACE_CIE, CSPACE_RGB): cv2.COLOR_XYZ2RGB,
    (CSPACE_CIE, CSPACE_BGR): cv2.COLOR_XYZ2BGR,
}

# This defines which colorspace pairs will be converted in-place in
# change_colorspace_(). Currently, all colorspaces seem to work fine with
# in-place transformations, which is why they are all set to True.
_CHANGE_COLORSPACE_INPLACE = {
    # RGB
    (CSPACE_RGB, CSPACE_BGR): True,
    (CSPACE_RGB, CSPACE_GRAY): True,
    (CSPACE_RGB, CSPACE_YCrCb): True,
    (CSPACE_RGB, CSPACE_HSV): True,
    (CSPACE_RGB, CSPACE_HLS): True,
    (CSPACE_RGB, CSPACE_Lab): True,
    (CSPACE_RGB, CSPACE_Luv): True,
    (CSPACE_RGB, CSPACE_YUV): True,
    (CSPACE_RGB, CSPACE_CIE): True,
    # BGR
    (CSPACE_BGR, CSPACE_RGB): True,
    (CSPACE_BGR, CSPACE_GRAY): True,
    (CSPACE_BGR, CSPACE_YCrCb): True,
    (CSPACE_BGR, CSPACE_HSV): True,
    (CSPACE_BGR, CSPACE_HLS): True,
    (CSPACE_BGR, CSPACE_Lab): True,
    (CSPACE_BGR, CSPACE_Luv): True,
    (CSPACE_BGR, CSPACE_YUV): True,
    (CSPACE_BGR, CSPACE_CIE): True,
    # GRAY
    # YCrCb
    (CSPACE_YCrCb, CSPACE_RGB): True,
    (CSPACE_YCrCb, CSPACE_BGR): True,
    # HSV
    (CSPACE_HSV, CSPACE_RGB): True,
    (CSPACE_HSV, CSPACE_BGR): True,
    # HLS
    (CSPACE_HLS, CSPACE_RGB): True,
    (CSPACE_HLS, CSPACE_BGR): True,
    # Lab
    (CSPACE_Lab, CSPACE_RGB): True,
    (CSPACE_Lab, CSPACE_BGR): True,
    # Luv
    (CSPACE_Luv, CSPACE_RGB): True,
    (CSPACE_Luv, CSPACE_BGR): True,
    # YUV
    (CSPACE_YUV, CSPACE_RGB): True,
    (CSPACE_YUV, CSPACE_BGR): True,
    # CIE
    (CSPACE_CIE, CSPACE_RGB): True,
    (CSPACE_CIE, CSPACE_BGR): True,
}


_CSPACE_OPENCV_CONV_VARS = {
    # RGB
    (CSPACE_RGB, CSPACE_BGR): cv2.COLOR_RGB2BGR,
    (CSPACE_RGB, CSPACE_GRAY): cv2.COLOR_RGB2GRAY,
    (CSPACE_RGB, CSPACE_YCrCb): _get_opencv_attr(["COLOR_RGB2YCR_CB"]),
    (CSPACE_RGB, CSPACE_HSV): cv2.COLOR_RGB2HSV,
    (CSPACE_RGB, CSPACE_HLS): cv2.COLOR_RGB2HLS,
    (CSPACE_RGB, CSPACE_Lab): _get_opencv_attr(["COLOR_RGB2LAB", "COLOR_RGB2Lab"]),
    (CSPACE_RGB, CSPACE_Luv): cv2.COLOR_RGB2LUV,
    (CSPACE_RGB, CSPACE_YUV): cv2.COLOR_RGB2YUV,
    (CSPACE_RGB, CSPACE_CIE): cv2.COLOR_RGB2XYZ,
    # BGR
    (CSPACE_BGR, CSPACE_RGB): cv2.COLOR_BGR2RGB,
    (CSPACE_BGR, CSPACE_GRAY): cv2.COLOR_BGR2GRAY,
    (CSPACE_BGR, CSPACE_YCrCb): _get_opencv_attr(["COLOR_BGR2YCR_CB"]),
    (CSPACE_BGR, CSPACE_HSV): cv2.COLOR_BGR2HSV,
    (CSPACE_BGR, CSPACE_HLS): cv2.COLOR_BGR2HLS,
    (CSPACE_BGR, CSPACE_Lab): _get_opencv_attr(["COLOR_BGR2LAB", "COLOR_BGR2Lab"]),
    (CSPACE_BGR, CSPACE_Luv): cv2.COLOR_BGR2LUV,
    (CSPACE_BGR, CSPACE_YUV): cv2.COLOR_BGR2YUV,
    (CSPACE_BGR, CSPACE_CIE): cv2.COLOR_BGR2XYZ,
    # GRAY
    # YCrCb
    (CSPACE_YCrCb, CSPACE_RGB): _get_opencv_attr(["COLOR_YCrCb2RGB", "COLOR_YCR_CB2RGB"]),
    (CSPACE_YCrCb, CSPACE_BGR): _get_opencv_attr(["COLOR_YCrCb2BGR", "COLOR_YCR_CB2BGR"]),
    # HSV
    (CSPACE_HSV, CSPACE_RGB): cv2.COLOR_HSV2RGB,
    (CSPACE_HSV, CSPACE_BGR): cv2.COLOR_HSV2BGR,
    # HLS
    (CSPACE_HLS, CSPACE_RGB): cv2.COLOR_HLS2RGB,
    (CSPACE_HLS, CSPACE_BGR): cv2.COLOR_HLS2BGR,
    # Lab
    (CSPACE_Lab, CSPACE_RGB): _get_opencv_attr(["COLOR_Lab2RGB", "COLOR_LAB2RGB"]),
    (CSPACE_Lab, CSPACE_BGR): _get_opencv_attr(["COLOR_Lab2BGR", "COLOR_LAB2BGR"]),
    # Luv
    (CSPACE_Luv, CSPACE_RGB): _get_opencv_attr(["COLOR_Luv2RGB", "COLOR_LUV2RGB"]),
    (CSPACE_Luv, CSPACE_BGR): _get_opencv_attr(["COLOR_Luv2BGR", "COLOR_LUV2BGR"]),
    # YUV
    (CSPACE_YUV, CSPACE_RGB): cv2.COLOR_YUV2RGB,
    (CSPACE_YUV, CSPACE_BGR): cv2.COLOR_YUV2BGR,
    # CIE
    (CSPACE_CIE, CSPACE_RGB): cv2.COLOR_XYZ2RGB,
    (CSPACE_CIE, CSPACE_BGR): cv2.COLOR_XYZ2BGR,
}

# This defines which colorspace pairs will be converted in-place in
# change_colorspace_(). Currently, all colorspaces seem to work fine with
# in-place transformations, which is why they are all set to True.
_CHANGE_COLORSPACE_INPLACE = {
    # RGB
    (CSPACE_RGB, CSPACE_BGR): True,
    (CSPACE_RGB, CSPACE_GRAY): True,
    (CSPACE_RGB, CSPACE_YCrCb): True,
    (CSPACE_RGB, CSPACE_HSV): True,
    (CSPACE_RGB, CSPACE_HLS): True,
    (CSPACE_RGB, CSPACE_Lab): True,
    (CSPACE_RGB, CSPACE_Luv): True,
    (CSPACE_RGB, CSPACE_YUV): True,
    (CSPACE_RGB, CSPACE_CIE): True,
    # BGR
    (CSPACE_BGR, CSPACE_RGB): True,
    (CSPACE_BGR, CSPACE_GRAY): True,
    (CSPACE_BGR, CSPACE_YCrCb): True,
    (CSPACE_BGR, CSPACE_HSV): True,
    (CSPACE_BGR, CSPACE_HLS): True,
    (CSPACE_BGR, CSPACE_Lab): True,
    (CSPACE_BGR, CSPACE_Luv): True,
    (CSPACE_BGR, CSPACE_YUV): True,
    (CSPACE_BGR, CSPACE_CIE): True,
    # GRAY
    # YCrCb
    (CSPACE_YCrCb, CSPACE_RGB): True,
    (CSPACE_YCrCb, CSPACE_BGR): True,
    # HSV
    (CSPACE_HSV, CSPACE_RGB): True,
    (CSPACE_HSV, CSPACE_BGR): True,
    # HLS
    (CSPACE_HLS, CSPACE_RGB): True,
    (CSPACE_HLS, CSPACE_BGR): True,
    # Lab
    (CSPACE_Lab, CSPACE_RGB): True,
    (CSPACE_Lab, CSPACE_BGR): True,
    # Luv
    (CSPACE_Luv, CSPACE_RGB): True,
    (CSPACE_Luv, CSPACE_BGR): True,
    # YUV
    (CSPACE_YUV, CSPACE_RGB): True,
    (CSPACE_YUV, CSPACE_BGR): True,
    # CIE
    (CSPACE_CIE, CSPACE_RGB): True,
    (CSPACE_CIE, CSPACE_BGR): True,
}


def change_colorspace_(
    image: Array, to_colorspace: ColorSpace, from_colorspace: ColorSpace = CSPACE_RGB
) -> Array:
    """Change the colorspace of an image inplace.

    .. note::

        All outputs of this function are `uint8`. For some colorspaces this
        may not be optimal.

    .. note::

        Output grayscale images will still have three channels.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        The image to convert from one colorspace into another.
        Usually expected to have shape ``(H,W,3)``.

    to_colorspace : str
        The target colorspace. See the ``CSPACE`` constants,
        e.g. ``imgaug2.augmenters.color.CSPACE_RGB``.

    from_colorspace : str, optional
        The source colorspace. Analogous to `to_colorspace`. Defaults
        to ``RGB``.

    Returns
    -------
    ndarray
        Image with target colorspace. *Can* be the same array instance as was
        originally provided (i.e. changed inplace). Grayscale images will
        still have three channels.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> import numpy as np
    >>> # fake RGB image
    >>> image_rgb = np.arange(4*4*3).astype(np.uint8).reshape((4, 4, 3))
    >>> image_bgr = iaa.change_colorspace_(np.copy(image_rgb), iaa.CSPACE_BGR)

    """
    # some colorspaces here should use image/255.0 according to
    # the docs, but at least for conversion to grayscale that
    # results in errors, ie uint8 is expected

    # this was once used to accomodate for image .flags -- still necessary?
    def _get_dst(image_: Array, from_to_cspace: tuple[ColorSpace, ColorSpace]) -> Array | None:
        if _CHANGE_COLORSPACE_INPLACE[from_to_cspace]:
            return image_
        return None

    # cv2 does not support height/width 0
    # we don't check here if the channel axis is zero-sized as for colorspace
    # transformations it should never be 0
    if 0 in image.shape[0:2]:
        return image

    iadt.allow_only_uint8({image.dtype})

    for arg_name in ["to_colorspace", "from_colorspace"]:
        assert locals()[arg_name] in CSPACE_ALL, (
            f"Expected `{arg_name}` to be one of: {CSPACE_ALL}. Got: {locals()[arg_name]}."
        )

    assert from_colorspace != CSPACE_GRAY, (
        "Cannot convert from grayscale to another colorspace as colors cannot be recovered."
    )

    assert image.ndim == 3, (
        "Expected image shape to be three-dimensional, i.e. (H,W,C), "
        f"got {image.ndim} dimensions with shape {image.shape}."
    )
    assert image.shape[2] == 3, (
        "Expected number of channels to be three, "
        f"got {image.shape[2]} channels (shape {image.shape})."
    )

    if from_colorspace == to_colorspace:
        return image

    from_to_direct = (from_colorspace, to_colorspace)
    from_to_indirect = [(from_colorspace, CSPACE_RGB), (CSPACE_RGB, to_colorspace)]

    image = _normalize_cv2_input_arr_(image)
    image_aug = image
    if from_to_direct in _CSPACE_OPENCV_CONV_VARS:
        from2to_var = _CSPACE_OPENCV_CONV_VARS[from_to_direct]
        dst = _get_dst(image_aug, from_to_direct)
        image_aug = cv2.cvtColor(image_aug, from2to_var, dst=dst)
    else:
        from2rgb_var = _CSPACE_OPENCV_CONV_VARS[from_to_indirect[0]]
        rgb2to_var = _CSPACE_OPENCV_CONV_VARS[from_to_indirect[1]]

        dst1 = _get_dst(image_aug, from_to_indirect[0])
        dst2 = _get_dst(image_aug, from_to_indirect[1])

        image_aug = cv2.cvtColor(image_aug, from2rgb_var, dst=dst1)
        image_aug = cv2.cvtColor(image_aug, rgb2to_var, dst=dst2)

    # for grayscale: covnert from (H, W) to (H, W, 3)
    if len(image_aug.shape) == 2:
        image_aug = image_aug[:, :, np.newaxis]
        image_aug = np.tile(image_aug, (1, 1, 3))

    return image_aug


def change_colorspaces_(
    images: Images,
    to_colorspaces: ColorSpaceInput,
    from_colorspaces: ColorSpaceInput = CSPACE_RGB,
) -> Images:
    """Change the colorspaces of a batch of images inplace.

    .. note::

        All outputs of this function are `uint8`. For some colorspaces this
        may not be optimal.

    .. note::

        Output grayscale images will still have three channels.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.color.change_colorspace_`.

    Parameters
    ----------
    images : ndarray or list of ndarray
        The images to convert from one colorspace into another.
        Either a list of ``(H,W,3)`` arrays or a single ``(N,H,W,3)`` array.

    to_colorspaces : str or iterable of str
        The target colorspaces. Either a single string (all images will be
        converted to the same colorspace) or an iterable of strings (one per
        image). See the ``CSPACE`` constants, e.g.
        ``imgaug2.augmenters.color.CSPACE_RGB``.

    from_colorspaces : str or list of str, optional
        The source colorspace. Analogous to `to_colorspace`. Defaults
        to ``RGB``.

    Returns
    -------
    ndarray or list of ndarray
        Images with target colorspaces. *Can* contain the same array instances
        as were originally provided (i.e. changed inplace). Grayscale images
        will still have three channels.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> import numpy as np
    >>> # fake RGB image
    >>> image_rgb = np.arange(4*4*3).astype(np.uint8).reshape((4, 4, 3))
    >>> images_rgb = [image_rgb, image_rgb, image_rgb]
    >>> images_rgb_copy = [np.copy(image_rgb) for image_rgb in images_rgb]
    >>> images_bgr = iaa.change_colorspaces_(images_rgb_copy, iaa.CSPACE_BGR)

    Create three example ``RGB`` images and convert them to ``BGR`` colorspace.

    >>> images_rgb_copy = [np.copy(image_rgb) for image_rgb in images_rgb]
    >>> images_various = iaa.change_colorspaces_(
    >>>     images_rgb_copy, [iaa.CSPACE_BGR, iaa.CSPACE_HSV, iaa.CSPACE_GRAY])

    Chnage the colorspace of the first image to ``BGR``, the one of the second
    image to ``HSV`` and the one of the third image to ``grayscale`` (note
    that in the latter case the image will still have shape ``(H,W,3)``,
    not ``(H,W,1)``).

    """

    def _validate(arg: ColorSpaceInput, arg_name: str) -> Sequence[ColorSpace]:
        if ia.is_string(arg):
            arg = [arg] * len(images)
        else:
            assert ia.is_iterable(arg), (
                f"Expected `{arg_name}` to be either an iterable of strings or a single "
                f"string. Got type: {type(arg).__name__}."
            )
            assert len(arg) == len(images), (
                f"If `{arg_name}` is provided as a list it must have the same length "
                f"as `images`. Got length {len(arg)}, expected {len(images)}."
            )

        return cast(Sequence[ColorSpace], arg)

    to_colorspaces = _validate(to_colorspaces, "to_colorspaces")
    from_colorspaces = _validate(from_colorspaces, "from_colorspaces")

    gen = zip(images, to_colorspaces, from_colorspaces, strict=True)
    for i, (image, to_colorspace, from_colorspace) in enumerate(gen):
        images[i] = change_colorspace_(image, to_colorspace, from_colorspace)
    return images


@ia.deprecated(alt_func="WithColorspace")
def InColorspace(
    to_colorspace: ColorSpace,
    from_colorspace: ColorSpace = "RGB",
    children: ChildrenInput = None,
    seed: RNGInput = None,
    name: str | None = None,
    random_state: RNGInput | Literal["deprecated"] = "deprecated",
    deterministic: bool | Literal["deprecated"] = "deprecated",
) -> WithColorspace:
    """Convert images to another colorspace."""
    return WithColorspace(
        to_colorspace,
        from_colorspace,
        children,
        seed=seed,
        name=name,
        random_state=random_state,
        deterministic=deterministic,
    )


class WithColorspace(meta.Augmenter):
    """
    Apply child augmenters within a specific colorspace.

    This augumenter takes a source colorspace A and a target colorspace B
    as well as children C. It changes images from A to B, then applies the
    child augmenters C and finally changes the colorspace back from B to A.
    See also ChangeColorspace() for more.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.color.change_colorspaces_`.

    Parameters
    ----------
    to_colorspace : str
        See :func:`~imgaug2.augmenters.color.change_colorspace_`.

    from_colorspace : str, optional
        See :func:`~imgaug2.augmenters.color.change_colorspace_`.

    children : imgaug2.augmenters.meta.Augmenter or list of imgaug2.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to converted images.

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
    >>> aug = iaa.WithColorspace(
    >>>     to_colorspace=iaa.CSPACE_HSV,
    >>>     from_colorspace=iaa.CSPACE_RGB,
    >>>     children=iaa.WithChannels(
    >>>         0,
    >>>         iaa.Add((0, 50))
    >>>     )
    >>> )

    Convert to ``HSV`` colorspace, add a value between ``0`` and ``50``
    (uniformly sampled per image) to the Hue channel, then convert back to the
    input colorspace (``RGB``).

    """

    def __init__(
        self,
        to_colorspace: ColorSpaceInput,
        from_colorspace: ColorSpace = CSPACE_RGB,
        children: ChildrenInput = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.to_colorspace = to_colorspace
        self.from_colorspace = from_colorspace
        self.children = meta.handle_children_list(children, self.name, "then")

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        with batch.propagation_hooks_ctx(self, hooks, parents):
            # TODO this did not fail in the tests when there was only one
            #      `if` with all three steps in it
            if batch.images is not None:
                batch.images = change_colorspaces_(
                    batch.images,
                    to_colorspaces=self.to_colorspace,
                    from_colorspaces=self.from_colorspace,
                )

            batch = self.children.augment_batch_(batch, parents=parents + [self], hooks=hooks)

            if batch.images is not None:
                batch.images = change_colorspaces_(
                    batch.images,
                    to_colorspaces=self.from_colorspace,
                    from_colorspaces=self.to_colorspace,
                )
        return batch

    def _to_deterministic(self) -> meta.Augmenter:
        aug = self.copy()
        aug.children = aug.children.to_deterministic()
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.to_colorspace, self.from_colorspace]

    def get_children_lists(self) -> list[list[meta.Augmenter]]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_children_lists`."""
        return cast(list[list[meta.Augmenter]], [self.children])

    def __str__(self) -> str:
        return (
            f"WithColorspace(from_colorspace={self.from_colorspace}, "
            f"to_colorspace={self.to_colorspace}, name={self.name}, children=[{self.children}], deterministic={self.deterministic})"
        )


class ChangeColorspace(meta.Augmenter):
    """
    Augmenter to change the colorspace of images.

    ..note::

        This augmenter tries to project the colorspace value range on
        0-255. It outputs dtype=uint8 images.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.color.change_colorspace_`.

    Parameters
    ----------
    to_colorspace : str or list of str or imgaug2.parameters.StochasticParameter
        The target colorspace.
        Allowed strings are: ``RGB``, ``BGR``, ``GRAY``, ``CIE``, ``YCrCb``,
        ``HSV``, ``HLS``, ``Lab``, ``Luv``.
        These are also accessible via
        ``imgaug2.augmenters.color.CSPACE_<NAME>``,
        e.g. ``imgaug2.augmenters.CSPACE_YCrCb``.

            * If a string, it must be among the allowed colorspaces.
            * If a list, it is expected to be a list of strings, each one
              being an allowed colorspace. A random element from the list
              will be chosen per image.
            * If a StochasticParameter, it is expected to return string. A new
              sample will be drawn per image.

    from_colorspace : str, optional
        The source colorspace (of the input images).
        See `to_colorspace`. Only a single string is allowed.

    alpha : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        The alpha value of the new colorspace when overlayed over the
        old one. A value close to 1.0 means that mostly the new
        colorspace is visible. A value close to 0.0 means, that mostly the
        old image is visible.

            * If an int or float, exactly that value will be used.
            * If a tuple ``(a, b)``, a random value from the range
              ``a <= x <= b`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

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

    """

    # TODO mark these as deprecated
    RGB = CSPACE_RGB
    BGR = CSPACE_BGR
    GRAY = CSPACE_GRAY
    CIE = CSPACE_CIE
    YCrCb = CSPACE_YCrCb
    HSV = CSPACE_HSV
    HLS = CSPACE_HLS
    Lab = CSPACE_Lab
    Luv = CSPACE_Luv
    COLORSPACES = {RGB, BGR, GRAY, CIE, YCrCb, HSV, HLS, Lab, Luv}
    # TODO access cv2 COLOR_ variables directly instead of indirectly via
    #      dictionary mapping
    CV_VARS = {
        # RGB
        "RGB2BGR": cv2.COLOR_RGB2BGR,
        "RGB2GRAY": cv2.COLOR_RGB2GRAY,
        "RGB2CIE": cv2.COLOR_RGB2XYZ,
        "RGB2YCrCb": cv2.COLOR_RGB2YCR_CB,
        "RGB2HSV": cv2.COLOR_RGB2HSV,
        "RGB2HLS": cv2.COLOR_RGB2HLS,
        "RGB2Lab": cv2.COLOR_RGB2LAB,
        "RGB2Luv": cv2.COLOR_RGB2LUV,
        # BGR
        "BGR2RGB": cv2.COLOR_BGR2RGB,
        "BGR2GRAY": cv2.COLOR_BGR2GRAY,
        "BGR2CIE": cv2.COLOR_BGR2XYZ,
        "BGR2YCrCb": cv2.COLOR_BGR2YCR_CB,
        "BGR2HSV": cv2.COLOR_BGR2HSV,
        "BGR2HLS": cv2.COLOR_BGR2HLS,
        "BGR2Lab": cv2.COLOR_BGR2LAB,
        "BGR2Luv": cv2.COLOR_BGR2LUV,
        # HSV
        "HSV2RGB": cv2.COLOR_HSV2RGB,
        "HSV2BGR": cv2.COLOR_HSV2BGR,
        # HLS
        "HLS2RGB": cv2.COLOR_HLS2RGB,
        "HLS2BGR": cv2.COLOR_HLS2BGR,
        # Lab
        "Lab2RGB": (cv2.COLOR_Lab2RGB if hasattr(cv2, "COLOR_Lab2RGB") else cv2.COLOR_LAB2RGB),
        "Lab2BGR": (cv2.COLOR_Lab2BGR if hasattr(cv2, "COLOR_Lab2BGR") else cv2.COLOR_LAB2BGR),
    }

    def __init__(
        self,
        to_colorspace: ToColorspaceChoiceInput,
        from_colorspace: ColorSpace = CSPACE_RGB,
        alpha: ParamInput = 1.0,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        # TODO somehow merge this with Alpha augmenter?
        self.alpha = iap.handle_continuous_param(
            alpha,
            "alpha",
            value_range=(0, 1.0),
            tuple_to_uniform=True,
            list_to_choice=True,
            prefetch=False,
        )

        if ia.is_string(to_colorspace):
            assert to_colorspace in CSPACE_ALL, (
                f"Expected 'to_colorspace' to be one of {CSPACE_ALL}. Got {to_colorspace}."
            )
            self.to_colorspace = iap.Deterministic(to_colorspace)
        elif ia.is_iterable(to_colorspace):
            all_strings = all([ia.is_string(colorspace) for colorspace in to_colorspace])
            assert all_strings, (
                "Expected list of 'to_colorspace' to only contain strings. Got types {}.".format(
                    ", ".join([str(type(v)) for v in to_colorspace])
                )
            )
            all_valid = all([(colorspace in CSPACE_ALL) for colorspace in to_colorspace])
            assert all_valid, (
                "Expected list of 'to_colorspace' to only contain strings "
                f"that are in {CSPACE_ALL}. Got strings {to_colorspace}."
            )
            self.to_colorspace = iap.Choice(to_colorspace)
        elif isinstance(to_colorspace, iap.StochasticParameter):
            self.to_colorspace = to_colorspace
        else:
            raise Exception(
                "Expected to_colorspace to be string, list of "
                f"strings or StochasticParameter, got {type(to_colorspace)}."
            )
        # Keep the raw parameter accessible (Choice/Deterministic/...), matching
        # the original imgaug API. (Prefetching can be applied internally if
        # needed, but should not change the public attribute type.)

        assert ia.is_string(from_colorspace), (
            f"Expected from_colorspace to be a single string, got type {type(from_colorspace)}."
        )
        assert from_colorspace in CSPACE_ALL, (
            "Expected from_colorspace to be one of: {}. Got: {}.".format(
                ", ".join(CSPACE_ALL), from_colorspace
            )
        )
        assert from_colorspace != CSPACE_GRAY, (
            "Cannot convert from grayscale images to other colorspaces."
        )
        self.from_colorspace = from_colorspace

        # epsilon value to check if alpha is close to 1.0 or 0.0
        self.eps = 0.001

    def _draw_samples(self, n_augmentables: int, random_state: iarandom.RNG) -> tuple[Array, Array]:
        rss = random_state.duplicate(2)
        alphas = self.alpha.draw_samples((n_augmentables,), random_state=rss[0])
        to_colorspaces = self.to_colorspace.draw_samples((n_augmentables,), random_state=rss[1])
        return alphas, to_colorspaces

    def _draw_to_colorspace(self, random_state: iarandom.RNG) -> ColorSpace:
        _, to_colorspaces = self._draw_samples(1, random_state)
        return to_colorspaces[0]

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

        images = batch.images
        nb_images = len(images)
        alphas, to_colorspaces = self._draw_samples(nb_images, random_state)
        for i, image in enumerate(images):
            alpha = alphas[i]
            to_colorspace = to_colorspaces[i]

            assert to_colorspace in CSPACE_ALL, (
                f"Expected 'to_colorspace' to be one of {CSPACE_ALL}. Got {to_colorspace}."
            )

            if alpha <= self.eps or self.from_colorspace == to_colorspace:
                pass  # no change necessary
            else:
                image_aug = change_colorspace_(image, to_colorspace, self.from_colorspace)
                batch.images[i] = blend_alpha(image_aug, image, alpha, self.eps)

        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        # Match the original imgaug API: include the (string) source colorspace
        # between the stochastic params for easier introspection.
        return [self.to_colorspace, self.from_colorspace, self.alpha]


# TODO This should rather do the blending in RGB or BGR space.
#      Currently, if the input image is in e.g. HSV space, it will blend in
#      that space.
# TODO rename to Grayscale3D and add Grayscale that keeps the image at 1D?
class Grayscale(ChangeColorspace):
    """Augmenter to convert images to their grayscale versions.

    .. note::

        Number of output channels is still ``3``, i.e. this augmenter just
        "removes" color.

    TODO check dtype support

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.color.change_colorspace_`.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        The alpha value of the grayscale image when overlayed over the
        old image. A value close to 1.0 means, that mostly the new grayscale
        image is visible. A value close to 0.0 means, that mostly the
        old image is visible.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the range
              ``a <= x <= b`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    from_colorspace : str, optional
        The source colorspace (of the input images).
        See :func:`~imgaug2.augmenters.color.change_colorspace_`.

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
    >>> aug = iaa.Grayscale(alpha=1.0)

    Creates an augmenter that turns images to their grayscale versions.

    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.Grayscale(alpha=(0.0, 1.0))

    Creates an augmenter that turns images to their grayscale versions with
    an alpha value in the range ``0 <= alpha <= 1``. An alpha value of 0.5 would
    mean, that the output image is 50 percent of the input image and 50
    percent of the grayscale image (i.e. 50 percent of color removed).

    """

    def __init__(
        self,
        alpha: ParamInput = 1,
        from_colorspace: ColorSpace = CSPACE_RGB,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            to_colorspace=CSPACE_GRAY,
            alpha=alpha,
            from_colorspace=from_colorspace,
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
        if batch.images is None:
            return batch

        # MLX fast-path
        from imgaug2.mlx._core import to_mlx

        if is_mlx_array(batch.images):
            images_mlx = batch.images
            nb_images = len(images_mlx)
            rss = random_state.duplicate(2)
            alphas = self.alpha.draw_samples((nb_images,), random_state=rss[0])

            # Convert alphas to MLX array and reshape for broadcasting: (N, 1, 1, 1)
            # We assume images are (N, H, W, C)
            alphas_mlx = to_mlx(alphas).reshape(-1, 1, 1, 1)

            # Get full grayscale version (alpha=1.0)
            gray_mlx = mlx_color.grayscale(images_mlx, alpha=1.0)

            # Blend: constant float/int math works with MLX arrays
            # image * (1 - alpha) + gray * alpha
            # Ensure alphas are float32 for blending
            alphas_mlx = alphas_mlx.astype(images_mlx.dtype)

            batch.images = images_mlx * (1 - alphas_mlx) + gray_mlx * alphas_mlx
            return batch

        if _is_mlx_list(batch.images):
            images_list = list(batch.images)
            nb_images = len(images_list)
            rss = random_state.duplicate(2)
            alphas = self.alpha.draw_samples((nb_images,), random_state=rss[0])

            for i, image in enumerate(images_list):
                alpha = float(alphas[i])
                gray = mlx_color.grayscale(image, alpha=1.0)
                images_list[i] = image * (1 - alpha) + gray * alpha

            batch.images = images_list
            return batch

        return super()._augment_batch_(batch, random_state, parents, hooks)
