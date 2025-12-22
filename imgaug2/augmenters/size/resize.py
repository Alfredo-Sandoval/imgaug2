"""Resize augmenters."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Literal

import numpy as np

import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, Images, RNGInput
from imgaug2.compat.markers import legacy

from ._utils import (
    ResizeInterpolationInput,
    ResizeSamplingResult,
    ResizeSizeDictValue,
    ResizeSizeInput,
    ResizeSizeParam,
    Shape,
    _mlx_resize_order_from_interpolation,
)
def Scale(*args: object, **kwargs: object) -> Resize:
    """Augmenter that resizes images to specified heights and widths."""
    return Resize(*args, **kwargs)


class Resize(meta.Augmenter):
    """Augmenter that resizes images to specified heights and widths.

    **Supported dtypes**:

    See :func:`~imgaug2.imgaug2.imresize_many_images`.

    Parameters
    ----------
    size : 'keep' or int or float or tuple of int or tuple of float or list of int or list of float or imgaug2.parameters.StochasticParameter or dict
        The new size of the images.

            * If this has the string value ``keep``, the original height and
              width values will be kept (image is not resized).
            * If this is an ``int``, this value will always be used as the new
              height and width of the images.
            * If this is a ``float`` ``v``, then per image the image's height
              ``H`` and width ``W`` will be changed to ``H*v`` and ``W*v``.
            * If this is a ``tuple``, it is expected to have two entries
              ``(a, b)``. If at least one of these are ``float`` s, a value
              will be sampled from range ``[a, b]`` and used as the ``float``
              value to resize the image (see above). If both are ``int`` s, a
              value will be sampled from the discrete range ``[a..b]`` and
              used as the integer value to resize the image (see above).
            * If this is a ``list``, a random value from the ``list`` will be
              picked to resize the image. All values in the ``list`` must be
              ``int`` s or ``float`` s (no mixture is possible).
            * If this is a ``StochasticParameter``, then this parameter will
              first be queried once per image. The resulting value will be used
              for both height and width.
            * If this is a ``dict``, it may contain the keys ``height`` and
              ``width`` or the keys ``shorter-side`` and ``longer-side``. Each
              key may have the same datatypes as above and describes the
              scaling on x and y-axis or the shorter and longer axis,
              respectively. Both axis are sampled independently. Additionally,
              one of the keys may have the value ``keep-aspect-ratio``, which
              means that the respective side of the image will be resized so
              that the original aspect ratio is kept. This is useful when only
              resizing one image size by a pixel value (e.g. resize images to
              a height of ``64`` pixels and resize the width so that the
              overall aspect ratio is maintained).

    interpolation : imgaug2.ALL or int or str or list of int or list of str or imgaug2.parameters.StochasticParameter, optional
        Interpolation to use.

            * If ``imgaug2.ALL``, then a random interpolation from ``nearest``,
              ``linear``, ``area`` or ``cubic`` will be picked (per image).
            * If ``int``, then this interpolation will always be used.
              Expected to be any of the following:
              ``cv2.INTER_NEAREST``, ``cv2.INTER_LINEAR``, ``cv2.INTER_AREA``,
              ``cv2.INTER_CUBIC``
            * If string, then this interpolation will always be used.
              Expected to be any of the following:
              ``nearest``, ``linear``, ``area``, ``cubic``
            * If ``list`` of ``int`` / ``str``, then a random one of the values
              will be picked per image as the interpolation.
            * If a ``StochasticParameter``, then this parameter will be
              queried per image and is expected to return an ``int`` or
              ``str``.

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
    >>> aug = iaa.Resize(32)

    Resize all images to ``32x32`` pixels.

    >>> aug = iaa.Resize(0.5)

    Resize all images to ``50`` percent of their original size.

    >>> aug = iaa.Resize((16, 22))

    Resize all images to a random height and width within the discrete
    interval ``[16..22]`` (uniformly sampled per image).

    >>> aug = iaa.Resize((0.5, 0.75))

    Resize all any input image so that its height (``H``) and width (``W``)
    become ``H*v`` and ``W*v``, where ``v`` is uniformly sampled from the
    interval ``[0.5, 0.75]``.

    >>> aug = iaa.Resize([16, 32, 64])

    Resize all images either to ``16x16``, ``32x32`` or ``64x64`` pixels.

    >>> aug = iaa.Resize({"height": 32})

    Resize all images to a height of ``32`` pixels and keeps the original
    width.

    >>> aug = iaa.Resize({"height": 32, "width": 48})

    Resize all images to a height of ``32`` pixels and a width of ``48``.

    >>> aug = iaa.Resize({"height": 32, "width": "keep-aspect-ratio"})

    Resize all images to a height of ``32`` pixels and resizes the
    x-axis (width) so that the aspect ratio is maintained.

    >>> aug = iaa.Resize(
    >>>     {"shorter-side": 224, "longer-side": "keep-aspect-ratio"})

    Resize all images to a height/width of ``224`` pixels, depending on which
    axis is shorter and resize the other axis so that the aspect ratio is
    maintained.

    >>> aug = iaa.Resize({"height": (0.5, 0.75), "width": [16, 32, 64]})

    Resize all images to a height of ``H*v``, where ``H`` is the original
    height and ``v`` is a random value sampled from the interval
    ``[0.5, 0.75]``. The width/x-axis of each image is resized to either
    ``16`` or ``32`` or ``64`` pixels.

    >>> aug = iaa.Resize(32, interpolation=["linear", "cubic"])

    Resize all images to ``32x32`` pixels. Randomly use either ``linear``
    or ``cubic`` interpolation.

    """

    def __init__(
        self,
        size: ResizeSizeInput,
        interpolation: ResizeInterpolationInput = "cubic",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        self.size, self.size_order = self._handle_size_arg(size, False)
        self.interpolation = self._handle_interpolation_arg(interpolation)

    @classmethod
    @iap._prefetchable_str
    def _handle_size_arg(
        cls, size: ResizeSizeInput, subcall: bool
    ) -> ResizeSizeParam | tuple[ResizeSizeParam, str]:
        def _dict_to_size_tuple(
            val1: ResizeSizeDictValue, val2: ResizeSizeDictValue
        ) -> tuple[iap.StochasticParameter, iap.StochasticParameter]:
            kaa = "keep-aspect-ratio"
            not_both_kaa = val1 != kaa or val2 != kaa
            assert not_both_kaa, (
                "Expected at least one value to not be \"keep-aspect-ratio\", but got it two times."
            )

            size_tuple = []
            for k in [val1, val2]:
                if k in ["keep-aspect-ratio", "keep"]:
                    entry = iap.Deterministic(k)
                else:
                    entry = cls._handle_size_arg(k, True)
                size_tuple.append(entry)
            return tuple(size_tuple)

        def _contains_any_key(dict_: dict[str, object], keys: Sequence[str]) -> bool:
            return any([key in dict_ for key in keys])

        # HW = height, width
        # SL = shorter, longer
        size_order = "HW"

        if size == "keep":
            result = iap.Deterministic("keep")
        elif ia.is_single_number(size):
            assert size > 0, f"Expected only values > 0, got {size}"
            result = iap.Deterministic(size)
        elif not subcall and isinstance(size, dict):
            if len(size.keys()) == 0:
                result = iap.Deterministic("keep")
            elif _contains_any_key(size, ["height", "width"]):
                height = size.get("height", "keep")
                width = size.get("width", "keep")
                result = _dict_to_size_tuple(height, width)
            elif _contains_any_key(size, ["shorter-side", "longer-side"]):
                shorter = size.get("shorter-side", "keep")
                longer = size.get("longer-side", "keep")
                result = _dict_to_size_tuple(shorter, longer)
                size_order = "SL"
            else:
                raise ValueError(
                    "Expected dictionary containing no keys, "
                    "the keys \"height\" and/or \"width\", "
                    "or the keys \"shorter-side\" and/or \"longer-side\". "
                    f"Got keys: {str(size.keys())}."
                )
        elif isinstance(size, tuple):
            assert len(size) == 2, (
                f"Expected size tuple to contain exactly 2 values, got {len(size)}."
            )
            assert size[0] > 0 and size[1] > 0, (
                f"Expected size tuple to only contain values >0, got {size[0]} and {size[1]}."
            )
            if ia.is_single_float(size[0]) or ia.is_single_float(size[1]):
                result = iap.Uniform(size[0], size[1])
            else:
                result = iap.DiscreteUniform(size[0], size[1])
        elif isinstance(size, list):
            if len(size) == 0:
                result = iap.Deterministic("keep")
            else:
                all_int = all([ia.is_single_integer(v) for v in size])
                all_float = all([ia.is_single_float(v) for v in size])
                assert all_int or all_float, "Expected to get only integers or floats."
                assert all([v > 0 for v in size]), "Expected all values to be >0."
                result = iap.Choice(size)
        elif isinstance(size, iap.StochasticParameter):
            result = size
        else:
            raise ValueError(
                "Expected number, tuple of two numbers, list of numbers, "
                "dictionary of form "
                "{'height': number/tuple/list/'keep-aspect-ratio'/'keep', "
                "'width': <analogous>}, dictionary of form "
                "{'shorter-side': number/tuple/list/'keep-aspect-ratio'/"
                "'keep', 'longer-side': <analogous>} "
                f"or StochasticParameter, got {type(size)}."
            )

        if subcall:
            return result
        return result, size_order

    @classmethod
    def _handle_interpolation_arg(
        cls, interpolation: ResizeInterpolationInput
    ) -> iap.StochasticParameter:
        if interpolation == ia.ALL:
            interpolation = iap.Choice(["nearest", "linear", "area", "cubic"])
        elif ia.is_single_integer(interpolation):
            interpolation = iap.Deterministic(interpolation)
        elif ia.is_string(interpolation):
            interpolation = iap.Deterministic(interpolation)
        elif ia.is_iterable(interpolation):
            interpolation = iap.Choice(interpolation)
        elif isinstance(interpolation, iap.StochasticParameter):
            pass
        else:
            raise Exception(
                "Expected int or string or iterable or StochasticParameter, "
                f"got {type(interpolation)}."
            )
        return interpolation

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        nb_rows = batch.nb_rows
        samples = self._draw_samples(nb_rows, random_state)

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images, samples)

        if batch.heatmaps is not None:
            # TODO this uses the same interpolation as for images for heatmaps
            #      while other augmenters resort to cubic
            batch.heatmaps = self._augment_maps_by_samples(batch.heatmaps, "arr_0to1", samples)

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps, "arr", (samples[0], samples[1], [None] * nb_rows)
            )

        for augm_name in ["keypoints", "bounding_boxes", "polygons", "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(self._augment_keypoints_by_samples, samples=samples)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    @legacy(version="0.4.0")
    def _augment_images_by_samples(self, images: Images, samples: ResizeSamplingResult) -> Images:
        input_was_array = False
        input_dtype = None
        if ia.is_np_array(images):
            input_was_array = True
            input_dtype = images.dtype

        samples_a, samples_b, samples_ip = samples
        result = []
        for i, image in enumerate(images):
            h, w = self._compute_height_width(
                image.shape, samples_a[i], samples_b[i], self.size_order
            )
            from imgaug2.mlx._core import is_mlx_array

            if is_mlx_array(image):
                import imgaug2.mlx as mlx

                if image.size == 0:
                    image_rs = image
                else:
                    order = _mlx_resize_order_from_interpolation(samples_ip[i])
                    if order is None:
                        image_np = mlx.to_numpy(image)
                        image_np = ia.imresize_single_image(
                            image_np, (h, w), interpolation=samples_ip[i]
                        )
                        image_rs = mlx.to_mlx(image_np)
                    else:
                        image_rs = mlx.geometry.resize(image, (h, w), order=order)
            else:
                image_rs = ia.imresize_single_image(image, (h, w), interpolation=samples_ip[i])
            result.append(image_rs)

        if input_was_array:
            all_same_size = len({image.shape for image in result}) == 1
            if all_same_size:
                result = np.array(result, dtype=input_dtype)

        return result

    @legacy(version="0.4.0")
    def _augment_maps_by_samples(
        self,
        augmentables: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage],
        arr_attr_name: str,
        samples: ResizeSamplingResult,
    ) -> list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage]:
        result = []
        samples_h, samples_w, samples_ip = samples

        for i, augmentable in enumerate(augmentables):
            arr = getattr(augmentable, arr_attr_name)
            arr_shape = arr.shape
            img_shape = augmentable.shape
            h_img, w_img = self._compute_height_width(
                img_shape, samples_h[i], samples_w[i], self.size_order
            )
            h = int(np.round(h_img * (arr_shape[0] / img_shape[0])))
            w = int(np.round(w_img * (arr_shape[1] / img_shape[1])))
            h = max(h, 1)
            w = max(w, 1)
            if samples_ip[0] is not None:
                # TODO change this for heatmaps to always have cubic or
                #      automatic interpolation?
                augmentable_resize = augmentable.resize((h, w), interpolation=samples_ip[i])
            else:
                augmentable_resize = augmentable.resize((h, w))
            augmentable_resize.shape = (h_img, w_img) + img_shape[2:]
            result.append(augmentable_resize)

        return result

    @legacy(version="0.4.0")
    def _augment_keypoints_by_samples(
        self, kpsois: list[ia.KeypointsOnImage], samples: ResizeSamplingResult
    ) -> list[ia.KeypointsOnImage]:
        result = []
        samples_a, samples_b, _samples_ip = samples
        for i, kpsoi in enumerate(kpsois):
            h, w = self._compute_height_width(
                kpsoi.shape, samples_a[i], samples_b[i], self.size_order
            )
            new_shape = (h, w) + kpsoi.shape[2:]
            keypoints_on_image_rs = kpsoi.on_(new_shape)

            result.append(keypoints_on_image_rs)

        return result

    def _draw_samples(self, nb_images: int, random_state: iarandom.RNG) -> ResizeSamplingResult:
        rngs = random_state.duplicate(3)
        if isinstance(self.size, tuple):
            samples_h = self.size[0].draw_samples(nb_images, random_state=rngs[0])
            samples_w = self.size[1].draw_samples(nb_images, random_state=rngs[1])
        else:
            samples_h = self.size.draw_samples(nb_images, random_state=rngs[0])
            samples_w = samples_h

        samples_ip = self.interpolation.draw_samples(nb_images, random_state=rngs[2])
        return samples_h, samples_w, samples_ip

    @classmethod
    def _compute_height_width(
        cls,
        image_shape: Shape,
        sample_a: float | int | str,
        sample_b: float | int | str,
        size_order: str,
    ) -> tuple[int, int]:
        imh, imw = image_shape[0:2]

        if size_order == 'SL':
            # size order: short, long
            if imh < imw:
                h, w = sample_a, sample_b
            else:
                w, h = sample_a, sample_b
        else:
            # size order: height, width
            h, w = sample_a, sample_b

        if ia.is_single_float(h):
            assert h > 0, f"Expected 'h' to be >0, got {h:.4f}"
            h = int(np.round(imh * h))
            h = h if h > 0 else 1
        elif h == "keep":
            h = imh
        if ia.is_single_float(w):
            assert w > 0, f"Expected 'w' to be >0, got {w:.4f}"
            w = int(np.round(imw * w))
            w = w if w > 0 else 1
        elif w == "keep":
            w = imw

        # at least the checks for keep-aspect-ratio must come after
        # the float checks, as they are dependent on the results
        # this is also why these are not written as elifs
        if h == "keep-aspect-ratio":
            h_per_w_orig = imh / imw
            h = int(np.round(w * h_per_w_orig))
        if w == "keep-aspect-ratio":
            w_per_h_orig = imw / imh
            w = int(np.round(h * w_per_h_orig))

        return int(h), int(w)

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.size, self.interpolation, self.size_order]


