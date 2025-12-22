from __future__ import annotations

import tempfile
from typing import Literal

import imageio
import numpy as np

import imgaug2.dtypes as iadt
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.compat.markers import legacy
from imgaug2.augmenters import meta
from imgaug2.augmenters._typing import Array, ParamInput, RNGInput


def compress_jpeg(image: Array, compression: int) -> Array:
    """Compress an image using jpeg compression.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: ?
        * ``uint32``: ?
        * ``uint64``: ?
        * ``int8``: ?
        * ``int16``: ?
        * ``int32``: ?
        * ``int64``: ?
        * ``float16``: ?
        * ``float32``: ?
        * ``float64``: ?
        * ``float128``: ?
        * ``bool``: ?

    Parameters
    ----------
    image : ndarray
        Image of dtype ``uint8`` and shape ``(H,W,[C])``. If ``C`` is provided,
        it must be ``1`` or ``3``.

    compression : int
        Strength of the compression in the interval ``[0, 100]``.

    Returns
    -------
    ndarray
        Input image after applying jpeg compression to it and reloading
        the result into a new array. Same shape and dtype as the input.

    """
    import PIL.Image

    if image.size == 0:
        return np.copy(image)

    # The value range 1 to 95 is suggested by PIL's save() documentation
    # Values above 95 seem to not make sense (no improvement in visual
    # quality, but large file size).
    # A value of 100 would mostly deactivate jpeg compression.
    # A value of 0 would lead to no compression (instead of maximum
    # compression).
    # We use range 1 to 100 here, because this augmenter is about
    # generating images for training and not for saving, hence we do not
    # care about large file sizes.
    maximum_quality = 100
    minimum_quality = 1

    iadt.allow_only_uint8({image.dtype})
    assert 0 <= compression <= 100, (
        f"Expected compression to be in the interval [0, 100], got {compression:.4f}."
    )

    has_no_channels = image.ndim == 2
    is_single_channel = image.ndim == 3 and image.shape[-1] == 1
    if is_single_channel:
        image = image[..., 0]

    assert has_no_channels or is_single_channel or image.shape[-1] == 3, (
        "Expected either a grayscale image of shape (H,W) or (H,W,1) or an "
        f"RGB image of shape (H,W,3). Got shape {image.shape}."
    )

    # Map from compression to quality used by PIL
    # We have valid compressions from 0 to 100, i.e. 101 possible
    # values
    quality = int(
        np.clip(
            np.round(
                minimum_quality + (maximum_quality - minimum_quality) * (1.0 - (compression / 101))
            ),
            minimum_quality,
            maximum_quality,
        )
    )

    image_pil = PIL.Image.fromarray(image)
    with tempfile.NamedTemporaryFile(mode="wb+", suffix=".jpg") as f:
        image_pil.save(f, quality=quality)

        # Read back from file.
        # We dont read from f.name, because that leads to PermissionDenied
        # errors on Windows. We add f.seek(0) here, because otherwise we get
        # `SyntaxError: index out of range` in PIL.
        f.seek(0)
        pilmode = "RGB"
        if has_no_channels or is_single_channel:
            pilmode = "L"
        image = imageio.imread(f, pilmode=pilmode, format="jpeg")
    if is_single_channel:
        image = image[..., np.newaxis]
    return image


class JpegCompression(meta.Augmenter):
    """
    Degrade the quality of images by JPEG-compressing them.

    During JPEG compression, high frequency components (e.g. edges) are removed.
    With low compression (strength) only the highest frequency components are
    removed, while very high compression (strength) will lead to only the
    lowest frequency components "surviving". This lowers the image quality.
    For more details, see https://en.wikipedia.org/wiki/Compression_artifact.

    Note that this augmenter still returns images as numpy arrays (i.e. saves
    the images with JPEG compression and then reloads them into arrays). It
    does not return the raw JPEG file content.

    **Supported dtypes**:

    See :func:`~imgaug2.augmenters.arithmetic.compress_jpeg`.

    Parameters
    ----------
    compression : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Degree of compression used during JPEG compression within value range
        ``[0, 100]``. Higher values denote stronger compression and will cause
        low-frequency components to disappear. Note that JPEG's compression
        strength is also often set as a *quality*, which is the inverse of this
        parameter. Common choices for the *quality* setting are around 80 to 95,
        depending on the image. This translates here to a *compression*
        parameter of around 20 to 5.

            * If a single number, then that value always will be used as the
              compression.
            * If a tuple ``(a, b)``, then the compression will be
              a value sampled uniformly from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image and used as the compression.
            * If a ``StochasticParameter``, then ``N`` samples will be drawn
              from that parameter per ``N`` input images, each representing the
              compression for the ``n``-th image.

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
    >>> aug = iaa.JpegCompression(compression=(70, 99))

    Remove high frequency components in images via JPEG compression with
    a *compression strength* between ``70`` and ``99`` (randomly and
    uniformly sampled per image). This corresponds to a (very low) *quality*
    setting of ``1`` to ``30``.

    """

    def __init__(
        self,
        compression: ParamInput = (0, 100),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        # will be converted to int during augmentation, which is why we allow
        # floats here
        self.compression = iap.handle_continuous_param(
            compression,
            "compression",
            value_range=(0, 100),
            tuple_to_uniform=True,
            list_to_choice=True,
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

        images = batch.images
        nb_images = len(images)
        samples = self.compression.draw_samples((nb_images,), random_state=random_state)

        for i, (image, sample) in enumerate(zip(images, samples, strict=True)):
            batch.images[i] = compress_jpeg(image, int(sample))

        return batch

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [self.compression]
