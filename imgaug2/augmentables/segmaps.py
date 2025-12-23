"""Classes dealing with segmentation maps.

E.g. masks, semantic or instance segmentation maps.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

import imgaug2.imgaug as ia

from imgaug2.augmenters._blend_utils import blend_alpha
from imgaug2.augmenters.size._utils import pad, pad_to_aspect_ratio


@ia.deprecated(
    alt_func="SegmentationMapsOnImage",
    comment="(Note the plural 'Maps' instead of old 'Map'.)",
)
def SegmentationMapOnImage(*args: object, **kwargs: object) -> SegmentationMapsOnImage:
    """Object representing a segmentation map associated with an image."""
    return SegmentationMapsOnImage(*args, **kwargs)


_ImageShape = tuple[int, ...]
_Color = tuple[int, int, int]
_SegmapArray = NDArray[np.generic]
_SegmapArrayInt32 = NDArray[np.int32]
_RGBImage = NDArray[np.uint8]


def _generate_default_segment_colors() -> list[_Color]:
    import matplotlib

    colors: list[_Color] = [(0, 0, 0)]
    cmap_names = ("tab20", "tab20b", "tab20c")
    if hasattr(matplotlib, "colormaps"):
        get_cmap = matplotlib.colormaps.__getitem__
    else:
        import matplotlib.cm as cm

        get_cmap = cm.get_cmap

    for cmap_name in cmap_names:
        cmap = get_cmap(cmap_name)
        samples = np.linspace(0, 1, cmap.N, endpoint=False)
        rgba = cmap(samples)
        rgb = np.rint(rgba[:, :3] * 255).astype(np.uint8)
        colors.extend([tuple(int(channel) for channel in row) for row in rgb])
    return colors[:42]


class SegmentationMapsOnImage:
    """
    Object representing a segmentation map associated with an image.

    Attributes
    ----------
    DEFAULT_SEGMENT_COLORS : list of tuple of int
        Standard RGB colors to use during drawing, ordered by class index.
        Generated from the matplotlib ``tab20``, ``tab20b`` and ``tab20c``
        colormaps (with black as the background color at index 0).

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Array representing the segmentation map(s). May have dtypes bool,
        int or uint.

    shape : tuple of int
        Shape of the image on which the segmentation map(s) is/are placed.
        **Not** the shape of the segmentation map(s) array, unless it is
        identical to the image shape (note the likely difference between the
        arrays in the number of channels).
        This is expected to be ``(H, W)`` or ``(H, W, C)`` with ``C`` usually
        being ``3``.
        If there is no corresponding image, use ``(H_arr, W_arr)`` instead,
        where ``H_arr`` is the height of the segmentation map(s) array
        (analogous ``W_arr``).

    nb_classes : None or int, optional
        Deprecated.

    """

    DEFAULT_SEGMENT_COLORS = _generate_default_segment_colors()

    def __init__(
        self, arr: _SegmapArray, shape: _ImageShape, nb_classes: int | None = None
    ) -> None:
        assert ia.is_np_array(arr), f"Expected to get numpy array, got {type(arr)}."
        assert arr.ndim in [2, 3], (
            "Expected segmentation map array to be 2- or "
            f"3-dimensional, got {arr.ndim} dimensions and shape {arr.shape}."
        )
        assert isinstance(shape, tuple), (
            "Expected 'shape' to be a tuple denoting the shape of the image "
            f"on which the segmentation map is placed. Got type {type(shape)} instead."
        )

        if arr.dtype.kind == "f":
            ia.warn_deprecated(
                "Got a float array as the segmentation map in "
                "SegmentationMapsOnImage. That is deprecated. Please provide "
                "instead a (H,W,[C]) array of dtype bool_, int or uint, where "
                "C denotes the segmentation map index."
            )

            if arr.ndim == 2:
                arr = arr > 0.5
            else:  # arr.ndim == 3
                arr = np.argmax(arr, axis=2).astype(np.int32)

        if arr.dtype.name == "bool":
            self._input_was: tuple[np.dtype, int] = (arr.dtype, arr.ndim)
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
        elif arr.dtype.kind in ["i", "u"]:
            assert np.min(arr.flat[0:100]) >= 0, (
                "Expected segmentation map array to only contain values >=0, "
                f"got a minimum of {np.min(arr)}."
            )
            if arr.dtype.kind == "u":
                # allow only <=uint16 due to conversion to int32
                assert arr.dtype.itemsize <= 2, (
                    "When using uint arrays as segmentation maps, only uint8 "
                    f"and uint16 are allowed. Got dtype {arr.dtype.name}."
                )
            elif arr.dtype.kind == "i":
                # allow only <=uint16 due to conversion to int32
                assert arr.dtype.itemsize <= 4, (
                    "When using int arrays as segmentation maps, only int8, "
                    f"int16 and int32 are allowed. Got dtype {arr.dtype.name}."
                )

            self._input_was = (arr.dtype, arr.ndim)
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
        else:
            raise Exception(
                "Input was expected to be an array of dtype 'bool', 'int' "
                f"or 'uint'. Got dtype '{arr.dtype.name}'."
            )

        if arr.dtype.name != "int32":
            arr = arr.astype(np.int32)

        self.arr: _SegmapArrayInt32 = cast(_SegmapArrayInt32, arr)
        self.shape = shape

        if nb_classes is not None:
            ia.warn_deprecated(
                "Providing nb_classes to SegmentationMapsOnImage is no longer "
                "necessary and hence deprecated. The argument is ignored "
                "and can be safely removed."
            )

    def get_arr(self) -> _SegmapArray:
        """Return the seg.map array, with original dtype and shape ndim.

        Here, "original" denotes the dtype and number of shape dimensions that
        was used when the `SegmentationMapsOnImage` instance was
        created, i.e. upon the call of
        `__init__()`.
        Internally, this class may use a different dtype and shape to simplify
        computations.

        .. note::

            The height and width may have changed compared to the original
            input due to e.g. pooling operations.

        Returns
        -------
        ndarray
            Segmentation map array.
            Same dtype and number of dimensions as was originally used when
            the `SegmentationMapsOnImage` instance was created.

        """
        input_dtype, input_ndim = self._input_was
        # The internally used int32 has a wider value range than any other
        # input dtype, hence we can simply convert via astype() here.
        arr_input = self.arr.astype(input_dtype)
        if input_ndim == 2:
            assert arr_input.shape[2] == 1, (
                "Originally got a (H,W) segmentation map. Internal array "
                f"should now have shape (H,W,1), but got {arr_input.shape}. This might be "
                "an internal error."
            )
            return arr_input[:, :, 0]
        return arr_input

    @ia.deprecated(alt_func="SegmentationMapsOnImage.get_arr()")
    def get_arr_int(self, *args: object, **kwargs: object) -> _SegmapArray:
        """Return the seg.map array, with original dtype and shape ndim."""
        return self.get_arr()

    def draw(
        self,
        size: float | int | Sequence[float | int] | None = None,
        colors: Sequence[_Color] | None = None,
    ) -> list[_RGBImage]:
        """
        Render the segmentation map as an RGB image.

        Parameters
        ----------
        size : None or float or iterable of int or iterable of float, optional
            Size of the rendered RGB image as ``(height, width)``.
            See `imresize_single_image()` for details.
            If set to ``None``, no resizing is performed and the size of the
            segmentation map array is used.

        colors : None or list of tuple of int, optional
            Colors to use. One for each class to draw.
            If ``None``, then default colors will be used.

        Returns
        -------
        list of (H,W,3) ndarray
            Rendered segmentation map (dtype is ``uint8``).
            One per ``C`` in the original input array ``(H,W,C)``.

        """

        def _handle_sizeval(sizeval: float | int | None, arr_axis_size: int) -> int:
            if sizeval is None:
                return arr_axis_size
            if ia.is_single_float(sizeval):
                return max(int(arr_axis_size * sizeval), 1)
            if ia.is_single_integer(sizeval):
                return int(sizeval)
            raise ValueError(f"Expected float or int, got {type(sizeval)}.")

        if size is None:
            size_seq: Sequence[float | int | None] = [None, None]
        elif not ia.is_iterable(size):
            size_seq = [cast(float | int, size), cast(float | int, size)]
        else:
            size_seq = cast(Sequence[float | int | None], size)

        height = _handle_sizeval(size_seq[0], self.arr.shape[0])
        width = _handle_sizeval(size_seq[1], self.arr.shape[1])
        image = np.zeros((height, width, 3), dtype=np.uint8)

        return self.draw_on_image(
            image,
            alpha=1.0,
            resize="segmentation_map",
            colors=colors,
            draw_background=True,
        )

    def draw_on_image(
        self,
        image: _RGBImage,
        alpha: float = 0.75,
        resize: Literal["segmentation_map", "image"] = "segmentation_map",
        colors: Sequence[_Color] | None = None,
        draw_background: bool = False,
        background_class_id: int = 0,
        background_threshold: object | None = None,
    ) -> list[_RGBImage]:
        """Draw the segmentation map as an overlay over an image.

        Parameters
        ----------
        image : (H,W,3) ndarray
            Image onto which to draw the segmentation map. Expected dtype
            is ``uint8``.

        alpha : float, optional
            Alpha/opacity value to use for the mixing of image and
            segmentation map. Larger values mean that the segmentation map
            will be more visible and the image less visible.

        resize : {'segmentation_map', 'image'}, optional
            In case of size differences between the image and segmentation
            map, either the image or the segmentation map can be resized.
            This parameter controls which of the two will be resized to the
            other's size.

        colors : None or list of tuple of int, optional
            Colors to use. One for each class to draw.
            If ``None``, then default colors will be used.

        draw_background : bool, optional
            If ``True``, the background will be drawn like any other class.
            If ``False``, the background will not be drawn, i.e. the respective
            background pixels will be identical with the image's RGB color at
            the corresponding spatial location and no color overlay will be
            applied.

        background_class_id : int, optional
            Class id to interpret as the background class.
            See `draw_background`.

        background_threshold : None, optional
            Deprecated.
            This parameter is ignored.

        Returns
        -------
        list of (H,W,3) ndarray
            Rendered overlays as ``uint8`` arrays.
            Always a **list** containing one RGB image per segmentation map
            array channel.

        """
        if background_threshold is not None:
            ia.warn_deprecated(
                "The argument `background_threshold` is deprecated and "
                "ignored. Please don't use it anymore."
            )

        assert image.ndim == 3, (
            f"Expected to draw on 3-dimensional image, got image with {image.ndim} dimensions."
        )
        assert image.shape[2] == 3, (
            f"Expected to draw on RGB image, got image with {image.shape[2]} channels instead."
        )
        assert image.dtype.name == "uint8", (
            f"Expected to get image with dtype uint8, got dtype {image.dtype.name}."
        )
        assert 0 - 1e-8 <= alpha <= 1.0 + 1e-8, (
            f"Expected 'alpha' to be in interval [0.0, 1.0], got {alpha:.4f}."
        )
        assert resize in ["segmentation_map", "image"], (
            f'Expected \'resize\' to be "segmentation_map" or "image", got {resize}.'
        )

        colors = colors if colors is not None else SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS

        if resize == "image":
            image = ia.imresize_single_image(image, self.arr.shape[0:2], interpolation="cubic")

        segmaps_drawn = []
        arr_channelwise = np.dsplit(self.arr, self.arr.shape[2])
        for arr in arr_channelwise:
            arr = arr[:, :, 0]

            nb_classes = 1 + np.max(arr)
            segmap_drawn = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
            assert nb_classes <= len(colors), (
                f"Can't draw all {nb_classes} classes as it would exceed the maximum "
                f"number of {len(colors)} available colors."
            )

            ids_in_map = np.unique(arr)
            for c, color in zip(range(int(nb_classes)), colors, strict=False):
                if c in ids_in_map:
                    class_mask = arr == c
                    segmap_drawn[class_mask] = color

            segmap_drawn = ia.imresize_single_image(
                segmap_drawn, image.shape[0:2], interpolation="nearest"
            )

            segmap_on_image = blend_alpha(segmap_drawn, image, alpha)

            if draw_background:
                mix = segmap_on_image
            else:
                foreground_mask = ia.imresize_single_image(
                    (arr != background_class_id),
                    image.shape[0:2],
                    interpolation="nearest",
                )
                # without this, the merge below does nothing
                foreground_mask = np.atleast_3d(foreground_mask)

                mix = (~foreground_mask) * image + foreground_mask * segmap_on_image
            segmaps_drawn.append(mix)
        return segmaps_drawn

    def pad(
        self,
        top: int = 0,
        right: int = 0,
        bottom: int = 0,
        left: int = 0,
        mode: str = "constant",
        cval: int | float = 0,
    ) -> SegmentationMapsOnImage:
        """Pad the segmentation maps at their top/right/bottom/left side.

        Parameters
        ----------
        top : int, optional
            Amount of pixels to add at the top side of the segmentation map.
            Must be ``0`` or greater.

        right : int, optional
            Amount of pixels to add at the right side of the segmentation map.
            Must be ``0`` or greater.

        bottom : int, optional
            Amount of pixels to add at the bottom side of the segmentation map.
            Must be ``0`` or greater.

        left : int, optional
            Amount of pixels to add at the left side of the segmentation map.
            Must be ``0`` or greater.

        mode : str, optional
            Padding mode to use. See `pad()` for details.

        cval : number, optional
            Value to use for padding if `mode` is ``constant``.
            See `pad()` for details.

        Returns
        -------
        imgaug2.augmentables.segmaps.SegmentationMapsOnImage
            Padded segmentation map with height ``H'=H+top+bottom`` and
            width ``W'=W+left+right``.

        """
        arr_padded = pad(
            self.arr,
            top=top,
            right=right,
            bottom=bottom,
            left=left,
            mode=mode,
            cval=cval,
        )
        return self.deepcopy(arr=arr_padded)

    def pad_to_aspect_ratio(
        self,
        aspect_ratio: float,
        mode: str = "constant",
        cval: int | float = 0,
        return_pad_amounts: bool = False,
    ) -> SegmentationMapsOnImage | tuple[SegmentationMapsOnImage, tuple[int, int, int, int]]:
        """Pad the segmentation maps until they match a target aspect ratio.

        Depending on which dimension is smaller (height or width), only the
        corresponding sides (left/right or top/bottom) will be padded. In
        each case, both of the sides will be padded equally.

        Parameters
        ----------
        aspect_ratio : float
            Target aspect ratio, given as width/height. E.g. ``2.0`` denotes
            the image having twice as much width as height.

        mode : str, optional
            Padding mode to use.
            See `pad()` for details.

        cval : number, optional
            Value to use for padding if `mode` is ``constant``.
            See `pad()` for details.

        return_pad_amounts : bool, optional
            If ``False``, then only the padded instance will be returned.
            If ``True``, a tuple with two entries will be returned, where
            the first entry is the padded instance and the second entry are
            the amounts by which each array side was padded. These amounts are
            again a tuple of the form ``(top, right, bottom, left)``, with
            each value being an integer.

        Returns
        -------
        imgaug2.augmentables.segmaps.SegmentationMapsOnImage
            Padded segmentation map as `SegmentationMapsOnImage`
            instance.

        tuple of int
            Amounts by which the instance's array was padded on each side,
            given as a tuple ``(top, right, bottom, left)``.
            This tuple is only returned if `return_pad_amounts` was set to
            ``True``.

        """
        arr_padded, pad_amounts = pad_to_aspect_ratio(
            self.arr,
            aspect_ratio=aspect_ratio,
            mode=mode,
            cval=cval,
            return_pad_amounts=True,
        )
        segmap = self.deepcopy(arr=arr_padded)
        if return_pad_amounts:
            if isinstance(pad_amounts, np.ndarray):
                pad_amounts_vals = tuple(int(v) for v in pad_amounts.tolist())
            else:
                pad_amounts_vals = tuple(int(v) for v in pad_amounts)
            assert len(pad_amounts_vals) == 4
            pad_amounts_4 = (
                pad_amounts_vals[0],
                pad_amounts_vals[1],
                pad_amounts_vals[2],
                pad_amounts_vals[3],
            )
            return segmap, pad_amounts_4
        return segmap

    @ia.deprecated(
        alt_func="SegmentationMapsOnImage.resize()",
        comment="resize() has the exactly same interface.",
    )
    def scale(self, *args: object, **kwargs: object) -> SegmentationMapsOnImage:
        """Resize the seg.map(s) array given a target size and interpolation."""
        return self.resize(*args, **kwargs)

    def resize(
        self, sizes: object, interpolation: str | int | None = "nearest"
    ) -> SegmentationMapsOnImage:
        """Resize the seg.map(s) array given a target size and interpolation.

        Parameters
        ----------
        sizes : float or iterable of int or iterable of float
            New size of the array in ``(height, width)``.
            See `imresize_single_image()` for details.

        interpolation : None or str or int, optional
            The interpolation to use during resize.
            Nearest neighbour interpolation (``"nearest"``) is almost always
            the best choice.
            See `imresize_single_image()` for details.

        Returns
        -------
        imgaug2.augmentables.segmaps.SegmentationMapsOnImage
            Resized segmentation map object.

        """
        if interpolation not in [None, "nearest"]:
            assert self.arr.dtype.kind not in ["i", "u"], (
                "Non-nearest interpolation is not allowed for integer segmentation maps. "
                "Use interpolation='nearest'."
            )
        arr_resized = ia.imresize_single_image(self.arr, sizes, interpolation=interpolation)
        return self.deepcopy(arr_resized)

    # TODO how best to handle changes to _input_was due to changed 'arr'?
    def copy(
        self, arr: _SegmapArrayInt32 | None = None, shape: _ImageShape | None = None
    ) -> SegmentationMapsOnImage:
        """Create a shallow copy of the segmentation map object.

        Parameters
        ----------
        arr : None or (H,W) ndarray or (H,W,C) ndarray, optional
            Optionally the `arr` attribute to use for the new segmentation map
            instance. Will be copied from the old instance if not provided.
            See
            `__init__()`
            for details.

        shape : None or tuple of int, optional
            Optionally the shape attribute to use for the the new segmentation
            map instance. Will be copied from the old instance if not provided.
            See
            `__init__()`
            for details.

        Returns
        -------
        imgaug2.augmentables.segmaps.SegmentationMapsOnImage
            Shallow copy.

        """
        segmap = SegmentationMapsOnImage(
            self.arr if arr is None else arr,
            shape=self.shape if shape is None else shape,
        )
        segmap._input_was = self._input_was
        return segmap

    def deepcopy(
        self, arr: _SegmapArrayInt32 | None = None, shape: _ImageShape | None = None
    ) -> SegmentationMapsOnImage:
        """Create a deep copy of the segmentation map object.

        Parameters
        ----------
        arr : None or (H,W) ndarray or (H,W,C) ndarray, optional
            Optionally the `arr` attribute to use for the new segmentation map
            instance. Will be copied from the old instance if not provided.
            See
            `__init__()`
            for details.

        shape : None or tuple of int, optional
            Optionally the shape attribute to use for the the new segmentation
            map instance. Will be copied from the old instance if not provided.
            See
            `__init__()`
            for details.

        Returns
        -------
        imgaug2.augmentables.segmaps.SegmentationMapsOnImage
            Deep copy.

        """
        segmap = SegmentationMapsOnImage(
            np.copy(self.arr if arr is None else arr),
            shape=self.shape if shape is None else shape,
        )
        segmap._input_was = self._input_was
        return segmap
