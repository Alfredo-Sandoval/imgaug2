"""MLX-accelerated crop and pad operations for image augmentation.

This module provides hardware-accelerated cropping and padding operations using
Apple's MLX framework. All functions support both NumPy and MLX arrays, preserving
the input array type in the output. Images are expected in HWC (Height, Width, Channel)
or NHWC (Batch, Height, Width, Channel) format.

Examples
--------
>>> import numpy as np  # xdoctest: +SKIP
>>> from imgaug2.mlx.crop import center_crop, pad, random_resized_crop  # xdoctest: +SKIP
>>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)  # xdoctest: +SKIP
>>> cropped = center_crop(img, height=50, width=50)  # xdoctest: +SKIP
>>> padded = pad(img, pad_top=10, pad_left=10, mode="reflect")  # xdoctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from ._core import is_mlx_array, mx, require, to_mlx, to_numpy

if TYPE_CHECKING:
    import mlx.core as _mx

    MlxArray: TypeAlias = _mx.array
else:
    MlxArray: TypeAlias = object


def center_crop(image: object, height: int, width: int) -> object:
    """Crop the center region of an image to the specified size.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W, C) or (N, H, W, C).
    height : int
        Target height of the cropped region.
    width : int
        Target width of the cropped region.

    Returns
    -------
    object
        Center-cropped image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If input shape is not 3D or 4D.

    Notes
    -----
    The crop is centered on the image. If the target size is larger than the
    input size, negative indices may result in unexpected behavior.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)
    if img_mlx.size == 0:
        return image
    ndim = len(img_mlx.shape)

    if ndim == 3:
        # HWC
        h, w, _c = img_mlx.shape
        y_start = (h - height) // 2
        x_start = (w - width) // 2
        result = img_mlx[y_start : y_start + height, x_start : x_start + width, :]
    elif ndim == 4:
        # NHWC
        _n, h, w, _c = img_mlx.shape
        y_start = (h - height) // 2
        x_start = (w - width) // 2
        result = img_mlx[:, y_start : y_start + height, x_start : x_start + width, :]
    else:
        raise ValueError(f"Expected 3D (HWC) or 4D (NHWC) image, got {ndim}D")

    if is_input_mlx:
        return result
    result_np = to_numpy(result)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def random_crop(
    image: object,
    height: int,
    width: int,
    seed: int | None = None,
) -> object:
    """Crop a random region from the image.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W, C) or (N, H, W, C).
    height : int
        Target height of the cropped region.
    width : int
        Target width of the cropped region.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    object
        Randomly cropped image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If input shape is not 3D or 4D.

    Notes
    -----
    For batched input (NHWC), the same random crop coordinates are applied
    to all images in the batch.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)
    if img_mlx.size == 0:
        return image
    ndim = len(img_mlx.shape)
    rng = np.random.default_rng(seed)

    if ndim == 3:
        # HWC
        h, w, _c = img_mlx.shape
        y_start = int(rng.integers(0, max(1, h - height + 1)))
        x_start = int(rng.integers(0, max(1, w - width + 1)))
        result = img_mlx[y_start : y_start + height, x_start : x_start + width, :]
    elif ndim == 4:
        # NHWC - same random crop for all images in batch
        _n, h, w, _c = img_mlx.shape
        y_start = int(rng.integers(0, max(1, h - height + 1)))
        x_start = int(rng.integers(0, max(1, w - width + 1)))
        result = img_mlx[:, y_start : y_start + height, x_start : x_start + width, :]
    else:
        raise ValueError(f"Expected 3D (HWC) or 4D (NHWC) image, got {ndim}D")

    if is_input_mlx:
        return result
    result_np = to_numpy(result)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def crop(
    image: object,
    y_start: int,
    x_start: int,
    height: int,
    width: int,
) -> object:
    """Crop a region from the image at specified coordinates.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W, C) or (N, H, W, C).
    y_start : int
        Top coordinate of the crop region (row index).
    x_start : int
        Left coordinate of the crop region (column index).
    height : int
        Height of the cropped region.
    width : int
        Width of the cropped region.

    Returns
    -------
    object
        Cropped image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If input shape is not 3D or 4D.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)
    ndim = len(img_mlx.shape)

    if ndim == 3:
        result = img_mlx[y_start : y_start + height, x_start : x_start + width, :]
    elif ndim == 4:
        result = img_mlx[:, y_start : y_start + height, x_start : x_start + width, :]
    else:
        raise ValueError(f"Expected 3D (HWC) or 4D (NHWC) image, got {ndim}D")

    if is_input_mlx:
        return result
    result_np = to_numpy(result)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def pad(
    image: object,
    pad_top: int = 0,
    pad_bottom: int = 0,
    pad_left: int = 0,
    pad_right: int = 0,
    mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"] = "constant",
    value: float = 0.0,
) -> object:
    """Pad an image with specified amounts on each side.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W, C) or (N, H, W, C).
    pad_top : int, optional
        Number of pixels to pad on the top edge. Default is 0.
    pad_bottom : int, optional
        Number of pixels to pad on the bottom edge. Default is 0.
    pad_left : int, optional
        Number of pixels to pad on the left edge. Default is 0.
    pad_right : int, optional
        Number of pixels to pad on the right edge. Default is 0.
    mode : {'constant', 'edge', 'reflect', 'symmetric', 'wrap'}, optional
        Padding mode. Default is 'constant'.
        - 'constant': Pad with constant value specified by `value`.
        - 'edge': Pad with edge pixel values.
        - 'reflect': Reflect pixels without edge duplication (cv2 BORDER_REFLECT_101).
        - 'symmetric': Reflect pixels with edge duplication.
        - 'wrap': Wrap pixels around to opposite edge.
    value : float, optional
        Constant value for 'constant' mode. Default is 0.0.

    Returns
    -------
    object
        Padded image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If input shape is not 3D or 4D, or if mode is unknown.
    """
    require()

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)
    if img_mlx.size == 0:
        return image
    ndim = len(img_mlx.shape)

    if ndim == 3:
        # HWC -> add batch dim for uniform handling
        img_mlx = img_mlx[None, ...]
        squeeze_batch = True
    elif ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError(f"Expected 3D (HWC) or 4D (NHWC) image, got {ndim}D")

    if mode == "constant":
        # Use mx.pad for constant padding
        # mx.pad expects [(before, after), ...] for each dimension
        pad_width = [(0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)]
        result = mx.pad(img_mlx, pad_width, constant_values=value)
    else:
        # For other modes, use index-based padding
        result = _pad_with_indices(img_mlx, pad_top, pad_bottom, pad_left, pad_right, mode)

    if squeeze_batch:
        result = result[0]

    if is_input_mlx:
        return result
    result_np = to_numpy(result)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def _pad_with_indices(
    img: MlxArray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    mode: str,
) -> MlxArray:
    """Pad image using index gathering for non-constant modes.

    Internal helper function for implementing edge, reflect, symmetric, and wrap
    padding modes by generating appropriate indices.
    """
    _n, h, w, _c = img.shape
    # Generate source indices
    y_indices = _generate_indices(h, pad_top, pad_bottom, mode)
    x_indices = _generate_indices(w, pad_left, pad_right, mode)

    # Gather rows then columns
    result = img[:, y_indices, :, :]
    result = result[:, :, x_indices, :]
    return result


def _generate_indices(size: int, pad_before: int, pad_after: int, mode: str) -> object:
    """Generate indices for padding along one axis.

    Internal helper that computes source indices for various padding modes.
    """
    total = size + pad_before + pad_after
    indices = mx.arange(total) - pad_before

    if mode == "edge":
        indices = mx.clip(indices, 0, size - 1)
    elif mode == "reflect":
        # Reflect without edge duplication (like cv2 BORDER_REFLECT_101)
        indices = _reflect101_indices(indices, size)
    elif mode == "symmetric":
        # Reflect with edge duplication
        indices = _symmetric_indices(indices, size)
    elif mode == "wrap":
        indices = indices % size
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

    return indices


def _reflect101_indices(indices: object, size: int) -> object:
    """Reflect indices without edge duplication (cv2 BORDER_REFLECT_101)."""
    # Period is 2*(size-1)
    if size <= 1:
        return mx.zeros_like(indices)

    period = 2 * (size - 1)
    indices = indices % period
    # Mirror: if index >= size, reflect it
    indices = mx.where(indices >= size, period - indices, indices)
    return indices


def _symmetric_indices(indices: object, size: int) -> object:
    """Reflect indices with edge duplication (symmetric padding)."""
    if size <= 1:
        return mx.zeros_like(indices)

    period = 2 * size
    indices = indices % period
    # Mirror: if index >= size, reflect it
    indices = mx.where(indices >= size, period - 1 - indices, indices)
    return indices


def pad_if_needed(
    image: object,
    min_height: int,
    min_width: int,
    mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"] = "constant",
    value: float = 0.0,
    position: Literal["center", "top_left", "random"] = "center",
    seed: int | None = None,
) -> object:
    """Conditionally pad image to ensure minimum dimensions.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W, C) or (N, H, W, C).
    min_height : int
        Minimum required height. No padding if image height >= min_height.
    min_width : int
        Minimum required width. No padding if image width >= min_width.
    mode : {'constant', 'edge', 'reflect', 'symmetric', 'wrap'}, optional
        Padding mode. See `pad` function for details. Default is 'constant'.
    value : float, optional
        Constant value for 'constant' mode. Default is 0.0.
    position : {'center', 'top_left', 'random'}, optional
        Where to position the original image within the padded result. Default is 'center'.
        - 'center': Center the image in the padded result.
        - 'top_left': Position at top-left corner.
        - 'random': Random position within padded area.
    seed : int, optional
        Random seed for 'random' position mode. Default is None.

    Returns
    -------
    object
        Padded image if dimensions were below minimum, otherwise unchanged.
        Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If input shape is not 3D or 4D, or if position mode is unknown.
    """
    require()

    img_mlx = to_mlx(image)
    ndim = len(img_mlx.shape)

    if ndim == 3:
        h, w = img_mlx.shape[:2]
    elif ndim == 4:
        h, w = img_mlx.shape[1:3]
    else:
        raise ValueError(f"Expected 3D (HWC) or 4D (NHWC) image, got {ndim}D")

    pad_h = max(0, min_height - h)
    pad_w = max(0, min_width - w)

    if pad_h == 0 and pad_w == 0:
        return image  # No padding needed

    if position == "center":
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    elif position == "top_left":
        pad_top = 0
        pad_bottom = pad_h
        pad_left = 0
        pad_right = pad_w
    elif position == "random":
        rng = np.random.default_rng(seed)
        pad_top = int(rng.integers(0, pad_h + 1)) if pad_h > 0 else 0
        pad_bottom = pad_h - pad_top
        pad_left = int(rng.integers(0, pad_w + 1)) if pad_w > 0 else 0
        pad_right = pad_w - pad_left
    else:
        raise ValueError(f"Unknown position: {position}")

    return pad(image, pad_top, pad_bottom, pad_left, pad_right, mode=mode, value=value)


def random_resized_crop(
    image: object,
    height: int,
    width: int,
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    seed: int | None = None,
) -> object:
    """Crop a random portion of image and resize to target size.

    This is the standard augmentation used in ImageNet training, implementing
    the "Inception-style" preprocessing.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are
        (H, W, C) or (N, H, W, C).
    height : int
        Target height after resizing.
    width : int
        Target width after resizing.
    scale : tuple of float, optional
        Range of proportion of image area to crop. Default is (0.08, 1.0).
    ratio : tuple of float, optional
        Range of aspect ratio (width/height) of the crop. Default is (3/4, 4/3).
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    object
        Cropped and resized image. Returns same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If input shape is not 3D or 4D.

    Notes
    -----
    The function attempts up to 10 times to find a valid random crop satisfying
    the scale and ratio constraints. If unsuccessful, it falls back to a center
    crop with the closest valid aspect ratio.
    """
    require()

    from .geometry import resize

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = to_mlx(image)
    ndim = len(img_mlx.shape)

    if ndim == 3:
        h, w = img_mlx.shape[:2]
    elif ndim == 4:
        h, w = img_mlx.shape[1:3]
    else:
        raise ValueError(f"Expected 3D (HWC) or 4D (NHWC) image, got {ndim}D")

    rng = np.random.default_rng(seed)
    area = h * w

    # Try to find valid crop parameters
    for _ in range(10):
        target_area = area * rng.uniform(scale[0], scale[1])
        log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
        aspect_ratio = np.exp(rng.uniform(log_ratio[0], log_ratio[1]))

        crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(np.sqrt(target_area / aspect_ratio)))

        if 0 < crop_w <= w and 0 < crop_h <= h:
            y_start = int(rng.integers(0, h - crop_h + 1))
            x_start = int(rng.integers(0, w - crop_w + 1))

            # Crop
            cropped = crop(img_mlx, y_start, x_start, crop_h, crop_w)
            # Resize
            result = resize(cropped, (height, width))

            if is_input_mlx:
                return result
            result_np = to_numpy(result)
            if original_dtype == np.uint8:
                return np.clip(result_np, 0, 255).astype(np.uint8)
            if original_dtype is not None and result_np.dtype != original_dtype:
                return result_np.astype(original_dtype)
            return result_np

    # Fallback to center crop + resize
    in_ratio = w / h
    if in_ratio < min(ratio):
        crop_w = w
        crop_h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        crop_h = h
        crop_w = int(round(h * max(ratio)))
    else:
        crop_w = w
        crop_h = h

    cropped = center_crop(img_mlx, crop_h, crop_w)
    result = resize(cropped, (height, width))

    if is_input_mlx:
        return result
    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


__all__ = [
    "center_crop",
    "crop",
    "pad",
    "pad_if_needed",
    "random_crop",
    "random_resized_crop",
]
