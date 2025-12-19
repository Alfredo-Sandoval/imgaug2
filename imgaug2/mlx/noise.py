"""
Noise augmentations for the MLX backend.

This module provides GPU-accelerated noise operations using Apple's MLX framework.
All functions accept both NumPy arrays and MLX arrays, returning the same type
as input.

Shapes
------
All functions support:
- ``(H, W)`` — grayscale (where applicable)
- ``(H, W, C)`` — single image with channels
- ``(N, H, W, C)`` — batch of images

Dtype Handling
--------------
- Input uint8 images are processed in float32 and returned as uint8 with
  values clipped to [0, 255].
- Float inputs are returned as float with the original dtype preserved.

Hybrid Operations
-----------------
Functions like ``cutout``, ``random_erasing``, ``shot_noise``, ``pixel_shuffle``,
and ``spatter`` may use NumPy for certain random sampling or array manipulations
before final processing on MLX. All other functions run fully on the MLX device.

See Also
--------
imgaug2.augmenters.arithmetic : High-level augmenter classes using these operations.
"""

from __future__ import annotations

import numpy as np

from ._core import ensure_float32, is_mlx_array, mx, require, to_mlx, to_numpy


def _validate_probability(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_image_ndim(img_mlx: mx.array, func_name: str) -> None:
    if img_mlx.ndim not in (2, 3, 4):
        raise ValueError(
            f"{func_name} expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mlx.shape)}."
        )


def _to_nhwc(img_mlx: mx.array, func_name: str) -> tuple[mx.array, bool, bool]:
    squeezed_batch_axis = False
    squeezed_channel_axis = False
    if img_mlx.ndim == 2:
        img_mlx = img_mlx[None, :, :, None]
        squeezed_batch_axis = True
        squeezed_channel_axis = True
    elif img_mlx.ndim == 3:
        img_mlx = img_mlx[None, :, :, :]
        squeezed_batch_axis = True
    elif img_mlx.ndim != 4:
        raise ValueError(
            f"{func_name} expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mlx.shape)}."
        )
    return img_mlx, squeezed_batch_axis, squeezed_channel_axis


def additive_gaussian_noise(image: object, scale: float, seed: int | None = None) -> object:
    """
    Add Gaussian noise to an image.

    Samples noise from a normal distribution with mean 0 and standard deviation
    specified by `scale`, then adds it to the input image.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    scale : float
        Standard deviation of the Gaussian noise. Higher values produce stronger noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Image with additive Gaussian noise. Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If block_size is invalid or input shape is invalid.
    ValueError
        If std/gauss_sigma/intensity/mode are invalid or input shape is invalid.
    ValueError
        If scale <= 0 or input shape is invalid.
    ValueError
        If color_shift/intensity are invalid or input shape is invalid.
    ValueError
        If p/scale/ratio are invalid or input shape is invalid.
    ValueError
        If num_holes or hole size is invalid, or input shape is invalid.
    ValueError
        If ratio/grid_size are invalid or input shape is invalid.
    ValueError
        If scale < 0 or input shape is invalid.
    ValueError
        If scale < 0 or input shape is invalid.
    """
    require()

    if scale < 0:
        raise ValueError(f"scale must be >= 0, got {scale}")

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))
    if img_mlx.size == 0:
        return image
    _validate_image_ndim(img_mlx, "additive_gaussian_noise")

    if seed is not None:
        mx.random.seed(seed)

    noise = mx.random.normal(shape=img_mlx.shape, scale=scale)
    result = img_mlx + noise

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def dropout(image: object, p: float, seed: int | None = None) -> object:
    """
    Apply per-pixel dropout to an image.

    Randomly sets pixels to zero with probability `p`. The dropout mask is shared
    across channels, meaning all channels of a pixel are dropped together.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    p : float
        Probability of dropping each pixel. Must be in [0, 1].
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Image with dropout applied. Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If p is not in [0, 1] or input shape is invalid.
    """
    require()

    _validate_probability("p", p)

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))
    if img_mlx.size == 0:
        return image
    _validate_image_ndim(img_mlx, "dropout")

    if seed is not None:
        mx.random.seed(seed)

    if img_mlx.ndim == 2:
        mask = (mx.random.uniform(shape=img_mlx.shape) > p).astype(mx.float32)
    else:
        mask = mx.random.uniform(shape=img_mlx.shape[:-1]) > p
        mask = mask[..., None].astype(mx.float32)

    result = img_mlx * mask

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return result_np.astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def salt_and_pepper(
    image: object,
    p: float,
    salt_ratio: float = 0.5,
    seed: int | None = None,
) -> object:
    """
    Apply salt and pepper noise to an image.

    Randomly replaces pixels with either white (salt) or black (pepper) values.
    The ratio between salt and pepper can be controlled.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    p : float
        Total probability of applying noise to each pixel. Must be in [0, 1].
    salt_ratio : float, default 0.5
        Proportion of salt vs pepper. 0.5 means equal amounts, 1.0 means all salt.
        Must be in [0, 1].
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Image with salt and pepper noise. Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    ValueError
        If p is not in [0, 1], salt_ratio is not in [0, 1], or input shape is invalid.
    """
    require()

    _validate_probability("p", p)
    _validate_probability("salt_ratio", salt_ratio)

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))
    if img_mlx.size == 0:
        return image
    _validate_image_ndim(img_mlx, "salt_and_pepper")

    if seed is not None:
        mx.random.seed(seed)

    r = mx.random.uniform(shape=img_mlx.shape)
    p_salt = p * salt_ratio

    salt_mask = r < p_salt
    pepper_mask = (r >= p_salt) & (r < p)

    result = mx.where(salt_mask, 255.0, img_mlx)
    result = mx.where(pepper_mask, 0.0, result)

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def coarse_dropout(
    image: object,
    p: float,
    size_px: int | tuple[int, int] | None = None,
    per_channel: bool = False,
    seed: int | None = None,
) -> object:
    """
    Apply coarse dropout (Cutout-like) to an image.

    Generates a low-resolution dropout mask and upsamples it, creating large
    rectangular dropout regions instead of individual pixels.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    p : float
        Probability of dropping each low-resolution grid cell. Must be in [0, 1].
    size_px : int or tuple of int, optional
        Size of each dropout region in pixels. If int, uses square regions.
        If tuple, ``(height, width)``. If None, applies pixel-level dropout.
    per_channel : bool, default False
        If True, generate independent dropout masks per channel.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Image with coarse dropout applied. Same type as input (NumPy or MLX).

    Raises
    ------
    ValueError
        If p/size_px are invalid or image shape is invalid.
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    _validate_probability("p", p)

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))
    if img_mlx.size == 0:
        return image
    _validate_image_ndim(img_mlx, "coarse_dropout")

    if seed is not None:
        mx.random.seed(seed)

    squeezed_batch_axis = False
    squeezed_channel_axis = False
    if img_mlx.ndim == 2:
        img_nhwc = img_mlx[None, :, :, None]
        squeezed_batch_axis = True
        squeezed_channel_axis = True
    elif img_mlx.ndim == 3:
        img_nhwc = img_mlx[None, :, :, :]
        squeezed_batch_axis = True
    elif img_mlx.ndim == 4:
        img_nhwc = img_mlx
    else:
        raise ValueError(
            "coarse_dropout expects (H, W), (H, W, C), or (N, H, W, C), "
            f"got shape {tuple(img_mlx.shape)}."
        )

    n, h, w, c = img_nhwc.shape
    if size_px is None:
        h_low, w_low = h, w
    elif isinstance(size_px, int):
        if size_px <= 0:
            raise ValueError(f"size_px must be > 0, got {size_px}")
        h_low = max(1, h // size_px)
        w_low = max(1, w // size_px)
    else:
        if size_px[0] <= 0 or size_px[1] <= 0:
            raise ValueError(f"size_px must be > 0, got {size_px}")
        h_low = max(1, h // size_px[0])
        w_low = max(1, w // size_px[1])

    mask_c = c if per_channel else 1
    mask_shape = (n, h_low, w_low, mask_c)

    mask_low = (mx.random.uniform(shape=mask_shape) > p).astype(mx.float32)

    if (h_low, w_low) != (h, w):
        from imgaug2.mlx.geometry import resize

        mask = resize(mask_low, (h, w), order=0, mode="edge")
    else:
        mask = mask_low

    result = img_nhwc * mask

    if squeezed_batch_axis:
        result = result[0]
    if squeezed_channel_axis:
        result = result[:, :, 0]

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return result_np.astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def multiplicative_noise(
    image: object,
    scale: float = 0.1,
    per_channel: bool = False,
    seed: int | None = None,
) -> object:
    """
    Apply multiplicative Gaussian noise to an image.

    Multiplies the image by (1 + noise), where noise is sampled from a Gaussian
    distribution with mean 0 and standard deviation `scale`.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    scale : float, default 0.1
        Standard deviation of the multiplicative noise.
    per_channel : bool, default False
        If True, generate independent noise per channel.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Image with multiplicative noise applied. Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    if scale < 0:
        raise ValueError(f"scale must be >= 0, got {scale}")

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))
    if img_mlx.size == 0:
        return image
    _validate_image_ndim(img_mlx, "multiplicative_noise")

    if seed is not None:
        mx.random.seed(seed)

    if per_channel:
        noise_shape = img_mlx.shape
    elif img_mlx.ndim == 2:
        noise_shape = img_mlx.shape
    else:
        noise_shape = img_mlx.shape[:-1]

    noise = mx.random.normal(shape=noise_shape, scale=scale)
    if not per_channel and img_mlx.ndim >= 3:
        noise = noise[..., None]

    result = img_mlx * (1.0 + noise)

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def grid_dropout(
    image: object,
    ratio: float = 0.5,
    grid_size: tuple[int, int] = (4, 4),
    seed: int | None = None,
) -> object:
    """
    Apply grid-based dropout to an image.

    Divides the image into a grid and randomly drops entire grid cells,
    creating a checkerboard-like dropout pattern.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    ratio : float, default 0.5
        Probability of dropping each grid cell. Must be in [0, 1].
    grid_size : tuple of int, default (4, 4)
        Grid dimensions as ``(rows, cols)``.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Image with grid dropout applied. Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    _validate_probability("ratio", ratio)

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))
    if img_mlx.size == 0:
        return image
    img_nhwc, squeezed_batch, squeezed_channel = _to_nhwc(img_mlx, "grid_dropout")

    if seed is not None:
        mx.random.seed(seed)

    n, h, w, _c = img_nhwc.shape
    grid_h, grid_w = grid_size
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"grid_size must be > 0, got {grid_size}")

    mask_low = (mx.random.uniform(shape=(n, grid_h, grid_w, 1)) > ratio).astype(mx.float32)

    from .geometry import resize

    mask = resize(mask_low, (h, w), order=0, mode="edge")

    result = img_nhwc * mask

    if squeezed_batch:
        result = result[0]
    if squeezed_channel:
        result = result[:, :, 0]

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return result_np.astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def cutout(
    image: object,
    num_holes: int = 1,
    hole_height: int = 16,
    hole_width: int = 16,
    fill_value: float = 0.0,
    seed: int | None = None,
) -> object:
    """
    Cut out rectangular regions from an image.

    Randomly places rectangular holes in the image, filling them with a
    specified value. This is a simple form of occlusion augmentation.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    num_holes : int, default 1
        Number of rectangular holes to cut out.
    hole_height : int, default 16
        Height of each rectangular hole in pixels.
    hole_width : int, default 16
        Width of each rectangular hole in pixels.
    fill_value : float, default 0.0
        Value to fill the holes with.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Image with rectangular holes cut out. Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    if num_holes < 0:
        raise ValueError(f"num_holes must be >= 0, got {num_holes}")
    if num_holes == 0:
        return image
    if hole_height <= 0 or hole_width <= 0:
        raise ValueError(
            f"hole_height and hole_width must be > 0, got {hole_height}x{hole_width}"
        )

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))
    if img_mlx.size == 0:
        return image
    rng = np.random.default_rng(seed)

    img_nhwc, squeezed_batch, squeezed_channel = _to_nhwc(img_mlx, "cutout")

    n, h, w, _c = img_nhwc.shape

    y_coords = mx.arange(h, dtype=mx.int32)[:, None]
    x_coords = mx.arange(w, dtype=mx.int32)[None, :]

    mask = mx.ones((h, w), dtype=mx.float32)

    for _ in range(num_holes):
        y_center = int(rng.integers(0, h))
        x_center = int(rng.integers(0, w))

        y1 = max(0, y_center - hole_height // 2)
        y2 = min(h, y_center + hole_height // 2)
        x1 = max(0, x_center - hole_width // 2)
        x2 = min(w, x_center + hole_width // 2)

        hole_mask = (y_coords >= y1) & (y_coords < y2) & (x_coords >= x1) & (x_coords < x2)
        mask = mx.where(hole_mask, 0.0, mask)

    mask = mask[None, :, :, None]
    result = img_nhwc * mask + fill_value * (1.0 - mask)

    if squeezed_batch:
        result = result[0]
    if squeezed_channel:
        result = result[:, :, 0]

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def random_erasing(
    image: object,
    p: float = 0.5,
    scale: tuple[float, float] = (0.02, 0.33),
    ratio: tuple[float, float] = (0.3, 3.3),
    fill_value: float = 0.0,
    seed: int | None = None,
) -> object:
    """
    Apply random erasing data augmentation.

    Randomly erases a rectangular region with aspect ratio and size sampled
    from specified ranges. This is the PyTorch-style random erasing augmentation.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array.
    p : float, default 0.5
        Probability of applying erasing to the image.
    scale : tuple of float, default (0.02, 0.33)
        Range of proportion of erased area as ``(min, max)`` relative to image area.
    ratio : tuple of float, default (0.3, 3.3)
        Range of aspect ratio of erased area as ``(min, max)``.
    fill_value : float, default 0.0
        Value to fill the erased region with.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Image with random erasing applied, or unchanged if not triggered.
        Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    Notes
    -----
    If no valid erasing region is found after 10 attempts, the original image
    is returned unchanged.
    """
    require()

    _validate_probability("p", p)
    if len(scale) != 2 or len(ratio) != 2:
        raise ValueError("scale and ratio must be length-2 tuples")
    if scale[0] < 0 or scale[1] < 0 or scale[0] > scale[1] or scale[1] > 1.0:
        raise ValueError(f"scale must be in [0,1] with min<=max, got {scale}")
    if ratio[0] <= 0 or ratio[1] <= 0 or ratio[0] > ratio[1]:
        raise ValueError(f"ratio must be > 0 with min<=max, got {ratio}")

    rng = np.random.default_rng(seed)

    if rng.random() > p:
        return image

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))
    if img_mlx.size == 0:
        return image

    img_nhwc, squeezed_batch, squeezed_channel = _to_nhwc(img_mlx, "random_erasing")

    n, h, w, _c = img_nhwc.shape
    area = h * w

    y_coords = mx.arange(h, dtype=mx.int32)[:, None]
    x_coords = mx.arange(w, dtype=mx.int32)[None, :]
    for _ in range(10):
        target_area = area * rng.uniform(scale[0], scale[1])
        aspect_ratio = rng.uniform(ratio[0], ratio[1])

        erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(np.sqrt(target_area / aspect_ratio)))

        if erase_h < h and erase_w < w:
            y1 = int(rng.integers(0, h - erase_h))
            x1 = int(rng.integers(0, w - erase_w))
            y2 = y1 + erase_h
            x2 = x1 + erase_w

            erase_mask = (y_coords >= y1) & (y_coords < y2) & (x_coords >= x1) & (x_coords < x2)
            mask = mx.where(erase_mask, 0.0, 1.0)[None, :, :, None]

            result = img_nhwc * mask + fill_value * (1.0 - mask)

            if squeezed_batch:
                result = result[0]
            if squeezed_channel:
                result = result[:, :, 0]

            if is_input_mlx:
                return result

            result_np = to_numpy(result)
            if original_dtype == np.uint8:
                return np.clip(result_np, 0, 255).astype(np.uint8)
            if original_dtype is not None and result_np.dtype != original_dtype:
                return result_np.astype(original_dtype)
            return result_np
    if squeezed_batch:
        img_mlx = img_nhwc[0]
    if squeezed_channel:
        img_mlx = img_mlx[:, :, 0]

    if is_input_mlx:
        return img_mlx
    result_np = to_numpy(img_mlx)
    if original_dtype == np.uint8:
        return np.clip(result_np, 0, 255).astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def iso_noise(
    image: object,
    color_shift: float = 0.05,
    intensity: float = 0.5,
    seed: int | None = None,
) -> object:
    """
    Apply camera sensor ISO noise simulation.

    Simulates the noise pattern typically seen in high-ISO photographs, which
    includes both luminance (brightness) noise and chroma (color) noise. Noise
    intensity is stronger in darker regions, mimicking real camera behavior.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array in [0, 255] range.
    color_shift : float, default 0.05
        Maximum amount of color shift for chroma noise.
    intensity : float, default 0.5
        Overall noise intensity multiplier.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Image with simulated ISO noise. Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    if color_shift < 0:
        raise ValueError(f"color_shift must be >= 0, got {color_shift}")
    if intensity < 0:
        raise ValueError(f"intensity must be >= 0, got {intensity}")

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))
    if img_mlx.size == 0:
        return image

    if seed is not None:
        mx.random.seed(seed)

    img_nhwc, squeezed_batch, squeezed_channel = _to_nhwc(img_mlx, "iso_noise")

    n, h, w, c = img_nhwc.shape

    img_normalized = img_nhwc / 255.0

    luma_noise_scale = intensity * 0.1
    luma_noise = mx.random.normal(shape=(n, h, w, 1)) * luma_noise_scale
    darkness_factor = 1.0 - mx.mean(img_normalized, axis=-1, keepdims=True)
    luma_noise = luma_noise * (0.5 + darkness_factor)

    if c >= 3 and color_shift > 0:
        chroma_noise = mx.random.normal(shape=(n, h, w, c)) * color_shift * intensity
    else:
        chroma_noise = mx.zeros_like(img_nhwc) / 255.0

    result = img_nhwc + (luma_noise * 255.0) + (chroma_noise * 255.0)
    result = mx.clip(result, 0, 255)

    if squeezed_batch:
        result = result[0]
    if squeezed_channel:
        result = result[:, :, 0]

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return result_np.astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def shot_noise(
    image: object,
    scale: float = 1.0,
    seed: int | None = None,
) -> object:
    """
    Apply shot noise (Poisson noise) to an image.

    Shot noise simulates the quantum nature of light and the randomness in
    photon arrival. The noise variance is proportional to the signal level,
    making bright regions noisier than dark regions.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array in [0, 255] range.
    scale : float, default 1.0
        Scale factor for noise intensity. Higher values increase noise strength.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Image with shot noise applied. Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    Notes
    -----
    This function uses NumPy's Poisson sampling on the CPU, then transfers
    the result to MLX for final processing.
    """
    require()

    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))
    if img_mlx.size == 0:
        return image
    _validate_image_ndim(img_mlx, "shot_noise")
    img_normalized = img_mlx / 255.0

    rng = np.random.default_rng(seed)
    img_np = to_numpy(img_normalized)
    lam = np.maximum(img_np * 255.0 / scale, 0.001)
    noisy_np = rng.poisson(lam).astype(np.float32) * scale

    result = mx.array(noisy_np)
    result = mx.clip(result, 0, 255)

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return result_np.astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def spatter(
    image: object,
    mean: float = 0.65,
    std: float = 0.3,
    gauss_sigma: float = 4.0,
    intensity: float = 0.6,
    mode: str = "rain",
    seed: int | None = None,
) -> object:
    """
    Apply spatter effect simulating rain drops or mud splashes.

    Creates blob-like spatter patterns by generating Gaussian noise, blurring it,
    and thresholding to create realistic rain or mud splatter effects.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array in [0, 255] range.
    mean : float, default 0.65
        Mean of the Gaussian noise used to generate spatter pattern.
    std : float, default 0.3
        Standard deviation of the Gaussian noise.
    gauss_sigma : float, default 4.0
        Sigma for Gaussian blur applied to create blob-like patterns.
    intensity : float, default 0.6
        Intensity of the spatter effect in [0, 1].
    mode : {"rain", "mud"}, default "rain"
        Type of spatter. "rain" creates light/white splashes, "mud" creates dark splashes.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Image with spatter effect applied. Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.
    """
    require()

    if std < 0:
        raise ValueError(f"std must be >= 0, got {std}")
    if gauss_sigma < 0:
        raise ValueError(f"gauss_sigma must be >= 0, got {gauss_sigma}")
    _validate_probability("intensity", intensity)
    if mode not in {"rain", "mud"}:
        raise ValueError(f"mode must be 'rain' or 'mud', got {mode!r}")

    from .blur import gaussian_blur

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))
    if img_mlx.size == 0:
        return image

    if seed is not None:
        mx.random.seed(seed)

    img_nhwc, squeezed_batch, squeezed_channel = _to_nhwc(img_mlx, "spatter")

    n, h, w, c = img_nhwc.shape

    noise = mx.random.normal(shape=(n, h, w, 1)) * std + mean

    noise_blurred = gaussian_blur(noise, sigma=gauss_sigma)
    if not is_mlx_array(noise_blurred):
        noise_blurred = mx.array(noise_blurred)

    threshold = mean
    spatter_mask = (noise_blurred > threshold).astype(mx.float32)

    if mode == "rain":
        rain_color = mx.array([200.0, 200.0, 220.0], dtype=mx.float32)
        if c == 1:
            rain_color = mx.array([210.0], dtype=mx.float32)
        rain_color = rain_color[:c]
        spatter_layer = mx.broadcast_to(rain_color, (n, h, w, c))
        alpha = spatter_mask * intensity * 0.7
        result = img_nhwc * (1.0 - alpha) + spatter_layer * alpha
    else:
        mud_color = mx.array([80.0, 60.0, 40.0], dtype=mx.float32)
        if c == 1:
            mud_color = mx.array([60.0], dtype=mx.float32)
        mud_color = mud_color[:c]
        spatter_layer = mx.broadcast_to(mud_color, (n, h, w, c))
        alpha = spatter_mask * intensity
        result = img_nhwc * (1.0 - alpha) + spatter_layer * alpha

    result = mx.clip(result, 0, 255)

    if squeezed_batch:
        result = result[0]
    if squeezed_channel:
        result = result[:, :, 0]

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return result_np.astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


def pixel_shuffle(
    image: object,
    block_size: int = 4,
    seed: int | None = None,
) -> object:
    """
    Shuffle pixels within local blocks.

    Divides the image into non-overlapping square blocks and randomly permutes
    the pixels within each block independently, creating a mosaic-like effect.

    Parameters
    ----------
    image : array-like
        Input image as NumPy array or MLX array in [0, 255] range.
    block_size : int, default 4
        Size of square blocks for local shuffling in pixels.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array-like
        Image with shuffled pixels within blocks. Same type as input (NumPy or MLX).

    Raises
    ------
    ImportError
        If MLX is not installed.
    RuntimeError
        If MLX is installed but not functional on this system.

    Notes
    -----
    This function uses NumPy for the shuffling operation on the CPU, then
    transfers the result to MLX.
    """
    require()

    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    is_input_mlx = is_mlx_array(image)
    original_dtype = None if is_input_mlx else to_numpy(image).dtype

    img_mlx = ensure_float32(to_mlx(image))
    if img_mlx.size == 0:
        return image
    rng = np.random.default_rng(seed)

    img_nhwc, squeezed_batch, squeezed_channel = _to_nhwc(img_mlx, "pixel_shuffle")

    n, h, w, c = img_nhwc.shape

    img_np = to_numpy(img_nhwc)

    num_blocks_h = h // block_size
    num_blocks_w = w // block_size

    for b in range(n):
        for by in range(num_blocks_h):
            for bx in range(num_blocks_w):
                y0 = by * block_size
                y1 = y0 + block_size
                x0 = bx * block_size
                x1 = x0 + block_size

                block = img_np[b, y0:y1, x0:x1, :].reshape(-1, c)
                perm = rng.permutation(len(block))
                block = block[perm]
                img_np[b, y0:y1, x0:x1, :] = block.reshape(block_size, block_size, c)

    result = mx.array(img_np)

    if squeezed_batch:
        result = result[0]
    if squeezed_channel:
        result = result[:, :, 0]

    if is_input_mlx:
        return result

    result_np = to_numpy(result)
    if original_dtype == np.uint8:
        return result_np.astype(np.uint8)
    if original_dtype is not None and result_np.dtype != original_dtype:
        return result_np.astype(original_dtype)
    return result_np


__all__ = [
    "additive_gaussian_noise",
    "coarse_dropout",
    "cutout",
    "dropout",
    "grid_dropout",
    "iso_noise",
    "multiplicative_noise",
    "pixel_shuffle",
    "random_erasing",
    "salt_and_pepper",
    "shot_noise",
    "spatter",
]
