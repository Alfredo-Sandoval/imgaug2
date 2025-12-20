"""JPEG compression artifacts for MLX-compatible pipelines.

This module provides JPEG compression simulation for use in MLX-based augmentation
pipelines. Note that JPEG encoding/decoding is CPU-bound and requires roundtripping
through host memory, as there is no native MLX implementation of JPEG codecs.

The functions preserve input type (NumPy or MLX) but internally use OpenCV for
JPEG operations on the CPU.

Examples
--------
>>> import numpy as np  # xdoctest: +SKIP
>>> from imgaug2.mlx.compression import jpeg_compression  # xdoctest: +SKIP
>>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)  # xdoctest: +SKIP
>>> compressed = jpeg_compression(img, quality=50)  # xdoctest: +SKIP
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, overload

import cv2
import numpy as np
from numpy.typing import NDArray

from ._core import is_mlx_array, require, to_mlx, to_numpy

if TYPE_CHECKING:
    import mlx.core as _mx

    MlxArray: TypeAlias = _mx.array
else:
    MlxArray: TypeAlias = object

NumpyArray: TypeAlias = NDArray[np.generic]


def _to_uint8_for_jpeg(image_np: np.ndarray) -> tuple[np.ndarray, bool]:
    """Convert array to uint8 for JPEG encoding.

    Internal helper that converts floating-point or integer arrays to uint8 format
    required by JPEG encoder. Detects [0,1] normalized floats for proper restoration.

    Returns
    -------
    tuple of (np.ndarray, bool)
        Converted uint8 image and flag indicating if input was [0,1] normalized float.
    """
    if image_np.dtype == np.uint8:
        return image_np, False

    if image_np.dtype.kind == "f":
        # Heuristic: treat [0,1] floats as normalized.
        max_val = float(np.max(image_np)) if image_np.size > 0 else 0.0
        was_01 = max_val <= 1.0 + 1e-6
        if was_01:
            u8 = np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8)
            return u8, True

        u8 = np.clip(image_np, 0.0, 255.0).astype(np.uint8)
        return u8, False

    # Int types
    u8 = np.clip(image_np, 0, 255).astype(np.uint8)
    return u8, False


def _restore_from_jpeg(
    decoded_u8: np.ndarray,
    *,
    original_dtype: np.dtype,
    was_01_float: bool,
) -> np.ndarray:
    if original_dtype == np.uint8:
        return decoded_u8

    if original_dtype.kind == "f":
        out = decoded_u8.astype(np.float32)
        if was_01_float:
            out = out / 255.0
        return out.astype(original_dtype, copy=False)

    return decoded_u8.astype(original_dtype, copy=False)


@overload
def jpeg_compression(image: NumpyArray, quality: int) -> NumpyArray: ...


@overload
def jpeg_compression(image: MlxArray, quality: int) -> MlxArray: ...


def jpeg_compression(image: object, quality: int) -> object:
    """Apply JPEG compression artifacts via encode/decode cycle.

    This function simulates JPEG compression artifacts by encoding the image to
    JPEG format and immediately decoding it. The operation is CPU-bound and
    roundtrips through host memory even for MLX inputs.

    Parameters
    ----------
    image : object
        Input image as NumPy array or MLX array. Supported shapes are:
        - (H, W) : Single-channel grayscale image
        - (H, W, C) : Multi-channel image where C in {1, 3}
        - (N, H, W, C) : Batch of multi-channel images where C in {1, 3}
    quality : int
        JPEG quality parameter in range [1, 100]. Higher values produce better
        quality with less compression. Typical values: 50-95.

    Returns
    -------
    object
        Image with JPEG compression artifacts. Returns same type as input
        (NumPy or MLX).

    Raises
    ------
    ImportError
        If OpenCV is not installed.
    RuntimeError
        If OpenCV encoding/decoding fails.
    ValueError
        If quality is not in [1, 100], if shape is not 2D/3D/4D, or if
        number of channels is not 1 or 3.

    Notes
    -----
    - This operation requires CPU processing and cannot be fully accelerated.
    - Input dtypes are preserved: uint8 stays uint8, float32 stays float32.
    - Floating-point images in [0, 1] range are automatically detected and
      scaled appropriately.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    >>> compressed = jpeg_compression(img, quality=30)  # Heavy compression
    >>> high_quality = jpeg_compression(img, quality=95)  # Light compression
    """
    is_input_mlx = is_mlx_array(image)
    if is_input_mlx:
        require()

    q = int(quality)
    if q < 1 or q > 100:
        raise ValueError(f"quality must be in [1,100], got {quality}")

    image_np = to_numpy(image)
    original_dtype = image_np.dtype

    if image_np.size == 0:
        return image

    squeezed_batch = False
    squeezed_channel = False

    if image_np.ndim == 2:
        image_np = image_np[:, :, None]
        squeezed_channel = True

    if image_np.ndim == 3:
        image_np = image_np[None, ...]
        squeezed_batch = True
    elif image_np.ndim != 4:
        raise ValueError(
            "jpeg_compression expects (H,W), (H,W,C), or (N,H,W,C), "
            f"got shape {tuple(image_np.shape)}."
        )

    n, h, w, c = image_np.shape
    if c not in (1, 3):
        raise ValueError(f"jpeg_compression supports C=1 or C=3, got C={c}.")

    out_list: list[np.ndarray] = []
    for i in range(n):
        img = image_np[i]
        img_u8, was_01_float = _to_uint8_for_jpeg(img)

        if c == 1:
            img_u8_enc = img_u8[:, :, 0]
        else:
            img_u8_enc = img_u8

        ok, enc = cv2.imencode(".jpg", img_u8_enc, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("cv2.imencode('.jpg', ...) failed")

        dec = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
        if dec is None:
            raise RuntimeError("cv2.imdecode(...) failed")

        if c == 1 and dec.ndim == 2:
            dec = dec[:, :, None]
        elif c == 3 and dec.ndim == 2:
            # Some OpenCV builds can decode grayscale when input is degenerate.
            dec = np.repeat(dec[:, :, None], 3, axis=2)

        dec = dec.reshape(h, w, c)
        restored = _restore_from_jpeg(dec, original_dtype=original_dtype, was_01_float=was_01_float)
        out_list.append(restored)

    out_np = np.stack(out_list, axis=0)

    if squeezed_batch:
        out_np = out_np[0]
    if squeezed_channel:
        out_np = out_np[:, :, 0]

    if is_input_mlx:
        return to_mlx(out_np)
    return out_np


__all__ = ["jpeg_compression"]
