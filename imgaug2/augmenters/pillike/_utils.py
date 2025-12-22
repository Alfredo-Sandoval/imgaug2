from __future__ import annotations

from typing import cast

from imgaug2.augmenters._typing import Array


def _ensure_valid_shape(image: Array, func_name: str) -> tuple[Array, bool]:
    is_hw1 = image.ndim == 3 and image.shape[-1] == 1
    if is_hw1:
        image = image[:, :, 0]
    assert image.ndim == 2 or (image.ndim == 3 and image.shape[-1] in [3, 4]), (
        f"Can apply {func_name} only to images of "
        "shape (H, W) or (H, W, 1) or (H, W, 3) or (H, W, 4). "
        f"Got shape {image.shape}."
    )
    return image, is_hw1


def _maybe_mlx(image: Array, func_name: str, *args: object, **kwargs: object) -> Array | None:
    from imgaug2.mlx._core import is_mlx_array

    if is_mlx_array(image):
        from imgaug2.mlx import pillike as mlx_pillike

        func = getattr(mlx_pillike, func_name)
        return cast(Array, func(image, *args, **kwargs))

    return None
