"""Shared helpers for geometry augmenters."""

from __future__ import annotations

from typing import Literal

import imgaug2.imgaug as ia
import imgaug2.parameters as iap


@iap._prefetchable
def _handle_order_arg(
    order: int | list[int] | iap.StochasticParameter | Literal["ALL"],
    backend: str,
) -> iap.StochasticParameter:
    # Performance in skimage for Affine:
    #  1.0x order 0
    #  1.5x order 1
    #  3.0x order 3
    # 30.0x order 4
    # 60.0x order 5
    # measurement based on 256x256x3 batches, difference is smaller
    # on smaller images (seems to grow more like exponentially with image size)
    if order == ia.ALL:
        if backend in ["auto", "cv2"]:
            return iap.Choice([0, 1, 3])
        # dont use order=2 (bi-quadratic) because that is apparently
        # currently not recommended (and throws a warning)
        return iap.Choice([0, 1, 3, 4, 5])
    if ia.is_single_integer(order):
        assert 0 <= order <= 5, (
            f"Expected order's integer value to be in the interval [0, 5], got {order}."
        )
        if backend == "cv2":
            assert order in [0, 1, 3], (
                f"Backend \"cv2\" and order={order} was chosen, but cv2 backend "
                "can only handle order 0, 1 or 3."
            )
        return iap.Deterministic(order)
    if isinstance(order, list):
        assert all([ia.is_single_integer(val) for val in order]), (
            "Expected order list to only contain integers, "
            f"got types {str([type(val) for val in order])}."
        )
        assert all([0 <= val <= 5 for val in order]), (
            f"Expected all of order's integer values to be in range 0 <= x <= 5, got {str(order)}."
        )
        if backend == "cv2":
            assert all([val in [0, 1, 3] for val in order]), (
                f"cv2 backend can only handle order 0, 1 or 3. Got order list of {order}."
            )
        return iap.Choice(order)
    if isinstance(order, iap.StochasticParameter):
        return order
    raise Exception(
        "Expected order to be imgaug2.ALL, int, list of int or "
        f"StochasticParameter, got {type(order)}."
    )


def _handle_mode_arg(
    mode: str | list[str] | iap.StochasticParameter | Literal["ALL"],
) -> iap.StochasticParameter:
    return iap.handle_categorical_string_param(
        mode,
        "mode",
        valid_values=["constant", "edge", "symmetric", "reflect", "wrap"],
    )


__all__ = ["_handle_order_arg", "_handle_mode_arg"]
