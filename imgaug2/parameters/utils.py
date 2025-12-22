from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

import imgaug2.imgaug as ia
from imgaug2.compat.markers import legacy

from .base import StochasticParameter


def force_np_float_dtype(val: NDArray) -> NDArray:
    if val.dtype.kind == "f":
        return val
    return val.astype(np.float64)


@legacy
def both_np_float_if_one_is_float(a: NDArray, b: NDArray) -> tuple[NDArray, NDArray]:
    a_f = a.dtype.type in ia.NP_FLOAT_TYPES
    b_f = b.dtype.type in ia.NP_FLOAT_TYPES
    if a_f and b_f:
        return a, b
    if a_f:
        return a, b.astype(np.float64)
    if b_f:
        return a.astype(np.float64), b
    return a.astype(np.float64), b.astype(np.float64)


@legacy
def draw_distributions_grid(
    params: Sequence[StochasticParameter],
    rows: int | None = None,
    cols: int | None = None,
    graph_sizes: tuple[int, int] = (350, 350),
    sample_sizes: Sequence[int] | None = None,
    titles: Sequence[str | None] | bool | None = None,
) -> NDArray:
    if titles is None:
        titles = [None] * len(params)
    elif titles is False:
        titles = [False] * len(params)  # type: ignore

    if sample_sizes is not None:
        images = [
            param_i.draw_distribution_graph(size=size_i, title=title_i)
            for param_i, size_i, title_i in zip(
                params, sample_sizes, titles, strict=False
            )  # type: ignore
        ]
    else:
        images = [
            param_i.draw_distribution_graph(title=title_i)
            for param_i, title_i in zip(params, titles, strict=False)  # type: ignore
        ]

    images_rs = ia.imresize_many_images(images, sizes=graph_sizes)
    grid = ia.draw_grid(images_rs, rows=rows, cols=cols)
    return grid


@legacy
def show_distributions_grid(
    params: Sequence[StochasticParameter],
    rows: int | None = None,
    cols: int | None = None,
    graph_sizes: tuple[int, int] = (350, 350),
    sample_sizes: Sequence[int] | None = None,
    titles: Sequence[str | None] | bool | None = None,
) -> None:
    ia.imshow(
        draw_distributions_grid(
            params,
            graph_sizes=graph_sizes,
            sample_sizes=sample_sizes,
            rows=rows,
            cols=cols,
            titles=titles,
        )
    )
