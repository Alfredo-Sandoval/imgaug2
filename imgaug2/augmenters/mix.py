"""
Batch-level mixing utilities (MixUp / CutMix / Mosaic).

These are intentionally NOT implemented as imgaug2 augmenters, because they operate
on batches and require label mixing. They are provided as helper functions.

All functions assume `images` is a numpy array of shape (N,H,W,C) (or (N,H,W)).
Labels may be:
  * class indices (N,)
  * one-hot vectors (N,K)
  * regression targets (N,...) (will be mixed linearly)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _default_rng(rng=None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        seed = int(rng.randint(0, 2**32 - 1))
        return np.random.default_rng(seed)
    return np.random.default_rng(int(rng))


def mixup(
    images: np.ndarray, labels: np.ndarray, alpha: float = 0.2, rng=None
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    MixUp.

    Returns: (images_mix, labels_mix, lam, perm)
    """
    g = _default_rng(rng)
    N = images.shape[0]
    perm = g.permutation(N)
    lam = 1.0 if alpha <= 0 else float(g.beta(alpha, alpha))

    images_mix = lam * images.astype(np.float32) + (1.0 - lam) * images[perm].astype(np.float32)
    if np.issubdtype(images.dtype, np.integer):
        images_mix = np.clip(images_mix, 0, np.iinfo(images.dtype).max).astype(images.dtype)
    else:
        images_mix = images_mix.astype(images.dtype)

    labels_mix = lam * labels.astype(np.float32) + (1.0 - lam) * labels[perm].astype(np.float32)
    return images_mix, labels_mix, lam, perm


def cutmix(
    images: np.ndarray, labels: np.ndarray, alpha: float = 1.0, rng=None
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    CutMix.

    Returns: (images_mix, labels_mix, lam, perm)
    """
    g = _default_rng(rng)
    N, H, W = images.shape[0], images.shape[1], images.shape[2]
    perm = g.permutation(N)

    lam = 1.0 if alpha <= 0 else float(g.beta(alpha, alpha))
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = int(g.integers(0, W))
    cy = int(g.integers(0, H))

    x1 = int(np.clip(cx - cut_w // 2, 0, W))
    x2 = int(np.clip(cx + cut_w // 2, 0, W))
    y1 = int(np.clip(cy - cut_h // 2, 0, H))
    y2 = int(np.clip(cy + cut_h // 2, 0, H))

    images_mix = images.copy()
    images_mix[:, y1:y2, x1:x2, ...] = images[perm, y1:y2, x1:x2, ...]

    area = float((x2 - x1) * (y2 - y1))
    lam_adj = 1.0 - area / float(H * W)

    labels_mix = lam_adj * labels.astype(np.float32) + (1.0 - lam_adj) * labels[perm].astype(
        np.float32
    )
    return images_mix, labels_mix, lam_adj, perm


def mosaic4(
    images: np.ndarray, rng=None, output_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mosaic with 4 images (image-only).

    Returns: (mosaic_images, indices)
      mosaic_images: (N, out_h, out_w, C)
      indices: (N,4) indices used for each mosaic.
    """
    g = _default_rng(rng)
    N, H, W = images.shape[0], images.shape[1], images.shape[2]
    C = 1 if images.ndim == 3 else images.shape[3]

    out_h, out_w = output_size if output_size is not None else (H, W)
    out = np.zeros((N, out_h, out_w, C), dtype=images.dtype)

    idx = np.stack(
        [
            g.integers(0, N, size=N),
            g.integers(0, N, size=N),
            g.integers(0, N, size=N),
            g.integers(0, N, size=N),
        ],
        axis=1,
    )

    yc = int(g.integers(int(0.3 * out_h), int(0.7 * out_h)))
    xc = int(g.integers(int(0.3 * out_w), int(0.7 * out_w)))

    for i in range(N):
        a, b, c, d = idx[i]
        ia = images[a]
        ib = images[b]
        ic = images[c]
        id_ = images[d]

        out[i, :yc, :xc, :] = (
            ia[:yc, :xc, ...]
            if ia.shape[0] >= yc and ia.shape[1] >= xc
            else np.resize(ia, (yc, xc, C))
        )
        out[i, :yc, xc:, :] = (
            ib[:yc, : out_w - xc, ...]
            if ib.shape[0] >= yc and ib.shape[1] >= out_w - xc
            else np.resize(ib, (yc, out_w - xc, C))
        )
        out[i, yc:, :xc, :] = (
            ic[: out_h - yc, :xc, ...]
            if ic.shape[0] >= out_h - yc and ic.shape[1] >= xc
            else np.resize(ic, (out_h - yc, xc, C))
        )
        out[i, yc:, xc:, :] = (
            id_[: out_h - yc, : out_w - xc, ...]
            if id_.shape[0] >= out_h - yc and id_.shape[1] >= out_w - xc
            else np.resize(id_, (out_h - yc, out_w - xc, C))
        )

    if images.ndim == 3:
        out = out[..., 0]
    return out, idx
