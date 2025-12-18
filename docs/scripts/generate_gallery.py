#!/usr/bin/env python3
"""Generate documentation gallery images (before/after grids).

This script is intentionally deterministic so that docs visuals don't churn
between runs.

Run from repo root:
    python docs/scripts/generate_gallery.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data
import imgaug2.augmenters.pillike as iaa_pil

from PIL import Image


@dataclass(frozen=True, slots=True)
class GalleryItem:
    filename: str
    augmenter: iaa.Augmenter
    seed: int
    n_aug: int = 8
    cols: int = 3


def _write_grid(path: Path, images: list[np.ndarray], *, cols: int) -> None:
    grid = ia.draw_grid(images, cols=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path, optimize=True, compress_level=9)


def generate(out_dir: Path) -> list[Path]:
    base = data.quokka(size=(256, 256))

    items = [
        GalleryItem(
            filename="arithmetic_ops.png",
            augmenter=iaa.SomeOf(
                (1, 3),
                [
                    iaa.Add((-35, 35), per_channel=0.2),
                    iaa.Multiply((0.75, 1.25), per_channel=0.2),
                    iaa.AdditiveGaussianNoise(scale=(0, 0.06 * 255)),
                    iaa.CoarseDropout((0.0, 0.06), size_percent=(0.02, 0.15)),
                ],
                random_order=True,
            ),
            seed=10,
        ),
        GalleryItem(
            filename="blend_ops.png",
            augmenter=iaa.OneOf(
                [
                    iaa.BlendAlphaSimplexNoise(
                        foreground=iaa.EdgeDetect(1.0),
                        per_channel=False,
                    ),
                    iaa.BlendAlphaCheckerboard(
                        nb_rows=8,
                        nb_cols=8,
                        foreground=iaa.AddToHueAndSaturation((-40, 40)),
                    ),
                    iaa.BlendAlphaVerticalLinearGradient(
                        foreground=iaa.Multiply((0.5, 1.3)),
                    ),
                ]
            ),
            seed=17,
        ),
        GalleryItem(
            filename="color_ops.png",
            augmenter=iaa.SomeOf(
                (1, 2),
                [
                    iaa.AddToHueAndSaturation((-25, 25), per_channel=0.2),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    iaa.ChangeColorTemperature((3000, 9000)),
                ],
                random_order=True,
            ),
            seed=11,
        ),
        GalleryItem(
            filename="collections_ops.png",
            augmenter=iaa.RandAugment(n=2, m=9),
            seed=18,
            n_aug=6,
        ),
        GalleryItem(
            filename="convolutional_ops.png",
            augmenter=iaa.OneOf(
                [
                    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.8, 1.6)),
                    iaa.Emboss(alpha=(0.0, 1.0), strength=(0.0, 2.0)),
                ]
            ),
            seed=19,
        ),
        GalleryItem(
            filename="contrast_ops.png",
            augmenter=iaa.SomeOf(
                (1, 2),
                [
                    iaa.LinearContrast((0.75, 1.25), per_channel=0.2),
                    iaa.GammaContrast((0.7, 1.6), per_channel=0.2),
                    iaa.SigmoidContrast(gain=(5, 12), cutoff=(0.35, 0.65)),
                ],
                random_order=True,
            ),
            seed=12,
        ),
        GalleryItem(
            filename="edges_ops.png",
            augmenter=iaa.Canny(alpha=(0.0, 0.9)),
            seed=20,
        ),
        GalleryItem(
            filename="flip_ops.png",
            augmenter=iaa.Fliplr(0.5),
            seed=13,
        ),
        GalleryItem(
            filename="geometric_affine.png",
            augmenter=iaa.Affine(
                rotate=(-25, 25),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                mode="edge",
            ),
            seed=1,
        ),
        GalleryItem(
            filename="blur_gaussian.png",
            augmenter=iaa.GaussianBlur(sigma=(0.0, 2.0)),
            seed=2,
        ),
        GalleryItem(
            filename="noise_additive_gaussian.png",
            augmenter=iaa.AdditiveGaussianNoise(scale=(0, 0.08 * 255)),
            seed=3,
        ),
        GalleryItem(
            filename="pillike_ops.png",
            augmenter=iaa.OneOf(
                [
                    iaa_pil.EnhanceColor((0.4, 1.6)),
                    iaa_pil.EnhanceContrast((0.6, 1.5)),
                    iaa_pil.Autocontrast(cutoff=(0, 5)),
                    iaa_pil.Equalize(),
                ]
            ),
            seed=21,
        ),
        GalleryItem(
            filename="pooling_ops.png",
            augmenter=iaa.OneOf(
                [
                    iaa.AveragePooling(kernel_size=(2, 5), keep_size=True),
                    iaa.MaxPooling(kernel_size=(2, 4), keep_size=True),
                    iaa.MedianPooling(kernel_size=(2, 5), keep_size=True),
                ]
            ),
            seed=22,
        ),
        GalleryItem(
            filename="size_ops.png",
            augmenter=iaa.CropAndPad(
                px=(-40, 40),
                pad_mode="edge",
            ),
            seed=14,
        ),
        GalleryItem(
            filename="segmentation_ops.png",
            augmenter=iaa.Superpixels(p_replace=(0.2, 0.9), n_segments=(30, 200)),
            seed=15,
        ),
        GalleryItem(
            filename="weather_ops.png",
            augmenter=iaa.Fog(),
            seed=16,
        ),
        GalleryItem(
            filename="artistic_ops.png",
            augmenter=iaa.Cartoon(),
            seed=23,
            n_aug=6,
        ),
    ]

    written: list[Path] = []
    for item in items:
        ia.seed(item.seed)
        images = [base] + [item.augmenter(image=base) for _ in range(item.n_aug)]
        out_path = out_dir / item.filename
        _write_grid(out_path, images, cols=item.cols)
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/assets/gallery"),
        help="Output directory for generated PNGs.",
    )
    args = parser.parse_args()

    paths = generate(args.out_dir)
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
