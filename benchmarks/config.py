"""Benchmark configuration (augmenters + image/batch sizes)."""

from __future__ import annotations

from collections.abc import Callable

import imgaug2.augmenters as iaa

# (batch_size, height, width, channels)
BENCHMARK_CONFIGS: list[tuple[int, int, int, int]] = [
    (1, 64, 64, 3),
    (1, 256, 256, 3),
    (16, 256, 256, 3),
    (32, 256, 256, 3),
]

# NOTE: Keep these factories side-effect free. The runner will call them many
# times and expects a fresh augmenter each time.
AUGMENTERS: dict[str, Callable[[], iaa.Augmenter]] = {
    # Fast
    "Identity": lambda: iaa.Identity(),
    "Fliplr": lambda: iaa.Fliplr(1.0),
    "Flipud": lambda: iaa.Flipud(1.0),
    "Rot90": lambda: iaa.Rot90(k=1),
    # Geometric
    "Affine_rotate": lambda: iaa.Affine(rotate=(-25, 25)),
    "Affine_scale": lambda: iaa.Affine(scale=(0.8, 1.2)),
    "Affine_all": lambda: iaa.Affine(
        rotate=(-25, 25),
        scale=(0.8, 1.2),
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        shear=(-8, 8),
    ),
    "PerspectiveTransform": lambda: iaa.PerspectiveTransform(scale=(0.01, 0.1)),
    "ElasticTransformation": lambda: iaa.ElasticTransformation(alpha=50, sigma=5),
    "PiecewiseAffine": lambda: iaa.PiecewiseAffine(scale=(0.01, 0.03)),
    # Blur
    "GaussianBlur_small": lambda: iaa.GaussianBlur(sigma=1.0),
    "GaussianBlur_large": lambda: iaa.GaussianBlur(sigma=3.0),
    "AverageBlur": lambda: iaa.AverageBlur(k=5),
    "MedianBlur": lambda: iaa.MedianBlur(k=5),
    "MotionBlur": lambda: iaa.MotionBlur(k=15),
    "BilateralBlur": lambda: iaa.BilateralBlur(d=5),
    # Noise
    "AdditiveGaussianNoise": lambda: iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
    "Dropout": lambda: iaa.Dropout(p=0.05),
    "CoarseDropout": lambda: iaa.CoarseDropout(p=0.05, size_percent=0.1),
    "SaltAndPepper": lambda: iaa.SaltAndPepper(p=0.05),
    # Color
    "Grayscale": lambda: iaa.Grayscale(alpha=1.0),
    "AddToHueAndSaturation": lambda: iaa.AddToHueAndSaturation((-20, 20)),
    "Multiply": lambda: iaa.Multiply((0.8, 1.2)),
    "LinearContrast": lambda: iaa.LinearContrast((0.75, 1.25)),
    "CLAHE": lambda: iaa.CLAHE(),
    # Segmentation (slow)
    "Superpixels": lambda: iaa.Superpixels(p_replace=0.5, n_segments=100),
    "Voronoi": lambda: iaa.Voronoi(iaa.RegularGridPointsSampler(n_rows=20, n_cols=20)),
    # Pipelines
    "Pipeline_light": lambda: iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Multiply((0.9, 1.1)),
        ]
    ),
    "Pipeline_medium": lambda: iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.Affine(rotate=(-15, 15)),
        ]
    ),
    "Pipeline_heavy": lambda: iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.SomeOf(
                (1, 3),
                [
                    iaa.GaussianBlur(sigma=(0, 1.5)),
                    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
                    iaa.Multiply((0.8, 1.2)),
                    iaa.Affine(rotate=(-20, 20)),
                ],
            ),
        ]
    ),
}
