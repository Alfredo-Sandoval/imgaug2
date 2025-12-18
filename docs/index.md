# imgaug2

**A powerful image augmentation library for machine learning.**

imgaug2 is a community-maintained fork of [aleju/imgaug](https://github.com/aleju/imgaug), focused on keeping the library usable on modern Python, NumPy, and OpenCV stacks while preserving the original API and deterministic behavior.

## What can imgaug2 do?

- **Augment Images**: Apply dozens of augmentation techniques including geometric transforms, blur, noise, color shifts, weather effects, and more.
- **Augment Annotations**: Keep keypoints, bounding boxes, polygons, line strings, heatmaps, and segmentation maps aligned with augmented images.
- **Compose Pipelines**: Build complex augmentation pipelines with `Sequential`, `SomeOf`, `Sometimes`, and random ordering.
- **Deterministic Mode**: Reproduce exact augmentations using deterministic augmenters or fixed seeds.

## Quick Example

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa

# Load an image
image = ia.quokka(size=(256, 256))

# Define augmentation pipeline
aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.Affine(rotate=(-20, 20), scale=(0.8, 1.2))
])

# Augment
image_aug = aug(image=image)
```

## Augmenting with Annotations

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa

image = ia.quokka(size=(256, 256))
bbs = ia.BoundingBoxesOnImage([
    ia.BoundingBox(x1=50, y1=50, x2=200, y2=200)
], shape=image.shape)

aug = iaa.Affine(rotate=(-25, 25))
aug_det = aug.to_deterministic()

image_aug = aug_det(image=image)
bbs_aug = aug_det(bounding_boxes=bbs)
```

## Installation

```bash
pip install imgaug2
```

Or install from source:

```bash
pip install git+https://github.com/Alfredo-Sandoval/imgaug2.git
```

## Why imgaug2?

The original `imgaug` library by Alexander Jung is no longer actively maintained. imgaug2 continues development with:

- **Python 3.10+** support
- **Modern NumPy** compatibility
- **Bug fixes** and community contributions
- **Same API** for easy migration

## Links

- [GitHub Repository](https://github.com/Alfredo-Sandoval/imgaug2)
- [Original imgaug](https://github.com/aleju/imgaug) (archived)
- [Original Documentation](https://imgaug.readthedocs.io/) (still useful for concepts)
