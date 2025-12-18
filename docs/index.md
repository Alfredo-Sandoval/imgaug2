# imgaug2

**Modern image augmentation (Python 3.10+). MIT licensed forever.**

imgaug2 is a maintained continuation of [aleju/imgaug](https://github.com/aleju/imgaug) with modern Python 3.10+ support and active development while preserving API compatibility and deterministic behavior.

> **Note:** imgaug2 currently runs augmentations on the CPU.

## Features

- **130+ Augmentations** — Geometric transforms, blur, noise, color shifts, weather effects, and more
- **Multi-Augmentable** — Automatically augments keypoints, bounding boxes, polygons, heatmaps, and segmentation maps
- **Pipeline Composition** — Build complex pipelines with `Sequential`, `SomeOf`, `Sometimes`, and random ordering
- **Deterministic Mode** — Reproduce exact augmentations with deterministic augmenters or fixed seeds
- **Framework Agnostic** — Works with PyTorch, JAX, NumPy
- **Optional Dict-based API** — `Compose` interface with uniform `p=` via `imgaug2.compat`

## Quick Example

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data

# Load an image
image = data.quokka(size=(256, 256))

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
import imgaug2.data as data

image = data.quokka(size=(256, 256))
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

imgaug2 extends the original `imgaug` with:

- **Python 3.10+** — Modern language features and NumPy/OpenCV compatibility
- **Active Development** — Regular updates and bug fixes
- **API Compatibility** — Drop-in replacement for existing imgaug code
- **MIT Licensed** — Free forever, no restrictions
- **Better DX (optional)** — `imgaug2.compat` for dict-based pipelines and bbox formats

## Links

- [GitHub Repository](https://github.com/Alfredo-Sandoval/imgaug2)
- [Original imgaug](https://github.com/aleju/imgaug) (archived)
- [Original Documentation](https://imgaug.readthedocs.io/) (still useful for concepts)
