# Getting Started

This page shows copy-paste examples of the most common imgaug2 workflows.

## Basic Example

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa

image = np.zeros((128, 128, 3), dtype=np.uint8)

aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur((0.0, 1.5)),
    iaa.Affine(rotate=(-10, 10)),
], random_order=True)

image_aug = aug(image=image)
```

## Batch Augmentation

```python
images = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(16)]

aug = iaa.SomeOf((1, 3), [
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
    iaa.Multiply((0.8, 1.2)),
    iaa.LinearContrast((0.75, 1.25)),
], random_order=True)

images_aug = aug(images=images)
```

## Deterministic Augmentation

Apply the same transform to images and annotations:

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

## Multiple Data Types

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa

image = ia.quokka(size=(256, 256))
segmap = ia.SegmentationMapsOnImage(
    np.zeros((256, 256), dtype=np.int32),
    shape=image.shape
)

aug = iaa.Affine(rotate=(-10, 10))
image_aug, segmap_aug = aug(image=image, segmentation_maps=segmap)
```

## Seeding

```python
import imgaug2 as ia

ia.seed(42)  # Set global seed for reproducibility
```

!!! tip
    Set seeds at the start of your training script, not deep in library code.
