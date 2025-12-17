# Bounding Box Examples

Bounding boxes are commonly used in object detection tasks. imgaug2 can augment bounding boxes alongside images.

## Basic Bounding Box Augmentation

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Load image
image = ia.quokka(size=(256, 256))

# Define bounding boxes (x1, y1, x2, y2 format)
bbs = BoundingBoxesOnImage([
    BoundingBox(x1=50, y1=50, x2=200, y2=200, label="quokka"),
    BoundingBox(x1=10, y1=10, x2=50, y2=50, label="background")
], shape=image.shape)

# Create augmenter
aug = iaa.Affine(rotate=(-25, 25), scale=(0.8, 1.2))

# Augment both
image_aug, bbs_aug = aug(image=image, bounding_boxes=bbs)
```

## Deterministic Augmentation

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

image = ia.quokka(size=(256, 256))
bbs = BoundingBoxesOnImage([
    BoundingBox(x1=50, y1=50, x2=200, y2=200)
], shape=image.shape)

# Make augmenter deterministic
aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-25, 25))
])
aug_det = aug.to_deterministic()

# Apply same transform to both
image_aug = aug_det(image=image)
bbs_aug = aug_det(bounding_boxes=bbs)
```

## Handling Edge Cases

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

image = ia.quokka(size=(256, 256))
bbs = BoundingBoxesOnImage([
    BoundingBox(x1=0, y1=0, x2=50, y2=50),      # Corner box
    BoundingBox(x1=200, y1=200, x2=256, y2=256) # Opposite corner
], shape=image.shape)

aug = iaa.Affine(translate_percent={"x": 0.3})
image_aug, bbs_aug = aug(image=image, bounding_boxes=bbs)

# Remove boxes that are fully outside the image
bbs_aug = bbs_aug.remove_out_of_image_()

# Clip boxes to image boundaries
bbs_aug = bbs_aug.clip_out_of_image_()

# Remove boxes that are mostly outside (< 50% visible)
bbs_aug = bbs_aug.remove_out_of_image_fraction_(0.5)
```

## Visualizing Bounding Boxes

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

image = ia.quokka(size=(256, 256))
bbs = BoundingBoxesOnImage([
    BoundingBox(x1=50, y1=50, x2=200, y2=200, label="quokka")
], shape=image.shape)

# Draw on image
image_with_bbs = bbs.draw_on_image(image, size=2)

# Augment and draw
aug = iaa.Affine(rotate=15)
image_aug, bbs_aug = aug(image=image, bounding_boxes=bbs)
image_aug_with_bbs = bbs_aug.draw_on_image(image_aug, size=2)

# Save
import imageio
imageio.imwrite("bbs_before.jpg", image_with_bbs)
imageio.imwrite("bbs_after.jpg", image_aug_with_bbs)
```

## Bounding Box Properties

```python
from imgaug2.augmentables.bbs import BoundingBox

bb = BoundingBox(x1=10, y1=20, x2=110, y2=120, label="object")

# Properties
print(f"Width: {bb.width}, Height: {bb.height}")
print(f"Area: {bb.area}")
print(f"Center: ({bb.center_x}, {bb.center_y})")
print(f"Label: {bb.label}")

# Convert formats
print(f"x1,y1,x2,y2: {bb.x1}, {bb.y1}, {bb.x2}, {bb.y2}")

# Intersection over Union (IoU)
bb2 = BoundingBox(x1=50, y1=60, x2=150, y2=160)
iou = bb.iou(bb2)
print(f"IoU: {iou:.3f}")

# Extend/contract
bb_extended = bb.extend(all_sides=10)
bb_contracted = bb.extend(all_sides=-5)
```

## Batch Processing

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Multiple images with different numbers of boxes
images = [ia.quokka(size=(256, 256)) for _ in range(4)]
bounding_boxes = [
    BoundingBoxesOnImage([
        BoundingBox(x1=50, y1=50, x2=200, y2=200)
    ], shape=images[0].shape),
    BoundingBoxesOnImage([
        BoundingBox(x1=30, y1=30, x2=100, y2=100),
        BoundingBox(x1=150, y1=150, x2=230, y2=230)
    ], shape=images[0].shape),
    BoundingBoxesOnImage([], shape=images[0].shape),  # No boxes
    BoundingBoxesOnImage([
        BoundingBox(x1=0, y1=0, x2=256, y2=256)
    ], shape=images[0].shape),
]

# Augment
aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-15, 15))
])
images_aug, bbs_aug = aug(images=images, bounding_boxes=bounding_boxes)
```

## Converting to/from Other Formats

```python
from imgaug2.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np

# From numpy array (N, 4) with x1, y1, x2, y2
arr = np.array([
    [10, 20, 100, 120],
    [50, 60, 150, 160]
])
bbs = BoundingBoxesOnImage.from_xyxy_array(arr, shape=(256, 256, 3))

# To numpy array
arr_back = bbs.to_xyxy_array()
```
