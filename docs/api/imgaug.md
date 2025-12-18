# imgaug2 Module

The main `imgaug2` module provides core functions and classes.

## Import

```python
import imgaug2 as ia
```

## Core Functions

### seed

Set global random seed for reproducibility.

```python
ia.seed(42)
```

### quokka

Load sample quokka image for testing.

```python
import imgaug2.data as data

image = data.quokka()  # Full size
image = data.quokka(size=(256, 256))  # Resized
```

### quokka_square

Load square-cropped quokka image.

```python
import imgaug2.data as data

image = data.quokka_square(size=(256, 256))
```

### draw_grid

Create grid visualization of multiple images.

```python
grid = ia.draw_grid(images, cols=4)
```

### imshow

Display image (requires matplotlib).

```python
ia.imshow(image)
```

## Image Operations

### pad

Pad image to specific size.

```python
image_padded = ia.pad(image, top=10, right=10, bottom=10, left=10)
```

### compute_paddings_for_aspect_ratio

Compute padding needed for target aspect ratio.

```python
paddings = ia.compute_paddings_for_aspect_ratio(image, aspect_ratio=1.0)
```

### pool

Apply pooling operation.

```python
image_pooled = ia.avg_pool(image, (2, 2))
image_pooled = ia.max_pool(image, (2, 2))
```

## Augmentable Classes

The main module re-exports key augmentable classes:

```python
# Bounding boxes
ia.BoundingBox(x1=10, y1=20, x2=100, y2=120)
ia.BoundingBoxesOnImage([...], shape=image.shape)

# Keypoints
ia.Keypoint(x=50, y=100)
ia.KeypointsOnImage([...], shape=image.shape)

# Polygons
ia.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
ia.PolygonsOnImage([...], shape=image.shape)

# Line strings
ia.LineString([(0, 0), (50, 50), (100, 100)])
ia.LineStringsOnImage([...], shape=image.shape)

# Heatmaps
ia.HeatmapsOnImage(arr, shape=image.shape)

# Segmentation maps
ia.SegmentationMapsOnImage(arr, shape=image.shape)
```

## Batch Class

For handling batches of data:

```python
batch = ia.Batch(
    images=[image1, image2],
    bounding_boxes=[bbs1, bbs2],
    keypoints=[kps1, kps2]
)
```

## Constants

### CSPACE_*

Color space constants:

```python
ia.CSPACE_RGB
ia.CSPACE_BGR
ia.CSPACE_HSV
ia.CSPACE_HLS
ia.CSPACE_Lab
ia.CSPACE_YCrCb
```

## Utility Functions

### is_np_array

Check if object is numpy array.

```python
ia.is_np_array(obj)
```

### is_single_number

Check if object is a single number.

```python
ia.is_single_number(obj)
```

### is_iterable

Check if object is iterable (but not string).

```python
ia.is_iterable(obj)
```
