# Augmenters Overview

imgaug2 provides a wide variety of image augmentation techniques organized into categories. Each augmenter can be used standalone or combined into pipelines using meta-augmenters.

## Categories

| Category | Description | Examples |
|----------|-------------|----------|
| [Arithmetic](arithmetic.md) | Pixel value operations | `Add`, `Multiply`, `Dropout`, `SaltAndPepper` |
| [Blur](blur.md) | Blurring effects | `GaussianBlur`, `AverageBlur`, `MotionBlur` |
| [Color](color.md) | Color space transforms | `Grayscale`, `ChangeColorspace`, `AddToHueAndSaturation` |
| [Contrast](contrast.md) | Contrast adjustments | `LinearContrast`, `GammaContrast`, `CLAHE` |
| [Geometric](geometric.md) | Spatial transforms | `Affine`, `Rotate`, `ElasticTransformation`, `PerspectiveTransform` |
| [Flip](flip.md) | Mirror operations | `Fliplr`, `Flipud` |
| [Segmentation](segmentation.md) | Superpixel effects | `Superpixels`, `Voronoi` |
| [Size](size.md) | Resize and crop | `Resize`, `CropAndPad`, `PadToFixedSize` |
| [Weather](weather.md) | Weather effects | `Clouds`, `Fog`, `Snow`, `Rain` |
| [Meta](meta.md) | Pipeline control | `Sequential`, `SomeOf`, `Sometimes`, `OneOf` |

## Quick Reference

### Most Common Augmenters

```python
import imgaug2.augmenters as iaa

# Geometric
iaa.Fliplr(0.5)                    # Horizontal flip with 50% probability
iaa.Affine(rotate=(-25, 25))       # Random rotation
iaa.Affine(scale=(0.8, 1.2))       # Random scaling

# Blur & Noise
iaa.GaussianBlur(sigma=(0, 1.0))   # Gaussian blur
iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))  # Gaussian noise

# Color & Contrast
iaa.Multiply((0.8, 1.2))           # Brightness adjustment
iaa.LinearContrast((0.75, 1.25))   # Contrast adjustment
iaa.Grayscale(alpha=(0.0, 1.0))    # Random grayscale

# Weather
iaa.Clouds()                        # Cloud overlay
iaa.Fog()                           # Fog effect
```

### Building Pipelines

```python
import imgaug2.augmenters as iaa

# Sequential - apply all in order
aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.Affine(rotate=(-20, 20))
])

# SomeOf - apply N random augmenters
aug = iaa.SomeOf((1, 3), [
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
    iaa.Multiply((0.8, 1.2)),
    iaa.LinearContrast((0.75, 1.25))
])

# Sometimes - apply with probability
aug = iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.0)))

# OneOf - pick exactly one
aug = iaa.OneOf([
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.AverageBlur(k=(2, 5)),
    iaa.MotionBlur(k=5)
])
```

## Parameters

Most augmenters accept flexible parameter types:

```python
# Single value - always use this value
iaa.Affine(rotate=45)

# Tuple (a, b) - uniform random in range [a, b]
iaa.Affine(rotate=(-45, 45))

# List [a, b, c] - pick one randomly
iaa.Affine(rotate=[0, 90, 180, 270])

# Stochastic parameter - custom distribution
from imgaug2 import parameters as iap
iaa.Affine(rotate=iap.Normal(0, 10))
```

## Applying to Images and Annotations

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa

# Load data
image = ia.quokka(size=(256, 256))
bbs = ia.BoundingBoxesOnImage([
    ia.BoundingBox(x1=50, y1=50, x2=200, y2=200)
], shape=image.shape)

# Create augmenter
aug = iaa.Affine(rotate=(-25, 25))

# For consistent augmentation across image and annotations
aug_det = aug.to_deterministic()
image_aug = aug_det(image=image)
bbs_aug = aug_det(bounding_boxes=bbs)
```

## Supported Data Types

| Data Type | Class | Notes |
|-----------|-------|-------|
| Images | `numpy.ndarray` | HWC format, uint8 or float32 |
| Bounding Boxes | `BoundingBoxesOnImage` | x1, y1, x2, y2 format |
| Keypoints | `KeypointsOnImage` | x, y coordinates |
| Polygons | `PolygonsOnImage` | List of (x, y) vertices |
| Line Strings | `LineStringsOnImage` | Connected line segments |
| Heatmaps | `HeatmapsOnImage` | Float arrays [0, 1] |
| Segmentation Maps | `SegmentationMapsOnImage` | Integer class labels |
