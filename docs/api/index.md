# API Reference

This section provides detailed API documentation for imgaug2.

For full coverage of docstrings, see the [Generated API Reference](generated/index.md).

## Modules

### Core Modules

| Module | Description |
|--------|-------------|
| [`imgaug2`](imgaug.md) | Main module with core functions and classes |
| [`imgaug2.augmenters`](augmenters.md) | All augmentation classes |
| [`imgaug2.augmentables`](augmentables.md) | Annotation classes (bboxes, keypoints, etc.) |
| [`imgaug2.parameters`](parameters.md) | Stochastic parameter classes |

### Augmenter Submodules

All augmenters are accessible via `imgaug2.augmenters` (commonly imported as `iaa`):

```python
import imgaug2.augmenters as iaa
```

| Submodule | Description |
|-----------|-------------|
| `iaa.meta` | Pipeline control augmenters |
| `iaa.arithmetic` | Pixel value operations |
| `iaa.blur` | Blurring effects |
| `iaa.color` | Color modifications |
| `iaa.contrast` | Contrast adjustments |
| `iaa.geometric` | Spatial transforms |
| `iaa.flip` | Mirror operations |
| `iaa.size` | Resize and crop |
| `iaa.weather` | Weather effects |
| `iaa.segmentation` | Superpixel effects |
| `iaa.convolutional` | Convolution operations |
| `iaa.blend` | Blending effects |
| `iaa.pooling` | Pooling operations |
| `iaa.imgcorruptlike` | ImageCorruptions-style augmenters |
| `iaa.pillike` | PIL-style augmenters |

## Quick Import Reference

```python
# Core imports
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2 import parameters as iap

# Augmentable imports
from imgaug2.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug2.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug2.augmentables.polys import Polygon, PolygonsOnImage
from imgaug2.augmentables.heatmaps import HeatmapsOnImage
from imgaug2.augmentables.segmaps import SegmentationMapsOnImage
```

## Common Classes

### Augmenters

Most commonly used augmenters:

- `iaa.Sequential` - Apply augmenters in sequence
- `iaa.SomeOf` - Apply random subset
- `iaa.OneOf` - Apply one random augmenter
- `iaa.Sometimes` - Apply with probability
- `iaa.Fliplr` - Horizontal flip
- `iaa.Flipud` - Vertical flip
- `iaa.Affine` - Affine transformations
- `iaa.GaussianBlur` - Gaussian blur
- `iaa.AdditiveGaussianNoise` - Add noise
- `iaa.Multiply` - Brightness adjustment
- `iaa.LinearContrast` - Contrast adjustment

### Augmentables

- `ia.BoundingBox` - Single bounding box
- `ia.BoundingBoxesOnImage` - Multiple bounding boxes
- `ia.Keypoint` - Single keypoint
- `ia.KeypointsOnImage` - Multiple keypoints
- `ia.Polygon` - Single polygon
- `ia.PolygonsOnImage` - Multiple polygons
- `ia.HeatmapsOnImage` - Heatmap arrays
- `ia.SegmentationMapsOnImage` - Segmentation maps

### Parameters

- `iap.Uniform` - Uniform distribution
- `iap.Normal` - Normal distribution
- `iap.Choice` - Random choice
- `iap.Deterministic` - Fixed value
