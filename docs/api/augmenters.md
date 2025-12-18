# imgaug2.augmenters Module

The augmenters module contains all image augmentation classes.

## Import

```python
import imgaug2.augmenters as iaa
```

## Base Class

### Augmenter

All augmenters inherit from `Augmenter`:

```python
class Augmenter(metaclass=ABCMeta):
    def __call__(self, image=None, images=None, ...):
        """Apply augmentation."""

    def augment(self, image=None, images=None, ...):
        """Apply augmentation (alias for __call__)."""

    def to_deterministic(self):
        """Create deterministic version."""

    def get_parameters(self):
        """Get augmenter parameters."""
```

## Calling Augmenters

Augmenters can be called directly:

```python
aug = iaa.GaussianBlur(sigma=1.0)

# Single image
image_aug = aug(image=image)

# Batch of images
images_aug = aug(images=[image1, image2])

# With annotations
image_aug, bbs_aug = aug(image=image, bounding_boxes=bbs)

# All supported data types
image_aug, bbs_aug, kps_aug, segmap_aug = aug(
    image=image,
    bounding_boxes=bbs,
    keypoints=kps,
    segmentation_maps=segmap
)
```

## Supported Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | ndarray | Single image |
| `images` | list[ndarray] | Batch of images |
| `bounding_boxes` | BoundingBoxesOnImage | Bounding boxes |
| `keypoints` | KeypointsOnImage | Keypoints |
| `polygons` | PolygonsOnImage | Polygons |
| `line_strings` | LineStringsOnImage | Line strings |
| `heatmaps` | HeatmapsOnImage | Heatmaps |
| `segmentation_maps` | SegmentationMapsOnImage | Segmentation maps |

## Deterministic Mode

Deterministic mode is useful when you need the **same sampled parameters**
applied to multiple inputs (e.g. stereo pairs, or image + annotations).

```python
import imgaug2.augmenters as iaa

aug = iaa.Affine(rotate=(-25, 25))
aug_det = aug.to_deterministic()

# Example 1: apply identical transform parameters to two batches
left_aug = aug_det(images=left_images)
right_aug = aug_det(images=right_images)

# Example 2: image + bounding boxes (recommended: single call)
image_aug, bbs_aug = aug(image=image, bounding_boxes=bbs)

# Example 3: image + bounding boxes (separate calls using deterministic mode)
image_aug = aug_det(image=image)
bbs_aug = aug_det(bounding_boxes=bbs)
```

See also: [Reproducibility & Determinism](../reproducibility.md).

## Submodules

| Module | Import | Contents |
|--------|--------|----------|
| meta | `iaa.meta` | Sequential, SomeOf, Sometimes, etc. |
| arithmetic | `iaa.arithmetic` | Add, Multiply, Dropout, etc. |
| blur | `iaa.blur` | GaussianBlur, MotionBlur, etc. |
| color | `iaa.color` | Grayscale, AddToHue, etc. |
| contrast | `iaa.contrast` | LinearContrast, CLAHE, etc. |
| geometric | `iaa.geometric` | Affine, Rotate, PerspectiveTransform, etc. |
| flip | `iaa.flip` | Fliplr, Flipud |
| size | `iaa.size` | Resize, Crop, Pad, etc. |
| weather | `iaa.weather` | Clouds, Fog, Snow, Rain |
| segmentation | `iaa.segmentation` | Superpixels, Voronoi |
| convolutional | `iaa.convolutional` | Sharpen, Emboss, EdgeDetect |
| blend | `iaa.blend` | BlendAlpha variants |
| pooling | `iaa.pooling` | AveragePooling, MaxPooling |
| imgcorruptlike | `iaa.imgcorruptlike` | ImageCorruptions-style augmenters |
| pillike | `iaa.pillike` | PIL-style augmenters |

## Complete Augmenter List

See the [Augmenters Overview](../augmenters/index.md) for a complete list of all available augmenters organized by category.
