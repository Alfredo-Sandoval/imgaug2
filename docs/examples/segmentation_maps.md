# Segmentation Maps Examples

Segmentation maps assign class labels to each pixel, used in semantic segmentation tasks.

## Basic Segmentation Map Augmentation

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

# Load image
image = ia.quokka(size=(256, 256))

# Create segmentation map (integer class labels)
segmap_arr = np.zeros((256, 256), dtype=np.int32)
segmap_arr[50:200, 50:200] = 1  # Class 1: foreground
segmap_arr[100:150, 100:150] = 2  # Class 2: special region

segmap = SegmentationMapsOnImage(segmap_arr, shape=image.shape)

# Create augmenter
aug = iaa.Affine(rotate=(-25, 25), scale=(0.8, 1.2))

# Augment both
image_aug, segmap_aug = aug(image=image, segmentation_maps=segmap)
```

## Deterministic Augmentation

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

image = ia.quokka(size=(256, 256))
segmap_arr = np.zeros((256, 256), dtype=np.int32)
segmap_arr[50:200, 50:200] = 1

segmap = SegmentationMapsOnImage(segmap_arr, shape=image.shape)

# Deterministic augmentation
aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-25, 25)),
    iaa.ElasticTransformation(alpha=50, sigma=5)
])
aug_det = aug.to_deterministic()

image_aug = aug_det(image=image)
segmap_aug = aug_det(segmentation_maps=segmap)
```

## Visualizing Segmentation Maps

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

image = ia.quokka(size=(256, 256))

# Create multi-class segmentation
segmap_arr = np.zeros((256, 256), dtype=np.int32)
segmap_arr[0:100, :] = 0     # Sky
segmap_arr[100:200, :] = 1   # Vegetation
segmap_arr[200:256, :] = 2   # Ground

segmap = SegmentationMapsOnImage(segmap_arr, shape=image.shape)

# Draw with colors
image_with_segmap = segmap.draw_on_image(image, alpha=0.5)

# Augment and draw
aug = iaa.Affine(rotate=15)
image_aug, segmap_aug = aug(image=image, segmentation_maps=segmap)
image_aug_with_segmap = segmap_aug.draw_on_image(image_aug, alpha=0.5)

# Save
import imageio
imageio.imwrite("segmap_before.jpg", image_with_segmap[0])
imageio.imwrite("segmap_after.jpg", image_aug_with_segmap[0])
```

## Handling Multiple Classes

```python
import numpy as np
import imgaug2 as ia
from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

image = ia.quokka(size=(256, 256))

# Multi-class segmentation (e.g., COCO classes)
segmap_arr = np.zeros((256, 256), dtype=np.int32)
segmap_arr[0:80, :] = 0      # Background (class 0)
segmap_arr[80:160, 0:128] = 1   # Person (class 1)
segmap_arr[80:160, 128:256] = 2 # Car (class 2)
segmap_arr[160:256, :] = 3     # Road (class 3)

segmap = SegmentationMapsOnImage(segmap_arr, shape=image.shape)

# Get unique classes
arr = segmap.get_arr()
unique_classes = np.unique(arr)
print(f"Classes in segmap: {unique_classes}")
```

## Working with Different Resolutions

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

image = ia.quokka(size=(256, 256))

# Lower resolution segmap (common in deep learning)
segmap_arr = np.zeros((64, 64), dtype=np.int32)
segmap_arr[16:48, 16:48] = 1

segmap = SegmentationMapsOnImage(segmap_arr, shape=image.shape)

# Augment (handles resolution differences automatically)
aug = iaa.Affine(rotate=30)
image_aug, segmap_aug = aug(image=image, segmentation_maps=segmap)

# Access at different resolutions
segmap_64 = segmap_aug.get_arr()  # Still 64x64
segmap_256 = segmap_aug.resize((256, 256)).get_arr()  # Resized
```

## Batch Processing

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

# Multiple images and segmentation maps
images = [ia.quokka(size=(256, 256)) for _ in range(4)]
segmaps = []
for i in range(4):
    arr = np.zeros((256, 256), dtype=np.int32)
    arr[50+i*10:150+i*10, 50+i*10:150+i*10] = 1
    segmaps.append(SegmentationMapsOnImage(arr, shape=images[0].shape))

aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-15, 15))
])
images_aug, segmaps_aug = aug(images=images, segmentation_maps=segmaps)
```

## Instance Segmentation

For instance segmentation where each object has a unique ID:

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

image = ia.quokka(size=(256, 256))

# Instance segmentation: each instance has unique ID
instance_map = np.zeros((256, 256), dtype=np.int32)
instance_map[50:100, 50:100] = 1   # Instance 1
instance_map[80:130, 120:180] = 2  # Instance 2
instance_map[150:200, 60:120] = 3  # Instance 3

segmap = SegmentationMapsOnImage(instance_map, shape=image.shape)

aug = iaa.Affine(rotate=(-25, 25))
image_aug, segmap_aug = aug(image=image, segmentation_maps=segmap)

# Instance IDs are preserved after augmentation
```

## Converting to/from Arrays

```python
import numpy as np
from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

# From numpy array
arr = np.random.randint(0, 5, size=(256, 256), dtype=np.int32)
segmap = SegmentationMapsOnImage(arr, shape=(256, 256, 3))

# To numpy array
arr_back = segmap.get_arr()

# Get boolean mask for specific class
class_1_mask = segmap.get_arr() == 1
```
