# Heatmaps Examples

Heatmaps are 2D arrays of float values representing spatial distributions, commonly used for pose estimation confidence maps, attention maps, or density predictions.

## Basic Heatmap Augmentation

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.heatmaps import HeatmapsOnImage

# Load image
image = ia.quokka(size=(256, 256))

# Create a sample heatmap (values in [0, 1])
heatmap_arr = np.zeros((256, 256), dtype=np.float32)
heatmap_arr[100:150, 100:150] = 1.0  # Hot region

heatmap = HeatmapsOnImage(heatmap_arr, shape=image.shape)

# Create augmenter
aug = iaa.Affine(rotate=(-25, 25), scale=(0.8, 1.2))

# Augment both
image_aug, heatmap_aug = aug(image=image, heatmaps=heatmap)
```

## Multiple Heatmaps

You can have multiple heatmap channels (e.g., one per keypoint):

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.heatmaps import HeatmapsOnImage

image = ia.quokka(size=(256, 256))

# Create multi-channel heatmap (H, W, C)
heatmap_arr = np.zeros((256, 256, 3), dtype=np.float32)
heatmap_arr[50:100, 50:100, 0] = 1.0    # Channel 0
heatmap_arr[100:150, 100:150, 1] = 1.0  # Channel 1
heatmap_arr[150:200, 150:200, 2] = 1.0  # Channel 2

heatmap = HeatmapsOnImage(heatmap_arr, shape=image.shape)

aug = iaa.Affine(rotate=45)
image_aug, heatmap_aug = aug(image=image, heatmaps=heatmap)

# Access individual channels
heatmap_channel_0 = heatmap_aug.get_arr()[:, :, 0]
```

## Deterministic Augmentation

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.heatmaps import HeatmapsOnImage

image = ia.quokka(size=(256, 256))
heatmap_arr = np.random.rand(256, 256).astype(np.float32)
heatmap = HeatmapsOnImage(heatmap_arr, shape=image.shape)

# Deterministic for consistency
aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-25, 25))
])
aug_det = aug.to_deterministic()

image_aug = aug_det(image=image)
heatmap_aug = aug_det(heatmaps=heatmap)
```

## Visualizing Heatmaps

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.heatmaps import HeatmapsOnImage

image = ia.quokka(size=(256, 256))

# Create Gaussian-like heatmap
y, x = np.mgrid[0:256, 0:256]
center_y, center_x = 128, 128
heatmap_arr = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 50**2))
heatmap_arr = heatmap_arr.astype(np.float32)

heatmap = HeatmapsOnImage(heatmap_arr, shape=image.shape)

# Draw heatmap overlay on image
image_with_heatmap = heatmap.draw_on_image(image, alpha=0.5)

# Augment and visualize
aug = iaa.Affine(rotate=30)
image_aug, heatmap_aug = aug(image=image, heatmaps=heatmap)
image_aug_with_heatmap = heatmap_aug.draw_on_image(image_aug, alpha=0.5)

# Save
import imageio
imageio.imwrite("heatmap_before.jpg", image_with_heatmap[0])
imageio.imwrite("heatmap_after.jpg", image_aug_with_heatmap[0])
```

## Working with Different Resolutions

Heatmaps can have different resolution than the image:

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.heatmaps import HeatmapsOnImage

image = ia.quokka(size=(256, 256))

# Lower resolution heatmap (will be resized internally)
heatmap_arr = np.zeros((64, 64), dtype=np.float32)
heatmap_arr[20:40, 20:40] = 1.0

# Specify image shape for proper correspondence
heatmap = HeatmapsOnImage(heatmap_arr, shape=image.shape)

aug = iaa.Affine(rotate=45)
image_aug, heatmap_aug = aug(image=image, heatmaps=heatmap)

# Get heatmap at original resolution
heatmap_64x64 = heatmap_aug.get_arr()  # Still 64x64

# Resize to match image
heatmap_256x256 = heatmap_aug.resize((256, 256))
```

## Batch Processing

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.heatmaps import HeatmapsOnImage

# Multiple images and heatmaps
images = [ia.quokka(size=(256, 256)) for _ in range(4)]
heatmaps = [
    HeatmapsOnImage(
        np.random.rand(256, 256).astype(np.float32),
        shape=images[0].shape
    )
    for _ in range(4)
]

aug = iaa.Affine(rotate=(-25, 25))
images_aug, heatmaps_aug = aug(images=images, heatmaps=heatmaps)
```

## Heatmap Properties

```python
import numpy as np
from imgaug2.augmentables.heatmaps import HeatmapsOnImage

heatmap_arr = np.random.rand(128, 128, 3).astype(np.float32)
heatmap = HeatmapsOnImage(heatmap_arr, shape=(256, 256, 3))

# Properties
print(f"Shape: {heatmap.shape}")
print(f"Heatmap array shape: {heatmap.get_arr().shape}")
print(f"Number of channels: {heatmap.get_arr().shape[-1] if len(heatmap.get_arr().shape) == 3 else 1}")
print(f"Min value: {heatmap.min_value}")
print(f"Max value: {heatmap.max_value}")
```
