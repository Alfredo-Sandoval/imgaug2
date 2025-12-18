# Heatmaps Examples

## Basic Usage

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data
from imgaug2.augmentables.heatmaps import HeatmapsOnImage

image = data.quokka(size=(256, 256))
heatmap_arr = np.zeros((256, 256), dtype=np.float32)
heatmap_arr[100:150, 100:150] = 1.0

heatmap = HeatmapsOnImage(heatmap_arr, shape=image.shape)

aug = iaa.Affine(rotate=(-25, 25))
image_aug, heatmap_aug = aug(image=image, heatmaps=heatmap)
```

## Visualization

```python
image_with_heatmap = heatmap_aug.draw_on_image(image_aug, alpha=0.5)
ia.imshow(image_with_heatmap)  # requires matplotlib
```
