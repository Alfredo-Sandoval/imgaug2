# Segmentation Maps Examples

## Basic Usage

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data
from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

image = data.quokka(size=(256, 256))
segmap_arr = np.zeros((256, 256), dtype=np.int32)
segmap_arr[50:200, 50:200] = 1

segmap = SegmentationMapsOnImage(segmap_arr, shape=image.shape)

aug = iaa.Affine(rotate=(-25, 25))
image_aug, segmap_aug = aug(image=image, segmentation_maps=segmap)

# Overlay segmentation (requires matplotlib for display if you want to show it)
image_with_segmap = segmap_aug.draw_on_image(image_aug, alpha=0.5)
ia.imshow(image_with_segmap)
```

## Visualization Notes

- `SegmentationMapsOnImage.draw_on_image(...)` is the quickest way to sanity-check sync.
- For training, you usually pass `segmap_aug.arr` (or convert to your framework tensor).
