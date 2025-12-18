# Bounding Box Examples

## Basic Usage

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data
from imgaug2.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

image = data.quokka(size=(256, 256))
bbs = BoundingBoxesOnImage([
    BoundingBox(x1=50, y1=50, x2=200, y2=200, label="quokka")
], shape=image.shape)

aug = iaa.Affine(rotate=(-25, 25))
image_aug, bbs_aug = aug(image=image, bounding_boxes=bbs)
```

## Handling Edge Cases

```python
# Clip to image boundaries
bbs_aug = bbs_aug.clip_out_of_image_()

# Remove out-of-image boxes
bbs_aug = bbs_aug.remove_out_of_image_()
```

## Visualization

```python
image_with_bbs = bbs_aug.draw_on_image(image_aug, size=2)
ia.imshow(image_with_bbs)  # requires matplotlib
```
