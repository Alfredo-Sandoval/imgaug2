# Keypoints Examples

## Basic Usage

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data
from imgaug2.augmentables.kps import Keypoint, KeypointsOnImage

image = data.quokka(size=(256, 256))
kps = KeypointsOnImage([
    Keypoint(x=65, y=100),
    Keypoint(x=75, y=200),
], shape=image.shape)

aug = iaa.Affine(rotate=(-25, 25))
image_aug, kps_aug = aug(image=image, keypoints=kps)
```

## Deterministic

```python
aug_det = aug.to_deterministic()
image_aug = aug_det(image=image)
kps_aug = aug_det(keypoints=kps)
```

## Handling Edge Cases (Out-of-Image Keypoints)

After strong geometric transforms (large translations/crops), some keypoints may
end up outside the image.

```python
# Remove keypoints that are outside the image plane.
kps_aug = kps_aug.clip_out_of_image()
```

If you need to keep a fixed keypoint tensor shape for training, you usually
should not “remove” keypoints; instead, keep them and carry a visibility mask
in your dataset/model code.

## Visualization

```python
image_with_kps = kps_aug.draw_on_image(image_aug, size=7)
ia.imshow(image_with_kps)  # requires matplotlib
```
