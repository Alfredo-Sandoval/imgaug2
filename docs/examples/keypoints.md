# Keypoints Examples

Keypoints represent specific locations in images, such as facial landmarks or body joints. imgaug2 can augment keypoints alongside images to maintain spatial correspondence.

## Basic Keypoint Augmentation

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.kps import Keypoint, KeypointsOnImage

# Load image
image = ia.quokka(size=(256, 256))

# Define keypoints
kps = KeypointsOnImage([
    Keypoint(x=65, y=100),
    Keypoint(x=75, y=200),
    Keypoint(x=100, y=100),
    Keypoint(x=200, y=80)
], shape=image.shape)

# Create augmenter
aug = iaa.Affine(rotate=(-45, 45))

# Augment image and keypoints together
image_aug, kps_aug = aug(image=image, keypoints=kps)
```

## Deterministic Augmentation

When you need the exact same transformation applied to multiple data types:

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.kps import Keypoint, KeypointsOnImage

image = ia.quokka(size=(256, 256))
kps = KeypointsOnImage([
    Keypoint(x=65, y=100),
    Keypoint(x=75, y=200),
], shape=image.shape)

# Create augmenter and make it deterministic
aug = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.GaussianBlur(sigma=(0, 1.0))
])
aug_det = aug.to_deterministic()

# Apply same transformation to both
image_aug = aug_det(image=image)
kps_aug = aug_det(keypoints=kps)
```

## Handling Out-of-Image Keypoints

After augmentation, some keypoints may fall outside the image boundaries:

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.kps import Keypoint, KeypointsOnImage

image = ia.quokka(size=(256, 256))
kps = KeypointsOnImage([
    Keypoint(x=10, y=10),   # Near edge
    Keypoint(x=250, y=250), # Near opposite edge
], shape=image.shape)

aug = iaa.Affine(translate_percent={"x": 0.2, "y": 0.2})
image_aug, kps_aug = aug(image=image, keypoints=kps)

# Check which keypoints are still in the image
for i, kp in enumerate(kps_aug.keypoints):
    if kp.is_out_of_image(image_aug):
        print(f"Keypoint {i} is outside the image")
    else:
        print(f"Keypoint {i}: ({kp.x:.1f}, {kp.y:.1f})")

# Remove out-of-image keypoints
kps_aug_clipped = kps_aug.remove_out_of_image_()
```

## Visualizing Keypoints

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.kps import Keypoint, KeypointsOnImage

image = ia.quokka(size=(256, 256))
kps = KeypointsOnImage([
    Keypoint(x=65, y=100),
    Keypoint(x=75, y=200),
    Keypoint(x=100, y=100),
    Keypoint(x=200, y=80)
], shape=image.shape)

# Draw keypoints on image
image_with_kps = kps.draw_on_image(image, size=7)

# Augment and visualize
aug = iaa.Affine(rotate=(-25, 25))
image_aug, kps_aug = aug(image=image, keypoints=kps)
image_aug_with_kps = kps_aug.draw_on_image(image_aug, size=7)

# Save
import imageio
imageio.imwrite("keypoints_before.jpg", image_with_kps)
imageio.imwrite("keypoints_after.jpg", image_aug_with_kps)
```

## Batch Augmentation

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
from imgaug2.augmentables.kps import Keypoint, KeypointsOnImage

# Multiple images and keypoints
images = [ia.quokka(size=(256, 256)) for _ in range(4)]
keypoints = [
    KeypointsOnImage([
        Keypoint(x=65, y=100),
        Keypoint(x=75, y=200),
    ], shape=images[0].shape)
    for _ in range(4)
]

# Augment batch
aug = iaa.Affine(rotate=(-25, 25), scale=(0.8, 1.2))
images_aug, keypoints_aug = aug(images=images, keypoints=keypoints)
```

## Keypoint Properties

```python
from imgaug2.augmentables.kps import Keypoint, KeypointsOnImage

kp = Keypoint(x=50.5, y=100.2)

# Access coordinates
print(f"x: {kp.x}, y: {kp.y}")
print(f"Integer coords: ({kp.x_int}, {kp.y_int})")

# Compute distance to another keypoint
kp2 = Keypoint(x=60, y=110)
distance = kp.distance(kp2)

# Shift keypoint
kp_shifted = kp.shift(x=10, y=-5)
```
