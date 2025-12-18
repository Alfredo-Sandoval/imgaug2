# Line Strings Examples

Line strings are useful for polyline annotations (lanes, trajectories, skeleton
edges, contours, etc).

They behave like other geometric annotations: geometric transforms move them
with the image.

## Basic Usage

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data
from imgaug2.augmentables.lines import LineString, LineStringsOnImage

image = data.quokka(size=(256, 256))

lines = LineStringsOnImage(
    [
        LineString([(60, 60), (200, 90), (220, 210)]),
        LineString([(30, 200), (120, 180), (230, 230)]),
    ],
    shape=image.shape,
)

aug = iaa.Affine(
    rotate=(-25, 25),
    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
    scale=(0.9, 1.1),
    mode="edge",
)

image_aug, lines_aug = aug(image=image, line_strings=lines)

# visualize (requires matplotlib)
vis = lines_aug.draw_on_image(image_aug, color=(255, 0, 0), size=2)
ia.imshow(vis)
```

## Deterministic (Separate Calls)

```python
aug_det = aug.to_deterministic()
image_aug = aug_det(image=image)
lines_aug = aug_det(line_strings=lines)
```

## Handling Edge Cases (Clipping / Removal)

Line strings can extend outside the image after strong geometric transforms.

These helpers require `Shapely`:

```python
# Clip line strings to the image plane.
lines_clipped = lines_aug.clip_out_of_image()

# Remove line strings that are fully outside.
lines_visible = lines_aug.remove_out_of_image(fully=True, partly=False)
```

See also: [All Augmentables Together](all_augmentables.md).
