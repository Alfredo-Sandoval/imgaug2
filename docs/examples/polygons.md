# Polygons Examples

Polygons are useful for instance/region annotations (e.g. segmentation outlines).
They are treated as geometric annotations: transforms like `Affine`, `Fliplr`,
`Resize`, etc will move polygons with the image.

## Basic Usage

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data

image = data.quokka(size=(256, 256))
polys = data.quokka_polygons(size=(256, 256))

aug = iaa.Affine(
    rotate=(-25, 25),
    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
    scale=(0.9, 1.1),
    mode="edge",
)

image_aug, polys_aug = aug(image=image, polygons=polys)

# visualize (requires matplotlib)
vis = polys_aug.draw_on_image(image_aug, color=(0, 255, 0), size=2)
ia.imshow(vis)
```

## Deterministic (Separate Calls)

If you must augment polygons and images in separate calls (less common), use
`to_deterministic()`:

```python
aug_det = aug.to_deterministic()
image_aug = aug_det(image=image)
polys_aug = aug_det(polygons=polys)
```

## Handling Edge Cases (Clipping / Removal)

After geometric transforms, polygons may extend beyond the image boundaries.

Polygons support clipping/removal helpers (requires `Shapely`, which is a normal
imgaug2 dependency):

```python
# Clip polygons to the image plane (may change polygon vertex count).
polys_clipped = polys_aug.clip_out_of_image()

# Remove polygons that are fully outside.
polys_visible = polys_aug.remove_out_of_image(fully=True, partly=False)
```

See also: [Reproducibility & Determinism](../reproducibility.md) and
[All Augmentables Together](all_augmentables.md).
