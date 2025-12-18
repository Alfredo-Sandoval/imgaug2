# All Augmentables Together

imgaug2 can augment **images and all annotation types in one call**, using the
same sampled random parameters. This is the recommended way to keep everything
perfectly in sync.

Supported “augmentables” include:

- images
- bounding boxes
- keypoints
- polygons
- line strings
- heatmaps
- segmentation maps

## Recommended Pattern (Single Call)

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data
from imgaug2.augmentables.lines import LineString, LineStringsOnImage

ia.seed(0)

# Base image + built-in example annotations.
image = data.quokka(size=(256, 256))
bbs = data.quokka_bounding_boxes(size=(256, 256))
kps = data.quokka_keypoints(size=(256, 256))
polys = data.quokka_polygons(size=(256, 256))
heat = data.quokka_heatmap(size=(256, 256))
seg = data.quokka_segmentation_map(size=(256, 256))

# Line strings are not shipped as a quokka fixture, so we create one manually.
lines = LineStringsOnImage(
    [LineString([(60, 60), (210, 90), (200, 210)])],
    shape=image.shape,
)

aug = iaa.Sequential(
    [
        # Geometry affects images + all annotation types.
        iaa.Affine(
            rotate=(-20, 20),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            mode="edge",
        ),
        iaa.Fliplr(0.5),
        # Photometric augmenters affect images, but are ignored for e.g.
        # segmaps/heatmaps/bounding boxes (as you would expect).
        iaa.GaussianBlur((0.0, 1.5)),
        iaa.LinearContrast((0.8, 1.2)),
    ]
)

batch_aug = aug(
    image=image,
    bounding_boxes=bbs,
    keypoints=kps,
    polygons=polys,
    line_strings=lines,
    heatmaps=heat,
    segmentation_maps=seg,
    return_batch=True,
)

# Pull results back out of the batch.
image_aug = batch_aug.images_aug[0]
bbs_aug = batch_aug.bounding_boxes_aug
kps_aug = batch_aug.keypoints_aug
polys_aug = batch_aug.polygons_aug
lines_aug = batch_aug.line_strings_aug
heat_aug = batch_aug.heatmaps_aug
seg_aug = batch_aug.segmentation_maps_aug

# Quick visualization (requires matplotlib).
vis = image_aug
vis = bbs_aug.draw_on_image(vis, size=2)
vis = kps_aug.draw_on_image(vis, size=7)
vis = polys_aug.draw_on_image(vis, color=(0, 255, 0), size=2)
vis = lines_aug.draw_on_image(vis, color=(255, 0, 0), size=2)
vis = heat_aug.draw_on_image(vis, alpha=0.35)
vis = seg_aug.draw_on_image(vis, alpha=0.35)
ia.imshow(vis)
```

## Notes / Gotchas

### Interpolation matters (especially for labels)

- Segmentation maps are **label images**. They must be warped with
  nearest-neighbor interpolation to avoid creating invalid intermediate label
  values. imgaug2 handles this for most geometric augmenters.
- Heatmaps are **continuous** values and may be warped with linear
  interpolation.

### Keep batch structure consistent for strict reproducibility

If you care about repeatability across runs, keep:

- batch sizes constant
- image sizes constant

See: [Reproducibility & Determinism](../reproducibility.md).

### Cleaning invalid / out-of-image annotations

After strong geometric transforms, it’s normal for some annotations to go
partially or fully out of frame. Typical post-processing:

```python
# Bounding boxes
bbs_aug = bbs_aug.clip_out_of_image()
bbs_aug = bbs_aug.remove_out_of_image()

# Keypoints
kps_aug = kps_aug.clip_out_of_image()

# Polygons / line strings (requires Shapely)
polys_aug = polys_aug.clip_out_of_image()
lines_aug = lines_aug.clip_out_of_image()
```

## Related Pages

- [Basics](basics.md)
- [Bounding Boxes](bounding_boxes.md)
- [Keypoints](keypoints.md)
- [Polygons](polygons.md)
- [Line Strings](line_strings.md)
- [Heatmaps](heatmaps.md)
- [Segmentation Maps](segmentation_maps.md)
- [Hooks](../hooks.md)
