# Basic Examples

## Simple Pipeline

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data

image = data.quokka(size=(256, 256))

aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.Affine(rotate=(-20, 20))
])

image_aug = aug(image=image)
```

## Common Augmentation

```python
aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.SomeOf((0, 3), [
        iaa.GaussianBlur(sigma=(0, 0.5)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        iaa.Multiply((0.8, 1.2)),
        iaa.LinearContrast((0.75, 1.25)),
        iaa.Affine(rotate=(-25, 25), scale=(0.9, 1.1))
    ], random_order=True)
])
```

## Batch Processing

```python
images = [data.quokka(size=(256, 256)) for _ in range(16)]
images_aug = aug(images=images)
```

## Reproducibility

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data

ia.seed(42)  # Set global seed for reproducible runs

image = data.quokka(size=(256, 256))
bbs = ia.BoundingBoxesOnImage(
    [ia.BoundingBox(x1=30, y1=40, x2=180, y2=200, label="obj")],
    shape=image.shape,
)

aug = iaa.Affine(rotate=(-25, 25), translate_px={"x": (-10, 10), "y": (-10, 10)})

# Best: pass images + annotations together (keeps them in sync)
image_aug, bbs_aug = aug(image=image, bounding_boxes=bbs)
```

See also: [Reproducibility & Determinism](../reproducibility.md).
