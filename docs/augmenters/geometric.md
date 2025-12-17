# Geometric Augmenters

Augmenters that perform spatial/geometric transformations on images.

## Affine

Apply affine transformations (rotation, scale, translation, shear).

```python
import imgaug2.augmenters as iaa

# Combined transformations
aug = iaa.Affine(
    scale=(0.8, 1.2),
    rotate=(-25, 25),
    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    shear=(-8, 8)
)

# Individual transforms
aug = iaa.Affine(rotate=(-45, 45))
aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})
```

## Rotate

Rotate images.

```python
import imgaug2.augmenters as iaa

aug = iaa.Rotate((-45, 45))
```

## Rot90

Rotate by 90 degree increments.

```python
import imgaug2.augmenters as iaa

aug = iaa.Rot90(k=(0, 3))  # 0, 90, 180, or 270 degrees
aug = iaa.Rot90(k=1)       # Always 90 degrees
```

## ShearX / ShearY

Shear along specific axis.

```python
import imgaug2.augmenters as iaa

aug = iaa.ShearX((-20, 20))  # Horizontal shear
aug = iaa.ShearY((-20, 20))  # Vertical shear
```

## ScaleX / ScaleY

Scale along specific axis.

```python
import imgaug2.augmenters as iaa

aug = iaa.ScaleX((0.5, 1.5))  # Horizontal scale
aug = iaa.ScaleY((0.5, 1.5))  # Vertical scale
```

## TranslateX / TranslateY

Translate along specific axis.

```python
import imgaug2.augmenters as iaa

aug = iaa.TranslateX(percent=(-0.1, 0.1))
aug = iaa.TranslateY(px=(-20, 20))  # Or in pixels
```

## PiecewiseAffine

Apply local affine transformations on a grid.

```python
import imgaug2.augmenters as iaa

aug = iaa.PiecewiseAffine(scale=(0.01, 0.05))
```

## PerspectiveTransform

Apply perspective transformation.

```python
import imgaug2.augmenters as iaa

aug = iaa.PerspectiveTransform(scale=(0.01, 0.15))
```

## ElasticTransformation

Apply elastic deformation.

```python
import imgaug2.augmenters as iaa

aug = iaa.ElasticTransformation(alpha=(0, 50), sigma=(4, 6))
```

## WithPolarWarping

Apply augmenters in polar coordinates.

```python
import imgaug2.augmenters as iaa

aug = iaa.WithPolarWarping(iaa.Affine(translate_percent={"x": (-0.1, 0.1)}))
```

## Jigsaw

Shuffle image patches.

```python
import imgaug2.augmenters as iaa

aug = iaa.Jigsaw(nb_rows=4, nb_cols=4)
```

## All Geometric Augmenters

| Augmenter | Description |
|-----------|-------------|
| `Affine` | Affine transformations |
| `AffineCv2` | Affine using OpenCV |
| `Rotate` | Rotation |
| `Rot90` | 90-degree rotations |
| `ShearX` / `ShearY` | Shear transforms |
| `ScaleX` / `ScaleY` | Scale transforms |
| `TranslateX` / `TranslateY` | Translation |
| `PiecewiseAffine` | Local affine grid |
| `PerspectiveTransform` | Perspective |
| `ElasticTransformation` | Elastic deformation |
| `WithPolarWarping` | Polar coordinate transforms |
| `Jigsaw` | Patch shuffling |
