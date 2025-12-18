# pillike Augmenters

Augmenters that mimic PIL/Pillow operations.

![pillike augmenter gallery](../assets/gallery/pillike_ops.png)

These are useful when you want behavior close to PIL-based augmentation recipes
(e.g. for RandAugment-style pipelines) or when you want output parity with
PIL filters/enhancers.

## Usage

```python
import imgaug2.augmenters.pillike as iaa_pil

aug = iaa_pil.EnhanceColor((0.5, 1.5))
aug = iaa_pil.EnhanceSharpness((0.5, 2.0))
```

## Notes / Gotchas

### dtype expectations

Most PIL-like operations are most predictable on `uint8` images in `[0, 255]`.

If you train with `float32`, validate the output on a few samples (some PIL-ish
ops may clip/cast internally).

### `pillike.Affine` is images-only

`imgaug2.augmenters.pillike.Affine` matches PIL’s affine transform behavior, but
currently **cannot** transform non-image augmentables (bbs/kps/polys/segmaps/heatmaps).

If you need label-safe affine transforms, use:

```python
import imgaug2.augmenters as iaa
aug = iaa.Affine(rotate=(-15, 15), translate_percent=(-0.05, 0.05))
```

## Performance Notes

PIL-like ops can be slower than pure NumPy/OpenCV ops in some pipelines. If
performance is critical, benchmark your target batch sizes and consider using
OpenCV-based equivalents.

## Enhance Operations

```python
import imgaug2.augmenters.pillike as iaa_pil

# Color enhancement (saturation)
aug = iaa_pil.EnhanceColor((0.5, 1.5))

# Contrast enhancement
aug = iaa_pil.EnhanceContrast((0.5, 1.5))

# Brightness enhancement
aug = iaa_pil.EnhanceBrightness((0.5, 1.5))

# Sharpness enhancement
aug = iaa_pil.EnhanceSharpness((0.5, 2.0))
```

## Filter Operations

```python
import imgaug2.augmenters.pillike as iaa_pil

# Apply PIL filters
aug = iaa_pil.FilterBlur()
aug = iaa_pil.FilterSmooth()
aug = iaa_pil.FilterSmoothMore()
aug = iaa_pil.FilterDetail()
aug = iaa_pil.FilterSharpen()
aug = iaa_pil.FilterContour()
aug = iaa_pil.FilterEdgeEnhance()
aug = iaa_pil.FilterEdgeEnhanceMore()
aug = iaa_pil.FilterEmboss()
aug = iaa_pil.FilterFindEdges()
```

## Other Operations

```python
import imgaug2.augmenters.pillike as iaa_pil

# Autocontrast
aug = iaa_pil.Autocontrast()
aug = iaa_pil.Autocontrast(cutoff=(0, 10))

# Equalize
aug = iaa_pil.Equalize()

# Posterize
aug = iaa_pil.Posterize(nb_bits=(4, 8))

# Solarize
aug = iaa_pil.Solarize(threshold=(32, 128))

# Invert
aug = iaa_pil.Invert()
```

## Affine Operations

```python
import imgaug2.augmenters.pillike as iaa_pil

aug = iaa_pil.Affine(
    scale=(0.8, 1.2),
    translate_percent={"x": (-0.1, 0.1)},
    rotate=(-25, 25)
)
```

## All pillike Augmenters

| Category | Augmenters |
|----------|------------|
| Enhance | `EnhanceColor`, `EnhanceContrast`, `EnhanceBrightness`, `EnhanceSharpness` |
| Filters | `FilterBlur`, `FilterSmooth`, `FilterSmoothMore`, `FilterDetail`, `FilterSharpen`, `FilterContour`, `FilterEdgeEnhance`, `FilterEdgeEnhanceMore`, `FilterEmboss`, `FilterFindEdges` |
| Color | `Autocontrast`, `Equalize`, `Posterize`, `Solarize`, `Invert` |
| Geometric | `Affine` |
