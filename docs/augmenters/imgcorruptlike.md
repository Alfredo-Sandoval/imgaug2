# imgcorruptlike Augmenters

Augmenters that mimic the imagecorruptions library. These provide standardized corruption benchmarks.

!!! note "Optional dependency"
    `imgaug2.augmenters.imgcorruptlike` wraps the upstream `imagecorruptions` package.
    Install it manually:

    ```bash
    pip install imagecorruptions
    ```

## Usage

```python
import imgaug2.augmenters.imgcorruptlike as iaa_corrupt

# Apply specific corruption
aug = iaa_corrupt.GaussianNoise(severity=3)
aug = iaa_corrupt.MotionBlur(severity=(1, 5))
aug = iaa_corrupt.Snow(severity=2)
```

## Severity Levels

All imgcorruptlike augmenters accept a `severity` parameter from 1-5:
- 1: Mild corruption
- 3: Moderate corruption
- 5: Severe corruption

## Notes / Gotchas

- These augmenters are intended for **robustness benchmarking**. They can be
  quite strong at severity 4–5.
- They are **image-only** (no bounding boxes/keypoints/polygons/segmaps/heatmaps).
- Outputs are typically `uint8` to match `imagecorruptions` behavior. If your
  training pipeline expects `float32`, convert explicitly at the edges.

## Noise Corruptions

```python
import imgaug2.augmenters.imgcorruptlike as iaa_corrupt

aug = iaa_corrupt.GaussianNoise(severity=(1, 5))
aug = iaa_corrupt.ShotNoise(severity=(1, 5))
aug = iaa_corrupt.ImpulseNoise(severity=(1, 5))
aug = iaa_corrupt.SpeckleNoise(severity=(1, 5))
```

## Blur Corruptions

```python
import imgaug2.augmenters.imgcorruptlike as iaa_corrupt

aug = iaa_corrupt.GaussianBlur(severity=(1, 5))
aug = iaa_corrupt.GlassBlur(severity=(1, 5))
aug = iaa_corrupt.MotionBlur(severity=(1, 5))
aug = iaa_corrupt.DefocusBlur(severity=(1, 5))
aug = iaa_corrupt.ZoomBlur(severity=(1, 5))
```

## Weather Corruptions

```python
import imgaug2.augmenters.imgcorruptlike as iaa_corrupt

aug = iaa_corrupt.Snow(severity=(1, 5))
aug = iaa_corrupt.Frost(severity=(1, 5))
aug = iaa_corrupt.Fog(severity=(1, 5))
aug = iaa_corrupt.Spatter(severity=(1, 5))
```

## Digital Corruptions

```python
import imgaug2.augmenters.imgcorruptlike as iaa_corrupt

aug = iaa_corrupt.Brightness(severity=(1, 5))
aug = iaa_corrupt.Contrast(severity=(1, 5))
aug = iaa_corrupt.Saturate(severity=(1, 5))
aug = iaa_corrupt.JpegCompression(severity=(1, 5))
aug = iaa_corrupt.Pixelate(severity=(1, 5))
aug = iaa_corrupt.ElasticTransform(severity=(1, 5))
```

## All imgcorruptlike Augmenters

| Category | Augmenters |
|----------|------------|
| Noise | `GaussianNoise`, `ShotNoise`, `ImpulseNoise`, `SpeckleNoise` |
| Blur | `GaussianBlur`, `GlassBlur`, `MotionBlur`, `DefocusBlur`, `ZoomBlur` |
| Weather | `Snow`, `Frost`, `Fog`, `Spatter` |
| Digital | `Brightness`, `Contrast`, `Saturate`, `JpegCompression`, `Pixelate`, `ElasticTransform` |
