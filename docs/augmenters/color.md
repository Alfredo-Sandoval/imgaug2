# Color Augmenters

Augmenters that modify color properties of images.

## Grayscale

Convert to grayscale with blending.

```python
import imgaug2.augmenters as iaa

aug = iaa.Grayscale(alpha=1.0)           # Full grayscale
aug = iaa.Grayscale(alpha=(0.0, 1.0))    # Random grayscale intensity
```

## ChangeColorspace

Convert between color spaces.

```python
aug = iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV")
```

## AddToHueAndSaturation

Modify hue and saturation.

```python
aug = iaa.AddToHueAndSaturation((-20, 20))  # Shift hue
aug = iaa.AddToHue((-20, 20))               # Hue only
aug = iaa.AddToSaturation((-20, 20))        # Saturation only
```

## MultiplyHueAndSaturation

Scale hue and saturation.

```python
aug = iaa.MultiplyHueAndSaturation((0.5, 1.5))
aug = iaa.MultiplyHue((0.8, 1.2))
aug = iaa.MultiplySaturation((0.5, 1.5))
```

## ChangeColorTemperature

Adjust color temperature (warm/cool).

```python
aug = iaa.ChangeColorTemperature((3000, 8000))  # Kelvin range
```

## WithColorspace

Apply augmenters in a different color space.

```python
aug = iaa.WithColorspace(
    to_colorspace="HSV",
    children=iaa.Add((-20, 20), per_channel=True)
)
```

## WithBrightnessChannels

Apply augmenters to brightness channels only.

```python
aug = iaa.WithBrightnessChannels(iaa.Add((-20, 20)))
```

## All Color Augmenters

| Augmenter | Description |
|-----------|-------------|
| `Grayscale` | Convert to grayscale |
| `ChangeColorspace` | Change color space |
| `RemoveSaturation` | Remove color saturation |
| `AddToHueAndSaturation` | Shift hue/saturation |
| `AddToHue` | Shift hue only |
| `AddToSaturation` | Shift saturation only |
| `MultiplyHueAndSaturation` | Scale hue/saturation |
| `MultiplyHue` | Scale hue only |
| `MultiplySaturation` | Scale saturation only |
| `ChangeColorTemperature` | Warm/cool adjustment |
| `WithColorspace` | Apply in different colorspace |
| `WithBrightnessChannels` | Apply to brightness |
| `WithHueAndSaturation` | Apply to H/S channels |
| `Posterize` | Reduce color depth |
| `UniformColorQuantization` | Quantize colors |
| `KMeansColorQuantization` | K-means color reduction |
